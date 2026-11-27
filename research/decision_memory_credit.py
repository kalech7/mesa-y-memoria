"""Local action enumeration credit, without action targets or a learned teacher.

A complete on-policy *manager* path visits states, without evaluating outcomes.
Given a phase sampled independently with probability 1/2, uniformly choose one
of its n non-forced decisions. Weight n/4 cancels that within-path selection:
E_j[(n/4) g(h_j)] = sum_eligible g(h_j)/4. Averaging the two phases estimates
the mean over eight decision positions. Eligibility is measured before acting.

All legal actions at that prefix get fresh future rollouts under the same fixed
actor and common random numbers, independent of the index selector. Their real
reader outcomes, and only those outcomes, can be judged. Forks are correlated
counterfactuals, not independent trajectories for whole-episode leave-one-out.

The task-gradient interpretation is local on-policy action enumeration. An
additional exact categorical KL is a separately declared local surrogate anchor;
this module does not claim to differentiate future state-visitation KL terms.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

from research.budget_memory_env import ASK_COST, WRITE_COST, canonical, fingerprint, run_episode

VERSION = "decision_memory_credit_v1"
PHASES = ("ask", "write")
DECISIONS_PER_PHASE = 4


def _seed(seed, domain):
    if type(seed) is not int or seed < 0:
        raise ValueError("A nonnegative integer rollout seed is required")
    return int(fingerprint([VERSION, seed, domain])[:15], 16)


def _write(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _actor_fingerprint(policy):
    """Hash the small controller, never copy a frozen billion-byte backbone."""
    hook = getattr(policy, "credit_parameter_fingerprint", None)
    if callable(hook):
        result = hook()
        if not isinstance(result, str) or not result:
            raise ValueError("credit_parameter_fingerprint must return a nonempty string")
        return result
    head = getattr(policy, "head", None)
    if head is None or not callable(getattr(head, "state_dict", None)):
        raise TypeError("The actor must expose head.state_dict or credit_parameter_fingerprint")
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(head.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(canonical([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _unchanged(policy, expected):
    actual = _actor_fingerprint(policy)
    if actual != expected:
        raise ValueError("Actor changed during on-policy/counterfactual collection")
    return actual


def _behavior_probabilities(decision):
    metadata = decision["metadata"]
    probabilities = metadata.get("probabilities")
    logs = metadata.get("old_logprobs")
    if probabilities is None and isinstance(logs, list):
        probabilities = [math.exp(value) for value in logs]
    candidates = decision["candidates"]
    if (not isinstance(probabilities, list) or len(probabilities) != len(candidates)
            or any(not isinstance(p, (float, int)) or not math.isfinite(p) or p <= 0
                   for p in probabilities)
            or not math.isclose(math.fsum(probabilities), 1., rel_tol=0., abs_tol=1e-6)):
        raise ValueError("Target requires the complete positive behavior distribution")
    if logs is not None:
        if (not isinstance(logs, list) or len(logs) != len(probabilities)
                or any(not math.isfinite(logp) or abs(math.exp(logp)-p) > 1e-6
                       for logp, p in zip(logs, probabilities))):
            raise ValueError("Behavior log-probabilities and probabilities disagree")
    index = candidates.index(decision["action"])
    if metadata.get("action_index", index) != index:
        raise ValueError("Behavior action index does not match the sampled action")
    # Normalize only the tolerated serialization round-off, not a bad policy.
    total = math.fsum(probabilities)
    return [float(p)/total for p in probabilities]


class _UnobservedOutcomeReader:
    """Terminal placeholders that can never become labels or measured success."""
    def __init__(self):
        self.calls = 0

    def answer(self, observation, candidates, *, seed, decision_id):
        self.calls += 1
        return {"answer": "DEFER", "metadata": {
            "outcome_not_evaluated": True, "real_reader_called": False,
            "purpose": "manager_state_visitation_only", "decision_id": decision_id,
        }}


class _RecordedReader:
    def __init__(self, reader):
        self.reader = reader
        self.calls = []

    def answer(self, observation, candidates, *, seed, decision_id):
        self.calls.append({"seed": seed, "decision_id": decision_id,
                           "observation_sha256": fingerprint(observation)})
        fn = self.reader.answer if hasattr(self.reader, "answer") else self.reader
        return fn(observation, candidates, seed=seed, decision_id=decision_id)


class _ForkPolicy:
    def __init__(self, policy, decisions, target_step, action):
        self.policy = policy
        self.decisions = decisions
        self.target_step = target_step
        self.action = action
        self.replayed = self.forced = self.sampled = 0

    def decide(self, observation, candidates, *, seed, sample, decision_id):
        if sample is not True:
            raise ValueError("Counterfactual future continuation must be sampled")
        step = 2 * observation["session_index"] + int(observation["decision_type"] == "write")
        if step <= self.target_step:
            original = self.decisions[step]
            if (canonical(observation) != canonical(original["observation"])
                    or candidates != original["candidates"]):
                raise ValueError("Counterfactual prefix is not identical to its on-policy source")
            metadata = {"decision_id": decision_id, "policy_evaluated": False,
                        "source_decision_id": original["decision_id"],
                        "source_observation_sha256": original["observation_sha256"]}
            if step < self.target_step:
                self.replayed += 1
                metadata.update(credit_role="replayed_prefix",
                                source_policy_output=deepcopy(original["policy_output"]))
                action = original["action"]
            else:
                self.forced += 1
                metadata.update(credit_role="forced_counterfactual", forced_action=self.action)
                action = self.action
            # Do not attach old_logprobs/action_index to a forced/replayed action
            # as if this were a newly sampled on-policy transition.
            return {"action": action, "metadata": metadata}
        fn = self.policy.decide if hasattr(self.policy, "decide") else self.policy
        result = fn(observation, candidates, seed=seed, sample=True, decision_id=decision_id)
        if not isinstance(result, dict):
            raise TypeError("A sampled suffix needs the actor's complete metadata")
        result = deepcopy(result)
        result.setdefault("metadata", {}).update(credit_role="sampled_suffix", policy_evaluated=True)
        self.sampled += 1
        return result


def collect_counterfactual_group(policy, reader, episode, seed, output_dir, phase):
    """Visit states, select one eligible decision and enumerate its legal actions.

    The caller MUST sample phase uniformly/independently before this call; fixed
    alternation is not described as an unbiased per-update phase sampler. Target
    index and fork-future randomness use different deterministic seed domains.
    The returned visitation path has no queries, labels, utility or judge payload.
    """
    if phase not in PHASES:
        raise ValueError("phase must be ask or write")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Counterfactual collection cannot overwrite a prior group")
    output.mkdir(parents=True, exist_ok=True)
    namespace = f"decision-credit:{fingerprint([str(output.resolve()), seed])[:20]}"
    visit_seed, selector_seed, fork_seed = (_seed(seed, name) for name in
                                         ("manager_visitation", "target_index", "fork_future"))
    initial = _actor_fingerprint(policy)
    stub = _UnobservedOutcomeReader()
    visited = run_episode(policy, stub, episode, visit_seed, output / "visitation",
                          sample=True, namespace=namespace + ":visitation")
    _unchanged(policy, initial)
    if len(visited["decisions"]) != 8:
        raise ValueError("This estimator requires the four-session/eight-decision contract")
    # The environment used DEFER only to finish state visitation. Erase its
    # non-evaluated endpoints so reward parsers fail closed instead of using them.
    for key in ("queries", "future_answers", "metrics", "public_judge_payload"):
        visited.pop(key, None)
    visited.update(collection_role="manager_state_visitation_only", outcome_not_evaluated=True,
                   real_reader_calls=0, omitted_terminal_queries=stub.calls)
    _write(output / "visitation" / "trace.json", visited)
    phase_decisions = [d for d in visited["decisions"] if d["phase"] == phase]
    if len(phase_decisions) != DECISIONS_PER_PHASE:
        raise ValueError("Each phase must have four pre-action states")
    eligible = [d for d in phase_decisions if len(d["candidates"]) > 1]
    if not eligible:
        raise ValueError("No eligible decision in this phase; do not silently resample")
    chosen = random.Random(selector_seed).choice(eligible)
    target = deepcopy(chosen)
    n = len(eligible)
    selection_weight = n / DECISIONS_PER_PHASE
    probabilities = _behavior_probabilities(target)
    past_cost = math.fsum(ASK_COST if d["phase"] == "ask" and d["action"] != "PROCEED"
                         else WRITE_COST if d["phase"] == "write" and d["action"] != "NOOP"
                         else 0. for d in visited["decisions"][:target["step_id"]])
    target.update(eligible_count=n, eligible_step_ids=[d["step_id"] for d in eligible],
                  within_phase_selection_probability=1/n, phase_selection_probability=.5,
                  selection_weight=selection_weight)
    branches, reader_calls = [], []
    for index, action in enumerate(target["candidates"]):
        wrapper = _ForkPolicy(policy, visited["decisions"], target["step_id"], action)
        audited_reader = _RecordedReader(reader)
        branch = run_episode(wrapper, audited_reader, episode, fork_seed,
                             output / f"branch_{index:02d}", sample=True,
                             namespace=f"{namespace}:branch:{index}")
        _unchanged(policy, initial)
        if (wrapper.replayed != target["step_id"] or wrapper.forced != 1
                or wrapper.sampled != 7-target["step_id"] or len(audited_reader.calls) != 2):
            raise ValueError("A fork must replay its prefix, force once and resample all future decisions")
        branch["counterfactual"] = dict(collection_role="local_action_enumeration",
            target_step=target["step_id"], target_phase=phase, forced_action=action,
            forced_action_index=index, source_decision_id=target["decision_id"],
            replayed_prefix_decisions=wrapper.replayed, forced_decisions=wrapper.forced,
            sampled_future_decisions=wrapper.sampled, fork_seed=fork_seed,
            past_cost=past_cost, future_cost=branch["metrics"]["costs"]-past_cost,
            real_reader_calls=2, actor_fingerprint=initial,
            source_path_on_policy=False, use_for_whole_episode_LOO=False)
        _write(output / f"branch_{index:02d}" / "trace.json", branch)
        branches.append(branch)
        reader_calls.append(audited_reader.calls)
    decision_seeds = [[d["seed"] for d in b["decisions"]] for b in branches]
    reader_seeds = [[call["seed"] for call in calls] for calls in reader_calls]
    if any(x != decision_seeds[0] for x in decision_seeds) or any(x != reader_seeds[0] for x in reader_seeds):
        raise ValueError("Fork continuation does not use common random numbers")
    group = dict(version=VERSION, episode_id=episode["episode_id"],
        root_id=episode.get("root_id", episode["episode_id"]), family=episode.get("family"),
        seed=seed, phase=phase, target=target,
        prefix=deepcopy(visited["decisions"][:target["step_id"]]), visitation=visited,
        branches=branches, behavior_probs=probabilities, selection_weight=selection_weight,
        diagnostics=dict(actor_before=initial, actor_after=_unchanged(policy, initial),
            actor_unchanged=True, visit_seed=visit_seed, selector_seed=selector_seed,
            fork_future_seed=fork_seed, rng_domains_separate=True,
            common_decision_seeds=decision_seeds[0], common_reader_seeds=reader_seeds[0],
            visitation_real_reader_calls=0, visitation_stub_reader_calls=stub.calls,
            fork_real_reader_calls=sum(map(len, reader_calls)),
            outcome_label_episodes=len(branches), past_cost=past_cost,
            phase_sampling_contract="caller uniform independent Bernoulli(1/2)",
            normalization="eligible_count/4; phase probability1/2 estimates mean8",
            teacher_actions=False, selected_by_return=False, optimizer_steps=0))
    _write(output / "group.json", group)
    return group


def enumerate_credit(group, rewards):
    """Center real fork returns with a detached behavior-weighted baseline.

    Rewards must be in branch/catalog order and come from those evaluated forks.
    They may be scalars or fixed-reward-module dictionaries. This helper creates
    no action target. The task loss is -weight*sum(pi_current*stopgrad(advantages));
    one actor step is permitted only after every branch and judgment is complete.
    """
    branches = group["branches"]
    if len(rewards) != len(branches) or len(branches) != len(group["target"]["candidates"]):
        raise ValueError("Every legal action needs one real future return")
    values = []
    for branch, reward in zip(branches, rewards):
        if branch.get("counterfactual", {}).get("collection_role") != "local_action_enumeration":
            raise ValueError("Only actual counterfactual-reader branches can receive outcome credit")
        value = reward.get("episodic_reward") if isinstance(reward, dict) else reward
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("A finite real fork return is required")
        values.append(float(value))
    probabilities = group["behavior_probs"]
    baseline = math.fsum(p*r for p, r in zip(probabilities, values))
    past_cost = group["diagnostics"]["past_cost"]
    future_returns = [r+past_cost for r in values]
    advantages = [r-baseline for r in values]
    return dict(version=VERSION, actions=list(group["target"]["candidates"]),
        behavior_probs=list(probabilities), full_returns=values, future_returns=future_returns,
        past_cost=past_cost, future_costs=[b["counterfactual"]["future_cost"] for b in branches],
        baseline_full=baseline, baseline_future=baseline+past_cost, advantages=advantages,
        weighted_advantage_mean=math.fsum(p*a for p, a in zip(probabilities, advantages)),
        selection_weight=group["selection_weight"], eligible_count=group["target"]["eligible_count"],
        task_loss="-selection_weight * sum(pi_current * stopgrad(advantages))",
        advantage_normalization="none", target_action=None, whole_episode_LOO=False)
