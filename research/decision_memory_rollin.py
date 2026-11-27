"""Exogenous-session roll-in with deferred ASK and exact prefix importance.

Phase is externally Bernoulli(1/2), session is Uniform(0..3) BEFORE roll-in.
For ASK at t, force earlier ASK to PROCEED and sample earlier WRITE under pi.
Every eligible on-policy ASK prefix has exactly this earlier-ASK pattern, so
prod_k<t pi_old(PROCEED|h_k) is its pi/proposal density ratio. Ineligible ASK
states have zero policy score. WRITE prefixes stay fully on-policy, ratio one.

The ratio is a detached scalar, without clipping or batch renormalization. With
phase/session probabilities 1/2 and 1/4, the weighted local action expectation
estimates the mean task gradient over eight decisions. No n/4 factor is used.
The actor remains fixed across prefix collection and all real outcome forks.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import random

from research.budget_memory_env import ASK_COST, WRITE_COST, fingerprint, run_episode
from research.decision_memory_credit import (
    _actor_fingerprint, _behavior_probabilities, _ForkPolicy, _RecordedReader,
    _seed, _unchanged, _write,
)

VERSION = "decision_memory_rollin_v1"
SAMPLING_CONTRACT = "uniform_session_deferred_ask_IS_v1"


class _TargetReached(Exception):
    pass


class _ForbiddenRollinReader:
    def answer(self, *args, **kwargs):
        raise ValueError("Prefix-only roll-in must never evaluate a terminal reader")


class _RollinPolicy:
    def __init__(self, policy, phase, session_index, *, defer_previous_ask=None, allow_forced_target=False):
        self.policy = policy
        self.phase = phase
        self.session_index = session_index
        self.target_step = 2*session_index + int(phase == "write")
        self.defer_previous_ask = phase == "ask" if defer_previous_ask is None else bool(defer_previous_ask)
        self.allow_forced_target = bool(allow_forced_target)
        self.prefix = []
        self.target = None
        self.proceed_prefix_logprobs = []
        self.density_factors = []
        self.ask_prefix_behavior = []

    def decide(self, observation, candidates, *, seed, sample, decision_id):
        step = 2*observation["session_index"] + int(observation["decision_type"] == "write")
        if sample is not True or step != len(self.prefix) or step > self.target_step:
            raise ValueError("Roll-in must sample sequentially and stop at its exogenous target")
        if self.prefix:
            self.prefix[-1]["slots_after_sha256"] = fingerprint(observation["slots"])
        fn = self.policy.decide if hasattr(self.policy, "decide") else self.policy
        raw = fn(observation, candidates, seed=seed, sample=True, decision_id=decision_id)
        if not isinstance(raw, dict) or raw.get("action") not in candidates:
            raise TypeError("Roll-in requires a real controller action and complete metadata")
        raw = deepcopy(raw)
        record = dict(decision_id=decision_id, step_id=step,
            phase=observation["decision_type"], decision_type=observation["decision_type"],
            session_index=observation["session_index"], seed=seed,
            observation=deepcopy(observation), observation_sha256=fingerprint(observation),
            candidates=list(candidates), action=raw["action"], metadata=deepcopy(raw["metadata"]),
            policy_output=deepcopy(raw), slots_before_sha256=fingerprint(observation["slots"]))
        if step == self.target_step:
            if len(candidates) != 3 and not (self.allow_forced_target and candidates == ["PROCEED"]):
                raise ValueError("The exogenous target must retain its complete legal catalogue")
            record["target_draw_not_executed"] = True
            record["metadata"]["target_draw_not_executed"] = True
            self.target = record
            raise _TargetReached()
        if observation["decision_type"] == "ask" and len(candidates) > 1:
            probabilities = _behavior_probabilities(record)
            probability = probabilities[candidates.index("PROCEED")]
            self.ask_prefix_behavior.append(dict(step_id=step, session_index=observation["session_index"],
                logprob_proceed=math.log(probability), probability_proceed=probability,
                sampled_action=raw["action"], observation_sha256=record["observation_sha256"]))
        force = self.defer_previous_ask and observation["decision_type"] == "ask"
        if force:
            if observation["session_index"] >= self.session_index or len(candidates) != 3:
                raise ValueError("Only earlier eligible ASK states can be deferred")
            probabilities = _behavior_probabilities(record)
            probability = probabilities[candidates.index("PROCEED")]
            log_probability = math.log(probability)
            self.proceed_prefix_logprobs.append(log_probability)
            self.density_factors.append(dict(step_id=step, session_index=observation["session_index"],
                action="PROCEED", target_policy_probability=probability,
                target_policy_logprob=log_probability, proposal_probability=1.,
                density_ratio_factor=probability))
            result = {"action": "PROCEED", "metadata": {
                "credit_role": "forced_prefix_proceed", "decision_id": decision_id,
                "policy_evaluated": True, "sampled": False, "executed_action": "PROCEED",
                "proposal_probability": 1., "density_ratio_factor": probability,
                "executed_action_logprob": log_probability,
                "sampled_action_unused": raw["action"], "source_policy_output": raw}}
        else:
            result = deepcopy(raw)
            result["metadata"].update(credit_role="sampled_rollin_prefix", policy_evaluated=True,
                                      density_ratio_factor=1.)
        record.update(action=result["action"], metadata=deepcopy(result["metadata"]),
                      policy_output=deepcopy(result))
        self.prefix.append(record)
        return result


def collect_counterfactual_group(policy, reader, episode, seed, output_dir, phase, session_index=None):
    """Collect an IS-corrected prefix and all three legal real-outcome forks.

    Default session selection preserves the legacy target_index RNG domain, but
    now draws uniformly from all four sessions before any observation/return.
    An explicitly supplied session must have been independently preselected by
    the caller. No resampling or search for a favorable state is permitted.
    """
    if phase not in ("ask", "write"):
        raise ValueError("phase must be ask or write")
    visit_seed, selector_seed, fork_seed = (_seed(seed, name) for name in
                                         ("manager_visitation", "target_index", "fork_future"))
    selection_mode = "preselected_by_caller" if session_index is not None else "uniform_rng_before_rollin"
    if session_index is None:
        session_index = random.Random(selector_seed).randrange(4)
    if type(session_index) is not int or not 0 <= session_index < 4:
        raise ValueError("session_index must be exogenous and in 0..3")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Roll-in collection cannot overwrite an earlier group")
    output.mkdir(parents=True, exist_ok=True)
    namespace = f"decision-rollin:{fingerprint([str(output.resolve()),seed])[:20]}"
    initial = _actor_fingerprint(policy)
    rollin_policy = _RollinPolicy(policy, phase, session_index)
    try:
        run_episode(rollin_policy, _ForbiddenRollinReader(), episode, visit_seed,
                    output/"rollin", sample=True, namespace=namespace+":rollin")
    except _TargetReached:
        pass
    else:
        raise ValueError("Roll-in did not stop before executing its target")
    _unchanged(policy, initial)
    if rollin_policy.target is None:
        raise ValueError("No exogenous target was captured")
    target = rollin_policy.target
    prefix = rollin_policy.prefix
    if len(prefix) != target["step_id"]:
        raise ValueError("Captured prefix has the wrong length")
    logs = rollin_policy.proceed_prefix_logprobs
    if len(logs) != (session_index if phase == "ask" else 0):
        raise ValueError("Deferred ASK density ratio has missing or extra factors")
    log_weight = math.fsum(logs)
    importance = math.exp(log_weight)
    if not math.isfinite(importance) or not 0 <= importance <= 1:
        raise ValueError("The exact prefix density ratio must be finite and at most one")
    on_policy = not bool(logs)
    target.update(phase_selection_probability=.5, within_phase_selection_probability=.25,
                  session_selection_probability=.25, session_selection_mode=selection_mode,
                  prefix_importance=importance, selection_weight=importance)
    probabilities = _behavior_probabilities(target)
    past_cost = math.fsum(ASK_COST if d["phase"] == "ask" and d["action"] != "PROCEED"
                         else WRITE_COST if d["phase"] == "write" and d["action"] != "NOOP"
                         else 0. for d in prefix)
    rollin = dict(collection_role="exogenous_prefix_only", outcome_not_evaluated=True,
        source_path_on_policy=on_policy, phase=phase, session_index=session_index,
        target_step=target["step_id"], decisions=deepcopy(prefix), real_reader_calls=0,
        target_draw_not_executed=True, prefix_importance=importance,
        proceed_prefix_logprobs=list(logs), density_factors=deepcopy(rollin_policy.density_factors))
    _write(output/"rollin"/"trace.json", rollin)
    branches, reader_calls = [], []
    replay_source = [*prefix, target]
    for index, action in enumerate(target["candidates"]):
        wrapper = _ForkPolicy(policy, replay_source, target["step_id"], action)
        audited_reader = _RecordedReader(reader)
        branch = run_episode(wrapper, audited_reader, episode, fork_seed,
                             output/f"branch_{index:02d}", sample=True,
                             namespace=f"{namespace}:branch:{index}")
        _unchanged(policy, initial)
        if (wrapper.replayed != target["step_id"] or wrapper.forced != 1
                or wrapper.sampled != 7-target["step_id"] or len(audited_reader.calls) != 2):
            raise ValueError("Fork replay, intervention or sampled continuation is incomplete")
        branch["counterfactual"] = dict(collection_role="local_action_enumeration",
            sampling_contract=SAMPLING_CONTRACT, target_step=target["step_id"], target_phase=phase,
            forced_action=action, forced_action_index=index,
            source_decision_id=target["decision_id"], replayed_prefix_decisions=wrapper.replayed,
            forced_decisions=1, sampled_future_decisions=wrapper.sampled, fork_seed=fork_seed,
            past_cost=past_cost, future_cost=branch["metrics"]["costs"]-past_cost,
            prefix_importance=importance, real_reader_calls=2, actor_fingerprint=initial,
            source_path_on_policy=False, use_for_whole_episode_LOO=False)
        _write(output/f"branch_{index:02d}"/"trace.json", branch)
        branches.append(branch)
        reader_calls.append(audited_reader.calls)
    decision_seeds = [[d["seed"] for d in b["decisions"]] for b in branches]
    reader_seeds = [[call["seed"] for call in calls] for calls in reader_calls]
    if any(s != decision_seeds[0] for s in decision_seeds) or any(s != reader_seeds[0] for s in reader_seeds):
        raise ValueError("Counterfactual continuations lost common random numbers")
    group = dict(version=VERSION, sampling_contract=SAMPLING_CONTRACT,
        episode_id=episode["episode_id"], root_id=episode.get("root_id",episode["episode_id"]),
        family=episode.get("family"), seed=seed, phase=phase, target=target,
        prefix=deepcopy(prefix), rollin=rollin, branches=branches,
        behavior_probs=probabilities, selection_weight=importance,
        diagnostics=dict(actor_before=initial, actor_after=_unchanged(policy,initial), actor_unchanged=True,
            rollin_mode="deferred_ask_importance_sampling" if phase=="ask" else "on_policy_prefix",
            source_path_on_policy=on_policy, proceed_prefix_logprobs=list(logs),
            density_factors=deepcopy(rollin_policy.density_factors), prefix_importance=importance,
            prefix_log_importance=log_weight, importance_detached=True,
            importance_clipped=False, importance_batch_renormalized=False,
            on_policy_support_proof=("Every eligible ASK prefix has all earlier ASK=PROCEED; "
                "WRITE is sampled from pi and cancels in pi/mu; forced ASK contributes pi_old(PROCEED). "
                "ASK prefixes absent from proposal are ineligible and have zero score. "
                "WRITE targets use a fully on-policy prefix and density ratio1."),
            visit_seed=visit_seed, selector_seed=selector_seed, fork_future_seed=fork_seed,
            session_selection_mode=selection_mode, selected_session_before_rollin=True,
            rng_domains_separate=True, common_decision_seeds=decision_seeds[0],
            common_reader_seeds=reader_seeds[0], rollin_real_reader_calls=0,
            visitation_real_reader_calls=0, fork_real_reader_calls=sum(map(len,reader_calls)),
            outcome_label_episodes=3, past_cost=past_cost,
            phase_sampling_contract="caller uniform independent Bernoulli(1/2)",
            session_sampling_contract="exogenous Uniform(0..3), no resampling",
            normalization="phase1/2 * session1/4; exact detached prefix pi/mu only",
            teacher_actions=False, selected_by_return=False, optimizer_steps=0))
    _write(output/"group.json",group)
    return group


collect_rollin_counterfactual_group = collect_counterfactual_group


def enumerate_credit(group, rewards):
    """Detached outcome centering, retaining exact roll-in importance weight."""
    if group.get("sampling_contract") != SAMPLING_CONTRACT:
        raise ValueError("This credit helper requires the uniform-session roll-in contract")
    branches = group["branches"]
    if len(rewards) != len(branches) or len(branches) != len(group["target"]["candidates"]):
        raise ValueError("Every legal action requires its evaluated future return")
    values = []
    for branch, reward in zip(branches,rewards):
        if branch.get("counterfactual",{}).get("sampling_contract") != SAMPLING_CONTRACT:
            raise ValueError("Only real roll-in counterfactual branches can receive outcome credit")
        value = reward.get("episodic_reward") if isinstance(reward,dict) else reward
        if not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError("Finite real fork rewards are required")
        values.append(float(value))
    probabilities = group["behavior_probs"]
    baseline = math.fsum(p*q for p,q in zip(probabilities,values))
    past_cost = group["diagnostics"]["past_cost"]
    advantages = [q-baseline for q in values]
    return dict(version=VERSION, sampling_contract=SAMPLING_CONTRACT,
        actions=list(group["target"]["candidates"]), behavior_probs=list(probabilities),
        full_returns=values, future_returns=[q+past_cost for q in values], past_cost=past_cost,
        future_costs=[b["counterfactual"]["future_cost"] for b in branches],
        baseline_full=baseline, baseline_future=baseline+past_cost, advantages=advantages,
        weighted_advantage_mean=math.fsum(p*a for p,a in zip(probabilities,advantages)),
        selection_weight=group["selection_weight"], prefix_importance=group["selection_weight"],
        task_loss="-detached_prefix_importance * sum(pi_current * stopgrad(advantages))",
        advantage_normalization="none", target_action=None, whole_episode_LOO=False)
