"""Compare the existing selectors with GRPO using sibling actions and local credit."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from time import monotonic

import numpy as np
import torch

from research import tesis_interactive_train as base
from research.src.analysis.paired_statistics import paired_bootstrap
from research.src.artifacts import canonical, digest, file_hash, write_json
from research.tesis_interactive_bank_audit import audit as audit_bank, compare_seeds
from research.tesis_interactive_reward import aggregate, paired_group


BRANCH = "GRPO-ramificado"
FILTERED = "SFT-por-retorno"
ARMS = (base.MORE, base.LONG, FILTERED, BRANCH)
METHODS = (base.SFT, base.RULE, *ARMS)
BANK_SEED = 684038
PAST_BANKS = (683915, 684023)
CHECKPOINTS = (25, 50, 75, 100)
PASSES = 2
DRAWS = len(base.ACTIONS)
WARMUP_STEPS = 60
METRICS = ("current_supported", "future_supported", "strict_joint", "questions",
           "longitudinal_return", "unsupported_writes")


class BranchRollouts(base.TrainingRollouts):
    """Use the same TRAIN prefix cache for ordinary and forced rollouts."""

    def complete(self, story, prefix):
        if story["split"] != "train":
            raise ValueError("Counterfactual returns are available only on TRAIN")
        transitions = self.cache[(story["root_id"], story["world_id"])]
        env = None
        try:
            while True:
                if prefix not in transitions:
                    if env is None:
                        env = base.InteractiveEpisode(story)
                        observation = env.observe()
                        for index in prefix:
                            observation, _ = env.step(base.ACTIONS[index])
                    transitions[prefix] = ((None, env.summary()) if env.done else
                                           (base.state_of(env, observation), None))
                state, trace = transitions[prefix]
                if trace is not None:
                    return trace
                prefix += (state.target,)
                if env is not None:
                    observation, _ = env.step(base.ACTIONS[state.target])
        finally:
            if env is not None:
                env.close()

    def slate(self, story, turn):
        states, reference, _ = self.collect(story, base.RULE)
        possible = [index for index, state in enumerate(states) if int(state.mask.sum()) > 1]
        if not possible:
            return None
        index = possible[turn % len(possible)]
        prefix = tuple(reference[:index])
        state = states[index]
        actions = [action for action, allowed in enumerate(state.mask.tolist()) if allowed]
        traces = [self.complete(story, prefix + (action,)) for action in actions]
        return state, prefix, actions, paired_group(traces)


def branch_update(actor, optimizer, parent, slate, *, filtered):
    if slate is None:
        # No action choice, hence neither a policy gradient nor a target.
        return {"actions": [], "steps": [], "task_varies": False}
    state, prefix, actions, measured = slate
    action_ids = torch.tensor(actions)
    returns = torch.tensor([item["total"] for item in measured["returns"]["longitudinal"]])
    with torch.no_grad():
        old = actor.logprobs([state])[0][action_ids]
        reference = parent.logprobs([state])[0][action_ids]
        mass = old.exp()
        # A fixed unit scale preserves the size of the question cost relative to task gains.
        advantages = returns - (mass * returns).sum()
        best = torch.isclose(returns, returns.max(), atol=1e-6, rtol=0)
    steps = []
    for _ in range(PASSES):
        now = actor.logprobs([state])[0][action_ids]
        if filtered:
            loss = -now[best].mean()
        else:
            ratio = (now - old).exp()
            surrogate = torch.minimum(ratio * advantages, ratio.clamp(1 - base.CLIP, 1 + base.CLIP) * advantages)
            loss = -(mass * surrogate).sum() + base.KL_WEIGHT * (now.exp() * (now - reference)).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
        optimizer.step()
        steps.append(float(loss.detach()))
    return {"prefix": [base.ACTIONS[index] for index in prefix], "actions": [base.ACTIONS[i] for i in actions],
            "teacher": base.ACTIONS[state.target], "best": [base.ACTIONS[i] for i, keep in zip(actions, best) if keep],
            "advantages": advantages.tolist(), "steps": steps, **measured}


def record_validation(directory, label, model, validation):
    result, rows = base.evaluate(model, validation, "validation")
    traces = directory / f"{label}.jsonl"
    weights = directory / f"{label}.json"
    base.write_rows(traces, rows)
    write_json(weights, base.model_artifact(model))
    return result, {"macro": result["macro"], "trace_sha256": file_hash(traces), "weights_sha256": file_hash(weights)}


def train(splits, seed, directory, cache):
    directory.mkdir()
    validation_dir = directory / "validation"
    validation_dir.mkdir()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_data, validation = splits["train"], splits["validation"]
    warm = base.Selector()
    demonstrations = [state for story in train_data for state in cache.collect(story, base.RULE)[0]]
    optimizer = torch.optim.AdamW(warm.parameters(), lr=.012, weight_decay=0.)
    for _ in range(WARMUP_STEPS):
        base.supervised_update(warm, optimizer, [demonstrations[i] for i in rng.integers(len(demonstrations), size=16)])
    initial, parent_evidence = record_validation(validation_dir, "SFT", warm, validation)
    schedule = base.order_train(train_data, CHECKPOINTS[-1], rng)
    selections = {base.SFT: {"selected": 0, "checkpoints": {0: parent_evidence}}}
    policies = {base.SFT: warm, base.RULE: base.RULE}
    turn_by_world = defaultdict(int)
    slates = []
    for story in schedule:
        world = story["world_id"]
        slates.append(cache.slate(story, turn_by_world[world]))
        turn_by_world[world] += 1
    for method in ARMS:
        start = monotonic()
        actor = base.clone(warm)
        optimizer = torch.optim.AdamW(actor.parameters(), lr=.008, weight_decay=0.)
        rollout_rng = torch.Generator().manual_seed(seed)
        incumbent, best_step, score = warm, 0, base.rank(initial, 0)
        checkpoints = {0: parent_evidence}
        ledger = []
        for index, (story, slate) in enumerate(zip(schedule, slates), 1):
            if method == base.MORE:
                states = cache.collect(story, base.RULE)[0]
                entry = {"steps": [base.supervised_update(actor, optimizer, states) for _ in range(PASSES)]}
            elif method == base.LONG:
                trajectories = [cache.collect(story, actor, rollout_rng) for _ in range(DRAWS)]
                entry = base.grouped_update(actor, optimizer, warm, trajectories, horizon="longitudinal", passes=PASSES)
            else:
                entry = branch_update(actor, optimizer, warm, slate, filtered=method == FILTERED)
            ledger.append({"root_id": story["root_id"], "world_id": story["world_id"], **entry})
            if index in CHECKPOINTS:
                summary, evidence = record_validation(validation_dir, f"{method}-{index}", actor, validation)
                checkpoints[index] = evidence
                rank = base.rank(summary, index)
                if base.passes_guards(summary, initial) and rank > score:
                    incumbent, best_step, score = base.clone(actor), index, rank
        log = directory / f"{method}-train.jsonl"
        base.write_rows(log, ledger)
        policies[method] = incumbent
        selections[method] = {"selected": best_step, "checkpoints": checkpoints,
                              "train_sha256": file_hash(log), "optimizer_updates": sum(len(row["steps"]) for row in ledger),
                              "trajectories_per_group": (1 if method == base.MORE else DRAWS)
                              if method in (base.MORE, base.LONG) else None,
                              "branch_outcomes": sum(len(row["actions"]) for row in ledger)
                              if method in (BRANCH, FILTERED) else None,
                              "task_groups": sum(bool(row.get("task_varies")) for row in ledger),
                              "seconds": monotonic() - start}
        terminal = checkpoints[CHECKPOINTS[-1]]["macro"]
        print(seed, method, best_step, checkpoints[best_step]["macro"]["future_supported"],
              "terminal", terminal["future_supported"], flush=True)
    return policies, selections


def effects(panels, families):
    rows = base.matched_rows({int(seed): panel for seed, panel in panels.items()}, families)
    return {other: {key: paired_bootstrap(rows, BRANCH, other, key, confidence=.95, resamples=2000)
                   for key in METRICS} for other in (*ARMS[:-1], base.SFT, base.RULE)}


def run(output, bank_seed, validation_only):
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    bank = audit_bank(bank_seed)
    past = [compare_seeds(bank_seed, earlier) for earlier in PAST_BANKS]
    if any(entry["count"] for item in past for entry in item["cross_experiment_identity_overlap"].values()):
        raise ValueError("The new bank shares an audited identity with a previous bank")
    splits = base.build_bank(seed=bank_seed)
    sources = [Path(__file__).resolve(), Path(base.__file__).resolve()]
    research = sources[0].parent
    dependencies = ("tesis_interactive_env.py", "tesis_interactive_reward.py", "tesis_bank.py",
                    "src/env/episode.py", "src/memory/executor.py", "src/memory/store.py",
                    "src/readers/symbolic.py", "src/policy/rule_controller.py", "src/policy/dialogue.py",
                    "src/scoring/evidence.py", "src/training/episodic_grpo.py")
    sources.extend(research / dependency for dependency in dependencies)
    plan = {"seeds": base.SEEDS, "bank_seed": bank_seed, "methods": METHODS,
            "bank_audit": bank, "previous_bank_comparisons": past,
            "source_sha256": {str(path.relative_to(research)): file_hash(path) for path in sources},
            "bank_sha256": {part: digest(rows) for part, rows in splits.items()},
            "warmup_steps": WARMUP_STEPS, "groups": CHECKPOINTS[-1], "passes": PASSES, "draws": DRAWS,
            "checkpoints": CHECKPOINTS, "reward": "unchanged longitudinal",
            "primary": {"method": BRANCH, "comparator": base.MORE, "metric": "future_supported",
                        "minimum_gain": .02, "paired_interval_lower_bound_above": 0.,
                        "maximum_current_loss": base.MAX_CURRENT_LOSS, "maximum_unsupported_write_increase": 0.},
            "branch_behavior": "Cycle over legal decision prefixes of the existing public teacher trajectory. "
                               "Execute each forced action and finish with the same public teacher.",
            "split_for_selection": "validation"}
    plan_path = output / "PLAN.json"
    write_json(plan_path, plan)
    for split, rows in splits.items():
        if split != "evaluation" or not validation_only:
            base.write_rows(output / f"{split}.jsonl", rows)
    policy_by_seed, training = {}, {}
    cache = BranchRollouts()
    for seed in base.SEEDS:
        policy_by_seed[seed], training[seed] = train(splits, seed, output / str(seed), cache)
    lock_path = output / "SELECTION.json"
    write_json(lock_path, {"plan_sha256": file_hash(plan_path), "training": training})
    if validation_only:
        return
    panels = {}
    for seed, policies in policy_by_seed.items():
        panels[seed] = {}
        for method, policy in policies.items():
            summary, rows = base.evaluate(policy, splits["evaluation"], "evaluation", disk=True)
            trace = output / str(seed) / f"{method}-evaluacion.jsonl"
            base.write_rows(trace, rows)
            panels[seed][method] = {**summary, "traces": str(trace.relative_to(output)), "trace_sha256": file_hash(trace)}
    families = {story["root_id"]: story["scenario_family"] for story in splits["evaluation"]}
    contrasts = effects(panels, families)
    primary = contrasts[base.MORE]
    future = primary["future_supported"]
    decision = (future["estimate"] >= plan["primary"]["minimum_gain"] and future["interval"][0] > 0
                and primary["current_supported"]["estimate"] >= -base.MAX_CURRENT_LOSS - 1e-10
                and primary["unsupported_writes"]["estimate"] <= 1e-10)
    means = {method: {key: mean(panel[method]["macro"][key] for panel in panels.values()) for key in METRICS}
             for method in METHODS}
    write_json(output / "RESULTS.json", {"selection_sha256": file_hash(lock_path), "means": means,
                                         "contrasts": contrasts, "primary_success": decision, "panels": panels})
    print(canonical({"means": means, "primary_success": decision}))


def audit(output):
    plan_path, lock_path = output / "PLAN.json", output / "SELECTION.json"
    plan, lock, result = [json.loads(path.read_bytes()) for path in (plan_path, lock_path, output / "RESULTS.json")]
    if lock["plan_sha256"] != file_hash(plan_path) or result["selection_sha256"] != file_hash(lock_path):
        raise ValueError("Selection or plan changed")
    research = Path(__file__).resolve().parent
    for path, checksum in plan["source_sha256"].items():
        source = research / path
        if path == Path(__file__).name and (output / "TRAINING_SOURCE.py").is_file():
            source = output / "TRAINING_SOURCE.py"
        if file_hash(source) != checksum:
            raise ValueError(f"Changed source: {path}")
    splits = {name: base.read_jsonl(output / f"{name}.jsonl") for name in plan["bank_sha256"]}
    if any(digest(rows) != plan["bank_sha256"][name] for name, rows in splits.items()):
        raise ValueError("Changed bank")
    ids = [(row["root_id"], row["world_id"]) for row in splits["evaluation"]]
    count = 0
    def verify_validation(directory, label, evidence):
        trace, weights = [directory / f"{label}.{suffix}" for suffix in ("jsonl", "json")]
        rows = base.read_jsonl(trace)
        if (file_hash(trace) != evidence["trace_sha256"] or file_hash(weights) != evidence["weights_sha256"]
                or any(row["split"] != "validation" for row in rows) or aggregate(rows)["macro"] != evidence["macro"]):
            raise ValueError("Validation trace or weights changed")
        return {"macro": evidence["macro"]}

    for seed in plan["seeds"]:
        seed = str(seed)
        directory = output / seed / "validation"
        per_seed = lock["training"][seed]
        parent = verify_validation(directory, "SFT", per_seed[base.SFT]["checkpoints"]["0"])
        schedules = []
        branch_ledgers = []
        for method in ARMS:
            training = per_seed[method]
            log = output / seed / f"{method}-train.jsonl"
            ledger = base.read_jsonl(log)
            if file_hash(log) != training["train_sha256"] or len(ledger) != plan["groups"]:
                raise ValueError("Training ledger changed")
            schedules.append([(row["root_id"], row["world_id"]) for row in ledger])
            if method in (BRANCH, FILTERED):
                branch_ledgers.append([{k: row.get(k) for k in ("prefix", "actions", "returns")} for row in ledger])
            candidates = {0: parent}
            for step in plan["checkpoints"]:
                entry = verify_validation(directory, f"{method}-{step}", training["checkpoints"][str(step)])
                if base.passes_guards(entry, parent):
                    candidates[step] = entry
            if max(candidates, key=lambda step: base.rank(candidates[step], step)) != training["selected"]:
                raise ValueError("Incorrect checkpoint selection")
        train_ids = {(row["root_id"], row["world_id"]) for row in splits["train"]}
        if any(schedule != schedules[0] for schedule in schedules) or not set(schedules[0]) <= train_ids:
            raise ValueError("Training schedules diverged")
        if branch_ledgers[0] != branch_ledgers[1]:
            raise ValueError("Equal-feedback controls saw different branches")
        for method in plan["methods"]:
            panel = result["panels"][seed][method]
            trace = output / panel["traces"]
            rows = base.read_jsonl(trace)
            if (file_hash(trace) != panel["trace_sha256"] or any(row["split"] != "evaluation" for row in rows)
                    or [(r["root_id"], r["world_id"]) for r in rows] != ids
                    or aggregate(rows) != {k: v for k, v in panel.items() if k not in ("traces", "trace_sha256")}):
                raise ValueError("Reserved trace or aggregation changed")
            count += len(rows)
    families = {story["root_id"]: story["scenario_family"] for story in splits["evaluation"]}
    if effects(result["panels"], families) != result["contrasts"]:
        raise ValueError("Reported paired contrasts changed")
    return {"status": "verified", "reserved_trajectories": count, "selection_sha256": file_hash(lock_path),
            "results_sha256": file_hash(output / "RESULTS.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank-seed", type=int, default=BANK_SEED)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    if args.audit:
        report = audit(args.output)
        write_json(args.output / "AUDIT.json", report)
        print(canonical(report))
    else:
        run(args.output, args.bank_seed, args.validation_only)
