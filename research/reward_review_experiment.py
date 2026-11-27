"""Compare joint and shaped GRPO from a shared write-capable checkpoint."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from math import isclose
from pathlib import Path
from statistics import mean, pstdev

from research.competence_data import load_and_check
from research.competence_training import PROMPTS, _release, verify_initialization
from research.joint_reward_experiment import evaluation_summary, joint_success
from research.reward_memory_credit import memory_credit
from research.src.artifacts import digest, file_hash, now, write_json
from research.src.closure_protocol import execution_context
from research.src.policy.llm_controller import SpecLocalPolicy
from research.src.policy.rule_controller import compatible_worlds
from research.src.training.episodic_grpo import group_advantages, train_grpo
from research.src.training.rollouts import make_rollout_collector
from research.src.training.sft import train_sft
from research.weak_sft_experiment import (
    SEED, SELECTION, authorize, evaluate, read, selection_record, training_cohort, validation_callback,
)
from research.weak_sft_qa import _rows

WARMUP_STEPS = 120


def informative_question(trace):
    if trace["user_reply"] is None:
        return False
    before, after = (trace["observations"][phase] for phase in ("select", "respond"))
    for query in (None, trace["episode"]["future_query"]):
        old = {world["value"] for world in compatible_worlds(before, query=query)}
        new = {world["value"] for world in compatible_worlds(after, query=query)}
        if (new and None not in new) and (new < old or old == {None} and len(new) == 1):
            return True
    return False


def components(row):
    metrics, trace = row["metrics"], row["trace"]
    memory = memory_credit(trace, {"success": metrics["FJS"]}, metrics["unsupported_operations"])
    current = metrics["C2"]
    future = int(metrics["FJS"] == 1 and memory["supported_write"])
    return {
        **memory,
        "current": current,
        "faithful_future": future,
        "joint": joint_success(current, metrics["FJS"], memory["valid_write"], metrics["unsupported_operations"]),
        "informative_question": informative_question(trace),
    }


def task_score(parts, shaped):
    joint = 2 * parts["joint"]
    if not shaped:
        return joint
    return joint + .50 * parts["current"] + .50 * parts["faithful_future"] + .50 * parts["benefit"]


def collect_with_reward(output, method, shaped):
    collect = make_rollout_collector(output, method=method, prompt_root=PROMPTS)

    def rollout(policy, episode, seed):
        result = collect(policy, episode, seed)
        row = read(Path(result.memory_path).with_suffix(".json"))
        parts = components(row)
        relief = .10 if shaped and parts["informative_question"] else 0.
        # The original optimizer sums these two task slots. The environment
        # trace retains the actual utilities and question count.
        return replace(result, r2=task_score(parts, shaped), r3=relief)

    return rollout


def warmup(run, checkpoint, examples, validation, initial):
    output = run / "warmup"
    actor = SpecLocalPolicy.from_checkpoint(checkpoint, trainable=True, device="cuda")
    try:
        with execution_context(run, "training"):
            return train_sft(actor, examples, output / "optimizer",
                validation_callback(validation, output / "validation", "SFT-write"),
                continuation=True, seed=SEED, initial_validation_metrics=initial,
                updates=WARMUP_STEPS, epochs=2, checkpoint_every=WARMUP_STEPS,
                learning_rate=1e-4, budget_seconds=240, checkpoint_save_budget_seconds=120,
                prompt_root=PROMPTS, closure_authorization=run, closure_selection=SELECTION)
    finally:
        _release(actor)


def train(run, method, shaped, parent, cohort, validation, initial, old):
    output = run / "continuations" / method
    output.mkdir(parents=True)
    actor = SpecLocalPolicy.from_checkpoint(parent, trainable=True, device="cuda")
    reference = None
    try:
        reference = SpecLocalPolicy.from_checkpoint(parent, trainable=False, device="cpu")
        actor.config.temperature = reference.config.temperature = old["temperature"]
        write_json(output / "initialization.json", verify_initialization(actor, parent, reference))
        with execution_context(run, "training"):
            return train_grpo(actor, reference, cohort,
                collect_with_reward(output / "environment_rollouts", method, shaped),
                output / "optimizer", validation_callback(validation, output / "validation", method),
                horizon="long", updates=old["max_updates"], seed=SEED,
                roots_per_batch=old["roots_per_batch"], rollouts_per_root=old["rollouts_per_root"],
                learning_rate=old["learning_rate"], kl_coefficient=old["kl_coefficient"],
                clip_epsilon=old["clip_epsilon"], checkpoint_every=old["max_updates"],
                budget_seconds=old["budget_seconds"], initial_validation_metrics=initial,
                closure_authorization=run, checkpoint_save_budget_seconds=120, closure_selection=SELECTION,
                reference_execution=old["reference_execution"])
    finally:
        _release(reference)
        _release(actor)


def training_audit(output, result, shaped):
    groups = defaultdict(list)
    counts = Counter()
    signatures = []
    rows = _rows(output / "optimizer/rollouts.jsonl")
    plan = result["plan"]
    if len(rows) != plan["max_updates"] * plan["roots_per_batch"] * plan["rollouts_per_root"]:
        raise ValueError("Incomplete matched rollout budget")
    if result["frozen_before"] != result["frozen_after"] or result["reference_before"] != result["reference_after"]:
        raise ValueError("GRPO changed frozen parameters")
    for row in rows:
        rollout = row["rollout"]
        path = Path(rollout["memory_path"])
        if not path.is_file() or rollout["split"] != "train":
            raise ValueError("Missing isolated TRAIN state")
        evidence = read(path.with_suffix(".json"))
        parts = components(evidence)
        trace, metrics = evidence["trace"], evidence["metrics"]
        relief = .10 if shaped and parts["informative_question"] else 0.
        task = task_score(parts, shaped)
        expected = task + relief - .10 * metrics["questions"] - .50 * metrics["unsupported_operations"] - .20 * len(trace["errors"])
        if not isclose(expected, row["return"], abs_tol=1e-9):
            raise ValueError("Reward disagrees with environment and optimizer traces")
        groups[row["group_id"]].append((task, expected, row["advantage"]))
        counts["joint_successes"] += parts["joint"]
        counts["valid_writes"] += parts["valid_write"]
        counts["write_attempts"] += any(op["op"] != "NOOP" for op in trace["executed_operations"])
        counts["changed_memory"] += trace["final_memory_hash"] != trace["initial_memory_hash"]
        counts["helpful_writes"] += parts["benefit"] > 0
        counts["harmful_writes"] += parts["benefit"] < 0
        counts["unsupported_writes"] += metrics["unsupported_operations"] > 0
        counts["questions"] += metrics["questions"]
        counts["informative_questions"] += parts["informative_question"]
        outputs = [phase["generation"]["completion_ids"] for phase in rollout["phases"]] if row["update"] == 1 else None
        signatures.append([rollout["root_id"], rollout["latent_world_id"], rollout["rollout_seed"], outputs])
    for group in groups.values():
        tasks, rewards, recorded = zip(*group)
        counts["groups_with_task_signal"] += pstdev(tasks) > 0
        counts["groups_with_total_signal"] += pstdev(rewards) > 0
        if any(not isclose(a, b, abs_tol=1e-9) for a, b in zip(group_advantages(rewards), recorded, strict=True)):
            raise ValueError("Wrong GRPO advantages")
    return {**counts, "trajectories": len(rows), "updates": result["resource_counters"]["updates"],
            "adapter_changed": result["trainable_before"] != result["trainable_after"],
            "matched_schedule_sha256": digest(signatures),
            "validation_J": {str(e["step"]): e["validation"]["J_primary"]
                             for e in result["history"] if "validation" in e},
            "selected_step": result["best_step"]}


def execute(previous, fresh_dir, run):
    previous, fresh_dir, run = (path.resolve() for path in (previous, fresh_dir, run))
    prior_plan = read(previous / "PILOT_PLAN.json")
    bank_dir = Path(prior_plan["bank_dir"])
    splits, manifest = load_and_check(bank_dir)
    fresh, fresh_manifest = load_and_check(fresh_dir)
    if any(digest(rows) != prior_plan["input_sha256"][role] for role, rows in splits.items()):
        raise ValueError("Training bank changed")
    cohort = training_cohort(splits["train"])
    if digest(cohort) != prior_plan["rl"]["cohort_sha256"]:
        raise ValueError("Training schedule changed")
    chosen = read(previous / "CALIBRATION.json")["selected"]
    controls = {row["method"]: row for row in read(previous / "SELECTION.json")["rows"]}
    old = read(previous / "continuations/B2-L/optimizer/training.json")["plan"]
    old["temperature"] = prior_plan["rl"]["temperature"]
    teacher_path = bank_dir / manifest["teachers"]["train"]["path"]
    examples = [row for row in _rows(teacher_path) if row["phase"] == "write" and row["route"] == "natural"]
    if not examples or any(row["split"] != "train" for row in examples):
        raise ValueError("Expected natural TRAIN write targets")
    source_hashes = {str(p.resolve()): file_hash(p) for p in (
        Path(__file__), Path(__file__).with_name("reward_memory_credit.py"))}
    plan = {"created_at": now(), "seed": SEED, "prior": str(previous), "source_sha256": source_hashes,
            "fresh_evaluation_sha256": digest(fresh["evaluation"]), "fresh_data_seed": fresh_manifest["seed"],
            "train_cohort_sha256": digest(cohort), "write_warmup": {"updates": WARMUP_STEPS,
            "examples": len(examples), "examples_sha256": digest(examples), "learning_rate": 1e-4,
            "rule": "Use terminal unless validation has fewer than two supported non-NOOP writes or invalid > .20; then use previously selected B3 for both arms."},
            "rl": old, "weak_parent": controls["B1-weak"], "more_sft_control": controls["B3"],
            "joint_reward": "2 * I(C2, FJS, valid WRITE, no unsupported WRITE) minus original penalties",
            "revised_reward": "same joint + .5*C2 + .5*faithful FJS + .5*causal memory delta; waive original .10 ASK cost only if observable possible answers narrow"}
    run.mkdir(parents=True)
    authorize(run, fresh_dir, plan)
    initial = dict(read(chosen["validation_summary"])["metrics"])
    initial["utility"] = initial["J_primary"]
    warm = warmup(run, chosen["checkpoint_path"], examples, splits["validation"], initial)
    if warm["resource_counters"]["updates"] != WARMUP_STEPS:
        raise ValueError("Incomplete write warmup")
    warm_summary = run / "warmup/validation/evaluation_00/summary.json"
    warm_result = read(warm_summary)
    warm_rows = _rows(warm_result["paths"]["trajectories"])
    supported = sum(row["metrics"]["write_operation_count"] > 0 and row["metrics"]["write_faithful_episode"]
                    for row in warm_rows)
    fallback = supported < 2 or warm_result["metrics"]["any_invalid_output"] > .20
    if fallback:
        baseline = read(previous / "continuations/B3/optimizer/training.json")
        metrics = next(entry["validation"] for entry in baseline["history"] if entry["step"] == baseline["best_step"])
        parent = controls["B3"]["checkpoint_path"]
    else:
        parent = warm["final_checkpoint"]
        metrics = warm_result["metrics"]
    metrics = {**metrics, "utility": metrics["J_primary"]}
    write_json(run / "STARTING_POLICY.json", {"checkpoint": parent, "fallback_to_more_sft": fallback,
        "warmup_supported_write_episodes": supported, "warmup_validation": warm_result["metrics"]})
    print("shared starting policy", parent, "fallback", fallback, "supported writes", supported, flush=True)
    records = [
        controls["B1-weak"], controls["B3"],
        selection_record("SFT-write", warm["final_checkpoint"], WARMUP_STEPS),
        selection_record("RL-start", parent, 0),
    ]
    audits = {}
    for method, shaped in (("RL-joint", False), ("RL-revised", True)):
        print("starting", method, flush=True)
        result = train(run, method, shaped, parent, cohort, splits["validation"], metrics, old)
        audit = training_audit(run / "continuations" / method, result, shaped)
        audits[method] = audit
        write_json(run / f"QA_{method}.json", audit)
        records.extend([selection_record(method, result["selected_checkpoint"], result["best_step"]),
                        selection_record(method + "-terminal", result["final_checkpoint"], audit["updates"])])
        print(method, audit, flush=True)
    if audits["RL-joint"]["matched_schedule_sha256"] != audits["RL-revised"]["matched_schedule_sha256"]:
        raise ValueError("Different rollout schedule or initial completions between arms")
    if any(file_hash(path) != sha for path, sha in source_hashes.items()):
        raise ValueError("Experiment source changed")
    selection = run / "SELECTION.json"
    evaluation = fresh["evaluation"]
    write_json(selection, {"version": "competence_selection_v1", "created_at": now(),
        "selection_complete": True, "selection_split": "validation", "selected_before_evaluation": True,
        "heldback_episodes_sha256": digest(evaluation), "rows": records})
    summaries = evaluate(run, evaluation, selection)
    panels = {method: evaluation_summary(Path(summary["summary"]), evaluation, selection,
              read(summary["summary"])["method"], SEED) for method, summary in summaries.items()}
    paired = {}
    revised = panels["RL-revised-terminal"]["by_root"]
    for method in ("RL-joint-terminal", "RL-start", "B1-weak", "B3"):
        other = panels[method]["by_root"]
        diffs = [revised[root] - other[root] for root in revised]
        paired[method] = {"delta_J_pp": 100 * mean(diffs), "improved": sum(d > 0 for d in diffs),
                          "worse": sum(d < 0 for d in diffs), "tied": sum(d == 0 for d in diffs)}
    write_json(run / "COMPARISON.json", {"status": "verified", "training": audits, "panels": panels,
        "evaluation_roots": len(revised), "evaluation_episodes": len(evaluation), "paired_revised_terminal_vs": paired})
    print("comparison verified", {name: row["metrics"]["J_primary"] for name, row in panels.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("previous", "fresh", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    execute(args.previous, args.fresh, args.output)


if __name__ == "__main__":
    main()
