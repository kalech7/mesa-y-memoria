"""Repeat the weak-SFT GRPO branch with a joint task reward."""
import argparse
from collections import defaultdict
from dataclasses import replace
from math import isclose
from pathlib import Path
from statistics import mean

from research.competence_data import load_and_check
from research.competence_evaluation import DEFAULT_THRESHOLDS, evaluate_selected
from research.competence_training import PROMPTS, _release, checkpoint_evidence, verify_initialization
from research.src.artifacts import digest, file_hash, now, write_json
from research.src.closure_protocol import execution_context, require_training_authorization
from research.src.policy.llm_controller import SpecLocalPolicy
from research.src.scoring.evidence import Evaluator
from research.src.training.episodic_grpo import group_advantages, train_grpo
from research.src.training.rollouts import make_rollout_collector
from research.weak_sft_experiment import authorize, read, selection_record, training_cohort, validation_callback
from research.weak_sft_qa import _panel, _rows


def joint_success(current, future, valid_write, unsupported):
    return int(current == 1 and future == 1 and valid_write and unsupported == 0)


def collect_joint(output):
    collect = make_rollout_collector(output, method="B2-L-joint", prompt_root=PROMPTS)

    def rollout(policy, episode, seed):
        result = collect(policy, episode, seed)
        row = read(Path(result.memory_path).with_suffix(".json"))
        trace, metrics = row["trace"], row["metrics"]
        success = joint_success(metrics["C2"], metrics["FJS"],
            (trace["write_result"] or {}).get("valid") is True, result.unsupported_operations)
        # GRPO uses the sum of its two task-score slots for the long horizon.
        # Carry the entire joint score in r2. Raw S2/S3 utilities stay in the
        # environment trace. Question, unsupported-write and format costs still apply.
        return replace(result, r2=2 * success, r3=0)

    return rollout


def train(run, checkpoint, cohort, validation, initial, old):
    parent = checkpoint["path"]
    actor = SpecLocalPolicy.from_checkpoint(parent, trainable=True, device="cuda")
    reference = None
    try:
        reference = SpecLocalPolicy.from_checkpoint(parent, trainable=False, device="cpu")
        actor.config.temperature = reference.config.temperature = old["temperature"]
        write_json(run / "INITIALIZATION.json", verify_initialization(actor, parent, reference))
        with execution_context(run, "training"):
            return train_grpo(actor, reference, cohort, collect_joint(run / "environment_rollouts"),
                run / "optimizer", validation_callback(validation, run / "validation", "B2-L-joint"),
                horizon=old["horizon"], updates=old["max_updates"], seed=old["seed"],
                roots_per_batch=old["roots_per_batch"], rollouts_per_root=old["rollouts_per_root"],
                learning_rate=old["learning_rate"], kl_coefficient=old["kl_coefficient"],
                clip_epsilon=old["clip_epsilon"], checkpoint_every=old["checkpoint_every"],
                budget_seconds=old["budget_seconds"], initial_validation_metrics=initial,
                closure_authorization=run, checkpoint_save_budget_seconds=old["checkpoint_save_budget_seconds"],
                closure_selection=old["closure_selection"], reference_execution=old["reference_execution"])
    finally:
        _release(reference)
        _release(actor)


def audit_training(run, original_dir, training, plan):
    traces = _rows(run / "optimizer/rollouts.jsonl")
    previous = _rows(original_dir / "rollouts.jsonl")
    groups = _rows(run / "optimizer/group_diagnostics.jsonl")
    config = plan["rl"]
    expected = config["max_updates"] * config["roots_per_batch"] * config["rollouts_per_root"]
    if len(traces) != len(previous) or len(traces) != expected:
        raise ValueError("Training rollout budgets differ")
    if training["resource_counters"]["updates"] != config["max_updates"]:
        raise ValueError("Incomplete optimizer run")
    if training["frozen_before"] != training["frozen_after"] or training["reference_before"] != training["reference_after"]:
        raise ValueError("Frozen parameters changed")
    evaluator = Evaluator()
    by_group = defaultdict(list)
    for row, old in zip(traces, previous, strict=True):
        rollout, old_rollout = row["rollout"], old["rollout"]
        for key in ("root_id", "latent_world_id", "rollout_seed", "split"):
            if rollout[key] != old_rollout[key]:
                raise ValueError(f"Training comparison differs on {key}")
        if row["update"] == 1 and any(a["generation"]["completion_ids"] != b["generation"]["completion_ids"]
                                      for a, b in zip(rollout["phases"], old_rollout["phases"], strict=True)):
            raise ValueError("First group differs before the first reward update")
        evidence = read(Path(rollout["memory_path"]).with_suffix(".json"))
        trace = evidence["trace"]
        current, future = evaluator.score_current(trace), evaluator.score_future(trace)
        unsupported = evaluator.score_memory(trace)["unsupported_operations"]
        joint = joint_success(current["success"], future["success"],
                              (trace["write_result"] or {}).get("valid") is True, unsupported)
        reward = 2 * joint - .10 * evidence["metrics"]["questions"] - .50 * unsupported - .20 * len(trace["errors"])
        if rollout["r2"] != 2 * joint or rollout["r3"] != 0 or not isclose(row["return"], reward, abs_tol=1e-9):
            raise ValueError("Joint objective disagrees with saved environment trace")
        by_group[row["group_id"]].append((joint, reward, row["advantage"]))
    if len(groups) != len(by_group):
        raise ValueError("Missing group diagnostics")
    for group in groups:
        rows = by_group[group["group_id"]]
        joints, rewards, advantages = map(list, zip(*rows))
        if group["task_returns"] != [2 * joint for joint in joints] or any(
            not isclose(a, b, abs_tol=1e-9) for a, b in zip(advantages, group_advantages(rewards), strict=True)
        ):
            raise ValueError("Wrong group objective or advantage")
    return {"updates": training["resource_counters"]["updates"], "trajectories": len(traces),
            "groups": len(groups), "groups_with_joint_signal": sum(g["std_task"] > 0 for g in groups),
            "groups_with_any_signal": sum(g["std_total"] > 0 for g in groups),
            "rewarded_trajectories": sum(j for rows in by_group.values() for j, _, _ in rows),
            "nonzero_gradient_steps": sum(row["grad_norm"] > 0 for row in training["history"][1:]),
            "adapter_changed": training["trainable_before"] != training["trainable_after"],
            "first_group_completions_matched": True}


def evaluation_summary(path, bank, selection_path, method, seed):
    result, _ = _panel(path, bank, "evaluation", seed, selection_path=selection_path, method=method)
    rows = _rows(read(path)["paths"]["trajectories"])
    for row in rows:
        row["metrics"]["J_faithful"] = joint_success(
            row["metrics"]["C2"], row["metrics"]["FJS"], row["phases"]["write"]["error"] is None,
            row["metrics"]["unsupported_write_count"])
    from research.competence_evaluation import aggregate_rows
    strict = aggregate_rows(rows)
    return {"summary": str(path), "metrics": {**result["metrics"], "J_faithful": strict["metrics"]["J_faithful"]},
            "by_root": {root: v["J_primary"] for root, v in result["by_root"].items()}}


def execute(prior, run):
    prior, run = prior.resolve(), run.resolve()
    require_training_authorization(prior)
    baseline_plan = read(prior / "PILOT_PLAN.json")
    bank = Path(baseline_plan["bank_dir"])
    splits, _ = load_and_check(bank)
    if any(digest(rows) != baseline_plan["input_sha256"][role] for role, rows in splits.items()):
        raise ValueError("Original bank changed")
    old_dir = prior / "continuations/B2-L/optimizer"
    old = read(old_dir / "training.json")["plan"]
    old["temperature"] = baseline_plan["rl"]["temperature"]
    chosen = read(prior / "CALIBRATION.json")["selected"]
    checkpoint = checkpoint_evidence(chosen["checkpoint_path"])
    if checkpoint != chosen["checkpoint"]:
        raise ValueError("Weak SFT weights changed")
    cohort = training_cohort(splits["train"])
    if digest(cohort) != baseline_plan["rl"]["cohort_sha256"]:
        raise ValueError("Original training schedule changed")
    prior_comparison = prior / "COMPARISON.json"
    plan = {
        "created_at": now(), "scope": "exploratory follow-up on already evaluated heldback bank",
        "baseline_run": str(prior), "bank_dir": str(bank), "seed": old["seed"], "parent": checkpoint, "rl": old,
        "cohort_sha256": digest(cohort), "heldback_sha256": digest(splits["evaluation"]),
        "source_sha256": file_hash(Path(__file__)), "prior_comparison_sha256": file_hash(prior_comparison),
        "reward": "2 * I(C2 and FJS and valid WRITE and zero unsupported writes) minus original penalties",
        "storage_mapping": "optimizer r2 stores entire joint task score, r3 is zero; environment traces keep raw S2/S3",
        "selection": "original validation J, step 0 versus terminal, ties retain step 0",
    }
    run.mkdir(parents=True)
    authorize(run, bank, plan)
    initial = dict(read(chosen["validation_summary"])["metrics"])
    initial["utility"] = initial["J_primary"]
    result = train(run, checkpoint, cohort, splits["validation"], initial, old)
    audit = audit_training(run, old_dir, result, plan)
    write_json(run / "QA_TRAINING.json", audit)
    print("training verified", audit, flush=True)
    bank_rows = splits["evaluation"]
    terminal = "B2-L-joint-terminal"
    final_step = result["resource_counters"]["updates"]
    selection = run / "SELECTION.json"
    write_json(selection, {"version": "competence_selection_v1", "created_at": now(),
        "selection_complete": True, "selection_split": "validation", "selected_before_evaluation": True,
        "heldback_episodes_sha256": digest(bank_rows), "rows": [
            selection_record("B2-L-joint", result["selected_checkpoint"], result["best_step"]),
            selection_record(terminal, result["final_checkpoint"], final_step)]})
    output = run / "evaluation" / terminal
    policy = SpecLocalPolicy.from_checkpoint(result["final_checkpoint"], trainable=False, device="cuda")
    try:
        summary = evaluate_selected(policy, bank_rows, output, prompt_root=PROMPTS, method=terminal,
            training_seed=old["seed"], role="evaluation", include_interventions=False, include_teacher=False,
            protocol={"thresholds": DEFAULT_THRESHOLDS, "heldback_authorization": {
                "selection_manifest_path": str(selection), "selection_manifest_sha256": file_hash(selection),
                "locked_bank_sha256": digest(bank_rows)}})
        if summary["status"] != "completed":
            raise RuntimeError(f"Evaluation incomplete: {summary['failure']}")
    finally:
        _release(policy)
    prior_summaries = read(prior_comparison)["summaries"]
    panels = {}
    for method in ("B1-weak", "B3", "B2-L-terminal"):
        panels[method] = evaluation_summary(Path(prior_summaries[method]["summary"]),
                                            bank_rows, prior / "SELECTION.json", method, old["seed"])
    panels[terminal] = evaluation_summary(output / "summary.json", bank_rows, selection, terminal, old["seed"])
    selected = terminal if result["best_step"] == final_step else "B1-weak"
    panels["B2-L-joint"] = {**panels[selected], "same_policy_as": selected}
    paired = {}
    for method in ("B1-weak", "B3", "B2-L-terminal"):
        left, right = panels[terminal], panels[method]
        differences = [left["by_root"][root] - right["by_root"][root] for root in left["by_root"]]
        paired[method] = {"delta_J_pp": 100 * mean(differences), "improved": sum(d > 0 for d in differences),
                          "worse": sum(d < 0 for d in differences), "tied": sum(d == 0 for d in differences)}
    if file_hash(Path(__file__)) != plan["source_sha256"] or file_hash(prior_comparison) != plan["prior_comparison_sha256"]:
        raise ValueError("Plan inputs changed during the comparison")
    comparison = {"status": "verified", "seed": old["seed"], "selected_step": result["best_step"],
                  "validation_J": {str(row["step"]): row["validation"]["J_primary"]
                                   for row in result["history"] if "validation" in row},
                  "evaluation_roots": len(panels[terminal]["by_root"]), "evaluation_episodes": len(bank_rows),
                  "training": audit, "panels": panels, "paired_terminal_vs": paired}
    write_json(run / "COMPARISON.json", comparison)
    print("comparison verified", {method: panel["metrics"]["J_primary"] for method, panel in panels.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute(args.baseline, args.output)


if __name__ == "__main__":
    main()
