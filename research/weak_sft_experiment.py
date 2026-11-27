"""Exploratory weak-SFT calibration and matched continuation on a new bank."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

from research.competence_campaign import _junit
from research.competence_data import load_and_check
from research.competence_evaluation import DEFAULT_THRESHOLDS, evaluate_selected
from research.competence_training import (
    PROMPTS, SelectionValidation, _release, checkpoint_evidence, historical_b1,
    profile_teacher, verify_initialization,
)
from research.imperfect_sft import load_examples, prepare_bundle
from research.src.artifacts import digest, file_hash, now, write_json
from research.src.closure_protocol import execution_context, record_stage, source_hashes
from research.src.policy.llm_controller import SpecLocalPolicy
from research.src.training.episodic_grpo import train_grpo
from research.src.training.rollouts import make_rollout_collector
from research.src.training.sft import train_sft

SEED = 101
TARGET = .50
TOLERANCE = .10
MAX_INVALID = .20
RL_UPDATES = 12
RL_BUDGET = 720
TEMPERATURE = 1.3
HISTORY = Path("reports/imperfections_campaign_20261101_001")
SELECTION = {"utility_tolerance": 1e-6, "validity_guards": {"utility": 1.}}


def read(path):
    return json.loads(Path(path).read_text())


def authorize(output, bank_dir, plan):
    technical = HISTORY / "technical_cuda/technical_smoke.json"
    qa = read(HISTORY / "QA_TECHNICAL.json")
    tests = HISTORY / "qa_preflight.xml"
    audit = bank_dir / "audit_contracts.json"
    if (qa["status"] != "passed" or qa["technical_report_sha256"] != file_hash(technical)
            or read(audit)["status"] != "verified"):
        raise ValueError("Verified CUDA and data evidence required")
    _junit(tests)
    plan_path = output / "PILOT_PLAN.json"
    write_json(plan_path, plan)
    write_json(output / "MANIFIESTO_V4.json", {
        "protocol_version": "closure_v4", "purpose": "exploratory_weak_sft_then_grpo",
        "code_snapshot_sha256": digest(source_hashes()),
        "pilot_plan_sha256": file_hash(plan_path), "stages": {},
    })
    record_stage(output, "inventory", {"status": "passed"}, [plan_path, technical, bank_dir / "manifest.json"])
    record_stage(output, "test-environment", {"status": "passed"}, [tests, audit])
    record_stage(output, "test-training", {"status": "passed"}, [tests, technical])
    record_stage(output, "diagnose", {"status": "passed", "critical_checks_passed": True,
                 "scope": "Technical feasibility only. Exploratory continuation measures task signal."},
                 [audit, technical, plan_path])


def training_cohort(rows):
    by_family = defaultdict(dict)
    for row in sorted(rows, key=lambda item: (item["root_id"], item["latent_world_id"], item["realization_id"])):
        by_family[row["scenario_family"]].setdefault(row["root_id"], row)
    rng = random.Random(7331)
    choices = []
    for _, roots in sorted(by_family.items()):
        cohort = list(roots.values())
        rng.shuffle(cohort)
        choices.append(cohort)
    return [row for batch in zip(*choices) for row in batch][:RL_UPDATES]


class JointValidation(SelectionValidation):
    """Keep the original episode metrics and select continuation on J."""

    def __call__(self, policy):
        metrics = super().__call__(policy)
        metrics["utility"] = metrics["J_primary"]
        return metrics


def validation_callback(episodes, destination, method):
    return JointValidation(episodes, destination, {
        "capability_thresholds": DEFAULT_THRESHOLDS,
        "budgets": {"max_validation_evaluations": 8, "validation_seconds": 3600},
    }, method, SEED)


def sft_inputs(output, episodes, rate, actor):
    bundle = output / "demonstrations"
    prepare_bundle(episodes, bundle, rate=rate, seed=7331)
    examples, _ = load_examples(bundle, train=episodes, expected_rate=rate, expected_seed=7331)
    # Use natural interactions only so imperfect targets are not diluted by
    # four clean forced-route demonstrations per episode.
    natural = [row for row in examples if row["route"] == "natural"]
    write_json(output / "token_profile.json", profile_teacher(actor, natural))
    return natural


def calibrate_attempt(run, attempt, rate, period, limit, parent, splits, initial_metrics=None):
    destination = run / "calibration" / f"attempt_{attempt}"
    destination.mkdir(parents=True)
    actor = SpecLocalPolicy.from_checkpoint(parent, trainable=True, device="cuda")
    try:
        examples = sft_inputs(destination, splits["train"], rate, actor)
        print("calibrating", attempt, "rate", rate, "updates", limit, flush=True)
        callback = validation_callback(splits["validation"], destination / "validation", "B1-calibration")
        if initial_metrics is not None and "J_primary" not in initial_metrics:
            raise ValueError("Missing initial J")
        with execution_context(run, "training"):
            result = train_sft(actor, examples, destination / "optimizer", callback, seed=SEED,
                updates=limit, epochs=1, checkpoint_every=period, learning_rate=1e-4,
                continuation=initial_metrics is not None, initial_validation_metrics=initial_metrics,
                budget_seconds=480, checkpoint_save_budget_seconds=120,
                prompt_root=PROMPTS, closure_authorization=run, closure_selection=SELECTION)
    finally:
        _release(actor)
    result_path = destination / "optimizer" / "training.json"
    candidates = []
    index = int(initial_metrics is None)
    for entry in result["history"]:
        step, metrics = entry["step"], entry.get("validation")
        if not step or metrics is None:
            continue
        checkpoint = result_path.parent / f"step_{step:06d}"
        summary = destination / "validation" / f"evaluation_{index:02d}" / "summary.json"
        index += 1
        candidate = {
            "attempt": attempt, "step": step, "checkpoint_path": str(checkpoint),
            "checkpoint": checkpoint_evidence(checkpoint),
            "validation_summary": str(summary), "validation_summary_sha256": file_hash(summary),
            "training_summary": str(result_path), "training_summary_sha256": file_hash(result_path),
            "j": metrics["J_primary"], "invalid": metrics["any_invalid_output"],
            "demonstration_rate": rate, "bundle_manifest": str(destination / "demonstrations/manifest.json"),
        }
        candidates.append(candidate)
        print("candidate", attempt, step, "J", candidate["j"], "invalid", candidate["invalid"], flush=True)
    return candidates, result["history"][0]["validation"]


def eligible(row):
    return abs(row["j"] - TARGET) <= TOLERANCE + 1e-12 and row["invalid"] <= MAX_INVALID + 1e-12


def calibration_key(row):
    return (not eligible(row), round(abs(row["j"] - TARGET), 12),
            row["invalid"], row["step"], row["attempt"])


def calibrate(run, parent, splits):
    candidates, baseline = calibrate_attempt(run, 0, .65, 80, 320, parent, splits)
    if not any(eligible(row) for row in candidates):
        if any(row["j"] >= TARGET for row in candidates):
            extra, _ = calibrate_attempt(run, 1, .85, 24, 120, parent, splits, baseline)
        else:
            extra, _ = calibrate_attempt(run, 1, .25, 80, 320, parent, splits, baseline)
        candidates.extend(extra)
    chosen = min(candidates, key=calibration_key)
    record = {
        "created_at": now(), "candidates": candidates, "selected": chosen,
        "selected_checkpoint_path": chosen["checkpoint_path"],
        "target_met": abs(chosen["j"] - TARGET) <= TOLERANCE + 1e-12,
        "format_guard_met": chosen["invalid"] <= MAX_INVALID + 1e-12,
    }
    write_json(run / "CALIBRATION.json", record)
    print("weak checkpoint", chosen["checkpoint_path"], "J", chosen["j"], "invalid", chosen["invalid"], flush=True)
    return chosen


def train_rl(run, chosen, rows, validation, initial):
    output = run / "continuations/B2-L"
    output.mkdir(parents=True)
    parent = chosen["checkpoint_path"]
    actor = SpecLocalPolicy.from_checkpoint(parent, trainable=True, device="cuda")
    reference = None
    try:
        reference = SpecLocalPolicy.from_checkpoint(parent, trainable=False, device="cpu")
        actor.config.temperature = reference.config.temperature = TEMPERATURE
        write_json(output / "initialization.json", verify_initialization(actor, parent, reference))
        with execution_context(run, "training"):
            return train_grpo(actor, reference, rows,
                make_rollout_collector(output / "environment_rollouts", method="B2-L", prompt_root=PROMPTS),
                output / "optimizer", validation_callback(validation, output / "validation", "B2-L"),
                horizon="long", seed=SEED, updates=RL_UPDATES, roots_per_batch=1, rollouts_per_root=4,
                learning_rate=1e-5, kl_coefficient=.02, clip_epsilon=.2,
                checkpoint_every=RL_UPDATES, budget_seconds=RL_BUDGET,
                initial_validation_metrics=initial, closure_authorization=run,
                checkpoint_save_budget_seconds=120, closure_selection=SELECTION,
                reference_execution="cpu_storage_gpu_compute")
    finally:
        _release(reference)
        _release(actor)


def train_more_sft(run, chosen, splits, initial, seconds):
    output = run / "continuations/B3"
    output.mkdir(parents=True)
    parent = chosen["checkpoint_path"]
    actor = SpecLocalPolicy.from_checkpoint(parent, trainable=True, device="cuda")
    try:
        bundle = Path(chosen["bundle_manifest"]).parent
        examples, _ = load_examples(bundle, train=splits["train"], expected_rate=chosen["demonstration_rate"], expected_seed=7331)
        natural = [row for row in examples if row["route"] == "natural"]
        write_json(output / "initialization.json", verify_initialization(actor, parent))
        with execution_context(run, "training"):
            return train_sft(actor, natural, output / "optimizer",
                validation_callback(splits["validation"], output / "validation", "B3"),
                continuation=True, epochs=1, checkpoint_every=100_000, seed=SEED,
                learning_rate=1e-4, budget_seconds=seconds, prompt_root=PROMPTS,
                initial_validation_metrics=initial, closure_authorization=run,
                checkpoint_save_budget_seconds=120, closure_selection=SELECTION)
    finally:
        _release(actor)


def selection_record(method, checkpoint, step):
    evidence = checkpoint_evidence(checkpoint)
    manifest = read(Path(checkpoint) / "local_checkpoint.json")
    artifacts = evidence["artifacts_sha256"]
    fingerprint = digest({"base": manifest["base_artifacts_sha256"],
        "precision": manifest["config"]["precision"],
        "artifacts": {name: value for name, value in artifacts.items() if name != "README.md"}})
    return {"method": method, "training_seed": SEED, "step": step, "checkpoint_path": evidence["path"],
            "checkpoint_manifest_sha256": evidence["manifest_sha256"],
            "adapter_sha256": artifacts["adapter_model.safetensors"], "policy_fingerprint": fingerprint}


def evaluate(run, bank, selection):
    protocol = {"thresholds": DEFAULT_THRESHOLDS, "heldback_authorization": {
        "selection_manifest_path": str(selection), "selection_manifest_sha256": file_hash(selection),
        "locked_bank_sha256": digest(bank),
    }}
    summaries, equivalent = {}, {}
    for record in read(selection)["rows"]:
        method, key = record["method"], record["policy_fingerprint"]
        if key in equivalent:
            source = equivalent[key]
            summaries[method] = {**summaries[source], "same_adapter_as": source}
            continue
        actor = SpecLocalPolicy.from_checkpoint(record["checkpoint_path"], trainable=False, device="cuda")
        try:
            destination = run / "evaluation" / method
            result = evaluate_selected(actor, bank, destination, prompt_root=PROMPTS, method=method,
                training_seed=SEED, role="evaluation", protocol=protocol,
                include_interventions=False, include_teacher=False)
            if result["status"] != "completed":
                raise RuntimeError(f"Incomplete evaluation for {method}: {result['failure']}")
            summaries[method] = {"summary": str(destination / "summary.json"), "metrics": result["metrics"],
                                 "by_root": result["by_root"], "families": result["families"]}
            equivalent[key] = method
            print("evaluated", method, result["completed_episodes"], "J", result["metrics"]["J_primary"], flush=True)
        finally:
            _release(actor)
            del actor
    return summaries


def execute(bank_dir, run):
    bank_dir, run = bank_dir.resolve(), run.resolve()
    splits, manifest = load_and_check(bank_dir)
    parent = historical_b1(SEED)
    rl_rows = training_cohort(splits["train"])
    if len(rl_rows) != RL_UPDATES:
        raise ValueError("Insufficient stratified TRAIN roots")
    plan = {
        "created_at": now(), "scope": "exploratory, synthetic richer conversations, one fitted seed",
        "bank_dir": str(bank_dir), "bank_manifest_sha256": file_hash(bank_dir / "manifest.json"),
        "seed": SEED, "parent": checkpoint_evidence(parent),
        "runner_path": str(Path(__file__).resolve()), "script_sha256": file_hash(Path(__file__)),
        "input_sha256": {name: digest(rows) for name, rows in splits.items()},
        "heldback_sha256": digest(splits["evaluation"]),
        "calibration_target": TARGET, "calibration_tolerance": TOLERANCE, "calibration_max_invalid": MAX_INVALID,
        "calibration": {"primary": {"rate": .65, "every": 80, "steps": 320},
            "fallback_if_no_eligible_and_any_at_least_half": {"rate": .85, "every": 24, "steps": 120},
            "fallback_if_no_eligible_and_all_below_half": {"rate": .25, "every": 80, "steps": 320},
            "ranking": "Eligible J .40 to .60 and invalid <= .20 first, nearest J to .50, least invalid, earliest step, attempt ordinal. Step zero excluded.",
            "demo_scope": "Imperfect natural-route only. Fixed seed 7331. Validation and rewards clean."},
        "rl": {"max_updates": RL_UPDATES, "roots_per_batch": 1, "rollouts_per_root": 4,
            "horizon": "long", "temperature": TEMPERATURE, "budget_seconds": RL_BUDGET,
            "learning_rate": 1e-5, "kl_coefficient": .02, "clip_epsilon": .2,
            "root_order": [row["root_id"] for row in rl_rows], "cohort_sha256": digest(rl_rows)},
        "more_sft": {"budget": "match GRPO measured optimization seconds, excluding validation and save",
            "learning_rate": 1e-4, "demonstrations": "same natural imperfect mixture as weak SFT"},
        "continuation_selection": "Step zero vs terminal based on validation J, ties keep step zero. Terminal panels fixed diagnostics.",
        "evaluation": "Once all checkpoints sealed, greedy natural O2 on same held-back bank; J by paired root/family.",
        "data_sizes": manifest["sizes"],
    }
    run.mkdir(parents=True)
    authorize(run, bank_dir, plan)
    started = time.monotonic()
    chosen = calibrate(run, parent, splits)
    initial = dict(read(chosen["validation_summary"])["metrics"])
    initial["utility"] = initial["J_primary"]
    print("starting B2-L", flush=True)
    rl = train_rl(run, chosen, rl_rows, splits["validation"], initial)
    if not rl["resource_counters"]["updates"]:
        raise RuntimeError("No GRPO optimizer update completed")
    seconds = rl["resource_counters"]["training_seconds"]
    print("completed B2-L", rl["resource_counters"]["updates"], "selected", rl["best_step"], flush=True)
    print("starting B3", seconds, "optimization seconds", flush=True)
    sft = train_more_sft(run, chosen, splits, initial, seconds)
    print("completed B3", sft["resource_counters"]["updates"], "selected", sft["best_step"], flush=True)
    records = [selection_record("B1-weak", chosen["checkpoint_path"], chosen["step"])]
    training = {}
    for method, result in (("B3", sft), ("B2-L", rl)):
        records.append(selection_record(method, result["selected_checkpoint"], result["best_step"]))
        records.append(selection_record(method + "-terminal", result["final_checkpoint"], result["resource_counters"]["updates"]))
        training[method] = str(run / "continuations" / method / "optimizer/training.json")
    selection = run / "SELECTION.json"
    write_json(selection, {"version": "competence_selection_v1", "created_at": now(),
        "selection_complete": True, "selection_split": "validation", "selected_before_evaluation": True,
        "heldback_episodes_sha256": plan["heldback_sha256"], "rows": records})
    summaries = evaluate(run, splits["evaluation"], selection)
    write_json(run / "COMPARISON.json", {"status": "completed", "training": training, "summaries": summaries,
        "evaluation_roots": len({row["root_id"] for row in splits["evaluation"]}),
        "evaluation_episodes": len(splits["evaluation"]), "wall_seconds": time.monotonic() - started})
    print("comparison completed", run, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    execute(args.data, args.output)


if __name__ == "__main__":
    main()
