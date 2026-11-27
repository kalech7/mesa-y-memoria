"""Verify repaired semantics and prospective data without training or inference.

python -m research.verify_synthetic_repair --bank research/data/development_v2_20261024_001/validation.jsonl --output reports/synthetic_repair_v2_20261024_001/verification_001
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile

from research.src.data.development_v2 import load_historical
from research.src.data.semantic_partition_v2 import assert_semantic_split_isolation_v2, SemanticSplitLeakage
from research.src.evaluation.runner import run_episode
from research.src.policy.rule_controller import RuleController
from research.src.scoring.evidence import Evaluator, EVALUATOR_VERSION
from research.src.scoring.reward import rewards
from research.reliable_evaluation import verify_sqlite

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def expected_reference_success(episode, action):
    if episode["truth"]["decisive_slot"] is None or action == "NATURAL_REFERENCE":
        return 1
    if action == "ASK:TEMPORAL_SCOPE":
        return 1
    if action == "ASK:VALID_UNTIL" and episode["truth"]["scope_kind"] == "interval":
        return 1
    return 0


def verify(bank, output, *, historical=None, saved_run=None):
    bank, output = Path(bank).resolve(), Path(output).resolve()
    historical = Path(historical or ROOT / "research/data/bank_1_1").resolve()
    saved_run = Path(saved_run or ROOT / "research/runs/evaluacion_confiable_20261024_001").resolve()
    # Historical results can never be destinations, even through an alias.
    forbidden = (ROOT / "research/data/bank_1_1", ROOT / "research/runs", bank.parent,
                 ROOT / "reports/audit_synthetic_v1_20261024_001")
    if any(output.is_relative_to(p.resolve()) for p in forbidden):
        raise ValueError("Choose a new repair report directory outside historical data/results")
    if output.exists():
        raise FileExistsError("Use a fresh report directory; never overwrite verification results")
    output.mkdir(parents=True)
    (output / "tmp").mkdir()
    (output / "states").mkdir()
    tempfile.tempdir = str(output / "tmp")
    manifest = json.loads(bank.with_name("manifest.json").read_text())
    if sha(bank) != manifest["bank_sha256"]:
        raise ValueError("Development bank changed after freezing")
    episodes = [json.loads(line) for line in bank.open()]
    old = load_historical(historical)
    before = {name: sha(historical / (name + ".jsonl")) for name in old}
    plan = {"bank": str(bank), "bank_sha256": sha(bank), "historical_hashes_before": before,
            "new_evaluator_version": EVALUATOR_VERSION, "reference_is_B4_not_B1": True,
            "training": False, "new_Qwen_calls": 0,
            "saved_B1_run": str(saved_run), "saved_trajectory_sha256": sha(saved_run / "diagnostico_50.jsonl"),
            "reward_formula_sha256": sha(ROOT / "research/src/scoring/reward.py"),
            "scope": "Deterministic infrastructure check and symmetric rescoring; no new B1 performance measurement."}
    write(output / "plan.json", plan)
    isolation = assert_semantic_split_isolation_v2({"validation_v2": episodes}, historical_splits=old)
    write(output / "new_data_isolation.json", isolation)
    try:
        assert_semantic_split_isolation_v2({"train": old["train"], "validation": old["validation"]})
    except SemanticSplitLeakage as exc:
        write(output / "historical_training_rejected.json", {"status": "verified", "reason": "SEMANTIC_SPLIT_LEAKAGE",
                                                            "findings": exc.findings, "model_loaded": False})
    else:
        raise AssertionError("Historical leaking train/validation was accepted")

    reference_errors, summary_rows = [], []
    with (output / "reference_trajectories.jsonl").open("x") as trace_file:
        for number, episode in enumerate(episodes):
            for action in ("NATURAL_REFERENCE", "PROCEED", "ASK:CURRENT_CONTEXT", "ASK:GENERIC", "ASK:TEMPORAL_SCOPE", "ASK:VALID_UNTIL"):
                options = {} if action == "NATURAL_REFERENCE" else ({"method": "C-NEVER"} if action == "PROCEED" else
                           {"intervention": "I2", "forced_slot": action.split(":", 1)[1], "method": "B4-forced"})
                row = run_episode(deepcopy(episode), RuleController(),
                    db_path=output / "states" / (str(number) + "_" + action.replace(":", "_") + ".sqlite"), **options)
                row["repair_validation_action"] = action
                row["performance_author"] = "B4_reference_not_Qwen"
                expected = expected_reference_success(episode, action)
                checked = {"C2": row["metrics"]["C2"], "FJS": row["metrics"]["FJS"],
                           "unsupported_operations": row["metrics"]["unsupported_operations"],
                           "invalid_outputs": row["metrics"]["invalid_outputs"]}
                wanted = {"C2": 1, "FJS": expected, "unsupported_operations": 0, "invalid_outputs": 0}
                if checked != wanted:
                    reference_errors.append({"row": len(summary_rows) + 1, "root_id": episode["root_id"],
                                             "world": episode["latent_world_id"], "realization": episode["realization_id"],
                                             "action": action, "expected": wanted, "observed": checked})
                trace_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                summary_rows.append({"root_id": episode["root_id"], "world": episode["latent_world_id"],
                                     "realization": episode["realization_id"], "stratum": episode["authoring"]["stratum"],
                                     "action": action, "expected_FJS": expected, **checked})
    with (output / "reference_action_checks.csv").open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(summary_rows[0]));writer.writeheader();writer.writerows(summary_rows)
    write(output / "reference_errors.json", reference_errors)

    previous_manifest = json.loads((saved_run / "completion.json").read_text())
    assert previous_manifest["status"] == "passed" and previous_manifest["optimizer_steps"] == 0
    for name, expected in previous_manifest["artifacts_sha256"].items():
        path = (saved_run / name).resolve()
        assert path.is_relative_to(saved_run) and sha(path) == expected, name
    evaluator, changed, saved_count, natural = Evaluator(), [], 0, []
    sqlite_source = ROOT / "research/runs/diagnostico_senal_50_20261024_001"
    with (output / "saved_B1_symmetric_rescoring.jsonl").open("x") as h:
        for line, text in enumerate((saved_run / "diagnostico_50.jsonl").open(), 1):
            row = json.loads(text)
            trace = row["trace"]
            scored = {"current": evaluator.score_current(trace), "future": evaluator.score_future(trace),
                      "memory": evaluator.score_memory(trace)}
            reward = rewards(scored["current"]["utility"], scored["future"]["utility"], row["metrics"]["questions"],
                             scored["memory"]["unsupported_operations"], row["metrics"]["invalid_outputs"])
            verify_sqlite(sqlite_source, row)
            before_metrics = [row["metrics"]["C2"], row["metrics"]["FJS"], row["rewards_by_component"]["R_short"], row["rewards_by_component"]["R_long"]]
            after_metrics = [scored["current"]["success"], scored["future"]["success"], reward["R_short"], reward["R_long"]]
            result = {"source_line": line, "source_case": row["diagnostic_case_id"], "action": row["action_id"],
                      "writer": row["writer"], "prior_evaluator_version": row["evaluator_version"],
                      "evaluator_version": EVALUATOR_VERSION, "before_metrics": before_metrics,
                      "after_metrics": after_metrics, "new_scores": scored, "new_rewards": reward,
                      "actor_and_memory_changed": False}
            if before_metrics != after_metrics: changed.append(result)
            if row["action_id"] == "NATURAL": natural.append(after_metrics)
            h.write(json.dumps(result, ensure_ascii=False) + "\n")
            saved_count += 1
    after = {name: sha(historical / (name + ".jsonl")) for name in old}
    assert before == after and sha(saved_run / "diagnostico_50.jsonl") == plan["saved_trajectory_sha256"]
    assert saved_count == 935 and len(natural) == 85 and len(summary_rows) == 720
    result = {"status": "verified" if not reference_errors else "failed", "evaluator_version": EVALUATOR_VERSION,
              "new_development_roots": len({e["root_id"] for e in episodes}), "new_development_episodes": len(episodes),
              "reference_trajectories": len(summary_rows), "reference_expected_result_mismatches": len(reference_errors),
              "new_B1_performance": "not_executed", "historical_B1_trajectories_rescored": saved_count,
              "historical_SQLite_verified_read_only": saved_count,
              "historical_B1_natural_C2": sum(r[0] for r in natural), "historical_B1_natural_FJS": sum(r[1] for r in natural),
              "historical_B1_natural_denominator": len(natural), "changed_old_C2_FJS_returns": len(changed),
              "historical_banks_unchanged": before == after, "historical_scores_unchanged_on_disk": True,
              "new_Qwen_calls": 0, "training_steps": 0, "reward_formula_modified": False,
              "note": "Reference successes are contract checks, not B1 scores or proof of learning/generalization."}
    write(output / "changed_saved_scores.json", changed)
    write(output / "verification.json", result)
    write(output / "completion.json", {**result, "artifacts_sha256": {
        p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != "completion.json"}})
    print(json.dumps(result, ensure_ascii=False))
    if reference_errors:
        raise RuntimeError("New development has reference contract mismatches; inspect reference_errors.json")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--historical-directory")
    parser.add_argument("--saved-run")
    args = parser.parse_args(argv)
    return verify(args.bank, args.output, historical=args.historical_directory, saved_run=args.saved_run)


if __name__ == "__main__":
    main()
