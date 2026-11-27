"""Verify the held-out Qwen comparison and compute intervals grouped by dialogue."""
from __future__ import annotations

import argparse
import json
from math import isclose
from pathlib import Path

from research.src.analysis.paired_statistics import paired_bootstrap
from research.src.artifacts import digest, file_hash, write_json
from research.tesis_meal_cases import CAPTURE, LETTERS
from research.tesis_meal_train import ARMS, BASELINE, COMMON, SEED, rank, scores, summary


def read(path):
    return json.loads(path.read_bytes())


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def audit(run):
    plan_path = run / "PLAN.json"
    selection_path = run / "SELECTION.json"
    results_path = run / "RESULTS.json"
    plan, selection, results = (read(path) for path in (plan_path, selection_path, results_path))
    if (selection["plan_sha256"] != file_hash(plan_path)
            or results["selection_sha256"] != file_hash(selection_path) or selection["split"] != "validation"):
        raise ValueError("Evaluation must reference a locked validation selection")
    for name, expected in plan["source_sha256"].items():
        if file_hash(Path(__file__).with_name(name)) != expected:
            raise ValueError("Training sources differ from the frozen plan")
    bank = {}
    for split, expected in plan["bank_sha256"].items():
        path = run / f"{split}.jsonl"
        if file_hash(path) != expected:
            raise ValueError("Bank differs from the frozen plan")
        bank[split] = read_rows(path)
    for split, rows in bank.items():
        for row in rows:
            if (row["split"] != split or row["id"] != digest({"turns": row["turns"], "options": row["options"]})
                    or not row["accepted"]):
                raise ValueError("Malformed fixed case")
    identities = [{digest(row["turns"]) for row in rows} for rows in bank.values()]
    if any(a & b for i, a in enumerate(identities) for b in identities[i+1:]):
        raise ValueError("Public dialogues crossed a split")
    if CAPTURE not in {text for row in bank["train"] for text in row["turns"]}:
        raise ValueError("Reported user case was omitted from training bank")
    for key in (BASELINE, COMMON):
        trace = read_rows(run / f"{key}-validation.jsonl")
        if [r["id"] for r in trace] != [r["id"] for r in bank["validation"]]:
            raise ValueError("Mismatched validation")
        if summary(trace) != selection["methods"][key]["validation"]:
            raise ValueError("Validation scores changed")
    group_counts = {}
    schedules = {}
    for method in ARMS:
        chosen = selection["methods"][method]
        step = chosen["step"]
        if step != max((int(i) for i in chosen["validation"]), key=lambda i: rank(chosen["validation"][str(i)], i)):
            raise ValueError("Validation selection differs from the declared rank")
        steps = sorted(int(i) for i in chosen["validation"] if i != "0")
        logs = [row for checkpoint in steps for row in read_rows(run / f"{method}-train-{checkpoint}.jsonl")]
        schedules[method] = [row["cases"] for row in logs]
        if len(logs) != plan["updates_per_arm"] or any(
                len(row["cases"]) != plan["batch"] for row in logs):
            raise ValueError("Training budget differs from the plan")
        if method == "grpo":
            training = {row["id"]: row for row in bank["train"]}
            from research.tesis_meal_train import rewards
            for row in logs:
                expected = [[rewards(training[item])[index] for index in actions]
                            for item, actions in zip(row["cases"], row["sampled"], strict=True)]
                if (any(len(samples) != plan["draws"] for samples in expected)
                        or any(not isclose(wanted, actual, abs_tol=1e-7)
                               for left, right in zip(expected, row["returns"], strict=True)
                               for wanted, actual in zip(left, right, strict=True))):
                    raise ValueError("Incorrect GRPO rewards")
                if row["nonconstant_groups"] != sum(len(set(values)) > 1 for values in expected):
                    raise ValueError("Incorrect reward variation count")
            group_counts = {"total": len(logs) * plan["batch"],
                            "varying": sum(row["nonconstant_groups"] for row in logs)}
        for checkpoint in steps:
            validation = read_rows(run / f"{method}-validation-{checkpoint}.jsonl")
            if ([r["id"] for r in validation] != [r["id"] for r in bank["validation"]]
                    or summary(validation) != chosen["validation"][str(checkpoint)]):
                raise ValueError("A checkpoint was not evaluated on the fixed validation split")
        folder = run / method
        if any(file_hash(folder / filename) != chosen[key] for filename, key in (
                ("local_checkpoint.json", "checkpoint_manifest_sha256"), ("adapter_model.safetensors", "adapter_sha256"))):
            raise ValueError("Adapter differs from the validation selection")
    if schedules["sft"] != schedules["grpo"]:
        raise ValueError("Continuation schedules differ")
    warmup = read_rows(run / "sft-comun-train.jsonl")
    if len(warmup) != plan["warmup"] or any(len(row["cases"]) != plan["batch"] for row in warmup):
        raise ValueError("Warmup differs from the plan")
    sampled = {item for row in warmup for item in row["cases"]} | {item for batch in schedules["sft"] for item in batch}
    train_ids = {row["id"] for row in bank["train"]}
    if not sampled <= train_ids:
        raise ValueError("The optimizer saw a case outside TRAIN")
    paired = []
    guard_free = {}
    evaluation = bank["evaluation"]
    for method, panel in results["panels"].items():
        trace_path = run / f"{method}-evaluation.jsonl"
        trace = read_rows(trace_path)
        if file_hash(trace_path) != panel["traces_sha256"] or summary(trace) != panel["summary"]:
            raise ValueError("Reserved trace differs from its reported scores")
        for row, result in zip(evaluation, trace, strict=True):
            if row["id"] != result["id"] or result["split"] != "evaluation":
                raise ValueError("Held-out panels differ in cases")
            selected = LETTERS.index(result["action"])
            if result["dish"] != row["options"][selected]["name"] or any(
                    result[key] != value for key, value in scores(row, selected).items()):
                raise ValueError("Reported evaluation disagrees with authored constraints")
            if method in ARMS:
                paired.append({"root_id": digest(row["turns"]), "seed": SEED, "method": method,
                               "scenario_family": row["family"], "realization_id": row["id"],
                               "success": result["success"]})
        guard_free[method] = panel["summary"]["all"]
    contrast = paired_bootstrap(paired, "grpo", "sft", "success", confidence=.95, resamples=2000)
    # Re-score the held-out menu without any reference to model predictions.
    fixed = {letter: sum(index in row["accepted"] for row in evaluation) / len(evaluation)
             for index, letter in enumerate(LETTERS)}
    uniform = sum(len(row["accepted"]) / len(LETTERS) for row in evaluation) / len(evaluation)
    return {"status": "verified", "results_sha256": file_hash(results_path),
            "cases": {name: len(rows) for name, rows in bank.items()}, "exact_dialogue_overlap": 0,
            "trajectories": len(evaluation) * len(results["panels"]), "groups": group_counts,
            "capture_in_train": True, "guard_free": guard_free, "grpo_minus_sft_by_dialogue": contrast,
            "position_controls": {"uniform": uniform, "always_pick": fixed}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    write_json(args.run / "AUDIT.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
