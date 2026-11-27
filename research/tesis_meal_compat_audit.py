"""Recompute the order-independent compatibility experiment from archived choices."""
from __future__ import annotations

import argparse
from math import isclose
from pathlib import Path
from statistics import mean

from research.src.analysis.paired_statistics import paired_bootstrap
from research.src.artifacts import digest, file_hash, write_json
from research.tesis_meal_audit import read, read_rows
from research.tesis_meal_cases import CAPTURE, LETTERS
from research.tesis_meal_train import BASELINE, COMMON, SEED, rewards, scores


METHODS = ("sft", "grpo")


def audit(run, previous):
    plan_path, selection_path, result_path = (run / f"{name}.json" for name in ("PLAN", "SELECTION", "RESULTS"))
    plan, selected, results = map(read, (plan_path, selection_path, result_path))
    if (file_hash(plan_path) != selected["plan_sha256"] or file_hash(selection_path) != results["selection_sha256"]
            or selected["split"] != "validation"
            or plan["source_sha256"] != file_hash(Path(__file__).with_name("tesis_meal_compat.py"))):
        raise ValueError("Protocol or selection drift")
    banks = {split: read_rows(run / f"{split}.jsonl") for split in plan["bank_sha256"]}
    dialogues = {split: {digest(case["turns"]) for case in rows} for split, rows in banks.items()}
    prior = {digest(case["turns"]) for split in banks for case in read_rows(previous / f"{split}.jsonl")}
    if (dialogues["evaluation"] & prior or any(left & right for split, left in dialogues.items()
                                              for other, right in dialogues.items() if split != other)):
        raise ValueError("Exact dialogue leakage")
    for split, rows in banks.items():
        if file_hash(run / f"{split}.jsonl") != plan["bank_sha256"][split]:
            raise ValueError("Changed frozen cases")
        if any(row["split"] != split or row["id"] != digest({"turns": row["turns"], "options": row["options"]})
               for row in rows):
            raise ValueError("Malformed case")
    train = {row["id"]: row for row in banks["train"]}
    if not any(CAPTURE in row["turns"] for row in train.values()):
        raise ValueError("User case missing from training")

    def rescore(path, split):
        traces = read_rows(path)
        columns = ("success", "conflict", "unnecessary_clarification", "missed_clarification")
        for case, trace in zip(banks[split], traces, strict=True):
            action = LETTERS.index(trace["action"])
            if (case["id"] != trace["id"] or case["options"][action]["name"] != trace["dish"]
                    or any(trace[key] != value for key, value in scores(case, action).items())):
                raise ValueError("Inconsistent archived choice")
        return {key: mean(row[key] for row in traces) for key in columns}, traces

    schedules, groups = [], []
    for method in METHODS:
        chosen = selected["methods"][method]
        ckpt = run / method
        if (file_hash(ckpt / "adapter_model.safetensors") != chosen["adapter_sha256"]
                or file_hash(ckpt / "local_checkpoint.json") != chosen["checkpoint_manifest_sha256"]):
            raise ValueError("Adapter changed since selection")
        candidates = {0: rescore(run / "sft-comun-validation.jsonl", "validation")[0]}
        ordered = []
        for step in plan["checkpoints"]:
            candidates[step] = rescore(run / f"{method}-validation-{step}.jsonl", "validation")[0]
            for entry in read_rows(run / f"{method}-train-{step}.jsonl"):
                case = train[entry["case"]]
                ordered.append(case["id"])
                if method == "grpo":
                    wanted = [rewards(case)[i] for i in entry["sampled"][0]]
                    measured, = entry["returns"]
                    if (len(wanted) != plan["draws"] or any(not isclose(x, y, abs_tol=1e-7)
                                                            for x, y in zip(wanted, measured, strict=True))):
                        raise ValueError("Incorrect group-relative reward")
                    groups.append(len(set(wanted)) > 1)
        winner = max(candidates, key=lambda x: (candidates[x]["success"], -candidates[x]["conflict"], -x))
        if winner != chosen["step"] or {str(i): value for i, value in candidates.items()} != chosen["validation"]:
            raise ValueError("Incorrect validation checkpoint selection")
        schedules.append(ordered)
    if schedules[0] != schedules[1] or len(schedules[0]) != plan["updates_per_arm"]:
        raise ValueError("Unmatched training schedule")
    warmup = read_rows(run / "warmup.jsonl")
    if len(warmup) != plan["warmup"] or any(row["case"] not in train for row in warmup):
        raise ValueError("Warmup differs from plan")
    paired = []
    for method in (BASELINE, COMMON, *METHODS):
        path = run / f"{method}-evaluation.jsonl"
        panel = results["panels"][method]
        metrics, traces = rescore(path, "evaluation")
        if file_hash(path) != panel["traces_sha256"] or metrics != panel["metrics"]:
            raise ValueError("Reserved results changed")
        for case, row in zip(banks["evaluation"], traces, strict=True):
            paired.append({**row, "root_id": digest(case["turns"]), "realization_id": case["id"],
                           "seed": SEED, "scenario_family": case["family"], "method": method})
    contrast = paired_bootstrap(paired, "grpo", "sft", "success", confidence=.95, resamples=2000)
    if contrast != results["grpo_minus_sft"]:
        raise ValueError("Contrast changed")
    from collections import Counter
    per_case = Counter(len(row["accepted"]) for row in banks["evaluation"])
    return {"status": "verified", "results_sha256": file_hash(result_path),
            "cases": {key: len(rows) for key, rows in banks.items()},
            "dialogues": {key: len(ids) for key, ids in dialogues.items()},
            "grpo_groups": {"total": len(groups), "varying": sum(groups)},
            "separate_from_previous_evaluation": True,
            "uniform_accuracy": mean(len(row["accepted"]) / len(row["options"]) for row in banks["evaluation"]),
            "cases_by_accepted_action_count": dict(per_case),
            "trajectories": len(banks["evaluation"]) * len(results["panels"]),
            "selection_sha256": file_hash(selection_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--previous", type=Path, default=Path("research/runs/tesis_meal_lora_v1"))
    args = parser.parse_args()
    report = audit(args.run, args.previous)
    write_json(args.run / "AUDIT.json", report)
    print(report, flush=True)
