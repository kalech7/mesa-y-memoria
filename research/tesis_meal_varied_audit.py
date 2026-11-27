"""Audit exact menu rotations, group returns, held-out choices and selection."""
from __future__ import annotations

import argparse
from collections import defaultdict
from math import isclose, isfinite
from pathlib import Path
from statistics import mean

from research.src.analysis.paired_statistics import paired_bootstrap
from research.src.artifacts import digest, file_hash, write_json
from research.tesis_meal_audit import read, read_rows
from research.tesis_meal_cases import CAPTURE, LETTERS
from research.tesis_meal_train import rewards, scores
from research.tesis_meal_varied import METHODS, rank, summarize
from research.tesis_meal_varied_bank import normalized, rotate


def verify_trace(run, filename, roots):
    winners = read_rows(run / filename)
    expected_ids = [(row["id"], offset) for row in roots for offset in range(6)]
    if [(row["root_id"], row["rotation"]) for row in winners] != expected_ids:
        raise ValueError(f"Missing or mismatched reserved rotations in {filename}")
    root_by_id = {row["id"]: row for row in roots}
    for row in winners:
        source = rotate(root_by_id[row["root_id"]], row["rotation"])
        action = LETTERS.index(row["option"])
        token = LETTERS.find(row["raw_token"]) if len(row["raw_token"]) == 1 else -1
        if (row["original_action"] != source["order"][action] or row["family"] != source["family"] or
                row["pair"] != source["pair"] or row["raw_action_success"] != int(token in source["accepted"]) or
                any(row[key] != value for key, value in scores(source, action).items())):
            raise ValueError(f"Inconsistent choice or reward in {filename}")
    return winners


def verify_banks(run, plan):
    bank = {}
    distinct = {}
    for split, checksum in plan["bank_sha256"].items():
        path = run / f"{split}.jsonl"
        if file_hash(path) != checksum:
            raise ValueError("Bank changed after freezing")
        bank[split] = rows = read_rows(path)
        distinct[split] = {normalized(row["turns"]) for row in rows}
        groups = defaultdict(list)
        for row in rows:
            if row["split"] != split or row["id"] != digest({"turns": row["turns"], "options": row["options"]}):
                raise ValueError("Invalid case identity")
            allowed = [index for index, item in enumerate(row["options"]) if item["ingredients"]
                       and not (set(item["ingredients"]) & set(row["forbidden"])) and not row["ambiguous"]]
            allowed = allowed or [index for index, item in enumerate(row["options"]) if not item["ingredients"]]
            if allowed != row["accepted"]:
                raise ValueError("Accepted options disagree with authored constraints")
            if row["pair"]:
                groups[row["pair"]].append(row)
        if any(len(pair) != 2 or pair[0]["options"] != pair[1]["options"] or
               pair[0]["accepted"] == pair[1]["accepted"] or
               any(len(row["accepted"]) != 1 for row in pair) for pair in groups.values()):
            raise ValueError("Invalid counterfactual pair")
    if sum(map(len, distinct.values())) != len(set().union(*distinct.values())):
        raise ValueError("Normalized dialogues cross partitions")
    if not any(CAPTURE in row["turns"] for row in bank["train"]):
        raise ValueError("Reported user case absent from train")
    prior = set()
    for name in ("tesis_meal_lora_v1", "tesis_meal_lora_v2"):
        for split in bank:
            path = run.parent / name / f"{split}.jsonl"
            if path.is_file():
                prior.update(normalized(row["turns"]) for row in read_rows(path))
    if prior & distinct["evaluation"]:
        raise ValueError("Reserved dialogue appeared in a previous study")
    return bank, {split: len(rows) for split, rows in distinct.items()}


def verify_schedule(run, name, epochs, train, plan):
    observations, total_groups, varying_groups = [], 0, 0
    for epoch in range(1, epochs + 1):
        path = run / (f"warm-{epoch}.jsonl" if name == "warm" else f"{name}-train-{epoch}.jsonl")
        ledger = read_rows(path)
        if any(not 1 <= len(entry["cases"]) <= plan["batch"] or
               len(entry["losses"]) != (1 if name == "warm" else plan["passes"]) or
               not all(isfinite(loss) for loss in entry["losses"]) for entry in ledger):
            raise ValueError("Invalid optimizer schedule")
        for entry in ledger:
            observations.extend(tuple(value) for value in entry["cases"])
            if name == "grpo":
                for (root, offset), draws, returns in zip(entry["cases"], entry["draws"], entry["returns"], strict=True):
                    expected = [rewards(rotate(train[root], offset))[choice] for choice in draws]
                    if len(draws) != plan["draws"] or any(not isclose(a, b, abs_tol=1e-7)
                                                         for a, b in zip(expected, returns, strict=True)):
                        raise ValueError("GRPO rewards disagree with authored evidence")
                    total_groups += 1
                    varying_groups += len(set(expected)) > 1
    coverage = {(root, offset) for root in train for offset in range(6)}
    if len(observations) != len(coverage) or set(observations) != coverage:
        raise ValueError("Each training loop must expose every case in each position exactly once")
    return observations, {"total": total_groups, "varying": varying_groups}


def audit(run):
    plan_path = run / "PLAN.json"
    selection_path = run / "SELECTION.json"
    results_path = run / "RESULTS.json"
    plan, selection, results = map(read, (plan_path, selection_path, results_path))
    if (selection["split"] != "validation" or selection["plan_sha256"] != file_hash(plan_path)
            or results["selection_sha256"] != file_hash(selection_path)):
        raise ValueError("Selection lock changed")
    for name, checksum in plan["source_sha256"].items():
        if file_hash(Path(__file__).with_name(name)) != checksum:
            raise ValueError(f"Frozen source changed: {name}")
    bank, distinct = verify_banks(run, plan)
    training = {row["id"]: row for row in bank["train"]}
    verify_schedule(run, "warm", plan["warm_epochs"], training, plan)
    common = summarize(verify_trace(run, "sft-comun-validation.jsonl", bank["validation"]))
    schedules = {}
    telemetry = {}
    for method in METHODS[2:]:
        chosen = selection["methods"][method]
        schedules[method], telemetry[method] = verify_schedule(run, method, plan["epochs"], training, plan)
        candidates = {0: common}
        for epoch in range(1, plan["epochs"] + 1):
            candidates[epoch] = summarize(verify_trace(run, f"{method}-validation-{epoch}.jsonl", bank["validation"]))
        if (chosen["epoch"] != max(candidates, key=lambda epoch: rank(candidates[epoch], epoch)) or
                chosen["validation"] != {str(epoch): stats for epoch, stats in candidates.items()}):
            raise ValueError("Checkpoint was not selected by validation")
        for artifact, key in (("adapter_model.safetensors", "adapter_sha256"), ("local_checkpoint.json", "manifest_sha256")):
            if file_hash(run / method / artifact) != chosen[key]:
                raise ValueError("Selected adapter differs")
    if schedules["sft"] != schedules["grpo"]:
        raise ValueError("SFT and GRPO had different exposure")
    roots = {row["id"]: row for row in bank["evaluation"]}
    paired = []
    for method, panel in results["panels"].items():
        name = f"{method}-evaluation.jsonl"
        observed = verify_trace(run, name, bank["evaluation"])
        if file_hash(run / name) != panel["trace_sha256"] or summarize(observed) != panel["metrics"]:
            raise ValueError("Changed held-out metrics")
        if method in METHODS[2:]:
            paired.extend({"root_id": normalized(roots[r["root_id"]]["turns"]),
                           "realization_id": r["root_id"] + ":" + r["realization_id"], "scenario_family": r["family"],
                           "seed": plan["seed"], "method": method, "success": r["success"]} for r in observed)
    if paired_bootstrap(paired, "grpo", "sft", "success", confidence=.95, resamples=2000) != results["grpo_minus_sft"]:
        raise ValueError("Paired inference differs")
    return {"status": "verified", "results_sha256": file_hash(results_path),
            "selection_sha256": file_hash(selection_path),
            "cases": {split: len(rows) for split, rows in bank.items()}, "dialogues": distinct,
            "paired_evaluation_groups": len({row["pair"] for row in roots.values() if row["pair"]}),
            "evaluation_trajectories": len(results["panels"]) * len(roots) * 6,
            "uniform_accuracy": mean(len(row["accepted"]) / 6 for row in roots.values()),
            "grpo_groups": telemetry["grpo"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    write_json(args.run / "AUDIT.json", result)
    print(result, flush=True)
