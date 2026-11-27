"""Score the frozen profile policies on the separately authored reserve."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path

from research import tesis_profiles_heldout
from research.src.artifacts import file_hash
from research.tesis_extended_analysis import validation_inputs
from research.tesis_extended_experiment import MODEL, PROJECT, open_model
from research.tesis_profiles_analysis import actor_trees, evaluate, population
from research.tesis_profiles_env import public_features, rule_action
from research.tesis_profiles_learning import load_policies
from research.tesis_profiles_study import BELLMAN, METRICS, SEEDS, trees_for
from research.tesis_shared_budget_pilot import RULE, paired, summarize
from research.tesis_shared_budget_qwen import atomic_json, model_policy
from research.tesis_shared_budget_qwen_matched import GRPO, PARENT, SFT, load_adapter


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def locked_selection(run, reserved):
    lock = read_json(run / "SELECTION_LOCK.json")
    if lock["status"] != "frozen_before_reserved_evaluation":
        raise ValueError("Select and lock all checkpoints before opening the reserve")
    expected = {
        run / "PLAN.json": lock["plan_sha256"],
        run / "QWEN_VALIDATION.json": lock["validation_sha256"],
        run / "TABULAR.json": lock["tabular_sha256"],
        reserved / "MANIFEST.json": lock["reserve_manifest_sha256"],
    }
    if set(lock["adapters"]) != {str(seed) for seed in SEEDS}:
        raise ValueError("Expected all prespecified seeds")
    for methods in lock["adapters"].values():
        if set(methods) != {PARENT, SFT, GRPO}:
            raise ValueError("Expected all prespecified adapters")
        for checkpoint in methods.values():
            adapter = (run / checkpoint["adapter"]).resolve()
            if not adapter.is_relative_to(run):
                raise ValueError("Selected adapter must be inside the frozen run")
            expected[adapter / "adapter_model.safetensors"] = checkpoint["weights_sha256"]
            expected[adapter / "adapter_config.json"] = checkpoint["config_sha256"]
    if any(file_hash(path) != checksum for path, checksum in expected.items()):
        raise ValueError("The checkpoint selection or bank changed after freeze")
    plan = read_json(run / "PLAN.json")
    if any(file_hash(PROJECT / name) != checksum for name, checksum in plan["source_hashes"].items()):
        raise ValueError("The environment, model prompt or scorer changed after freeze")
    if file_hash(Path(tesis_profiles_heldout.__file__)) != plan["reserve_seal"]["loader_sha256"]:
        raise ValueError("The reserve loader changed after freeze")
    if any(file_hash(MODEL / name) != checksum for name, checksum in plan["model_hashes"].items()):
        raise ValueError("The base model or tokenizer changed after freeze")
    return lock, plan


def contrasts(rows, method, reference):
    return {key: paired(rows, method, reference, key) for key in METRICS if key in rows[0]["metrics"]}


def score_locked(run, reserved, workers):
    import torch

    output = run / "RESERVED_RESULTS.json"
    if output.exists():
        raise FileExistsError("Preserve the original reserved evaluation")
    lock, plan = locked_selection(run, reserved)
    tesis_profiles_heldout.verify(reserved)
    stories = tesis_profiles_heldout.load_executable(reserved)
    if any(row["split"] != "reserved" for row in stories):
        raise ValueError("Expected only independently authored reserved stories")
    print("Enumerating the locked user profile reserve", flush=True)
    trees = trees_for(stories, workers)
    policies = load_policies(run / "TABULAR.json", public_features=public_features, fallback=rule_action)
    tabular = [
        row for method, policy in {RULE: rule_action, **policies}.items()
        for row in evaluate(trees, policy, method=method, seed=0)
    ]

    observed = actor_trees(trees)
    chosen = lock["adapters"][str(SEEDS[0])][PARENT]
    model, tokenizer, labels = open_model(MODEL, run / chosen["adapter"], trainable=False)
    examples = validation_inputs(observed, tokenizer, plan["max_tokens"])
    rows, ledgers = [], {}
    for seed in SEEDS:
        ledgers[str(seed)] = {}
        for method in (PARENT, SFT, GRPO):
            checkpoint = lock["adapters"][str(seed)][method]
            load_adapter(model, run / checkpoint["adapter"])
            counter = Counter()
            policy = model_policy(model, examples, tokenizer, labels, "cuda", plan["batch_size"], counter)
            rows.extend(evaluate(observed, policy, method=method, seed=seed))
            ledgers[str(seed)][method] = {**checkpoint, **counter}
            print(f"Scored {method}, seed {seed}", flush=True)
    atomic_json(output, {
        "status": "independently_authored_reserve",
        "selection_sha256": file_hash(run / "SELECTION_LOCK.json"),
        "reserve_manifest_sha256": lock["reserve_manifest_sha256"],
        "scorer_sha256": file_hash(Path(__file__)),
        "device": torch.cuda.get_device_name(0),
        "population": population(trees),
        "sequential": {
            "summary": summarize(tabular), "rows": tabular,
            "paired": {other: contrasts(tabular, BELLMAN, other)
                       for other in (RULE, *policies) if other != BELLMAN},
            "coverage": {method: policy.coverage() for method, policy in policies.items()},
        },
        "qwen": {
            "summary": summarize(rows), "rows": rows, "inference": ledgers,
            "paired_grpo_minus_sft": contrasts(rows, GRPO, SFT),
        },
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--reserved", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Use at least one worker")
    run = args.run.resolve()
    with (run / "RESERVED.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        score_locked(run, args.reserved.resolve(), args.workers)


if __name__ == "__main__":
    main()
