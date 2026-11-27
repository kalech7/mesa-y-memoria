"""Evaluate only the frozen, selected policies on the separately authored reserve."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path

from research.src.artifacts import canonical, file_hash
from research.tesis_extended_analysis import actor_trees, evaluate, validation_inputs
from research.tesis_extended_env import rule_action
from research.tesis_extended_experiment import METRICS, MODEL, PROJECT, SEEDS, open_model, trees_for
from research import tesis_extended_heldout
from research.tesis_extended_heldout import load_executable, verify
from research.tesis_shared_budget_pilot import PLAN, RULE, paired, summarize
from research.tesis_shared_budget_planner import public_state
from research.tesis_shared_budget_qwen import atomic_json, model_policy
from research.tesis_shared_budget_qwen_matched import GRPO, PARENT, SFT, load_adapter


def locked_selection(run, reserved):
    lock = json.loads((run / "SELECTION_LOCK.json").read_text(encoding="utf-8"))
    if lock["status"] != "frozen_before_reserved_evaluation":
        raise ValueError("Select and lock all checkpoints before opening the reserve")
    expected = {
        run / "PLAN.json": lock["plan_sha256"],
        run / "RESULTS.json": lock["validation_sha256"],
        run / "PUBLIC_PLANNER.json": lock["planner_sha256"],
        reserved / "MANIFEST.json": lock["reserve_manifest_sha256"],
    }
    for methods in lock["adapters"].values():
        for checkpoint in methods.values():
            adapter = (run / checkpoint["path"]).resolve()
            if not adapter.is_relative_to(run):
                raise ValueError("Selected adapter must be inside the frozen run")
            expected[adapter / "adapter_model.safetensors"] = checkpoint["weights_sha256"]
            expected[adapter / "adapter_config.json"] = checkpoint["config_sha256"]
    if any(file_hash(path) != expected_sha for path, expected_sha in expected.items()):
        raise ValueError("The checkpoint selection or bank changed after freeze")
    return lock


def score_locked(run, reserved, workers):
    import torch

    output = run / "RESERVED_RESULTS.json"
    if output.exists():
        raise FileExistsError("Preserve the original reserved evaluation")
    lock = locked_selection(run, reserved)
    plan = json.loads((run / "PLAN.json").read_text(encoding="utf-8"))
    if any(file_hash(PROJECT / name) != checksum for name, checksum in plan["source_hashes"].items()):
        raise ValueError("The environment, model prompt or scorer changed after freeze")
    if file_hash(Path(tesis_extended_heldout.__file__)) != plan["reserve_seal"]["loader_sha256"]:
        raise ValueError("The independently authored reserve loader changed after freeze")
    if any(file_hash(MODEL / name) != checksum for name, checksum in plan["model_hashes"].items()):
        raise ValueError("The base model or tokenizer changed after freeze")
    verify(reserved)
    stories = load_executable(reserved)
    if any(row["split"] != "reserved" for row in stories):
        raise ValueError("Expected only independently authored reserved stories")
    print("Enumerating the locked reserve", flush=True)
    trees = trees_for(stories, workers)
    observed = actor_trees(trees)
    chosen = lock["adapters"][str(SEEDS[0])][PARENT]
    model, tokenizer, labels = open_model(MODEL, run / chosen["path"], trainable=False)
    examples = validation_inputs(observed, tokenizer, plan["max_tokens"])
    frozen_table = {
        canonical(key): action for key, action in json.loads((run / "PUBLIC_PLANNER.json").read_text())
    }

    def planner(observation, legal):
        action = frozen_table.get(canonical(public_state(observation, legal)))
        return action if action in legal else rule_action(observation, legal)

    rows, ledgers = [], {}
    for seed in SEEDS:
        for label, reference in ((RULE, rule_action), (PLAN, planner)):
            rows.extend(evaluate(trees, reference, method=label, seed=seed))
        ledgers[str(seed)] = {}
        for method in (PARENT, SFT, GRPO):
            checkpoint = lock["adapters"][str(seed)][method]
            load_adapter(model, run / checkpoint["path"])
            counter = Counter()
            policy = model_policy(model, examples, tokenizer, labels, "cuda", plan["batch_size"], counter)
            rows.extend(evaluate(observed, policy, method=method, seed=seed))
            ledgers[str(seed)][method] = {**checkpoint, **counter}
            print(f"Scored {method}, seed {seed}", flush=True)
    atomic_json(output, {
        "status": "independently_authored_ai_reviewed_reserve",
        "summary": summarize(rows), "rows": rows, "inference": ledgers,
        "selection_sha256": file_hash(run / "SELECTION_LOCK.json"),
        "reserve_manifest_sha256": lock["reserve_manifest_sha256"],
        "scorer_sha256": file_hash(Path(__file__)),
        "device": torch.cuda.get_device_name(0),
        "paired_grpo_minus_sft": {key: paired(rows, GRPO, SFT, key) for key in METRICS
                                  if key in rows[0]["metrics"]},
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
