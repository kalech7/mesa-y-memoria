"""Secondary reserve diagnostic of both trained final-epoch adapters."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from research.src.artifacts import file_hash
from research.tesis_extended_analysis import validation_inputs
from research.tesis_extended_experiment import MODEL, open_model
from research.tesis_profiles_analysis import actor_trees, evaluate
from research.tesis_profiles_heldout import load_executable
from research.tesis_profiles_score import contrasts, locked_selection
from research.tesis_profiles_study import SEEDS, trees_for
from research.tesis_shared_budget_pilot import summarize
from research.tesis_shared_budget_qwen import atomic_json, model_policy
from research.tesis_shared_budget_qwen_matched import GRPO, SFT, load_adapter


def diagnostic(run, reserved, workers):
    """Score only the last epoch of each arm after the primary evaluation."""
    output = run / "LAST_EPOCH_DIAGNOSTIC.json"
    if output.exists():
        raise FileExistsError(output)
    primary = run / "RESERVED_RESULTS.json"
    primary_hash = file_hash(primary)
    lock, plan = locked_selection(run, reserved)
    if json.loads(primary.read_text())["selection_sha256"] != file_hash(run / "SELECTION_LOCK.json"):
        raise ValueError("The primary evaluation must match the selection lock")
    epoch = plan["epochs"]
    validation_hash = lock["validation_sha256"]
    validation = json.loads((run / "QWEN_VALIDATION.json").read_text())
    paths = {}
    for seed in SEEDS:
        for method in (SFT, GRPO):
            path = run / f"seed-{seed}" / method / f"checkpoint-{epoch:03d}"
            ledger = json.loads((path / "LEDGER.json").read_text())
            if ledger != validation["training"][str(seed)]["arms"][method] or ledger[-1]["epoch"] != epoch:
                raise ValueError(f"Unexpected final epoch ledger for {method}, seed {seed}")
            paths[seed, method] = path / "adapter"

    trees = actor_trees(trees_for(load_executable(reserved), workers))
    model, tokenizer, labels = open_model(MODEL, paths[SEEDS[0], SFT], trainable=False)
    examples = validation_inputs(trees, tokenizer, plan["max_tokens"])
    rows, checkpoints = [], {}
    for (seed, method), path in paths.items():
        load_adapter(model, path)
        counter = Counter()
        policy = model_policy(model, examples, tokenizer, labels, "cuda", plan["batch_size"], counter)
        rows.extend(evaluate(trees, policy, method=method, seed=seed))
        checkpoints.setdefault(str(seed), {})[method] = {
            "adapter": str(path.relative_to(run)), "weights_sha256": file_hash(path / "adapter_model.safetensors"),
            "config_sha256": file_hash(path / "adapter_config.json"), "inference": dict(counter),
        }
    atomic_json(output, {
        "status": "exploratory_after_primary_reserved_evaluation",
        "epoch": epoch, "primary_sha256": primary_hash, "validation_sha256": validation_hash,
        "checkpoints": checkpoints, "rows": rows, "summary": summarize(rows),
        "paired_grpo_minus_sft": contrasts(rows, GRPO, SFT),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--reserved", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Use at least one worker")
    diagnostic(args.run.resolve(), args.reserved.resolve(), args.workers)


if __name__ == "__main__":
    main()
