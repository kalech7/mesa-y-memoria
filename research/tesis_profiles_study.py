"""Fit sequential tabular RL and compare matched Qwen SFT and branched GRPO."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
from importlib.metadata import version
import json
from pathlib import Path

from research.src.artifacts import digest, file_hash
from research.tesis_extended_analysis import branch_data, validation_inputs
from research.tesis_extended_experiment import MODEL, OLD_PARENT, PROJECT, competently_initialized, open_model, parent_path
from research.tesis_profiles_analysis import actor_trees, evaluate, population
from research.tesis_profiles_bank import build_development
from research.tesis_profiles_env import enumerate_world, public_features, rule_action
from research.tesis_profiles_learning import RL as BELLMAN, fit_from_trees
from research.tesis_shared_budget_pilot import RULE, paired, source_hashes, stratified_mean, summarize
from research.tesis_shared_budget_qwen import atomic_json, status
from research.tesis_shared_budget_qwen_matched import GRPO, SFT, fit_matched, load_adapter


SEEDS = (101, 202, 303)
METRICS = ("longitudinal", "current_supported", "future_supported", "questions",
           "questions_first", "questions_second", "questions_third", "future_unsupported_reuse",
           "unsupported_writes", "expired_reuse", "delivered_recontacts",
           "counterfactually_unnecessary_questions")


def trees_for(stories, workers):
    if workers == 1:
        return [enumerate_world(story) for story in stories]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(enumerate_world, stories, chunksize=2))


def freeze_profile_plan(args, output, splits):
    sources = source_hashes()
    for name in (
        "tesis_profiles_env", "tesis_profiles_bank", "tesis_profiles_learning", "tesis_profiles_analysis",
        "tesis_profiles_study", "tesis_profiles_score", "tesis_extended_env", "tesis_extended_analysis",
        "tesis_extended_experiment", "tesis_shared_budget_qwen", "tesis_shared_budget_qwen_context",
        "tesis_shared_budget_qwen_matched",
    ):
        relative = f"research/{name}.py"
        sources[relative] = file_hash(PROJECT / relative)
    seal = json.loads(args.reserve_seal.read_text(encoding="utf-8"))
    if seal["status"] != "frozen_before_selection" or not seal["manifest_sha256"]:
        raise ValueError("The independently authored reserve must be sealed before selection")
    settings = {
        "experiment": "diverse-public-user-profiles-v4",
        "seeds": list(SEEDS), "epochs": args.epochs, "batch_size": args.batch_size,
        "passes_per_batch": 2, "learning_rate": args.learning_rate, "max_tokens": args.max_tokens,
        "source_hashes": sources, "data_hashes": {role: digest(rows) for role, rows in splits.items()},
        "reserve_seal": seal, "reserved_access": False,
        "base_model": MODEL.relative_to(PROJECT).as_posix(),
        "model_hashes": {path.name: file_hash(path) for path in sorted(MODEL.iterdir())
                         if path.is_file() and path.suffix in {".json", ".safetensors", ".txt"}},
        "inherited_adapter": OLD_PARENT.relative_to(PROJECT).as_posix(),
        "inherited_adapter_hashes": {name: file_hash(OLD_PARENT / name)
                                     for name in ("adapter_model.safetensors", "adapter_config.json")},
        "selection": "current/reuse guards, utility, future support, fewer questions, earliest epoch",
        "primary_metric": "longitudinal",
        "reserve_interpretation": {
            "interval": "paired root bootstrap, stratified by scenario family, 95%",
            "minimum_lower_bound_for_advantage": 0,
            "max_current_supported_loss": .02,
            "max_extra_questions_per_episode": .10,
            "max_increase_future_unsupported_reuse": 0,
        },
        "versions": {name: version(name) for name in ("torch", "transformers", "peft")},
    }
    path = output / "PLAN.json"
    if args.resume and path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != settings:
            raise ValueError("Resume requires identical sources, data, parent and settings")
    else:
        atomic_json(path, settings)
    return settings


def sequential_experiment(train, dev, output):
    study = fit_from_trees(train, public_features=public_features, fixed_policy=rule_action)
    study.save(output / "TABULAR.json")
    refs = {RULE: rule_action, **study.policies}
    rows = [row for method, policy in refs.items() for row in evaluate(dev, policy, method=method, seed=0)]
    comparisons = {
        other: {key: paired(rows, BELLMAN, other, key) for key in METRICS if key in rows[0]["metrics"]}
        for other in refs if other != BELLMAN
    }
    atomic_json(output / "SEQUENTIAL_VALIDATION.json", {
        "summary": summarize(rows), "rows": rows, "paired": comparisons, "telemetry": study.telemetry,
        "coverage": {method: policy.coverage(reset=True) for method, policy in study.policies.items()},
        "population": {"train": population(train), "validation": population(dev)},
    })
    return study, rows


def lock_selection(output, settings, selections):
    artifacts = {}
    for seed, methods in selections.items():
        artifacts[seed] = {}
        for method, checkpoint in methods.items():
            path = Path(f"seed-{seed}") / checkpoint["adapter"]
            artifacts[seed][method] = {
                "adapter": path.as_posix(), "epoch": checkpoint["epoch"], "eligible": checkpoint["eligible"],
                "weights_sha256": file_hash(output / path / "adapter_model.safetensors"),
                "config_sha256": file_hash(output / path / "adapter_config.json"),
            }
    atomic_json(output / "SELECTION_LOCK.json", {
        "status": "frozen_before_reserved_evaluation",
        "plan_sha256": file_hash(output / "PLAN.json"),
        "validation_sha256": file_hash(output / "QWEN_VALIDATION.json"),
        "tabular_sha256": file_hash(output / "TABULAR.json"),
        "reserve_manifest_sha256": settings["reserve_seal"]["manifest_sha256"],
        "adapters": artifacts,
    })


def run_locked(args, output):
    import torch
    from transformers import AutoTokenizer

    splits = build_development()
    if set(splits) != {"train", "validation"}:
        raise ValueError("Optimization only accepts development partitions")
    settings = freeze_profile_plan(args, output, splits)
    status(output, "enumerating_training_profiles")
    train = trees_for(splits["train"], args.workers)
    status(output, "enumerating_validation_profiles")
    dev = trees_for(splits["validation"], args.workers)
    status(output, "fitting_sequential_rl")
    study, cpu_rows = sequential_experiment(train, dev, output)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The Qwen pair needs a bfloat16-capable CUDA device")
    status(output, "preparing_matched_qwen_groups")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    records, ledger = branch_data(train, actor_trees(train), study.policies[BELLMAN], tokenizer, args.max_tokens)
    dev_public = actor_trees(dev)
    examples = validation_inputs(dev_public, tokenizer, args.max_tokens)
    atomic_json(output / "QWEN_BRANCHES.json", ledger)
    torch.manual_seed(1301)
    model, tokenizer, labels = open_model(MODEL, OLD_PARENT)
    baseline = [row for row in cpu_rows if row["method"] == RULE]
    floor = max(0., stratified_mean(baseline, "current_supported") - .02)
    ceiling = stratified_mean(baseline, "future_unsupported_reuse")

    def rank(rows):
        def avg(name):
            return stratified_mean(rows, name)
        eligible = (avg("current_supported") >= floor - 1e-12
                    and avg("future_unsupported_reuse") <= ceiling + 1e-12)
        return (eligible, round(avg("longitudinal"), 12), round(avg("future_supported"), 12),
                round(-avg("questions"), 12))

    warm = competently_initialized(
        model, output, records, examples, tokenizer, labels,
        lambda policy, method: evaluate(dev_public, policy, method=method, seed=0), floor, settings,
    )
    common = parent_path(output, warm)
    rows, training, selections = [], {}, {}
    for seed in SEEDS:
        directory = output / f"seed-{seed}"
        directory.mkdir(exist_ok=args.resume)
        archive = directory / "SEED_RESULT.json"
        if archive.exists():
            result = json.loads(archive.read_text(encoding="utf-8"))
        else:
            status(output, "training_matched_qwen", seed=seed)
            load_adapter(model, common)
            result = fit_matched(
                model, directory, records=records, evaluation=examples, tokenizer=tokenizer, labels=labels,
                device="cuda", settings={**settings, "seed": seed}, selection_rank=rank,
                validation_rows=lambda policy, method: evaluate(dev_public, policy, method=method, seed=seed),
                resume=args.resume,
            )
            atomic_json(archive, result)
        key = str(seed)
        selections[key] = result["selected_checkpoints"]
        training[key] = result["accounting"]
        rows.extend(result["rows"])
    atomic_json(output / "QWEN_VALIDATION.json", {
        "status": "development_only", "summary": summarize(rows), "rows": rows,
        "selected": selections, "training": training,
        "paired_grpo_minus_sft": {key: paired(rows, GRPO, SFT, key) for key in METRICS if key in rows[0]["metrics"]},
        "shared_parent": {"current_supported": warm["current_supported"], "floor": floor,
                          "warm_updates": sum(row.get("optimizer_updates", 0) for row in warm["attempts"])},
        "reserved_access": False,
    })
    lock_selection(output, settings, selections)
    status(output, "completed", results="QWEN_VALIDATION.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reserve-seal", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (min(args.workers, args.epochs, args.batch_size, args.max_tokens) < 1 or args.epochs > 999
            or not 0 < args.learning_rate < 1):
        parser.error("Invalid training configuration")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    with (output / "RUN.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume and (output / "SELECTION_LOCK.json").is_file():
            status(output, "completed", results="QWEN_VALIDATION.json")
            return
        try:
            run_locked(args, output)
        except Exception as error:
            status(output, "failed", error_type=type(error).__name__, message=str(error))
            raise


if __name__ == "__main__":
    main()
