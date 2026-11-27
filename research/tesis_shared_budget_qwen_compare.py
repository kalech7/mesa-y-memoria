"""Run the matched SFT versus branched GRPO experiment on Qwen and an A100."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path

from research.src.artifacts import digest, file_hash
from research.tesis_shared_budget_bank import audit_development, build_development
from research.tesis_shared_budget_env import ACTIONS, decision_state, rule_action
from research.tesis_shared_budget_pilot import (
    PLAN, RULE, enumerate_split, paired, source_hashes, stratified_mean, summarize,
)
from research.tesis_shared_budget_planner import evaluate, fit_public_policy, group_trees, traverse, weighted
from research.tesis_shared_budget_qwen import LETTERS, LORA, atomic_json, model_inputs, status
from research.tesis_shared_budget_qwen_curriculum import MASS, WRITE_COST, build_records


PROJECT = Path(__file__).resolve().parents[1]
MAX_CURRENT_LOSS = .02
METRICS = ("longitudinal", "future_supported", "current_supported", "questions",
           "questions_first", "questions_second", "correct_first_choice",
           "counterfactually_unnecessary_questions", "future_unsupported_reuse")


def stage_metrics(trees, policy, *, method, seed):
    rows = evaluate(trees, policy, method=method, seed=seed)
    roots = group_trees(trees)
    for row in rows:
        worlds = roots[row["root_id"]]
        should_ask_first = row["scenario_family"].startswith("ready")
        asked_first = row["metrics"]["questions_first"]
        row["metrics"]["correct_first_choice"] = asked_first if should_ask_first else 1. - asked_first
        second_opportunity, second_resolved = [], []
        for world in worlds:
            path, _ = traverse(world, policy)
            valid, asked = False, False
            for index, action in enumerate(path):
                node = world.decisions[path[:index]]
                obs = node.observation
                if obs["visit"] == 2:
                    valid = ("TEMPORAL_SCOPE" in node.legal
                             and decision_state(obs)["missing_slot"] == "TEMPORAL_SCOPE")
                    asked = valid and action == "TEMPORAL_SCOPE"
                    break
            second_opportunity.append((world.mass, valid))
            second_resolved.append((world.mass, asked))
        row["metrics"]["second_ambiguous_with_budget"] = weighted(second_opportunity)
        row["metrics"]["asked_at_second_opportunity"] = weighted(second_resolved)
    return rows


def profile(rows):
    summary = summarize(rows)
    for method, item in summary.items():
        own = [row for row in rows if row["method"] == method]
        opportunity = stratified_mean(own, "second_ambiguous_with_budget")
        item["second_ask_given_opportunity"] = (stratified_mean(own, "asked_at_second_opportunity") / opportunity
                                                 if opportunity else None)
    return summary


def settings_for(args, splits, base):
    sources = source_hashes()
    for name in ("qwen", "qwen_curriculum", "qwen_matched", "qwen_compare"):
        filename = f"research/tesis_shared_budget_{name}.py"
        sources[filename] = file_hash(PROJECT / filename)
    model_files = [path for path in base.iterdir()
                   if path.is_file() and path.suffix in {".json", ".safetensors", ".txt"}]
    return {
        "experiment": "qwen-1.5b-shared-budget-matched-v1",
        "base_model": base.relative_to(PROJECT).as_posix(),
        "model_hashes": {file.name: file_hash(file) for file in sorted(model_files)},
        "source_hashes": sources,
        "data_hashes": {role: digest(rows) for role, rows in splits.items()},
        "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
        "passes_per_batch": 2, "learning_rate": args.learning_rate,
        "max_tokens": args.max_tokens, "lora": LORA, "dtype": "bfloat16",
        "curriculum_weights": MASS, "write_cost_in_train_returns": WRITE_COST,
        "max_current_loss_vs_public_rule": MAX_CURRENT_LOSS,
        "require_no_more_unsupported_reuse_vs_public_rule": True,
        "selection": ["eligible", "longitudinal_utility", "correct_first_choice", "future_supported",
                      "fewer_questions", "earliest_epoch"],
        "primary_comparison_metric": "longitudinal",
        "versions": {name: version(name) for name in ("torch", "transformers", "peft")},
        "reserved_access": False,
    }


def run_locked(args, output):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from research.tesis_shared_budget_qwen_matched import GRPO, PARENT, SFT, fit_matched

    base = args.model.resolve()
    splits = build_development()
    audit = audit_development(splits)
    settings = settings_for(args, splits, base)
    plan_file = output / "PLAN.json"
    if args.resume and plan_file.exists():
        if json.loads(plan_file.read_text(encoding="utf-8")) != settings:
            raise ValueError("Resume requires identical settings, bank, sources, model and dependencies")
    else:
        atomic_json(plan_file, settings)
    atomic_json(output / "BANK_AUDIT.json", audit)

    status(output, "enumerating_train")
    train = enumerate_split(splits["train"], args.workers)
    status(output, "preparing_shared_branches")
    planner = fit_public_policy(train, plan=True)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    records, ledger = build_records(train, planner, tokenizer, args.max_tokens)
    atomic_json(output / "BRANCHES.json", ledger)
    status(output, "enumerating_validation")
    dev = enumerate_split(splits["validation"], args.workers)
    examples = model_inputs(dev, tokenizer, args.max_tokens)

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This run requires CUDA with bfloat16 support")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokens = [tokenizer.encode(LETTERS[action], add_special_tokens=False) for action in ACTIONS]
    if any(len(token) != 1 for token in tokens) or len({token[0] for token in tokens}) != len(ACTIONS):
        raise ValueError("The tokenizer needs distinct single-token action labels")
    labels = torch.tensor([token[0] for token in tokens], device=device)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, local_files_only=True).to(device)
    if model.config.model_type != "qwen2":
        raise ValueError("The scorer needs a Qwen2-compatible model")
    model = get_peft_model(model, LoraConfig(**LORA))

    def validation_rows(policy, method):
        return stage_metrics(dev, policy, method=method, seed=args.seed)

    reference = validation_rows(rule_action, RULE)
    planner_rows = validation_rows(planner, PLAN)
    current_floor = stratified_mean(reference, "current_supported") - MAX_CURRENT_LOSS
    unsupported_ceiling = stratified_mean(reference, "future_unsupported_reuse")

    def rank(rows):
        eligible = (stratified_mean(rows, "current_supported") >= current_floor - 1e-12
                    and stratified_mean(rows, "future_unsupported_reuse") <= unsupported_ceiling + 1e-12)
        return (eligible, *[round(stratified_mean(rows, key), 12) for key in
                            ("longitudinal", "correct_first_choice", "future_supported")],
                round(-stratified_mean(rows, "questions"), 12))

    result = fit_matched(
        model, output, records=records, evaluation=examples, tokenizer=tokenizer,
        labels=labels, device=device, settings=settings, validation_rows=validation_rows,
        selection_rank=rank, resume=args.resume,
    )
    rows = [*result["rows"], *reference, *planner_rows]
    selection = result["selected_checkpoints"]
    chosen = {"sft": selection[SFT], "grpo": selection[GRPO]}
    lab_models = {}
    for name, item in chosen.items():
        relative = Path(item["adapter"])
        weights = output / relative / "adapter_model.safetensors"
        lab_models[name] = {"adapter": relative.as_posix(), "adapter_sha256": file_hash(weights)}
    atomic_json(output / "RESULTS.json", {
        "status": "development_only", "matched": result["accounting"], "selected_checkpoints": selection,
        "summary": profile(rows), "rows": rows,
        "paired_grpo_minus_sft": {key: paired(rows, GRPO, SFT, key) for key in METRICS},
        "paired_minus_parent": {method: {key: paired(rows, method, PARENT, key) for key in METRICS}
                                for method in (SFT, GRPO)},
        "device": torch.cuda.get_device_name(device), "curriculum": ledger,
        "simulator_steps": {"train": sum(tree.environment_steps for tree in train),
                            "validation": sum(tree.environment_steps for tree in dev)},
        "reserved_access": False,
    })
    atomic_json(output / "LAB_MODELS.json", {
        "status": "completed", "base_model": settings["base_model"], "models": lab_models,
        "selection": chosen, "planner": list(planner.actions.items()),
        "results": "RESULTS.json",
    })
    status(output, "completed", results="RESULTS.json", lab="LAB_MODELS.json")


def run(args):
    import fcntl

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    with (output / "RUN.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume and (output / "LAB_MODELS.json").is_file():
            status(output, "completed", results="RESULTS.json", lab="LAB_MODELS.json")
            return
        try:
            run_locked(args, output)
        except Exception as error:
            status(output, "failed", error_type=type(error).__name__, message=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=PROJECT / "models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not (1 <= args.epochs <= 999 and min(args.batch_size, args.workers, args.max_tokens) >= 1
            and 0 < args.learning_rate < 1):
        parser.error("Invalid training configuration")
    run(args)


if __name__ == "__main__":
    main()
