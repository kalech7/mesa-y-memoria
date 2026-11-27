"""Repeat the matched Qwen experiment with a visible evidence summary."""
from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import json
from pathlib import Path

from research.src.artifacts import file_hash
from research.tesis_shared_budget_bank import audit_development, build_development
from research.tesis_shared_budget_env import ACTIONS, rule_action
from research.tesis_shared_budget_pilot import PLAN, RULE, enumerate_split, paired, stratified_mean
from research.tesis_shared_budget_planner import Decision, fit_public_policy
from research.tesis_shared_budget_qwen import LETTERS, LORA, atomic_json, model_inputs, status
from research.tesis_shared_budget_qwen_compare import (
    MAX_CURRENT_LOSS, METRICS, PROJECT, profile, settings_for, stage_metrics,
)
from research.tesis_shared_budget_qwen_context import PROMPT_FORMAT, with_public_context
from research.tesis_shared_budget_qwen_curriculum import build_records


def annotated(trees):
    """Keep simulation and outcomes fixed and transform only actor observations."""
    return [replace(tree, decisions={
        prefix: Decision(with_public_context(state.observation), state.legal)
        for prefix, state in tree.decisions.items()
    }) for tree in trees]


def run_locked(args, output):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from research.tesis_shared_budget_qwen_matched import GRPO, PARENT, SFT, fit_matched

    base = args.model.resolve()
    splits = build_development()
    settings = settings_for(args, splits, base)
    settings["experiment"] = "qwen-1.5b-shared-budget-matched-context-v2"
    settings["prompt_format"] = PROMPT_FORMAT
    settings["status"] = "exploratory_followup_on_same_development_bank"
    for name in ("context", "context_run"):
        filename = f"research/tesis_shared_budget_qwen_{name}.py"
        settings["source_hashes"][filename] = file_hash(PROJECT / filename)
    plan = output / "PLAN.json"
    if args.resume and plan.exists():
        if json.loads(plan.read_text(encoding="utf-8")) != settings:
            raise ValueError("Resume requires the same plan and sources")
    else:
        atomic_json(plan, settings)
    atomic_json(output / "BANK_AUDIT.json", audit_development(splits))

    status(output, "enumerating_train")
    train = enumerate_split(splits["train"], args.workers)
    planner = fit_public_policy(train, plan=True)
    status(output, "preparing_shared_branches")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    records, branches = build_records(annotated(train), planner, tokenizer, args.max_tokens)
    atomic_json(output / "BRANCHES.json", branches)
    status(output, "enumerating_validation")
    dev = enumerate_split(splits["validation"], args.workers)
    dev_public = annotated(dev)
    examples = model_inputs(dev_public, tokenizer, args.max_tokens)

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This comparison needs CUDA with bfloat16 support")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    ids = [tokenizer.encode(LETTERS[action], add_special_tokens=False) for action in ACTIONS]
    if any(len(token) != 1 for token in ids) or len({token[0] for token in ids}) != len(ACTIONS):
        raise ValueError("The action menu requires distinct single-token labels")
    labels = torch.tensor([token[0] for token in ids], device=device)
    base_model = AutoModelForCausalLM.from_pretrained(
        base, local_files_only=True, torch_dtype=torch.bfloat16,
    ).to(device)
    if base_model.config.model_type != "qwen2":
        raise ValueError("The action scorer requires Qwen2")
    model = get_peft_model(base_model, LoraConfig(**LORA))

    def evaluate_policy(policy, method):
        return stage_metrics(dev, policy, method=method, seed=args.seed)

    reference, planner_rows = evaluate_policy(rule_action, RULE), evaluate_policy(planner, PLAN)
    current_floor = stratified_mean(reference, "current_supported") - MAX_CURRENT_LOSS
    unsupported_ceiling = stratified_mean(reference, "future_unsupported_reuse")

    def rank(rows):
        def mean(name):
            return stratified_mean(rows, name)

        eligible = (mean("current_supported") >= current_floor - 1e-12
                    and mean("future_unsupported_reuse") <= unsupported_ceiling + 1e-12)
        return (eligible, *[round(mean(key), 12) for key in
                            ("longitudinal", "correct_first_choice", "future_supported")],
                round(-mean("questions"), 12))

    result = fit_matched(
        model, output, records=records, evaluation=examples, tokenizer=tokenizer,
        labels=labels, device=device, settings=settings,
        validation_rows=lambda policy, method: stage_metrics(dev_public, policy, method=method, seed=args.seed),
        selection_rank=rank, resume=args.resume,
    )
    rows = [*result["rows"], *reference, *planner_rows]
    selected = result["selected_checkpoints"]
    publication = {"sft": selected[SFT], "grpo": selected[GRPO]}
    atomic_json(output / "RESULTS.json", {
        "status": "development_only", "prompt_format": PROMPT_FORMAT,
        "matched": result["accounting"], "selected_checkpoints": selected,
        "summary": profile(rows), "rows": rows, "curriculum": branches,
        "paired_grpo_minus_sft": {key: paired(rows, GRPO, SFT, key) for key in METRICS},
        "paired_minus_parent": {arm: {key: paired(rows, arm, PARENT, key) for key in METRICS}
                                for arm in (SFT, GRPO)},
        "device": torch.cuda.get_device_name(device), "reserved_access": False,
    })
    atomic_json(output / "LAB_MODELS.json", {
        "status": "completed", "base_model": settings["base_model"], "prompt_format": PROMPT_FORMAT,
        "models": {name: {"adapter": item["adapter"],
                          "adapter_sha256": file_hash(output / item["adapter"] / "adapter_model.safetensors")}
                   for name, item in publication.items()},
        "selection": publication, "planner": list(planner.actions.items()), "results": "RESULTS.json",
    })
    status(output, "completed", results="RESULTS.json", lab="LAB_MODELS.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=PROJECT / "models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not (1 <= args.epochs <= 999 and min(args.batch_size, args.workers, args.max_tokens) >= 1
            and 0 < args.learning_rate < 1):
        parser.error("Invalid training configuration")
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


if __name__ == "__main__":
    main()
