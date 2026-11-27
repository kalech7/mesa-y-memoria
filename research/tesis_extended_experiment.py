"""Run a paired multiseed Qwen study on the three-visit environment."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
import random
from statistics import mean

from research.src.artifacts import digest, file_hash
from research.tesis_extended_analysis import actor_trees, branch_data, evaluate, validation_inputs
from research.tesis_extended_bank import build_development
from research.tesis_extended_env import enumerate_world, rule_action
from research.tesis_shared_budget_env import ACTIONS
from research.tesis_shared_budget_pilot import PLAN, RULE, paired, source_hashes, stratified_mean, summarize
from research.tesis_shared_budget_planner import fit_public_policy
from research.tesis_shared_budget_qwen import LETTERS, atomic_json, batches, checkpoint, status
from research.tesis_shared_budget_qwen_matched import (
    GRPO, PARENT, SFT, fit_matched, load_adapter, prepare, train_epoch, validate_model,
)


PROJECT = Path(__file__).resolve().parents[1]
MODEL = PROJECT / "models/Qwen2.5-1.5B-Instruct"
OLD_PARENT = (PROJECT / "reports/tesis_shared_budget_qwen_matched_a100_v2"
              / "SFT-ramas/checkpoint-001/adapter")
SEEDS = (101, 202, 303)
MYOPIC = "valor-un-paso"
MAX_CURRENT_LOSS = .02
MIN_MARGIN = .005
METRICS = ("longitudinal", "current_supported", "future_supported", "questions",
           "questions_first", "questions_second", "questions_third", "future_unsupported_reuse")


def trees_for(stories, workers):
    if workers == 1:
        return [enumerate_world(story) for story in stories]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(enumerate_world, stories, chunksize=2))


def open_model(base, adapter, *, trainable=True):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    ids = [tokenizer.encode(LETTERS[action], add_special_tokens=False) for action in ACTIONS]
    if any(len(token) != 1 for token in ids) or len({token[0] for token in ids}) != len(ids):
        raise ValueError("The tokenizer needs distinct one-token action labels")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
    if model.config.model_type != "qwen2":
        raise ValueError("The branch scorer requires Qwen2")
    model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable, local_files_only=True)
    return model, tokenizer, torch.tensor([token[0] for token in ids], device="cuda")


def parent_path(output, evidence):
    return (output if evidence["adapter_scope"] == "run" else PROJECT) / evidence["adapter"]


def competently_initialized(model, output, records, examples, tokenizer, labels, validator, floor, settings):
    """Verify the inherited parent and, if needed, warm it on current TRAIN outcomes."""
    import torch

    directory = output / "shared-competence"
    directory.mkdir(exist_ok=True)
    report = directory / "PREFLIGHT.json"
    if report.is_file():
        result = json.loads(report.read_text())
        if result["passed"]:
            load_adapter(model, parent_path(output, result))
            return result
    saved = sorted(directory.glob("checkpoint-*"))
    parent = OLD_PARENT
    attempts = []
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=settings["learning_rate"])
    if saved:
        parent = saved[-1] / "adapter"
        load_adapter(model, parent)
        optimizer.load_state_dict(torch.load(saved[-1] / "OPTIMIZER.pt", map_location="cpu", weights_only=True))
        attempts = json.loads((saved[-1] / "LEDGER.json").read_text())
        current = attempts[-1]["current_supported"]
    else:
        raw = validate_model(model, examples, tokenizer, labels, "cuda", settings["batch_size"], PARENT, validator)
        current = stratified_mean(raw["rows"], "current_supported")
        attempts = [{"epoch": 0, "current_supported": current, **raw}]
    if current < floor - 1e-12:
        train = prepare([{**row, "returns": row["current_returns"]} for row in records])
        anchor = torch.zeros((len(train), len(ACTIONS)), device="cuda")
        for epoch in range(len(attempts), 3):
            order = list(range(len(train)))
            random.Random(1301 + epoch).shuffle(order)
            ledger = train_epoch(model, optimizer, SFT, train, list(batches(order, settings["batch_size"])),
                                 anchor, tokenizer, labels, "cuda", output=output, epoch=epoch,
                                 passes=2, total_weight=mean(row.weight for row in train))
            audit = validate_model(model, examples, tokenizer, labels, "cuda", settings["batch_size"], PARENT, validator)
            current = stratified_mean(audit["rows"], "current_supported")
            attempts.append({"epoch": epoch, "current_supported": current, **ledger, **audit})
            parent = checkpoint(directory, epoch, model, optimizer, attempts) / "adapter"
            if current >= floor - 1e-12:
                break
    adapter_scope = "run" if parent.is_relative_to(output) else "project"
    result = {
        "passed": current >= floor - 1e-12, "floor": floor, "current_supported": current,
        "attempts": attempts, "adapter_scope": adapter_scope,
        "adapter": str(parent.relative_to(output if adapter_scope == "run" else PROJECT)),
        "training_split": "train", "warm_start_role": "current supported response",
    }
    atomic_json(report, result)
    if not result["passed"]:
        raise ValueError("The common parent did not reach current-response competence")
    return result


def run_locked(args, output):
    import torch

    splits = build_development()
    if set(splits) != {"train", "validation"}:
        raise ValueError("Optimization only accepts TRAIN and validation")
    seal = json.loads(args.reserve_seal.read_text(encoding="utf-8"))
    if seal.get("status") != "frozen_before_selection" or not seal.get("manifest_sha256"):
        raise ValueError("A sealed independent reserve is required before checkpoint selection")
    sources = source_hashes()
    for name in ("tesis_extended_env", "tesis_extended_bank", "tesis_extended_analysis", "tesis_extended_experiment",
                 "tesis_extended_score_reserved",
                 "tesis_shared_budget_qwen", "tesis_shared_budget_qwen_context", "tesis_shared_budget_qwen_matched"):
        relative = f"research/{name}.py"
        sources[relative] = file_hash(PROJECT / relative)
    settings = {
        "experiment": "qwen-1.5b-three-visits-sequential-acquisition-v3",
        "seeds": SEEDS, "epochs": args.epochs, "batch_size": args.batch_size,
        "passes_per_batch": 2, "learning_rate": args.learning_rate, "max_tokens": args.max_tokens,
        "source_hashes": sources, "data_hashes": {role: digest(rows) for role, rows in splits.items()},
        "reserve_seal": seal, "reserved_access": False,
        "base_model": str(MODEL.relative_to(PROJECT)),
        "model_hashes": {p.name: file_hash(p) for p in sorted(MODEL.glob("*"))
                         if p.is_file() and p.suffix in {".json", ".safetensors", ".txt"}},
        "inherited_adapter": str(OLD_PARENT.relative_to(PROJECT)),
        "inherited_adapter_hashes": {filename: file_hash(OLD_PARENT / filename)
                                    for filename in ("adapter_model.safetensors", "adapter_config.json")},
        "min_planning_margin": MIN_MARGIN, "max_current_loss": MAX_CURRENT_LOSS,
        "primary_metric": "longitudinal", "selection": "validation guards, utility, future, fewer questions, earliest",
        "versions": {name: version(name) for name in ("torch", "transformers", "peft")},
    }
    manifest = output / "PLAN.json"
    if args.resume and manifest.exists():
        if manifest.read_text(encoding="utf-8") != json.dumps(settings, ensure_ascii=False, indent=2, allow_nan=False) + "\n":
            raise ValueError("Resume requires the exact frozen plan")
    else:
        atomic_json(manifest, settings)

    status(output, "enumerating_train")
    train = trees_for(splits["train"], args.workers)
    planner = fit_public_policy(train, plan=True)
    myopic = fit_public_policy(train, plan=False)
    status(output, "enumerating_validation")
    dev = trees_for(splits["validation"], args.workers)
    refs = {RULE: rule_action, PLAN: planner, MYOPIC: myopic}
    cpu_rows = [row for method, policy in refs.items() for row in evaluate(dev, policy, method=method, seed=0)]
    margin = paired(cpu_rows, PLAN, MYOPIC, "longitudinal")
    atomic_json(output / "CPU_MARGIN.json", {"paired_planner_minus_myopic": margin, "rows": cpu_rows,
                                             "summary": summarize(cpu_rows)})
    if margin["estimate"] < MIN_MARGIN:
        raise ValueError("The extended bank did not create the prespecified margin over one-step decisions")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The experiment needs one bfloat16-capable CUDA device")
    status(output, "encoding_common_branches")
    torch.manual_seed(1301)
    model, tokenizer, labels = open_model(MODEL, OLD_PARENT)
    records, ledger = branch_data(train, actor_trees(train), planner, tokenizer, args.max_tokens)
    dev_public = actor_trees(dev)
    examples = validation_inputs(dev_public, tokenizer, args.max_tokens)
    atomic_json(output / "BRANCHES.json", ledger)
    atomic_json(output / "PUBLIC_PLANNER.json", list(planner.actions.items()))

    baseline = [row for row in cpu_rows if row["method"] == RULE]
    floor = max(0., stratified_mean(baseline, "current_supported") - MAX_CURRENT_LOSS)
    ceiling = stratified_mean(baseline, "future_unsupported_reuse")

    def rank(rows):
        def average(name):
            return stratified_mean(rows, name)
        eligible = (average("current_supported") >= floor - 1e-12
                    and average("future_unsupported_reuse") <= ceiling + 1e-12)
        return (eligible, round(average("longitudinal"), 12),
                round(average("future_supported"), 12), round(-average("questions"), 12))

    status(output, "checking_shared_parent")
    warm = competently_initialized(
        model, output, records, examples, tokenizer, labels,
        lambda policy, method: evaluate(dev_public, policy, method=method, seed=0), floor, settings,
    )
    parent_adapter = parent_path(output, warm)
    rows, seeds = [], {}
    for seed in SEEDS:
        directory = output / f"seed-{seed}"
        directory.mkdir(exist_ok=args.resume)
        seed_report = directory / "SEED_RESULT.json"
        if seed_report.is_file():
            result = json.loads(seed_report.read_text())
        else:
            status(output, "training_seed", seed=seed)
            load_adapter(model, parent_adapter)
            result = fit_matched(
                model, directory, records=records, evaluation=examples, tokenizer=tokenizer, labels=labels,
                device="cuda", settings={**settings, "seed": seed}, selection_rank=rank,
                validation_rows=lambda policy, method: evaluate(dev_public, policy, method=method, seed=seed),
                resume=args.resume,
            )
            atomic_json(seed_report, result)
        seeds[str(seed)] = {key: result[key] for key in ("selected_checkpoints", "accounting")}
        rows.extend(result["rows"])
        rows.extend({**row, "seed": seed} for row in cpu_rows)
    atomic_json(output / "RESULTS.json", {
        "status": "development_only", "summary": summarize(rows), "rows": rows, "seeds": seeds,
        "paired_grpo_minus_sft": {key: paired(rows, GRPO, SFT, key) for key in METRICS if key in rows[0]["metrics"]},
        "shared_parent": {"current_supported": warm["current_supported"], "floor": floor,
                          "warm_updates": sum(entry.get("optimizer_updates", 0) for entry in warm["attempts"])},
        "reserved_access": False,
    })
    adapters = {}
    for seed in SEEDS:
        adapters[str(seed)] = {}
        for method, item in seeds[str(seed)]["selected_checkpoints"].items():
            path = Path(f"seed-{seed}") / item["adapter"]
            adapters[str(seed)][method] = {
                "path": path.as_posix(), "epoch": item["epoch"], "eligible": item["eligible"],
                "weights_sha256": file_hash(output / path / "adapter_model.safetensors"),
                "config_sha256": file_hash(output / path / "adapter_config.json"),
            }
    atomic_json(output / "SELECTION_LOCK.json", {
        "status": "frozen_before_reserved_evaluation",
        "plan_sha256": file_hash(manifest), "validation_sha256": file_hash(output / "RESULTS.json"),
        "planner_sha256": file_hash(output / "PUBLIC_PLANNER.json"),
        "reserve_manifest_sha256": seal["manifest_sha256"], "adapters": adapters,
    })
    status(output, "completed", results="RESULTS.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reserve-seal", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.epochs, args.batch_size, args.max_tokens) < 1 or not 0 < args.learning_rate < 1:
        parser.error("Invalid worker count, epochs, batch size, token limit or rate")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    with (output / "RUN.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume and (output / "SELECTION_LOCK.json").is_file():
            status(output, "completed", results="RESULTS.json")
            return
        try:
            run_locked(args, output)
        except Exception as error:
            status(output, "failed", error_type=type(error).__name__, message=str(error))
            raise


if __name__ == "__main__":
    main()
