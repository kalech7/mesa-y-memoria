"""Paired Qwen retraining with a revised public parser and a GRPO temperature control."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import fcntl
from importlib.metadata import version
import json
from pathlib import Path

from research.src.artifacts import canonical, digest, file_hash
from research.tesis_extended_analysis import branch_data, validation_inputs
from research.tesis_extended_experiment import MODEL, OLD_PARENT, PROJECT, SEEDS, competently_initialized, open_model, parent_path
from research.tesis_profiles_analysis import evaluate, population
from research.tesis_profiles_bank import build_development
from research.tesis_profiles_learning import RL, fit_from_trees, load_policies
from research.tesis_profiles_optimizer_v5 import GRPO_V5, fit_matched_v5
from research.tesis_profiles_parser_v5 import coverage_diagnostic, enumerate_world, public_features, rule_action
from research.tesis_shared_budget_pilot import RULE, paired, source_hashes, stratified_mean, summarize
from research.tesis_shared_budget_planner import Decision
from research.tesis_shared_budget_qwen import atomic_json, model_policy, status
from research.tesis_shared_budget_qwen_matched import GRPO, PARENT, SFT, load_adapter


ARMS = (SFT, GRPO, GRPO_V5)
METRICS = ("longitudinal", "future_supported", "current_supported", "future_unsupported_reuse",
           "questions", "questions_first", "questions_third", "delivered_recontacts")
GUARDS = {"max_current_loss": .02, "max_extra_questions": .10, "max_extra_unsupported_reuse": 0.}
RESERVE = PROJECT / "research/data/tesis_profiles_heldout_v5"
SEAL = PROJECT / "reports/tesis_profiles_review_v5/SEAL.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def trees(stories, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(enumerate_world, stories, chunksize=2))


def actor_observation(observation, legal):
    note = {"role": "assistant", "turn_id": "assistant_profile_v5",
            "text": "Indicadores estimados del diálogo visible: " + canonical(public_features(observation, legal))}
    return {**observation, "user_messages": [*observation["user_messages"], note]}


def actor_view(forest):
    cache, result = {}, []
    for tree in forest:
        decisions = {}
        for path, node in tree.decisions.items():
            key = canonical([node.observation, node.legal])
            if key not in cache:
                cache[key] = Decision(actor_observation(node.observation, node.legal), node.legal)
            decisions[path] = cache[key]
        result.append(replace(tree, decisions=decisions))
    return result


def freeze_plan(run, args, splits):
    sources = source_hashes()
    for name in ("tesis_profiles_env", "tesis_profiles_bank", "tesis_profiles_learning", "tesis_profiles_analysis",
                 "tesis_profiles_parser_v5", "tesis_profiles_optimizer_v5", "tesis_profiles_semantic_v5", "tesis_profiles_v5",
                 "tesis_extended_env", "tesis_extended_analysis", "tesis_extended_experiment", "tesis_extended_heldout",
                 "tesis_shared_budget_qwen", "tesis_shared_budget_qwen_context", "tesis_shared_budget_qwen_matched"):
        relative = f"research/{name}.py"
        sources[relative] = file_hash(PROJECT / relative)
    seal = read(SEAL)
    manifest_hash = file_hash(RESERVE / "MANIFEST.json")
    loader_hash = file_hash(PROJECT / "research/tesis_profiles_heldout_v5.py")
    if seal["manifest_sha256"] != manifest_hash or seal["author_source_sha256"] != loader_hash:
        raise ValueError("Reserve seal does not match the manifest and loader")
    plan = {
        "seeds": list(SEEDS), "epochs": args.epochs, "batch_size": args.batch_size,
        "passes_per_batch": 2, "learning_rate": 1e-4, "max_tokens": 2048,
        "shared_bank": {s: digest(rows) for s, rows in splits.items()}, "reserve_seal": seal,
        "model_hashes": {p.name: file_hash(p) for p in sorted(MODEL.iterdir()) if p.suffix in
                         {".json", ".safetensors", ".txt"}},
        "initial_adapter_hashes": {p.name: file_hash(p) for p in sorted(OLD_PARENT.iterdir())
                                   if p.name in {"adapter_model.safetensors", "adapter_config.json"}},
        "sources": sources, "versions": {p: version(p) for p in ("torch", "peft", "transformers")},
        "methods": [PARENT, *ARMS], "primary_contrast": [GRPO_V5, SFT], "secondary_reference": GRPO,
        "selection": "eligible on current and unsupported reuse vs public rule, then utility, future, cost, earliest",
        "reserved_panels": ["validation_selected", "last_epoch"], "primary_panel": "validation_selected",
        "advantage": "both selected arms eligible on all seeds and lower paired 95% root CI on utility > 0 and all guards",
        "guards": GUARDS, "primary_metric": "longitudinal", "reserved_access_for_training": False,
    }
    if (run / "PLAN.json").exists():
        if not args.resume or read(run / "PLAN.json") != plan:
            raise ValueError("Existing run plan differs")
    else:
        atomic_json(run / "PLAN.json", plan)
    return plan


def pack_adapter(run, path, epoch, eligible):
    return {"adapter": str(path.relative_to(run)), "epoch": epoch, "eligible": eligible,
            "weights": file_hash(path / "adapter_model.safetensors"), "config": file_hash(path / "adapter_config.json")}


def reward_guard_diagnostic(forest, public, policy, tokenizer, limit, records):
    """Check whether utility-maximizing TRAIN actions can conflict with the reuse guard."""
    guard_forest = [replace(tree, outcomes={path: {**outcome, "longitudinal": -outcome["future_unsupported_reuse"]}
                          for path, outcome in tree.outcomes.items()}) for tree in forest]
    guard_records, _ = branch_data(guard_forest, public, policy, tokenizer, limit)
    by_role = {}
    for record, guard in zip(records, guard_records, strict=True):
        if record["record"] != guard["record"] or record["root_id"] != guard["root_id"]:
            raise ValueError("Unmatched return diagnostic")
        returns, safe = record["returns"], guard["returns"]
        optimal = [i for i, value in enumerate(returns) if value >= max(returns) - 1e-8]
        gap = max(safe) - max(safe[i] for i in optimal)
        for role in ("all", record["role"]):
            stats = by_role.setdefault(role, Counter())
            stats.update(groups=1, all_utility_optima_have_extra_reuse=int(gap > 1e-8),
                         weighted_extra_reuse=record["weight"] * gap, training_weight=record["weight"])
    return by_role


def train(run, args):
    import torch
    from transformers import AutoTokenizer

    splits = build_development()
    plan = freeze_plan(run, args, splits)
    if args.resume and (run / "LOCK.json").exists():
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires a bfloat16-capable GPU")
    status(run, "enumerating")
    train_trees, dev = (trees(splits[role], args.workers) for role in ("train", "validation"))
    study = fit_from_trees(train_trees, public_features=public_features, fixed_policy=rule_action)
    study.save(run / "TABULAR.json")
    cpu_rows = [row for label, policy in {RULE: rule_action, **study.policies}.items()
                for row in evaluate(dev, policy, method=label, seed=0)]
    atomic_json(run / "CPU.json", {"summary": summarize(cpu_rows), "rows": cpu_rows,
                "population": {"train": population(train_trees), "validation": population(dev)},
                "telemetry": study.telemetry})
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    public_train = actor_view(train_trees)
    records, ledger = branch_data(train_trees, public_train, study.policies[RL], tokenizer, plan["max_tokens"])
    ledger["reward_guard_diagnostic"] = reward_guard_diagnostic(
        train_trees, public_train, study.policies[RL], tokenizer, plan["max_tokens"], records)
    atomic_json(run / "BRANCHES.json", ledger)
    observed = actor_view(dev)
    prompts = validation_inputs(observed, tokenizer, plan["max_tokens"])
    torch.manual_seed(1301)
    model, tokenizer, labels = open_model(MODEL, OLD_PARENT)
    rule = summarize(cpu_rows)[RULE]["macro"]
    floor = max(0., rule["current_supported"] - GUARDS["max_current_loss"])
    ceiling = rule["future_unsupported_reuse"]

    def rank(rows):
        def avg(key):
            return stratified_mean(rows, key)
        eligible = avg("current_supported") + 1e-12 >= floor and avg("future_unsupported_reuse") <= ceiling + 1e-12
        return (eligible, round(avg("longitudinal"), 12), round(avg("future_supported"), 12), -round(avg("questions"), 12))

    warm = competently_initialized(model, run, records, prompts, tokenizer, labels,
                  lambda policy, label: evaluate(observed, policy, method=label, seed=0), floor, plan)
    rows, panels = [], {"validation_selected": {}, "last_epoch": {}}
    for seed in SEEDS:
        status(run, "training", seed=seed)
        directory = run / f"seed-{seed}"
        directory.mkdir(exist_ok=args.resume)
        archive = directory / "SEED_RESULT.json"
        if archive.exists() and args.resume:
            result = read(archive)
        else:
            load_adapter(model, parent_path(run, warm))
            result = fit_matched_v5(model, directory, records=records, evaluation=prompts, tokenizer=tokenizer,
                                    labels=labels, device="cuda", settings={**plan, "seed": seed},
                                    validation_rows=lambda policy, label: evaluate(observed, policy, method=label, seed=seed),
                                    selection_rank=rank, resume=args.resume)
            atomic_json(archive, result)
        rows.extend(result["rows"])
        selected, last = {}, {}
        for label in (PARENT, *ARMS):
            item = result["selected_checkpoints"][label]
            selected[label] = pack_adapter(run, directory / item["adapter"], item["epoch"], item["eligible"])
            if label != PARENT:
                ledger = result["accounting"]["arms"][label][-1]
                last[label] = pack_adapter(run, directory / label / f"checkpoint-{args.epochs:03d}" / "adapter",
                                           args.epochs, rank(ledger["rows"])[0])
        panels["validation_selected"][str(seed)] = selected
        panels["last_epoch"][str(seed)] = last
    atomic_json(run / "VALIDATION.json", {"summary": summarize(rows), "rows": rows, "current_floor": floor,
                                         "unsupported_ceiling": ceiling})
    frozen = ("PLAN.json", "VALIDATION.json", "TABULAR.json", "BRANCHES.json",
              *(f"seed-{s}/SEED_RESULT.json" for s in SEEDS))
    atomic_json(run / "LOCK.json", {"panels": panels, "hashes": {p: file_hash(run / p) for p in frozen}})
    status(run, "selection_locked")


def comparison(rows, other, eligible):
    stats = {k: paired(rows, GRPO_V5, other, k) for k in METRICS}
    passed = (eligible and stats["longitudinal"]["interval"][0] > 0
              and stats["current_supported"]["estimate"] >= -GUARDS["max_current_loss"] - 1e-12
              and stats["questions"]["estimate"] <= GUARDS["max_extra_questions"] + 1e-12
              and stats["future_unsupported_reuse"]["estimate"] <= GUARDS["max_extra_unsupported_reuse"] + 1e-12)
    return {"both_eligible_on_all_seeds": eligible, "passes": passed, "contrasts": stats}


def score(run, args):
    from research import tesis_profiles_heldout_v5 as bank
    from research.tesis_profiles_semantic_v5 import semantic_comparison
    import torch

    output = run / "RESERVED_RESULTS.json"
    if output.exists():
        raise FileExistsError(output)
    plan, lock = read(run / "PLAN.json"), read(run / "LOCK.json")
    expected = {run / p: sha for p, sha in lock["hashes"].items()}
    expected.update({PROJECT / p: sha for p, sha in plan["sources"].items()})
    expected.update({MODEL / p: sha for p, sha in plan["model_hashes"].items()})
    expected[RESERVE / "MANIFEST.json"] = plan["reserve_seal"]["manifest_sha256"]
    expected[Path(bank.__file__)] = plan["reserve_seal"]["author_source_sha256"]
    for seeds in lock["panels"].values():
        for methods in seeds.values():
            for adapter in methods.values():
                path = (run / adapter["adapter"]).resolve()
                if not path.is_relative_to(run):
                    raise ValueError("Checkpoint outside the locked run")
                expected[path / "adapter_model.safetensors"] = adapter["weights"]
                expected[path / "adapter_config.json"] = adapter["config"]
    for path, sha in expected.items():
        if file_hash(path) != sha:
            raise ValueError(f"Locked artifact changed: {path}")
    bank.verify(RESERVE)
    raw = bank.load_executable(RESERVE)
    if set(s["split"] for s in raw) != {"reserved"}:
        raise ValueError("Expected reserved stories")
    status(run, "scoring_reserved")
    coverage = coverage_diagnostic(raw)
    semantic = semantic_comparison(raw, bank.load_gold(RESERVE))
    reserved = trees(raw, args.workers)
    tables = load_policies(run / "TABULAR.json", public_features=public_features, fallback=rule_action)
    tb = [row for label, policy in {RULE: rule_action, **tables}.items()
          for row in evaluate(reserved, policy, method=label, seed=0)]
    observed = actor_view(reserved)
    initial = lock["panels"]["validation_selected"][str(SEEDS[0])][PARENT]
    model, tokenizer, labels = open_model(MODEL, run / initial["adapter"], trainable=False)
    prompts = validation_inputs(observed, tokenizer, plan["max_tokens"])
    cache, panels = {}, {}
    for panel_name, seeds in lock["panels"].items():
        rows = []
        for seed, methods in seeds.items():
            for label, item in methods.items():
                key = item["weights"], item["config"]
                if key not in cache:
                    load_adapter(model, run / item["adapter"])
                    counter = Counter()
                    policy = model_policy(model, prompts, tokenizer, labels, "cuda", plan["batch_size"], counter)
                    cache[key] = evaluate(observed, policy, method=label, seed=int(seed))
                rows.extend({**row, "method": label, "seed": int(seed)} for row in cache[key])
        panels[panel_name] = {"summary": summarize(rows), "rows": rows,
            "comparisons": {other: comparison(rows, other,
                all(methods[m]["eligible"] for methods in seeds.values() for m in (other, GRPO_V5)))
                for other in (SFT, GRPO)}}
    atomic_json(output, {"device": torch.cuda.get_device_name(0), "population": population(reserved),
                "selection_sha256": file_hash(run / "LOCK.json"), "panels": panels,
                "parser": coverage, "semantic": semantic, "tabular": {"summary": summarize(tb), "rows": tb,
                    "coverage": {m: policy.coverage() for m, policy in tables.items()}}})
    status(run, "completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "score"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.epochs, args.batch_size) < 1:
        parser.error("Positive sizes required")
    run = args.run.resolve()
    if args.stage == "train":
        run.mkdir(parents=True, exist_ok=args.resume)
    with (run / "RUN.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            (train if args.stage == "train" else score)(run, args)
        except Exception as exc:
            status(run, "failed", message=str(exc))
            raise


if __name__ == "__main__":
    main()
