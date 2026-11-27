"""Matched Qwen LoRA pilot for recognizing opposite public agendas."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from statistics import mean

from research.src.artifacts import digest, file_hash
from research.tesis_shared_budget_qwen import atomic_json, batches, padded_scoring
from research.tesis_timing_v6_bank import build_worlds, FROZEN_WORLD_SHA256, verify_worlds
from research.tesis_timing_v6_env import fit_public_planner, prompt_text
from research.tesis_timing_v6_eval import evaluate_pool, group_worlds, independent_timing, paired_difference, score_public
from research.tesis_timing_v6_experiment import (
    BASE_FILES, PROJECT, SEEDS, V6_SOURCES, WEIGHTS, PublicScorer, calibrate, forest, signature,
)
from research.tesis_timing_v6_train import MODEL, PARENT, MarginScorer
from research.tesis_timing_v7_bank import reserve_worlds

ARMS = ("sft", "sft-pares")
EPOCHS = (2, 4, 6)
LR = 1e-4
PAIRS_PER_BATCH = 2
RESERVE_SHA = "2dc0cc00a44040a06600bd9bffd9875cefd010bbadc3662f6b50a5b454d86791"


def read(path):
    return json.loads(path.read_text())


def training_pairs(roots):
    """Labels come from weighted SQLite outcomes, never a private agenda flag."""
    grouped = defaultdict(list)
    early, late = ("ASK", "WAIT"), ("WAIT", "ASK")
    for root in roots:
        earlier = root.outcomes[early]["future_supported"]
        later = root.outcomes[late]["future_supported"]
        if earlier == later or later <= root.outcomes[("WAIT", "WAIT")]["future_supported"]:
            raise ValueError("Timing and second-visit targets must have strict branch evidence")
        grouped[root.pair].append((earlier > later, root))
    selected = []
    for pair, members in sorted(grouped.items()):
        if sorted(flag for flag, _ in members) != [False, True]:
            raise ValueError(f"Expected opposite training targets for pair {pair}")
        negative, positive = (root for _, root in sorted(members, key=lambda row: row[0]))
        selected.append(tuple(prompt_text(obs) for obs in (
            positive.first, negative.first, positive.second_after_wait, negative.second_after_wait)))
    return selected


def loss_for(margins, arm):
    import torch
    import torch.nn.functional as F

    if arm not in ARMS:
        raise ValueError(f"Unknown objective {arm}")
    first_pos, first_neg, second_pos, second_neg = margins.reshape(-1, 4).unbind(1)
    signed = torch.stack((first_pos, -first_neg, second_pos, second_neg))
    loss = F.softplus(-signed).mean()
    if arm == "sft-pares":
        loss = loss + F.softplus(first_neg - first_pos).mean()
    return loss


def measure(roots, margins, cutoff):
    pooled = evaluate_pool(roots, margins, cutoff)
    return {"macro": pooled["macro"], "packets": {
        p: {"metrics": row["metrics"]} for p, row in pooled["packets"].items()
    }, "individual": independent_timing(roots, margins)}


def fit(run, seed, arm, train, valid, scorer):
    import torch

    directory = run / f"seed-{seed}" / arm
    directory.mkdir(parents=True, exist_ok=True)
    finished = directory / "FIT.json"
    if finished.is_file():
        outcome = read(finished)
        if signature(directory / outcome["adapter"], WEIGHTS) != outcome["weights"]:
            raise ValueError("Previously selected adapter has changed")
        return outcome
    # A partial arm is deterministically retrained from the shared parent.
    scorer.active = None
    scorer.load(PARENT)
    optimizer = torch.optim.AdamW((p for p in scorer.model.parameters() if p.requires_grad), lr=LR)
    torch.manual_seed(seed)
    history, candidates, counter = [], [], Counter()
    for epoch in range(1, EPOCHS[-1] + 1):
        scorer.model.train()
        order = list(range(len(train)))
        random.Random(seed + epoch).shuffle(order)
        losses = []
        for indexes in batches(order, PAIRS_PER_BATCH):
            records = [scorer.record(text) for i in indexes for text in train[i]]
            logits = padded_scoring(scorer.model, records, scorer.labels, scorer.tokenizer.pad_token_id, "cuda", counter)
            loss = loss_for(logits[:, 1] - logits[:, 0], arm)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        history.append({"epoch": epoch, "mean_loss": mean(losses), "order_sha256": digest(order)})
        if epoch not in EPOCHS:
            continue
        scorer.cache.clear()
        margins = score_public(valid, scorer)
        cutoff = calibrate(valid, margins)[0]["cutoff"]
        metrics = measure(valid, margins, cutoff)
        adapter = f"epoch-{epoch}/adapter"
        scorer.model.save_pretrained(directory / adapter, safe_serialization=True)
        candidates.append({"adapter": adapter, "epoch": epoch, "cutoff": cutoff,
                           "future": metrics["macro"]["future_supported"],
                           "free_pairs": metrics["individual"]["both_timings_correct_native"]})
    best = max(candidates, key=lambda r: (r["free_pairs"], r["future"], -r["epoch"]))
    outcome = {**best, "weights": signature(directory / best["adapter"], WEIGHTS),
               "candidates": candidates, "training": history, "budget": dict(counter)}
    atomic_json(finished, outcome)
    print(f"seed={seed} {arm}: validation={best}", flush=True)
    return outcome


def setup(run, v6):
    import torch
    from importlib.metadata import version

    sources = V6_SOURCES + ["research/tesis_timing_v7.py", "research/tesis_timing_v7_bank.py"]
    cfg = {
        "seeds": list(SEEDS), "arms": list(ARMS), "epochs_considered": list(EPOCHS), "learning_rate": LR,
        "pair_minibatch": PAIRS_PER_BATCH, "train_validation_worlds": {k: FROZEN_WORLD_SHA256[k] for k in ("train", "validation")},
        "reserve_worlds": RESERVE_SHA, "reserve_pairs": 80, "reserve_packets": 8,
        "qwen": signature(MODEL, BASE_FILES), "parent": signature(PARENT, WEIGHTS),
        "source_files": signature(PROJECT, sources), "historical_lock": file_hash(v6 / "LOCK.json"),
        "selection": "maximize native free pair accuracy, then matched future accuracy, then earliest epoch",
        "allocation": "10 questions per 20 histories; validate first-visit threshold before scoring",
        "primary": "sft-pares minus matched sft; fraction of pairs with both native decisions correct",
        "secondary": "future accuracy at equal expenditure, paired percentile interval over full packets",
        "environment": {"device": torch.cuda.get_device_name(), **{k: version(k) for k in ("torch", "peft", "transformers")}},
    }
    if (run / "PLAN.json").exists() and read(run / "PLAN.json") != cfg:
        raise ValueError("Different experimental plan for this run")
    atomic_json(run / "PLAN.json", cfg)


def develop(run):
    train_worlds, validation_worlds = build_worlds("train"), build_worlds("validation")
    verify_worlds("train", train_worlds)
    verify_worlds("validation", validation_worlds)
    train_forest = forest(train_worlds, 1)
    planner = PublicScorer(fit_public_planner(train_forest))
    train = training_pairs(group_worlds(train_forest))
    valid = group_worlds(forest(validation_worlds, 1))
    planner_cut = calibrate(valid, score_public(valid, planner))[0]["cutoff"]
    scorer = MarginScorer(trainable=True)
    selected = {}
    for seed in SEEDS:
        results = {arm: fit(run, seed, arm, train, valid, scorer) for arm in ARMS}
        if results["sft"]["budget"] != results["sft-pares"]["budget"]:
            raise ValueError("SFT and SFT-pairs used unequal training budgets")
        selected[str(seed)] = {arm: {k: row[k] for k in ("epoch", "adapter", "weights", "cutoff")}
                               for arm, row in results.items()}
    lock = {"plan_sha256": file_hash(run / "PLAN.json"), "selected": selected, "planner_cutoff": planner_cut}
    if (run / "LOCK.json").exists() and read(run / "LOCK.json") != lock:
        raise ValueError("Selection differs from the locked evaluation")
    atomic_json(run / "LOCK.json", lock)
    return planner


def score(run, v6, planner):
    lock = read(run / "LOCK.json")
    worlds = reserve_worlds()
    if digest(worlds) != RESERVE_SHA:
        raise ValueError("New reserve differs from sealed digest")
    roots = group_worlds(forest(worlds, 1))
    scorer, methods = MarginScorer(), defaultdict(dict)
    historic = read(v6 / "LOCK.json")["selected"]
    for seed, choices in lock["selected"].items():
        for name, row in {**choices, **{arm: historic[seed][arm] for arm in ("SFT-ramas", "GRPO-continuacion")}}.items():
            path = run / f"seed-{seed}" / name / row["adapter"] if name in ARMS else v6 / row["adapter"]
            if signature(path, WEIGHTS) != (row["weights"] if name in ARMS else row["sha256"]):
                raise ValueError(f"Mismatch in checkpoint checksum for {name}")
            scorer.load(path)
            methods[name][seed] = measure(roots, score_public(roots, scorer), row["cutoff"])
    public = measure(roots, score_public(roots, planner), lock["planner_cutoff"])
    methods["planificador-publico"] = {str(s): public for s in SEEDS}
    primary = {}
    for name in methods:
        rates = [methods[name][str(s)]["individual"]["both_timings_correct_native"] for s in SEEDS]
        primary[name] = {"per_seed": rates, "mean": mean(rates)}
    a, b = ARMS
    result = {"lock_sha256": file_hash(run / "LOCK.json"), "reserve_worlds_sha256": RESERVE_SHA,
              "pairs": 80, "histories": len(roots), "primary_free_pairs": primary,
              "primary_difference": primary[b]["mean"] - primary[a]["mean"],
              "matched_future_difference": paired_difference(methods[b], methods[a]), "methods": methods}
    atomic_json(run / "RESERVED_RESULTS.json", result)


def main():
    import fcntl

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True)
    cli.add_argument("--v6-run", type=Path, default=PROJECT / "reports/tesis_timing_qwen_a100_v6")
    args = cli.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    with (args.run / ".pipeline.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        setup(args.run, args.v6_run)
        planner = develop(args.run)
        score(args.run, args.v6_run, planner)


if __name__ == "__main__":
    main()
