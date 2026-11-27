"""Matched Qwen LoRA SFT and two-pass categorical GRPO on rotated dialogue menus."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from statistics import mean

import torch

from research.src.analysis.paired_statistics import paired_bootstrap
from research.src.artifacts import file_hash, write_json
from research.tesis_meal_cases import LETTERS
from research.tesis_meal_train import action_tokens, load, logits, reset, rewards, scores, state, write_rows
from research.tesis_meal_varied_bank import build_varied, prompt, rotate, normalized
from web_demo.cases import MODELS


SEED = 416
BATCH = 4
WARM_EPOCHS = 2
CONT_EPOCHS = 2
PASSES = 2
DRAWS = 8
LR = 2e-4
TEMPERATURE = 1.4
CLIP = .2
KL = .025
METHODS = ("inicial", "sft-comun", "sft", "grpo")
METRICS = ("success", "conflict", "unnecessary_clarification", "missed_clarification", "raw_action_success")


def pack(actor, roots, offsets=range(6)):
    return [(item, actor.encode_prompt(prompt(item["turns"], item["options"])))
            for row in roots for offset in offsets for item in (rotate(row, offset),)]


def schedule(roots, seed, epochs):
    rng = random.Random(seed)
    offsets = {row["id"]: rng.randrange(6) for row in roots}
    for epoch in range(epochs):
        yield [(row["id"], (offsets[row["id"]] + 3 * epoch + shift) % 6)
               for row in rng.sample(roots, len(roots)) for shift in (0, 1, 2)]


def optimizer(actor):
    return torch.optim.AdamW([p for p in actor.model.parameters() if p.requires_grad], lr=LR, weight_decay=0.)


def distribution(array, tokens):
    return (array[:, tokens] / TEMPERATURE).log_softmax(-1)


def supervised(array, tokens, rows, *, warmup=False):
    logp = array.log_softmax(-1)[:, tokens] if warmup else distribution(array, tokens)
    return -torch.stack([line[row["accepted"]].mean() for line, row in zip(logp, rows, strict=True)]).mean()


def advantages(returns):
    return (returns - returns.mean(1, keepdim=True)) / returns.std(1, unbiased=False, keepdim=True).clamp_min(.5)


def optimize(actor, tokens, data, order, opt, method, rng, path, reference=None):
    ledger = []
    for start in range(0, len(order), BATCH):
        keys = order[start:start+BATCH]
        rows, inputs = zip(*(data[key] for key in keys))
        if any(row["split"] != "train" for row in rows):
            raise ValueError("Training includes held-out dialogue")
        chosen = returns = old = None
        if method == "grpo":
            with torch.no_grad():
                old = distribution(logits(actor, inputs), tokens)
                chosen = torch.multinomial(old.exp(), DRAWS, replacement=True, generator=rng)
                returns = torch.tensor([rewards(row) for row in rows], device=actor.device).gather(1, chosen)
                relative = advantages(returns)
                target = torch.stack([reference[key] for key in keys])
        steps = []
        for _ in range(1 if method == "warm" else PASSES):
            opt.zero_grad(set_to_none=True)
            array = logits(actor, inputs)
            if method == "grpo":
                logp = distribution(array, tokens)
                ratio = (logp.gather(1, chosen) - old.gather(1, chosen)).exp()
                surrogate = torch.minimum(ratio * relative, ratio.clamp(1 - CLIP, 1 + CLIP) * relative)
                divergence = (logp.exp() * (logp - target)).sum(-1).mean()
                loss = -surrogate.mean() + KL * divergence
            else:
                loss = supervised(array, tokens, rows, warmup=method == "warm")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in actor.model.parameters() if p.requires_grad], 1.)
            opt.step()
            steps.append(float(loss.detach()))
        entry = {"cases": keys, "losses": steps}
        if chosen is not None:
            entry.update(draws=chosen.tolist(), returns=returns.tolist())
        ledger.append(entry)
        if len(ledger) % 100 == 0:
            print(path.stem, len(ledger), "/", (len(order) + BATCH - 1)//BATCH, flush=True)
    write_rows(path, ledger)


def summarize(rows):
    grouped = defaultdict(list)
    pairs = defaultdict(list)
    families = defaultdict(list)
    for row in rows:
        grouped[row["root_id"]].append(row)
        if row["pair"]:
            pairs[row["pair"], row["rotation"]].append(row)
        families[row["family"]].append(row)
    return {"all": {key: mean(row[key] for row in rows) for key in METRICS},
            "family": {family: {key: mean(row[key] for row in cases) for key in METRICS}
                       for family, cases in sorted(families.items())},
            "all_orders_success": mean(all(r["success"] for r in group) for group in grouped.values()),
            "same_valid_dish": mean(all(r["success"] for r in group) and
                                    len({row["original_action"] for row in group}) == 1 for group in grouped.values()),
            "paired_flips": mean(len(group) == 2 and all(r["success"] for r in group)
                                 and len({r["original_action"] for r in group}) == 2 for group in pairs.values())}


def evaluate(actor, tokens, data, path):
    rows = []
    with torch.inference_mode():
        for start in range(0, len(data), BATCH):
            batch = data[start:start+BATCH]
            array = logits(actor, [ids for _, ids in batch])
            menu_actions = array[:, tokens].argmax(-1).tolist()
            raw = array.argmax(-1).tolist()
            for (source, _), decision, raw_id in zip(batch, menu_actions, raw, strict=True):
                raw_index = tokens.index(raw_id) if raw_id in tokens else -1
                rows.append({"root_id": source["id"], "realization_id": str(source["rotation"]),
                             "rotation": source["rotation"], "pair": source["pair"], "family": source["family"],
                             "option": LETTERS[decision], "original_action": source["order"][decision],
                             "raw_token": actor.tokenizer.decode([raw_id]),
                             "raw_action_success": int(raw_index in source["accepted"]),
                             **scores(source, decision)})
    write_rows(path, rows)
    return summarize(rows), rows


def rank(metrics, epoch):
    return metrics["all"]["success"], metrics["paired_flips"], -metrics["all"]["conflict"], -epoch


def run(output, device, seed):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    bank = build_varied()
    output.mkdir(parents=True)
    for name, rows in bank.items():
        write_rows(output / f"{name}.jsonl", rows)
    parent = MODELS["sft"].path
    sources = {name: file_hash(Path(__file__).with_name(name)) for name in (
        "tesis_meal_varied.py", "tesis_meal_varied_bank.py", "tesis_meal_dialogues_train.py",
        "tesis_meal_dialogues_eval.py", "tesis_meal_train.py", "tesis_meal_cases.py")}
    plan_path = output / "PLAN.json"
    write_json(plan_path, {"seed": seed, "device": device, "source_sha256": sources,
                          "bank_sha256": {name: file_hash(output / f"{name}.jsonl") for name in bank},
                          "parent_sha256": file_hash(parent / "adapter_model.safetensors"),
                          "warm_epochs": WARM_EPOCHS, "epochs": CONT_EPOCHS, "draws": DRAWS, "batch": BATCH,
                          "passes": PASSES, "rotation_offsets_per_root": 6, "reward": [1., -1., -.35],
                          "primary": "mean guard-free correct decision across six rotations per root",
                          "tie_break": "semantic flips, fewer conflicts, earliest", "data": "authored synthetic dialogues"})
    actor = load(parent, device)
    tokens = action_tokens(actor.tokenizer)
    train = {(row["id"], row["rotation"]): (row, ids) for row, ids in pack(actor, bank["train"])}
    validation = pack(actor, bank["validation"])
    starting = state(actor)
    rng = torch.Generator(device=actor.device).manual_seed(seed)
    opt = optimizer(actor)
    for epoch, order in enumerate(schedule(bank["train"], seed, WARM_EPOCHS), 1):
        optimize(actor, tokens, train, order, opt, "warm", rng, output / f"warm-{epoch}.jsonl")
    common = state(actor)
    val, _ = evaluate(actor, tokens, validation, output / "sft-comun-validation.jsonl")
    print("SFT común", val["all"], val["all_orders_success"], val["paired_flips"], flush=True)
    with torch.no_grad():
        reference = {}
        keys = list(train)
        for i in range(0, len(keys), BATCH):
            group = keys[i:i+BATCH]
            values = distribution(logits(actor, [train[key][1] for key in group]), tokens)
            reference.update(zip(group, values.unbind(), strict=True))
    selections = {}
    weights = {"inicial": starting, "sft-comun": common}
    orders = list(schedule(bank["train"], seed + 1, CONT_EPOCHS))
    for method in METHODS[2:]:
        reset(actor, common)
        opt = optimizer(actor)
        best_weights, best_epoch, validation_scores = common, 0, {0: val}
        for epoch, order in enumerate(orders, 1):
            optimize(actor, tokens, train, order, opt, method, rng, output / f"{method}-train-{epoch}.jsonl", reference)
            result, _ = evaluate(actor, tokens, validation, output / f"{method}-validation-{epoch}.jsonl")
            validation_scores[epoch] = result
            print(method, epoch, result["all"], result["all_orders_success"], result["paired_flips"], flush=True)
            if rank(result, epoch) > rank(validation_scores[best_epoch], best_epoch):
                best_weights, best_epoch = state(actor), epoch
        reset(actor, best_weights)
        actor.save_checkpoint(output / method)
        weights[method] = best_weights
        selections[method] = {"epoch": best_epoch, "validation": validation_scores,
                              "manifest_sha256": file_hash(output / method / "local_checkpoint.json"),
                              "adapter_sha256": file_hash(output / method / "adapter_model.safetensors")}
    path = output / "SELECTION.json"
    write_json(path, {"split": "validation", "plan_sha256": file_hash(plan_path), "methods": selections})
    reserved = pack(actor, bank["evaluation"])
    roots = {row["id"]: row for row in bank["evaluation"]}
    results, paired = {}, []
    for method, values in weights.items():
        reset(actor, values)
        path_trace = output / f"{method}-evaluation.jsonl"
        stats, observed = evaluate(actor, tokens, reserved, path_trace)
        results[method] = {"metrics": stats, "trace_sha256": file_hash(path_trace)}
        if method in METHODS[2:]:
            paired.extend({"root_id": normalized(roots[r["root_id"]]["turns"]),
                           "realization_id": r["root_id"] + ":" + r["realization_id"], "scenario_family": r["family"],
                           "seed": seed, "method": method, "success": r["success"]} for r in observed)
    contrast = paired_bootstrap(paired, "grpo", "sft", "success", confidence=.95, resamples=2000)
    result = {"selection_sha256": file_hash(path), "panels": results, "grpo_minus_sft": contrast}
    write_json(output / "RESULTS.json", result)
    print({key: value["metrics"]["all"] for key, value in results.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run(args.output, args.device, args.seed)
