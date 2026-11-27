"""Fit and compare real Qwen LoRA recommendation policies on held-out Spanish chats."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from statistics import mean

import torch

from research.grpo_select_checkpoint import load_selector
from research.src.analysis.paired_statistics import paired_bootstrap
from research.src.artifacts import canonical, file_hash, write_json
from research.tesis_meal_cases import LETTERS, FAMILIES, build, prompt
from web_demo.cases import BASE, MODELS
from web_demo.meal_policy import option_tokens as action_tokens, final_logits as logits


SEED = 416
WARMUP = 144
STEPS = 96
BATCH = 2
DRAWS = 6
LR = .00015
CLIP = .2
BETA = .03
TEMPERATURE = 1.5
CHECKPOINTS = (48, STEPS)
ARMS = ("sft", "grpo")
BASELINE = "inicial"
COMMON = "sft-comun"


def load(checkpoint, device):
    actor = load_selector(checkpoint, model_path=BASE, device=device, trainable=True)
    if device == "cpu":
        actor.model.float()
    actor.config.gradient_checkpointing = False
    actor.model.gradient_checkpointing_disable()
    return actor


def encode(actor, row):
    return actor.encode_prompt(prompt(row["turns"], row["options"]))


def action_logprobs(array, tokens, temperature=1.):
    return (array[:, tokens] / temperature).log_softmax(-1)


def scores(row, selected):
    option = row["options"][selected]
    avoided_conflict = not bool(set(option["ingredients"]) & set(row["forbidden"]))
    accepted = selected in row["accepted"]
    clarify = not option["ingredients"]
    expected_clarify = not row["options"][row["accepted"][0]]["ingredients"]
    return {"success": int(accepted), "conflict": int(not avoided_conflict),
            "unnecessary_clarification": int(clarify and not expected_clarify),
            "missed_clarification": int(not clarify and expected_clarify)}


def rewards(row):
    values = []
    for action in range(len(row["options"])):
        m = scores(row, action)
        values.append(1. if m["success"] else -1. if m["conflict"] or m["missed_clarification"] else -.35)
    return values


def sft_loss(array, tokens, rows):
    logp = array.log_softmax(-1)[:, tokens]
    # Uniform supervised targets across every author-accepted action.
    return -torch.stack([line[example["accepted"]].mean() for line, example in zip(logp, rows)]).mean()


def grpo_loss(array, tokens, rows, reference, generator):
    logp = action_logprobs(array, tokens, TEMPERATURE)
    old = logp.detach()
    # Sample each group from the actual six-action policy and use its same
    # distribution for the recentered advantage and clipped likelihood ratio.
    sampled = torch.multinomial(old.exp(), DRAWS, replacement=True, generator=generator)
    returns = torch.tensor([rewards(row) for row in rows], device=array.device).gather(1, sampled)
    centered = returns - returns.mean(dim=1, keepdim=True)
    advantages = centered / returns.std(dim=1, unbiased=False, keepdim=True).clamp_min(.5)
    ratio = (logp.gather(1, sampled) - old.gather(1, sampled)).exp()
    surrogate = torch.minimum(ratio * advantages, ratio.clamp(1 - CLIP, 1 + CLIP) * advantages)
    kl = (logp.exp() * (logp - reference)).sum(dim=-1).mean()
    loss = -surrogate.mean() + BETA * kl
    return loss, {"sampled": sampled.tolist(), "returns": returns.tolist(),
                  "nonconstant_groups": int((returns.max(1).values != returns.min(1).values).sum())}


def evaluate(actor, tokens, packed, destination):
    rows = []
    with torch.inference_mode():
        for offset in range(0, len(packed), BATCH):
            batch = packed[offset:offset+BATCH]
            output = logits(actor, [ids for _, ids in batch])
            legal = output[:, tokens].argmax(-1).tolist()
            unconstrained = output.argmax(-1).tolist()
            for (source, _), index, raw_id in zip(batch, legal, unconstrained):
                decoded = actor.tokenizer.decode([raw_id])
                # The raw metric allows the whole vocabulary and penalizes malformed choices.
                raw_action = tokens.index(raw_id) if raw_id in tokens else None
                raw_success = int(raw_action is not None and raw_action in source["accepted"])
                rows.append({"id": source["id"], "family": source["family"], "split": source["split"],
                             "action": LETTERS[index], "dish": source["options"][index]["name"],
                             "raw_first_token": decoded, "raw_action_success": raw_success,
                             **scores(source, index)})
    write_rows(destination, rows)
    return summary(rows)


def summary(rows):
    keys = ("success", "conflict", "unnecessary_clarification", "missed_clarification", "raw_action_success")
    return {"all": {key: mean(row[key] for row in rows) for key in keys},
            "families": {family: {key: mean(row[key] for row in rows if row["family"] == family) for key in keys}
                         for family in FAMILIES if any(row["family"] == family for row in rows)}}


def write_rows(path, rows):
    with path.open("x", encoding="utf-8") as output:
        for row in rows:
            output.write(canonical(row) + "\n")


def pack(actor, rows):
    return [(row, encode(actor, row)) for row in rows]


def state(actor):
    return {name: p.detach().cpu().clone() for name, p in actor.model.named_parameters() if p.requires_grad}


def reset(actor, weights):
    with torch.no_grad():
        for name, param in actor.model.named_parameters():
            if param.requires_grad:
                param.copy_(weights[name])


def optimize(actor, tokens, schedule, optimizer, *, method, references=None, filename):
    generator = torch.Generator(device=actor.device).manual_seed(SEED)
    history = []
    for offset in range(0, len(schedule), BATCH):
        batch = schedule[offset:offset+BATCH]
        rows, ids = zip(*batch)
        if any(row["split"] != "train" for row in rows):
            raise ValueError("Validation or evaluation data reached the optimizer")
        optimizer.zero_grad(set_to_none=True)
        array = logits(actor, ids)
        if method == "grpo":
            reference = torch.stack([references[row["id"]] for row in rows])
            loss, telemetry = grpo_loss(array, tokens, rows, reference, generator)
        else:
            loss, telemetry = sft_loss(array, tokens, rows), {}
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in actor.model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        step = offset // BATCH + 1
        history.append({"step": step, "cases": [row["id"] for row in rows], "loss": float(loss.detach()), **telemetry})
        if step % 8 == 0:
            print(f"{filename.stem} · {step}/{len(schedule)//BATCH} · loss {history[-1]['loss']:.4f}", flush=True)
    write_rows(filename, history)


def optimizer(actor):
    return torch.optim.AdamW([p for p in actor.model.parameters() if p.requires_grad], lr=LR, weight_decay=0.)


def rank(metrics, step):
    m = metrics["all"]
    return m["success"], -m["conflict"], m["raw_action_success"], -step


def run(output, device):
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    output.mkdir(parents=True)
    bank = build()
    for name, rows in bank.items():
        write_rows(output / f"{name}.jsonl", rows)
    parent = MODELS["sft"].path
    source_files = (Path(__file__), Path(__file__).with_name("tesis_meal_cases.py"))
    plan = {"seed": SEED, "device": device, "parent": str(parent),
            "parent_sha256": file_hash(parent / "adapter_model.safetensors"),
            "source_sha256": {path.name: file_hash(path) for path in source_files},
            "bank_sha256": {name: file_hash(output / f"{name}.jsonl") for name in bank},
            "warmup": WARMUP, "updates_per_arm": STEPS, "batch": BATCH, "draws": DRAWS,
            "learning_rate": LR, "temperature": TEMPERATURE, "kl_weight": BETA,
            "primary": "compatible choice before chat guard", "selection": "validation",
            "variant": "categorical GRPO, single decision, one optimizer step per sampled group",
            "scope": "exploratory single seed; held-out recipes, turns, wordings and typo forms"}
    write_json(output / "PLAN.json", plan)
    actor = load(parent, device)
    tokens = action_tokens(actor.tokenizer)
    train, validation = pack(actor, bank["train"]), pack(actor, bank["validation"])
    selections = {}
    selections[BASELINE] = {"validation": evaluate(actor, tokens, validation, output / "inicial-validation.jsonl")}
    parent_weights = state(actor)
    rng = random.Random(SEED)
    warm_order = [row for _ in range(WARMUP * BATCH // len(train)) for row in rng.sample(train, len(train))]
    order = [row for _ in range(STEPS * BATCH // len(train)) for row in rng.sample(train, len(train))]
    optimize(actor, tokens, warm_order, optimizer(actor), method="sft", filename=output / "sft-comun-train.jsonl")
    warm = state(actor)
    base = evaluate(actor, tokens, validation, output / "sft-comun-validation.jsonl")
    selections[COMMON] = {"validation": base}
    references = {}
    with torch.inference_mode():
        for index in range(0, len(train), BATCH):
            subset = train[index:index+BATCH]
            values = action_logprobs(logits(actor, [ids for _, ids in subset]), tokens, TEMPERATURE)
            references.update((row["id"], v.clone()) for (row, _), v in zip(subset, values))
    selected = {}
    for arm in ARMS:
        reset(actor, warm)
        opt = optimizer(actor)
        best, winner, selected_weights = rank(base, 0), 0, warm
        validations = {0: base}
        previous = 0
        for step in CHECKPOINTS:
            optimize(actor, tokens, order[previous*BATCH:step*BATCH], opt, method=arm, references=references,
                     filename=output / f"{arm}-train-{step}.jsonl")
            val = evaluate(actor, tokens, validation, output / f"{arm}-validation-{step}.jsonl")
            validations[step] = val
            print(arm, step, val["all"], flush=True)
            if rank(val, step) > best:
                best, winner, selected_weights = rank(val, step), step, state(actor)
            previous = step
        selected[arm] = selected_weights
        reset(actor, selected_weights)
        checkpoint = output / arm
        actor.save_checkpoint(checkpoint)
        selections[arm] = {"step": winner, "validation": validations,
                           "checkpoint_manifest_sha256": file_hash(checkpoint / "local_checkpoint.json"),
                           "adapter_sha256": file_hash(checkpoint / "adapter_model.safetensors")}
        del opt
    sealed = output / "SELECTION.json"
    write_json(sealed, {"plan_sha256": file_hash(output / "PLAN.json"), "split": "validation", "methods": selections})
    evaluation = pack(actor, bank["evaluation"])
    panels = {}
    rows = {}
    for method, weights in ((BASELINE, parent_weights), (COMMON, warm), *selected.items()):
        reset(actor, weights)
        trace = output / f"{method}-evaluation.jsonl"
        panels[method] = {"summary": evaluate(actor, tokens, evaluation, trace),
                          "traces_sha256": file_hash(trace)}
        rows[method] = [json.loads(line) for line in trace.read_text().splitlines()]
    paired = [{"root_id": row["id"], "seed": SEED, "method": method, "scenario_family": row["family"],
               "success": row["success"]} for method in ARMS for row in rows[method]]
    result = {"selection_sha256": file_hash(sealed), "evaluation_cases": len(evaluation), "panels": panels,
              "grpo_minus_sft": paired_bootstrap(paired, "grpo", "sft", "success", resamples=2000, confidence=.95)}
    write_json(output / "RESULTS.json", result)
    print(canonical({key: value["summary"]["all"] for key, value in panels.items()}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    run(args.output, args.device)


if __name__ == "__main__":
    main()
