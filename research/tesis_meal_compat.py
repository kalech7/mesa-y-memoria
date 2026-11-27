"""Qwen LoRA learns ingredient compatibility directly, independent of menu order."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from statistics import mean

import torch

from research.src.analysis.paired_statistics import paired_bootstrap
from research.src.artifacts import digest, file_hash, write_json
from research.tesis_meal_cases import LETTERS, FAMILIES, build, wording
from research.tesis_meal_train import (BASELINE, BETA, COMMON, DRAWS, SEED, action_tokens, grpo_loss, load, logits,
                                     optimizer, reset, rewards, scores, state, write_rows)
from web_demo.cases import MODELS


STEPS = 144
WARMUP = 144
CHECKPOINTS = (48, 96, STEPS)
METHODS = ("sft", "grpo")


def candidate_prompt(turns, option):
    dialogue = "\n".join("Usuario: " + text for text in turns)
    ingredients = ", ".join(option["ingredients"])
    return ("Determina si la opción satisface la comida solicitada. Distingue lo que ya comió, "
            "lo que quiere evitar y las correcciones. A = opción compatible; B = opción incompatible. "
            "Si falta una restricción concreta, no afirmes que un plato cumple. Responde con A o B.\n"
            + dialogue + f"\nOpción: {option['name']}. Ingredientes: {ingredients}.\nLetra:")


def pack(actor, row):
    return row, [actor.encode_prompt(candidate_prompt(row["turns"], option)) if option["ingredients"]
                 else None for option in row["options"]]


def decision(actor, ids, tokens):
    """Each dish receives its own binary Qwen logit. A neutral score represents clarification."""
    chunks = [item for item in ids if item is not None]
    z = logits(actor, chunks)[:, tokens[:2]]
    values = iter(z[:, 0] - z[:, 1])
    return torch.stack([next(values) if item is not None else z.new_zeros(()) for item in ids])


def supervised(actor, ids, tokens, row, rng, step):
    # Sample one valid and one incompatible dish when both exist; every restaurant
    # menu and every dialogue is shared across SFT and GRPO.
    positive = [i for i, item in enumerate(ids) if item is not None and i in row["accepted"]]
    negative = [i for i, item in enumerate(ids) if item is not None and i not in row["accepted"]]
    sample = [(rng.choice(group), tag) for tag, group in enumerate((positive, negative)) if group]
    if len(sample) == 1:
        group = positive or negative
        sample.append((rng.choice(group), sample[0][1]))
    array = logits(actor, [ids[i] for i, _ in sample])
    if step <= WARMUP:
        array = array[:, tokens[:2]]
        target = torch.tensor([tag for _, tag in sample], device=actor.device)
    else:
        target = torch.tensor([tokens[tag] for _, tag in sample], device=actor.device)
    return torch.nn.functional.cross_entropy(array, target)


def evaluate(actor, tokens, bank, path):
    output = []
    with torch.inference_mode():
        for row, ids in bank:
            choice = int(decision(actor, ids, tokens).argmax())
            output.append({"id": row["id"], "family": row["family"], "dish": row["options"][choice]["name"],
                           "action": LETTERS[choice], **scores(row, choice)})
    write_rows(path, output)
    columns = ("success", "conflict", "unnecessary_clarification", "missed_clarification")
    return {key: mean(row[key] for row in output) for key in columns}


def update(actor, tokens, schedule, opt, method, path, seed, references=None):
    randomizer = random.Random(seed)
    generator = torch.Generator(device=actor.device).manual_seed(seed)
    log = []
    for step, (row, ids) in enumerate(schedule, 1):
        if row["split"] != "train":
            raise ValueError("Only TRAIN cases may reach the optimizer")
        opt.zero_grad(set_to_none=True)
        telemetry = {}
        if method == "grpo":
            array = decision(actor, ids, tokens).unsqueeze(0)
            loss, telemetry = grpo_loss(array, list(range(len(ids))), [row], references[row["id"]], generator)
        else:
            loss = supervised(actor, ids, tokens, row, randomizer, step if method == "warmup" else WARMUP + 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in actor.model.parameters() if p.requires_grad], 1.)
        opt.step()
        log.append({"step": step, "case": row["id"], "loss": float(loss.detach()), **telemetry})
    write_rows(path, log)


def cases():
    bank = build(seed=51893, sizes={"train": 144, "validation": 32, "evaluation": 48})
    # This is a prospective check with a separate held-out dialogue vocabulary.
    evaluation = bank["evaluation"]
    alternate_intro = ("Para el almuerzo, ¿cuál de las opciones me iría mejor?", "De esa lista, ¿qué menú me conviene?",
                       "Ayúdame a escoger algo para comer.")
    pattern = {
        "rechazo": "Hoy no voy a probar {food}.",
        "errata": "Lo que no me apetece para nada es {food}.",
        "multiple": "Quisiera una opción que excluya {food} y {other}.",
        "variedad": "En la comida pasada ya tomé {food}. Busco una opción diferente a esa.",
        "rectificacion": "Me corrijo: el {food} está permitido. Lo que quiero descartar es {other}.",
        "ambiguo": "Quiero dejar fuera el que te dije, pero olvidé poner su nombre.",
        "todas": "Hoy me quedé sin {food}.",
        "historial": "La semana pasada comí {food}.",
    }
    from research.tesis_meal_cases import WORD
    for i, row in enumerate(evaluation):
        family = row["family"]
        if family == "rectificacion":
            previous = row["turns"][0]
            restored = next(key for key, value in WORD.items() if value in previous and key not in row["forbidden"])
            food, other = WORD[restored], WORD[row["forbidden"][0]]
            turns = [f"En un principio descarté {food} y {other}."]
        else:
            food = ", ".join(WORD.get(key, key) for key in row["forbidden"]) if family == "todas" else (
                WORD.get(row["forbidden"][0], row["forbidden"][0]) if row["forbidden"] else tuple(WORD.values())[i % len(WORD)])
            other = WORD[row["forbidden"][1]] if family == "multiple" else ""
            turns = []
        if family == "errata":
            food = {"pollo":"poollo", "huevo":"güebo", "queso":"kesito", "pescado":"peskao",
                    "garbanzos":"garvansos"}[food]
        turns.append(pattern[family].format(food=food, other=other) + " " + alternate_intro[i % 3])
        row["turns"] = turns
        row["id"] = digest({"turns": turns, "options": row["options"]})
    return bank


def run(output, device):
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    output.mkdir(parents=True)
    bank = cases()
    for split, rows in bank.items():
        write_rows(output / f"{split}.jsonl", rows)
    parent = MODELS["sft"].path
    plan = {"seed": SEED, "device": device, "warmup": WARMUP, "updates_per_arm": STEPS,
            "draws": DRAWS, "checkpoints": CHECKPOINTS, "parent": str(parent),
            "parent_sha256": file_hash(parent / "adapter_model.safetensors"),
            "source_sha256": file_hash(Path(__file__)),
            "bank_sha256": {split: file_hash(output / f"{split}.jsonl") for split in bank},
            "form": "Qwen assigns each dish a compatibility score; the menu order is absent from candidate prompts"}
    write_json(output / "PLAN.json", plan)
    actor = load(parent, device)
    tokens = action_tokens(actor.tokenizer)
    train, val = ([pack(actor, row) for row in bank[part]] for part in ("train", "validation"))
    start = state(actor)
    init = evaluate(actor, tokens, val, output / "inicial-validation.jsonl")
    rng = random.Random(SEED)
    schedule = rng.sample(train, len(train))
    update(actor, tokens, schedule, optimizer(actor), "warmup", output / "warmup.jsonl", SEED)
    common = state(actor)
    base = evaluate(actor, tokens, val, output / "sft-comun-validation.jsonl")
    print("validación inicial y ajuste común", init, base, flush=True)
    with torch.inference_mode():
        references = {row["id"]: (decision(actor, ids, tokens) / 1.5).log_softmax(-1).unsqueeze(0)
                      for row, ids in train}
    choices = {BASELINE: start, COMMON: common}
    selection = {}
    for method in METHODS:
        reset(actor, common)
        opt = optimizer(actor)
        selected = common
        best = (base["success"], -base["conflict"], 0)
        previous = 0
        validation = {0: base}
        for step in CHECKPOINTS:
            update(actor, tokens, schedule[previous:step], opt, method,
                   output / f"{method}-train-{step}.jsonl", SEED + step, references)
            metric = evaluate(actor, tokens, val, output / f"{method}-validation-{step}.jsonl")
            print(method, step, metric, flush=True)
            rank = (metric["success"], -metric["conflict"], -step)
            if rank > best:
                best, selected = rank, state(actor)
            validation[step] = metric
            previous = step
        reset(actor, selected)
        actor.save_checkpoint(output / method)
        choices[method] = selected
        selection[method] = {"step": -best[-1], "validation": validation,
                             "checkpoint_manifest_sha256": file_hash(output / method / "local_checkpoint.json"),
                             "adapter_sha256": file_hash(output / method / "adapter_model.safetensors")}
    write_json(output / "SELECTION.json", {"plan_sha256": file_hash(output / "PLAN.json"), "split": "validation",
                                           "methods": selection})
    reserved = [pack(actor, row) for row in bank["evaluation"]]
    panels = {}
    paired = []
    for method, weights in choices.items():
        reset(actor, weights)
        path = output / f"{method}-evaluation.jsonl"
        metrics = evaluate(actor, tokens, reserved, path)
        panels[method] = {"metrics": metrics, "traces_sha256": file_hash(path)}
        for raw in path.read_text().splitlines():
            row = json.loads(raw)
            source = next(case for case, _ in reserved if case["id"] == row["id"])
            paired.append({**row, "root_id": digest(source["turns"]), "realization_id": row["id"],
                           "seed": SEED, "scenario_family": row["family"], "method": method})
    contrast = paired_bootstrap(paired, "grpo", "sft", "success", confidence=.95, resamples=2000)
    result = {"selection_sha256": file_hash(output / "SELECTION.json"), "cases": len(reserved),
              "panels": panels, "grpo_minus_sft": contrast}
    write_json(output / "RESULTS.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run(args.output, args.device)
