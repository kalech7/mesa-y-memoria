"""Evaluate frozen menu adapters on three arbitrary orders and on blinded text."""
from __future__ import annotations

import argparse
from pathlib import Path
from random import Random
from statistics import mean

from research.src.artifacts import file_hash, write_json
from research.tesis_meal_cases import prompt as previous_prompt
from research.tesis_meal_varied_bank import prompt


def variants(roots):
    rng = Random(81774)
    for root in roots:
        for index in range(3):
            order = rng.sample(range(6), 6)
            while any(order == list(range(6))[i:] + list(range(6))[:i] for i in range(6)):
                order = rng.sample(range(6), 6)
            case = {**root, "options": [root["options"][i] for i in order],
                    "accepted": [order.index(i) for i in root["accepted"]]}
            yield case, index, order


def check(run, device):
    import torch

    from research.tesis_meal_audit import read, read_rows
    from research.tesis_meal_train import action_tokens, load, logits, scores, write_rows

    result = run / "RESULTS.json"
    selection = read(run / "SELECTION.json")
    if read(result)["selection_sha256"] != file_hash(run / "SELECTION.json"):
        raise ValueError("Selection changed")
    evaluation = read_rows(run / "evaluation.jsonl")
    variants_list = list(variants(evaluation))
    report = {}
    rows = []
    previous = run.parent / "tesis_meal_lora_v1"
    previous_selection = read(previous / "SELECTION.json")
    adapters = [(method, run, selection, prompt) for method in ("sft", "grpo")]
    adapters.extend((method + "-anterior", previous, previous_selection, previous_prompt) for method in ("sft", "grpo"))
    for name, directory, manifest, render in adapters:
        method = name.removesuffix("-anterior")
        folder = directory / method
        if manifest["methods"][method]["adapter_sha256"] != file_hash(folder / "adapter_model.safetensors"):
            raise ValueError("Selected adapter changed")
        actor = load(folder, device)
        tokens = action_tokens(actor.tokenizer)
        observed = []
        with torch.inference_mode():
            for start in range(0, len(variants_list), 4):
                group = variants_list[start:start+4]
                full = [actor.encode_prompt(render(row["turns"], row["options"])) for row, _, _ in group]
                blind = [actor.encode_prompt(render(["Elige una opción."], row["options"])) for row, _, _ in group]
                raw = logits(actor, full)[:, tokens].argmax(-1).tolist()
                control = logits(actor, blind)[:, tokens].argmax(-1).tolist()
                for (row, index, order), i, j in zip(group, raw, control, strict=True):
                    observed.append({"id": row["id"], "family": row["family"], "method": name, "permutation": index,
                                     "order": order, "answer": i, "blinded_answer": j,
                                     **scores(row, i), "blinded_success": scores(row, j)["success"]})
        report[name] = {key: mean(row[key] for row in observed)
                          for key in ("success", "conflict", "blinded_success")}
        del actor
        rows.extend(observed)
        print(name, report[name], flush=True)
    trace = run / "SHUFFLE_TRACES.jsonl"
    write_rows(trace, rows)
    write_json(run / "SHUFFLE.json", {"results_sha256": file_hash(result), "traces_sha256": file_hash(trace),
                                      "previous_selection_sha256": file_hash(previous / "SELECTION.json"),
                                      "permutations_per_case": 3, "metrics": report})
    print(report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    check(args.run, args.device)
