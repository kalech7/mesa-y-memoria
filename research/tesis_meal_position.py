"""Check the selected Qwen adapters under every cyclic ordering of each menu."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean

import torch

from research.src.artifacts import file_hash, write_json
from research.tesis_meal_audit import read, read_rows
from research.tesis_meal_train import action_tokens, encode, load, logits, write_rows


def evaluate(run, device):
    torch.set_num_threads(4)
    selection = read(run / "SELECTION.json")
    result = read(run / "RESULTS.json")
    if result["selection_sha256"] != file_hash(run / "SELECTION.json"):
        raise ValueError("The selected adapters changed")
    cases = read_rows(run / "evaluation.jsonl")
    rows = []
    report = {}
    for method in ("sft", "grpo"):
        checkpoint = run / method
        if file_hash(checkpoint / "adapter_model.safetensors") != selection["methods"][method]["adapter_sha256"]:
            raise ValueError("The adapter changed")
        policy = load(checkpoint, device)
        ids = action_tokens(policy.tokenizer)
        hits = Counter()
        same_option = Counter()
        first = []
        for case in cases:
            options = case["options"]
            indices = list(range(len(options)))
            orderings = [indices[i:] + indices[:i] for i in indices]
            prompts = [encode(policy, {**case, "options": [options[i] for i in order]}) for order in orderings]
            with torch.inference_mode():
                choices = []
                for i in range(0, len(prompts), 2):
                    choices.extend(logits(policy, prompts[i:i+2])[:, ids].argmax(dim=1).tolist())
            reconstructed = [order[choice] for order, choice in zip(orderings, choices, strict=True)]
            first.append(int(reconstructed[0] in case["accepted"]))
            for original in reconstructed:
                hits[case["id"]] += original in case["accepted"]
                same_option[case["id"]] += original == reconstructed[0]
            rows.append({"method": method, "id": case["id"], "selected_original_indices": reconstructed,
                         "accepted": case["accepted"]})
        report[method] = {"accuracy": mean(value / 6 for value in hits.values()),
                          "consistent_choice": mean(value == 6 for value in same_option.values()),
                          "all_orders_compatible": mean(value == 6 for value in hits.values()),
                          "original_order": mean(first)}
        del policy
    path = run / "POSITION_TRACES.jsonl"
    write_rows(path, rows)
    write_json(run / "POSITION.json", {"results_sha256": file_hash(run / "RESULTS.json"),
                                       "traces_sha256": file_hash(path), "orders_per_case": 6, "metrics": report})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    print(evaluate(args.run, args.device))
