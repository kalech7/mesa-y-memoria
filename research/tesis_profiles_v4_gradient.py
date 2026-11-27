"""Recover initial action-gradient evidence from the archived v4 TRAIN groups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.src.artifacts import file_hash
from research.tesis_extended_analysis import branch_data
from research.tesis_extended_experiment import MODEL, PROJECT, open_model
from research.tesis_profiles_analysis import actor_trees
from research.tesis_profiles_bank import build_development
from research.tesis_profiles_env import public_features, rule_action
from research.tesis_profiles_learning import RL, load_policies
from research.tesis_profiles_optimizer_v5 import initial_diagnostics
from research.tesis_profiles_study import trees_for
from research.tesis_shared_budget_qwen import atomic_json
from research.tesis_shared_budget_qwen_matched import PARENT, prepare, reference_logprobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = PROJECT / "reports/tesis_profiles_qwen_a100_v4"
    plan = json.loads((archive / "PLAN.json").read_text())
    selection = json.loads((archive / "SELECTION_LOCK.json").read_text())
    checkpoint = selection["adapters"]["101"][PARENT]
    parent = archive / checkpoint["adapter"]
    if file_hash(parent / "adapter_model.safetensors") != checkpoint["weights_sha256"]:
        raise ValueError("Archived common parent changed")
    forest = trees_for(build_development()["train"], 16)
    policies = load_policies(archive / "TABULAR.json", public_features=public_features, fallback=rule_action)
    model, tokenizer, labels = open_model(MODEL, parent, trainable=False)
    records, ledger = branch_data(forest, actor_trees(forest), policies[RL], tokenizer, plan["max_tokens"])
    if ledger != json.loads((archive / "QWEN_BRANCHES.json").read_text()):
        raise ValueError("Original training groups and returns did not reproduce")
    examples = prepare(records)
    reference, counts = reference_logprobs(model, [e.record for e in examples], labels, tokenizer.pad_token_id,
                                             "cuda", plan["batch_size"])
    atomic_json(args.output, {"v4_train_groups": ledger, "v4_parent_sha256": checkpoint["weights_sha256"],
                              "reference_counts": counts, "diagnostic": initial_diagnostics(examples, records, reference)})


if __name__ == "__main__":
    main()
