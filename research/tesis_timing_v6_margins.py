"""Post hoc sensing of the agenda cue, with frozen models and no refitting."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from research.src.artifacts import digest, file_hash
from research.tesis_shared_budget_qwen import atomic_json
from research.tesis_timing_v6_bank import build_bank
from research.tesis_timing_v6_env import replay
from research.tesis_timing_v6_experiment import ARMS, check_sources, read, signature


def compare_pairs(rows):
    """Use language-model scores produced *before* attaching the variant label."""
    paired = defaultdict(dict)
    for pair, variant, margin in rows:
        if variant in paired[pair]:
            raise ValueError("Duplicate agenda in timing pair")
        paired[pair][variant] = margin
    if not paired or any(set(v) != {"ask_now", "save"} for v in paired.values()):
        raise ValueError("Need both agendas of each pair")
    deltas = [v["ask_now"] - v["save"] for v in paired.values()]
    return {
        "pairs": len(deltas),
        "higher_when_ask_now": sum(d > 0 for d in deltas),
        "higher_when_should_save": sum(d < 0 for d in deltas),
        "tied": sum(d == 0 for d in deltas),
        "mean_paired_logit_delta": mean(deltas),
        "median_paired_logit_delta": median(deltas),
        "crosses_zero_correctly": sum(v["ask_now"] > 0 >= v["save"] for v in paired.values()),
    }


def main():
    from research.tesis_timing_v6_train import MarginScorer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    run = parser.parse_args().run.resolve()
    cfg = check_sources(run)
    lock = read(run / "LOCK.json")
    primary = read(run / "RESERVED_RESULTS.json")
    if (
        primary["lock_sha256"] != file_hash(run / "LOCK.json")
        or signature(run, lock["sha256"]) != lock["sha256"]
    ):
        raise ValueError("Primary evaluation or selection differs")
    study = {}
    public = {}
    for split in ("train", "validation", "reserved"):
        roots = build_bank(split)
        worlds = [w for root in roots for w in root["worlds"]]
        if digest(worlds) != cfg["sealed_generator_world_hashes"][split]:
            raise ValueError("Bank seal changed")
        public[split] = (
            [replay(root["worlds"][0], (), budget=1).observation for root in roots],
            [(root["pair_id"], root["variant"]) for root in roots],
        )
    scorer = MarginScorer()
    for seed, selected in lock["selected"].items():
        study[seed] = {}
        for arm in ARMS:
            item = selected[arm]
            path = run / item["adapter"]
            if signature(path, item["sha256"]) != item["sha256"]:
                raise ValueError("Adapter changed")
            scorer.load(path)
            split_rows = {}
            for split, (observations, labels) in public.items():
                margins = scorer.margins(observations)
                split_rows[split] = compare_pairs(
                    [(p, v, m) for (p, v), m in zip(labels, margins, strict=True)]
                )
            study[seed][arm] = split_rows
    atomic_json(
        run / "PAIR_MARGIN_DIAGNOSTIC.json",
        {
            "stage": "post hoc descriptive after primary evaluation; no fit or checkpoint selection",
            "lock_sha256": primary["lock_sha256"],
            "primary_sha256": file_hash(run / "RESERVED_RESULTS.json"),
            "measure": "difference between ASK-minus-WAIT logits from the two visit-one prompts",
            "results": study,
        },
    )


if __name__ == "__main__":
    main()
