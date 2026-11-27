"""Post-hoc paired bootstrap for the v7 primary endpoint using its frozen decisions."""
import argparse
from collections import Counter, defaultdict
from math import isclose
from pathlib import Path
from statistics import mean

from research.src.artifacts import digest, file_hash
from research.tesis_shared_budget_qwen import atomic_json
from research.tesis_timing_v6_bank import WEEKDAYS, _agenda_words
from research.tesis_timing_v6_eval import group_worlds, independent_timing, paired_difference, score_public
from research.tesis_timing_v6_experiment import PROJECT, SEEDS, forest, signature
from research.tesis_timing_v6_train import MarginScorer
from research.tesis_timing_v7 import ARMS, read
from research.tesis_timing_v7_bank import WORDS, reserve_worlds
from research.tesis_timing_v7_verify import verify


def corpus_index(worlds):
    cues = {}
    for i, template in enumerate(WORDS["cue"], 1):
        for other in ("breakfast_preference", "drink_preference"):
            for most, least in (("dinner_preference", other), (other, "dinner_preference")):
                for day in WEEKDAYS:
                    cues[template.format(**_agenda_words(most, least), single_weekday=day)] = i
    rows = {}
    for w in worlds:
        key = w["pair_id"]
        tag = {"packet": w["packet_id"], "template": cues[w["cue"]]}
        if rows.setdefault(key, tag) != tag:
            raise ValueError("One pair spans inconsistent templates or packets")
    if sorted(Counter(r["packet"] for r in rows.values()).values()) != [10] * 8:
        raise ValueError("Expected ten pairs in each of eight packets")
    if sorted(Counter(r["template"] for r in rows.values()).values()) != [10] * 8:
        raise ValueError("Expected ten pairs per agenda wording")
    return rows


def replay(run, worlds):
    grouped = defaultdict(list)
    roots = group_worlds(forest(worlds, 1))
    for r in roots:
        grouped[r.pair].append(r)
    rows = corpus_index(worlds)
    scorer = MarginScorer()
    for arm in ARMS:
        for seed in SEEDS:
            scorer.load(run / "adaptadores" / f"seed-{seed}" / arm)
            margins = score_public(roots, scorer)
            for pair, members in grouped.items():
                # Use the experiment's original greedy rollout, including WAIT twice.
                diag = independent_timing(members, margins)
                if len(members) != 2 or diag["evaluable_flipped_pairs"] != 1:
                    raise ValueError(f"Non-evaluable pair {pair}")
                rows[pair].setdefault(arm, []).append(int(diag["both_timings_correct_native"]))
    return rows


def summarize(run, worlds, rows):
    reference = read(run / "RESERVED_RESULTS.json")
    expected = corpus_index(worlds)
    if rows.keys() != expected.keys():
        raise ValueError("Unexpected reserve pairs")
    arms = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for index, seed in enumerate(SEEDS):
            by_packet = defaultdict(list)
            for pair, row in rows.items():
                if any(row[k] != v for k, v in expected[pair].items()):
                    raise ValueError("Pair labels do not match reserve")
                if len(row[arm]) != len(SEEDS) or any(type(x) is not int or x not in (0, 1) for x in row[arm]):
                    raise ValueError("Invalid primary indicator")
                by_packet[row["packet"]].append(row[arm][index])
            fraction = mean(mean(flags) for flags in by_packet.values())
            if not isclose(fraction, reference["primary_free_pairs"][arm]["per_seed"][index], abs_tol=1e-12):
                raise ValueError("Replay counts differ from the sealed primary outcome")
            arms[arm][str(seed)] = {"packets": {p: {"metrics": {"pairs": mean(flags)}}
                                                for p, flags in by_packet.items()}}
    point = paired_difference(arms["sft-pares"], arms["sft"], metric="pairs")
    if not isclose(point["estimate"], reference["primary_difference"], abs_tol=1e-12):
        raise ValueError("Primary contrast differs from the sealed result")
    sensitivity = {}
    for omitted in range(1, 9):
        sensitivity[str(omitted)] = mean(
            b - a for r in rows.values() if r["template"] != omitted
            for a, b in zip(r["sft"], r["sft-pares"], strict=True))
    return {
        "analysis": "post_hoc_paired_packet_percentile_bootstrap_for_frozen_primary",
        "seeds_in_column_order": list(SEEDS), "bootstrap_seed": 60421,
        "source_sha256": file_hash(Path(__file__)),
        "inputs_sha256": signature(run, ["PLAN.json", "LOCK.json", "RESERVED_RESULTS.json", "ADAPTADORES.json"]),
        "reserve_worlds_sha256": digest(worlds), "indicators_sha256": digest(rows),
        "contrast_sft_pares_minus_sft": point, "difference_omitting_template": sensitivity,
        "indicators_by_pair": dict(sorted(rows.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=PROJECT / "reports/tesis_timing_pares_a100_v7")
    parser.add_argument("--replay", action="store_true", help="Regenerate each binary decision on CUDA")
    args = parser.parse_args()
    run = args.run
    verify(run)
    path = run / "PRIMARY_PAIR_DIAGNOSTIC.json"
    worlds = reserve_worlds()
    if args.replay:
        rows = replay(run, worlds)
    else:
        rows = read(path)["indicators_by_pair"]
    report = summarize(run, worlds, rows)
    if path.exists():
        if read(path) != report:
            raise ValueError("Diagnostic differs from the published artifact")
    elif args.replay:
        atomic_json(path, report)
    print(report["contrast_sft_pares_minus_sft"])


if __name__ == "__main__":
    main()
