"""Check the published pilot locally, optionally replay its six portable adapters on CUDA."""
import argparse
from pathlib import Path
from statistics import mean

from research.src.artifacts import digest, file_hash
from research.tesis_timing_v6_eval import group_worlds, paired_difference, score_public
from research.tesis_timing_v6_experiment import PROJECT, SEEDS, signature, forest
from research.tesis_timing_v6_train import MarginScorer
from research.tesis_timing_v7 import ARMS, RESERVE_SHA, measure, read
from research.tesis_timing_v7_bank import reserve_worlds


def verify(run, replay=False):
    plan, lock, result = (read(run / name) for name in ("PLAN.json", "LOCK.json", "RESERVED_RESULTS.json"))

    def require(condition, detail):
        if not condition:
            raise ValueError(detail)

    require(lock["plan_sha256"] == file_hash(run / "PLAN.json"), "Plan digest mismatch")
    require(result["lock_sha256"] == file_hash(run / "LOCK.json"), "Lock digest mismatch")
    require(signature(PROJECT, plan["source_files"]) == plan["source_files"], "Experiment sources changed")
    worlds = reserve_worlds()
    require(digest(worlds) == plan["reserve_worlds"] == result["reserve_worlds_sha256"] == RESERVE_SHA,
            "Reserve digest mismatch")
    portable = read(run / "ADAPTADORES.json")
    methods = result["methods"]
    roots = group_worlds(forest(worlds, 1)) if replay else None
    scorer = MarginScorer() if replay else None
    for seed in map(str, SEEDS):
        fits = {}
        for arm in ARMS:
            selected = lock["selected"][seed][arm]
            relative = f"adaptadores/seed-{seed}/{arm}"
            files = portable[relative]
            require(files == signature(run / relative, files), f"Invalid portable files: {relative}")
            require(files["adapter_model.safetensors"] == selected["weights"]["adapter_model.safetensors"],
                    f"Portable adapter differs from trained weights: {relative}")
            fit = read(run / f"seed-{seed}" / arm / "FIT.json")
            require(all(fit[k] == v for k, v in selected.items()), f"Selection and ledger disagree: {relative}")
            fits[arm] = fit
            row = methods[arm][seed]
            require(len(row["packets"]) == 8 and all(abs(p["metrics"]["questions"] - .5) < 1e-9
                    for p in row["packets"].values()), f"Unequal spending: {relative}")
            if replay:
                scorer.load(run / relative)
                require(measure(roots, score_public(roots, scorer), selected["cutoff"]) == row,
                        f"GPU replay differs from sealed outputs: {relative}")
        require(fits[ARMS[0]]["budget"] == fits[ARMS[1]]["budget"], "Unequal training budgets")
    for name, values in result["primary_free_pairs"].items():
        rates = [methods[name][str(s)]["individual"]["both_timings_correct_native"] for s in SEEDS]
        require(values["per_seed"] == rates and values["mean"] == mean(rates), "Primary summary mismatch")
    left, right = methods["sft-pares"], methods["sft"]
    diff = mean(left[str(s)]["individual"]["both_timings_correct_native"] for s in SEEDS) - mean(
        right[str(s)]["individual"]["both_timings_correct_native"] for s in SEEDS)
    require(result["primary_difference"] == diff, "Primary paired difference mismatch")
    require(result["matched_future_difference"] == paired_difference(left, right),
            "Bootstrap differs from packet-level results")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, default=PROJECT / "reports/tesis_timing_pares_a100_v7")
    cli.add_argument("--replay", action="store_true", help="Recompute all six v7 policies with the local Qwen base on CUDA")
    args = cli.parse_args()
    verify(args.run, args.replay)
    print("Verificación correcta" + (" con replay CUDA de seis adaptadores." if args.replay else "."))
