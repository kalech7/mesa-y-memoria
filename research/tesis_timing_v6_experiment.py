"""Preregister, select under a matched budget, lock, and score the timing experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from research.src.artifacts import digest, file_hash, now
from research.tesis_shared_budget_qwen import atomic_json
from research.tesis_timing_v6_eval import (
    METRICS,
    PACKET_SIZE,
    QUESTIONS,
    aggregate,
    evaluate_pool,
    group_worlds,
    independent_timing,
    paired_difference,
    score_public,
    thresholds,
)

PROJECT = Path(__file__).resolve().parents[1]
SEEDS = (101, 202, 303)
ROUNDS = 2
GUARDS = {"current_loss": 0.02, "extra_unsupported_reuse": 0.0}
ARMS = ("SFT-ramas", "GRPO-continuacion", "GRPO-inmediato", "GRPO-continuacion-fija")
PLANNER = "planificador-publico"
RULE = "preguntar-en-cuanto-se-pueda"
V6_SOURCES = [
    "research/tesis_timing_v6_" + name + ".py" for name in ("bank", "env", "eval", "experiment", "train")
]
BASE_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)
V6_SOURCES += [
    "research/" + f
    for f in (
        "tesis_shared_budget_planner.py",
        "tesis_shared_budget_qwen.py",
        "tesis_shared_budget_qwen_matched.py",
        "tesis_extended_experiment.py",
        "tesis_profiles_optimizer_v5.py",
        "src/artifacts.py",
        "src/data/build_bank.py",
        "src/memory/store.py",
        "src/memory/executor.py",
        "src/readers/symbolic.py",
    )
]
WEIGHTS = ("adapter_model.safetensors", "adapter_config.json")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def signature(folder, names):
    return {n: file_hash(folder / n) for n in names}


def trees(split, budget):
    from research.tesis_timing_v6_bank import build_worlds

    return forest(build_worlds(split), budget)


def forest(worlds, budget):
    from research.tesis_timing_v6_env import enumerate_world

    return [enumerate_world(w, budget=budget) for w in worlds]


def plan(run):
    from research.tesis_timing_v6_bank import build_worlds
    from research.tesis_timing_v6_bank import FROZEN_WORLD_SHA256
    from research.tesis_timing_v6_train import PARENT

    base = PROJECT / "models/Qwen2.5-1.5B-Instruct"
    # The reserved generator, including the reserved wording, is fixed by hash;
    # neither creating this plan nor fitting a selector executes reserved roots.
    return {
        "schema": 1,
        "seeds": list(SEEDS),
        "rounds": ROUNDS,
        "models": list(ARMS),
        "source_sha256": signature(PROJECT, V6_SOURCES),
        "base": str(base.relative_to(PROJECT)),
        "base_sha256": signature(base, BASE_FILES),
        "parent": str(Path(PARENT).resolve().relative_to(PROJECT)),
        "parent_sha256": signature(Path(PARENT), WEIGHTS),
        "sealed_generator_world_hashes": FROZEN_WORLD_SHA256,
        "development_worlds_sha256": {s: digest(build_worlds(s)) for s in ("train", "validation")},
        "reserved_opening": "only after all validation checkpoints and calibration cutoffs locked",
        "primary": {
            "contrast": [ARMS[1], ARMS[0]],
            "metric": "future_supported",
            "questions": QUESTIONS,
            "packet_size": PACKET_SIZE,
            "per_root_maximum": 1,
            "guards": GUARDS,
            "success": "lower paired 95% packet bootstrap limit > 0, current and reuse guards satisfied",
            "repeats": "fixed three seeds, packets resampled in common, conditional on synthetic wording",
        },
        "selection": "validation future supported, current supported, lower unsupported reuse, earliest round",
        "threshold_grid": "always, zero, validation first-margin quartiles and p90, never; first in grid on tie",
        "allocation": "synchronous first arrivals, threshold + descending margins; unused quota to top waiting visit-2 margins",
        "diagnostics": [
            "budget 1/2 same roots on CPU",
            "forced 1 per story and unconstrained greedy",
            "immediate-reward GRPO",
            "longitudinal GRPO with fixed parent continuation",
            PLANNER,
            RULE,
        ],
        "cpu_gate": {
            "paired_flips_min": 0.75,
            "expected_root_gain_min": 0.04,
            "public_planner_gain_min": 0.03,
            "public_questions": 1,
        },
    }


def check_sources(run):
    cfg = read(run / "PLAN.json")
    if cfg["source_sha256"] != signature(PROJECT, V6_SOURCES):
        raise ValueError("Frozen v6 sources changed")
    if cfg["parent_sha256"] != signature(PROJECT / cfg["parent"], WEIGHTS):
        raise ValueError("Common parent changed")
    if cfg["base_sha256"] != signature(PROJECT / cfg["base"], BASE_FILES):
        raise ValueError("Qwen base changed")
    return cfg


def prepare(run):
    from research.tesis_timing_v6_env import cpu_audit

    run.mkdir(parents=True, exist_ok=True)
    cfg = plan(run)
    if (run / "PLAN.json").exists() and cfg != read(run / "PLAN.json"):
        raise ValueError("Different experiment already occupies this directory")
    atomic_json(run / "PLAN.json", cfg)
    if (run / "CPU.json").exists():
        if not read(run / "CPU.json")["passed"]:
            raise ValueError("CPU gate failed, see CPU.json")
        return
    checks = {}
    for budget in (1, 2):
        checks[str(budget)] = cpu_audit(trees("train", budget), trees("validation", budget))
    paired = checks["1"]["paired_timing"]
    controls = checks["1"]["policies"]
    best_fixed = max(controls[n]["future_supported"] for n in ("preguntar-primero", "preguntar-segundo"))
    gain = controls[PLANNER]["future_supported"] - best_fixed
    gate = (
        paired["robust_flip_rate"] >= 0.75
        and paired["mean_gain_over_best_constant"] > 0.04
        and gain > 0.03
        and abs(controls[PLANNER]["questions"] - 1.0) < 1e-8
    )
    atomic_json(
        run / "CPU.json", {"split": "validation", "budgets": checks, "public_gain": gain, "passed": gate}
    )
    if not gate:
        raise ValueError("No demonstrated margin for timing in development; see CPU.json")


def checkpoint(run, seed, arm, rnd):
    return run / f"seed-{seed}" / arm / f"checkpoint-{rnd:03d}" / "adapter"


def rank(evaluation):
    m = evaluation["macro"]
    return (
        round(m["future_supported"], 10),
        round(m["current_supported"], 10),
        -round(m["future_unsupported_reuse"], 10),
    )


def calibrate(roots, margins):
    candidates = []
    for cutoff in thresholds(margins):
        row = evaluate_pool(roots, margins, cutoff)
        candidates.append({"cutoff": cutoff, "macro": row["macro"]})
    return max(candidates, key=rank), candidates


class PublicScorer:
    def __init__(self, policy):
        self.policy = policy

    def margins(self, obs):
        return [self.policy.margin(o, ("WAIT", "ASK")) for o in obs]


def select(run):
    from research.tesis_timing_v6_train import MarginScorer
    from research.tesis_timing_v6_env import fit_public_planner

    check_sources(run)
    if (run / "LOCK.json").exists():
        return
    if not read(run / "CPU.json")["passed"] or not (run / "TRAIN_RESULT.json").is_file():
        raise ValueError("Missing completed training and passing CPU gate")
    roots = group_worlds(trees("validation", 1))
    scorer = MarginScorer()
    choice, history = {}, {}
    for seed in SEEDS:
        current = {}
        for arm in ARMS:
            outcomes = []
            for rnd in range(1, ROUNDS + 1):
                adapter = checkpoint(run, seed, arm, rnd)
                scorer.load(adapter)
                best, grid = calibrate(roots, score_public(roots, scorer))
                outcomes.append(
                    {**best, "round": rnd, "adapter": str(adapter.relative_to(run)), "grid": grid}
                )
            best = max(outcomes, key=rank)
            current[arm] = {k: v for k, v in best.items() if k != "grid"}
            current[arm]["sha256"] = signature(run / best["adapter"], WEIGHTS)
            history[f"{seed}/{arm}"] = outcomes
        choice[str(seed)] = current
    public = PublicScorer(fit_public_planner(trees("train", 1)))
    best, grid = calibrate(roots, score_public(roots, public))
    atomic_json(run / "VALIDATION.json", {"candidates": history, "public_candidates": grid})
    frozen_files = ["PLAN.json", "CPU.json", "VALIDATION.json", "TRAIN_RESULT.json", "TRAIN_PLAN.json"]
    atomic_json(
        run / "LOCK.json",
        {
            "selected": choice,
            "public_cutoff": best["cutoff"],
            "sha256": signature(run, frozen_files),
            "locked_utc": now(),
        },
    )


def score(run):
    from research.tesis_timing_v6_env import fit_public_planner
    from research.tesis_timing_v6_train import MarginScorer
    from research.tesis_timing_v6_bank import build_worlds
    import torch

    check_sources(run)
    lock = read(run / "LOCK.json")
    if lock["sha256"] != signature(run, lock["sha256"]):
        raise ValueError("A selected input changed")
    for items in lock["selected"].values():
        for item in items.values():
            if item["sha256"] != signature(run / item["adapter"], WEIGHTS):
                raise ValueError("A selected model changed")
    output = run / "RESERVED_RESULTS.json"
    if output.exists():
        if read(output)["lock_sha256"] != file_hash(run / "LOCK.json"):
            raise ValueError("Reserved results refer to a different run")
        return
    train_worlds = build_worlds("train")
    val_worlds = build_worlds("validation")
    reserved_worlds = build_worlds("reserved")
    if digest(reserved_worlds) != read(run / "PLAN.json")["sealed_generator_world_hashes"]["reserved"]:
        raise ValueError("Reserved world seal changed")
    for key in ("root_id", "pair_id", "packet_id"):
        spaces = [{w[key] for w in split} for split in (train_worlds, val_worlds, reserved_worlds)]
        if any(a & b for i, a in enumerate(spaces) for b in spaces[i + 1 :]):
            raise ValueError("Partition overlap in " + key)
    roots = group_worlds(forest(reserved_worlds, budget=1))
    scorer = MarginScorer()
    results = defaultdict(dict)
    diagnostics = defaultdict(dict)
    for seed, arms in lock["selected"].items():
        for arm, item in arms.items():
            scorer.load(run / item["adapter"])
            margins = score_public(roots, scorer)
            results[arm][seed] = evaluate_pool(roots, margins, item["cutoff"])
            diagnostics[arm][seed] = independent_timing(roots, margins)
    public = PublicScorer(fit_public_planner(forest(train_worlds, budget=1)))
    margins = score_public(roots, public)
    for method, data, cutoff in [
        (PLANNER, margins, lock["public_cutoff"]),
        (RULE, {r.identity: (1.0, 1.0) for r in roots}, "always"),
    ]:
        row = evaluate_pool(roots, data, cutoff)
        results[method] = {str(s): row for s in SEEDS}
        diag = independent_timing(roots, data)
        diagnostics[method] = {str(s): diag for s in SEEDS}
    comparisons = {}
    for other in (ARMS[0], *ARMS[2:], PLANNER, RULE):
        contrasts = {m: paired_difference(results[ARMS[1]], results[other], m) for m in METRICS}
        passes = (
            contrasts["future_supported"]["interval"][0] > 0
            and contrasts["current_supported"]["estimate"] >= -GUARDS["current_loss"] - 1e-9
            and contrasts["future_unsupported_reuse"]["estimate"]
            <= GUARDS["extra_unsupported_reuse"] + 1e-9
            and abs(contrasts["questions"]["estimate"]) < 1e-9
        )
        comparisons[other] = {"contrasts": contrasts, "passes": passes}
    summary = {m: aggregate([x["macro"] for x in by_seed.values()]) for m, by_seed in results.items()}
    atomic_json(
        output,
        {
            "device": torch.cuda.get_device_name(0),
            "lock_sha256": file_hash(run / "LOCK.json"),
            "reserved_worlds_sha256": digest(reserved_worlds),
            "worlds": len(reserved_worlds),
            "roots": len(roots),
            "pairs": len({r.pair for r in roots}),
            "packets": len({r.packet for r in roots}),
            "primary_contrast": ARMS[:2][::-1],
            "summary": summary,
            "comparisons": comparisons,
            "diagnostics": diagnostics,
            "pooled": results,
            "completed_utc": now(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "select", "score"])
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "select": select, "score": score}[args.stage](args.run.resolve())


if __name__ == "__main__":
    main()
