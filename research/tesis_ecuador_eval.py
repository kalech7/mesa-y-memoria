"""Freeze, evaluate on CUDA, and independently re-score the Ecuador transfer set."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from math import isclose, isfinite
from random import Random
from statistics import mean

from research.src.artifacts import file_hash
from research.tesis_ecuador_worlds import audit, build_worlds
from research.tesis_shared_budget_qwen import atomic_json
from research.tesis_timing_v6_env import fit_public_planner
from research.tesis_timing_v6_eval import group_worlds, independent_timing, score_public
from research.tesis_timing_v6_experiment import BASE_FILES, PROJECT, SEEDS, WEIGHTS, PublicScorer, forest, signature, trees
from research.tesis_timing_v6_train import MODEL, PARENT, MarginScorer
from research.tesis_timing_v7 import ARMS, measure, read
from research.tesis_timing_v7_verify import verify as verify_adapters

RUN = PROJECT / "reports/tesis_ecuador_30_v1"
V7 = PROJECT / "reports/tesis_timing_pares_a100_v7"
PLANNER = "planificador-publico"
INPUTS = ("PARES.json", "CLAVE.json", "REVISION_CIEGA.json", "JUICIOS.json")
SOURCES = ("research/tesis_ecuador_worlds.py", "research/tesis_ecuador_eval.py",
           "research/tesis_timing_v7_verify.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def review(directory):
    key = read(directory / "CLAVE.json")
    texts = {row["id"]: row for row in read(directory / "PARES.json")}
    blind = {row["caso"]: row for row in read(directory / "REVISION_CIEGA.json")}
    judgments = read(directory / "JUICIOS.json")
    require(len(key) == len(blind) == len(judgments) == len(texts) == 30, "Incomplete review")
    require(set(blind) == set(key) == {r["caso"] for r in judgments}, "Unmatched review IDs")
    require({r["id"] for r in key.values()} == set(texts), "Unmatched original IDs")
    correct = pairs = 0
    for row in judgments:
        answer = key[row["caso"]]
        source = texts[answer["id"]]
        item = blind[row["caso"]]
        require({answer["X"], answer["Y"]} == {1, 2}, "Invalid orientation key")
        require(item["pedido_primera_visita"] == source["opening"] and
                item["pedido_segunda_visita"] == source["second"], "Blind review saw different orders")
        for side in ("X", "Y"):
            cue = source["agenda_first" if answer[side] == 1 else "agenda_second"]
            require(item[side] == cue, "Blind review saw a different agenda")
        hits = sum(answer[side] == row[side] for side in ("X", "Y"))
        correct += hits
        pairs += hits == 2
    return {"matching_orientations": correct, "matching_pairs": pairs}


def seal():
    verify_adapters(V7)
    audits = audit(read(RUN / "PARES.json"))
    rounds = ("ronda1", "ronda2")
    archived = [f"{directory}/{name}" for directory in rounds for name in INPUTS]
    return {"inputs": signature(RUN, (*INPUTS, "PROTOCOLO.md", "CONCILIACION_REVISION.json", *archived)),
            "sources": signature(PROJECT, SOURCES), "v7": signature(V7, ("PLAN.json", "LOCK.json", "ADAPTADORES.json")),
            "earlier_reviews": {directory: review(RUN / directory) for directory in rounds},
            "review": review(RUN), "audit": audits}


def stored_or_new(name, value):
    path = RUN / name
    if path.exists():
        require(read(path) == value, f"{name} differs from the archived version")
    else:
        atomic_json(path, value)


def run_cuda(roots, *, replay=False):
    import torch

    path = RUN / "MARGENES.json"
    require(replay or not path.exists(), "GPU margins already exist")
    device = torch.cuda.get_device_name()
    require("A100" in device, f"Expected an A100, found {device}")
    plan = read(V7 / "PLAN.json")
    parent = signature(PARENT, WEIGHTS)
    expected = dict(plan["parent"])
    if parent["adapter_config.json"] != expected["adapter_config.json"]:
        provenance = read(PROJECT / "reports/tesis_timing_qwen_a100_v6/PROCEDENCIA.json")
        expected["adapter_config.json"] = provenance["configuraciones_sha256"][expected["adapter_config.json"]]
    require(signature(MODEL, BASE_FILES) == plan["qwen"] and parent == expected,
            "Qwen base or parent differs from the original experiment")
    scorer = MarginScorer()
    methods = {arm: {} for arm in ARMS}
    started = datetime.now(timezone.utc)
    for seed in map(str, SEEDS):
        for arm in ARMS:
            print(f"Scoring {arm} {seed}", flush=True)
            scorer.load(V7 / f"adaptadores/seed-{seed}" / arm)
            methods[arm][seed] = {key: list(values) for key, values in score_public(roots, scorer).items()}
    planner = PublicScorer(fit_public_planner(trees("train", 1)))
    raw = {"seal_sha256": file_hash(RUN / "SELLO.json"), "started_utc": started.isoformat(),
           "finished_utc": datetime.now(timezone.utc).isoformat(), "gpu": device, "cuda": torch.version.cuda,
           "software": {k: version(k) for k in ("torch", "peft", "transformers")},
           "public_prompt_tokens_max": max(len(ids) for ids, _ in scorer.encodings.values()),
           "methods": methods, "planner": {key: list(values) for key, values in score_public(roots, planner).items()}}
    if replay:
        previous = read(path)
        for field in ("seal_sha256", "methods", "planner", "public_prompt_tokens_max"):
            require(previous[field] == raw[field], f"CUDA replay differs in {field}")
    else:
        atomic_json(path, raw)


def pair_scores(roots, margins):
    require(set(margins) == {r.identity for r in roots}, "Incorrect set of margin rows")
    require(all(len(v) == 2 and all(isfinite(x) for x in v) for v in margins.values()), "Invalid margins")
    by_pair = {}
    for root in roots:
        by_pair.setdefault(root.pair, []).append(root)
    hits = {}
    for pair, members in sorted(by_pair.items()):
        result = independent_timing(members, margins)
        require(result["evaluable_flipped_pairs"] == 1, "A pair lacks opposed optimal visits")
        hits[pair] = int(result["both_timings_correct_native"])
    return hits


def summarise(roots, raw):
    require(raw["seal_sha256"] == file_hash(RUN / "SELLO.json"), "Margins came from a different seal")
    require(set(raw["methods"]) == set(ARMS), "Missing an arm")
    lock = read(V7 / "LOCK.json")
    results = {}
    for arm in ARMS:
        results[arm] = {}
        require(set(raw["methods"][arm]) == set(map(str, SEEDS)), "Missing paired seeds")
        for seed, margins in raw["methods"][arm].items():
            hits = pair_scores(roots, margins)
            result = measure(roots, margins, lock["selected"][seed][arm]["cutoff"])
            require(isclose(mean(hits.values()), result["individual"]["both_timings_correct_native"]), "Inconsistent primary")
            first = {(r.pair, r.variant): margins[r.identity][0] for r in roots}
            contrasts = [first[p, "ask_now"] - first[p, "save"] for p in hits]
            ranking = {"correct": sum(x > 0 for x in contrasts), "tie": sum(x == 0 for x in contrasts),
                       "reversed": sum(x < 0 for x in contrasts)}
            results[arm][seed] = {**result, "correct_pairs": sum(hits.values()), "hits": hits,
                                  "first_visit_margin_order": ranking}
    a, b = (results[arm] for arm in ARMS)
    pairs = sorted({r.pair for r in roots})
    differences = [sum(b[s]["hits"][p] - a[s]["hits"][p] for s in a) for p in pairs]
    delta = sum(differences) / (len(pairs) * len(SEEDS))
    rng = Random(231127)
    draws = sorted(sum(rng.choices(differences, k=len(pairs))) / (len(SEEDS) * len(pairs)) for _ in range(5000))
    planner_hits = pair_scores(roots, raw["planner"])
    return {"margins_sha256": file_hash(RUN / "MARGENES.json"), "pairs": len(pairs), "histories": len(roots),
            "primary_difference": delta, "primary_pair_bootstrap_95": [draws[125], draws[4874]],
            "net_wins_per_pair": dict(sorted(Counter(str(x) for x in differences).items())),
            "matched_future_difference": mean(b[s]["macro"]["future_supported"] - a[s]["macro"]["future_supported"] for s in a),
            "root_to_pair": {r.identity: {"pair": r.pair, "variant": r.variant, "packet": r.packet} for r in roots},
            "methods": results, "planner": {**measure(roots, raw["planner"], lock["planner_cutoff"]),
                                            "correct_pairs": sum(planner_hits.values()), "hits": planner_hits}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "run", "summarize", "replay", "verify"))
    mode = parser.parse_args().mode
    manifest = seal()
    if mode == "freeze":
        stored_or_new("SELLO.json", manifest)
        return
    require(read(RUN / "SELLO.json") == manifest, "Frozen inputs or code changed")
    roots = group_worlds(forest(build_worlds(read(RUN / "PARES.json")), 1))
    if mode in ("run", "replay"):
        run_cuda(roots, replay=mode == "replay")
    result = summarise(roots, read(RUN / "MARGENES.json"))
    if mode in ("verify", "replay"):
        require(result == read(RUN / "RESULTADOS.json"), "Recomputed results differ")
    else:
        stored_or_new("RESULTADOS.json", result)
    print(f"{len(roots)} histories. Paired difference {result['primary_difference']:+.2%}.")


if __name__ == "__main__":
    main()
