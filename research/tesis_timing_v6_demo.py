"""Reproduce one paired TRAIN example with the actual validation-selected v6 adapters."""
from __future__ import annotations

import argparse
from collections import Counter
from math import isfinite
from pathlib import Path

from research.src.artifacts import digest, file_hash
from research.tesis_shared_budget_qwen import atomic_json
from research.tesis_timing_v6_bank import build_worlds
from research.tesis_timing_v6_env import enumerate_world, replay
from research.tesis_timing_v6_eval import ASK, WAIT, group_worlds
from research.tesis_timing_v6_experiment import ARMS, SEEDS, WEIGHTS, check_sources, read, signature


EXAMPLE = "Hoy quiero sopa para cenar."
ROUTES = {"ask_today": (ASK, WAIT), "save_for_tomorrow": (WAIT, ASK), "never_ask": (WAIT, WAIT)}


def demo_pair(worlds):
    """Deterministically select the first exact soup pair in the original TRAIN order."""
    if not worlds or any(w["split"] != "train" for w in worlds):
        raise ValueError("The demonstration may only use TRAIN")
    match = next((w for w in worlds if w["opening"] == EXAMPLE), worlds[0])
    return [w for w in worlds if w["pair_id"] == match["pair_id"]]


def fulfilled_example(worlds):
    for w in worlds:
        first, second = w["visits"][:2]
        if first["scope"] != "ongoing" or second["scope"] != "ongoing":
            continue
        counts = Counter(q["attribute"] for q in w["future_queries"])
        heavy, light = (first, second) if w["variant"] == "ask_now" else (second, first)
        if counts[heavy["attribute"]] > counts[light["attribute"]]:
            return w
    raise ValueError("No illustrative world with both habits and the announced plan")


def checked_adapter(run, seed, arm, selection):
    relative = Path(selection["adapter"])
    expected = Path(f"seed-{seed}") / arm / f"checkpoint-{selection['round']:03d}" / "adapter"
    folder = (run / relative).resolve()
    if relative != expected or not folder.is_relative_to(run):
        raise ValueError("Adapter must belong to its locked seed, arm, and round")
    if signature(folder, WEIGHTS) != selection["sha256"]:
        raise ValueError("Selected checkpoint weights or configuration changed")
    return folder


def native_choices(scorer, roots):
    first = scorer.margins([root.first for root in roots])
    waiting = [root for root, margin in zip(roots, first, strict=True) if margin <= 0]
    second = dict(zip((r.identity for r in waiting),
                      scorer.margins([r.second_after_wait for r in waiting]) if waiting else [], strict=True))
    if not all(isfinite(v) for v in [*first, *second.values()]):
        raise ValueError("Nonfinite Qwen margin")
    decisions = {}
    for root, margin in zip(roots, first, strict=True):
        later = second.get(root.identity)
        path = (ASK, WAIT) if margin > 0 else (WAIT, ASK) if later > 0 else (WAIT, WAIT)
        decisions[root.identity] = {"actions": list(path), "ask_minus_wait_logit_v1": margin,
                                    "ask_minus_wait_logit_v2": later,
                                    "v2_source": "one_legal_action" if later is None else "qwen",
                                    "expected_metrics_over_worlds": root.outcomes[path]}
    return decisions


def build_demo(worlds, run, lock, scorer):
    pair = demo_pair(worlds)
    trees = [enumerate_world(w, budget=1) for w in pair]
    roots = group_worlds(trees)
    if sorted(r.variant for r in roots) != ["ask_now", "save"]:
        raise ValueError("The TRAIN example must contain both counterpart agendas")
    members = {}
    for root in roots:
        same = [w for w in pair if w["root_id"] == root.identity]
        example = fulfilled_example(same)
        for path in ROUTES.values():
            root.outcomes[path]["longitudinal"] = sum(tree.mass * tree.outcomes[path]["longitudinal"]
                                                      for tree in trees if tree.story["root_id"] == root.identity)
        members[root.identity] = {
            "agenda": root.variant, "cue": example["cue"],
            "second_utterance": example["visits"][1]["utterance"],
            "world_count": len(same), "world_weights": {w["world_id"]: w["mass"] for w in same},
            "expected_over_all_worlds": {key: root.outcomes[path] for key, path in ROUTES.items()},
            "fulfilled_world_with_two_habits": {
                "world_id": example["world_id"],
                "traces": {key: replay(example, ROUTES[key], budget=1)
                           for key in ("ask_today", "save_for_tomorrow")}},
        }
    models = {}
    for seed in SEEDS:
        models[str(seed)] = {}
        for arm in ARMS:
            item = lock["selected"][str(seed)][arm]
            scorer.load(checked_adapter(run, seed, arm, item))
            models[str(seed)][arm] = {"checkpoint": item["adapter"], "round": item["round"],
                                      "sha256": item["sha256"], "roots": native_choices(scorer, roots)}
    return {"split": "train", "purpose": "Ejemplo ilustrativo de entrenamiento, separado del resultado reservado.",
            "budget_per_story": 1, "policy": "Elección nativa de Qwen con mayor logit entre las acciones legales. Los empates eligen WAIT.",
            "pair_id": roots[0].pair, "opening": pair[0]["opening"], "members": members, "models": models}


def main():
    from research.tesis_timing_v6_train import MarginScorer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    run = parser.parse_args().run.resolve()
    plan = check_sources(run)
    lock = read(run / "LOCK.json")
    if lock["sha256"] != signature(run, lock["sha256"]):
        raise ValueError("Validation selection changed after locking")
    worlds = build_worlds("train")
    if digest(worlds) != plan["development_worlds_sha256"]["train"]:
        raise ValueError("TRAIN changed since preregistration")
    result = build_demo(worlds, run, lock, MarginScorer())
    result["lock_sha256"] = file_hash(run / "LOCK.json")
    result["demo_source_sha256"] = file_hash(Path(__file__))
    atomic_json(run / "DEMO_PAR.json", result)
    print("Ejemplo de TRAIN —", result["opening"])
    for root_id, member in result["members"].items():
        print(member["cue"], member["second_utterance"])
        print("Acierto futuro esperado al preguntar hoy / mañana / nunca", [
            round(member["expected_over_all_worlds"][key]["future_supported"], 4) for key in ROUTES])
        for arm, selected in result["models"]["101"].items():
            print(arm, "semilla 101", selected["roots"][root_id]["actions"])
    print("Resultados, citas SQLite y decisiones en las tres semillas:", run / "DEMO_PAR.json")


if __name__ == "__main__":
    main()
