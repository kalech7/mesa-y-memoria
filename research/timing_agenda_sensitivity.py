"""Reweight the v6 validation worlds after replay, keeping all decisions fixed."""
from collections import defaultdict
import json
from statistics import mean

from research.tesis_shared_budget_planner import traverse
from research.tesis_timing_v6_bank import build_worlds
from research.tesis_timing_v6_env import enumerate_world, fit_public_planner


def measure():
    planner = fit_public_planner(enumerate_world(w) for w in build_worlds("train"))
    # Every root has four equally probable scope combinations under either
    # calendar outcome. Classify that outcome only after choosing the route.
    scores = defaultdict(lambda: defaultdict(list))
    roots = set()
    for world in build_worlds("validation"):
        tree = enumerate_world(world)
        _, planned = traverse(tree, planner)
        first = tree.outcomes[("ASK", "WAIT")]
        second = tree.outcomes[("WAIT", "ASK")]
        announced = world["visits"][0 if world["variant"] == "ask_now" else 1]["attribute"]
        followed = sum(q["attribute"] == announced for q in world["future_queries"]) == 5
        roots.add(world["root_id"])
        for name, outcome in (("public_planner", planned), ("always_first", first), ("always_second", second)):
            if outcome["questions"] != 1:
                raise ValueError("La comparación requiere una pregunta por historia.")
            scores[name][followed].append(outcome["future_supported"])

    rows = []
    for p in (0.2, 0.5, 0.65, 0.8, 1.0):
        row = {"agenda_fulfillment_probability": p}
        for name, outcomes in scores.items():
            if any(len(outcomes[flag]) != 4 * len(roots) for flag in (True, False)):
                raise ValueError("Faltan combinaciones de alcance en validación.")
            row[name] = round(p * mean(outcomes[True]) + (1 - p) * mean(outcomes[False]), 6)
        rows.append(row)
    return {"split": "validation", "stories": len(roots), "questions_per_story": 1,
            "scope_probability_ongoing": 0.5, "metric": "future_supported", "rows": rows}


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2))
