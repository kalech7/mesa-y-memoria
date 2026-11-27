"""Reward supported outcomes from complete, interactive food-memory episodes.

The scorer reads terminal traces. Action selection receives only the observation
defined by the environment. The immediate ablation drops the later query score
and shares all other reward terms, including the actual cost of a question.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields
from math import isfinite
from statistics import mean


@dataclass(frozen=True)
class RewardConfig:
    current_weight: float = 1.0
    future_weight: float = 1.5
    question_cost: float = .08
    invalid_output_cost: float = .25
    unsupported_write_cost: float = .75
    unsupported_commitment_cost: float = .35

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError(f"Invalid reward coefficient: {item.name}")


DEFAULT_CONFIG = RewardConfig()
HORIZONS = ("longitudinal", "immediate")
_COUNTERS = (
    "questions", "informative_questions", "redundant_questions", "failed_questions",
    "invalid_outputs", "unsupported_writes", "unsupported_commitments",
    "harmful_persistence", "omitted_confirmed_updates",
)


def _count(value, name, maximum=None):
    if not isinstance(value, int) or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid counter {name}: {value!r}")
    return int(value)


def metrics(trace):
    """Return per-episode measures; later queries never multiply sample size."""
    visits, future = trace["visits"], trace["future"]
    if not visits or not future:
        raise ValueError("A complete trace requires visits and future queries")
    counts = dict.fromkeys(_COUNTERS, 0)
    current_scores = []
    for visit in visits:
        current_scores.append(_count(visit["current_supported"], "current_supported", 1))
        questions = _count(visit["questions"], "questions")
        counts["questions"] += questions
        for name in _COUNTERS[1:]:
            counts[name] += _count(visit.get(name, 0), name, questions if name.endswith("questions") else None)
        if visit.get("unsupported_commitments", 0) and visit["current_supported"]:
            raise ValueError("An unsupported commitment cannot also be a supported response")
    if "questions" in trace and _count(trace["questions"], "episode questions") != counts["questions"]:
        raise ValueError("Questions in the terminal trace differ from visit counters")
    future_scores = [_count(item["supported"], "future supported", 1) for item in future]
    factual_scores = [_count(item.get("correct", supported), "future correct", 1)
                      for item, supported in zip(future, future_scores)]
    if any(supported and not correct for supported, correct in zip(future_scores, factual_scores)):
        raise ValueError("A supported future success must also be factually correct")
    counts["harmful_persistence"] += _count(trace.get("harmful_persistence", 0), "terminal harmful persistence")
    strict = (all(current_scores) and all(future_scores)
              and not any(counts[key] for key in ("invalid_outputs", "unsupported_writes",
                                                  "unsupported_commitments", "harmful_persistence")))
    return {**counts, "current_supported": mean(current_scores), "future_supported": mean(future_scores),
            "future_correct": mean(factual_scores), "strict_joint": int(strict),
            "visits": len(visits), "future_queries": len(future)}


def reward(trace, *, horizon="longitudinal", config=DEFAULT_CONFIG):
    """Compute return from supported successes, emitted questions and actual harm."""
    if horizon not in HORIZONS:
        raise ValueError(f"Unknown reward horizon: {horizon}")
    return _reward(metrics(trace), horizon, config)


def _reward(measured, horizon, config):
    current = config.current_weight * measured["current_supported"]
    future = config.future_weight * measured["future_supported"] if horizon == "longitudinal" else 0.
    penalties = {
        "question_penalty": -config.question_cost * measured["questions"],
        "invalid_output_penalty": -config.invalid_output_cost * measured["invalid_outputs"],
        "unsupported_write_penalty": -config.unsupported_write_cost * measured["unsupported_writes"],
        "unsupported_commitment_penalty": -config.unsupported_commitment_cost * measured["unsupported_commitments"],
    }
    task = current + future
    return {"total": task + sum(penalties.values()), "task": task, "current": current,
            "future": future, **penalties}


def paired_group(traces, *, config=DEFAULT_CONFIG):
    """Diagnose sampled action outcomes for the same starting story and world."""
    if len(traces) < 2:
        raise ValueError("A comparison group needs at least two trajectories")
    keys = {(row["root_id"], row["world_id"], row["split"]) for row in traces}
    if len(keys) != 1:
        raise ValueError("Group trajectories must share an initial root, world and split")
    if len({(len(row["visits"]), len(row["future"])) for row in traces}) != 1:
        raise ValueError("Group trajectories must cover the same scheduled outcomes")
    measurements = [metrics(row) for row in traces]
    returns = {horizon: [_reward(row, horizon, config) for row in measurements] for horizon in HORIZONS}

    def varies(values):
        return len({round(value, 10) for value in values}) > 1

    def preferred(horizon):
        totals = [round(row["total"], 10) for row in returns[horizon]]
        maximum = max(totals)
        return [index for index, value in enumerate(totals) if value == maximum]

    current_varies = varies(row["current_supported"] for row in measurements)
    future_varies = varies(row["future_supported"] for row in measurements)
    task_varies = varies(row["task"] for row in returns["longitudinal"])
    return_varies = varies(row["total"] for row in returns["longitudinal"])
    return {"metrics": measurements, "returns": returns, "task_varies": task_varies,
            "return_varies": return_varies, "only_penalties_vary": return_varies and not task_varies,
            "current_varies": current_varies, "future_varies": future_varies,
            "future_varies_with_current_fixed": future_varies and not current_varies,
            "preferred_trajectories": {name: preferred(name) for name in HORIZONS},
            "horizon_changes_preference": preferred("longitudinal") != preferred("immediate")}


def aggregate(traces, *, config=DEFAULT_CONFIG):
    """Average worlds within each story and give each scenario family equal weight."""
    roots = defaultdict(dict)
    splits = set()
    for row in traces:
        root = row["root_id"]
        splits.add(row["split"])
        identity = (row["world_id"], row.get("realization_id"))
        if identity in roots[root]:
            raise ValueError(f"Duplicate world/realization in root {root}")
        roots[root][identity] = row
    if not roots:
        raise ValueError("Need at least one complete trace")
    if len(splits) != 1:
        raise ValueError("Aggregate one split at a time")
    by_root, families = {}, defaultdict(list)
    for root, items in roots.items():
        episodes = list(items.values())
        family, = {row.get("family", row.get("scenario_family", "all")) for row in episodes}
        worlds = defaultdict(list)
        for row in episodes:
            measured = metrics(row)
            worlds[row["world_id"]].append({**measured, **{f"{name}_return": _reward(measured, name, config)["total"]
                                                        for name in HORIZONS}})
        by_root[root] = {key: mean(mean(value[key] for value in values) for values in worlds.values())
                         for key in next(iter(worlds.values()))[0]}
        families[family].append(by_root[root])
    by_family = {family: {key: mean(row[key] for row in members) for key in members[0]}
                 for family, members in sorted(families.items())}
    return {"roots": len(roots), "episodes": len(traces), "by_root": by_root, "by_family": by_family,
            "macro": {key: mean(panel[key] for panel in by_family.values()) for key in next(iter(by_family.values()))}}
