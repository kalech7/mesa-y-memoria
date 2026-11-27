"""V4 inference: realizations -> worlds -> roots -> equally weighted families.

This module deliberately does not change historical analyses. Training seeds
are paired replications of the recipe, never additional independent users.
"""
from __future__ import annotations

from collections import defaultdict
import math
import random
from statistics import mean

from .analysis.paired_statistics import _quantile, binomial_interval

CLAIMS = 11
CONFIDENCE = 1 - .05 / CLAIMS


def _value(row, metric):
    value = row.get(metric, row.get("metrics", {}).get(metric))
    if value is None or isinstance(value, bool):
        if isinstance(value, bool):
            return float(value)
        raise ValueError("Missing metric: " + metric)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite metric: " + metric)
    return value


def aggregate_condition(rows, selector, metric):
    """Require an unambiguous condition and preserve unequal world sizes."""
    worlds, identities, condition_sets = defaultdict(list), set(), defaultdict(set)
    for row in rows:
        if not all(row.get(k) == v for k, v in selector.items()):
            continue
        seed = row.get("training_seed", row.get("seed"))
        required = ("root_id", "latent_world_id", "realization_id")
        if seed is None or any(row.get(k) is None for k in required):
            raise ValueError("Explicit seed/root/world/realization required")
        condition = tuple((k, str(row.get(k))) for k in (
            "condition_id", "checkpoint_id", "observation_condition", "intervention", "reader_id", "extraction_track",
            "nonresponse_rate", "future_probe_id", "generation_seed", "max_questions_s3"))
        condition_sets[seed].add(tuple((k, str(row.get(k))) for k in (
            "checkpoint_id", "observation_condition", "intervention", "reader_id", "extraction_track",
            "nonresponse_rate", "future_probe_id", "max_questions_s3")))
        identity = (seed, *(row[k] for k in required), row.get("stochastic_replica_id"), condition)
        if identity in identities:
            raise ValueError("Duplicate episode identity; do not silently deduplicate")
        identities.add(identity)
        family = row.get("scenario_family", row.get("family"))
        if family is None:
            raise ValueError("Explicit family required")
        worlds[(seed, row["root_id"], family, row["latent_world_id"], condition)].append(_value(row, metric))
    if any(len(values) != 1 for values in condition_sets.values()):
        raise ValueError("Ambiguous condition across roots: selectors must fix reader, extraction and other factors")
    roots, conditions = defaultdict(list), defaultdict(set)
    for (seed, rid, family, world, condition), values in worlds.items():
        roots[(seed, rid, family)].append(mean(values))
        conditions[(seed, rid)].add(condition)
    if any(len(c) != 1 for c in conditions.values()):
        raise ValueError("Ambiguous condition: selector must fix all intervention factors and probes")
    result = {}
    for (seed, rid, family), values in roots.items():
        if (seed, rid) in result:
            raise ValueError("A root cannot belong to multiple families")
        result[(seed, rid)] = (family, mean(values))
    return result


def paired_contrast(rows, candidate, control, metric, *, resamples=20000,
                    confidence=CONFIDENCE, analysis_seed=7331, harm_margin=None):
    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid inference configuration")
    rows = list(rows)
    def episode_keys(selector):
        return {(r.get("training_seed", r.get("seed")), r.get("root_id"), r.get("latent_world_id"),
                 r.get("realization_id"), r.get("future_probe_id"), r.get("stochastic_replica_id"))
                for r in rows if all(r.get(k) == v for k, v in selector.items())}
    if episode_keys(candidate) != episode_keys(control):
        raise ValueError("Complete paired episode worlds, realizations and probes required")
    left, right = (aggregate_condition(rows, selector, metric) for selector in (candidate, control))
    if not left or left.keys() != right.keys():
        raise ValueError("Complete paired roots and training seeds required")
    by_root, families = defaultdict(dict), {}
    for (seed, rid), (family, value) in left.items():
        if family != right[(seed, rid)][0] or (rid in families and families[rid] != family):
            raise ValueError("Paired family mismatch")
        by_root[rid][seed] = value - right[(seed, rid)][1]
        families[rid] = family
    seeds = sorted({s for p in by_root.values() for s in p}, key=str)
    if any(set(p) != set(seeds) for p in by_root.values()):
        raise ValueError("Every root needs the complete paired seed profile")
    strata = defaultdict(list)
    for rid, family in families.items():
        strata[family].append(rid)
    profiles = {f: [mean(by_root[r].values()) for r in roots] for f, roots in strata.items()}
    estimate = mean(mean(values) for values in profiles.values())
    rng = random.Random(analysis_seed)
    boot = [mean(mean(rng.choices(values, k=len(values))) for values in profiles.values())
            for _ in range(resamples)]
    alpha = (1 - confidence) / 2
    interval = [_quantile(boot, alpha), _quantile(boot, 1 - alpha)]
    zero_discordances = all(all(v == 0 for v in p.values()) for p in by_root.values())
    result = {"status": "executed", "metric": metric, "candidate": candidate, "control": control,
              "estimate": estimate, "interval": interval, "confidence": confidence,
              "familywise_alpha": .05, "planned_claims": CLAIMS,
              "multiplicity_method": "Bonferroni individual two-sided intervals for 11 planned claims",
              "resamples": resamples, "analysis_seed": analysis_seed,
              "independent_roots": len(by_root), "training_seeds": seeds,
              "roots_by_family": {f: len(v) for f, v in strata.items()},
              "per_family": {f: mean(v) for f, v in profiles.items()},
              "per_seed": {str(s): mean(mean(by_root[r][s] for r in roots) for roots in strata.values()) for s in seeds},
              "zero_discordances": zero_discordances,
              "interpretation": "Conditional on observed seeds and templates; a null interval is not equivalence"}
    if zero_discordances:
        # An event is defined on independent roots, not on a continuous mean.
        # Union-bound across strata preserves the requested confidence.
        event_confidence = 1 - (1 - confidence) / len(strata)
        bounds = {f: binomial_interval(0, len(roots), event_confidence) for f, roots in strata.items()}
        result["zero_discordance_sensitivity"] = {
            "event": "At least one paired training-seed root mean differs between methods",
            "bounds_by_family": bounds,
            "max_absolute_mean_difference_if_metric_in_0_1": mean(v["interval"][1] for v in bounds.values()),
            "limitation": "Bound requires independent roots, the stated root event and a metric in [0,1]; not a binomial CI for the continuous mean"}
    if harm_margin is not None:
        if not 0 < harm_margin < 1:
            raise ValueError("Predeclared harm margin must be in (0,1)")
        upper = interval[1]
        if zero_discordances:
            upper = max(upper, result["zero_discordance_sensitivity"]["max_absolute_mean_difference_if_metric_in_0_1"])
        result["guard"] = {"harm_margin": harm_margin, "upper_bound_used": upper,
                           "passed": upper < harm_margin,
                           "rule": "upper bound of candidate minus control strictly below harm margin"}
    seed_effects = list(result["per_seed"].values())
    result["between_seed_sensitivity"] = {"observed_effect_range": [min(seed_effects), max(seed_effects)],
                                           "interpretation": "Sensitivity over observed paired training seeds; three seeds do not estimate a user population"}
    return result
