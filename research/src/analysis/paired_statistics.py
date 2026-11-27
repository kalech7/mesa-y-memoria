"""Paired root inference. Worlds, wording and seeds are never extra users."""
from __future__ import annotations

from collections import defaultdict
import math
import random
from statistics import NormalDist, mean


def _metric(row, name):
    value = row.get(name, row.get("metrics", {}).get(name))
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"Missing/nonfinite metric {name}")
    return float(value)


def aggregate_rows(rows, metric="future_justified_success"):
    """One output per method, seed, root and condition, averaging realizations.

    Repeated trace rows with the same episode identity are deduplicated before
    averaging; contradictory repeated rows indicate malformed input.
    """
    episodes, groups = {}, defaultdict(list)
    for row in rows:
        condition = row.get("condition", row.get("observation_condition", "O2"))
        # Intervention is part of condition; never average across causal arms.
        condition = (condition, row.get("intervention"))
        key = (row["method"], row["seed"], row["root_id"], condition)
        identity = key + (row.get("latent_world_id", "single"), row.get("realization_id", "single"))
        value = _metric(row, metric)
        if identity in episodes:
            if episodes[identity] != value:
                raise ValueError("Conflicting duplicate episode result")
            continue
        episodes[identity] = value
        groups[key].append((value, row.get("scenario_family", row.get("family"))))
    result = []
    for (method, seed, rid, condition), items in groups.items():
        families = {family for _, family in items}
        if len(families) != 1 or None in families:
            raise ValueError("Exactly one explicit family is required per root")
        result.append({"method": method, "seed": seed, "root_id": rid,
                       "condition": condition[0], "intervention": condition[1],
                       "scenario_family": next(iter(families)), metric: mean(v for v, _ in items),
                       "episodes": len(items)})
    return result


def _quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def paired_bootstrap(rows, left, right, metric="future_justified_success", *, condition=None,
                     resamples=20000, confidence=.99, seed=7331, resample_seeds=False):
    """Stratified root bootstrap with identical root indices for both methods.

    left/right can be method strings or (method, condition, intervention) tuples.
    The default 99% interval corresponds to the five prespecified contrasts.
    Sensitivity optionally resamples whole, paired training seeds as well.
    """
    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap configuration")
    aggregated = aggregate_rows(rows, metric)
    def selected(row, selector):
        if isinstance(selector, (tuple, list)):
            return (row["method"], row["condition"], row["intervention"]) == tuple(selector)
        return row["method"] == selector and (condition is None or row["condition"] == condition)
    maps = []
    for selector in (left, right):
        mapping = {}
        for row in aggregated:
            if selected(row, selector):
                key = (row["seed"], row["root_id"])
                if key in mapping:
                    raise ValueError("Specify condition/intervention to obtain one paired row")
                mapping[key] = (row["scenario_family"], row[metric])
        maps.append(mapping)
    if not maps[0] or maps[0].keys() != maps[1].keys():
        raise ValueError("PAIRED_ROOTS_REQUIRED: both methods must contain exactly the same roots and seeds")
    differences = defaultdict(dict)
    families = {}
    for key, (family, value) in maps[0].items():
        if family != maps[1][key][0]:
            raise ValueError("Paired root family mismatch")
        training_seed, rid = key
        differences[rid][training_seed] = value - maps[1][key][1]
        families[rid] = family
    seeds = sorted({s for values in differences.values() for s in values})
    if any(set(values) != set(seeds) for values in differences.values()):
        raise ValueError("Complete paired seeds required for each root")
    strata = defaultdict(list)
    for rid, family in families.items():
        strata[family].append(rid)
    def statistic(sample, seed_sample):
        return mean(mean(mean(differences[r][s] for s in seed_sample) for r in roots)
                    for roots in sample.values())
    estimate = statistic(strata, seeds)
    rng = random.Random(seed)
    boot = []
    for _ in range(resamples):
        sample = {f: [rng.choice(roots) for _ in roots] for f, roots in strata.items()}
        chosen_seeds = [rng.choice(seeds) for _ in seeds] if resample_seeds else seeds
        boot.append(statistic(sample, chosen_seeds))
    alpha = (1 - confidence) / 2
    return {"estimate": estimate, "interval": [_quantile(boot, alpha), _quantile(boot, 1 - alpha)],
            "confidence": confidence, "resamples": resamples, "analysis_seed": seed,
            "independent_roots": len(differences), "roots_by_family": {f: len(r) for f, r in strata.items()},
            "training_seeds": seeds, "seed_sensitivity": resample_seeds,
            "per_seed": {str(s): statistic(strata, [s]) for s in seeds},
            "interpretation": "Root uncertainty conditional on observed training seeds" if not resample_seeds
                              else "Sensitivity to paired seed resampling; three seeds limit precision"}


def binomial_interval(events, roots, confidence=.95):
    """Exact conservative Clopper-Pearson interval, including zero/all events."""
    if roots < 1 or not 0 <= events <= roots or not 0 < confidence < 1:
        raise ValueError("Invalid binomial count")
    alpha = (1 - confidence) / 2
    def cdf(k, p):
        if p <= 0:
            return 1.0
        if p >= 1:
            return float(k >= roots)
        terms = [math.lgamma(roots + 1) - math.lgamma(i + 1) - math.lgamma(roots - i + 1)
                 + i * math.log(p) + (roots - i) * math.log1p(-p) for i in range(k + 1)]
        largest = max(terms)
        return min(1.0, math.exp(largest) * sum(math.exp(t - largest) for t in terms))
    def solve(k, target):
        low, high = 0.0, 1.0
        for _ in range(65):
            mid = (low + high) / 2
            if cdf(k, mid) > target:
                low = mid
            else:
                high = mid
        return (low + high) / 2
    lower = 0.0 if events == 0 else solve(events - 1, 1 - alpha)
    upper = 1.0 if events == roots else solve(events, alpha)
    return {"estimate": events / roots, "interval": [lower, upper], "confidence": confidence,
            "independent_roots": roots, "events": events, "method": "clopper_pearson"}


def select_checkpoint(checkpoints):
    """Common validation utility, then false writes/questions, then earliest.

    Step zero must be supplied rather than retrospectively ruled out.
    """
    if not checkpoints or not any(c["step"] == 0 for c in checkpoints):
        raise ValueError("STEP_ZERO_REQUIRED")
    for c in checkpoints:
        if c.get("split", "validation") != "validation":
            raise ValueError("Checkpoint selection is validation-only")
        if not math.isfinite(c["utility"]):
            raise ValueError("Invalid validation utility")
    return min(checkpoints, key=lambda c: (-c["utility"], c.get("unsupported_writes", 0),
                                           c.get("questions", 0), c["step"]))


def plan_test_size(validation_differences, *, sizes=range(300, 1201, 150), effect=.05,
                   confidence=.99, target_power=.8, simulations=2000, seed=7331):
    """Pre-test paired-root power simulation with an explicit conservative arm.

    Input: validation-only rows {root_id,scenario_family,seed,difference}.
    Resamples paired root profiles and complete seeds; shifts their mean to the
    prespecified effect. Variance x1.5 is a sensitivity scenario, not known power.
    This planning routine never opens or extends the test based on test results.
    """
    if not validation_differences:
        raise ValueError("Validation pilot differences required")
    grouped, families = defaultdict(dict), {}
    for row in validation_differences:
        if row.get("split", "validation") != "validation":
            raise ValueError("Power planning accepts validation only")
        grouped[row["root_id"]][row["seed"]] = float(row["difference"])
        families[row["root_id"]] = row["scenario_family"]
    seeds = sorted({s for profile in grouped.values() for s in profile})
    if any(set(p) != set(seeds) for p in grouped.values()):
        raise ValueError("Complete paired seed profiles required")
    strata = defaultdict(list)
    for rid in grouped:
        strata[families[rid]].append(rid)
    center = mean(mean(mean(grouped[r][s] for s in seeds) for r in rs) for rs in strata.values())
    critical = NormalDist().inv_cdf((1 + confidence) / 2)
    rng = random.Random(seed)
    results = []
    for size in sizes:
        if size < 300 or size > 1200 or size % len(strata):
            raise ValueError("Prespecified size range must be balanced within 300..1200")
        for scale in (1.0, 1.5):
            detected = 0
            for _ in range(simulations):
                seed_sample = [rng.choice(seeds) for _ in seeds]
                values = []
                for roots in strata.values():
                    for _ in range(size // len(strata)):
                        rid = rng.choice(roots)
                        values.append(effect + scale * (mean(grouped[rid][s] for s in seed_sample) - center))
                mu = mean(values)
                variance = sum((v - mu) ** 2 for v in values) / max(1, size - 1)
                # Normal approximation applied to paired root differences, not
                # independent proportions; report its limitation explicitly.
                detected += mu > critical * math.sqrt(variance / size)
            results.append({"roots": size, "variance_scale": scale, "estimated_power": detected / simulations})
    adequate = [r["roots"] for r in results if r["variance_scale"] == 1.5 and r["estimated_power"] >= target_power]
    return {"recommended_roots": min(adequate) if adequate else None, "scenarios": results,
            "status": "planning_only", "method": "paired_root_and_seed_resampling_normal_test",
            "limitation": "Pilot uncertainty and three-seed variability limit power estimation"}
