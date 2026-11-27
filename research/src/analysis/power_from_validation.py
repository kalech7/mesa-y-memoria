"""Pre-test paired-root/whole-seed Monte Carlo planning from real validation.

This plans precision under declared sensitivity models, not guaranteed power.
Unobserved discordance receives a conservative variance floor, especially when
the pilot has zero differences. Test outcomes are never accepted as inputs.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from .paired_statistics import aggregate_rows, binomial_interval


def _wilson(successes, simulations):
    z = 1.959963984540054
    p = successes / simulations
    denominator = 1 + z * z / simulations
    center = (p + z * z / (2 * simulations)) / denominator
    half = z * math.sqrt(p * (1 - p) / simulations + z * z / (4 * simulations ** 2)) / denominator
    return [max(0., center - half), min(1., center + half)]


def _simulate(profiles, families, *, sizes, simulations, effect, seed, root_fraction=1.):
    import numpy as np
    seeds = sorted({s for values in profiles.values() for s in values})
    if any(set(values) != set(seeds) for values in profiles.values()):
        raise ValueError("Complete paired training-seed profiles are required")
    grouped = defaultdict(list)
    for root in sorted(profiles):
        grouped[families[root]].append([profiles[root][s] for s in seeds])
    arrays = [np.asarray(values, dtype=float) for _, values in sorted(grouped.items())]
    centered_at = float(np.mean([array.mean() for array in arrays]))
    root_values = np.concatenate([a.mean(axis=1) for a in arrays])
    observed_variance = float(np.var(root_values, ddof=1)) if len(root_values) > 1 else 0.
    homogeneous = bool(np.max(root_values) - np.min(root_values) < 1e-12)
    # If no pilot root differs from its common observed value, an unobserved
    # discordant root can still exist. CP upper event probability gives a
    # transparent sensitivity variance for a bounded +/-1 paired difference.
    zero_deviations = int(sum(abs(v - float(root_values.mean())) > 1e-12 for v in root_values))
    q_upper = binomial_interval(0, len(root_values))["interval"][1]
    extra_variance = max(0., q_upper - observed_variance)
    amplitude = math.sqrt(extra_variance / q_upper) if q_upper > 0 else 0.
    rng = np.random.default_rng(seed)
    scenarios = []
    critical = 2.5758293035489004  # two-sided alpha .01
    for total_roots in sizes:
        actual_roots = int(round(total_roots * root_fraction))
        if actual_roots < len(arrays) * 2 or actual_roots % len(arrays):
            raise ValueError("Candidate root sizes must be balanced and at least two per stratum")
        per_family = actual_roots // len(arrays)
        for variance_multiplier in (1., 1.5):
            successes = 0
            effect_sum = 0.
            for start in range(0, simulations, 128):
                count = min(128, simulations - start)
                seed_indices = rng.integers(0, len(seeds), size=(count, len(seeds)))
                estimates = np.zeros(count)
                variances = np.zeros(count)
                for array in arrays:
                    root_indices = rng.integers(0, len(array), size=(count, per_family))
                    sampled = array[root_indices]
                    sampled = np.take_along_axis(sampled, seed_indices[:, None, :], axis=2).mean(axis=2)
                    deviations = sampled - centered_at
                    if extra_variance > 0:
                        # Root-level unobserved discordance is shared across
                        # seeds; resampling more seeds cannot make it disappear.
                        active = rng.random((count, per_family)) < q_upper
                        signs = rng.integers(0, 2, size=(count, per_family)) * 2 - 1
                        deviations += active * signs * amplitude
                    values = effect + math.sqrt(variance_multiplier) * deviations
                    estimates += values.mean(axis=1) / len(arrays)
                    variances += values.var(axis=1, ddof=1) / per_family / len(arrays) ** 2
                detected = estimates > critical * np.sqrt(variances)
                successes += int(detected.sum())
                effect_sum += float(estimates.sum())
            # A finite run with all successes does not establish power=1.
            scenarios.append({"candidate_total_id_roots": total_roots, "contrast_independent_roots": actual_roots,
                              "variance_multiplier": variance_multiplier,
                              "estimated_power": (successes + .5) / (simulations + 1),
                              "monte_carlo_95_interval": _wilson(successes, simulations),
                              "monte_carlo_successes": successes, "simulations": simulations,
                              "monte_carlo_mean_effect": effect_sum / simulations,
                              "power_estimator": "Jeffreys mean for finite Monte Carlo success count"})
    adequate = [s["candidate_total_id_roots"] for s in scenarios
                if s["variance_multiplier"] == 1.5 and s["estimated_power"] >= .8]
    return {"status": "executed", "planning_model": "paired_root_stratified_and_whole_seed_resampling_normal_test",
            "roots_in_validation_pilot": len(root_values), "training_seeds": seeds,
            "observed_paired_root_variance": observed_variance, "homogeneous_pilot": homogeneous,
            "observed_nonconstant_root_differences": zero_deviations,
            "unobserved_discordance_probability_upper_95": q_upper,
            "additional_root_variance_floor": extra_variance,
            "nominal_effect": effect, "alpha": .01, "target_power": .8,
            "recommended_total_id_roots_in_conservative_model": min(adequate) if adequate else None,
            "scenarios": scenarios,
            "limitation": "Sensitivity model, not identified population power; pilot and three-seed uncertainty remain. Zero observed variation never proves certainty."}


def plan_from_rows(rows, destination, *, simulations=2000, sizes=range(300, 1201, 150), effect=.05,
                   seed=7331, feasible_id_roots=300, selected_steps=None):
    """Plan from selected-checkpoint validation traces only; never choose by test.

    If multiple callback checkpoints are present, selected_steps must explicitly
    map (method, seed) to the selected validation checkpoint step.
    """
    destination = Path(destination)
    if destination.suffix != ".json":
        destination.mkdir(parents=True, exist_ok=True)
        destination = destination / "POWER_VALIDATION.json"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
    if simulations < 20 or feasible_id_roots < 300 or feasible_id_roots > 1200 or feasible_id_roots % 5:
        raise ValueError("Invalid Monte Carlo or feasible-size configuration")
    compact = []
    for row in rows:
        if row.get("split") != "validation":
            raise ValueError("POWER_PLANNING_REQUIRES_VALIDATION_ONLY")
        if selected_steps is not None and "checkpoint_step" in row:
            if row["checkpoint_step"] != selected_steps[(row["method"], row["seed"])]:
                continue
        flat = {**row, **row.get("metrics", {})}
        compact.append({key: value for key, value in flat.items() if value is None or isinstance(value, (str, int, float, bool))}
                       | {"eligible": row.get("trace", {}).get("episode", {}).get("opportunity", {}).get("eligible", False)})
    contrasts = []
    definitions = [("B2L_minus_B3", "B2-L", "B3", "FJS", 1.),
                   ("B2L_minus_B1", "B2-L", "B1", "FJS", 1.),
                   ("B2L_minus_B2I", "B2-L", "B2-I", "FJS", 1.),
                   ("B0_O2_minus_O1", ("B0", "O2", None), ("B0", "O1", None), "C2", 1.),
                   ("B1_I2_minus_I1_F4", ("B1", "O2", "I2"), ("B1", "O2", "I1"), "FJS", .2)]
    for name, left, right, metric, fraction in definitions:
        relevant = [r for r in compact if r.get(metric) is not None and (fraction != .2 or r["eligible"])]
        if isinstance(left, str):
            relevant = [r for r in relevant if r.get("observation_condition", "O2") == "O2" and not r.get("intervention")
                        and r.get("reader_id", "fixed_symbolic") == "fixed_symbolic" and r.get("extraction_track", "REF") == "REF"
                        and not r.get("nonresponse_rate")]
        maps = []
        try:
            aggregated = aggregate_rows(relevant, metric)
            for selector in (left, right):
                mapping = {}
                for row in aggregated:
                    matched = row["method"] == selector if isinstance(selector, str) else (
                        row["method"], row["condition"], row["intervention"]) == selector
                    if matched:
                        key = (row["seed"], row["root_id"])
                        if key in mapping:
                            raise ValueError("VALIDATION_CHECKPOINT_SELECTION_REQUIRED")
                        mapping[key] = (row["scenario_family"], row[metric])
                maps.append(mapping)
            if not maps[0] or not maps[1]:
                contrasts.append({"contrast": name, "status": "not_executed", "reason": "Missing selected validation method/condition"})
                continue
            if maps[0].keys() != maps[1].keys():
                raise ValueError("Selected validation roots and seeds are not exactly paired")
            profiles, families = defaultdict(dict), {}
            for (training_seed, rid), (family, value) in maps[0].items():
                if maps[1][(training_seed, rid)][0] != family:
                    raise ValueError("Validation family mismatch")
                profiles[rid][training_seed] = value - maps[1][(training_seed, rid)][1]
                families[rid] = family
            result = _simulate(profiles, families, sizes=list(sizes), simulations=simulations,
                               effect=effect, seed=seed, root_fraction=fraction)
            result["contrast"] = name
            contrasts.append(result)
        except (ValueError, KeyError) as exc:
            contrasts.append({"contrast": name, "status": "inconclusive", "reason": str(exc)})
    required = [c for c in contrasts[:3] if c["status"] == "executed"]
    recommendations = [c["recommended_total_id_roots_in_conservative_model"] for c in required]
    fully_executed = len(required) == 3
    recommended = max(recommendations) if fully_executed and all(v is not None for v in recommendations) else None
    executed = [c for c in contrasts if c["status"] == "executed"]
    all_recommendations = [c["recommended_total_id_roots_in_conservative_model"] for c in executed]
    family_planning_complete = len(executed) == 5
    family_recommended = max(all_recommendations) if family_planning_complete and all(v is not None for v in all_recommendations) else None
    report = {"status": "executed" if required else "inconclusive", "validation_only": True,
              "analysis_seed": seed, "simulations_per_scenario": simulations,
              "contrasts": contrasts, "recommended_id_roots": recommended,
              "recommended_id_roots_scope": "Three training-continuation contrasts only",
              "all_five_contrasts_planned": family_planning_complete,
              "all_five_contrasts_recommended_id_roots": family_recommended,
              "resource_feasible_locked_id_roots": feasible_id_roots,
              "precision_limited": not family_planning_complete or family_recommended is None or family_recommended > feasible_id_roots,
              "size_decision": "Resource-feasible size chosen before test; never enlarged after inspecting test",
              "source_compact_sha256": hashlib.sha256(json.dumps(compact, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
              "warning": "Planning is conditional on noisy pilot estimates and explicit conservative assumptions; it is not a power guarantee."}
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def plan_from_traces(paths, destination, **options):
    """Load explicitly selected validation JSONL callbacks with bounded memory."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    rows = []
    sources = []
    for path in paths:
        source = Path(path)
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for line in handle:
                digest.update(line)
                if line.strip():
                    row = json.loads(line)
                    rows.append({key: row.get(key) for key in ("method", "seed", "root_id", "latent_world_id", "realization_id",
                                 "scenario_family", "split", "observation_condition", "intervention", "reader_id", "extraction_track", "checkpoint_step") if key in row}
                                | {"metrics": row.get("metrics", {}), "C2": row.get("C2", row.get("metrics", {}).get("C2")),
                                   "FJS": row.get("FJS", row.get("metrics", {}).get("FJS")),
                                   "trace": {"episode": {"opportunity": row.get("trace", {}).get("episode", {}).get("opportunity", {})}}})
        sources.append({"path": str(source), "sha256": digest.hexdigest()})
    result = plan_from_rows(rows, destination, **options)
    result["sources"] = sources
    path = Path(destination)
    if path.suffix != ".json":
        path = path / "POWER_VALIDATION.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
