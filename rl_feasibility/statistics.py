"""Episode-level descriptive and paired bootstrap statistics, standard library."""
import math
import random
import statistics


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def percentile(values, quantile):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def bootstrap_mean(values, repetitions=1000, seed=819):
    values = list(values)
    if not values:
        return {"mean": None, "ci95": None, "n": 0}
    if min(values) == max(values):
        return {"mean": mean(values), "ci95": [values[0], values[0]], "n": len(values)}
    rng = random.Random(seed)
    estimates = [sum(rng.choices(values, k=len(values))) / len(values) for _ in range(repetitions)]
    return {"mean": mean(values), "ci95": [percentile(estimates, .025), percentile(estimates, .975)], "n": len(values)}


def paired_difference(first, second, field="return", repetitions=1000):
    """Rows must be unique independent users; compare first minus second."""
    left = {row["scenario_id"]: row for row in first}
    right = {row["scenario_id"]: row for row in second}
    if len(left) != len(first) or len(right) != len(second) or left.keys() != right.keys():
        raise ValueError("Paired comparison requires identical, unique episode IDs")
    values = [float(left[key][field]) - float(right[key][field]) for key in sorted(left)
              if left[key].get(field) is not None and right[key].get(field) is not None]
    return bootstrap_mean(values, repetitions)
