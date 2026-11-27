"""Evaluation-only user-clustered summaries; never imported by a policy."""
from collections import Counter
from copy import deepcopy
from numbers import Real
import random
from time import monotonic

from rl_feasibility.statistics import bootstrap_mean, percentile
from .environment import MinimalEnv

RATIOS = {
    'conflict_detection_rate': ('conflicts_detected', 'potential_conflicts'),
    'relevant_memory_recall': ('retrieved_relevant_count', 'relevant_memory_count'),
    'temporal_error_rate': ('temporal_interpretation_errors', 'temporal_comparisons'),
    'temporal_type_error_rate': ('temporal_interpretation_errors', 'temporal_comparisons'),
    'interval_applicability_error_rate': ('interval_applicability_errors', 'temporal_comparisons'),
    'extraction_error_rate': ('extraction_errors', 'extraction_total'),
    'original_parser_error_rate': ('original_parser_errors', 'extraction_total'),
    'necessary_question_recall': ('necessary_asks', 'needed_questions'),
}


def ratio_estimate(rows, numerator, denominator, repetitions=1000, other=None):
    def value(group):
        den = sum(r[denominator] for r in group)
        return sum(r[numerator] for r in group)/den if den else None
    observed = value(rows)
    reference = value(other) if other is not None else None
    if observed is None or (other is not None and reference is None):
        return {'mean': None, 'ci95': None, 'n': len(rows),
                'numerator':sum(r[numerator] for r in rows),
                'denominator':sum(r[denominator] for r in rows)}
    rng=random.Random(819)
    samples=[]
    for _ in range(repetitions):
        indices=rng.choices(range(len(rows)),k=len(rows))
        estimate=value([rows[i] for i in indices])
        baseline=value([other[i] for i in indices]) if other is not None else 0
        if estimate is not None and baseline is not None:
            samples.append(estimate-baseline)
    return {'mean':observed-(reference if other is not None else 0),
            'ci95':[percentile(samples,.025),percentile(samples,.975)],'n':len(rows),
            'numerator':sum(r[numerator] for r in rows),
            'denominator':sum(r[denominator] for r in rows)}


def rollout(policy, scenario, representation='O3', extraction='VISIBLE_STRUCTURED'):
    start = monotonic()
    env = MinimalEnv(scenario, representation=representation, extraction=extraction)
    observations = []
    try:
        initial = env.reset()
        obs = initial[0] if isinstance(initial, tuple) else initial
        for _ in range(8):
            observations.append(deepcopy(obs))
            obs, _, done, truncated, _ = env.step(policy.select_action(deepcopy(obs)))
            if done or truncated:
                break
        else:
            raise RuntimeError('Minimal episode exceeded eight actions')
        result = env.diagnostic_metrics()
        result['policy_observations'] = observations
        result['elapsed_seconds'] = monotonic() - start
        result['method'] = policy.name
        return result
    finally:
        env.close()


def evaluate(policy, scenarios, representation='O3', extraction='VISIBLE_STRUCTURED'):
    return [rollout(policy, s, representation, extraction) for s in scenarios]


def scalar_fields(rows):
    return sorted({k for row in rows for k, v in row.items()
                   if isinstance(v, Real) and not isinstance(v, bool)
                   and k not in ('seed',)})


def summarize(rows, repetitions=1000):
    if len({r['user_id'] for r in rows}) != len(rows):
        raise ValueError('A summary requires one independent trajectory per synthetic user')
    fields = scalar_fields(rows)
    result = {'users': len(rows), 'metrics': {
        key: bootstrap_mean([r[key] for r in rows if isinstance(r.get(key), Real)], repetitions)
        for key in fields}}
    for key,(num,den) in RATIOS.items():
        if all(num in row and den in row for row in rows):
            result['metrics'][key]=ratio_estimate(rows,num,den,repetitions)
    result['actions_proposed'] = dict(Counter(
        s['proposed_action']['conversation'] for r in rows for s in r.get('steps', [])))
    result['by_family'] = {}
    for family in sorted({r['family'] for r in rows}):
        group = [r for r in rows if r['family'] == family]
        result['by_family'][family] = {'users': len(group), 'metrics': {
            k: bootstrap_mean([r[k] for r in group if isinstance(r.get(k), Real)], repetitions)
            for k in fields}}
        for key,(num,den) in RATIOS.items():
            if all(num in row and den in row for row in group):
                result['by_family'][family]['metrics'][key]=ratio_estimate(group,num,den,repetitions)
    return result


def paired(left, right, repetitions=1000):
    """Left minus right, resampling entire users (both sessions together)."""
    a = {r['user_id']: r for r in left}
    b = {r['user_id']: r for r in right}
    if len(a) != len(left) or len(b) != len(right) or a.keys() != b.keys():
        raise ValueError('Paired comparisons require identical unique users')
    fields = set(scalar_fields(left)) & set(scalar_fields(right))
    result={k: bootstrap_mean([a[u][k] - b[u][k] for u in sorted(a)], repetitions)
            for k in sorted(fields) if k != 'elapsed_seconds'}
    for key,(num,den) in RATIOS.items():
        if num in fields and den in fields:
            result[key]=ratio_estimate([a[u] for u in sorted(a)],num,den,repetitions,
                                       [b[u] for u in sorted(a)])
    return result


def average_seed_rows(groups):
    """Collapse seeds inside a user; seeds never inflate the user sample size."""
    indexes = [{r['user_id']: r for r in rows} for rows in groups]
    if not indexes or any(i.keys() != indexes[0].keys() for i in indexes):
        raise ValueError('Seed groups must contain identical users')
    fields = set.intersection(*(set(scalar_fields(rows)) for rows in groups))
    result = []
    for user in sorted(indexes[0]):
        row = deepcopy(indexes[0][user])
        for key in fields:
            row[key] = sum(i[user][key] for i in indexes) / len(indexes)
        result.append(row)
    return result
