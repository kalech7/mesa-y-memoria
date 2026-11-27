"""Evaluation-only metrics with paired complete-user bootstrap intervals."""
from collections import Counter
from copy import deepcopy
import random
from statistics import mean
from time import monotonic

from rl_feasibility.statistics import bootstrap_mean, percentile
from .environment_adapter import PilotEnv


def select_observation(validation_results):
    """Select O2/O3 by fixed-rule validation return; ties prefer O2."""
    def score(rep):
        result=validation_results[rep]
        if 'mean_return' in result:
            return result['mean_return']
        return result['metrics']['return']['mean']
    return max(('O2','O3'),key=score)


def rollout(policy,scenario,representation='O2',condition='ORACLE',db_path=None):
    env=PilotEnv(scenario,representation=representation,condition=condition,db_path=db_path)
    start=monotonic()
    try:
        obs=env.reset()
        for _ in range(8):
            obs,_,done,truncated,_=env.step(policy.select_action(deepcopy(obs)))
            if done or truncated:
                break
        else:
            raise RuntimeError('Episode exceeded the declared action budget')
        result=env.diagnostic_metrics()
        result['method']=getattr(policy,'name',type(policy).__name__)
        result['elapsed_seconds']=monotonic()-start
        return result
    finally:
        env.close()


def evaluate(policy,scenarios,representation='O2',condition='ORACLE',directory=None,retain=0):
    rows=[]
    for index,scenario in enumerate(scenarios):
        db=directory/f'{index}.sqlite' if directory is not None and index<retain else None
        rows.append(rollout(policy,scenario,representation,condition,db))
    return rows


MEANS=('error_rate','accuracy','return','future_accuracy','questions',
       'necessary_asks','unnecessary_asks','incorrect_updates','future_errors',
       'shield_interventions','unsupported_answers')
RATIOS={
    'relevant_memory_recall':('retrieved_relevant_count','relevant_memory_count'),
    'conflict_detection_rate':('conflicts_detected','potential_conflicts'),
    'necessary_question_recall':('necessary_asks','needed_questions'),
    'extraction_error_rate':('extraction_errors','extraction_total'),
    'interval_applicability_error_rate':('interval_applicability_errors','temporal_comparisons'),
}


def _ratio(rows,num,den):
    denominator=sum(r.get(den,0) for r in rows)
    return sum(r.get(num,0) for r in rows)/denominator if denominator else None


def ratio_ci(rows,num,den,repetitions=1000,other=None):
    point=_ratio(rows,num,den)
    reference=_ratio(other,num,den) if other is not None else 0
    result={'mean':None,'ci95':None,'n':len(rows),
            'numerator':sum(r.get(num,0) for r in rows),'denominator':sum(r.get(den,0) for r in rows)}
    if point is None or reference is None:
        return result
    rng=random.Random(51005)
    values=[]
    for _ in range(repetitions):
        indices=rng.choices(range(len(rows)),k=len(rows))
        a=_ratio([rows[i] for i in indices],num,den)
        b=_ratio([other[i] for i in indices],num,den) if other is not None else 0
        if a is not None and b is not None:
            values.append(a-b)
    result.update(mean=point-reference,ci95=[percentile(values,.025),percentile(values,.975)])
    return result


def summarize(rows,repetitions=1000):
    if not rows or len({r['user_id'] for r in rows})!=len(rows):
        raise ValueError('Expected nonempty, unique complete users')
    metrics={key:bootstrap_mean([r.get(key,0) for r in rows],repetitions,seed=51005) for key in MEANS}
    metrics['reward_per_session']=bootstrap_mean([r['return']/len(r['sessions']) for r in rows],repetitions,seed=51005)
    for key,(num,den) in RATIOS.items():
        metrics[key]=ratio_ci(rows,num,den,repetitions)
    # Counts include all final decisions; clarification steps are costed separately.
    counts={'users':len(rows),'sessions':sum(len(r['sessions']) for r in rows),
        'correct':sum(r['accuracy']*len(r['sessions']) for r in rows),
        'incorrect':sum(r['error_rate']*len(r['sessions']) for r in rows),
        'questions':sum(r['questions'] for r in rows),'necessary_asks':sum(r['necessary_asks'] for r in rows),
        'unnecessary_asks':sum(r['unnecessary_asks'] for r in rows),
        'incorrect_updates':sum(r['incorrect_updates'] for r in rows),
        'safety_interventions':sum(r['shield_interventions'] for r in rows)}
    families={}
    for family in sorted({r['family'] for r in rows}):
        group=[r for r in rows if r['family']==family]
        families[family]={'users':len(group),**{key:mean(r.get(key,0) for r in group) for key in MEANS}}
    actions=Counter()
    for row in rows:
        actions.update(row.get('seed_mean_action_counts',Counter(s['proposed_action']['conversation'] for s in row['steps'])))
    actions=dict(actions)
    return {'counts':counts,'metrics':metrics,'by_family':families,'actions':actions,
        'confidence_unit':'complete synthetic user; both sessions together'}


def paired(left,right,repetitions=1000):
    a={r['user_id']:r for r in left};b={r['user_id']:r for r in right}
    if len(a)!=len(left) or len(b)!=len(right) or a.keys()!=b.keys():
        raise ValueError('Expected identical unique users in both methods')
    order=sorted(a)
    metrics={key:bootstrap_mean([a[u].get(key,0)-b[u].get(key,0) for u in order],
        repetitions,seed=51005) for key in MEANS}
    for key,(num,den) in RATIOS.items():
        metrics[key]=ratio_ci([a[u] for u in order],num,den,repetitions,[b[u] for u in order])
    return metrics


def average_seeds(groups):
    indexes=[{r['user_id']:r for r in group} for group in groups]
    if not indexes or any(i.keys()!=indexes[0].keys() for i in indexes):
        raise ValueError('Seed averages require identical users')
    keys=set(MEANS)|{k for pair in RATIOS.values() for k in pair}
    rows=[]
    for uid in sorted(indexes[0]):
        row=deepcopy(indexes[0][uid])
        for key in keys:
            row[key]=mean(i[uid].get(key,0) for i in indexes)
        counts=Counter()
        for index in indexes:
            counts.update({k:v/len(indexes) for k,v in Counter(
                s['proposed_action']['conversation'] for s in index[uid]['steps']).items()})
        row['seed_mean_action_counts']=dict(counts)
        # Numeric aggregation only. Source trajectories remain in individual files.
        rows.append(row)
    return rows
