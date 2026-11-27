"""Post-experiment descriptive analyses; never used to choose/train a policy.

Added after the primary run. Its own hash/snapshot is recorded separately from
the frozen experiment source; no primary selection gate is recalculated here.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil

from rl_feasibility.statistics import bootstrap_mean
from .metrics import summarize
from .reward import instant


def load_rows(path):
    return [json.loads(line) for line in Path(path).open()]


def analyze(run):
    run=Path(run)
    def save(name,value):
        (run/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    groups={'CPC':['CPC_DAY','CPC_MEAL'],
            'SCC':['SOURCE_CHANGE','AMBIGUOUS_BEHAVIOR'],
            'TEMPORAL_AMBIGUITY_PROXY':['UNANSWERED'],
            'SAME_SCOPE_IRRELEVANT':['IRRELEVANT_CONFLICT'],
            'RESTRICTION':['RESTRICTION'],
            'RESOLVED_TEMPORAL_CONTROLS':['STABLE_EXCEPTION','EXPLICIT_CHANGE','LATE_RECORDED_HISTORY']}
    conflict_results={}
    horizon_results={}
    operation_results={}
    e1={}
    for directory in ('test','pipeline','shift'):
        conflict_results[directory]={}
        horizon_results[directory]={}
        operation_results[directory]={}
        for file in sorted((run/directory).glob('*.jsonl')):
            rows=load_rows(file)
            if directory=='test':
                e1[file.stem]=rows
            conflict_results[directory][file.stem]={}
            for label,families in groups.items():
                selected=[r for r in rows if r['family'] in families]
                if selected:
                    conflict_results[directory][file.stem][label]=summarize(selected)
            time_bins=defaultdict(list)
            for row in rows:
                bins=defaultdict(list)
                first=instant(row['sessions'][0]['date'])
                for session in row['sessions']:
                    days=(instant(session['date'])-first).days
                    label='0_days' if days==0 else '1_to_90_days' if days<=90 else '91_to_180_days' if days<=180 else '181_plus_days'
                    bins[label].append(session)
                for label,sessions in bins.items():
                    time_bins[label].append({'user_accuracy':sum(s['correct'] for s in sessions)/len(sessions),
                                             'correct':sum(s['correct'] for s in sessions),'sessions':len(sessions)})
            horizon_results[directory][file.stem]={
                label:{'users':len(items),'sessions':sum(i['sessions'] for i in items),
                       'correct':sum(i['correct'] for i in items),
                       'user_macro_accuracy':bootstrap_mean([i['user_accuracy'] for i in items])}
                for label,items in time_bins.items()}
            operation_results[directory][file.stem]={
                'sqlite_operations':dict(Counter(op['operation'] for row in rows for op in row['operations'])),
                'proposed_operations_per_decision':dict(Counter(step['proposed_action']['memory_operation'] for row in rows for step in row['steps'])),
                'executed_operations_per_incoming_fact':dict(Counter(op['executed'] for row in rows for step in row['steps'] for op in step.get('actual_memory_operations',[]))),
                'shielded_decisions':sum(bool(step['shield_reasons'] or step['memory_violations']) for row in rows for step in row['steps'])}
    save('conflict_type_metrics.json',{
        'post_experiment_descriptive':True,'family_mapping':groups,'metrics':conflict_results,
        'BOC':{'evaluation_status':'INSUFFICIENT_COVERAGE_FOR_REPEATED_OSCILLATION',
               'reason':'UNANSWERED represents an ambiguous temporal disagreement, not repeated adoption/reversal cycles.',
               'proxy_reported_separately':'TEMPORAL_AMBIGUITY_PROXY'},
        'interpretation':'CPC/SCC groupings describe authored synthetic families; no externally adjudicated conflict labels.'})
    save('temporal_horizon_metrics.json',{
        'post_experiment_descriptive':True,'unit':'days since first observed session',
        'clustering':'complete synthetic user within each bin; bins are not independent cohorts',
        'metrics':horizon_results})
    save('memory_operation_metrics.json',operation_results)
    scenarios=json.loads((run/'evaluator_private/test.json').read_text())
    upper=sum(sum(1 if s['gold']['foods'] else .7 for s in e['sessions'])/len(e['sessions']) for e in scenarios)/len(scenarios)
    observed=sum(r['return']/len(r['sessions']) for r in e1['PLANNING_BASELINE'])/len(scenarios)
    save('privileged_relaxed_bound.json',{
        'kind':'POSTHOC_PRIVILEGED_RELAXED_UPPER_BOUND_NOT_A_POLICY',
        'free_perfect_information_upper_bound_per_session':upper,
        'planning_observed_return_per_session':observed,'remaining_relaxed_headroom':upper-observed,
        'minimum_difference_preregistered':.1,'used_for_gate_selection':False,
        'assumption':'At most +1 for an accepted grounded food or +0.7 for accepted uncertainty; all other costs nonpositive. This relaxes information/interaction costs and does not describe a reachable policy.'})
    example_ids={}
    for row in e1['CLARIFICATION_RULE_BASED']:
        example_ids.setdefault(row['family'],row['episode_id'])
    examples={}
    for family,uid in example_ids.items():
        examples[family]={name:next(r for r in e1[name] if r['episode_id']==uid)
                          for name in ('CLARIFICATION_RULE_BASED','LAST_WRITE_WINS')}
    save('trace_examples.json',examples)
    source=Path(__file__)
    snapshot=run/'analysis_source_snapshot'
    snapshot.mkdir(exist_ok=True)
    shutil.copy2(source,snapshot/source.name)
    save('descriptive_analysis_manifest.json',{
        'post_experiment':True,'used_for_selection':False,
        'command':f'uv run --offline python -B -m rl_longitudinal.diagnostics --run {run}',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'primary_inference_sources_modified':False})
    return {'conflict_groups':list(groups),'BOC':'repeated cycles not evaluated',
            'remaining_privileged_relaxed_headroom':upper-observed}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    print(json.dumps(analyze(parser.parse_args().run),ensure_ascii=False))
