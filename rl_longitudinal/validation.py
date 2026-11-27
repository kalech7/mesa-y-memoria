"""Mechanical gates and paired causal probes, without changing acceptance gates."""
from collections import Counter
from copy import deepcopy
import csv
import json


def validate_environment(scenarios, output):
    from .evaluate import run_episode, write_json, write_jsonl
    from .longitudinal_environment import LongitudinalEnv
    from .baselines import ClarificationRuleBased
    policy=ClarificationRuleBased()
    rows=[]
    counterfactuals=[]
    probes=[]
    for scenario in scenarios[:120]:
        retained=run_episode(policy,scenario)
        matched_control=run_episode(policy,scenario,enforce_invariants=False)
        no_memory=run_episode(policy,scenario,no_persistence=True,enforce_invariants=False)
        no_reply_memory=run_episode(policy,scenario,no_persist_clarifications=True,enforce_invariants=False)
        no_asking=run_episode(policy,scenario,no_clarification=True,enforce_invariants=False)
        def drop_first(observation,action):
            if observation['session_index']==0:
                action['memory_operation']='NOOP'
            return action
        first_drop=run_episode(policy,scenario,intervention=drop_first,enforce_invariants=False)
        def score(row):
            return sum(bool(s['correct']) for s in row['sessions'][1:])
        differences=[i for i,(a,b) in enumerate(zip(matched_control['sessions'],first_drop['sessions']))
                     if i>0 and a['correct']!=b['correct']]
        counterfactuals.append({
            'episode_id':scenario['episode_id'],'family':scenario['family'],
            'only_intervention':'NOOP instead of committing incoming evidence in first session; history/clarify guards disabled explicitly',
            'same_initial_observation':matched_control['policy_observations'][0]==first_drop['policy_observations'][0],
            'same_guard_configuration':matched_control['invariant_enforcement']==first_drop['invariant_enforcement'],
            'same_first_decision_quality':matched_control['sessions'][0]['correct']==first_drop['sessions'][0]['correct'],
            'later_decisions_differ_at_sessions':differences,
            'retained_future_correct':score(matched_control),'dropped_first_future_correct':score(first_drop),
            'retained_return':matched_control['return'],'drop_return':first_drop['return'],
            'retained_trajectory':matched_control['sessions'],'drop_trajectory':first_drop['sessions']})
        rows.append(retained)
        probes.append({'episode_id':scenario['episode_id'],'family':scenario['family'],
                       'retained':retained,'matched_control':matched_control,'no_persistence':no_memory,
                       'no_persist_clarifications':no_reply_memory,'no_clarification':no_asking})
    # Reset reproducibility and private-label perturbation tests use equivalent
    # public initial conditions, never the evaluator's labels as observations.
    reproducible=True
    gold_independent=True
    for scenario in scenarios[:10]:
        env1=LongitudinalEnv(scenario)
        env2=LongitudinalEnv(scenario)
        altered=deepcopy(scenario)
        for session in altered['sessions']:
            session['gold']['foods']=['PRIVATE_SENTINEL_NOT_A_FOOD']
        env3=LongitudinalEnv(altered)
        try:
            o1,_=env1.reset(seed=991)
            o2,_=env2.reset(seed=991)
            o3,_=env3.reset(seed=991)
            reproducible &= o1==o2
            gold_independent &= o1==o3
        finally:
            env1.close();env2.close();env3.close()
    required_steps=[s for row in rows for s in row['steps']]
    valid_traces=all(all(s['trace_id'] in {t['trace_id'] for t in row['retrieval_traces']}
                         for s in row['steps']) for row in rows)
    correct=all(s['correct'] for row in rows for s in row['sessions'])
    safe=all(not s['restriction_violation'] for row in rows for s in row['sessions'])
    necessary=sum(s['needed_asks'] for row in rows for s in row['sessions'])
    performed=sum(s['needed_asks_performed'] for row in rows for s in row['sessions'])
    unnecessary=sum(s['unnecessary_asks'] for row in rows for s in row['sessions'])
    repeated=sum(s['repeated_asks'] for row in rows for s in row['sessions'])
    future_effect=sum(bool(c['later_decisions_differ_at_sessions']) for c in counterfactuals)
    same_now_future_effect=sum(c['same_first_decision_quality'] and bool(c['later_decisions_differ_at_sessions'])
                               for c in counterfactuals)
    result={'gate_a':bool(reproducible and gold_independent and valid_traces and safe and correct),
            'gate_b':bool(necessary>0 and performed==necessary and unnecessary==0 and repeated==0),
            'gate_c':bool(same_now_future_effect>0),
            'controlled_users':len(rows),'controlled_sessions':sum(len(r['sessions']) for r in rows),
            'all_decisions_have_actual_sqlite_retrieval':valid_traces,
            'reset_reproducible':reproducible,'private_gold_mutation_does_not_change_initial_observation':gold_independent,
            'controlled_all_correct':correct,'controlled_no_restriction_violations':safe,
            'needed_clarifications':necessary,'performed_needed_clarifications':performed,
            'unnecessary_questions':unnecessary,'repeated_questions':repeated,
            'counterfactual_users_with_future_effect':future_effect,
            'same_current_quality_but_future_effect':same_now_future_effect,
            'all_causal_pairs_have_same_guards':all(c['same_guard_configuration'] for c in counterfactuals),
            'external_validation':'NOT_PERFORMED; annotations below are proposed synthetic cases',
            'ablation_interpretation':'Disabling persistence/clarification breaks selected invariants; not a compliant agent.'}
    write_json(output/'sequentiality_analysis.json',{'summary':result,'counterfactuals':counterfactuals})
    write_jsonl(output/'ablation_episodes.jsonl',probes)
    from .metrics import summarize, paired
    ablations={name:[probe[name] for probe in probes] for name in
               ('retained','matched_control','no_persistence','no_persist_clarifications','no_clarification')}
    write_json(output/'ablation_metrics.json',{
        'validation_only':True,
        'metrics':{name:summarize(items) for name,items in ablations.items()},
        'paired_matched_control_minus_ablation':{
            name:{field:paired(ablations['matched_control'],items,field=field)
                  for field in ('return_per_session','accuracy','followup_accuracy')}
            for name,items in ablations.items() if name not in ('retained','matched_control')},
        'no_delayed_reward_training':'NOT_EXECUTED; no learned policy was compared under alternative reward horizons.'})
    conflicts=[]
    for row in rows:
        for observation,trace in zip(row['policy_observations'],row['steps']):
            if observation['reconciliation']['conflicts'] or observation['reconciliation']['missing_slots']:
                conflicts.append({'episode_id':row['episode_id'],'family':row['family'],
                                  'observation':observation,'step':trace})
    write_jsonl(output/'temporal_conflicts.jsonl',conflicts)
    # Blind review is prepared, never silently adjudicated by code or subagents.
    with (output/'human_review_blind.csv').open('w',newline='') as stream:
        fields=['review_id','history','query','query_at','response','later_session',
                'reviewer','acceptable_actions','evidence_ids','uncertainty','rationale']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        chosen={}
        for row in rows:
            chosen.setdefault(row['family'],row)
        for family,row in sorted(chosen.items()):
            for i,session in enumerate(row['sessions']):
                writer.writerow({'review_id':f"case_{len(chosen)}_{row['episode_id'][:8]}_{i}",
                                 'history':json.dumps({'persisted':session['memory_before'],
                                                      'incoming':[s['pending_before'] for s in session['steps']]},ensure_ascii=False),
                                 'query':session['query'],'query_at':session['query_at'],
                                 'response':json.dumps(session['action'],ensure_ascii=False),
                                 'later_session':json.dumps(row['sessions'][i+1]['action'],ensure_ascii=False) if i+1<len(row['sessions']) else ''})
    return result
