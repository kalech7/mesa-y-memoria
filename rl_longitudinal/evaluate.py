"""Reproducible CPU-only SPEC-RL-003 experiment; never overwrites a run."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

from .audit import audit_partitions, verify_historical_manifest
from .metrics import summarize, paired, annotate


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                    allow_nan=False)+'\n')


def write_jsonl(path, rows):
    with Path(path).open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')


def run_episode(policy, scenario, *, condition='ORACLE_STRUCTURED', db_path=None,
                enforce_invariants=None, no_persistence=False, intervention=None,
                no_persist_clarifications=False, no_clarification=False):
    from .longitudinal_environment import LongitudinalEnv
    conforming = not bool(getattr(policy,'ablations',())) if enforce_invariants is None else enforce_invariants
    kwargs = dict(condition=condition, enforce_invariants=conforming,
                  no_persistence=no_persistence, no_clarification=no_clarification,
                  no_persist_clarifications=no_persist_clarifications)
    if db_path is not None:
        kwargs['db_path'] = db_path
    start = perf_counter()
    env = LongitudinalEnv(scenario, **kwargs)
    policy_observations, policy_actions = [], []
    try:
        obs, info = env.reset()
        for step in range(100):
            policy_observations.append(obs)
            action = policy.select_action(obs)
            if intervention is not None:
                action = intervention(obs,dict(action))
            policy_actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        else:
            raise RuntimeError('Episode exceeded the declared bounded interaction budget')
        result = env.diagnostic_metrics()
        result['policy_observations'] = policy_observations
        result['policy_actions'] = policy_actions
        result['policy_name'] = policy.name
        result['invariant_ablations'] = list(getattr(policy,'ablations',()))
        result['elapsed_ms'] = (perf_counter()-start)*1000
        return annotate(result,scenario)
    finally:
        if hasattr(env,'close'):
            env.close()


def run_policies(policies, scenarios, directory, *, condition='ORACLE_STRUCTURED',
                 repetitions=1000, retain_databases=0):
    directory.mkdir(parents=True,exist_ok=False)
    rows, metrics = {}, {}
    for policy in policies:
        if hasattr(policy,'reset_usage'):
            policy.reset_usage()
        result = []
        for index, scenario in enumerate(scenarios):
            db = None
            if index < retain_databases:
                database_directory = directory/'sqlite'/policy.name
                database_directory.mkdir(parents=True,exist_ok=True)
                db = database_directory/(scenario['episode_id']+'.sqlite')
            result.append(run_episode(policy,scenario,condition=condition,db_path=db))
        rows[policy.name] = result
        metrics[policy.name] = summarize(result,repetitions)
        if hasattr(policy,'usage_summary'):
            metrics[policy.name]['policy_usage']=policy.usage_summary()
        write_jsonl(directory/(policy.name+'.jsonl'),result)
        print(f'{condition} {policy.name}: {len(result)} users',flush=True)
    write_json(directory/'metrics.json',metrics)
    return rows,metrics


def gate_headroom(validation_rows, repetitions=1000):
    reference = 'PLANNING_BASELINE'
    competitors = [name for name in validation_rows if name != reference]
    best = max(competitors,key=lambda name:sum(r['return']/len(r['sessions'])
                                             for r in validation_rows[name])/len(validation_rows[name]))
    estimate = paired(validation_rows[reference],validation_rows[best],repetitions=repetitions)
    difference, ci = estimate['mean'],estimate['ci95']
    status = ('PASS' if difference>=.10 and ci[0]>0 else
              'NO_MEANINGFUL_MARGIN' if ci[1]<.10 else 'INCONCLUSIVE')
    return {'status':status,'validation_only':True,'reference_method':reference,
            'best_non_rl_method':best,'minimum_difference':.10,'difference':difference,
            'paired_ci':ci,'n':estimate['n'],
            'interpretation':'Measured against this visible model-based planner, not an optimality proof.'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--train-users',type=int,default=600)
    parser.add_argument('--validation-users',type=int,default=120)
    parser.add_argument('--test-users',type=int,default=240)
    parser.add_argument('--shift-users',type=int,default=120)
    parser.add_argument('--seed-offset',type=int,default=0)
    parser.add_argument('--bootstrap',type=int,default=1000)
    parser.add_argument('--skip-tests',action='store_true')
    parser.add_argument('--retain-databases',type=int,default=2)
    args=parser.parse_args(argv)
    if min(args.train_users,args.validation_users,args.test_users,args.shift_users,args.bootstrap)<1:
        parser.error('All data sizes and bootstrap repetitions must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    start=perf_counter()
    test_result={'executed':False,'passed':False}
    if not args.skip_tests:
        with (args.output/'unittest.log').open('w') as stream:
            check=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],
                                 stdout=stream,stderr=subprocess.STDOUT)
        test_result={'executed':True,'passed':check.returncode==0,'returncode':check.returncode}
        if check.returncode:
            write_json(args.output/'memory_validation.json',{'gate_a':False,'tests':test_result})
            raise SystemExit('Critical tests failed. See unittest.log; no experiment was run.')
    from .scenario_generator import generate_episodes
    from .baselines import make_baselines, train_supervised
    from .reward import RewardConfig
    partitions={name:generate_episodes(size,seed=seed+args.seed_offset,split=name,
                                       distribution='shift' if name=='shift' else 'standard')
                for name,size,seed in [('train',args.train_users,42),('validation',args.validation_users,43),
                                       ('test',args.test_users,44),('shift',args.shift_users,45)]}
    private=args.output/'evaluator_private'
    private.mkdir(mode=0o700)
    for name,scenarios in partitions.items():
        write_json(private/(name+'.json'),scenarios)
    audit=audit_partitions(partitions)
    write_json(args.output/'data_audit.json',audit)
    from .longitudinal_environment import LongitudinalEnv
    trained_supervised=train_supervised(partitions['train'],env_factory=LongitudinalEnv,
                                       output_dir=args.output/'supervised')
    supervised=trained_supervised.pop('policy')
    policies=list(make_baselines(supervised).values())
    write_json(args.output/'supervised_training.json',trained_supervised)
    validation_rows,validation_metrics=run_policies(policies,partitions['validation'],args.output/'validation',
                                                  repetitions=args.bootstrap)
    headroom=gate_headroom(validation_rows,args.bootstrap)
    write_json(args.output/'headroom_analysis.json',headroom)
    # Independent transition evidence is required BEFORE any RL invocation.
    from .validation import validate_environment
    validation=validate_environment(partitions['validation'],args.output)
    gate_report={'gate_a':test_result['passed'] and validation['gate_a'],
                 'gate_b':validation['gate_b'],'gate_c':validation['gate_c'],'gate_d':headroom}
    write_json(args.output/'memory_validation.json',{'tests':test_result,**validation,**gate_report})
    training={'executed':False,'rl_steps':0,'seeds_executed':[],
              'reason':'Gates A-D did not justify RL; an untrained compatible implementation is not evidence of learning.'}
    if gate_report['gate_a'] and gate_report['gate_b'] and gate_report['gate_c'] and headroom['status']=='PASS':
        from .train_rl import train_three_seeds
        training_result=train_three_seeds(partitions['train'],partitions['validation'],
                                         gate_report=gate_report,output_dir=args.output/'rl_training')
        learned=training_result.pop('selected_policy')
        training={'executed':True,**training_result}
        policies.append(learned)
    write_json(args.output/'training_metrics.json',training)
    sources=[p for root in ('rl_longitudinal','tests','memory','simulator') for p in Path(root).glob('*.py')]
    sources += list(Path('docs').glob('rl_longitudinal*.md'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for source in sources:
        target=args.output/'source_snapshot'/source
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    write_json(args.output/'freeze.json',{'sources':hashes,'selection_used_test':False,
                                        'gate_report':gate_report,'reward':asdict(RewardConfig())})
    rows,metrics=run_policies(policies,partitions['test'],args.output/'test',
                             repetitions=args.bootstrap,retain_databases=args.retain_databases)
    pipeline_rows,pipeline_metrics=run_policies(policies,partitions['test'],args.output/'pipeline',
                                               condition='PIPELINE',repetitions=args.bootstrap)
    shift_rows,shift_metrics=run_policies(policies,partitions['shift'],args.output/'shift',
                                         repetitions=args.bootstrap)
    comparisons={}
    for name in rows:
        if name!='PLANNING_BASELINE':
            comparisons['PLANNING_BASELINE_minus_'+name]={field:paired(rows['PLANNING_BASELINE'],rows[name],
                                                                      field=field,repetitions=args.bootstrap)
                                                        for field in ('return_per_session','accuracy','followup_accuracy')}
    write_json(args.output/'baseline_comparison.json',{'validation':validation_metrics,'test':metrics,
                                                      'paired_test':comparisons})
    write_json(args.output/'longitudinal_metrics.json',{'ORACLE_STRUCTURED':metrics,'PIPELINE':pipeline_metrics,
                                                       'FAMILY_HOLDOUT':shift_metrics})
    write_json(args.output/'clarification_metrics.json',{
        name:{key:value for key,value in m.items() if 'clarif' in key or key=='shield_interventions'}
        for name,m in metrics.items()})
    from rl_feasibility.statistics import bootstrap_mean
    sensitivity={}
    for cost in (.05,.10,.20):
        sensitivity[str(cost)]={
            name:bootstrap_mean([(row['return']+(.10-cost)*sum(s['asks'] for s in row['sessions']))/len(row['sessions'])
                                 for row in episodes],args.bootstrap)
            for name,episodes in rows.items()}
    write_json(args.output/'reward_sensitivity.json',{
        'clarification_costs':sensitivity,'frozen_trajectories':True,'retraining_performed':False,
        'interpretation':'Descriptive reweighting; not an estimate of a policy relearned under another reward.'})
    errors=[]
    for condition,collection in [('ORACLE_STRUCTURED',rows),('PIPELINE',pipeline_rows),('FAMILY_HOLDOUT',shift_rows)]:
        for policy_name,episodes in collection.items():
            for episode in episodes:
                for session in episode['sessions']:
                    if not session['correct']:
                        errors.append({'episode_id':episode['episode_id'],'family':episode['family'],
                                       'condition':condition,'policy':policy_name,'session':session,
                                       'input':{'query':session['query'],'query_at':session['query_at'],
                                                'history':session['memory_before'],
                                                'steps_incoming':[step['pending_before'] for step in session['steps']]},
                                       'expected':session['expected'],'obtained':session['action'],
                                       'component':('extraction_and_temporal_adapter' if condition=='PIPELINE' else 'baseline_policy'),
                                       'possible_cause':('Missing or misinterpreted food, scope, provenance, temporal kind or supersession links; inspect extraction against original evidence.' if condition=='PIPELINE' else 'Recency or no-clarification ablation ignores information required by this task; other failures require individual review.'),
                                       'recommended_check':('Compare communicated text, extracted facts and lineage before changing the policy; keep this test held out.' if condition=='PIPELINE' else 'Inspect dated applicable evidence and question/operation traces; do not relabel gold solely to fit the policy.'),
                                       'classification':'automatic_proposed_cause_not_human_adjudicated'})
    write_jsonl(args.output/'error_analysis.jsonl',errors)
    write_json(args.output/'historical_preservation.json',verify_historical_manifest('reports/rl_longitudinal/initial_manifest.json'))
    write_json(args.output/'manifest.json',{'command':sys.argv,'parameters':vars(args)|{'output':str(args.output)},
                                         'elapsed_seconds':perf_counter()-start,'training_executed':training['executed'],
                                         'human_adjudication':'NOT_PERFORMED','gates':gate_report,
                                         'errors':len(errors),'offline':True,'new_installations':False})
    from .report import generate_report
    generate_report(args.output)
    print(json.dumps({'output':str(args.output),'gates':gate_report,'training_executed':training['executed']},ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':
    main()
