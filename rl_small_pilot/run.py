"""SPEC-RL-005: CPU neural-weights pilot with validation-only selection."""
# Keep the default project Python intact. macOS Intel needs the explicitly
# provisioned PyTorch/Python-3.12 environment, so re-exec before torch imports.
import importlib.util
import os
from pathlib import Path
import sys


def _training_runtime():
    if importlib.util.find_spec('torch') is not None:
        return
    root=Path(__file__).resolve().parent.parent
    alternate=root/'.venv-rl-small/bin/python'
    if not alternate.exists() or os.environ.get('RL_SMALL_REEXEC')=='1':
        raise SystemExit('PyTorch unavailable. Provision the authorized isolated environment; no installation is performed by this command.')
    env=dict(os.environ,RL_SMALL_REEXEC='1',RL_SMALL_LAUNCHER_PYTHON=sys.executable)
    os.execve(str(alternate),[str(alternate),'-B','-m','rl_small_pilot.run',*sys.argv[1:]],env)


def main():
    _training_runtime()
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-users',type=int,default=300)
    p.add_argument('--validation-users',type=int,default=60)
    p.add_argument('--test-users',type=int,default=120)
    p.add_argument('--seed',type=int,default=51005)
    p.add_argument('--seeds',nargs='+',type=int,default=[101,202,303])
    p.add_argument('--m0-epochs',type=int,default=20)
    p.add_argument('--m0-lr',type=float,default=.003)
    p.add_argument('--optimizer-steps',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=20)
    p.add_argument('--learning-rates',nargs='+',type=float,default=[.0001,.0003])
    p.add_argument('--eval-every',type=int,default=10)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--skip-tests',action='store_true',help='Only for a small technical smoke; main run executes all tests.')
    p.add_argument('--llm-base',default=os.environ.get('LLM_BASE_URL',''))
    p.add_argument('--llm-token-env',default='USFQ_API_KEY')
    p.add_argument('--llm-workers',type=int,default=2)
    p.add_argument('--llm-users',type=int,default=0,help='0 disables inference; use test-users for fully matched M4, a subset is marked exploratory.')
    execute(p.parse_args())


def execute(args):
    from collections import Counter
    from dataclasses import asdict
    from datetime import datetime,timezone
    import hashlib
    import json
    import platform
    import shutil
    import subprocess
    from time import monotonic
    import torch
    from rl_observability.policies import HistoricalClarificationRules
    from .environment_adapter import PilotEnv,generate_splits,RewardConfig
    from .evaluate import evaluate,summarize,paired,average_seeds,select_observation
    from .supervised_baseline import train_supervised,train_additional_supervised
    from .rl_trainer import train_reinforce
    from .error_diagnostics import diagnose_rows,compare_paired

    if min(args.train_users,args.validation_users,args.test_users)<10 or len(set(args.seeds))<3:
        raise ValueError('Each split must include both directions of all five families; at least three seeds')
    if min(args.m0_epochs,args.optimizer_steps,args.batch_size,args.eval_every)<1 or args.bootstrap<100:
        raise ValueError('Positive bounded training/evaluation configuration required')
    if not 0<=args.llm_users<=args.test_users or len(args.learning_rates)>3 or any(x<=0 for x in args.learning_rates):
        raise ValueError('Invalid inference or learning-rate budget')
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    start=monotonic()
    def save(path,value):
        path=out/path;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')
    def rows_file(path,rows):
        path=out/path;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n' for r in rows))
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(created_at=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),
        torch=torch.__version__,device='CPU',threads=1,algorithm='REINFORCE',
        algorithm_justification='Tabular Q cannot update neural weights; SPEC-RL-005 authorizes the elementary policy-gradient alternative. Choice is unrelated to POMDP status.',
        official_thesis_question_unchanged=True,
        thesis_question='¿Cuándo y cómo debe solicitar aclaración un asistente con memorias incompatibles, y cómo mejora la respuesta actual y la coherencia posterior frente a prompting y heurísticas?',
        quantity_reduction='300/60/120 defaults reduce CPU replication cost relative to suggested 600/120/240; all five mechanisms and both preference directions retained.',
        m0_selection='validation return, earliest tie',continuation_selection='mean validation return across three seeds; learning-rate grid only; each checkpoint earliest tie',
        preregistered_meaningful_return_margin=.10,
        no_private_inputs=True,external_policy_is_not_a_trained_llm=True)
    save(Path('config.json'),config)
    shutil.copyfile(Path(__file__).parent/'PROTOCOL.md',out/'PROTOCOL.md')
    from importlib.metadata import distributions
    save(Path('training_environment.json'),{'python':platform.python_version(),
        'platform':platform.platform(),'interpreter':sys.executable,
        'packages':{d.metadata['Name']:d.version for d in distributions()},
        'pytorch_installation_authorized':True,'test_plan_approved':True,
        'failed_default_install':'No matching torch wheel for macOS Intel / CPython 3.13',
        'isolated_runtime':'Python 3.12 / torch 2.2.2; existing project Python unchanged'})
    for directory in ('rl_small_pilot','rl_observability','rl_longitudinal','memory','simulator'):
        for source in Path(directory).glob('*.py'):
            dest=out/'source_snapshot'/source;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    save(Path('source_hashes.json'),{str(p.relative_to(out/'source_snapshot')):sha(p) for p in (out/'source_snapshot').rglob('*.py')})
    # Record immutable historical sources/results so preservation can be checked.
    old_paths=[Path('reports/rl_observability/run_001/freeze.json'),Path('reports/rl_observability/run_001/test/metrics.json'),Path('reports/rl_longitudinal/INFORME_RL_LONGITUDINAL.md')]
    save(Path('historical_hashes.json'),{str(p):sha(p) for p in old_paths})
    if not args.skip_tests:
        print('Validating the entire unittest suite, including neural-weights tests.',flush=True)
        with (out/'unittest.log').open('w') as stream:
            check=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],stdout=stream,stderr=subprocess.STDOUT)
        if check.returncode:
            save(Path('failure.json'),{'stage':'tests','returncode':check.returncode})
            raise RuntimeError('Test gate failed; no training. See unittest.log')
    splits=generate_splits(args.train_users,args.validation_users,args.test_users,args.seed)
    for split,scenarios in splits.items():
        save(Path('evaluator_private')/f'{split}.json',scenarios)
    manifests={}
    for split,scenarios in splits.items():
        manifests[split]={'users':len(scenarios),'sha256':sha(out/'evaluator_private'/f'{split}.json'),
            'families':dict(Counter(s['family'] for s in scenarios)),
            'initial_directions':dict(Counter(s['initial_direction'] for s in scenarios)),
            'contexts':sorted({tuple(s['context_combination']) for s in scenarios}),
            'template_ids':sorted({s['template_id'] for s in scenarios}),
            'shared_five_mechanisms':True,'symbolic_support_can_overlap':True}
    for a,b in (('train','validation'),('train','test'),('validation','test')):
        assert {s['user_id'] for s in splits[a]}.isdisjoint(s['user_id'] for s in splits[b])
        assert {tuple(s['context_combination']) for s in splits[a]}.isdisjoint(tuple(s['context_combination']) for s in splits[b])
    save(Path('dataset_manifest.json'),manifests)
    rules=HistoricalClarificationRules()
    val_observations={};obs_rows={}
    for representation in ('O1','O2','O3'):
        rows=evaluate(rules,splits['validation'],representation)
        val_observations[representation]=summarize(rows,args.bootstrap)
        obs_rows[representation]=rows
        rows_file(Path('validation/observability')/f'{representation}.jsonl',rows)
    selected=select_observation(val_observations)
    save(Path('observation_selection.json'),{'selected':selected,'validation_only':True,
        'candidates':['O2','O3'],'tie_prefers':'O2','results':val_observations})
    print(f'Observation selected with fixed rules on validation: {selected}',flush=True)
    factory=lambda scenario:PilotEnv(scenario,representation=selected,condition='ORACLE')
    # Reward sanity checks run before training, without changing coefficients.
    from .reward_audit import audit_rewards
    audit=audit_rewards(splits['validation'][:10],selected)
    save(Path('reward_audit.json'),audit)
    if not audit['passed']:
        raise RuntimeError('Reward audit failed; training blocked')
    # Baseline failures and extraction discrepancies are diagnosed before learning.
    before_pipe=evaluate(rules,splits['validation'],selected,'PIPELINE')
    rows_file(Path('validation/pipeline_rules.jsonl'),before_pipe)
    save(Path('diagnostics_before_training.json'),{
        'oracle':diagnose_rows(obs_rows[selected]),'pipeline':diagnose_rows(before_pipe),
        'paired':compare_paired(obs_rows[selected],before_pipe)})
    models={};train_results={};demos={};curves=[]
    for seed in args.seeds:
        print(f'M0 supervised neural seed {seed}',flush=True)
        result=train_supervised(splits['train'],splits['validation'],env_factory=factory,
            seed=seed,lr=args.m0_lr,epochs=args.m0_epochs,batch_size=args.batch_size,eval_every=4)
        models[('M0',seed)]=result['policy'];demos[seed]=result['demos']
        path=out/'checkpoints'/f'seed_{seed}'/'M0';path.mkdir(parents=True,exist_ok=True)
        result['initial_policy'].save(path/'random_initial.pt')
        result['policy'].save(path/'selected.pt');result['last_policy'].save(path/'last.pt')
        serial={k:v for k,v in result.items() if k in ('metrics','curve','episode_log')}
        save(Path('training')/f'M0_seed{seed}.json',serial)
        train_results[f'M0_seed{seed}']=serial
        for point in result['curve']:curves.append({'method':'M0','seed':seed,'lr':args.m0_lr,**point})
    candidates={method:{} for method in ('M1','M2')}
    validation_scores={method:{} for method in ('M1','M2')}
    for lr in args.learning_rates:
        key=f'{lr:.6g}'
        for method in ('M1','M2'):
            candidates[method][key]={};validation_scores[method][key]=[]
            for seed in args.seeds:
                print(f'{method} seed {seed}, lr={lr}',flush=True)
                common=dict(env_factory=factory,seed=seed,lr=lr,optimizer_steps=args.optimizer_steps,
                    batch_size=args.batch_size,eval_every=args.eval_every)
                result=(train_reinforce(models[('M0',seed)],splits['train'],splits['validation'],gamma=1,**common)
                        if method=='M1' else train_additional_supervised(models[('M0',seed)],demos[seed],splits['validation'],**common))
                assert result['metrics']['initial_hash']==models[('M0',seed)].parameter_hash()
                assert result['metrics']['last_parameter_l2_delta']>0
                assert result['metrics']['nonzero_gradient_updates']>0
                path=out/'checkpoints'/f'seed_{seed}'/f'{method}_lr{key}';path.mkdir(parents=True,exist_ok=True)
                result['policy'].save(path/'selected.pt');result['last_policy'].save(path/'last.pt')
                candidates[method][key][seed]=result['policy']
                score=result['metrics']['after']['mean_return']
                validation_scores[method][key].append(score)
                serial={k:v for k,v in result.items() if k in ('metrics','curve','episode_log')}
                label=f'{method}_seed{seed}_lr{key}'
                save(Path('training')/(label+'.json'),serial);train_results[label]=serial
                for point in result['curve']:curves.append({'method':method,'seed':seed,'lr':lr,**point})
    chosen={method:max(validation_scores[method],key=lambda lr:sum(validation_scores[method][lr])/len(args.seeds)) for method in ('M1','M2')}
    for method in ('M1','M2'):
        for seed in args.seeds:models[(method,seed)]=candidates[method][chosen[method]][seed]
    freeze={'created_at':datetime.now(timezone.utc).isoformat(),'created_before_test':True,
        'observation':selected,'learning_rates':chosen,'validation_scores':validation_scores,
        'reward':asdict(RewardConfig()),'hyperparameters':config,
        'checkpoint_hashes':{str(p.relative_to(out)):sha(p) for p in (out/'checkpoints').rglob('*.pt')},
        'selected_parameter_hashes':{f'{m}_seed{s}':p.parameter_hash() for (m,s),p in models.items()},
        'meaningful_return_margin':.10,'test_used_for_selection':False}
    save(Path('freeze.json'),freeze)
    print('Policies, learning rates and reward frozen. Starting test.',flush=True)
    summaries={};test_rows={}
    for condition in ('ORACLE','PIPELINE'):
        summaries[condition]={};test_rows[condition]={}
        for (method,seed),policy in models.items():
            label=f'{method}_seed{seed}'
            print(f'Test {condition}/{label}',flush=True)
            rows=evaluate(policy,splits['test'],selected,condition,
                directory=out/'sqlite'/condition/label,retain=1)
            rows_file(Path('test')/condition/(label+'.jsonl'),rows)
            test_rows[condition][label]=rows;summaries[condition][label]=summarize(rows,args.bootstrap)
        rows=evaluate(rules,splits['test'],selected,condition,directory=out/'sqlite'/condition/'M3',retain=1)
        test_rows[condition]['M3']=rows;summaries[condition]['M3']=summarize(rows,args.bootstrap)
        rows_file(Path('test')/condition/'M3.jsonl',rows)
        for method in ('M0','M1','M2'):
            averaged=average_seeds([test_rows[condition][f'{method}_seed{s}'] for s in args.seeds])
            test_rows[condition][method]=averaged;summaries[condition][method]=summarize(averaged,args.bootstrap)
    # Observation ablation on test uses exactly the same strong fixed rule policy.
    observations={selected:test_rows['ORACLE']['M3']}
    for representation in ('O1','O2','O3'):
        if representation not in observations:observations[representation]=evaluate(rules,splits['test'],representation)
        rows_file(Path('test/observability')/(representation+'.jsonl'),observations[representation])
    comparisons={condition:{f'{a}-{b}':paired(test_rows[condition][a],test_rows[condition][b],args.bootstrap)
        for a,b in (('M1','M0'),('M1','M2'),('M1','M3'))} for condition in ('ORACLE','PIPELINE')}
    comparisons['observability']={f'{a}-{b}':paired(observations[a],observations[b],args.bootstrap) for a,b in (('O2','O1'),('O3','O2'))}
    comparisons['pipeline_minus_oracle']={method:paired(test_rows['PIPELINE'][method],test_rows['ORACLE'][method],args.bootstrap) for method in ('M0','M1','M2','M3')}
    diagnoses={condition:{method:diagnose_rows(rows) for method,rows in test_rows[condition].items()
        if method=='M3' or '_seed' in method} for condition in ('ORACLE','PIPELINE')}
    save(Path('metrics.json'),{'test':summaries,'observations':{r:summarize(v,args.bootstrap) for r,v in observations.items()},
        'comparisons':comparisons,'selected_observation':selected,'selected_learning_rates':chosen})
    save(Path('error_diagnostics.json'),diagnoses)
    save(Path('training_curves.json'),curves)
    if args.llm_users and args.llm_base and os.environ.get(args.llm_token_env):
        from .llm_comparator import run_comparator
        llm=run_comparator(splits['test'][:args.llm_users],selected,out,args.llm_base,args.llm_token_env,args.llm_workers,args.bootstrap)
        save(Path('llm_status.json'),llm)
        if llm['status']=='EXECUTED':
            combined=json.loads((out/'metrics.json').read_text())
            llm_summaries=json.loads((out/'llm/metrics.json').read_text())
            for condition,value in llm_summaries.items():
                if value is None:continue
                combined['test'][condition]['M4']=value
                llm_rows=[json.loads(line) for line in (out/'llm'/(condition+'.jsonl')).read_text().splitlines()]
                ids={r['user_id'] for r in llm_rows}
                matched=[r for r in test_rows[condition]['M3'] if r['user_id'] in ids]
                combined['comparisons'][condition]['M4-M3']=paired(llm_rows,matched,args.bootstrap)
            save(Path('metrics.json'),combined)
    else:
        save(Path('llm_status.json'),{'status':'PENDING','reason':'No inference enabled and authenticated in this command; prior GLM pilot retained separately.',
            'importance':'The thesis concerns LLM assistants; an external policy is not LLM weight training.'})
    save(Path('resources.json'),{'elapsed_seconds':monotonic()-start,'device':'CPU','torch':torch.__version__,
        'python':platform.python_version(),'threads':torch.get_num_threads(),'gpu_used':False,
        'main_methods':['M0','M1','M2','M3'],'llm_weights_trained':False,
        'selected_parameter_count':models[('M0',args.seeds[0])].parameter_count,
        'train_runs':3*len(args.seeds)+2*len(args.seeds)*(len(args.learning_rates)-1),
        'neural_rl_runs':len(args.seeds)*len(args.learning_rates)})
    assert all(sha(Path(p))==h for p,h in json.loads((out/'historical_hashes.json').read_text()).items())
    from .report import generate_report
    generate_report(out)
    print(f'COMPLETE: {out}',flush=True)


if __name__=='__main__':
    main()
