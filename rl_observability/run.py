"""SPEC-RL-004: bounded CPU experiment with validation-only selection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
from time import monotonic

from .scenarios import generate_splits
from .policies import LastWriteWins, HistoricalClarificationRules, QLearningPolicy
from .learning import train_supervised, train_q_learning
from .evaluation import evaluate, summarize, paired, average_seed_rows


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+'\n')


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe_llm():
    """Read-only resource check; do not serialize private service addresses."""
    from agents.client import http_json, TechnicalError
    base = os.environ.get('LLM_BASE_URL', '')
    if not base and Path('SPEC_CODEX.md').exists():
        match = re.search(r'http://[0-9.]+:[0-9]+/v1', Path('SPEC_CODEX.md').read_text())
        base = match.group() if match else ''
    if not base:
        return {'status': 'NOT_RUN', 'reason': 'No LLM resource configured', 'inference_calls': 0}
    try:
        response = http_json(base.rstrip('/')+'/models', timeout=3,
                             token=os.environ.get('VLLM_API_KEY',''))
        return {'status':'AVAILABLE_NOT_EVALUATED','models':[r['id'] for r in response.get('data',[])],
                'inference_calls':0,'reason':'Resource available; separate prompting protocol required'}
    except TechnicalError as error:
        return {'status':'NOT_RUN','reason':'Configured LLM resource inaccessible',
                'error_kind':error.kind,'timeout_seconds':3,'inference_calls':0}


def run(args):
    if min(args.train_users, args.validation_users, args.test_users) < 5:
        raise ValueError('Each split must cover all five families')
    if args.episodes < 1 or not 0 < args.max_seconds <= 300 or args.bootstrap < 100:
        raise ValueError('Invalid bounded compute/bootstrap budget')
    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('At least three distinct seeds are required')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    config = vars(args).copy()
    config.update(started_at=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                  hardware=platform.machine(), algorithm='tabular Q-learning',
                  representation_selection='B2 validation mean return; ties prefer simpler O1 then O2 then O3',
                  test_used_for_selection=False, unit='synthetic_user',
                  clarification_budget_session_2=1,
                  pilot_authorization='SPEC-RL-004 section 10: trainability pilot independent of superiority')
    write_json(out/'config.json', config)
    llm_probe = probe_llm()
    write_json(out/'llm_probe.json',llm_probe)
    snapshots = out/'source_snapshot'
    snapshots.mkdir()
    for directory in ('rl_observability','rl_longitudinal','simulator','memory'):
        for source in sorted(Path(directory).glob('*.py')):
            dest = snapshots/source
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,dest)
    write_json(out/'source_hashes.json', {str(p.relative_to(snapshots)):digest(p)
                for p in sorted(snapshots.rglob('*.py'))})
    datasets = generate_splits(args.train_users,args.validation_users,args.test_users,args.seed)
    for split, scenarios in datasets.items():
        write_json(out/'evaluator_private'/f'{split}.json',scenarios)
    write_json(out/'dataset_manifest.json', {split:{'users':len(scenarios),
        'sha256':digest(out/'evaluator_private'/f'{split}.json'),
        'families':sorted({s['family'] for s in scenarios}),
        'abstract_templates_shared':True,'surface_template_partition':split}
        for split,scenarios in datasets.items()})
    policies, training = {}, {}
    validation, summaries = {}, {}
    for representation in ('O1','O2','O3'):
        candidates={'B1':LastWriteWins(),'B2':HistoricalClarificationRules(), 'B4_INITIAL':QLearningPolicy()}
        for seed in args.seeds:
            print(f'{representation}: supervised + Q-learning seed {seed}',flush=True)
            sft=train_supervised(datasets['train'],datasets['validation'],observation=representation,seed=seed)
            rl=train_q_learning(datasets['train'],datasets['validation'],observation=representation,
                seed=seed,episodes=args.episodes,max_seconds=args.max_seconds,block_size=args.block_size)
            for prefix,result in (('B3',sft),('B4',rl)):
                key=f'{prefix}_seed{seed}'
                candidates[key]=result['policy']
                serial={k:v for k,v in result.items() if k!='policy'}
                write_json(out/'training'/representation/f'{key}.json',serial)
                policy=result['policy']
                if hasattr(policy,'to_dict'):
                    write_json(out/'checkpoints'/representation/f'{key}.json',policy.to_dict())
                training[f'{representation}/{key}']=serial
        policies[representation]=candidates
        validation[representation]={}
        summaries[representation]={}
        for method,policy in candidates.items():
            rows=evaluate(policy,datasets['validation'],representation)
            validation[representation][method]=rows
            summaries[representation][method]=summarize(rows,args.bootstrap)
            write_rows(out/'validation'/representation/f'{method}.jsonl',rows)
    write_json(out/'validation'/'metrics.json',summaries)
    def mean_return(rows):
        return sum(r['return'] for r in rows)/len(rows)
    selected=max(('O1','O2','O3'),key=lambda rep:mean_return(validation[rep]['B2']))
    best_nonrl=max((k for k in policies[selected] if k.startswith(('B1','B2','B3'))),
                   key=lambda k:mean_return(validation[selected][k]))
    best_rl=max((k for k in policies[selected] if k.startswith('B4_seed')),
                key=lambda k:mean_return(validation[selected][k]))
    from .environment import RewardConfig
    from dataclasses import asdict
    freeze={'created_before_test':True,'selected_observation':selected,
        'selected_non_rl':best_nonrl,'selected_rl':best_rl,'reward':asdict(RewardConfig()),
        'hyperparameters':config,'criterion':'validation return only; deterministic insertion-order ties',
        'checkpoint_hashes':{str(p.relative_to(out)):digest(p) for p in sorted((out/'checkpoints').rglob('*.json'))}}
    write_json(out/'freeze.json',freeze)
    print(f'Frozen: {selected}, comparator {best_nonrl}, RL {best_rl}. Starting test.',flush=True)
    test, test_summaries = {}, {}
    for representation,candidates in policies.items():
        test[representation],test_summaries[representation]={},{}
        for method,policy in candidates.items():
            rows=evaluate(policy,datasets['test'],representation)
            test[representation][method]=rows
            test_summaries[representation][method]=summarize(rows,args.bootstrap)
            write_rows(out/'test'/representation/f'{method}.jsonl',rows)
        for prefix in ('B3','B4'):
            averaged=average_seed_rows([test[representation][f'{prefix}_seed{s}'] for s in args.seeds])
            test[representation][prefix+'_seed_mean']=averaged
            test_summaries[representation][prefix+'_seed_mean']=summarize(averaged,args.bootstrap)
    write_json(out/'test'/'metrics.json',test_summaries)
    comparisons={
        'observability_fixed_B2':{f'{a}-{b}':paired(test[a]['B2'],test[b]['B2'],args.bootstrap)
                                  for a,b in (('O2','O1'),('O3','O2'))},
        'learning_selected_observation':{
            f'{a}-{b}':paired(test[selected][a],test[selected][b],args.bootstrap)
            for a,b in ((best_rl,best_nonrl),(best_rl,'B4_INITIAL'),
                        ('B4_seed_mean','B3_seed_mean'),('B4_seed_mean','B2'))},
        'pipeline_B2':{}}
    # Extraction is a separate diagnostic, never used to select models or tune rewards.
    for representation in ('O1','O2','O3'):
        rows=evaluate(policies[representation]['B2'],datasets['test'],representation,'PIPELINE')
        write_rows(out/'pipeline'/f'{representation}_B2.jsonl',rows)
        comparisons['pipeline_B2'][representation]={'metrics':summarize(rows,args.bootstrap),
            'pipeline_minus_visible_structured':paired(rows,test[representation]['B2'],args.bootstrap)}
    write_json(out/'comparisons.json',comparisons)
    write_json(out/'resources.json',{'elapsed_seconds':monotonic()-started,
        'device':'CPU','gpu_used':False,'network_inference_calls':0,
        'B5':llm_probe,
        'training_runs':len(args.seeds)*3,'supervised_runs':len(args.seeds)*3,
        'bootstrap_repetitions':args.bootstrap})
    from .report import generate_report
    generate_report(out)
    print(f'Completed: {out}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='reports/rl_observability/run_001')
    p.add_argument('--train-users',type=int,default=200)
    p.add_argument('--validation-users',type=int,default=100)
    p.add_argument('--test-users',type=int,default=100)
    p.add_argument('--seed',type=int,default=41004)
    p.add_argument('--seeds',type=int,nargs='+',default=[101,202,303])
    p.add_argument('--episodes',type=int,default=1000)
    p.add_argument('--max-seconds',type=float,default=60)
    p.add_argument('--block-size',type=int,default=100)
    p.add_argument('--bootstrap',type=int,default=1000)
    run(p.parse_args())


if __name__=='__main__':
    main()
