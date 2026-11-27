"""One-seed controller RL: balanced gates and local simulated action credit.

No action demonstrations or SFT. The language encoder and the separate reader
remain frozen. Pilot review precedes optimization; validation selection precedes
one fresh reserved evaluation. Historical budget experiments are untouched.
"""
from copy import deepcopy
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev
import argparse
import gc
import json
import math
import random
import shutil
import time

from research.budget_memory_experiment import (
    BASE, SEED, read, rows, append, balanced_cohort, aggregate, paired, feedback_group,
    fresh_reader, ScriptedManager, fresh_manager, better_selection)
from research.budget_memory_env import run_episode, score_trace
from research.budget_memory_training import parameter_hashes
from research.decision_memory_credit import collect_counterfactual_group, enumerate_credit
from research.decision_memory_policy import DecisionMemoryPolicy
from research.decision_memory_training import optimize_counterfactual, module_hash, enumerated_loss
from research.decision_memory_metrics import metrics, sampling_metrics
from research.src.artifacts import digest, file_hash, now, write_json


def sources():
    root = Path(__file__).resolve().parent
    files = [*root.glob('decision_memory_*.py'), *root.glob('budget_memory_*.py'),
             root.parent/'llm_local_001/model.py', root/'src/policy/llm_controller.py']
    return {str(p): file_hash(p) for p in sorted(set(files))}


def check(plan):
    if sources() != plan['source_sha256']:
        raise ValueError('Frozen prospective source changed')
    for path, sha in plan['data_sha256'].items():
        if file_hash(path) != sha:
            raise ValueError('Frozen bank changed')
    for relative, sha in plan['base_artifacts_sha256'].items():
        if file_hash(BASE/relative) != sha:
            raise ValueError('Provider artifact changed')


def load_bank(path):
    path = Path(path).resolve()
    manifest = read(path/'MANIFEST.json')
    bank = {role: rows(entry['path']) for role, entry in manifest['files'].items()}
    hashes = {entry['path']: entry['sha256'] for entry in manifest['files'].values()}
    hashes[str(path/'MANIFEST.json')] = file_hash(path/'MANIFEST.json')
    if any(file_hash(f) != h for f, h in hashes.items()):
        raise ValueError('Invalid bank artifact')
    return bank, hashes


def release(actor=None, reader=None):
    import torch
    if actor is not None:
        actor.model.to('cpu')
        if hasattr(actor, 'encoder_backend'):
            actor.encoder_backend.model.to('cpu')
    if reader is not None:
        reader.backend.model.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()


def controller(trainable=False):
    return DecisionMemoryPolicy(BASE, device='cuda', precision='bfloat16', seed=SEED, trainable=trainable)


class HierRandom:
    def decide(self, observation, candidates, *, seed, sample, decision_id):
        rng = random.Random(seed)
        chosen = candidates[0] if len(candidates) == 1 or rng.random() < .5 else rng.choice(candidates[1:])
        return {'action': chosen, 'metadata': {'decision_id': decision_id, 'hierarchical_random': True}}


def evaluate_policy(actor, reader, episodes, output, method):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    result = []
    for index, episode in enumerate(episodes):
        policy = (HierRandom() if actor == 'hier_random' else ScriptedManager(actor)) if isinstance(actor,str) else actor
        trace = run_episode(policy, reader, episode, SEED*100000+index, output/f'episode_{index:03d}',
                            sample=False, namespace=f'{method}:{output.name}:{index}')
        row = dict(episode_id=episode['episode_id'],root_id=episode['root_id'],family=episode['family'],
                   method=method,metrics=metrics(trace,episode))
        append(output/'metrics.jsonl',row)
        result.append(row)
        print('evaluation',method,index+1,'/',len(episodes),flush=True)
    panel = aggregate(result)
    panel['reader_profile'] = reader.backend.policy_profile()
    panel['reader_frozen'] = not any(p.requires_grad for p in reader.backend.model.parameters())
    write_json(output/'summary.json',panel)
    return panel


def pilot(run, data):
    import torch
    from llm_local_001.model import artifact_hashes
    run = Path(run).resolve()
    if run.exists(): raise FileExistsError(run)
    bank, data_hashes = load_bank(data)
    run.mkdir(parents=True)
    plan = dict(created_at=now(),seed=SEED,sft_updates=0,trainable_scope='contextual_controller_head_only',
                source_sha256=sources(),data_sha256=data_hashes,base_artifacts_sha256=artifact_hashes(BASE),
                cohort=balanced_cohort(bank['train'],5),pilot_groups=10,pilot_branches=30,
                state_sampling='diagnostic both phases; no optimization; TRAIN only',
                gates=dict(min_task_variable_groups=1,min_positive_supported_queries=2,
                           min_rules_reader_query_success=.6,require_both_gates_explored=True,
                           require_frozen_weights_and_zero_steps=True))
    write_json(run/'PILOT_PLAN.json',plan)
    project=Path(__file__).resolve().parents[1]
    for path in plan['source_sha256']:
        target=run/'source_snapshot'/Path(path).relative_to(project)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    actor,reader=controller(),fresh_reader()
    head_before=module_hash(actor.head)
    encoder_before=parameter_hashes(actor.encoder_backend)
    reader_before=parameter_hashes(reader.backend)
    write_json(run/'POLICY_PROFILE.json',actor.policy_profile())
    groups=[];visits=[];all_traces=[];started=time.monotonic()
    try:
        for index,episode in enumerate(plan['cohort']):
            for phase in ('ask','write'):
                group_index=len(groups)
                start=time.monotonic()
                group=collect_counterfactual_group(actor,reader,episode,SEED*100000+group_index,
                                                  run/'pilot'/f'group_{group_index:03d}',phase)
                model_seconds=time.monotonic()-start
                rewards=feedback_group(run,f'pilot_group_{group_index:03d}',group['branches'])
                credit=enumerate_credit(group,rewards)
                utilities=[x['mean_utility'] for x in rewards]
                row=dict(group=group_index,phase=phase,session=group['target']['session_index'],
                         credit=credit,utilities=utilities,task_std=pstdev(utilities),
                         model_seconds=model_seconds,total_seconds=time.monotonic()-start)
                append(run/'PILOT_GROUPS.jsonl',row);groups.append(row)
                visits.append(sampling_metrics(group['visitation']));all_traces.extend(group['branches'])
                print('pilot',group_index,phase,'Q',credit['full_returns'],'task_std',row['task_std'],flush=True)
        rules=evaluate_policy('semantic',reader,plan['cohort'],run/'pilot_rules','Public-rules-pilot')
        torch.cuda.reset_peak_memory_stats()
        actor.enable_training()
        ref=deepcopy(actor.head).requires_grad_(False)
        features=group['target']['metadata']['feature_vector']
        lp=actor.distribution(features,group['target']['candidates'])
        with torch.no_grad(): rp=actor.distribution(features,group['target']['candidates'],head=ref)
        smoke_q=torch.tensor([0.,1.,-.1],device=actor.device)
        losses=enumerated_loss(lp,rp,smoke_q)
        losses[0].backward()
        finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in actor.head.parameters())
        actor.head.zero_grad(set_to_none=True);actor.head.requires_grad_(False);actor.training_enabled=False
        positive=sum(score_trace(t)['query_success']*2 for t in all_traces)
        values=dict(task_variable_groups=sum(x['task_std']>0 for x in groups),
                    positive_supported_queries=positive,rules_reader_query_success=rules['metrics']['query_success'],
                    sampled_ask=sum(v.get('sampled_ask_count',v.get('ask_count',0)) for v in visits),
                    sampled_noop=sum(v.get('sampled_noop_count',v.get('noop_count',0)) for v in visits),
                    head_unchanged=module_hash(actor.head)==head_before,
                    encoder_unchanged=parameter_hashes(actor.encoder_backend)==encoder_before,
                    reader_unchanged=parameter_hashes(reader.backend)==reader_before,finite_head_gradient=finite)
        # Mechanical visitation counts avoid relying on a plotting-key spelling.
        trajectories=[read(run/'pilot'/f'group_{i:03d}'/'visitation'/'trace.json') for i in range(10)]
        values['sampled_ask']=sum(d['phase']=='ask' and d['action']!='PROCEED' for t in trajectories for d in t['decisions'])
        values['sampled_noop']=sum(d['phase']=='write' and d['action']=='NOOP' for t in trajectories for d in t['decisions'])
        passed=(values['task_variable_groups']>=1 and positive>=2 and values['rules_reader_query_success']>=.6
                and values['sampled_ask']>0 and values['sampled_noop']>0
                and all(values[k] for k in ('head_unchanged','encoder_unchanged','reader_unchanged','finite_head_gradient')))
        check(plan)
        result=dict(status='passed' if passed else 'failed_feasibility',gates=values,groups=groups,
                    rules=rules,elapsed_seconds=time.monotonic()-started,optimizer_steps=0,sft_updates=0,
                    peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
                    scope='Real fork outcomes; synthetic Q above only checks head autograd, zero updates')
        write_json(run/'PILOT_RESULT.json',result);print('PILOT_RESULT',result['status'],values,flush=True)
    finally:release(actor,reader)


def select_key(panel):
    return (panel['metrics']['net_return'],panel['metrics']['query_success'],-panel['metrics']['ask_count'])


def train(run,data,groups=64,lr=.003,budget_seconds=3600):
    import torch
    run=Path(run).resolve();pilot_plan=read(run/'PILOT_PLAN.json')
    if groups not in (32,64) or not math.isfinite(lr) or lr<=0 or budget_seconds<=0:
        raise ValueError('Invalid prospective one-seed controller schedule')
    if (read(run/'PILOT_RESULT.json')['status']!='passed' or read(run/'PILOT_REVIEW.json')['status']!='passed'
            or read(run/'PILOT_REVIEW.json')['pilot_result_sha256']!=file_hash(run/'PILOT_RESULT.json')):
        raise ValueError('Reviewed pilot must pass before optimization')
    check(pilot_plan);bank,_=load_bank(data)
    output=run/'optimizer';output.mkdir(exist_ok=False)
    rng=random.Random(SEED+8191)
    phases=[rng.choice(('ask','write')) for _ in range(groups)]
    plan=dict(created_at=now(),seed=SEED,sft_updates=0,max_updates=groups,learning_rate=lr,
              kl_coefficient=.02,local_KL_regularizer=True,budget_seconds=budget_seconds,
              phase_sampling='independent uniform Bernoulli, separate seeded RNG',phases=phases,
              phase_counts=dict(Counter(phases)),cohort=balanced_cohort(bank['train'],groups),
              checkpoints=[0,groups//2,groups],trainable_scope='contextual_head_only',
              selection='Max validation net_return, ties query_success then fewer ASK then earliest',
              primary_claim='Head0 improvement alone is insufficient; also compare original LM, FIFO, public rules',
              terminal_diagnostic='Predeclared from start; terminal never replaces validation selection',
              **{key:pilot_plan[key] for key in ('source_sha256','data_sha256','base_artifacts_sha256')})
    write_json(run/'TRAIN_PLAN.json',plan)
    actor,reader=controller(True),fresh_reader();ref=deepcopy(actor.head).requires_grad_(False)
    initial=dict(head=module_hash(actor.head),reference=module_hash(ref),
                 encoder=parameter_hashes(actor.encoder_backend),reader=parameter_hashes(reader.backend))
    write_json(run/'INITIALIZATION.json',initial)
    actor.save_checkpoint(output/'step_000000')
    optimizer=torch.optim.Adam(actor.head.parameters(),lr=lr)
    journal=output/'progress.jsonl';append(journal,dict(stage='started',step=0,plan=plan))
    history=[];elapsed_training=0.;start=time.monotonic()
    try:
        baseline=evaluate_policy(actor,reader,bank['validation'],run/'validation/step_000000','Head0')
        best=baseline;best_step=0
        append(journal,dict(stage='validation_completed',step=0,metrics=baseline['metrics']))
        for index,(episode,phase) in enumerate(zip(plan['cohort'],phases,strict=True),start=1):
            if elapsed_training>=budget_seconds:raise TimeoutError('Declared training budget exhausted')
            then=time.monotonic()
            group=collect_counterfactual_group(actor,reader,episode,SEED*1000000+index,
                                              run/'train_groups'/f'group_{index:03d}',phase)
            rewards=feedback_group(run,f'train_group_{index:03d}',group['branches'])
            credit=enumerate_credit(group,rewards)
            stats=optimize_counterfactual(actor,ref,group,rewards,optimizer,beta=.02)
            elapsed_training+=time.monotonic()-then
            row=dict(stage='training',step=index,phase=phase,**stats,elapsed_seconds=time.monotonic()-start,
                     resource_counters=dict(updates=index,optimizer_steps=index,outcome_episodes=index*3,
                                            visitation_episodes=index,sft_updates=0,training_seconds=elapsed_training))
            append(journal,row);history.append(row)
            append(output/'credit_diagnostics.jsonl',dict(step=index,phase=phase,credit=credit,
                                                       target_session=group['target']['session_index']))
            behavior=sampling_metrics(group['visitation'])
            vals={f'behavior/{k}':v for k,v in behavior.items() if isinstance(v,(int,float))}
            vals.update({'credit/expected_return':stats['baseline'],
                         'credit/task_std':pstdev(r['mean_utility'] for r in rewards),
                         'credit/return_std':pstdev(r['episodic_reward'] for r in rewards),
                         f'credit/{phase}_positive_probability':1-stats['probabilities_before'][0]})
            append(output/'training_metrics.jsonl',dict(step=index,values=vals))
            print('training',index,phase,'Q',stats['returns'],'loss',stats['loss'],flush=True)
            if index in plan['checkpoints']:
                actor.save_checkpoint(output/f'step_{index:06d}')
                panel=evaluate_policy(actor,reader,bank['validation'],run/'validation'/f'step_{index:06d}',f'Head-step-{index}')
                append(journal,dict(stage='validation_completed',step=index,metrics=panel['metrics']))
                if better_selection(panel['metrics'],best['metrics']):best,best_step=panel,index
        final=dict(head=module_hash(actor.head),reference=module_hash(ref),
                   encoder=parameter_hashes(actor.encoder_backend),reader=parameter_hashes(reader.backend))
        if initial['encoder']!=final['encoder'] or initial['reader']!=final['reader'] or initial['reference']!=final['reference']:
            raise ValueError('Frozen encoder, reader or reference changed')
        check(plan)
        result=dict(status='completed',seed=SEED,sft_updates=0,best_step=best_step,history=history,
                    initial=initial,final=final,head_changed=initial['head']!=final['head'],
                    resource_counters=dict(updates=groups,optimizer_steps=groups,outcome_episodes=groups*3,
                                           visitation_episodes=groups,training_seconds=elapsed_training,
                                           total_seconds=time.monotonic()-start))
        write_json(output/'training.json',result)
        write_json(run/'SELECTION.json',dict(created_at=now(),selected_before_evaluation=True,selection_split='validation',
                   selected_step=best_step,selected_checkpoint=str(output/f'step_{best_step:06d}'),
                   initial_checkpoint=str(output/'step_000000'),terminal_checkpoint=str(output/f'step_{groups:06d}'),
                   source_sha256=plan['source_sha256'],data_sha256=plan['data_sha256']))
        append(journal,dict(stage='completed',step=groups,selected_step=best_step))
        print('training_completed selected',best_step,flush=True)
    finally:release(actor,reader)


def evaluate(run,data):
    run=Path(run).resolve();plan=read(run/'TRAIN_PLAN.json');check(plan)
    selection=read(run/'SELECTION.json');selection_sha=file_hash(run/'SELECTION.json')
    bank,_=load_bank(data);reader=fresh_reader();panels={};paths={}
    try:
        for method,path in (('Head0',selection['initial_checkpoint']),('RL-selected',selection['selected_checkpoint']),
                            ('RL-terminal',selection['terminal_checkpoint'])):
            if path in paths:panels[method]={**deepcopy(panels[paths[path]]),'identical_to':paths[path]};continue
            actor=DecisionMemoryPolicy.from_checkpoint(path,device='cuda',trainable=False)
            try:panels[method]=evaluate_policy(actor,reader,bank['evaluation'],run/'evaluation'/method,method)
            finally:release(actor);del actor
            paths[path]=method
        actor=fresh_manager()
        try:panels['LM-frozen']=evaluate_policy(actor,reader,bank['evaluation'],run/'evaluation/LM-frozen','LM-frozen')
        finally:release(actor);del actor
        for method,mode in (('FIFO','fifo'),('HierRandom','hier_random'),('Public-rules','semantic')):
            panels[method]=evaluate_policy(mode,reader,bank['evaluation'],run/'evaluation'/method,method)
        if selection_sha!=file_hash(run/'SELECTION.json'):raise ValueError('Selection changed during test')
        check(plan)
        result=dict(status='completed',seed=SEED,sft_updates=0,selected_step=selection['selected_step'],
                    panels=panels,paired_vs_LM={k:paired(panels['LM-frozen'],v) for k,v in panels.items() if k!='LM-frozen'},
                    paired_vs_Head0={k:paired(panels['Head0'],v) for k,v in panels.items() if k in ('RL-selected','RL-terminal')},
                    selection_sha256=selection_sha,training=read(run/'optimizer/training.json')['resource_counters'],
                    terminal_role='Predeclared descriptive terminal; not used to change selection')
        write_json(run/'COMPARISON.json',result)
        print('comparison_completed',{k:v['metrics']['query_success'] for k,v in panels.items()},flush=True)
    finally:release(reader=reader)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('pilot','train','evaluate'),required=True)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--groups',type=int,default=64);parser.add_argument('--learning-rate',type=float,default=.003)
    parser.add_argument('--budget-seconds',type=int,default=3600)
    args=parser.parse_args()
    if args.stage=='pilot':pilot(args.run,args.data)
    elif args.stage=='train':train(args.run,args.data,args.groups,args.learning_rate,args.budget_seconds)
    else:evaluate(args.run,args.data)


if __name__=='__main__':main()
