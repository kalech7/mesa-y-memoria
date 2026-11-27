"""Prospective stable controller iteration; historical session runs remain intact.

Same task, seed, macro estimator and comparator interface. This version adds
nonaffine hidden LayerNorm and FP64 categorical arithmetic, lowers the declared
head learning rate, and increases Adam epsilon. No SFT or reserved-run tuning.
"""
from copy import deepcopy
from pathlib import Path
from statistics import pstdev
import argparse
import math
import shutil
import time

from research.decision_memory_experiment import (
    BASE, SEED, read, append, balanced_cohort, fresh_reader, better_selection,
    load_bank, release, evaluate_policy, parameter_hashes, feedback_group,
    fresh_manager, paired, sources as legacy_sources, check as legacy_check,
)
from research.decision_memory_session_credit import collect_session_group, enumerate_session_credit
from research.decision_memory_session_training import joint_logprobs, session_loss
from research.stable_controller_training import optimize_session
from research.decision_memory_training import module_hash
from research.decision_memory_metrics import sampling_metrics
from research.budget_memory_env import score_trace
from research.src.artifacts import file_hash, now, write_json
from research.stable_controller_policy import StableControllerPolicy

VERSION = 'stable_controller_experiment_v1'
LEARNING_RATE = .0003
ADAM_EPSILON = 1e-6


def sources():
    result=legacy_sources()
    root=Path(__file__).resolve().parent
    for path in sorted(root.glob('stable_controller_*.py')):
        result[str(path)]=file_hash(path)
    return result


def check(plan):
    if sources()!=plan['source_sha256']:
        raise ValueError('Frozen prospective stable controller sources changed')
    for path,sha in plan['data_sha256'].items():
        if file_hash(path)!=sha:raise ValueError('Frozen fresh bank changed')
    for relative,sha in plan['base_artifacts_sha256'].items():
        if file_hash(BASE/relative)!=sha:raise ValueError('Frozen provider artifact changed')


def controller(trainable=False):
    return StableControllerPolicy(BASE,device='cuda',precision='bfloat16',seed=SEED,trainable=trainable)


def _head_activation_diagnostics(actor,group):
    """Tiny-head telemetry on cached public targets; no encoder/reader call."""
    import torch
    targets=[group['ask_target'],*group['write_targets']]
    features=torch.tensor([t['metadata']['feature_vector'] for t in targets],
                          device=actor.device,dtype=torch.float32)
    with torch.no_grad():
        pre=actor.head[0](features)
        normalized=actor.head[1](pre)
        hidden=actor.head[2](normalized)
        saturated=hidden.abs()>.999
    return dict(hidden_saturation_fraction=float(saturated.float().mean()),
        hidden_saturation_fraction_max_state=float(saturated.float().mean(dim=1).max()),
        firstlinear_preactivation_absmean=float(pre.abs().mean()),
        firstlinear_preactivation_absmax=float(pre.abs().max()),
        firstlinear_preactivation_min=float(pre.min()),firstlinear_preactivation_max=float(pre.max()),
        post_layernorm_variance_max=float(normalized.var(dim=1,unbiased=False).max()),
        cached_target_count=len(targets),encoder_calls=0,reader_calls=0)



def _session_diagnostics(group,rewards):
    """Post-outcome diagnostics, never a selector or an expert action target."""
    utility=[float(row['mean_utility']) for row in rewards]
    pairs=group['action_pairs']
    ask_values=[]
    write_variations=[]
    for ask_index,probabilities in enumerate(group['behavior_write_probs']):
        indices=[pair['branch_index'] for pair in pairs if pair['ask_index']==ask_index]
        if len(indices)!=3:
            raise ValueError('Every public ASK branch requires all three WRITE outcomes')
        values=[utility[index] for index in indices]
        ask_values.append(sum(probability*value for probability,value in zip(probabilities,values)))
        write_variations.append(pstdev(values))
    expected={(i,j) for i in range(len(group['ask_target']['candidates'])) for j in range(3)}
    complete={(pair['ask_index'],pair['write_index']) for pair in pairs}==expected and len(pairs)==len(expected)
    return dict(task_std=pstdev(utility),return_std=pstdev(float(row['episodic_reward']) for row in rewards),
                ask_expected_conditional_utilities=ask_values,ask_task_std=pstdev(ask_values),
                write_conditional_task_stds=write_variations,
                any_write_task_variation=any(value>0 for value in write_variations),
                complete_macro_enumeration=complete,branch_count=len(group['branches']),
                query_count=sum(len(branch['queries']) for branch in group['branches']))


def pilot(run,data,inherit_pilot):
    """Inherit unchanged outcome feasibility; independently review new numerics.

    Loads only the frozen backends and already cached public TRAIN vectors.
    There is no new LLM/reader forward, no synthetic rollout and no optimizer.
    A reviewer must sign this run's resulting PILOT_RESULT before training.
    """
    import torch
    from llm_local_001.model import artifact_hashes
    run=Path(run).resolve();inherited=Path(inherit_pilot).resolve()
    if run.exists():raise FileExistsError(run)
    inherited_plan=read(inherited/'PILOT_PLAN.json')
    inherited_result=read(inherited/'PILOT_RESULT.json')
    inherited_review=read(inherited/'PILOT_REVIEW.json')
    if (inherited_plan.get('version')!='decision_memory_session_experiment_v1'
            or inherited_result.get('optimizer_steps')!=0 or inherited_result.get('sft_updates')!=0
            or inherited_result['status']!='passed' or inherited_review['status']!='passed'
            or inherited_review['pilot_result_sha256']!=file_hash(inherited/'PILOT_RESULT.json')):
        raise ValueError('Reviewed passing session pilot required for inheritance')
    legacy_check(inherited_plan)
    project=Path(__file__).resolve().parents[1]
    inherited_evidence={}
    for path,sha in inherited_plan['source_sha256'].items():
        snapshot=inherited/'source_snapshot'/Path(path).relative_to(project)
        if file_hash(snapshot)!=sha:raise ValueError('Inherited source snapshot mismatch')
        inherited_evidence[str(snapshot)]=sha
    for path in sorted([*inherited.glob('pilot/group_*/group.json'),*inherited.glob('pilot/group_*/branch_*/trace.json'),
                        *inherited.glob('judge_queue/pilot_group_*.request.json'),*inherited.glob('judge_queue/pilot_group_*.response.json'),
                        inherited/'PILOT_GROUPS.jsonl']):
        inherited_evidence[str(path)]=file_hash(path)
    if len(list(inherited.glob('pilot/group_*/branch_*/trace.json')))!=inherited_result['outcome_episodes']:
        raise ValueError('Inherited real-outcome trace inventory mismatch')
    if len(list(inherited.glob('judge_queue/pilot_group_*.response.json')))!=10:
        raise ValueError('All ten macro feedback groups must be preserved')
    _,data_hashes=load_bank(data)
    if data_hashes!=inherited_plan['data_sha256']:
        raise ValueError('Outcome inheritance requires exactly the same data artifacts')
    run.mkdir(parents=True)
    inherited_artifacts={name:file_hash(inherited/name) for name in
                         ('PILOT_PLAN.json','PILOT_RESULT.json','PILOT_REVIEW.json')}
    plan=dict(version=VERSION,created_at=now(),seed=SEED,sft_updates=0,
        trainable_scope='fresh_contextual_head_only',source_sha256=sources(),data_sha256=data_hashes,
        base_artifacts_sha256=artifact_hashes(BASE),cohort=inherited_plan['cohort'],
        pilot_kind='inherited_outcome_feasibility_plus_zero_update_technical_check',
        inherited_pilot=str(inherited),inherited_artifact_sha256=inherited_artifacts,
        inherited_evidence_sha256=inherited_evidence,
        inherited_outcome_episodes=inherited_result['outcome_episodes'],
        inherited_terminal_queries=inherited_result['terminal_queries'],new_outcome_episodes=0,new_terminal_queries=0,
        prior_development='Run002 TRAIN/VAL exposed hidden tanh saturation; no TEST002 opened or used to design this correction.',
        changes=['Nonaffine hidden LayerNorm(64) before tanh','FP64 categorical log probabilities and local loss',
                 'Fresh seed101 initialization; learning_rate0.0003; Adam epsilon1e-6'],
        inheritance_rationale='Initial output logits remain exactly zero and action distributions match in law; no claim of bitwise sampled trajectories under FP64 multinomial. Environment, dataset, reader, reward and macro collector are unchanged.',
        technical_gates=['FP64_normalized_probabilities','same_initial_joint_distribution','hidden_LayerNorm_nonaffine',
                         'finite_gradients','save_load_identity','hidden_saturation_at_most_one_eighth','head_encoder_reader_unchanged','zero_optimizer_and_SFT'])
    write_json(run/'PILOT_PLAN.json',plan)
    project=Path(__file__).resolve().parents[1]
    for path in plan['source_sha256']:
        target=run/'source_snapshot'/Path(path).relative_to(project)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    for name in inherited_artifacts:
        target=run/'inherited_pilot'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(inherited/name,target)
    for path in inherited.glob('judge_queue/pilot_group_*.json'):
        target=run/'inherited_pilot/judge_queue'/path.name;target.parent.mkdir(exist_ok=True);shutil.copy2(path,target)
    shutil.copy2(inherited/'PILOT_GROUPS.jsonl',run/'inherited_pilot/PILOT_GROUPS.jsonl')
    write_json(run/'INHERITED_EVIDENCE.json',inherited_evidence)
    actor,reader=controller(),fresh_reader()
    initial=dict(head=module_hash(actor.head),encoder=parameter_hashes(actor.encoder_backend),reader=parameter_hashes(reader.backend))
    write_json(run/'POLICY_PROFILE.json',actor.policy_profile())
    started=time.monotonic()
    try:
        group=read(inherited/'pilot/group_000/group.json')
        targets=[group['ask_target'],*group['write_targets']]
        actor.enable_training()
        layer=actor.head[1]
        normalization=(isinstance(layer,torch.nn.LayerNorm) and tuple(layer.normalized_shape)==(64,)
                       and layer.elementwise_affine is False)
        initial_joint=True;fp64=True;normalized=True
        hidden=[]
        for target in targets:
            values=target['metadata']['feature_vector'];candidates=target['candidates']
            lp=actor.distribution(values,candidates)
            fp64=fp64 and lp.dtype==torch.float64
            p=lp.detach().exp()
            expected=torch.tensor([1.] if len(candidates)==1 else [.5,.25,.25],device=actor.device,dtype=torch.float64)
            normalized=normalized and abs(float(p.sum())-1)<1e-12
            initial_joint=initial_joint and bool(torch.allclose(p,expected,atol=1e-12,rtol=0))
            with torch.no_grad():
                features=torch.tensor(values,device=actor.device,dtype=torch.float32)
                h=actor.head[2](actor.head[1](actor.head[0](features)))
                hidden.append(float((h.abs()>.999).float().mean()))
        lp=joint_logprobs(actor,group)
        reference=lp.detach().clone()
        q=torch.linspace(-.1,1.,len(lp),device=actor.device,dtype=lp.dtype)
        session_loss(lp,reference,q,float(group['selection_weight']))[0].backward()
        finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in actor.head.parameters())
        actor.head.zero_grad(set_to_none=True);actor.head.requires_grad_(False);actor.training_enabled=False
        actor.save_checkpoint(run/'technical_checkpoint')
        restored=StableControllerPolicy.from_checkpoint(run/'technical_checkpoint',device='cuda',trainable=False,
                                                        encoder_backend=actor.encoder_backend)
        restored_identity=module_hash(restored.head)==initial['head'] and restored.policy_profile()==actor.policy_profile()
        with torch.no_grad():
            restored_identity=restored_identity and bool(torch.equal(joint_logprobs(restored,group),joint_logprobs(actor,group)))
        del restored
        activation_diagnostics=_head_activation_diagnostics(actor,group)
        gates=dict(hidden_saturation_at_most_one_eighth=activation_diagnostics['hidden_saturation_fraction_max_state']<=.125,
            FP64_normalized_probabilities=fp64 and normalized,same_initial_joint_distribution=initial_joint,
            hidden_LayerNorm_nonaffine=normalization,finite_gradients=finite,save_load_identity=restored_identity,
            head_unchanged=module_hash(actor.head)==initial['head'],
            encoder_unchanged=parameter_hashes(actor.encoder_backend)==initial['encoder'],
            reader_unchanged=parameter_hashes(reader.backend)==initial['reader'])
        check(plan)
        for path,sha in inherited_evidence.items():
            if file_hash(path)!=sha:raise ValueError('Inherited trace, feedback or source snapshot changed')
        for name,sha in inherited_artifacts.items():
            if file_hash(inherited/name)!=sha:raise ValueError('Inherited pilot artifact changed during technical check')
        result=dict(status='passed' if all(gates.values()) else 'failed_technical_check',version=VERSION,
            pilot_kind=plan['pilot_kind'],gates=gates,initial_fingerprints=initial,
            inherited_gates=inherited_result['gates'],outcome_episodes=0,terminal_queries=0,
            inherited_outcome_episodes=inherited_result['outcome_episodes'],inherited_terminal_queries=inherited_result['terminal_queries'],
            new_LLM_calls=0,new_reader_calls=0,optimizer_steps=0,sft_updates=0,
            cached_target_tanh_saturation_abs_gt0999=hidden,activation_diagnostics=activation_diagnostics,elapsed_seconds=time.monotonic()-started,
            inherited_artifact_sha256=inherited_artifacts)
        write_json(run/'PILOT_RESULT.json',result)
        print('PILOT_RESULT',result['status'],gates,flush=True)
    finally:release(actor,reader)


def train(run,data,groups=32,lr=LEARNING_RATE,budget_seconds=3600):
    import torch
    if type(groups) is not int or groups!=32 or not math.isfinite(lr) or lr!=LEARNING_RATE or not math.isfinite(budget_seconds) or budget_seconds!=3600:
        raise ValueError('Prospective schedule requires32 macro updates and positive learning rate/time budget')
    run=Path(run).resolve();pilot_plan=read(run/'PILOT_PLAN.json')
    review=read(run/'PILOT_REVIEW.json')
    if (read(run/'PILOT_RESULT.json')['status']!='passed' or review['status']!='passed'
            or review['pilot_result_sha256']!=file_hash(run/'PILOT_RESULT.json')):
        raise ValueError('A signed independent review of the passing pilot is required before optimization')
    check(pilot_plan);bank,_=load_bank(data)
    output=run/'optimizer';output.mkdir(exist_ok=False)
    plan=dict(version=VERSION,created_at=now(),seed=SEED,sft_updates=0,max_updates=groups,learning_rate=lr,adam_epsilon=ADAM_EPSILON,
              development_iteration='003 correction after run002 TRAIN/VAL saturation, without TEST002',
              controller_architecture='FP32 parameters; nonaffine hidden LayerNorm64 before tanh; FP64 categorical loss',
              kl_coefficient=.02,local_joint_KL_regularizer=True,budget_seconds=budget_seconds,
              budget_excludes_validation=True,cohort=balanced_cohort(bank['train'],groups),checkpoints=[0,16,32],
              trainable_scope='contextual_head_only',credit_unit='session_macro_ASK_x_WRITE',
              session_sampling='Independent Uniform(0..3) before roll-in',
              prefix_distribution='mu=.5*pi+.5*defer_earlier_ASK; WRITE follows pi',
              prefix_importance='2 if earlier ASK; otherwise 2*r/(1+r); detached, unclipped, unnormalized',
              gradient_normalization='E[weighted session gradient]=grad E[R]/4',
              max_outcome_episodes=groups*9,max_terminal_queries=groups*18,
              selection='Max validation net_return, ties query_success then fewer ASK then earliest',
              primary_claim='Head0 improvement alone is insufficient; also compare original LM, FIFO and public rules',
              terminal_diagnostic='Predeclared before fit; terminal never replaces validation-selected model',
              **{key:pilot_plan[key] for key in ('source_sha256','data_sha256','base_artifacts_sha256')})
    write_json(run/'TRAIN_PLAN.json',plan)
    actor,reader=controller(True),fresh_reader();reference=deepcopy(actor.head).requires_grad_(False)
    initial=dict(head=module_hash(actor.head),reference=module_hash(reference),
                 encoder=parameter_hashes(actor.encoder_backend),reader=parameter_hashes(reader.backend))
    if initial['head']!=initial['reference']:raise ValueError('Initial reference differs from the controller')
    write_json(run/'INITIALIZATION.json',initial);actor.save_checkpoint(output/'step_000000')
    optimizer=torch.optim.Adam(actor.head.parameters(),lr=lr,eps=ADAM_EPSILON)
    journal=output/'progress.jsonl';append(journal,dict(stage='started',step=0,plan=plan))
    history=[];training_seconds=0.;outcomes=queries=0;start=time.monotonic()
    try:
        baseline=evaluate_policy(actor,reader,bank['validation'],run/'validation/step_000000','Head0')
        best,best_step=baseline,0
        append(journal,dict(stage='validation_completed',step=0,metrics=baseline['metrics']))
        for index,episode in enumerate(plan['cohort'],start=1):
            if training_seconds>=budget_seconds:raise TimeoutError('Declared training budget exhausted')
            then=time.monotonic()
            group=collect_session_group(actor,reader,episode,SEED*1000000+index,
                                        run/'train_groups'/f'group_{index:03d}',session_index=None)
            rewards=feedback_group(run,f'train_group_{index:03d}',group['branches'])
            credit=enumerate_session_credit(group,rewards)
            stats=optimize_session(actor,reference,group,rewards,optimizer,beta=.02)
            stability=_head_activation_diagnostics(actor,group)
            diagnostics=_session_diagnostics(group,rewards)
            training_seconds+=time.monotonic()-then
            outcomes+=diagnostics['branch_count'];queries+=diagnostics['query_count']
            counters=dict(updates=index,optimizer_steps=index,outcome_episodes=outcomes,terminal_queries=queries,
                          rollin_prefixes=index,sft_updates=0,training_seconds=training_seconds)
            row=dict(stage='training',step=index,phase='session',session=group['session_index'],
                     **stats,elapsed_seconds=time.monotonic()-start,resource_counters=counters)
            append(journal,row);history.append(row)
            append(output/'stability_diagnostics.jsonl',dict(step=index,**stability))
            append(output/'session_diagnostics.jsonl',dict(step=index,phase='session',session=group['session_index'],
                                                       **stats,**diagnostics,prefix_component=group['diagnostics']['mixture_component']))
            append(output/'credit_diagnostics.jsonl',dict(step=index,credit=credit,session=group['session_index']))
            behavior=sampling_metrics(group['rollin'])
            values={f'behavior/{key}':value for key,value in behavior.items() if isinstance(value,(int,float))}
            values.update({'credit/expected_return':stats['baseline'],'credit/task_std':diagnostics['task_std'],
                           'credit/return_std':diagnostics['return_std'],'credit/prefix_importance':group['selection_weight'],
                           'credit/ask_task_std':diagnostics['ask_task_std'],
                           'credit/branch_count':diagnostics['branch_count']})
            values.update({f'stability/{key}':value for key,value in stability.items() if isinstance(value,(int,float))})
            append(output/'training_metrics.jsonl',dict(step=index,values=values,
                                                       visitation_distribution='mu_prefix_mixture',credit_unit='session_macro'))
            print('training',index,'session',group['session_index'],'branches',diagnostics['branch_count'],
                  'Q',stats['returns'],'loss',stats['loss'],flush=True)
            if index in plan['checkpoints']:
                actor.save_checkpoint(output/f'step_{index:06d}')
                panel=evaluate_policy(actor,reader,bank['validation'],run/'validation'/f'step_{index:06d}',f'Head-step-{index}')
                append(journal,dict(stage='validation_completed',step=index,metrics=panel['metrics']))
                if better_selection(panel['metrics'],best['metrics']):best,best_step=panel,index
        final=dict(head=module_hash(actor.head),reference=module_hash(reference),
                   encoder=parameter_hashes(actor.encoder_backend),reader=parameter_hashes(reader.backend))
        if any(initial[key]!=final[key] for key in ('encoder','reader','reference')):
            raise ValueError('Frozen encoder, reader or reference changed')
        check(plan)
        counters=dict(updates=groups,optimizer_steps=groups,outcome_episodes=outcomes,terminal_queries=queries,
                      rollin_prefixes=groups,training_seconds=training_seconds,total_seconds=time.monotonic()-start)
        result=dict(status='completed',version=VERSION,seed=SEED,sft_updates=0,best_step=best_step,history=history,
                    initial=initial,final=final,head_changed=initial['head']!=final['head'],resource_counters=counters)
        write_json(output/'training.json',result)
        write_json(run/'SELECTION.json',dict(created_at=now(),selected_before_evaluation=True,selection_split='validation',
                   selected_step=best_step,selected_checkpoint=str(output/f'step_{best_step:06d}'),
                   initial_checkpoint=str(output/'step_000000'),terminal_checkpoint=str(output/f'step_{groups:06d}'),
                   source_sha256=plan['source_sha256'],data_sha256=plan['data_sha256']))
        append(journal,dict(stage='completed',step=groups,selected_step=best_step))
        print('training_completed selected',best_step,flush=True)
    except Exception as error:
        append(journal,dict(stage='failed',step=len(history),error_type=type(error).__name__,error=str(error),
                            outcome_episodes=outcomes,terminal_queries=queries,training_seconds=training_seconds))
        raise
    finally:release(actor,reader)


def evaluate(run,data):
    run=Path(run).resolve();plan=read(run/'TRAIN_PLAN.json');check(plan)
    selection=read(run/'SELECTION.json');selection_sha=file_hash(run/'SELECTION.json')
    bank,_=load_bank(data);reader=fresh_reader();panels={};paths={}
    try:
        for method,path in (('Head0',selection['initial_checkpoint']),('RL-selected',selection['selected_checkpoint']),
                            ('RL-terminal',selection['terminal_checkpoint'])):
            if path in paths:panels[method]={**deepcopy(panels[paths[path]]),'identical_to':paths[path]};continue
            actor=StableControllerPolicy.from_checkpoint(path,device='cuda',trainable=False)
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
    parser.add_argument('--groups',type=int,default=32);parser.add_argument('--learning-rate',type=float,default=LEARNING_RATE)
    parser.add_argument('--budget-seconds',type=int,default=3600)
    parser.add_argument('--inherit-pilot',type=Path)
    options=parser.parse_args()
    if options.stage=='pilot':
        if options.inherit_pilot is None:parser.error('--inherit-pilot is required for reviewed feasibility inheritance')
        pilot(options.run,options.data,options.inherit_pilot)
    elif options.stage=='train':train(options.run,options.data,options.groups,options.learning_rate,options.budget_seconds)
    else:evaluate(options.run,options.data)


if __name__=='__main__':main()
