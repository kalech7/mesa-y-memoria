"""Session-macro controller RL with an independently reviewed TRAIN pilot.

No SFT, LoRA or optimizer on the provider LLM. The encoder and reader remain
frozen; only the contextual controller head is trained. Their interfaces are
unchanged. Prefixes come from a full-support exploration
mixture mu; exact detached importance ratios correct the local macro gradient.
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
    load_bank, controller, release, evaluate_policy, evaluate, sources, check,
    parameter_hashes, feedback_group,
)
from research.decision_memory_session_credit import collect_session_group, enumerate_session_credit
from research.decision_memory_session_training import joint_logprobs, session_loss, optimize_session
from research.decision_memory_training import module_hash
from research.decision_memory_metrics import sampling_metrics
from research.budget_memory_env import score_trace
from research.src.artifacts import file_hash, now, write_json

VERSION = 'decision_memory_session_experiment_v1'


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


def pilot(run,data):
    import torch
    from llm_local_001.model import artifact_hashes
    run=Path(run).resolve()
    if run.exists():raise FileExistsError(run)
    bank,data_hashes=load_bank(data)
    run.mkdir(parents=True)
    plan=dict(version=VERSION,created_at=now(),seed=SEED,sft_updates=0,
              trainable_scope='contextual_controller_head_only',
              source_sha256=sources(),data_sha256=data_hashes,base_artifacts_sha256=artifact_hashes(BASE),
              cohort=balanced_cohort(bank['train'],5),pilot_groups=10,
              pilot_max_branches=90,pilot_max_queries=180,
              session_sampling='Uniform(0..3) independently before each roll-in; seed101*100000+group_index',
              prefix_distribution='mu=.5*pi+.5*defer_earlier_ASK; WRITE follows pi',
              prefix_importance='2 if earlier ASK; otherwise 2*r/(1+r), r=product pi(PROCEED)',
              macro_distribution='pi_ASK * pi_WRITE_given_actual_user_reply',
              gates=dict(min_task_variable_groups=1,min_positive_supported_queries=2,
                         min_rules_reader_query_success=.6,require_complete_macro_enumeration=True,
                         require_frozen_weights_and_zero_steps=True),
              task_diagnostics='ASK variation after averaging conditional WRITE; WRITE variation within each ASK; not gates')
    write_json(run/'PILOT_PLAN.json',plan)
    project=Path(__file__).resolve().parents[1]
    for path in plan['source_sha256']:
        target=run/'source_snapshot'/Path(path).relative_to(project)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    actor,reader=controller(),fresh_reader()
    initial=dict(head=module_hash(actor.head),encoder=parameter_hashes(actor.encoder_backend),reader=parameter_hashes(reader.backend))
    write_json(run/'POLICY_PROFILE.json',actor.policy_profile())
    groups=[];branches=[];positive=0;started=time.monotonic()
    try:
        for index in range(10):
            episode=plan['cohort'][index%5]
            start=time.monotonic()
            group=collect_session_group(actor,reader,episode,SEED*100000+index,
                                        run/'pilot'/f'group_{index:03d}',session_index=None)
            model_seconds=time.monotonic()-start
            rewards=feedback_group(run,f'pilot_group_{index:03d}',group['branches'])
            credit=enumerate_session_credit(group,rewards)
            diagnostics=_session_diagnostics(group,rewards)
            positive+=sum(value==1 for reward in rewards for value in reward['query_utilities'])
            row=dict(group=index,session=group['session_index'],credit=credit,
                     prefix_importance=group['selection_weight'],rollin_component=group['diagnostics']['mixture_component'],
                     model_seconds=model_seconds,total_seconds=time.monotonic()-start,**diagnostics)
            append(run/'PILOT_GROUPS.jsonl',row);groups.append(row);branches.extend(group['branches'])
            print('pilot',index,'session',group['session_index'],'branches',diagnostics['branch_count'],
                  'Q',credit['full_returns'],'task_std',diagnostics['task_std'],flush=True)
        rules=evaluate_policy('semantic',reader,plan['cohort'],run/'pilot_rules','Public-rules-pilot')
        # Numeric Q only checks head autograd. No optimizer is created or run.
        torch.cuda.reset_peak_memory_stats()
        actor.enable_training();reference=deepcopy(actor.head).requires_grad_(False)
        lp=joint_logprobs(actor,group)
        with torch.no_grad():reference_lp=joint_logprobs(actor,group,head=reference)
        synthetic_q=torch.linspace(-.1,1.,len(lp),device=actor.device,dtype=lp.dtype)
        synthetic_loss=session_loss(lp,reference_lp,synthetic_q,float(group['selection_weight']))[0]
        synthetic_loss.backward()
        finite=all(value.grad is None or bool(torch.isfinite(value.grad).all()) for value in actor.head.parameters())
        actor.head.zero_grad(set_to_none=True);actor.head.requires_grad_(False);actor.training_enabled=False
        values=dict(task_variable_groups=sum(row['task_std']>0 for row in groups),
                    ask_variable_groups=sum(row['ask_task_std']>0 for row in groups),
                    write_variable_groups=sum(row['any_write_task_variation'] for row in groups),
                    positive_supported_queries=positive,
                    deterministic_positive_supported_queries=sum(score_trace(trace)['query_success']*2 for trace in branches),
                    rules_reader_query_success=rules['metrics']['query_success'],
                    complete_macro_enumeration=all(row['complete_macro_enumeration'] for row in groups),
                    head_unchanged=module_hash(actor.head)==initial['head'],
                    encoder_unchanged=parameter_hashes(actor.encoder_backend)==initial['encoder'],
                    reader_unchanged=parameter_hashes(reader.backend)==initial['reader'],finite_head_gradient=finite)
        passed=(values['task_variable_groups']>=1 and positive>=2 and values['rules_reader_query_success']>=.6
                and all(values[key] for key in ('complete_macro_enumeration','head_unchanged','encoder_unchanged','reader_unchanged','finite_head_gradient')))
        check(plan)
        result=dict(status='passed' if passed else 'failed_feasibility',version=VERSION,gates=values,
                    groups=groups,rules=rules,elapsed_seconds=time.monotonic()-started,
                    optimizer_steps=0,sft_updates=0,
                    outcome_episodes=len(branches),terminal_queries=sum(len(trace['queries']) for trace in branches),
                    peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
                    scope='Real TRAIN macro fork outcomes; synthetic Q only checks head autograd, zero updates')
        write_json(run/'PILOT_RESULT.json',result)
        print('PILOT_RESULT',result['status'],values,flush=True)
    finally:release(actor,reader)


def train(run,data,groups=32,lr=.003,budget_seconds=3600):
    import torch
    if type(groups) is not int or groups!=32 or not math.isfinite(lr) or lr<=0 or not math.isfinite(budget_seconds) or budget_seconds<=0:
        raise ValueError('Prospective schedule requires32 macro updates and positive learning rate/time budget')
    run=Path(run).resolve();pilot_plan=read(run/'PILOT_PLAN.json')
    review=read(run/'PILOT_REVIEW.json')
    if (read(run/'PILOT_RESULT.json')['status']!='passed' or review['status']!='passed'
            or review['pilot_result_sha256']!=file_hash(run/'PILOT_RESULT.json')):
        raise ValueError('A signed independent review of the passing pilot is required before optimization')
    check(pilot_plan);bank,_=load_bank(data)
    output=run/'optimizer';output.mkdir(exist_ok=False)
    plan=dict(version=VERSION,created_at=now(),seed=SEED,sft_updates=0,max_updates=groups,learning_rate=lr,
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
    optimizer=torch.optim.Adam(actor.head.parameters(),lr=lr)
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
            diagnostics=_session_diagnostics(group,rewards)
            training_seconds+=time.monotonic()-then
            outcomes+=diagnostics['branch_count'];queries+=diagnostics['query_count']
            counters=dict(updates=index,optimizer_steps=index,outcome_episodes=outcomes,terminal_queries=queries,
                          rollin_prefixes=index,sft_updates=0,training_seconds=training_seconds)
            row=dict(stage='training',step=index,phase='session',session=group['session_index'],
                     **stats,elapsed_seconds=time.monotonic()-start,resource_counters=counters)
            append(journal,row);history.append(row)
            append(output/'session_diagnostics.jsonl',dict(step=index,phase='session',session=group['session_index'],
                                                       **stats,**diagnostics,prefix_component=group['diagnostics']['mixture_component']))
            append(output/'credit_diagnostics.jsonl',dict(step=index,credit=credit,session=group['session_index']))
            behavior=sampling_metrics(group['rollin'])
            values={f'behavior/{key}':value for key,value in behavior.items() if isinstance(value,(int,float))}
            values.update({'credit/expected_return':stats['baseline'],'credit/task_std':diagnostics['task_std'],
                           'credit/return_std':diagnostics['return_std'],'credit/prefix_importance':group['selection_weight'],
                           'credit/ask_task_std':diagnostics['ask_task_std'],
                           'credit/branch_count':diagnostics['branch_count']})
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('pilot','train','evaluate'),required=True)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--groups',type=int,default=32);parser.add_argument('--learning-rate',type=float,default=.003)
    parser.add_argument('--budget-seconds',type=int,default=3600)
    options=parser.parse_args()
    if options.stage=='pilot':pilot(options.run,options.data)
    elif options.stage=='train':train(options.run,options.data,options.groups,options.learning_rate,options.budget_seconds)
    else:evaluate(options.run,options.data)


if __name__=='__main__':main()
