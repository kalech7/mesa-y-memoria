"""CPU-only macro credit integration, with numeric Q fixtures and no real SGD.

The controller/environment/collector are production code. Only the encoder and
reader are test doubles. Synthetic returns test autograd plumbing, never supply
action demonstrations or measured learning results to an experiment.
"""
from copy import deepcopy
import random

import pytest
import torch

from research.budget_memory_data import make_episode
from research.decision_memory_metrics import sampling_metrics
from research.decision_memory_policy import DecisionMemoryPolicy
from research.decision_memory_session_credit import collect_session_group, _seed
from research.decision_memory_session_training import joint_logprobs, session_loss, optimize_session
from research.decision_memory_training import module_hash
from research.tests.test_decision_memory_policy_integration import WordEncoder, DeferReader


@pytest.fixture
def actor():
    return DecisionMemoryPolicy(encoder_backend=WordEncoder(),trainable=True)


@pytest.fixture
def group(tmp_path,actor):
    return collect_session_group(actor,DeferReader(),make_episode('temporal',0,101),101,tmp_path/'macro',session_index=0)


def test_real_controller_macro_enumeration_metadata_and_provider_freeze(actor,group):
    assert len(group['branches'])==len(group['action_pairs'])==9
    assert len(group['write_targets'])==len(group['ask_target']['candidates'])==3
    expected={(i,j) for i in range(3) for j in range(3)}
    assert {(pair['ask_index'],pair['write_index']) for pair in group['action_pairs']}==expected
    joint=joint_logprobs(actor,group)
    torch.testing.assert_close(joint.exp().sum(),torch.tensor(1.),rtol=0,atol=0)
    ask=group['ask_target'];writes=group['write_targets']
    for pair in group['action_pairs']:
        i,j=pair['ask_index'],pair['write_index']
        old=ask['metadata']['old_logprobs'][i]+writes[i]['metadata']['old_logprobs'][j]
        assert float(joint[pair['branch_index']].detach())==pytest.approx(old,abs=2e-7)
        branch=group['branches'][pair['branch_index']]
        assert branch['counterfactual']['ask_action']==ask['candidates'][i]
        assert branch['counterfactual']['write_action']==writes[i]['candidates'][j]
        forced=[d for d in branch['decisions'] if d['metadata'].get('credit_role') in {'forced_counterfactual','forced_session_write'}]
        assert len(forced)==2
        assert all('old_logprobs' not in d['metadata'] for d in forced)
    assert group['diagnostics']['actor_before']==group['diagnostics']['actor_after']==actor.credit_parameter_fingerprint()
    assert group['rollin']['outcome_not_evaluated']
    assert not {'queries','metrics','future_answers','public_judge_payload'} & group['rollin'].keys()
    assert all(not value.requires_grad and value.grad is None for value in actor.encoder_backend.model.parameters())


def test_shared_ask_write_autograd_matches_direct_expected_return(actor,group):
    # Nonzero controller output weights exercise the shared hidden layer too.
    with torch.no_grad():
        actor.head[-1].weight.copy_(torch.linspace(-.02,.03,actor.head[-1].weight.numel()).reshape_as(actor.head[-1].weight))
        actor.head[-1].bias.copy_(torch.linspace(-.1,.15,8))
    lp=joint_logprobs(actor,group)
    q=torch.linspace(-.3,.9,len(lp),requires_grad=True)
    reference=lp.detach().clone()
    weight=1.7
    loss,*_=session_loss(lp,reference,q,weight,beta=0.)
    direct=-weight*(lp.exp()*q.detach()).sum()
    params=tuple(actor.head.parameters())
    actual=torch.autograd.grad(loss,params,retain_graph=True)
    expected=torch.autograd.grad(direct,params)
    for left,right in zip(actual,expected):
        torch.testing.assert_close(left,right,rtol=3e-5,atol=2e-7)
    assert actual[-1][:4].abs().sum()>0 and actual[-1][4:].abs().sum()>0
    assert actual[0].abs().sum()>0
    assert q.grad is None


def test_joint_kl_equals_ask_plus_ask_weighted_write_kl_and_gradient(actor,group):
    reference_head=deepcopy(actor.head).requires_grad_(False)
    with torch.no_grad():actor.head[-1].bias.copy_(torch.tensor([.2,-.2,.1,.4,.3,-.4,.5,-.1]))
    current=joint_logprobs(actor,group)
    with torch.no_grad():reference=joint_logprobs(actor,group,head=reference_head)
    exact=(current.exp()*(current-reference)).sum()
    ask=group['ask_target']
    a=actor.distribution(ask['metadata']['feature_vector'],ask['candidates'])
    with torch.no_grad():ar=actor.distribution(ask['metadata']['feature_vector'],ask['candidates'],head=reference_head)
    factor=(a.exp()*(a-ar)).sum()
    for i,target in enumerate(group['write_targets']):
        w=actor.distribution(target['metadata']['feature_vector'],target['candidates'])
        with torch.no_grad():wr=actor.distribution(target['metadata']['feature_vector'],target['candidates'],head=reference_head)
        factor=factor+a[i].exp()*(w.exp()*(w-wr)).sum()
    torch.testing.assert_close(exact,factor,rtol=1e-5,atol=1e-7)
    left=torch.autograd.grad(exact,tuple(actor.head.parameters()),retain_graph=True)
    right=torch.autograd.grad(factor,tuple(actor.head.parameters()))
    for x,y in zip(left,right):torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-7)
    assert all(value.grad is None for value in reference_head.parameters())


def test_macro_update_learns_both_gates_and_keeps_provider_and_reference_frozen(actor,group):
    initial=actor.credit_parameter_fingerprint()
    reference=deepcopy(actor.head).requires_grad_(False)
    reference_hash=module_hash(reference)
    encoder=actor.encoder_backend.model.base_model.scale.clone()
    q=[1. if pair['ask_index']==1 and pair['write_index']==2 else -.1 for pair in group['action_pairs']]
    rewards=[{'episodic_reward':value} for value in q]
    before=float((joint_logprobs(actor,group).exp()*torch.tensor(q)).sum().detach())
    result=optimize_session(actor,reference,group,rewards,torch.optim.SGD(actor.head.parameters(),lr=.1),beta=0.)
    after=float((joint_logprobs(actor,group).exp()*torch.tensor(q)).sum().detach())
    assert after>before and result['max_old_distribution_error']==0
    bias=actor.head[-1].bias.detach()
    assert bias[1]>bias[0] and bias[5]>bias[4]  # ASK and STORE gates both learn.
    assert bias[2]>bias[3] and bias[7]>bias[6]  # Temporal ASK and slot2 conditionals learn.
    assert actor.credit_parameter_fingerprint()!=initial
    assert module_hash(reference)==reference_hash
    torch.testing.assert_close(encoder,actor.encoder_backend.model.base_model.scale,rtol=0,atol=0)
    assert all(value.grad is None and not value.requires_grad for value in actor.encoder_backend.model.parameters())


def test_equal_returns_and_identical_reference_give_exact_zero_gradient(actor,group):
    lp=joint_logprobs(actor,group)
    q=torch.full_like(lp,-.12)
    loss,policy,kl,_,advantages=session_loss(lp,lp.detach().clone(),q,2.)
    loss.backward()
    assert float(loss.detach())==float(policy.detach())==float(kl.detach())==0.
    assert torch.equal(advantages,torch.zeros_like(advantages))
    assert all(value.grad is None or torch.equal(value.grad,torch.zeros_like(value.grad)) for value in actor.head.parameters())


@pytest.mark.parametrize('corrupt',['duplicate_pair','branch_action','feature_digest'])
def test_macro_binding_corruption_rejected_before_update(actor,group,corrupt):
    broken=deepcopy(group)
    if corrupt=='duplicate_pair':
        broken['action_pairs'][1]=deepcopy(broken['action_pairs'][2])
    elif corrupt=='branch_action':
        broken['branches'][0]['counterfactual']['write_action']='STORE_SLOT_2'
    else:
        broken['write_targets'][0]['metadata']['feature_vector'][0]+=.1
    before=actor.credit_parameter_fingerprint()
    with pytest.raises(ValueError):
        optimize_session(actor,deepcopy(actor.head).requires_grad_(False),broken,
                         [{'episodic_reward':0.} for _ in broken['branches']],torch.optim.SGD(actor.head.parameters(),lr=.1))
    assert actor.credit_parameter_fingerprint()==before


def test_exhausted_question_budget_has_three_macros_and_no_ask_gradient(tmp_path,actor):
    episode=make_episode('retention',0,101)
    # Select a fixture solely by exogenous mixture coin and observed budget.
    group=None
    for seed in range(30):
        if random.Random(_seed(seed,'rollin_mixture')).random()<.5:continue
        candidate=collect_session_group(actor,DeferReader(),episode,seed,tmp_path/f'budget-{seed}',session_index=3)
        if len(candidate['ask_target']['candidates'])==1:
            group=candidate;break
    assert group is not None
    assert len(group['branches'])==3 and len(group['write_targets'])==1
    assert group['ask_target']['candidates']==['PROCEED']
    lp=joint_logprobs(actor,group)
    torch.testing.assert_close(lp.exp(),torch.tensor([.5,.25,.25]),rtol=0,atol=0)
    (-lp[1]).backward()
    assert torch.equal(actor.head[-1].bias.grad[:4],torch.zeros(4))
    assert actor.head[-1].bias.grad[4:].abs().sum()>0
    diagnostic=sampling_metrics(group['rollin'])
    assert diagnostic['sampled_ask_gate_count']<=3


@pytest.mark.parametrize('corrupt',[
    'selection_weight','density_ratio','mixture_probability',
    'ask_step','write_step','branch_session','write_target_swap',
])
def test_macro_sampling_and_conditional_identity_guards_reject_before_update(actor,group,corrupt):
    """Ratios/steps cannot silently relabel a scalar outcome to another state."""
    broken=deepcopy(group)
    if corrupt=='selection_weight':
        broken['selection_weight']=1.5  # True t0 mixture ratio is exactly one.
    elif corrupt=='density_ratio':
        broken['diagnostics']['prefix_importance']=1.5
    elif corrupt=='mixture_probability':
        broken['diagnostics']['mixture_probability']=.75
    elif corrupt=='ask_step':
        broken['ask_target']['step_id']+=2
    elif corrupt=='write_step':
        broken['write_targets'][0]['step_id']+=2
    elif corrupt=='branch_session':
        broken['branches'][0]['counterfactual']['session_index']=1
    elif corrupt=='write_target_swap':
        # Initial logits are equal, so a likelihood-only replay check would miss
        # this swap. The actual post-reply WRITE observations must bind to ASK.
        broken['write_targets'][1],broken['write_targets'][2]=(broken['write_targets'][2],
                                                             broken['write_targets'][1])
    before=actor.credit_parameter_fingerprint()
    encoder_before=actor.encoder_backend.model.base_model.scale.clone()
    reference=deepcopy(actor.head).requires_grad_(False)
    reference_before=module_hash(reference)
    optimizer=torch.optim.Adam(actor.head.parameters(),lr=.01)
    with pytest.raises(ValueError):
        optimize_session(actor,reference,broken,
                         [{'episodic_reward':0.} for _ in broken['branches']],optimizer,beta=0.)
    assert actor.credit_parameter_fingerprint()==before
    assert module_hash(reference)==reference_before
    assert not optimizer.state  # No Adam moments, hence no optimizer step.
    torch.testing.assert_close(encoder_before,actor.encoder_backend.model.base_model.scale,rtol=0,atol=0)
