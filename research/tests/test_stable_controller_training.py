"""Production macro credit with the stable head and a CPU encoder test double."""
from copy import deepcopy
import pytest
import torch
from research.budget_memory_data import make_episode
from research.decision_memory_session_credit import collect_session_group
from research.decision_memory_session_training import joint_logprobs, session_loss
from research.stable_controller_policy import StableControllerPolicy
from research.stable_controller_training import optimize_session
from research.tests.test_decision_memory_policy_integration import WordEncoder, DeferReader


@pytest.fixture
def setup(tmp_path):
    actor=StableControllerPolicy(encoder_backend=WordEncoder(),trainable=True)
    group=collect_session_group(actor,DeferReader(),make_episode('temporal',0,101),101,
                                tmp_path/'macro',session_index=0)
    return actor,group


def test_fp64_replay_and_real_macro_update(setup):
    actor,group=setup
    reference=deepcopy(actor.head).requires_grad_(False)
    q=[1. if p['ask_index']==1 and p['write_index']==2 else -.1 for p in group['action_pairs']]
    before=float((joint_logprobs(actor,group).exp()*torch.tensor(q,dtype=torch.float64)).sum().detach())
    optimizer=torch.optim.Adam(actor.head.parameters(),lr=.0003,eps=1e-6)
    stats=optimize_session(actor,reference,group,[{'episodic_reward':x} for x in q],optimizer,beta=0.)
    after=float((joint_logprobs(actor,group).exp()*torch.tensor(q,dtype=torch.float64)).sum().detach())
    assert after>before
    assert stats['max_old_distribution_error']==0
    assert stats['distribution_dtype']=='torch.float64'
    assert all(p.dtype==torch.float32 for p in actor.head.parameters())
    assert all(p.grad is None for p in reference.parameters())
    assert all(p.grad is None and not p.requires_grad for p in actor.encoder_backend.model.parameters())


def test_nonzero_fp64_replay_error_is_detected_before_sgd(setup):
    actor,group=setup
    group['ask_target']['metadata']['old_logprobs'][0]+=1e-10
    before=actor.credit_parameter_fingerprint()
    optimizer=torch.optim.Adam(actor.head.parameters(),lr=.0003,eps=1e-6)
    with pytest.raises(ValueError,match='probabilities differ'):
        optimize_session(actor,deepcopy(actor.head).requires_grad_(False),group,
                         [{'episodic_reward':0.} for _ in group['branches']],optimizer)
    assert actor.credit_parameter_fingerprint()==before
    assert not optimizer.state


def test_fp64_centered_objective_retains_direct_gradient_and_detaches_returns(setup):
    actor,group=setup
    lp=joint_logprobs(actor,group)
    assert lp.dtype==torch.float64
    q=torch.linspace(-.2,.8,len(lp),dtype=torch.float64,requires_grad=True)
    loss,*_=session_loss(lp,lp.detach().clone(),q,1.7,beta=0.)
    direct=-1.7*(lp.exp()*q.detach()).sum()
    actual=torch.autograd.grad(loss,tuple(actor.head.parameters()),retain_graph=True)
    expected=torch.autograd.grad(direct,tuple(actor.head.parameters()))
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)
    assert q.grad is None
