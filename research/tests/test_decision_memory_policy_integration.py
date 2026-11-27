"""CPU-only controller → real simulator forks → enum-loss interface tests.

The returns here are synthetic numeric fixtures to check differentiation, not
measured task outcomes or training data. No provider model is loaded or updated.
"""
import pytest
import torch
from copy import deepcopy

from research.budget_memory_data import make_episode
from research.decision_memory_credit import collect_counterfactual_group, enumerate_credit
from research.decision_memory_policy import DecisionMemoryPolicy
from research.decision_memory_training import optimize_counterfactual
from research.tests.test_decision_memory_policy import FrozenEncoder


class WordEncoder(FrozenEncoder):
    def encode_prompt(self, text):
        return [1]+[sum(word.encode())%251+3 for word in text.split()]+[2]


class DeferReader:
    def answer(self, observation, candidates, *, seed, decision_id):
        return {'answer':'DEFER', 'metadata':{'decision_id':decision_id}}


@pytest.mark.parametrize('phase',['ask','write'])
def test_real_policy_counterfactual_metadata_and_cpu_optimizer(tmp_path,phase):
    actor=DecisionMemoryPolicy(encoder_backend=WordEncoder(),trainable=True)
    reference=deepcopy(actor.head).requires_grad_(False)
    initial=actor.credit_parameter_fingerprint()
    episode=make_episode('correction',3,202611061)
    group=collect_counterfactual_group(actor,DeferReader(),episode,101,tmp_path/phase,phase)
    assert actor.credit_parameter_fingerprint()==initial
    assert group['diagnostics']['actor_before']==group['diagnostics']['actor_after']==initial
    metadata=group['target']['metadata']
    candidates=group['target']['candidates']
    replay=actor.distribution(metadata['feature_vector'],candidates)
    torch.testing.assert_close(replay.detach(),torch.tensor(metadata['old_logprobs']),rtol=0,atol=0)
    assert metadata['probabilities']==[.5,.25,.25]
    assert group['target']['decision_id']==metadata['decision_id']
    assert all(key not in group['visitation'] for key in ('queries','metrics','public_judge_payload'))
    assert len(actor._used_decision_ids)==8+3*(7-group['target']['step_id'])
    returns=[-.1,.2,.3]
    credit=enumerate_credit(group,returns)
    assert credit['weighted_advantage_mean']==pytest.approx(0.,abs=1e-15)
    assert credit['target_action'] is None
    rewards=[{'episodic_reward':value,'mean_utility':value} for value in returns]
    before=float((replay.exp()*torch.tensor(returns)).sum().detach())
    result=optimize_counterfactual(actor,reference,group,rewards,torch.optim.SGD(actor.head.parameters(),lr=.1))
    after=float((actor.distribution(metadata['feature_vector'],candidates).exp()*torch.tensor(returns)).sum().detach())
    assert after>before
    assert result['max_old_distribution_error']==0
    assert result['grad_norm']>0
    assert actor.credit_parameter_fingerprint()!=initial
    assert all(value.grad is None and not value.requires_grad for value in actor.encoder_backend.model.parameters())
