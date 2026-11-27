"""CPU contracts for learned binary gates on a frozen transformer encoder."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from research.decision_memory_policy import (
    ASK_ACTIONS, WRITE_ACTIONS, DecisionMemoryPolicy, joint_distribution, structural_features,
)


class Body(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(4), requires_grad=False)
        self.calls = 0
    def forward(self, input_ids, attention_mask, use_cache=False, return_dict=True):
        self.calls += 1
        x = input_ids.float()
        summary = torch.stack((x.mean(), x.sum()%7, x.std()+1, x[0,-1]%11)) * self.scale
        return SimpleNamespace(last_hidden_state=summary[None,None,:].expand(1,input_ids.shape[1],-1))


class FrozenEncoder:
    def __init__(self, identity='toy-fixed'):
        self.torch = torch
        self.model = torch.nn.Module()
        self.model.add_module('base_model', Body())
        self.model.config = SimpleNamespace(hidden_size=4, model_type='toy')
        self.model.requires_grad_(False)
        self.config = SimpleNamespace(model_path='/cpu-only-provider-fixture', precision='float32')
        self.origin_checkpoint = None
        self.identity = identity
    @property
    def device(self):
        return self.model.base_model.scale.device
    def encode_prompt(self, text):
        return [1]+list(map(ord,text))+[2]
    def base_fingerprint(self):
        return {'fake-weights':self.identity}


def obs(phase='write', index=0):
    return {'decision_type':phase,'session_index':index,'session_id':f'S{index+1}','date':'2030-01-01',
            'exchange':[{'role':'user','turn_id':'u1','text':'Prefiero agua.'}],
            'questions_remaining':1,
            'slots':[{'slot_id':i,'source_session':None,'stored_at':None,'exchange':[]} for i in (1,2)]}


def policy(trainable=True):
    return DecisionMemoryPolicy(encoder_backend=FrozenEncoder(),trainable=trainable)


@pytest.mark.parametrize('catalogue',[list(ASK_ACTIONS),list(WRITE_ACTIONS)])
def test_balanced_binary_mass_is_not_cardinality_biased(catalogue):
    result=joint_distribution(torch.zeros(8,requires_grad=True),catalogue)
    torch.testing.assert_close(result['joint_logprobs'].exp(),torch.tensor([.5,.25,.25]),rtol=0,atol=0)
    torch.testing.assert_close(result['gate_logprobs'].exp(),torch.tensor([.5,.5]),rtol=0,atol=0)
    torch.testing.assert_close(result['conditional_logprobs'].exp(),torch.tensor([.5,.5]),rtol=0,atol=0)
    assert result['joint_logprobs'].exp().sum()==1
    order=list(reversed(catalogue))
    torch.testing.assert_close(joint_distribution(torch.zeros(8),order)['joint_logprobs'].exp(),torch.tensor([.25,.25,.5]),rtol=0,atol=0)


def test_exact_joint_kl_equals_gate_plus_weighted_conditional_kl():
    current=torch.tensor([.2,-.3,.1,.5,-.2,.8,.3,-.6],requires_grad=True)
    old=torch.tensor([.1,.4,-.5,.9,-.1,.3,.7,.6])
    for catalogue in (list(ASK_ACTIONS),list(WRITE_ACTIONS)):
        a,b=joint_distribution(current,catalogue),joint_distribution(old,catalogue)
        exact=(a['joint_logprobs'].exp()*(a['joint_logprobs']-b['joint_logprobs'])).sum()
        hierarchical=(a['gate_logprobs'].exp()*(a['gate_logprobs']-b['gate_logprobs'])).sum()
        hierarchical+=a['gate_logprobs'][1].exp()*(a['conditional_logprobs'].exp()*(a['conditional_logprobs']-b['conditional_logprobs'])).sum()
        torch.testing.assert_close(exact,hierarchical,rtol=1e-6,atol=1e-7)
        exact_grad=torch.autograd.grad(exact,current,retain_graph=True)[0]
        hierarchical_grad=torch.autograd.grad(hierarchical,current,retain_graph=True)[0]
        torch.testing.assert_close(exact_grad,hierarchical_grad,rtol=1e-5,atol=1e-7)


def test_selected_branch_gradients_train_gate_and_conditional_without_encoder_gradient():
    p=policy()
    record=p.feature(obs())
    reference=deepcopy(p.head).requires_grad_(False)
    before_encoder=p.encoder_backend.model.base_model.scale.clone()
    before_head=p.credit_parameter_fingerprint()
    lp=p.distribution(record['feature_vector'],list(WRITE_ACTIONS))
    torch.testing.assert_close(lp,p.distribution(record['feature_vector'],list(WRITE_ACTIONS),head=reference),rtol=0,atol=0)
    (-lp[1]).backward()
    bias=p.head[-1].bias.grad
    assert bias[4]>0 and bias[5]<0 and bias[6]<0 and bias[7]>0
    assert torch.equal(bias[:4],torch.zeros(4))
    assert p.encoder_backend.model.base_model.scale.grad is None
    optimizer=torch.optim.SGD(p.head.parameters(),lr=.1)
    optimizer.step()
    assert p.credit_parameter_fingerprint()!=before_head
    torch.testing.assert_close(p.encoder_backend.model.base_model.scale,before_encoder,rtol=0,atol=0)
    assert all(not v.requires_grad for v in p.encoder_backend.model.parameters())
    assert all(v.requires_grad for v in p.head.parameters())


def test_negative_branch_updates_gate_but_not_conditional():
    logits=torch.zeros(8,requires_grad=True)
    (-joint_distribution(logits,list(WRITE_ACTIONS))['joint_logprobs'][0]).backward()
    assert logits.grad[4]<0 and logits.grad[5]>0
    assert torch.equal(logits.grad[6:],torch.zeros(2))


def test_greedy_is_hierarchical_and_has_no_empty_slot_action_rule():
    p=policy()
    with torch.no_grad():
        p.head[-1].bias[5]=torch.log(torch.tensor(1.5))  # STORE .6; each slot .3, NOOP .4.
    result=p.decide(obs(),list(WRITE_ACTIONS),seed=101,sample=False,decision_id='r1/S1/write/1')
    assert result['action']=='STORE_SLOT_1'
    assert result['metadata']['probabilities'][0]>result['metadata']['probabilities'][1]
    changed=obs(index=1)
    changed['slots'][0].update(exchange=[{'role':'user','turn_id':'old','text':'Un dato.'}],source_session=0,stored_at='2029-12-31')
    result2=p.decide(changed,list(WRITE_ACTIONS),seed=101,sample=False,decision_id='r1/S2/write/3')
    assert result2['action']=='STORE_SLOT_1'  # No forced preference for the empty slot.


def test_initial_greedy_negative_ties_and_joint_sampling_replay():
    p=policy()
    record=p.feature(obs('ask'))
    result=p.decide(obs('ask'),list(ASK_ACTIONS),seed=101,sample=False,decision_id='r1/S1/ask/0')
    assert result['action']=='PROCEED'
    seen=set()
    before_rng=torch.random.get_rng_state().clone()
    for seed in range(30):
        r=p.decide(obs('ask'),list(ASK_ACTIONS),seed=seed,sample=True,decision_id=f'rollout{seed}/S1/ask/0')
        seen.add(r['action'])
        m=r['metadata']
        torch.testing.assert_close(p.distribution(m['feature_vector'],list(ASK_ACTIONS)).detach(),torch.tensor(m['old_logprobs']),rtol=0,atol=0)
        assert m['old_logprobs'][m['action_index']]==pytest.approx(m['chosen_gate_logprob']+m['chosen_conditional_logprob'])
    assert seen==set(ASK_ACTIONS)
    assert torch.equal(torch.random.get_rng_state(),before_rng)
    assert record['feature_vector']==result['metadata']['feature_vector']


def test_cache_is_prompt_based_normalized_and_private_ids_never_enter_prompt():
    p=policy()
    first,second=p.feature(obs()),p.feature(obs())
    assert first['prompt_sha256']==second['prompt_sha256'] and second['encoder_cache_hit']
    assert p.encoder_backend.model.base_model.calls==1
    normalized=torch.tensor(first['feature_vector'][:4])
    assert normalized.mean()==pytest.approx(0.,abs=1e-5)
    assert normalized.var(unbiased=False)==pytest.approx(1.,abs=1e-5)
    r=p.decide(obs(),list(WRITE_ACTIONS),seed=923,sample=False,decision_id='PRIVATE_episode_seed923')
    assert 'PRIVATE_episode_seed923' not in r['metadata']['prompt']
    assert '923' not in r['metadata']['prompt']
    changed=obs();changed['exchange'][0]['text']='Otra frase pública.'
    assert p.feature(changed)['prompt_sha256']!=first['prompt_sha256']
    assert p.encoder_backend.model.base_model.calls==2
    json.dumps(r,allow_nan=False)
    with pytest.raises(ValueError,match='unique'):
        p.decide(obs(),list(WRITE_ACTIONS),seed=923,sample=False,decision_id='PRIVATE_episode_seed923')


def test_public_structural_features_do_not_contain_semantic_rules():
    value=obs('ask',index=3)
    value['slots'][0].update(exchange=[{'role':'user','turn_id':'x','text':'Nada semántico.'}],source_session=0,stored_at='2029-01-01')
    assert structural_features(value)==[1.,0.,1.,1.,0.,1.,1.,0.]
    value['exchange'][0]['text']='ahora habitual casa oficina cualquier palabra'
    assert structural_features(value)==[1.,0.,1.,1.,0.,1.,1.,0.]
    p=policy()
    private=obs();private['truth']={'correct':'NOOP'}
    with pytest.raises(ValueError):p.feature(private)
    with pytest.raises(ValueError):p.distribution([0.]*p.feature_size,['NOOP','STORE_SLOT_1'])


def test_forced_proceed_distribution_exactly_one_and_gradient_zero():
    logits=torch.randn(8,requires_grad=True)
    result=joint_distribution(logits,['PROCEED'])['joint_logprobs']
    assert result.tolist()==[0.]
    result.sum().backward()
    assert torch.equal(logits.grad,torch.zeros(8))
    p=policy();value=obs('ask');value['questions_remaining']=0
    r=p.decide(value,['PROCEED'],seed=0,sample=True,decision_id='forced/S1/ask/0')
    assert r['action']=='PROCEED' and r['metadata']['probabilities']==[1.]


def test_singleton_catalogue_requires_exhausted_public_budget():
    p=policy()
    with pytest.raises(ValueError,match='budget'):
        p.decide(obs('ask'),['PROCEED'],seed=0,sample=False,decision_id='bad-budget')
    exhausted=obs('ask');exhausted['questions_remaining']=0
    with pytest.raises(ValueError,match='budget'):
        p.decide(exhausted,list(ASK_ACTIONS),seed=0,sample=False,decision_id='bad-full-catalogue')


def test_checkpoint_head_roundtrip_and_checksum_base_profile_guards(tmp_path):
    p=policy()
    before=p.policy_profile()
    modified=p.policy_profile();modified['encoder']['model_path']='corrupt external copy'
    assert p.policy_profile()==before
    p.save_checkpoint(tmp_path/'head0')
    loaded=DecisionMemoryPolicy.from_checkpoint(tmp_path/'head0',encoder_backend=FrozenEncoder())
    assert loaded.credit_parameter_fingerprint()==p.credit_parameter_fingerprint()
    assert loaded.policy_profile()==p.policy_profile()
    with pytest.raises(ValueError,match='encoder'):
        DecisionMemoryPolicy.from_checkpoint(tmp_path/'head0',encoder_backend=FrozenEncoder('different-base'))
    data=bytearray((tmp_path/'head0/head.safetensors').read_bytes());data[-1]^=1
    (tmp_path/'head0/head.safetensors').write_bytes(data)
    with pytest.raises(ValueError,match='checksum'):
        DecisionMemoryPolicy.from_checkpoint(tmp_path/'head0',encoder_backend=FrozenEncoder())
