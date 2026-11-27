"""Stable-head CPU contracts; no provider model or real training is executed."""
from copy import deepcopy
import json

import pytest
import torch

from research.decision_memory_policy import ASK_ACTIONS, WRITE_ACTIONS, DecisionMemoryPolicy
from research.stable_controller_policy import StableControllerPolicy, VERSION
from research.decision_memory_session_training import joint_logprobs, session_loss
from research.tests.test_decision_memory_policy import obs
from research.tests.test_decision_memory_policy_integration import WordEncoder


def policy(real_dimension=False):
    encoder=WordEncoder()
    if real_dimension:
        # Only exercise the head: the toy BODY is deliberately never called.
        encoder.model.config.hidden_size=896
    return StableControllerPolicy(encoder_backend=encoder,trainable=True)


def test_architecture_and_parameter_count_exact_with_unchanged_provider():
    actor=policy(True)
    assert actor.feature_size==904
    assert sum(p.numel() for p in actor.head.parameters())==58440
    assert isinstance(actor.head[1],torch.nn.LayerNorm)
    assert actor.head[1].normalized_shape==(64,) and actor.head[1].eps==1e-5
    assert actor.head[1].elementwise_affine is False
    assert all(p.dtype==torch.float32 and p.requires_grad for p in actor.head.parameters())
    assert all(not p.requires_grad for p in actor.encoder_backend.model.parameters())
    assert actor.encoder_backend.model.base_model.calls==0


@pytest.mark.parametrize('actions',[list(ASK_ACTIONS),list(WRITE_ACTIONS)])
def test_initial_law_balanced_and_replay_fp64_exact(actions):
    actor=policy();features=torch.linspace(-1,1,actor.feature_size)
    old=DecisionMemoryPolicy(encoder_backend=WordEncoder(),trainable=True)
    new_lp=actor.distribution(features,actions)
    assert new_lp.dtype==torch.float64
    torch.testing.assert_close(new_lp.exp(),torch.tensor([.5,.25,.25],dtype=torch.float64),rtol=0,atol=0)
    torch.testing.assert_close(new_lp.exp().float(),old.distribution(features,actions).exp(),rtol=0,atol=0)
    torch.testing.assert_close(new_lp,actor.distribution(features,actions),rtol=0,atol=0)
    # Equality in law does not promise identical FP32/FP64 multinomial draws.


def test_fixed_layernorm_prevents_complete_tanh_saturation_under_stress():
    actor=policy(True);features=torch.linspace(-1,1,904)
    with torch.no_grad():
        actor.head[0].weight.mul_(512.)
        actor.head[0].bias.add_(1000.)
    raw=actor.head[0](features)
    assert bool(raw.tanh().abs().gt(.99).all())  # Stress actually saturates old head.
    normalized=actor.head[1](raw)
    hidden=actor.head[2](normalized)
    assert float(hidden.abs().gt(.99).float().mean())<.2
    assert bool(torch.isfinite(hidden).all())
    assert float(normalized.mean().detach())==pytest.approx(0.,abs=2e-5)
    assert float(normalized.var(unbiased=False).detach())==pytest.approx(1.,abs=2e-4)


def test_both_gates_and_shared_layer_receive_finite_gradients():
    actor=policy();features=torch.linspace(-1,1,actor.feature_size)
    with torch.no_grad():
        actor.head[-1].weight.copy_(torch.linspace(-.1,.2,512).reshape(8,64))
    ask=actor.distribution(features,list(ASK_ACTIONS))
    write=actor.distribution(features.flip(0),list(WRITE_ACTIONS))
    loss=-ask[1]-write[2]
    loss.backward()
    bias=actor.head[-1].bias.grad
    assert bias[:4].abs().sum()>0 and bias[4:].abs().sum()>0
    assert actor.head[0].weight.grad.abs().sum()>0
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in actor.head.parameters())
    assert all(p.grad is None for p in actor.encoder_backend.model.parameters())


def test_reference_kl_replay_and_serialization_roundtrip(tmp_path):
    actor=policy();reference=deepcopy(actor.head).requires_grad_(False)
    features=torch.linspace(-1,1,actor.feature_size)
    before=actor.distribution(features,list(WRITE_ACTIONS))
    torch.testing.assert_close(before,actor.distribution(features,list(WRITE_ACTIONS),head=reference),rtol=0,atol=0)
    with torch.no_grad():actor.head[-1].bias.copy_(torch.linspace(-.1,.2,8))
    current=actor.distribution(features,list(WRITE_ACTIONS))
    old=actor.distribution(features,list(WRITE_ACTIONS),head=reference)
    kl=(current.exp()*(current-old)).sum()
    assert kl>0;kl.backward()
    assert all(p.grad is None for p in reference.parameters())
    actor.save_checkpoint(tmp_path/'stable')
    manifest=json.loads((tmp_path/'stable/decision_memory_checkpoint.json').read_text())
    assert manifest['controller_policy_version']==VERSION
    assert manifest['profile']['version']==VERSION
    loaded=StableControllerPolicy.from_checkpoint(tmp_path/'stable',encoder_backend=WordEncoder(),trainable=True)
    assert loaded.policy_profile()==actor.policy_profile()
    assert loaded.credit_parameter_fingerprint()==actor.credit_parameter_fingerprint()
    torch.testing.assert_close(loaded.distribution(features,list(WRITE_ACTIONS)),current.detach(),rtol=0,atol=0)
    historical=DecisionMemoryPolicy(encoder_backend=WordEncoder())
    historical.save_checkpoint(tmp_path/'old')
    with pytest.raises(ValueError,match='historical'):
        StableControllerPolicy.from_checkpoint(tmp_path/'old',encoder_backend=WordEncoder())


def test_decide_metadata_and_public_feature_extraction_unchanged():
    actor=policy();old=DecisionMemoryPolicy(encoder_backend=WordEncoder())
    assert actor.feature(obs())['feature_vector']==old.feature(obs())['feature_vector']
    result=actor.decide(obs(),list(WRITE_ACTIONS),seed=101,sample=False,decision_id='stable/S1/write')
    assert result['action']=='NOOP'
    assert result['metadata']['policy_version']==VERSION
    assert result['metadata']['distribution_dtype']=='float64'
    assert result['metadata']['profile_sha256']==actor.policy_profile()['profile_sha256']
    assert actor.policy_profile()['head']['normalization']['elementwise_affine'] is False
    json.dumps(result,allow_nan=False)


def test_greedy_numerical_tie_tolerance_changes_neither_sampling_nor_likelihood():
    actor=policy();value=obs('ask');features=actor.feature(value)['feature_vector']
    with torch.no_grad():actor.head[-1].bias[1]=5e-9
    lp=actor.distribution(features,list(ASK_ACTIONS)).detach()
    result=actor.decide(value,list(ASK_ACTIONS),seed=101,sample=False,decision_id='tie')
    assert result['action']=='PROCEED'
    metadata=result['metadata']
    assert 0<metadata['gate_logprobs'][1]-metadata['gate_logprobs'][0]<1e-8
    torch.testing.assert_close(torch.tensor(metadata['old_logprobs'],dtype=torch.float64),lp,rtol=0,atol=0)
    assert metadata['probabilities'][0]<.5  # The distribution was not snapped.
    assert metadata['action_index']==0 and metadata['chosen_gate_logprob']==metadata['gate_logprobs'][0]
    sampled=actor.decide(value,list(ASK_ACTIONS),seed=101,sample=True,decision_id='tie-sampled')
    assert sampled['metadata']['old_logprobs']==metadata['old_logprobs']
    with torch.no_grad():
        actor.head[-1].bias[1]=1e-3
        actor.head[-1].bias[3]=5e-9
    first=actor.decide(value,list(ASK_ACTIONS),seed=101,sample=False,decision_id='conditional-tie')
    assert first['action']=='ASK_TEMPORAL_SCOPE'
    with torch.no_grad():actor.head[-1].bias[3]=1e-3
    second=actor.decide(value,list(ASK_ACTIONS),seed=101,sample=False,decision_id='conditional-signal')
    assert second['action']=='ASK_CURRENT_CONTEXT'
    assert actor.policy_profile()['greedy_tie_logprob_tolerance']==1e-8


def numerical_group(actor):
    ask=actor.feature(obs('ask'));write=actor.feature(obs('write'))
    return {'ask_target':{'metadata':ask,'candidates':list(ASK_ACTIONS)},
            'write_targets':[{'metadata':write,'candidates':list(WRITE_ACTIONS)} for _ in range(3)],
            'action_pairs':[{'ask_index':i,'write_index':j} for i in range(3) for j in range(3)]}


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('qrow',[[-1.1,-.12,-.12],[.1,-.2,-.3]])
def test_identical_ask_q_rows_adam_cannot_flip_greedy_numerical_tie(device,qrow):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    actor=policy();group=numerical_group(actor)
    calls=actor.encoder_backend.model.base_model.calls
    actor.head.to(device)  # Only the tiny head, never an encoder, enters CUDA.
    lp=joint_logprobs(actor,group)
    q=torch.tensor(qrow*3,dtype=torch.float64,device=device)
    loss=session_loss(lp,lp.detach().clone(),q,1.,beta=.02)[0]
    optimizer=torch.optim.Adam(actor.head.parameters(),lr=.0003,eps=1e-6)
    loss.backward()
    assert float(actor.head[-1].weight.grad[:4].norm())<1e-12
    assert float(actor.head[-1].weight.grad[4:].norm())>1e-4
    torch.nn.utils.clip_grad_norm_(actor.head.parameters(),1.);optimizer.step()
    result=actor.decide(obs('ask'),list(ASK_ACTIONS),seed=101,sample=False,decision_id=f'after/{device}')
    assert result['action']=='PROCEED'
    assert abs(result['metadata']['probabilities'][0]-.5)<1e-8
    assert actor.encoder_backend.model.base_model.calls==calls


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_substantive_ask_advantage_crosses_tolerance_after_real_tiny_head_step(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    actor=policy();group=numerical_group(actor);actor.head.to(device)
    lp=joint_logprobs(actor,group)
    q=torch.tensor([-.1]*3+[1.]*3+[-.1]*3,dtype=torch.float64,device=device)
    loss=session_loss(lp,lp.detach().clone(),q,1.,beta=0.)[0]
    optimizer=torch.optim.Adam(actor.head.parameters(),lr=.0003,eps=1e-6)
    loss.backward();torch.nn.utils.clip_grad_norm_(actor.head.parameters(),1.);optimizer.step()
    result=actor.decide(obs('ask'),list(ASK_ACTIONS),seed=101,sample=False,decision_id=f'advantage/{device}')
    assert result['action']=='ASK_TEMPORAL_SCOPE'
    assert result['metadata']['gate_logprobs'][1]-result['metadata']['gate_logprobs'][0]>1e-5
