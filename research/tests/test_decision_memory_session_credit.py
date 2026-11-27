"""CPU transport and credit tests for the complete session intervention."""
from copy import deepcopy
import math
import random

import pytest

from research.budget_memory_data import make_episode
from research.decision_memory_credit import _seed
from research.decision_memory_session_credit import collect_session_group, enumerate_session_credit, SAMPLING_CONTRACT
from research.tests.test_decision_memory_credit import FakePolicy,FakeReader


def _seed_for_component(component, *, first_ask=None):
    for seed in range(1000):
        candidate='defer_ask' if random.Random(_seed(seed,'rollin_mixture')).random()<.5 else 'on_policy'
        first_call_seed=random.Random(_seed(seed,'manager_visitation')).randrange(2**31)
        takes_ask=random.Random(first_call_seed).choices([False,True,True],weights=[.55,.25,.20])[0]
        if candidate==component and (first_ask is None or takes_ask==first_ask):
            return seed
    raise AssertionError('No mechanical RNG fixture seed found')


class ConditionalPolicy(FakePolicy):
    def decide(self,observation,candidates,**kwargs):
        result=super().decide(observation,candidates,**kwargs)
        if observation['decision_type']=='write':
            questions=[t['text'] for t in observation['exchange'] if t['role']=='assistant']
            if any('¿Esto es' in text for text in questions):
                probabilities=[.2,.3,.5]
            elif any('¿En qué lugar' in text for text in questions):
                probabilities=[.4,.5,.1]
            else:
                probabilities=[.55,.25,.20]
            action=random.Random(kwargs['seed']).choices(candidates,weights=probabilities)[0]
            result['action']=action
            result['metadata'].update(probabilities=probabilities,old_logprobs=[math.log(p) for p in probabilities],
                                      action_index=candidates.index(action))
        return result


@pytest.fixture
def episode():
    return make_episode('context',3,202611061)


@pytest.fixture
def group(tmp_path,episode):
    return collect_session_group(ConditionalPolicy(),FakeReader(),episode,101,tmp_path,0)


def test_available_budget_enumerates_all_nine_pairs_and_conditional_observations(group):
    assert group['sampling_contract']==SAMPLING_CONTRACT
    assert group['selection_weight']==1
    assert len(group['branches'])==len(group['action_pairs'])==9
    assert len(group['write_targets'])==len(group['ask_target']['candidates'])==3
    assert sum(group['behavior_probs'])==pytest.approx(1)
    assert len({(p['ask_index'],p['write_index']) for p in group['action_pairs']})==9
    for index,pair in enumerate(group['action_pairs']):
        branch=group['branches'][index]
        assert pair['branch_index']==index
        assert branch['counterfactual']['ask_index']==pair['ask_index']
        assert branch['counterfactual']['write_index']==pair['write_index']
        assert branch['decisions'][0]['action']==pair['ask_action']
        assert branch['decisions'][1]['action']==pair['write_action']
        target=group['write_targets'][pair['ask_index']]
        assert target['observation']==branch['decisions'][1]['observation']
        assert target['step_id']==1 and group['ask_target']['step_id']==0
        expected=group['behavior_ask_probs'][pair['ask_index']]*group['behavior_write_probs'][pair['ask_index']][pair['write_index']]
        assert group['behavior_probs'][index]==pytest.approx(expected)
    # WRITE probability genuinely depends on the emitted reply/question branch.
    assert len({tuple(p) for p in group['behavior_write_probs']})==3
    assert len(group['write_targets'][0]['observation']['exchange'])==1
    assert len(group['write_targets'][1]['observation']['exchange'])==3
    assert len(group['write_targets'][2]['observation']['exchange'])==3


def test_spent_budget_has_three_pairs_and_marginal_mixture_ratio_two(tmp_path,episode):
    seed=_seed_for_component('on_policy',first_ask=True)
    g=collect_session_group(FakePolicy(),FakeReader(),episode,seed,tmp_path,1)
    assert g['diagnostics']['asked_before']
    assert g['diagnostics']['mixture_component']=='on_policy'
    assert g['selection_weight']==2
    assert g['ask_target']['candidates']==['PROCEED']
    assert g['behavior_ask_probs']==[1.]
    assert len(g['branches'])==3 and len(g['write_targets'])==1
    assert g['diagnostics']['no_ask_prefix_probability'] is None
    assert g['diagnostics']['proceed_prefix_logprobs']==[]
    assert sum(g['behavior_probs'])==pytest.approx(1)


@pytest.mark.parametrize('component',['defer_ask','on_policy'])
def test_unspent_prefix_uses_marginal_mixture_density_not_component_density(tmp_path,episode,component):
    seed=_seed_for_component(component,first_ask=False)
    g=collect_session_group(FakePolicy(),FakeReader(),episode,seed,tmp_path,1)
    assert not g['diagnostics']['asked_before']
    assert g['diagnostics']['mixture_component']==component
    assert g['diagnostics']['no_ask_prefix_probability']==pytest.approx(.55)
    assert g['selection_weight']==pytest.approx(2*.55/(1+.55))
    assert g['diagnostics']['importance_detached']
    assert not g['diagnostics']['importance_clipped']
    assert not g['diagnostics']['importance_batch_renormalized']


def test_prefix_replay_two_forced_actions_and_sampled_future_with_common_randomness(tmp_path,episode):
    seed=_seed_for_component('defer_ask')
    g=collect_session_group(FakePolicy(),FakeReader(),episode,seed,tmp_path,2)
    assert g['diagnostics']['source_path_on_policy'] is False
    for branch in g['branches']:
        assert branch['counterfactual']['forced_decisions']==2
        assert branch['counterfactual']['replayed_prefix_decisions']==4
        assert branch['counterfactual']['sampled_future_decisions']==2
        assert [d['seed'] for d in branch['decisions']]==g['diagnostics']['common_decision_seeds']
        assert [q['reader_metadata']['seed'] for q in branch['queries']]==g['diagnostics']['common_reader_seeds']
        for source,replay in zip(g['prefix'],branch['decisions']):
            assert source['observation']==replay['observation']
            assert source['action']==replay['action']
            assert replay['metadata']['credit_role']=='replayed_prefix'
        assert branch['decisions'][4]['metadata']['credit_role']=='forced_counterfactual'
        assert branch['decisions'][5]['metadata']['credit_role']=='forced_session_write'
        assert branch['decisions'][5]['metadata']['sampled'] is False
        assert all(d['metadata']['credit_role']=='sampled_suffix' for d in branch['decisions'][6:])


def test_no_real_reader_in_rollin_actor_is_fixed_and_namespaces_unique(tmp_path,episode):
    policy,reader=FakePolicy(),FakeReader()
    a=collect_session_group(policy,reader,episode,101,tmp_path/'a',0)
    b=collect_session_group(policy,reader,episode,101,tmp_path/'b',0)
    assert len(reader.calls)==36
    assert a['diagnostics']['rollin_real_reader_calls']==0
    assert a['diagnostics']['fork_real_reader_calls']==18
    assert a['diagnostics']['actor_before']==a['diagnostics']['actor_after']
    assert not {'queries','metrics','public_judge_payload'} & a['rollin'].keys()
    ids_a={d['decision_id'] for trace in a['branches'] for d in trace['decisions']}
    ids_b={d['decision_id'] for trace in b['branches'] for d in trace['decisions']}
    assert not ids_a & ids_b


def test_credit_is_centered_detached_and_never_creates_an_action_target(group):
    values=[-.2,.1,.3,-.1,.8,.2,-.4,.1,.5]
    c=enumerate_session_credit(group,values)
    assert c['baseline_full']==pytest.approx(sum(p*q for p,q in zip(group['behavior_probs'],values)))
    assert c['weighted_advantage_mean']==pytest.approx(0,abs=1e-12)
    assert all(type(x) is float for x in c['advantages'])
    assert c['target_action'] is None and c['whole_episode_LOO'] is False
    assert c['future_returns']==pytest.approx([q+c['past_cost'] for q in values])


def test_both_target_costs_remain_in_future_credit(tmp_path,episode):
    seed=_seed_for_component('defer_ask')
    g=collect_session_group(FakePolicy(),FakeReader(),episode,seed,tmp_path,3)
    rewards=[b['metrics']['net_return'] for b in g['branches']]  # fake reader always DEFER
    c=enumerate_session_credit(g,rewards)
    baseline_pair=next(p['branch_index'] for p in g['action_pairs'] if p['ask_action']=='PROCEED' and p['write_action']=='NOOP')
    cost_pair=next(p['branch_index'] for p in g['action_pairs'] if p['ask_action']=='ASK_TEMPORAL_SCOPE' and p['write_action']=='STORE_SLOT_1')
    assert c['future_returns'][baseline_pair]-c['future_returns'][cost_pair]==pytest.approx(.12)
    assert all(b['counterfactual']['past_cost']==pytest.approx(g['diagnostics']['past_cost']) for b in g['branches'])


def test_centering_preserves_gradients_of_ask_and_every_conditional_write(group):
    torch=pytest.importorskip('torch')
    values=[-.2,.1,.3,-.1,.8,.2,-.4,.1,.5]
    def gradient(q):
        ask=torch.tensor([.1,.2,-.1],dtype=torch.float64,requires_grad=True)
        write=torch.tensor([[.1,.2,.3],[.2,-.2,.1],[.3,-.1,.2]],dtype=torch.float64,requires_grad=True)
        joint=(ask.softmax(-1)[:,None]*write.softmax(-1)).reshape(-1)
        loss=-(joint*torch.tensor(q,dtype=torch.float64)).sum()
        return torch.autograd.grad(loss,(ask,write))
    q=enumerate_session_credit(group,values)
    for a,b in zip(gradient(values),gradient(q['advantages'])):
        assert torch.allclose(a,b,atol=1e-12,rtol=0)
        assert bool(a.ne(0).any())


def test_invalid_contract_session_overwrite_or_parameter_mutation_fails(tmp_path,episode,group):
    with pytest.raises(ValueError,match='Session'):
        collect_session_group(FakePolicy(),FakeReader(),episode,1,tmp_path/'invalid',4)
    with pytest.raises(ValueError,match='Actor changed'):
        collect_session_group(FakePolicy(mutate=True),FakeReader(),episode,1,tmp_path/'mutate',1)
    with pytest.raises(FileExistsError):
        collect_session_group(FakePolicy(),FakeReader(),episode,1,tmp_path,1)
    bad=deepcopy(group);bad['sampling_contract']='uniform_eligible_old'
    with pytest.raises(ValueError,match='session-joint'):
        enumerate_session_credit(bad,[0.]*9)
