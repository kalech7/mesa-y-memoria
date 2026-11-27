"""CPU contract tests of exogenous targets and detached prefix IS."""
from copy import deepcopy
import math

import pytest

from research.budget_memory_data import make_episode
from research.decision_memory_credit import collect_counterfactual_group as legacy_collect, _seed
from research.decision_memory_rollin import collect_counterfactual_group, enumerate_credit, SAMPLING_CONTRACT
from research.tests.test_decision_memory_credit import FakePolicy, FakeReader


@pytest.mark.parametrize('session',[0,1,2,3])
def test_ask_rollin_reaches_all_sessions_and_records_exact_ratio(tmp_path,session):
    episode=make_episode('context',3,202611061)
    policy,reader=FakePolicy(),FakeReader()
    g=collect_counterfactual_group(policy,reader,episode,101,tmp_path,'ask',session)
    assert g['sampling_contract']==SAMPLING_CONTRACT
    assert g['target']['session_index']==session and g['target']['step_id']==2*session
    assert g['target']['observation']['questions_remaining']==1
    assert g['selection_weight']==pytest.approx(.55**session)
    assert g['diagnostics']['proceed_prefix_logprobs']==pytest.approx([math.log(.55)]*session)
    assert g['diagnostics']['source_path_on_policy'] == (session==0)
    assert g['diagnostics']['importance_clipped'] is False
    assert g['diagnostics']['importance_batch_renormalized'] is False
    assert len(reader.calls)==6 and g['diagnostics']['rollin_real_reader_calls']==0
    assert not {'queries','metrics','public_judge_payload'} & g['rollin'].keys()
    for decision in g['prefix']:
        if decision['phase']=='ask':
            assert decision['action']=='PROCEED'
            assert decision['metadata']['credit_role']=='forced_prefix_proceed'
            assert 'old_logprobs' not in decision['metadata']
            assert decision['metadata']['proposal_probability']==1
        else:
            assert decision['metadata']['credit_role']=='sampled_rollin_prefix'
    assert len(policy.calls)==2*session+1+3*(7-2*session)


@pytest.mark.parametrize('session',[0,1,2,3])
def test_write_prefix_is_on_policy_with_unit_weight(tmp_path,session):
    g=collect_counterfactual_group(FakePolicy(),FakeReader(),make_episode('retention',2,61),9,tmp_path,'write',session)
    assert g['selection_weight']==1
    assert g['diagnostics']['source_path_on_policy']
    assert g['diagnostics']['proceed_prefix_logprobs']==[]
    assert all(d['metadata']['credit_role']=='sampled_rollin_prefix' for d in g['prefix'])
    assert 'eligible_count' not in g['target']
    assert g['target']['within_phase_selection_probability']==.25


def test_default_write_target_and_rng_match_legacy_to_isolate_change(tmp_path):
    episode=make_episode('correction',7,61)
    old=legacy_collect(FakePolicy(),FakeReader(),episode,42,tmp_path/'old','write')
    new=collect_counterfactual_group(FakePolicy(),FakeReader(),episode,42,tmp_path/'new','write')
    assert old['target']['step_id']==new['target']['step_id']
    assert old['target']['observation']==new['target']['observation']
    assert old['diagnostics']['visit_seed']==new['diagnostics']['visit_seed']==_seed(42,'manager_visitation')
    assert old['diagnostics']['selector_seed']==new['diagnostics']['selector_seed']==_seed(42,'target_index')
    assert old['diagnostics']['fork_future_seed']==new['diagnostics']['fork_future_seed']==_seed(42,'fork_future')
    for a,b in zip(old['branches'],new['branches']):
        assert [d['observation'] for d in a['decisions']]==[d['observation'] for d in b['decisions']]
        assert [d['action'] for d in a['decisions']]==[d['action'] for d in b['decisions']]
        assert a['metrics']==b['metrics']


def test_fork_replay_identity_and_new_common_future_seeds(tmp_path):
    g=collect_counterfactual_group(FakePolicy(),FakeReader(),make_episode('temporal',1,61),19,tmp_path,'ask',2)
    for branch in g['branches']:
        for original,replayed in zip(g['prefix'],branch['decisions']):
            assert original['observation']==replayed['observation'] and original['action']==replayed['action']
        assert branch['decisions'][4]['metadata']['credit_role']=='forced_counterfactual'
        assert all(d['metadata']['credit_role']=='sampled_suffix' for d in branch['decisions'][5:])
        assert [d['seed'] for d in branch['decisions']]==g['diagnostics']['common_decision_seeds']
        assert [q['reader_metadata']['seed'] for q in branch['queries']]==g['diagnostics']['common_reader_seeds']
    assert len({g['diagnostics'][k] for k in ('visit_seed','selector_seed','fork_future_seed')})==3


def test_credit_has_detached_is_not_legacy_eligible_weight(tmp_path):
    g=collect_counterfactual_group(FakePolicy(),FakeReader(),make_episode('retention',1,61),91,tmp_path,'ask',3)
    c=enumerate_credit(g,[-.1,.3,.2])
    assert c['selection_weight']==pytest.approx(.55**3)
    assert 'eligible_count' not in c
    assert type(c['prefix_importance']) is float and type(c['baseline_full']) is float
    assert c['weighted_advantage_mean']==pytest.approx(0,abs=1e-12)
    assert c['target_action'] is None
    assert c['future_returns']==pytest.approx([q+c['past_cost'] for q in c['full_returns']])
    assert g['diagnostics']['actor_before']==g['diagnostics']['actor_after']


def test_invalid_or_overwritten_rollin_and_actor_mutation_fail(tmp_path):
    episode=make_episode('retention',1,61)
    with pytest.raises(ValueError,match='0..3'):
        collect_counterfactual_group(FakePolicy(),FakeReader(),episode,1,tmp_path/'bad','ask',4)
    with pytest.raises(ValueError,match='Actor changed'):
        collect_counterfactual_group(FakePolicy(mutate=True),FakeReader(),episode,1,tmp_path/'mutate','ask',1)
    collect_counterfactual_group(FakePolicy(),FakeReader(),episode,1,tmp_path/'ok','ask',1)
    with pytest.raises(FileExistsError):
        collect_counterfactual_group(FakePolicy(),FakeReader(),episode,1,tmp_path/'ok','ask',1)
