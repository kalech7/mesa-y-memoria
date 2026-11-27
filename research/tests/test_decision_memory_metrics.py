"""Post-action provenance diagnostics never supply an actor action target."""
from copy import deepcopy
import math

import pytest

from research.budget_memory_data import make_episode
from research.budget_memory_env import run_episode, score_trace
from research.decision_memory_metrics import metrics, sampling_metrics


class FixedActions:
    def __init__(self,write='STORE_SLOT_1',ask_index=None,ask_type='ASK_TEMPORAL_SCOPE'):
        self.write,self.ask_index,self.ask_type=write,ask_index,ask_type
    def decide(self,observation,candidates,**kwargs):
        if observation['decision_type']=='write':
            return self.write
        return self.ask_type if observation['session_index']==self.ask_index and len(candidates)>1 else 'PROCEED'


class DeferReader:
    def answer(self,observation,candidates,**kwargs):return 'DEFER'


def rollout(tmp_path,family='retention',**options):
    episode=make_episode(family,0,202611061)
    return run_episode(FixedActions(**options),DeferReader(),episode,101,tmp_path),episode


def test_neutral_writes_and_noops_are_literal_fact_provenance_only(tmp_path):
    stored,episode=rollout(tmp_path/'stores')
    result=metrics(stored,episode)
    assert result['write_count']==4 and result['noop_count']==0
    assert result['neutral_write_count']==2 and result['fact_write_count']==2
    assert sum(result[f'write_session_{session}_count'] for session in range(1,5))==4
    skipped,_=rollout(tmp_path/'noops',write='NOOP')
    result2=metrics(skipped,episode)
    assert result2['write_count']==0 and result2['noop_count']==4
    assert result2['neutral_noop_count']==2 and result2['fact_noop_count']==2
    assert all(type(value) in (int,float) for value in result.values())
    original=score_trace(stored)
    assert all(result[key]==value for key,value in original.items() if type(value) in (int,float))


def test_temporary_fact_is_not_misclassified_as_neutral_because_future_queries_expire_it(tmp_path):
    trace,episode=rollout(tmp_path,'temporary_distractor')
    result=metrics(trace,episode)
    assert result['neutral_write_count']==1 and result['fact_write_count']==3
    assert result['temporal_fact_write_count']==1
    assert result['ask_count']==result['informative_ask_count']==0


def temporal_fixture():
    for seed in range(20):
        episode=make_episode('temporal',0,seed)
        worlds=episode['private']['worlds']
        for index in range(4):
            texts={world['replies'][str(index)]['ASK_TEMPORAL_SCOPE']['text'] for world in worlds}
            if len(texts)==3:
                return episode,index
    raise AssertionError('Fixture requires one available authored temporal clarification')


def test_informative_ask_counts_real_before_after_worlds_not_ask_name(tmp_path):
    episode,index=temporal_fixture()
    informative=run_episode(FixedActions(ask_index=index),DeferReader(),episode,101,tmp_path/'informative')
    result=metrics(informative,episode)
    assert result['informative_ask_count']==1 and result['uninformative_ask_count']==0
    assert result['ask_worlds_removed']==2
    assert result['ask_information_gain_bits']==pytest.approx(math.log2(3))
    assert result[f'ask_session_{index+1}_count']==1
    assert result['ask_temporal_scope_count']==1
    wrong=run_episode(FixedActions(ask_index=index,ask_type='ASK_CURRENT_CONTEXT'),DeferReader(),episode,101,tmp_path/'wrong')
    result2=metrics(wrong,episode)
    assert result2['informative_ask_count']==0 and result2['uninformative_ask_count']==1
    assert result2['ask_worlds_removed']==0 and result2['ask_information_gain_bits']==0
    assert result2['ask_current_context_count']==1
    changed=deepcopy(episode)
    changed['private']['actual_world_id']=changed['private']['worlds'][-1]['world_id']
    changed['queries']=[{'private_test_only':'not consulted by diagnostics'}]
    assert metrics(informative,changed)==result


def test_sampling_gate_summary_excludes_greedy_replay_and_forced_records():
    def row(phase,p,**overrides):
        return {'phase':phase,'candidates':['a','b','c'],'metadata':{
            'sampled':True,'gate_logprobs':[math.log(1-p),math.log(p)],
            'conditional_logprobs':[math.log(.3),math.log(.7)],**overrides}}
    rows=[row('ask',.2),row('ask',.6),row('write',.8),
          row('write',.99,credit_role='forced_counterfactual'),
          row('write',.99,credit_role='replayed_prefix'),row('write',.99,sampled=False),
          row('ask',.99,forced_action=True)]
    rows.append({'phase':'ask','candidates':['PROCEED'],'metadata':{'sampled':True}})
    result=sampling_metrics({'decisions':rows})
    assert result['sampled_ask_gate_count']==2 and result['sampled_store_gate_count']==1
    assert result['sampled_ask_gate_probability']==pytest.approx(.4)
    assert result['sampled_store_gate_probability']==pytest.approx(.8)
    assert result['sampled_store_first_conditional_probability']==pytest.approx(.3)
    assert sampling_metrics({'decisions':[]})=={'sampled_ask_gate_count':0,'sampled_store_gate_count':0}


def test_metric_input_audit_rejects_wrong_episode_visitation_and_counter_drift(tmp_path):
    trace,episode=rollout(tmp_path)
    bad=deepcopy(trace);bad['outcome_not_evaluated']=True
    with pytest.raises(ValueError,match='Visitation'):metrics(bad,episode)
    bad=deepcopy(trace);bad['episode_id']='another'
    with pytest.raises(ValueError,match='identities'):metrics(bad,episode)
    bad=deepcopy(trace);bad['write_count']=3
    with pytest.raises(ValueError,match='counters'):metrics(bad,episode)


def test_diagnostics_do_not_mutate_trace_episode_or_policy_observation(tmp_path):
    trace,episode=rollout(tmp_path,'correction')
    before_trace,before_episode=deepcopy(trace),deepcopy(episode)
    result=metrics(trace,episode)
    assert trace==before_trace and episode==before_episode
    assert result['fact_write_count']==3 and result['neutral_write_count']==1
    assert all('neutral' not in decision['observation'] for decision in trace['decisions'])
