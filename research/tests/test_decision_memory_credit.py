"""CPU tests of causal enumeration mechanics and its centered local objective."""
from copy import deepcopy
import json
import math
import random

import pytest

from research.budget_memory_data import make_episode
from research.budget_memory_env import canonical, fingerprint
from research.decision_memory_credit import (
    _ForkPolicy, collect_counterfactual_group, enumerate_credit,
)


class FakePolicy:
    def __init__(self, mutate=False):
        self.weight = .31
        self.calls = []
        self.ids = set()
        self.mutate = mutate

    def credit_parameter_fingerprint(self):
        return fingerprint([self.weight])

    def decide(self, observation, candidates, *, seed, sample, decision_id):
        assert decision_id not in self.ids
        self.ids.add(decision_id)
        self.calls.append((deepcopy(observation), list(candidates), seed, sample, decision_id))
        probabilities = [1.] if len(candidates) == 1 else [.55, .25, .20]
        action = random.Random(seed).choices(candidates, weights=probabilities)[0]
        if self.mutate:
            self.weight += .01
        return {"action": action, "metadata": {
            "decision_id": decision_id, "seed": seed, "sampled": sample,
            "probabilities": probabilities, "old_logprobs": [math.log(p) for p in probabilities],
            "action_index": candidates.index(action), "feature_vector": [1., observation['session_index']],
            "prompt_ids": [2, 3], "opaque_metadata": {"preserve": True}}}


class FakeReader:
    def __init__(self):
        self.calls = []
        self.ids = set()

    def answer(self, observation, candidates, *, seed, decision_id):
        assert decision_id not in self.ids
        self.ids.add(decision_id)
        self.calls.append((deepcopy(observation), seed, decision_id))
        return {"answer": "DEFER", "metadata": {"seed": seed, "decision_id": decision_id}}


@pytest.fixture
def episode():
    return make_episode('correction', 3, 202611061)


@pytest.fixture
def group(tmp_path, episode):
    return collect_counterfactual_group(FakePolicy(), FakeReader(), episode, 101, tmp_path, 'write')


def test_prefix_identity_and_exactly_one_forced_action(group):
    target = group['target']
    for branch, action in zip(group['branches'], target['candidates']):
        for original, replay in zip(group['prefix'], branch['decisions']):
            assert replay['observation'] == original['observation']
            assert replay['candidates'] == original['candidates']
            assert replay['action'] == original['action']
            assert replay['metadata']['credit_role'] == 'replayed_prefix'
            assert replay['metadata']['source_policy_output'] == original['policy_output']
            assert 'old_logprobs' not in replay['metadata']
        forced = branch['decisions'][target['step_id']]
        assert forced['action'] == action
        assert forced['observation'] == target['observation']
        assert forced['metadata']['credit_role'] == 'forced_counterfactual'
        assert forced['metadata']['policy_evaluated'] is False
        assert 'old_logprobs' not in forced['metadata']
        assert branch['counterfactual']['forced_decisions'] == 1
        for future in branch['decisions'][target['step_id']+1:]:
            assert future['metadata']['credit_role'] == 'sampled_suffix'
            assert future['metadata']['sampled'] is True
            assert future['metadata']['opaque_metadata'] == {'preserve': True}


def test_all_legal_actions_have_real_reader_outcomes_and_no_visitation_labels(tmp_path, episode):
    policy, reader = FakePolicy(), FakeReader()
    g = collect_counterfactual_group(policy, reader, episode, 102, tmp_path, 'ask')
    assert len(g['branches']) == len(g['target']['candidates']) == 3
    assert len(reader.calls) == 6
    assert g['diagnostics']['visitation_real_reader_calls'] == 0
    assert g['diagnostics']['visitation_stub_reader_calls'] == 2
    assert g['visitation']['outcome_not_evaluated'] is True
    assert not {'queries','metrics','future_answers','public_judge_payload'} & g['visitation'].keys()
    artifact = json.loads((tmp_path/'visitation/trace.json').read_text())
    assert artifact == g['visitation']
    assert all(len(b['queries']) == 2 for b in g['branches'])
    assert all(b['counterfactual']['use_for_whole_episode_LOO'] is False for b in g['branches'])


def test_common_future_randomness_is_new_and_independent_of_selector(group):
    d = group['diagnostics']
    assert len({d['visit_seed'], d['selector_seed'], d['fork_future_seed']}) == 3
    assert all(b['rollout_seed'] == d['fork_future_seed'] for b in group['branches'])
    assert group['visitation']['rollout_seed'] == d['visit_seed']
    assert all([x['seed'] for x in b['decisions']] == d['common_decision_seeds'] for b in group['branches'])
    assert all([q['reader_metadata']['seed'] for q in b['queries']] == d['common_reader_seeds'] for b in group['branches'])


def test_actual_actor_not_called_for_replayed_or_forced_decisions(tmp_path, episode):
    policy = FakePolicy()
    g = collect_counterfactual_group(policy, FakeReader(), episode, 103, tmp_path, 'write')
    target = g['target']['step_id']
    assert len(policy.calls) == 8 + 3*(7-target)
    assert len(policy.ids) == len(policy.calls)
    assert all(call[3] is True for call in policy.calls)
    assert g['diagnostics']['actor_before'] == g['diagnostics']['actor_after']


def test_namespace_is_unique_across_groups_with_common_episode_seed(tmp_path, episode):
    policy, reader = FakePolicy(), FakeReader()
    a = collect_counterfactual_group(policy,reader,episode,101,tmp_path/'a','write')
    b = collect_counterfactual_group(policy,reader,episode,101,tmp_path/'b','write')
    assert a['target']['step_id'] == b['target']['step_id']
    assert a['target']['observation'] == b['target']['observation']
    ids_a = {d['decision_id'] for x in a['branches'] for d in x['decisions']}
    ids_b = {d['decision_id'] for x in b['branches'] for d in x['decisions']}
    assert not ids_a & ids_b


def test_actor_mutation_during_collection_fails_before_credit(tmp_path, episode):
    with pytest.raises(ValueError,match='Actor changed'):
        collect_counterfactual_group(FakePolicy(mutate=True), FakeReader(), episode, 101, tmp_path,'write')


def test_prefix_mismatch_is_not_repaired(group):
    target = group['target']
    wrapper = _ForkPolicy(FakePolicy(), group['visitation']['decisions'], target['step_id'], target['candidates'][0])
    bad = deepcopy(target['observation']); bad['date'] = '2000-01-01'
    with pytest.raises(ValueError, match='not identical'):
        wrapper.decide(bad,target['candidates'],seed=1,sample=True,decision_id='new')


def test_eligible_count_weight_and_pre_action_measurement(tmp_path, episode):
    for phase in ('ask','write'):
        g = collect_counterfactual_group(FakePolicy(),FakeReader(),episode,107,tmp_path/phase,phase)
        eligible = [d for d in g['visitation']['decisions'] if d['phase']==phase and len(d['candidates'])>1]
        assert g['target']['eligible_count'] == len(eligible)
        assert g['selection_weight'] == len(eligible)/4
        assert g['target']['within_phase_selection_probability'] == 1/len(eligible)
        assert g['target']['step_id'] in [d['step_id'] for d in eligible]
    assert len(eligible) == 4  # every WRITE state remains legal, including NOOP


def test_n_over_four_cancels_conditional_index_selection_pointwise():
    gradients = [2., -1., 5., .5]
    weighted, desired, unweighted = 0., 0., 0.
    ask_probability = .25
    for first_ask in range(5):  # 4 denotes no question during the episode
        probability = ((1-ask_probability)**first_ask * ask_probability
                       if first_ask < 4 else (1-ask_probability)**4)
        n = min(first_ask+1,4)
        selected_mean = sum(gradients[:n])/n
        assert (n/4)*selected_mean == pytest.approx(sum(gradients[:n])/4)
        weighted += probability*(n/4)*selected_mean
        unweighted += probability*selected_mean
        desired += probability*sum(gradients[:n])/4
    assert weighted == pytest.approx(desired)
    assert unweighted != pytest.approx(desired)


def test_past_cost_excludes_target_and_is_shared_by_all_forks(group):
    past = sum(.1 if d['phase']=='ask' and d['action']!='PROCEED' else
               .02 if d['phase']=='write' and d['action']!='NOOP' else 0.
               for d in group['prefix'])
    assert group['diagnostics']['past_cost'] == pytest.approx(past)
    for branch in group['branches']:
        assert branch['counterfactual']['past_cost'] == pytest.approx(past)
        assert branch['counterfactual']['future_cost'] == pytest.approx(branch['metrics']['costs']-past)
        assert branch['counterfactual']['future_cost'] >= 0


def test_credit_centers_full_distribution_without_action_labels(group):
    values = [-.2,.8,.1]
    credit = enumerate_credit(group, [{'episodic_reward': x} for x in values])
    baseline = sum(p*q for p,q in zip(group['behavior_probs'],values))
    assert credit['baseline_full'] == pytest.approx(baseline)
    assert credit['weighted_advantage_mean'] == pytest.approx(0,abs=1e-12)
    assert credit['advantages'] == pytest.approx([q-baseline for q in values])
    assert credit['future_returns'] == pytest.approx([q+credit['past_cost'] for q in values])
    assert credit['target_action'] is None and credit['whole_episode_LOO'] is False
    assert all(type(x) is float for x in credit['advantages'])


def test_target_cost_remains_in_future_credit(group):
    # Same outcome and later costs: a target STORE still costs .02 more than NOOP.
    credit = enumerate_credit(group, [0.,-.02,-.02])
    assert credit['future_returns'][0]-credit['future_returns'][1] == pytest.approx(.02)
    assert credit['advantages'][0]-credit['advantages'][1] == pytest.approx(.02)


def test_centered_gradient_matches_full_and_future_returns(group):
    torch = pytest.importorskip('torch')
    q = [-.2,.8,.1]
    credit = enumerate_credit(group,q)
    def gradient(values):
        logits = torch.tensor([.1,-.2,.5],dtype=torch.float64,requires_grad=True)
        loss = -group['selection_weight']*(logits.softmax(-1)*torch.tensor(values,dtype=torch.float64)).sum()
        return torch.autograd.grad(loss,logits)[0]
    assert torch.allclose(gradient(q),gradient(credit['advantages']),atol=1e-12,rtol=0)
    assert torch.allclose(gradient(q),gradient(credit['future_returns']),atol=1e-12,rtol=0)


def test_no_reward_is_permitted_for_visitation_path(group):
    bad = deepcopy(group);bad['branches'][0] = bad['visitation']
    with pytest.raises(ValueError,match='Only actual'):
        enumerate_credit(bad,[0.,0.,0.])
    with pytest.raises(ValueError,match='Every legal'):
        enumerate_credit(group,[0.,0.])
    with pytest.raises(ValueError,match='finite'):
        enumerate_credit(group,[0.,float('nan'),0.])


def test_invalid_phase_or_reuse_fails_closed(tmp_path, episode):
    with pytest.raises(ValueError,match='phase'):
        collect_counterfactual_group(FakePolicy(),FakeReader(),episode,1,tmp_path,'teacher')
    collect_counterfactual_group(FakePolicy(),FakeReader(),episode,1,tmp_path,'write')
    with pytest.raises(FileExistsError):
        collect_counterfactual_group(FakePolicy(),FakeReader(),episode,1,tmp_path,'write')
