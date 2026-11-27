"""Independent CPU audit of complete-group RLOO gradient and decision identities."""
from contextlib import contextmanager
from copy import deepcopy

import pytest
import torch

from research.budget_memory_training import leave_one_out, optimize_group


class ToyPolicy:
    def __init__(self, trainable):
        self.torch = torch
        self.model = torch.nn.Linear(3, 8, bias=False)
        with torch.no_grad():
            self.model.weight.copy_(torch.arange(24).reshape(8, 3) / 40)
        self.model.requires_grad_(trainable)

    @property
    def device(self):
        return self.model.weight.device

    @contextmanager
    def likelihood_mode(self):
        yield

    def policy_profile(self):
        return {'profile_sha256': 'CPU-toy-protocol-v1'}

    def candidate_token_ids(self, candidates):
        return [list(map(ord, candidate)) + [0] for candidate in candidates]

    def distribution(self, prompt_ids, candidates):
        return self.model.weight[prompt_ids[0], :len(candidates)].log_softmax(-1)


def trajectories(policy, *, all_forced=False):
    output = []
    for rollout in range(4):
        decisions = []
        for step in range(8):
            forced = all_forced or step in (2, 4, 6)
            candidates = ['PROCEED'] if forced else ['A', 'B', 'C']
            old = [0.] if forced else policy.distribution([step], candidates).detach().tolist()
            index = 0 if forced else (rollout + step) % 3
            identifier = f'root/seed{rollout}/session{step//2}/step{step}'
            decisions.append({'decision_id': identifier, 'phase': 'ask' if step%2 == 0 else 'write',
                              'candidates': candidates, 'action': candidates[index],
                              'metadata': {'decision_id': identifier, 'prompt_ids': [step],
                                           'old_logprobs': old, 'action_index': index,
                                           'candidates': candidates, 'profile_sha256': policy.policy_profile()['profile_sha256'],
                                           'candidate_token_ids': policy.candidate_token_ids(candidates)}})
        output.append({'decisions': decisions})
    return output


def test_complete_group_update_matches_independent_categorical_gradient_and_denominator():
    actor, reference = ToyPolicy(True), ToyPolicy(False)
    group = trajectories(actor)
    returns = [1., 0., -.5, .2]
    initial = actor.model.weight.detach().clone()
    expected = initial.clone().requires_grad_()
    objective = torch.zeros(())
    for trajectory, advantage in zip(group, leave_one_out(returns), strict=True):
        for decision in trajectory['decisions']:
            if len(decision['candidates']) == 1:
                continue
            metadata = decision['metadata']
            logp = expected[metadata['prompt_ids'][0]].log_softmax(-1)[metadata['action_index']]
            objective = objective - advantage * logp / (4 * 8)
    objective.backward()
    assert expected.grad.norm() < 1  # Clipping does not confound the exact gradient comparison.
    optimizer = torch.optim.SGD(actor.model.parameters(), lr=.1)
    result = optimize_group(actor, reference, group, returns, optimizer, beta=0.)
    torch.testing.assert_close(actor.model.weight, initial - .1 * expected.grad, rtol=0, atol=1e-8)
    assert result['reference_cache_keys'] == 20
    assert result['decision_count'] == 32
    assert result['max_old_distribution_error'] == 0
    assert reference.device.type == 'cpu'
    torch.testing.assert_close(reference.model.weight, initial, rtol=0, atol=0)


def test_flat_real_cost_return_has_no_task_update_at_identical_reference():
    actor, reference = ToyPolicy(True), ToyPolicy(False)
    before = actor.model.weight.detach().clone()
    result = optimize_group(actor, reference, trajectories(actor), [.38] * 4,
                            torch.optim.SGD(actor.model.parameters(), lr=.1), beta=.03)
    assert result['advantages'] == [0.] * 4 and result['zero_signal']
    torch.testing.assert_close(actor.model.weight, before, rtol=0, atol=0)


def test_repeated_session_decision_id_and_stale_old_distribution_abort_before_step():
    actor, reference = ToyPolicy(True), ToyPolicy(False)
    optimizer = torch.optim.SGD(actor.model.parameters(), lr=.1)
    initial = actor.model.weight.detach().clone()
    group = trajectories(actor)
    group[1]['decisions'][1]['metadata']['decision_id'] = group[0]['decisions'][1]['metadata']['decision_id']
    group[1]['decisions'][1]['decision_id'] = group[0]['decisions'][1]['decision_id']
    with pytest.raises(ValueError, match='Repeated'):
        optimize_group(actor, reference, group, [1., 0., 0., 0.], optimizer)
    stale = trajectories(actor)
    stale[0]['decisions'][1]['metadata']['old_logprobs'][0] += .1
    with pytest.raises(ValueError, match='distribution mismatch'):
        optimize_group(actor, reference, stale, [1., 0., 0., 0.], optimizer)
    torch.testing.assert_close(actor.model.weight, initial, rtol=0, atol=0)
    assert reference.device.type == 'cpu'


def test_all_forced_catalogues_have_exact_zero_gradient_and_keep_decision_denominator():
    actor, reference = ToyPolicy(True), ToyPolicy(False)
    initial = actor.model.weight.detach().clone()
    result = optimize_group(actor, reference, trajectories(actor, all_forced=True), [0.] * 4,
                            torch.optim.SGD(actor.model.parameters(), lr=.1))
    assert result['decision_count'] == 32 and result['reference_cache_keys'] == 0
    assert result['grad_norm'] == 0 and result['zero_signal']
    torch.testing.assert_close(actor.model.weight, initial, rtol=0, atol=0)


@pytest.mark.parametrize('poison', ['catalogue', 'tokens', 'action', 'profile'])
def test_collected_identity_integrity_failures_abort_before_parameter_update(poison):
    actor, reference = ToyPolicy(True), ToyPolicy(False)
    group = trajectories(actor)
    decision = group[0]['decisions'][1]
    if poison == 'catalogue':
        decision['candidates'] = list(reversed(decision['candidates']))
    elif poison == 'tokens':
        decision['metadata']['candidate_token_ids'][0][0] += 1
    elif poison == 'action':
        decision['action'] = 'WRONG'
    else:
        decision['metadata']['profile_sha256'] = 'WRONG'
    initial = actor.model.weight.detach().clone()
    with pytest.raises(ValueError):
        optimize_group(actor, reference, group, [1., 0., 0., 0.],
                       torch.optim.SGD(actor.model.parameters(), lr=.1))
    torch.testing.assert_close(actor.model.weight, initial, rtol=0, atol=0)
