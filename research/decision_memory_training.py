"""Enumerated Monte Carlo action gradients for the memory controller.

All Q values are observed simulator outcomes of legal action forks, never
expert action targets. The actor stays fixed while each group is collected.
State selection is phase-uniform, then uniform among eligible visited states;
the n/4 factor recovers the mean-over-eight-decisions policy gradient.
The reference KL is an explicit local regularizer, not a future KL return.
"""
from __future__ import annotations

import hashlib
import math
from research.src.artifacts import digest


def module_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(__import__('torch').uint8).numpy().tobytes())
    return digest.hexdigest()


def enumerated_loss(logprobs, reference_logprobs, q_values, *, selection_weight=1., beta=.02):
    """Expected action gradient with a detached state baseline and exact KL."""
    import torch
    if logprobs.ndim != 1 or reference_logprobs.shape != logprobs.shape or q_values.shape != logprobs.shape:
        raise ValueError('Action values and categorical distributions must have identical vector shapes')
    if not 0 < selection_weight <= 1 or beta < 0:
        raise ValueError('Invalid state-selection weight or KL coefficient')
    if not all(bool(torch.isfinite(value).all()) for value in (logprobs, reference_logprobs, q_values)):
        raise ValueError('Nonfinite action distribution or observed return')
    q_values = q_values.detach()
    probabilities = logprobs.exp()
    baseline = (probabilities.detach() * q_values).sum().detach()
    advantages = (q_values - baseline).detach()
    if bool(q_values.eq(q_values[0]).all()):
        advantages = torch.zeros_like(q_values)
    policy_loss = -(probabilities * advantages).sum()
    kl = (logprobs.sum() * 0. if logprobs.detach().equal(reference_logprobs) else
          (probabilities * (logprobs - reference_logprobs)).sum())
    return selection_weight * (policy_loss + beta * kl), policy_loss, kl, baseline, advantages


def optimize_counterfactual(actor, reference_head, group, rewards, optimizer, *, beta=.02):
    """One update on a visited public state, after all legal forks finish."""
    import torch
    target = group['target']
    metadata = target['metadata']
    candidates = target['candidates']
    n = target['eligible_count']
    step = target['step_id']
    phase = group['phase']
    if (type(n) is not int or not 1 <= n <= 4
            or group['selection_weight'] != n / 4
            or target['selection_weight'] != n / 4
            or not math.isclose(target['within_phase_selection_probability'], 1/n, abs_tol=1e-12)
            or target['phase_selection_probability'] != .5
            or target['phase'] != phase or phase not in ('ask', 'write')
            or type(step) is not int or not 0 <= step <= 7
            or step % 2 != (0 if phase == 'ask' else 1)
            or target['session_index'] != step // 2
            or metadata['decision_id'] != target['decision_id']):
        raise ValueError('Malformed state selection, identity, phase or importance weight')
    if len(candidates) != len(rewards) or len(candidates) < 2:
        raise ValueError('One real terminal outcome is required for every eligible legal action')
    if group['diagnostics']['actor_after'] != actor.credit_parameter_fingerprint():
        raise ValueError('Actor parameters changed after collecting action outcomes')
    for index, branch in enumerate(group['branches']):
        proof = branch['counterfactual']
        if (proof['forced_action_index'] != index or proof['forced_action'] != candidates[index]
                or proof['target_step'] != step
                or proof['source_decision_id'] != target['decision_id']
                or branch['decisions'][step]['action'] != candidates[index]
                or branch['decisions'][step]['observation_sha256'] != target['observation_sha256']):
            raise ValueError('Fork ordering or public target state changed')
    if metadata['profile_sha256'] != actor.policy_profile()['profile_sha256']:
        raise ValueError('Collected controller profile changed')
    if metadata.get('candidates') != candidates:
        raise ValueError('Collected catalogue order changed')
    features = metadata['feature_vector']
    recomputed = actor.feature(target['observation'])
    if (recomputed['feature_sha256'] != metadata['feature_sha256']
            or digest(features) != metadata['feature_sha256']):
        raise ValueError('Stored feature does not match the visited public state')
    distribution = actor.distribution(features, candidates)
    old = torch.tensor(metadata['old_logprobs'], dtype=distribution.dtype, device=distribution.device)
    replay_error = float((distribution.detach() - old).abs().max().cpu())
    if replay_error > 2e-6:
        raise ValueError(f'Actor changed while collecting counterfactual outcomes: {replay_error}')
    with torch.no_grad():
        reference = actor.distribution(features, candidates, head=reference_head)
    q = torch.tensor([float(row['episodic_reward']) for row in rewards],
                     dtype=distribution.dtype, device=distribution.device)
    weight = float(group['selection_weight'])
    losses = enumerated_loss(distribution, reference, q, selection_weight=weight, beta=beta)
    optimizer.zero_grad(set_to_none=True)
    losses[0].backward()
    parameters = list(actor.head.parameters())
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise ValueError('Nonfinite controller gradient')
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        updated = actor.distribution(features, candidates).exp().cpu().tolist()
    return {'loss': float(losses[0].detach().cpu()), 'policy_loss': float(losses[1].detach().cpu()),
            'kl': float(losses[2].detach().cpu()), 'baseline': float(losses[3].cpu()),
            'advantages': losses[4].cpu().tolist(), 'returns': q.cpu().tolist(),
            'grad_norm': float(norm.cpu()), 'selection_weight': weight,
            'max_old_distribution_error': replay_error,
            'probabilities_before': distribution.detach().exp().cpu().tolist(),
            'probabilities_after': updated,
            'task_signal': any(row['mean_utility'] != rewards[0]['mean_utility'] for row in rewards[1:]),
            'return_signal': any(row['episodic_reward'] != rewards[0]['episodic_reward'] for row in rewards[1:])}
