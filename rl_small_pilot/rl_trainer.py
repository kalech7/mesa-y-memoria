"""Monte Carlo REINFORCE on real neural parameters; no language-model training.

The scalar baseline contains returns of earlier batches only. It is frozen
before each batch, so it cannot depend on the sampled action being scored.
"""
from collections import Counter
from copy import deepcopy
import random
from statistics import mean
from time import monotonic

import torch

from .small_policy import seed_everything
from .supervised_baseline import _factory, _reset, _score, evaluate_policy


def discounted_returns(rewards, gamma=1):
    values, running = [], 0.
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        values.append(running)
    return list(reversed(values))


def train_reinforce(initial, scenarios, validation=(), *, env_factory=None, observation='O2',
                    condition='ORACLE', seed=101, lr=.0003, optimizer_steps=30,
                    batch_size=20, gamma=1., eval_every=10, validation_every=None):
    if validation_every is not None:
        eval_every = validation_every
    if not scenarios or optimizer_steps < 1 or batch_size < 1 or eval_every < 1 or lr <= 0 or not 0 <= gamma <= 1:
        raise ValueError('Nonempty TRAIN users, positive budget and gamma in [0,1] required')
    seed_everything(seed)
    started = monotonic()
    factory = env_factory or _factory(observation, condition)
    policy = initial.clone()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = random.Random(seed)
    before = evaluate_policy(policy, validation, env_factory=factory)
    curve = [{'optimizer_steps': 0, 'episodes': 0, 'training_mean_return': None,
              'validation': before}]
    selected, selected_step, best = policy.clone(), 0, _score(before)
    logs, gradients, block_returns = [], [], []
    total_steps = 0
    baseline_sum, baseline_count = 0., 0
    all_returns, action_counts = [], Counter()
    for update in range(1, optimizer_steps + 1):
        baseline = baseline_sum / baseline_count if baseline_count else 0.
        losses, batch_targets = [], []
        for _ in range(batch_size):
            scenario = rng.choice(scenarios)
            env = factory(scenario)
            try:
                obs = _reset(env)
                log_probs, rewards, chosen = [], [], []
                for _ in range(8):
                    action, log_prob = policy.sample_action(obs)
                    log_probs.append(log_prob)
                    chosen.append(deepcopy(action))
                    action_counts[action['conversation']] += 1
                    obs, reward, done, truncated, _ = env.step(action)
                    rewards.append(float(reward))
                    total_steps += 1
                    if done or truncated:
                        break
                else:
                    raise RuntimeError('REINFORCE trajectory exceeded eight steps')
                targets = discounted_returns(rewards, gamma)
                # Sum across time, average across independent user trajectories.
                losses.append(-torch.stack([log_prob * (target - baseline)
                                            for log_prob, target in zip(log_probs, targets)]).sum())
                batch_targets.extend(targets)
                total = sum(rewards)
                all_returns.append(total)
                block_returns.append(total)
                diagnostic = env.diagnostic_metrics()
                logs.append({'episode': len(logs) + 1, 'optimizer_step': update,
                             'return': total, 'baseline_before_batch': baseline, 'actions': chosen,
                             'decision_error_rate': diagnostic.get('error_rate'),
                             'necessary_asks': diagnostic.get('necessary_asks'),
                             'unnecessary_asks': diagnostic.get('unnecessary_asks'),
                             'incorrect_updates': diagnostic.get('incorrect_updates'),
                             'future_accuracy': diagnostic.get('future_accuracy')})
            finally:
                env.close()
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('REINFORCE produced a nonfinite loss')
        optimizer.zero_grad()
        loss.backward()
        gradient = sum(float(p.grad.detach().square().sum().item())
                       for p in policy.parameters() if p.grad is not None) ** .5
        gradients.append(gradient)
        optimizer.step()
        # Update only after gradients for the whole current batch are computed.
        baseline_sum += sum(batch_targets)
        baseline_count += len(batch_targets)
        if update % eval_every == 0 or update == optimizer_steps:
            evaluation = evaluate_policy(policy, validation, env_factory=factory)
            curve.append({'optimizer_steps': update, 'episodes': len(logs),
                          'training_mean_return': mean(block_returns),
                          'training_mean_loss': float(loss.detach().item()),
                          'gradient_norm': gradient, 'validation': evaluation,
                          'elapsed_seconds': monotonic() - started})
            block_returns = []
            if _score(evaluation) > best:
                selected, selected_step, best = policy.clone(), update, _score(evaluation)
    if not validation:
        selected, selected_step = policy.clone(), optimizer_steps
    after = evaluate_policy(selected, validation, env_factory=factory)
    metrics = {'algorithm': 'REINFORCE Monte Carlo policy gradient', 'seed': seed,
               'training_users': len(scenarios), 'completed_episodes': len(logs),
               'sampled_trajectories': len(logs), 'optimizer_steps': optimizer_steps,
               'batch_size_trajectories': batch_size, 'action_forward_passes': total_steps,
               'lr': lr, 'gamma': gamma, 'baseline': 'scalar mean of discounted returns from earlier batches only',
               'entropy_bonus': 0., 'critic': False,
               'initial_hash': initial.parameter_hash(), 'last_hash': policy.parameter_hash(),
               'initial_parameter_hash': initial.parameter_hash(),
               'selected_hash': selected.parameter_hash(),
               'last_parameter_l2_delta': policy.parameter_delta(initial),
               'selected_parameter_l2_delta': selected.parameter_delta(initial),
               'nonzero_gradient_updates': sum(g > 0 for g in gradients),
               'gradient_norm_max': max(gradients, default=0),
               'selected_step': selected_step, 'before': before, 'after': after,
               'final_checkpoint_validation': evaluate_policy(policy, validation, env_factory=factory),
               'mean_training_return': mean(all_returns), 'cumulative_training_return': sum(all_returns),
               'actions': dict(action_counts), 'elapsed_seconds': monotonic() - started,
               'test_used_for_selection': False, 'uses_private_labels_as_features': False,
               'uses_rewards_for_gradients': True, 'rule_fallback': False,
               'selection_basis': 'validation_mean_return; ties prefer earliest',
               'compute_comparison': 'same optimizer steps and users/batch as M2; actual action forward passes reported'}
    return {'policy': selected, 'last_policy': policy, 'initial_policy': initial.clone(),
            'metrics': metrics, 'curve': curve, 'episode_log': logs}
