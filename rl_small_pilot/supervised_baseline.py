"""Train-only demonstrations and supervised controls sharing the RL network."""
from collections import Counter, defaultdict
from copy import deepcopy
import random
from statistics import mean
from time import monotonic

import torch
from torch.nn import functional as F

from rl_observability.policies import HistoricalClarificationRules
from .observations import action_index
from .small_policy import SmallPolicy, seed_everything


def _factory(observation, condition):
    from .environment_adapter import PilotEnv
    return lambda scenario: PilotEnv(scenario, representation=observation, condition=condition)


def _reset(env):
    value = env.reset()
    return value[0] if isinstance(value, tuple) else value


def evaluate_policy(policy, scenarios, *, env_factory=None, observation='O2', condition='ORACLE'):
    """Validation only. Caller controls the split; trainers have no test argument."""
    factory = env_factory or _factory(observation, condition)
    rows, actions = [], Counter()
    for scenario in scenarios:
        env = factory(scenario)
        try:
            obs = _reset(env)
            for _ in range(8):
                action = policy.select_action(obs)
                actions[action['conversation']] += 1
                obs, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    break
            else:
                raise RuntimeError('Validation trajectory exceeded eight steps')
            rows.append(env.diagnostic_metrics())
        finally:
            env.close()
    if not rows:
        return {'mean_return': None, 'users': 0, 'actions': {}}
    total = lambda key: sum(row.get(key, 0) for row in rows)
    decisions = sum(len(row['sessions']) for row in rows)
    errors = sum(sum(not session['correct'] for session in row['sessions']) for row in rows)
    return {'users': len(rows), 'mean_return': mean(row['return'] for row in rows),
            'total_return': total('return'), 'decision_errors': errors, 'decisions': decisions,
            'decision_error_rate': errors / max(1, decisions),
            'accuracy': 1 - errors / max(1, decisions),
            'future_accuracy': mean(row['future_accuracy'] for row in rows),
            'necessary_asks': total('necessary_asks'), 'unnecessary_asks': total('unnecessary_asks'),
            'questions': total('questions'), 'incorrect_updates': total('incorrect_updates'),
            'future_errors': total('future_errors'), 'repeated_asks': total('repeated_asks'),
            'shield_interventions': total('shield_interventions'), 'actions': dict(actions)}


def collect_demonstrations(scenarios, *, env_factory=None, observation='O2', condition='ORACLE', teacher=None):
    factory = env_factory or _factory(observation, condition)
    teacher = teacher or HistoricalClarificationRules()
    demos = []
    for i, scenario in enumerate(scenarios):
        env = factory(scenario)
        try:
            obs = _reset(env)
            for _ in range(8):
                action = teacher.select_action(obs)
                action_index(action)  # Reject any action outside the shared vocabulary.
                demos.append({'observation': deepcopy(obs), 'action': deepcopy(action),
                              'trajectory_id': scenario.get('user_id', str(i))})
                obs, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    break
            else:
                raise RuntimeError('Teacher trajectory exceeded eight steps')
        finally:
            env.close()
    return demos


def _groups(demonstrations):
    groups = defaultdict(list)
    for i, demo in enumerate(demonstrations):
        groups[demo.get('trajectory_id', str(i))].append(demo)
    return list(groups.values())


def _ce_step(policy, optimizer, groups):
    # Average per-user losses; sessions from one user are not independent units.
    losses = []
    examples = 0
    for group in groups:
        trajectory = []
        for demo in group:
            target = torch.tensor([action_index(demo['action'])], dtype=torch.long)
            trajectory.append(F.cross_entropy(policy.logits(demo['observation']).unsqueeze(0), target))
            examples += 1
        losses.append(torch.stack(trajectory).mean())
    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    gradient = sum(float(p.grad.detach().square().sum().item())
                   for p in policy.parameters() if p.grad is not None) ** .5
    optimizer.step()
    return float(loss.detach().item()), gradient, examples


def _score(evaluation):
    value = evaluation.get('mean_return')
    return value if value is not None else float('-inf')


def train_supervised(scenarios, validation=(), *, env_factory=None, observation='O2',
                     condition='ORACLE', seed=101, lr=.003, epochs=20, batch_size=20,
                     eval_every=1, teacher=None, optimizer_steps=None):
    """M0: strong teacher CE; all demonstrations originate in TRAIN users."""
    if not scenarios or epochs < 1 or batch_size < 1 or eval_every < 1 or lr <= 0 or (optimizer_steps is not None and optimizer_steps < 1):
        raise ValueError('Positive bounded training budget and nonempty TRAIN users required')
    seed_everything(seed)
    policy = SmallPolicy(seed=seed)
    initial = policy.clone()
    started = monotonic()
    demos = collect_demonstrations(scenarios, env_factory=env_factory, observation=observation,
                                   condition=condition, teacher=teacher)
    groups = _groups(demos)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = random.Random(seed)
    before = evaluate_policy(policy, validation, env_factory=env_factory, observation=observation, condition=condition)
    curve = [{'optimizer_steps': 0, 'epoch': 0, 'validation': before, 'training_mean_loss': None}]
    selected, selected_step, best = policy.clone(), 0, _score(before)
    updates = examples = 0
    gradients, loss_history = [], []
    for epoch in range(1, epochs + 1):
        order = list(groups)
        rng.shuffle(order)
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            if optimizer_steps is not None and updates >= optimizer_steps:
                break
            loss, gradient, count = _ce_step(policy, optimizer, order[start:start + batch_size])
            updates += 1
            examples += count
            epoch_losses.append(loss)
            loss_history.append(loss)
            gradients.append(gradient)
        exhausted = optimizer_steps is not None and updates >= optimizer_steps
        if epoch % eval_every == 0 or epoch == epochs or exhausted:
            evaluation = evaluate_policy(policy, validation, env_factory=env_factory, observation=observation, condition=condition)
            curve.append({'optimizer_steps': updates, 'epoch': epoch, 'validation': evaluation,
                          'training_mean_loss': mean(epoch_losses) if epoch_losses else None, 'elapsed_seconds': monotonic() - started})
            if _score(evaluation) > best:
                selected, selected_step, best = policy.clone(), updates, _score(evaluation)
        if exhausted:
            break
    if not validation:
        selected, selected_step = policy.clone(), updates
    after = evaluate_policy(selected, validation, env_factory=env_factory, observation=observation, condition=condition)
    metrics = {'algorithm': 'supervised cross entropy', 'seed': seed,
               'training_users': len(scenarios), 'demonstrations': len(demos),
               'optimizer_steps': updates, 'action_forward_passes': examples,
               'epochs': epochs, 'batch_size_trajectories': batch_size, 'lr': lr,
               'parameter_count': policy.parameter_count,
               'initial_hash': initial.parameter_hash(), 'last_hash': policy.parameter_hash(),
               'initial_parameter_hash': initial.parameter_hash(),
               'selected_hash': selected.parameter_hash(),
               'last_parameter_l2_delta': policy.parameter_delta(initial),
               'selected_parameter_l2_delta': selected.parameter_delta(initial),
               'gradient_norm_max': max(gradients, default=0),
               'nonzero_gradient_updates': sum(g > 0 for g in gradients),
               'selected_step': selected_step, 'before': before, 'after': after,
               'final_checkpoint_validation': evaluate_policy(policy, validation, env_factory=env_factory,
                                                              observation=observation, condition=condition),
               'elapsed_seconds': monotonic() - started, 'test_used_for_selection': False,
               'selection_basis': 'validation_mean_return; ties prefer earliest',
               'uses_private_labels_as_features': False, 'teacher': 'historical_clarification_rules',
               'uses_rewards_for_gradients': False, 'rule_fallback': False}
    return {'policy': selected, 'last_policy': policy, 'initial_policy': initial,
            'demos': demos, 'demonstrations': demos, 'metrics': metrics, 'curve': curve, 'episode_log': []}


def train_additional_supervised(initial, demonstrations, validation=(), *, env_factory=None,
                                observation='O2', condition='ORACLE', seed=101, lr=.0003,
                                optimizer_steps=30, batch_size=20, eval_every=10, gamma=1,
                                validation_every=None):
    """M2: same M0 weights, same update/user-batch budget as M1, CE gradients."""
    if validation_every is not None:
        eval_every = validation_every
    if not demonstrations or optimizer_steps < 1 or batch_size < 1 or eval_every < 1 or lr <= 0:
        raise ValueError('Nonempty TRAIN demonstrations and positive budget required')
    seed_everything(seed)
    started = monotonic()
    policy = initial.clone()
    rng = random.Random(seed)
    groups = _groups(demonstrations)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    before = evaluate_policy(policy, validation, env_factory=env_factory, observation=observation, condition=condition)
    curve = [{'optimizer_steps': 0, 'validation': before, 'training_mean_loss': None}]
    selected, selected_step, best = policy.clone(), 0, _score(before)
    gradients, recent, traces = [], [], []
    examples = 0
    for step in range(1, optimizer_steps + 1):
        batch = [rng.choice(groups) for _ in range(batch_size)]
        loss, gradient, count = _ce_step(policy, optimizer, batch)
        examples += count
        recent.append(loss)
        gradients.append(gradient)
        traces.append({'optimizer_step': step, 'loss': loss, 'gradient_norm': gradient,
                       'trajectories': batch_size, 'action_examples': count})
        if step % eval_every == 0 or step == optimizer_steps:
            evaluation = evaluate_policy(policy, validation, env_factory=env_factory, observation=observation, condition=condition)
            curve.append({'optimizer_steps': step, 'validation': evaluation,
                          'training_mean_loss': mean(recent), 'elapsed_seconds': monotonic() - started})
            recent = []
            if _score(evaluation) > best:
                selected, selected_step, best = policy.clone(), step, _score(evaluation)
    if not validation:
        selected, selected_step = policy.clone(), optimizer_steps
    after = evaluate_policy(selected, validation, env_factory=env_factory, observation=observation, condition=condition)
    metrics = {'algorithm': 'additional supervised cross entropy', 'seed': seed,
               'training_users': len(groups), 'demonstrations': len(demonstrations),
               'optimizer_steps': optimizer_steps, 'sampled_trajectories': optimizer_steps * batch_size,
               'action_forward_passes': examples, 'batch_size_trajectories': batch_size, 'lr': lr,
               'initial_hash': initial.parameter_hash(), 'last_hash': policy.parameter_hash(),
               'initial_parameter_hash': initial.parameter_hash(),
               'selected_hash': selected.parameter_hash(), 'last_parameter_l2_delta': policy.parameter_delta(initial),
               'selected_parameter_l2_delta': selected.parameter_delta(initial),
               'nonzero_gradient_updates': sum(g > 0 for g in gradients), 'gradient_norm_max': max(gradients, default=0),
               'selected_step': selected_step, 'before': before, 'after': after,
               'final_checkpoint_validation': evaluate_policy(policy, validation, env_factory=env_factory,
                                                              observation=observation, condition=condition),
               'elapsed_seconds': monotonic() - started, 'test_used_for_selection': False,
               'selection_basis': 'validation_mean_return; ties prefer earliest',
               'uses_rewards_for_gradients': False, 'rule_fallback': False,
               'compute_comparison': 'same optimizer steps and users/batch; actual action forward passes reported'}
    return {'policy': selected, 'last_policy': policy, 'initial_policy': initial.clone(),
            'metrics': metrics, 'curve': curve, 'episode_log': traces}
