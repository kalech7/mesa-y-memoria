"""Bounded SPEC-RL-004 pilot, retaining the project's tabular Q-learning.

This additive pilot is explicitly authorized by SPEC-RL-004 H3, independently
of superiority H4. It neither modifies nor bypasses the historical pipeline's
gated train_rl.py. Validation alone selects checkpoints; no test argument exists.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
import random
from statistics import mean
from time import monotonic

from .policies import (HistoricalClarificationRules, QLearningPolicy,
                       SupervisedPolicy, action_catalog, feature_key)


def _factory(observation):
    from .environment import MinimalEnv
    return lambda scenario: MinimalEnv(scenario, representation=observation)


def _reset(env):
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def _close(env):
    if hasattr(env, 'close'):
        env.close()


def evaluate_policy(policy, scenarios, *, observation='O3', env_factory=None):
    """Small validation helper; detailed final analysis belongs to the runner."""
    env_factory = env_factory or _factory(observation)
    returns, errors, decisions, actions = [], 0, 0, Counter()
    for scenario in scenarios:
        env = env_factory(scenario)
        try:
            obs = _reset(env)
            total = 0.0
            for _ in range(8):
                action = policy.select_action(obs)
                actions[action['conversation']] += 1
                obs, reward, done, truncated, info = env.step(action)
                total += reward
                correct = info.get('decision_correct', info.get('correct'))
                if correct is not None:
                    decisions += 1
                    errors += int(not correct)
                if done or truncated:
                    break
            else:
                raise RuntimeError('Minimal environment exceeded eight steps')
            returns.append(total)
        finally:
            _close(env)
    return {'mean_return': mean(returns) if returns else None,
            'mean_normalized_return': mean(returns) / 2 if returns else None,
            'total_return': sum(returns), 'users': len(returns),
            'decision_error_rate': errors / decisions if decisions else None,
            'decision_errors': errors, 'decisions': decisions, 'actions': dict(actions)}


def train_supervised(scenarios, validation=(), *, observation='O3', seed=101,
                     env_factory=None, teacher=None):
    """Clone a strong public-rule teacher, with no privileged demonstrations.

    One teacher trajectory per independent training user. Dataset size and all
    labels are reported; the learner receives identical state/action features to
    Q-learning. Unseen states use only a training empirical action prior.
    """
    env_factory = env_factory or _factory(observation)
    teacher = teacher or HistoricalClarificationRules()
    counts, prior = defaultdict(Counter), Counter()
    examples = 0
    start = monotonic()
    before = evaluate_policy(SupervisedPolicy(), validation, observation=observation, env_factory=env_factory)
    ordered = list(scenarios)
    random.Random(seed).shuffle(ordered)
    for scenario in ordered:
        env = env_factory(scenario)
        try:
            obs = _reset(env)
            for _ in range(8):
                action = teacher.select_action(obs)
                encoded = json.dumps(action, sort_keys=True, separators=(',', ':'))
                if encoded not in action_catalog(obs):
                    raise ValueError('Teacher action is outside the shared action catalog')
                counts[feature_key(obs)][encoded] += 1
                prior[encoded] += 1
                examples += 1
                obs, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    break
            else:
                raise RuntimeError('Teacher trajectory exceeded eight steps')
        finally:
            _close(env)
    policy = SupervisedPolicy({k: dict(v) for k, v in counts.items()}, dict(prior))
    after = evaluate_policy(policy, validation, observation=observation, env_factory=env_factory)
    metrics = {'algorithm': 'tabular categorical behavior cloning', 'teacher': teacher.name,
               'seed': seed, 'training_users': len(scenarios), 'demonstrations': examples,
               'unique_states': len(counts), 'table_cells': sum(len(v) for v in counts.values()),
               'elapsed_seconds': monotonic() - start, 'before': before, 'after': after,
               'uses_private_labels': False, 'uses_rewards': False, 'rule_fallback': False,
               'test_used_for_selection': False, 'observation': observation,
               'architecture': 'categorical state table; same feature_key/action_catalog as Q-learning'}
    return {'policy': policy, 'metrics': metrics,
            'curve': [{'episodes': 0, 'validation': before}, {'episodes': len(scenarios), 'validation': after}]}


def train_q_learning(scenarios, validation=(), *, observation='O3', seed=101,
                     episodes=1000, max_seconds=60.0, block_size=100, alpha=.3,
                     gamma=1.0, epsilon_start=1.0, epsilon_end=.05, env_factory=None):
    """Retain Q-learning, with total user episodes and wall time both bounded.

    Users are resampled from train only. Repeated visits are explicitly reported
    and never counted as independent evaluation users. A frozen alpha/gamma and
    exploration schedule are shared across representations and seeds. Selection
    uses validation return at preregistered block boundaries, including step 0.
    """
    if not scenarios or not 0 < episodes <= 100000 or not 0 < max_seconds <= 3600 or block_size <= 0:
        raise ValueError('Nonempty training users and bounded positive budget required')
    if not 0 < alpha <= 1 or not 0 <= gamma <= 1 or not 0 <= epsilon_end <= epsilon_start <= 1:
        raise ValueError('Invalid Q-learning hyperparameters')
    env_factory = env_factory or _factory(observation)
    rng = random.Random(seed)
    q_values = defaultdict(dict)
    start = monotonic()
    before = evaluate_policy(QLearningPolicy(), validation, observation=observation, env_factory=env_factory)
    curve = [{'episodes': 0, 'training_mean_return': None, 'validation': before}]
    selected_values = {}
    selected_episode = 0
    selected_return = before['mean_return'] if before['mean_return'] is not None else float('-inf')
    traces, recent_returns, all_returns = [], [], []
    for episode_index in range(episodes):
        if monotonic() - start >= max_seconds:
            break
        scenario = scenarios[rng.randrange(len(scenarios))]
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * episode_index / max(1, episodes - 1)
        env = env_factory(scenario)
        try:
            obs = _reset(env)
            total, chosen = 0.0, []
            for _ in range(8):
                key, actions = feature_key(obs), action_catalog(obs)
                action = rng.choice(actions) if rng.random() < epsilon else max(
                    actions, key=lambda a: (q_values[key].get(a, 0.0), -actions.index(a)))
                next_obs, reward, done, truncated, info = env.step(json.loads(action))
                bootstrap = 0.0 if done or truncated else max(
                    (q_values[feature_key(next_obs)].get(a, 0.0) for a in action_catalog(next_obs)), default=0.0)
                previous = q_values[key].get(action, 0.0)
                q_values[key][action] = previous + alpha * (reward + gamma * bootstrap - previous)
                total += reward
                chosen.append(json.loads(action))
                obs = next_obs
                if done or truncated:
                    break
            else:
                raise RuntimeError('Q-learning trajectory exceeded eight steps')
            traces.append({'episode': episode_index + 1, 'return': total,
                           'epsilon': epsilon, 'actions': chosen})
            recent_returns.append(total)
            all_returns.append(total)
            if (episode_index + 1) % block_size == 0 or episode_index + 1 == episodes:
                evaluation = evaluate_policy(QLearningPolicy(dict(q_values)), validation,
                                             observation=observation, env_factory=env_factory)
                curve.append({'episodes': episode_index + 1, 'training_mean_return': mean(recent_returns),
                              'validation': evaluation, 'unique_states': len(q_values),
                              'elapsed_seconds': monotonic() - start})
                score = evaluation['mean_return']
                if score is not None and score > selected_return:
                    selected_values = deepcopy(dict(q_values))
                    selected_episode, selected_return = episode_index + 1, score
                recent_returns = []
        finally:
            _close(env)
    # If no validation users were supplied, return the final pilot transparently;
    # the experiment runner always supplies validation for actual selection.
    if not validation:
        selected_values, selected_episode = dict(q_values), len(traces)
    policy = QLearningPolicy(selected_values)
    after = evaluate_policy(policy, validation, observation=observation, env_factory=env_factory)
    final_validation = evaluate_policy(QLearningPolicy(dict(q_values)), validation,
                                       observation=observation, env_factory=env_factory)
    metrics = {'algorithm': 'tabular Q-learning', 'seed': seed, 'training_users': len(scenarios),
               'requested_episodes': episodes, 'completed_episodes': len(traces),
               'unique_states': len(selected_values), 'table_cells': sum(len(v) for v in selected_values.values()),
               'mean_training_return': mean(all_returns) if all_returns else None,
               'cumulative_training_return': sum(all_returns), 'before': before, 'after': after,
               'final_checkpoint_validation': final_validation,
               'selected_episode': selected_episode, 'selection_basis': 'validation_mean_return_only',
               'elapsed_seconds': monotonic() - start, 'time_limit_reached': len(traces) < episodes,
               'observation': observation, 'test_used_for_selection': False,
               'uses_private_labels_as_features': False, 'rule_fallback': False,
               'hyperparameters': {'alpha': alpha, 'gamma': gamma, 'epsilon_start': epsilon_start,
                                   'epsilon_end': epsilon_end, 'block_size': block_size,
                                   'episodes': episodes, 'max_seconds': max_seconds},
               'algorithm_change': False, 'pilot_superiority_gate_required': False,
               'architecture': 'categorical state table; same feature_key/action_catalog as supervision'}
    return {'policy': policy, 'metrics': metrics, 'curve': curve, 'episode_log': traces}


def train_three_seeds(scenarios, validation, *, seeds=(101, 202, 303), **kwargs):
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError('At least three distinct training seeds are required')
    return {seed: train_q_learning(scenarios, validation, seed=seed, **kwargs) for seed in seeds}
