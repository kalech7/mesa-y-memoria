"""Small bounded Q-learning pilot; training requires prior validation-only gates.

No test-set parameter exists. The policy uses exactly the SFT observation
abstraction, action space, and fallback. It learns neither the extractor nor the
symbolic temporal reconciler or safety shield. Checkpoints are selected only by
validation normalized episodic return.
"""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
from time import monotonic

from .baselines import (ClarificationRuleBased, action_catalog, feature_key, _real_action)


def validate_training_gates(report):
    """Fail closed, before creating an environment or output directory."""
    if isinstance(report, (str, Path)):
        report = json.loads(Path(report).read_text())
    if not isinstance(report, dict):
        raise ValueError('A recorded longitudinal validation gate report is required')
    if not all(report.get(key) is True for key in ('gate_a', 'gate_b', 'gate_c')):
        raise ValueError('Historical, clarification, and longitudinal gates must all pass')
    gate = report.get('gate_d', {})
    if not isinstance(gate, dict) or gate.get('status') != 'PASS' or gate.get('validation_only') is not True:
        raise ValueError('Gate D must pass using validation only; test results cannot authorize training')
    minimum = gate.get('minimum_difference', .10)
    ci = gate.get('paired_ci', ())
    if minimum < .10 or gate.get('difference', float('-inf')) < minimum or len(ci) != 2 or ci[0] <= 0:
        raise ValueError('Gate D lacks the preregistered meaningful positive paired margin')
    if not gate.get('reference_method') or not gate.get('best_non_rl_method'):
        raise ValueError('Gate D must identify the reference and strongest conventional comparator')
    return report


class QLearningLongitudinal:
    name = 'RL_LONGITUDINAL'
    ablations = ()

    def __init__(self, q_values=None):
        self.q_values = q_values or {}
        self.fallback = ClarificationRuleBased()

    def select_action(self, observation):
        values = self.q_values.get(feature_key(observation))
        if not values:
            return self.fallback.select_action(observation)
        actions = action_catalog(observation)
        selected = max(actions, key=lambda action: (values.get(action, 0.0), -actions.index(action)))
        return _real_action(selected, observation)

    def to_dict(self):
        return {'policy_type': 'longitudinal_tabular_q_learning', 'feature_version': 1,
                'q_values': self.q_values, 'unseen_state_fallback': self.fallback.name}

    @classmethod
    def from_dict(cls, value):
        if value.get('policy_type') != 'longitudinal_tabular_q_learning':
            raise ValueError('Unsupported longitudinal RL checkpoint')
        return cls(value['q_values'])

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def _factory(condition):
    from .longitudinal_environment import LongitudinalEnv
    return lambda scenario: LongitudinalEnv(scenario, condition=condition)


def validation_return(policy, scenarios, *, env_factory=None, condition='ORACLE_STRUCTURED'):
    env_factory = env_factory or _factory(condition)
    values = []
    for scenario in scenarios:
        env = env_factory(scenario)
        try:
            obs, _ = env.reset()
            total = 0.0
            sessions = max(1, obs.get('remaining_sessions', len(scenario.get('sessions', []))))
            for _ in range(100):
                obs, reward, done, truncated, _ = env.step(policy.select_action(obs))
                total += reward
                if done or truncated:
                    break
            else:
                raise RuntimeError('Validation episode exceeded 100 actions')
            values.append(total / sessions)
        finally:
            if hasattr(env, 'close'):
                env.close()
    return mean(values) if values else None


def train_q_learning(scenarios, validation_scenarios=(), *, gate_report=None,
                     seed=101, episodes=None, alpha=.3, gamma=1.0,
                     epsilon_start=1.0, epsilon_end=.05, max_seconds=120.0,
                     block_size=50, env_factory=None, condition='ORACLE_STRUCTURED', output_dir=None):
    gates = validate_training_gates(gate_report)
    episodes = len(scenarios) if episodes is None else episodes
    if not 0 < episodes <= min(2000, len(scenarios)):
        raise ValueError('Training is limited to one pass and at most 2000 independent user episodes')
    if not 0 < alpha <= 1 or not 0 <= gamma <= 1 or not 0 <= epsilon_end <= epsilon_start <= 1:
        raise ValueError('Invalid fixed learning hyperparameters')
    if not 0 < max_seconds <= 300 or block_size <= 0:
        raise ValueError('Invalid modest compute budget')
    env_factory = env_factory or _factory(condition)
    rng = random.Random(seed)
    ordered = list(scenarios)
    rng.shuffle(ordered)
    q_values = defaultdict(dict)
    blocks, traces, returns = [], [], []
    start = monotonic()
    completed = 0
    for episode_index, scenario in enumerate(ordered[:episodes]):
        if monotonic() - start >= max_seconds:
            break
        env = env_factory(scenario)
        try:
            obs, _ = env.reset()
            total = 0.0
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * episode_index / max(1, episodes - 1)
            actions_taken = []
            sessions = max(1, len(scenario.get('sessions', [])))
            for step in range(100):
                key, actions = feature_key(obs), action_catalog(obs)
                if rng.random() < epsilon:
                    action = rng.choice(actions)
                else:
                    action = max(actions, key=lambda a: (q_values[key].get(a, 0.0), -actions.index(a)))
                real = _real_action(action, obs)
                next_obs, reward, done, truncated, _ = env.step(real)
                next_key = feature_key(next_obs)
                next_value = 0.0 if done or truncated else max(
                    (q_values[next_key].get(a, 0.0) for a in action_catalog(next_obs)), default=0.0)
                previous = q_values[key].get(action, 0.0)
                q_values[key][action] = previous + alpha * (reward + gamma * next_value - previous)
                total += reward
                actions_taken.append(real)
                obs = next_obs
                if done or truncated:
                    break
            else:
                raise RuntimeError('Training episode exceeded the fixed 100 action safety limit')
            completed += 1
            returns.append(total / sessions)
            traces.append({'training_episode_index': episode_index, 'return': total,
                           'normalized_return': total / sessions, 'actions': actions_taken, 'epsilon': epsilon})
            if completed % block_size == 0 or completed == episodes:
                blocks.append({'episodes': completed, 'mean_normalized_training_return': mean(returns),
                               'epsilon': epsilon, 'unique_observation_keys': len(q_values),
                               'elapsed_seconds': monotonic() - start})
                returns = []
        finally:
            if hasattr(env, 'close'):
                env.close()
    policy = QLearningLongitudinal(dict(q_values))
    metrics = {'executed': True, 'completed_episodes': completed, 'requested_episodes': episodes,
               'elapsed_seconds': monotonic() - start, 'time_limit_reached': completed < episodes,
               'validation_normalized_return': validation_return(policy, validation_scenarios,
                                                                env_factory=env_factory, condition=condition),
               'hyperparameters': {'algorithm': 'tabular Q-learning', 'seed': seed, 'alpha': alpha,
                                   'gamma': gamma, 'epsilon_start': epsilon_start, 'epsilon_end': epsilon_end,
                                   'max_seconds': max_seconds, 'block_size': block_size, 'condition': condition},
               'blocks': blocks, 'gates_authorizing_training': gates,
               'private_labels_used_for_actions_or_features': False,
               'test_used_for_selection': False}
    result = {'policy': policy, 'training_metrics': metrics, 'episode_log': traces}
    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=False)
        checkpoint = path / 'checkpoint.json'
        policy.save(checkpoint)
        (path / 'training_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        (path / 'episode_log.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in traces))
        result['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return result


def train_three_seeds(scenarios, validation_scenarios, *, gate_report, output_dir,
                      seeds=(101, 202, 303), **kwargs):
    validate_training_gates(gate_report)
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError('At least three independent preregistered training seeds are required')
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=False)
    results = [train_q_learning(scenarios, validation_scenarios, gate_report=gate_report, seed=seed,
                                output_dir=path / str(seed), **kwargs) for seed in seeds]
    best = max(results, key=lambda row: row['training_metrics']['validation_normalized_return'])
    serializable = [{key: value for key, value in result.items() if key != 'policy'} for result in results]
    return {'results': serializable, 'selected_policy': best['policy'],
            'selection_basis': 'validation_normalized_return_only'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--gate-report', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--max-seconds', type=float, default=120.0)
    args = parser.parse_args()
    try:
        gates = validate_training_gates(args.gate_report)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    train = json.loads(Path(args.train).read_text())
    validation = json.loads(Path(args.validation).read_text())
    train_three_seeds(train, validation, gate_report=gates, output_dir=args.output,
                      episodes=args.episodes, max_seconds=args.max_seconds)


if __name__ == '__main__':
    main()
