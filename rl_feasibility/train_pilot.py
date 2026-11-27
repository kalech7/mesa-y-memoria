"""Bounded standard-library tabular Q-learning pilot, run only after prior gates.

The training runner deliberately has no test-set parameter. Checkpoint selection
uses validation return, with no access to private diagnostic labels in the policy.
"""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
from time import monotonic

from rl_feasibility.baselines import action_catalog, feature_key, _real_action, CurrentHeuristic


class QLearningPolicy:
    name = 'Q_LEARNING'
    def __init__(self, q_values=None, seed=0):
        self.q_values = q_values or {}
        self.rng = random.Random(seed)
        self.fallback = CurrentHeuristic()

    def select_action(self, observation):
        key = feature_key(observation)
        values = self.q_values.get(key)
        if not values:
            return self.fallback.select_action(observation)
        actions = action_catalog(observation)
        # A fixed tie break makes the frozen policy deterministic and reproducible.
        action = max(actions, key=lambda action: (values.get(action, 0.0), -actions.index(action)))
        return _real_action(action, observation)

    def to_dict(self):
        return {'policy_type':'tabular_q_learning', 'feature_version':1, 'q_values':self.q_values,
                'unseen_state_fallback':'CURRENT_HEURISTIC'}

    @classmethod
    def from_dict(cls, value):
        if value.get('policy_type') != 'tabular_q_learning':
            raise ValueError('Unsupported checkpoint type')
        return cls(value['q_values'])

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def validation_return(policy, scenarios, condition='ORACLE_STRUCTURED', reward_config=None):
    from rl_feasibility.environment import FoodPreferenceEnv
    returns = []
    for scenario in scenarios:
        env = FoodPreferenceEnv(scenario, condition=condition, reward_config=reward_config)
        obs, _ = env.reset()
        total = 0.0
        for _ in range(4):
            obs, reward, terminated, truncated, _ = env.step(policy.select_action(obs))
            total += reward
            if terminated or truncated:
                break
        returns.append(total)
    return mean(returns) if returns else None


def train_q_learning(scenarios, validation_scenarios=(), *, seed=101, episodes=2000,
                     alpha=0.3, gamma=1.0, epsilon_start=1.0, epsilon_end=0.05,
                     condition='ORACLE_STRUCTURED', reward_config=None,
                     output_dir=None, block_size=100, max_seconds=120.0):
    """One fixed-budget shuffled pass by default, no adaptive budget increases.

    `episodes` may not exceed available training episodes: use new independently
    generated training scenarios to enlarge a preregistered run. This prevents
    hidden replay expansions after seeing a disappointing result.
    """
    from rl_feasibility.environment import FoodPreferenceEnv
    if episodes <= 0 or episodes > len(scenarios):
        raise ValueError('episodes must be positive and no greater than training scenario count')
    if not 0 < alpha <= 1 or not 0 <= gamma <= 1:
        raise ValueError('Invalid learning parameters')
    if not 0 <= epsilon_end <= epsilon_start <= 1:
        raise ValueError('Invalid exploration schedule')
    rng = random.Random(seed)
    ordered = list(scenarios)
    rng.shuffle(ordered)
    q_values = defaultdict(dict)
    history = []
    traces = []
    block_returns = []
    started = monotonic()
    completed = 0
    for episode_index, scenario in enumerate(ordered[:episodes]):
        if monotonic() - started >= max_seconds:
            break
        env = FoodPreferenceEnv(scenario, condition=condition, reward_config=reward_config)
        obs, _ = env.reset()
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * episode_index / max(1, episodes-1)
        total = 0.0
        actions_taken = []
        for _ in range(4):
            key = feature_key(obs)
            actions = action_catalog(obs)
            if rng.random() < epsilon:
                action = rng.choice(actions)
            else:
                action = max(actions, key=lambda a:(q_values[key].get(a, 0.0), -actions.index(a)))
            real_action = _real_action(action, obs)
            next_obs, reward, terminated, truncated, _ = env.step(real_action)
            next_key = feature_key(next_obs)
            bootstrap = 0.0 if terminated or truncated else max(
                (q_values[next_key].get(a, 0.0) for a in action_catalog(next_obs)), default=0.0)
            old = q_values[key].get(action, 0.0)
            q_values[key][action] = old + alpha * (reward + gamma * bootstrap - old)
            total += reward
            actions_taken.append(real_action)
            obs = next_obs
            if terminated or truncated:
                break
        completed += 1
        block_returns.append(total)
        traces.append({'training_episode_index':episode_index, 'return':total,
                       'actions':actions_taken, 'epsilon':epsilon})
        if completed % block_size == 0 or completed == episodes:
            history.append({'episodes':completed, 'mean_training_return':mean(block_returns),
                            'epsilon':epsilon, 'unique_observation_keys':len(q_values),
                            'elapsed_seconds':monotonic()-started})
            block_returns = []
    policy = QLearningPolicy(dict(q_values), seed)
    valid = validation_return(policy, validation_scenarios, condition, reward_config)
    hyperparameters = {'seed':seed, 'requested_episodes':episodes, 'alpha':alpha, 'gamma':gamma,
                      'epsilon_start':epsilon_start, 'epsilon_end':epsilon_end, 'block_size':block_size,
                      'max_seconds':max_seconds, 'condition':condition, 'algorithm':'tabular Q-learning'}
    result = {'policy':policy, 'training_metrics':{'executed':True, 'completed_episodes':completed,
              'elapsed_seconds':monotonic()-started, 'time_limit_reached':completed < episodes,
              'validation_return':valid, 'hyperparameters':hyperparameters, 'blocks':history},
              'episode_log':traces}
    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=False)
        checkpoint = path/'checkpoint.json'
        policy.save(checkpoint)
        (path/'training_metrics.json').write_text(json.dumps(result['training_metrics'], indent=2))
        (path/'episode_log.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in traces))
        result['checkpoint'] = str(checkpoint)
        result['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--max-seconds', type=float, default=120.0)
    parser.add_argument('--condition', default='ORACLE_STRUCTURED', choices=['ORACLE_STRUCTURED','PIPELINE'])
    parser.add_argument('--gate-report', required=True, help='Directory with validated environment_validation.json and headroom_analysis.json')
    args = parser.parse_args()
    gate_report = Path(args.gate_report)
    try:
        gate_one = json.loads((gate_report / 'environment_validation.json').read_text())['gate_1']
        gate_two = json.loads((gate_report / 'headroom_analysis.json').read_text())['validation']['gate_2']
    except (OSError, KeyError, json.JSONDecodeError):
        parser.error('Training requires readable prior Gate 1 and Gate 2 artifacts')
    if gate_one is not True or gate_two != 'PASS':
        parser.error('Training not justified: Gate 1 and Gate 2 have not both passed')
    def read(path):
        data = json.loads(Path(path).read_text())
        return data['scenarios'] if isinstance(data, dict) else data
    result = train_q_learning(read(args.train), read(args.validation), seed=args.seed,
                              episodes=args.episodes, alpha=args.alpha, max_seconds=args.max_seconds,
                              condition=args.condition, output_dir=args.output)
    print(json.dumps(result['training_metrics'], indent=2))


if __name__ == '__main__':
    main()
