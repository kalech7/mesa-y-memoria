"""Approved SPEC-RL-005 checks of the experimental boundary and real learning.

Torch-dependent checks skip only when torch is absent. The recorded pilot runs
these same checks under its isolated CPU torch interpreter without skips.
"""
from copy import deepcopy
import importlib.util
import inspect
import json
from pathlib import Path
import tempfile
import unittest


TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


def scenario_for(family, split='train'):
    from rl_small_pilot.environment_adapter import generate_splits
    scenarios = generate_splits(train_n=20, validation_n=20, test_n=20, seed=1701)
    return deepcopy(next(s for s in scenarios[split] if s['family'] == family))


def composed(conversation, memory='NOOP', polarity=None, slot=None):
    return dict(conversation=conversation, memory_operation=memory,
                polarity=polarity, slot=slot)


def recursive_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from recursive_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from recursive_keys(child)


class ExperimentalBoundaryTests(unittest.TestCase):
    def test_partition_ids_templates_and_context_combinations_are_disjoint(self):
        from rl_small_pilot.environment_adapter import generate_splits
        splits = generate_splits(train_n=40, validation_n=20, test_n=20, seed=1702)
        for key in ('user_id', 'template_id'):
            sets = {split: {s[key] for s in scenarios}
                    for split, scenarios in splits.items()}
            for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
                self.assertFalse(sets[left] & sets[right])
        contexts = {split: {tuple(s['context_combination']) for s in scenarios}
                    for split, scenarios in splits.items()}
        self.assertFalse(contexts['train'] & contexts['test'])
        self.assertFalse(contexts['validation'] & contexts['test'])
        for scenarios in splits.values():
            self.assertEqual(len({s['family'] for s in scenarios}), 5)
            for family in {s['family'] for s in scenarios}:
                self.assertEqual({s['initial_direction'] for s in scenarios if s['family'] == family},
                                 {'LIKE', 'DISLIKE'})

    def test_private_gold_and_uncommunicated_replies_cannot_change_observation(self):
        from rl_small_pilot.environment_adapter import PilotEnv
        from rl_small_pilot.observations import feature_vector
        original = scenario_for('RELEVANT_RESOLVABLE')
        altered = deepcopy(original)
        for session in altered['sessions']:
            session['gold'] = dict(polarity='FORBID', needs_clarification=False,
                                   relevant_memory_ids=[], temporal=False, conflict=False)
        altered['sessions'][1]['responses'] = {
            'preference': dict(text='SECRET_FUTURE_RESPONSE', status='UNRESOLVED', fact=None)}
        for representation in ('O1', 'O2', 'O3'):
            with self.subTest(representation=representation):
                a = PilotEnv(original, representation=representation, condition='ORACLE')
                b = PilotEnv(altered, representation=representation, condition='ORACLE')
                try:
                    left, right = a.reset(), b.reset()
                    self.assertEqual(left, right)
                    self.assertEqual(feature_vector(left), feature_vector(right))
                    forbidden = {'gold', 'required_decision', 'expected_polarity',
                                 'responses', 'reward', 'family', 'template_id', 'user_id'}
                    self.assertFalse(forbidden.intersection(recursive_keys(left)))
                    self.assertNotIn('SECRET_FUTURE_RESPONSE', json.dumps(left))
                finally:
                    a.close()
                    b.close()

    def test_policy_vector_ignores_identity_and_absolute_date_artifacts(self):
        from datetime import datetime, timedelta
        from rl_small_pilot.environment_adapter import PilotEnv
        from rl_small_pilot.observations import feature_vector
        env = PilotEnv(scenario_for('EXPLICIT_CHANGE'), condition='ORACLE')
        try:
            original = env.reset()
            altered = deepcopy(original)
            visited = set()
            def change(value):
                if isinstance(value, (dict, list)):
                    if id(value) in visited:
                        return
                    visited.add(id(value))
                if isinstance(value, dict):
                    for key, item in list(value.items()):
                        if key in ('memory_id', 'evidence_id'):
                            value[key] = 'opaque_' + str(item)
                        elif key in ('at', 'query_at', 'current_at', 'observed_at',
                                     'recorded_at', 'valid_from', 'valid_to') and item:
                            date = datetime.fromisoformat(item.replace('Z', '+00:00'))
                            value[key] = (date + timedelta(days=364)).isoformat()
                        elif key == 'supersedes':
                            value[key] = ['opaque_' + str(x) for x in item]
                        else:
                            change(item)
                elif isinstance(value, list):
                    for item in value:
                        change(item)
            change(altered)
            self.assertEqual(feature_vector(original), feature_vector(altered))
        finally:
            env.close()

    def test_sqlite_retains_history_and_reload_between_sessions(self):
        from rl_small_pilot.environment_adapter import PilotEnv
        from rl_observability.policies import HistoricalClarificationRules
        from rl_longitudinal.temporal_memory import TemporalMemoryStore
        scenario = scenario_for('EXPLICIT_CHANGE')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'memory.sqlite'
            env = PilotEnv(scenario, condition='ORACLE', db_path=path)
            try:
                obs = env.reset()
                initial_ids = {f.memory_id for f in env.store.facts()}
                first_connection = env.store.conn
                obs, _, done, _, _ = env.step(HistoricalClarificationRules().select_action(obs))
                self.assertFalse(done)
                self.assertEqual(obs['session_index'], 2)
                self.assertIsNot(env.store.conn, first_connection)
                self.assertTrue(initial_ids.issubset({f.memory_id for f in env.store.facts()}))
                self.assertTrue(any(op['operation'] == 'UPDATE_WITH_HISTORY'
                                    for op in env.store.operations()))
            finally:
                env.close()
            self.assertEqual(path.read_bytes()[:16], b'SQLite format 3\x00')
            store = TemporalMemoryStore(str(path), scenario['user_id'])
            try:
                self.assertGreater(len(store.facts()), len(initial_ids))
            finally:
                store.close()
            retry = PilotEnv(scenario, condition='ORACLE', db_path=path)
            try:
                with self.assertRaises(FileExistsError):
                    retry.reset()
            finally:
                retry.close()

    def test_correct_and_unsupported_rewards_have_exact_composition(self):
        from rl_small_pilot.environment_adapter import PilotEnv
        scenario = scenario_for('EXPLICIT_CHANGE')
        expected = scenario['sessions'][1]['incoming'][0]['polarity']
        for polarity, reward, correct in ((expected, 1., True),
                ('DISLIKE' if expected == 'LIKE' else 'LIKE', -1.6, False)):
            with self.subTest(polarity=polarity):
                env = PilotEnv(scenario, condition='ORACLE')
                try:
                    env.reset()
                    _, actual, _, _, info = env.step(composed('ANSWER', 'UPDATE_WITH_HISTORY', polarity))
                    self.assertAlmostEqual(actual, reward)
                    self.assertEqual(info['correct'], correct)
                    trace = env.diagnostic_metrics()['steps'][-1]
                    self.assertEqual(trace['proposed_action'], trace['executed_action'])
                finally:
                    env.close()

    def test_unjustified_update_penalty_and_future_consequence(self):
        from rl_small_pilot.environment_adapter import PilotEnv
        scenario = scenario_for('TEMPORARY_EXCEPTION')
        polarity = scenario['sessions'][1]['incoming'][0]['polarity']
        env = PilotEnv(scenario, condition='ORACLE')
        try:
            env.reset()
            obs, reward, _, _, _ = env.step(composed('ANSWER', 'UPDATE_WITH_HISTORY', polarity))
            self.assertAlmostEqual(reward, .5)  # correct 1, unjustified update -.5
            self.assertEqual(env.diagnostic_metrics()['incorrect_updates'], 1)
            _, _, done, _, info = env.step(composed('ANSWER', polarity=polarity))
            self.assertTrue(done)
            self.assertFalse(info['correct'])
        finally:
            env.close()

    def test_clarification_cost_preservation_and_resolution_are_different(self):
        from rl_small_pilot.environment_adapter import PilotEnv
        from rl_observability.policies import HistoricalClarificationRules
        scenario = scenario_for('RELEVANT_RESOLVABLE')
        returns = []
        for asking in (False, True):
            env = PilotEnv(scenario, condition='ORACLE')
            try:
                obs = env.reset()
                if asking:
                    obs, reward, done, _, _ = env.step(composed('CLARIFY', slot='preference'))
                    self.assertAlmostEqual(reward, -.1)
                    self.assertFalse(done)
                    self.assertTrue(any(f.get('source') == 'CLARIFICATION'
                                        for f in obs['incoming_evidence']))
                else:
                    obs, reward, done, _, _ = env.step(composed('PRESERVE_CONFLICT', 'ADD_MEMORY'))
                    self.assertAlmostEqual(reward, .7)
                while not done:
                    obs, _, done, _, _ = env.step(HistoricalClarificationRules().select_action(obs))
                returns.append(env.diagnostic_metrics()['return'])
            finally:
                env.close()
        self.assertGreater(returns[1], returns[0])

    def test_security_logs_proposal_and_execution_without_gold_repair(self):
        from rl_small_pilot.environment_adapter import PilotEnv
        scenario = scenario_for('EXPLICIT_CHANGE')
        scenario['sessions'][1]['target_food'] = 'maní'
        env = PilotEnv(scenario, condition='ORACLE')
        try:
            env.reset()
            proposed = composed('ANSWER', polarity='LIKE')
            _, reward, _, _, _ = env.step(proposed)
            trace = env.diagnostic_metrics()['steps'][-1]
            self.assertEqual(trace['proposed_action'], proposed)
            self.assertNotEqual(trace['executed_action'], proposed)
            self.assertIn('EXPLICIT_RESTRICTION', trace['shield_reasons'])
            self.assertLessEqual(reward, -2.)
        finally:
            env.close()


@unittest.skipUnless(TORCH_AVAILABLE, 'torch is absent; run under the recorded isolated pilot interpreter')
class RealNeuralLearningTests(unittest.TestCase):
    def setUp(self):
        import torch

        # SmallPolicy configures deterministic CPU training for its run.
        self.addCleanup(torch.set_deterministic_debug_mode, torch.get_deterministic_debug_mode())
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())

    def test_discounted_returns_include_later_session_rewards(self):
        from rl_small_pilot.rl_trainer import discounted_returns
        self.assertEqual(discounted_returns([1., -1., 2.], gamma=.5), [1., 0., 2.])
        self.assertEqual(discounted_returns([-.1, 1., 1.], gamma=1.), [1.9, 2., 1.])

    def test_reinforce_changes_weights_and_keeps_initial_checkpoint_immutable(self):
        from rl_small_pilot.environment_adapter import generate_splits
        from rl_small_pilot.small_policy import SmallPolicy
        from rl_small_pilot.rl_trainer import train_reinforce
        splits = generate_splits(train_n=5, validation_n=5, test_n=5, seed=1901)
        initial = SmallPolicy(seed=77)
        before = initial.parameter_hash()
        result = train_reinforce(initial, splits['train'], splits['validation'],
            seed=77, optimizer_steps=2, batch_size=3, validation_every=1)
        self.assertEqual(initial.parameter_hash(), before)
        self.assertNotEqual(result['last_policy'].parameter_hash(), before)
        self.assertGreater(result['last_policy'].parameter_delta(initial), 0.)
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                            for p in result['last_policy'].parameters()))
        self.assertNotIn('test', inspect.signature(train_reinforce).parameters)

    def test_supervised_control_and_rl_start_from_identical_m0_weights(self):
        from rl_small_pilot.environment_adapter import generate_splits
        from rl_small_pilot.small_policy import SmallPolicy
        from rl_small_pilot.rl_trainer import train_reinforce
        from rl_small_pilot.supervised_baseline import train_supervised, train_additional_supervised
        splits = generate_splits(train_n=5, validation_n=5, test_n=5, seed=1902)
        trained = train_supervised(splits['train'], splits['validation'],
                                  seed=78, optimizer_steps=2)
        initial = trained['policy']
        before = initial.parameter_hash()
        rl = train_reinforce(initial, splits['train'], splits['validation'],
                             seed=78, optimizer_steps=1, batch_size=2, validation_every=1)
        sl = train_additional_supervised(initial, trained['demonstrations'], splits['validation'],
                             seed=78, optimizer_steps=1, batch_size=2, validation_every=1)
        self.assertEqual(initial.parameter_hash(), before)
        self.assertEqual(rl['metrics']['initial_parameter_hash'], before)
        self.assertEqual(sl['metrics']['initial_parameter_hash'], before)
        self.assertNotIn('test', inspect.signature(train_additional_supervised).parameters)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'M0.pt'
            initial.save(path)
            restored = SmallPolicy.load(path)
            self.assertEqual(restored.parameter_hash(), before)

    def test_selection_uses_validation_and_excludes_o1_as_primary(self):
        from rl_small_pilot.evaluate import select_observation
        validation_results = {'O1': {'mean_return': 9.},
                              'O2': {'mean_return': 1.},
                              'O3': {'mean_return': 1.}}
        self.assertEqual(select_observation(validation_results), 'O2')
        validation_results['O3']['mean_return'] = 1.1
        self.assertEqual(select_observation(validation_results), 'O3')
        self.assertNotIn('test', inspect.signature(select_observation).parameters)


if __name__ == '__main__':
    unittest.main()
