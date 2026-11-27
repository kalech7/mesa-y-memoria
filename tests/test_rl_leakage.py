"""Boundary tests: policy observations must not encode evaluator-only labels."""
import copy
import json
import unittest

from rl_feasibility.environment import FoodPreferenceEnv
from rl_feasibility.scenario_generator import generate_scenarios


class RLLeakageTests(unittest.TestCase):
    def setUp(self):
        self.scenario = generate_scenarios(1, 971, split='train')[0]

    def assert_public(self, value):
        forbidden = {'scenario_id', 'seed', 'category', 'split', 'latent',
                     'acceptable_foods', 'response_map', 'unresolvable', 'gold',
                     'expected', 'optimal_action', 'reward_label', 'true_state'}
        if isinstance(value, dict):
            self.assertFalse(forbidden.intersection(value))
            if 'user_id' in value:
                self.assertEqual(value['user_id'], 'user')
            for child in value.values():
                self.assert_public(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                self.assert_public(child)

    def test_reset_observation_and_info_exclude_private_fields(self):
        env = FoodPreferenceEnv(self.scenario)
        observation, info = env.reset(seed=999)
        self.assert_public(observation)
        self.assert_public(info)
        self.assertEqual(info, {})

    def test_metadata_and_seed_changes_leave_observation_identical(self):
        changed = copy.deepcopy(self.scenario)
        changed.update(scenario_id='other-id', user_id='other-user', category='SCC', split='test')
        first = FoodPreferenceEnv(self.scenario).reset(seed=1)[0]
        second = FoodPreferenceEnv(changed).reset(seed=987654)[0]
        self.assertEqual(first, second)

    def test_private_alternative_states_are_initially_indistinguishable(self):
        changed = copy.deepcopy(self.scenario)
        changed['latent']['context']['company'] = 'friends'
        changed['latent']['acceptable_foods'] = ['pizza']
        changed['response_map']['company'] = {'value': 'friends', 'text': 'Comeré con amigos.'}
        first_env, second_env = FoodPreferenceEnv(self.scenario), FoodPreferenceEnv(changed)
        first, _ = first_env.reset()
        second, _ = second_env.reset()
        self.assertEqual(first, second)
        first_after = first_env.step('CLARIFY_COMPANY')[0]
        second_after = second_env.step('CLARIFY_COMPANY')[0]
        self.assertNotEqual(first_after, second_after)
        self.assert_public(first_after)
        self.assert_public(second_after)

    def test_unasked_future_reply_not_visible(self):
        altered = copy.deepcopy(self.scenario)
        marker = 'FUTURE_REPLY_MARKER_61a9'
        altered['response_map']['place']['text'] = marker
        env = FoodPreferenceEnv(altered)
        observation, info = env.reset()
        self.assertNotIn(marker, json.dumps([observation, info]))
        observation, _, _, _, info = env.step('CLARIFY_DAY')
        self.assertNotIn(marker, json.dumps([observation, info]))
        observation, _, _, _, info = env.step('CLARIFY_PLACE')
        self.assertIn(marker, json.dumps(observation))
        self.assert_public(info)

    def test_step_diagnostics_never_return_terminal_labels(self):
        for action in ('CLARIFY_DAY', 'ANSWER:ensalada', 'PRESERVE_CONFLICT', 'ABSTAIN'):
            env = FoodPreferenceEnv(self.scenario)
            env.reset()
            observation, _, _, _, info = env.step(action)
            self.assert_public(observation)
            self.assert_public(info)
            self.assertEqual(info, {})

    def test_observation_mutation_cannot_inject_private_truth(self):
        env = FoodPreferenceEnv(self.scenario)
        observation, _ = env.reset()
        observation['context']['injected_private_truth'] = 'pizza'
        observation['preferences'].clear()
        next_observation = env.step('CLARIFY_DAY')[0]
        self.assertNotIn('injected_private_truth', next_observation['context'])
        self.assertTrue(next_observation['preferences'])


if __name__ == '__main__':
    unittest.main()
