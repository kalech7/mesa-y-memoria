import unittest

from rl_feasibility.environment import FoodPreferenceEnv, diagnostic_metrics, public_support
from rl_feasibility.reward import RewardConfig
from tests.test_rl_environment import case


class RewardTests(unittest.TestCase):
    def test_supported_answer_is_better_than_abstention(self):
        scenario = case('CLEAR')
        good = FoodPreferenceEnv(scenario)
        obs, _ = good.reset()
        food = public_support(obs)['supported_foods'][0]
        self.assertEqual(good.step('ANSWER:' + food)[1], 1)
        abstain = FoodPreferenceEnv(scenario)
        abstain.reset()
        self.assertLess(abstain.step('ABSTAIN')[1], 1)

    def test_correct_hidden_guess_still_penalized_for_lack_of_evidence(self):
        scenario = case('CPC_DAY')
        env = FoodPreferenceEnv(scenario)
        env.reset()
        reward = env.step('ANSWER:' + scenario['latent']['acceptable_foods'][0])[1]
        self.assertAlmostEqual(reward, .4)
        self.assertFalse(diagnostic_metrics(env)['final_correct'])

    def test_relevant_ask_improves_total_return(self):
        scenario = case('CPC_DAY')
        env = FoodPreferenceEnv(scenario)
        env.reset()
        obs, cost, *_ = env.step('CLARIFY_DAY')
        food = public_support(obs)['supported_foods'][0]
        terminal = env.step('ANSWER:' + food)[1]
        self.assertAlmostEqual(cost + terminal, .9)

    def test_irrelevant_ask_is_costly_on_clear_case(self):
        env = FoodPreferenceEnv(case('CLEAR'))
        obs, _ = env.reset()
        env.step('CLARIFY_DAY')
        env.step('ANSWER:' + public_support(obs)['supported_foods'][0])
        self.assertAlmostEqual(diagnostic_metrics(env)['return'], .9)
        self.assertEqual(diagnostic_metrics(env)['unnecessary_clarifications'], 1)

    def test_restriction_attempt_is_penalized_and_recorded(self):
        scenario = case('RESTRICTION')
        env = FoodPreferenceEnv(scenario)
        obs, _ = env.reset()
        forbidden = public_support(obs)['forbidden_foods'][0]
        reward = env.step('ANSWER:' + forbidden)[1]
        self.assertLess(reward, -2)
        self.assertFalse(diagnostic_metrics(env)['restriction_violation'])
        self.assertTrue(diagnostic_metrics(env)['attempted_restriction_violation'])
        self.assertIsNone(diagnostic_metrics(env)['selected_food'])
        self.assertFalse(diagnostic_metrics(env)['final_correct'])

    def test_genuine_conflict_preservation_is_rewarded(self):
        env = FoodPreferenceEnv(case('SAME_SCOPE'))
        env.reset()
        self.assertAlmostEqual(env.step('PRESERVE_CONFLICT')[1], .7)
        self.assertTrue(diagnostic_metrics(env)['final_correct'])

    def test_preserve_always_is_not_optimal(self):
        env = FoodPreferenceEnv(case('CLEAR'))
        env.reset()
        self.assertLess(env.step('PRESERVE_CONFLICT')[1], 1)

    def test_information_bonus_disabled_no_reward_for_message_length(self):
        with self.assertRaises(ValueError):
            RewardConfig(information_bonus=.2)
        with self.assertRaises(ValueError):
            RewardConfig(clarification_cost=-1)


class CorrectAbstentionTest(unittest.TestCase):
    def test_absent_evidence_allows_correct_abstention_without_making_it_profitable(self):
        scenario = case('CLEAR')
        scenario['oracle_preferences'] = []
        scenario['events'] = []
        scenario['latent']['acceptable_foods'] = []
        env = FoodPreferenceEnv(scenario)
        env.reset()
        self.assertLess(env.step('ABSTAIN')[1], 0)
        self.assertTrue(diagnostic_metrics(env)['decision_correct'])


class IndependentAuditTest(unittest.TestCase):
    def test_corrupt_policy_support_cannot_create_reward_or_grounding(self):
        from unittest.mock import patch
        scenario = case('CLEAR')
        env = FoodPreferenceEnv(scenario)
        obs, _ = env.reset()
        wrong = next(f for f in ('ensalada', 'pizza', 'sopa', 'arroz') if f not in scenario['latent']['acceptable_foods'])
        corrupted = {'supported_foods': [wrong], 'possible_foods': [wrong],
                     'forbidden_foods': [], 'conflict': False, 'missing_slots': [], 'unresolved': False}
        with patch('rl_feasibility.environment.public_support', return_value=corrupted):
            reward = env.step('ANSWER:' + wrong)[1]
        self.assertAlmostEqual(reward, -1.6)
        self.assertFalse(diagnostic_metrics(env)['evidence_supported'])
        self.assertFalse(diagnostic_metrics(env)['decision_correct'])


class PipelineSafetyAuditTest(unittest.TestCase):
    def test_missed_extraction_does_not_hide_actual_constraint_violation(self):
        from unittest.mock import patch
        from dataclasses import asdict
        from simulator.preferences import ParsedEvidence, Preference
        scenario = case('RESTRICTION')
        forbidden = scenario['oracle_restrictions'][0]['food_item']
        # Simulate an extractor that loses the stated medical restriction but
        # still extracts the taste. The public shield cannot know that record.
        parsed = ParsedEvidence(preferences=[Preference('p0', 'user', forbidden, 'LIKE',
            day_scope='ANY', meal_scope='ANY', evidence_id='e0')])
        with patch('rl_feasibility.environment.extract_preferences', return_value=parsed):
            env = FoodPreferenceEnv(scenario, condition='PIPELINE')
            env.reset()
        env.step('ANSWER:' + forbidden)
        metrics = diagnostic_metrics(env)
        self.assertTrue(metrics['restriction_violation'])
        self.assertTrue(metrics['attempted_restriction_violation'])
        self.assertFalse(metrics['safety_blocked'])
        self.assertFalse(metrics['decision_correct'])
