"""Open recommendations admit independent, safe alternatives to disputed foods."""
from copy import deepcopy
from dataclasses import asdict
import unittest

from simulator.preferences import Preference
from rl_feasibility.environment import FoodPreferenceEnv, audit_evidence, diagnostic_metrics, public_support
from rl_feasibility.scenario_generator import generate_scenarios

TIME = '2026-10-19T12:00:00Z'


def record(food, polarity='LIKE', source='USER_STATEMENT', **kwargs):
    return asdict(Preference('p', 'user', food, polarity, day_scope='ANY',
        meal_scope='ANY', source=source, timestamp=TIME, evidence_id='e', **kwargs))


def observation(records):
    return {'preferences': records, 'restrictions': [], 'context': {}, 'reference_time': TIME}


class LocalEvidenceTests(unittest.TestCase):
    def test_disputed_other_food_does_not_invalidate_clean_recommendation(self):
        obs = observation([record('arroz'), record('pizza', 'DISLIKE'),
                           record('pizza', source='ORDER_HISTORY')])
        self.assertEqual(public_support(obs)['supported_foods'], ['arroz'])
        self.assertTrue(public_support(obs)['conflict'])
        self.assertTrue(audit_evidence(obs, 'arroz')['supported'])
        self.assertFalse(audit_evidence(obs, 'pizza')['supported'])

    def test_reordering_or_more_contradictory_evidence_about_other_food_is_invariant(self):
        records = [record('arroz'), record('pizza', 'DISLIKE'),
                   record('pizza', source='ORDER_HISTORY')]
        for changed in (list(reversed(records)), records + [record('pizza', stability='OSCILLATING')]):
            obs = observation(changed)
            self.assertIn('arroz', public_support(obs)['supported_foods'])
            self.assertTrue(audit_evidence(obs, 'arroz')['supported'])
            self.assertTrue(audit_evidence(obs)['conflict'])

    def test_missing_scope_for_other_food_does_not_block_general_option(self):
        conditional = record('pizza')
        conditional['meal_scope'] = 'LUNCH'
        obs = observation([record('arroz'), conditional])
        self.assertEqual(public_support(obs)['supported_foods'], ['arroz'])
        self.assertIn('meal', public_support(obs)['missing_slots'])
        self.assertTrue(audit_evidence(obs, 'arroz')['supported'])

    def test_local_conflict_still_blocks_the_disputed_food(self):
        obs = observation([record('arroz'), record('arroz', 'DISLIKE'), record('pizza')])
        self.assertEqual(public_support(obs)['supported_foods'], ['pizza'])
        self.assertFalse(audit_evidence(obs, 'arroz')['supported'])
        self.assertTrue(audit_evidence(obs, 'pizza')['supported'])


class OpenRecommendationAuthorTruthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = generate_scenarios(240, 314)

    def find(self, category):
        return next(s for s in self.scenarios if s['category'] == category)

    def test_scc_clean_declaration_is_always_an_accepted_answer(self):
        for scenario in self.scenarios:
            if not scenario['category'].startswith('SCC'):
                continue
            clean = scenario['oracle_preferences'][0]['food_item']
            self.assertIn(clean, scenario['latent']['acceptable_foods'])
            env = FoodPreferenceEnv(scenario)
            env.reset()
            self.assertEqual(env.step('ANSWER:' + clean)[1], 1.0)
            metrics = diagnostic_metrics(env)
            self.assertTrue(metrics['decision_correct'])
            self.assertFalse(metrics['necessary_clarification_required'])

    def test_current_choice_is_separate_and_multiple_answers_can_remain_justified(self):
        scenario = next(s for s in self.scenarios if s['category'] == 'SCC_RESOLVABLE'
                        and len(s['latent']['acceptable_foods']) == 2)
        self.assertEqual(scenario['response_map']['current_preference']['value'],
                         scenario['latent']['current_choice'])
        for food in scenario['latent']['acceptable_foods']:
            env = FoodPreferenceEnv(scenario)
            env.reset()
            obs, *_ = env.step('CLARIFY_CURRENT_PREFERENCE')
            self.assertIn(food, public_support(obs)['supported_foods'])
            self.assertTrue(audit_evidence(obs, food)['supported'])
            env.step('ANSWER:' + food)
            self.assertTrue(diagnostic_metrics(env)['decision_correct'])

    def test_same_scope_conflict_allows_clean_alternative_or_preservation(self):
        scenario = self.find('SAME_SCOPE')
        clean = scenario['oracle_preferences'][2]['food_item']
        self.assertEqual(scenario['latent']['acceptable_foods'], [clean])
        for action, expected_reward in (('ANSWER:' + clean, 1.0), ('PRESERVE_CONFLICT', .7)):
            env = FoodPreferenceEnv(scenario)
            env.reset()
            self.assertAlmostEqual(env.step(action)[1], expected_reward)
            metrics = diagnostic_metrics(env)
            self.assertTrue(metrics['decision_correct'])
            self.assertTrue(metrics['conflict_unresolvable'])
            self.assertFalse(metrics['decision_unresolvable'])

    def test_unresolvable_oscillation_can_still_prevent_any_grounded_answer(self):
        env = FoodPreferenceEnv(self.find('BOC_UNRESOLVABLE'))
        obs, _ = env.reset()
        self.assertFalse(public_support(obs)['supported_foods'])
        env.step('PRESERVE_CONFLICT')
        self.assertTrue(diagnostic_metrics(env)['decision_unresolvable'])


class ClarificationTaskUtilityTests(unittest.TestCase):
    def test_informative_question_is_still_avoidable_when_clean_answer_exists(self):
        scenario = next(s for s in generate_scenarios(240, 314)
                        if s['category'] == 'SCC_RESOLVABLE'
                        and len(s['latent']['acceptable_foods']) == 2)
        env = FoodPreferenceEnv(scenario)
        initial, _ = env.reset()
        clean = public_support(initial)['supported_foods'][0]
        env.step('CLARIFY_CURRENT_PREFERENCE')
        env.step('ANSWER:' + clean)
        metrics = diagnostic_metrics(env)
        self.assertEqual(metrics['useful_clarifications'], 1)
        self.assertEqual(metrics['task_useful_clarifications'], 0)
        self.assertEqual(metrics['unnecessary_clarifications'], 1)
        self.assertTrue(metrics['transitions'][0]['sufficient_before'])
        self.assertAlmostEqual(metrics['return'], .9)


class SameFoodPartitionTests(unittest.TestCase):
    def test_opposite_tastes_in_disjoint_meals_require_scope_not_conflict_resolution(self):
        lunch = record('arroz')
        dinner = record('arroz', 'DISLIKE')
        lunch['meal_scope'], dinner['meal_scope'] = 'LUNCH', 'DINNER'
        obs = observation([lunch, dinner])
        self.assertFalse(public_support(obs)['conflict'])
        self.assertEqual(public_support(obs)['missing_slots'], ['meal'])
        self.assertFalse(public_support(obs)['supported_foods'])
        self.assertFalse(audit_evidence(obs)['conflict'])
        obs['context']['meal'] = 'LUNCH'
        self.assertEqual(public_support(obs)['supported_foods'], ['arroz'])
        self.assertTrue(audit_evidence(obs, 'arroz')['supported'])
        self.assertEqual(len(obs['preferences']), 2)
        obs['context']['meal'] = 'DINNER'
        self.assertFalse(public_support(obs)['supported_foods'])
        self.assertFalse(audit_evidence(obs, 'arroz')['supported'])
        self.assertEqual(len(obs['preferences']), 2)

    def test_disjoint_company_conditions_are_not_same_scope_conflict(self):
        alone = record('arroz', context={'company': 'alone'})
        friends = record('arroz', 'DISLIKE', context={'company': 'friends'})
        obs = observation([alone, friends])
        self.assertFalse(public_support(obs)['conflict'])
        self.assertIn('company', public_support(obs)['missing_slots'])
        obs['context']['company'] = 'alone'
        self.assertEqual(public_support(obs)['supported_foods'], ['arroz'])
