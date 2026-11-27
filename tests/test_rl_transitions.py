import unittest

from rl_feasibility.environment import FoodPreferenceEnv, public_support, diagnostic_metrics
from tests.test_rl_environment import case


class TransitionTests(unittest.TestCase):
    def test_relevant_and_irrelevant_questions_have_different_consequences(self):
        scenario = case('CPC_COMPANY')
        a, b = FoodPreferenceEnv(scenario), FoodPreferenceEnv(scenario)
        self.assertEqual(a.reset()[0], b.reset()[0])
        good = a.step('CLARIFY_COMPANY')[0]
        bad = b.step('CLARIFY_MEAL')[0]
        self.assertTrue(public_support(good)['supported_foods'])
        self.assertFalse(public_support(bad)['supported_foods'])

    def test_two_questions_resolve_conjunction_but_one_does_not(self):
        env = FoodPreferenceEnv(case('CPC_TWO_SLOT'))
        env.reset()
        obs, *_ = env.step('CLARIFY_MEAL')
        self.assertFalse(public_support(obs)['supported_foods'])
        obs, *_ = env.step('CLARIFY_COMPANY')
        foods = public_support(obs)['supported_foods']
        self.assertEqual(len(foods), 1)
        env.step('ANSWER:' + foods[0])
        self.assertTrue(diagnostic_metrics(env)['final_correct'])

    def test_current_clarification_keeps_history_and_resolves_choice(self):
        env = FoodPreferenceEnv(case('BOC_RESOLVABLE'))
        obs, _ = env.reset()
        original_ids = {p['evidence_id'] for p in obs['preferences']}
        obs, *_ = env.step('CLARIFY_CURRENT_PREFERENCE')
        self.assertTrue(original_ids <= {p['evidence_id'] for p in obs['preferences']})
        foods = public_support(obs)['supported_foods']
        self.assertEqual(len(foods), 1)
        env.step('ANSWER:' + foods[0])
        self.assertTrue(diagnostic_metrics(env)['final_correct'])

    def test_unresolvable_reply_confirms_uncertainty(self):
        env = FoodPreferenceEnv(case('SCC_UNRESOLVABLE'))
        env.reset()
        obs, *_ = env.step('CLARIFY_CURRENT_PREFERENCE')
        self.assertTrue(public_support(obs)['unresolved'])
        # Unresolved evidence about one food does not erase another clean taste.
        self.assertTrue(public_support(obs)['conflict'])
        self.assertTrue(public_support(obs)['supported_foods'])
        env.step('PRESERVE_CONFLICT')
        self.assertTrue(diagnostic_metrics(env)['correct_conflict_preservation'])

    def test_asking_twice_is_not_new_information(self):
        env = FoodPreferenceEnv(case('CPC_DAY'))
        env.reset()
        env.step('CLARIFY_DAY')
        env.step('CLARIFY_DAY')
        env.step('ABSTAIN')
        metrics = diagnostic_metrics(env)
        self.assertEqual(metrics['useful_clarifications'], 1)
        self.assertEqual(metrics['unnecessary_clarifications'], 1)

    def test_source_contradiction_never_removes_original_evidence(self):
        env = FoodPreferenceEnv(case('SCC_RESOLVABLE'))
        before, _ = env.reset()
        after = env.step('CLARIFY_CURRENT_PREFERENCE')[0]
        self.assertEqual(after['preferences'][:len(before['preferences'])], before['preferences'])
        self.assertTrue(any(p['source'] == 'ORDER_HISTORY' for p in after['preferences']))
