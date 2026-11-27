import unittest
import json
from pathlib import Path

from rl_feasibility.evaluate import compare_headroom, summarize
from rl_feasibility.statistics import paired_difference


class RLEvaluationTests(unittest.TestCase):
    def rows(self, value):
        return [{'scenario_id': str(i), 'return': value, 'actions': [], 'category': 'NONE'} for i in range(12)]

    def test_supervised_is_eligible_for_best_baseline_and_can_close_gate(self):
        rows = {'STRONG_REFERENCE': self.rows(.9), 'CURRENT_HEURISTIC': self.rows(.4),
                'SUPERVISED_POLICY': self.rows(.9)}
        result = compare_headroom(rows, bootstrap=20)
        self.assertEqual(result['best_baseline_selected_on_validation'], 'SUPERVISED_POLICY')
        self.assertEqual(result['gate_2'], 'NO_MEANINGFUL_MARGIN')
        self.assertEqual(result['headroom']['mean'], 0)

    def test_baseline_selection_can_be_frozen_before_test(self):
        rows = {'STRONG_REFERENCE': self.rows(.9), 'CURRENT_HEURISTIC': self.rows(.8),
                'SUPERVISED_POLICY': self.rows(.6)}
        result = compare_headroom(rows, selected='SUPERVISED_POLICY', bootstrap=20)
        self.assertEqual(result['best_baseline_selected_on_validation'], 'SUPERVISED_POLICY')

    def test_paired_statistics_keep_episode_alignment(self):
        left = [{'scenario_id': 'a', 'return': 1}, {'scenario_id': 'b', 'return': 0}]
        right = [{'scenario_id': 'b', 'return': -.1}, {'scenario_id': 'a', 'return': .9}]
        result = paired_difference(left, right, repetitions=20)
        self.assertAlmostEqual(result['mean'], .1)
        self.assertAlmostEqual(result['ci95'][0], .1)
        with self.assertRaises(ValueError):
            paired_difference(left, right[:1])

    def test_empty_clarification_denominator_not_perfect_score(self):
        result = summarize(self.rows(1), bootstrap=20)
        self.assertIsNone(result['clarification_quality']['value'])
        self.assertIsNone(result['restriction_violation_rate_exposed']['value'])

    def test_manual_candidates_use_the_public_environment_contract(self):
        from rl_feasibility.environment import FoodPreferenceEnv
        path = Path(__file__).parent / 'fixtures/rl_feasibility/manual_review_candidates.json'
        fixture = json.loads(path.read_text())
        self.assertFalse(fixture['provenance']['human_reviewed'])
        for scenario in fixture['scenarios']:
            with self.subTest(scenario=scenario['scenario_id']):
                observation, info = FoodPreferenceEnv(scenario).reset()
                self.assertNotIn('latent', observation)
                self.assertEqual(info, {})

    def test_wrong_answer_after_useful_question_is_not_a_missing_question(self):
        from rl_feasibility.environment import FoodPreferenceEnv, diagnostic_metrics
        from rl_feasibility.scenario_generator import generate_scenarios
        scenario = next(s for s in generate_scenarios(12, 42) if s['category'] == 'CPC_DAY')
        env = FoodPreferenceEnv(scenario)
        env.reset()
        env.step('CLARIFY_DAY')
        wrong = next(f for f in ('arroz', 'pizza', 'ensalada', 'sopa') if f not in scenario['latent']['acceptable_foods'])
        env.step('ANSWER:' + wrong)
        metrics = diagnostic_metrics(env)
        self.assertFalse(metrics['decision_correct'])
        self.assertTrue(metrics['necessary_information_acquired'])
        self.assertFalse(metrics['necessary_clarification_omitted'])
        self.assertTrue(metrics['initial_need_without_final_success'])

    def test_historical_adapter_preserves_communicated_company_and_place(self):
        from unittest.mock import patch
        from simulator.resolution import resolve_preferences
        from rl_feasibility.baselines import CurrentHeuristic
        from simulator.preferences import parse_context
        observation = {'query': '¿Qué debería comer?', 'context': {'company': 'friends', 'place': 'work'},
                       'preferences': [], 'restrictions': [], 'remaining_questions': 2, 'remaining_steps': 4}
        with patch('simulator.resolution.resolve_preferences', wraps=resolve_preferences) as resolver:
            CurrentHeuristic().select_action(observation)
            communicated = parse_context(resolver.call_args.args[2]).context
        self.assertEqual(communicated['company'], 'friends')
        self.assertEqual(communicated['place'], 'work')


if __name__ == '__main__':
    unittest.main()
