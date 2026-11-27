"""Policy privacy, imitation and checkpoint tests; no experimental training here."""
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from rl_feasibility.baselines import (
    StrongReference, NeverAsk, AlwaysAsk, SupervisedPolicy, feature_key,
    _canon_action, _real_action, action_catalog,
)
from rl_feasibility.train_pilot import QLearningPolicy, train_q_learning


def observation():
    return {'query':'¿Qué debería comer?', 'preferences':[{'food_item':'ensalada', 'polarity':'LIKE',
           'source':'USER_STATEMENT', 'preference_id':'p0', 'user_id':'u0'}], 'restrictions':[],
           'context':{}, 'messages':[], 'asked_slots':[], 'remaining_questions':2, 'remaining_steps':4}


SUPPORT = {'supported_foods':['ensalada'], 'possible_foods':['ensalada'], 'safe_foods':['ensalada','pizza','sopa','arroz'],
           'missing_slots':[], 'conflict':False, 'unresolved':False}


class TestPolicyPublicContract(unittest.TestCase):
    @patch('rl_feasibility.baselines._support', return_value=SUPPORT)
    def test_reference_avoids_unnecessary_questions(self, _):
        self.assertEqual(StrongReference().select_action(observation()), 'ANSWER:ensalada')
        self.assertEqual(NeverAsk().select_action(observation()), 'ANSWER:ensalada')

    @patch('rl_feasibility.baselines._support', return_value={**SUPPORT, 'conflict':True})
    def test_clean_option_does_not_require_resolving_another_foods_conflict(self, _):
        self.assertEqual(StrongReference().select_action(observation()), 'ANSWER:ensalada')
        self.assertEqual(NeverAsk().select_action(observation()), 'ANSWER:ensalada')

    @patch('rl_feasibility.baselines._support', return_value={**SUPPORT, 'supported_foods':[], 'missing_slots':['company']})
    def test_different_context_questions_are_selected(self, _):
        obs = observation()
        self.assertEqual(StrongReference().select_action(obs), 'CLARIFY_COMPANY')
        obs['asked_slots'] = ['company']
        self.assertNotEqual(StrongReference().select_action(obs), 'CLARIFY_COMPANY')

    @patch('rl_feasibility.baselines._support', return_value={**SUPPORT, 'conflict':True})
    def test_conflict_has_one_relevant_question_then_preserves(self, _):
        obs = observation()
        # This control has no clean alternative; a dispute about another food
        # must not force a question when an answer is already grounded.
        _.return_value['supported_foods'] = []
        self.assertEqual(StrongReference().select_action(obs), 'CLARIFY_CURRENT_PREFERENCE')
        obs['asked_slots'] = ['current_preference']
        self.assertEqual(StrongReference().select_action(obs), 'PRESERVE_CONFLICT')

    @patch('rl_feasibility.baselines._support', return_value=SUPPORT)
    def test_ids_seed_category_not_in_features(self, _):
        obs = observation()
        changed = copy.deepcopy(obs)
        changed.update(seed=98571, scenario_id='SECRET_OPTIMAL_ACTION_PIZZA', category='SCC', hidden_state={'food':'pizza'})
        changed['preferences'][0].update(preference_id='different', user_id='different', evidence_id='secret')
        self.assertEqual(feature_key(obs), feature_key(changed))
        self.assertEqual(StrongReference().select_action(obs), StrongReference().select_action(changed))

    @patch('rl_feasibility.baselines._support', return_value=SUPPORT)
    def test_actions_mask_budget_and_horizon(self, _):
        obs = observation()
        obs['remaining_questions'] = 0
        self.assertFalse(any(a.startswith('CLARIFY') for a in action_catalog(obs)))
        obs['remaining_questions'] = 2
        obs['remaining_steps'] = 1
        self.assertFalse(any(a.startswith('CLARIFY') for a in action_catalog(obs)))

    @patch('rl_feasibility.baselines._support', return_value=SUPPORT)
    def test_canonical_action_round_trip(self, _):
        obs = observation()
        for food in ('ensalada', 'pizza', 'sopa', 'arroz'):
            action = 'ANSWER:' + food
            self.assertEqual(_real_action(_canon_action(action, obs), obs), action)

    @patch('rl_feasibility.baselines._support', return_value=SUPPORT)
    def test_frozen_sft_and_q_checkpoint_roundtrip(self, _):
        obs = observation()
        key = feature_key(obs)
        sft = SupervisedPolicy({key:{'ANSWER@0':2}})
        self.assertEqual(SupervisedPolicy.from_dict(sft.to_dict()).select_action(obs),'ANSWER:ensalada')
        policy = QLearningPolicy({key:{'ANSWER@0':1.25, 'ABSTAIN':-.2}})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'policy.json'
            policy.save(path)
            restored = QLearningPolicy.load(path)
            self.assertEqual(restored.select_action(obs), 'ANSWER:ensalada')
            self.assertEqual(policy.to_dict(), restored.to_dict())

    def test_training_rejects_hidden_budget_expansion(self):
        # This validation happens before the lazy environment import can run an episode.
        with self.assertRaises(ValueError):
            train_q_learning([], episodes=1)


class TestPolicyTransitionIntegration(unittest.TestCase):
    def test_two_context_questions_then_grounded_food(self):
        from rl_feasibility.scenario_generator import generate_scenarios
        from rl_feasibility.environment import FoodPreferenceEnv, diagnostic_metrics
        scenario = next(s for s in generate_scenarios(24, seed=11) if s['category']=='CPC_TWO_SLOT')
        env = FoodPreferenceEnv(scenario)
        obs, _ = env.reset()
        actions = []
        teacher = StrongReference()
        for _ in range(4):
            action = teacher.select_action(obs)
            actions.append(action)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        self.assertEqual(actions[:2], ['CLARIFY_MEAL', 'CLARIFY_COMPANY'])
        self.assertTrue(actions[-1].startswith('ANSWER:'))
        self.assertTrue(diagnostic_metrics(env)['final_correct'])

    def test_teacher_cannot_predict_whether_user_can_resolve(self):
        from rl_feasibility.scenario_generator import generate_scenarios
        from rl_feasibility.environment import FoodPreferenceEnv
        scenarios = generate_scenarios(24, seed=11)
        resolvable = next(s for s in scenarios if s['category']=='SCC_RESOLVABLE')
        unresolved = copy.deepcopy(resolvable)
        unresolved['response_map']['current_preference'] = {'value':None, 'text':'No puedo aclarar mi preferencia actual.'}
        first, _ = FoodPreferenceEnv(resolvable).reset()
        second, _ = FoodPreferenceEnv(unresolved).reset()
        self.assertEqual(first, second)
        self.assertEqual(StrongReference().select_action(first), StrongReference().select_action(second))
