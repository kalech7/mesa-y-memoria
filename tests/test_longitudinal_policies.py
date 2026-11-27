"""Observable policy contracts; no scientific RL training is run by these tests."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl_longitudinal.baselines import (LastWriteWins, HistoricalRuleBased,
    ClarificationRuleBased, PlanningBaseline, SupervisedLongitudinal,
    feature_key, action_catalog, _canonical_action, _real_action, memory_operation)
from rl_longitudinal.memory_reconciliation import reconcile
from rl_longitudinal.train_rl import QLearningLongitudinal, validate_training_gates, train_q_learning, train_three_seeds


def fact(food='pizza', polarity='LIKE', *, day='ANY', kind='STABLE_PREFERENCE',
         at='2026-01-01T00:00:00Z', mid='m1', supersedes=()):
    return dict(memory_id=mid, user_id='user', food=food, polarity=polarity,
                scope=dict(day=day, meal='ANY', company='ANY', place='ANY'), kind=kind,
                observed_at=at, recorded_at=at, valid_from=at, valid_to=None,
                source='USER_STATEMENT', evidence_id='e' + mid, status='ACTIVE',
                supersedes=list(supersedes), uncertainty='KNOWN', text='Declaración visible')


def observation(facts=(), incoming=(), context=None, required=None):
    context = context or {}
    when = '2026-10-01T00:00:00Z'
    return dict(historical_memory=list(facts), incoming_evidence=list(incoming),
                context=context, query='¿Qué me recomiendas?', query_at=when,
                session_index=2, remaining_sessions=3, remaining_clarifications=2,
                clarification_history=[], user_available=True,
                reconciliation=asdict(reconcile(list(facts) + list(incoming), context, when)),
                required_decision=required or {'action': 'ANSWER', 'question_key': None})


class LongitudinalPolicyTests(unittest.TestCase):
    def test_last_write_is_actual_last_not_last_positive(self):
        obs = observation([fact(), fact(polarity='DISLIKE', at='2026-09-01T00:00:00Z', mid='m2')])
        self.assertEqual(LastWriteWins().select_action(obs)['conversation'], 'ABSTAIN')

    def test_history_rule_respects_scope_lww_ablation_does_not(self):
        obs = observation([fact('ensalada', day='WEEKDAY'),
                           fact('pizza', day='WEEKEND', at='2026-09-01T00:00:00Z', mid='m2')],
                          context={'day': 'WEEKDAY'})
        self.assertEqual(HistoricalRuleBased().select_action(obs)['food'], 'ensalada')
        self.assertEqual(LastWriteWins().select_action(obs)['food'], 'pizza')

    def test_lww_does_not_override_restriction(self):
        restriction = fact('maní', 'FORBID', kind='RESTRICTION')
        obs = observation([restriction, fact('maní', at='2026-09-01T00:00:00Z', mid='m2')])
        self.assertNotEqual(LastWriteWins().select_action(obs)['food'], 'maní')

    def test_clarification_rule_asks_specific_question(self):
        obs = observation([fact('pizza', day='WEEKEND')],
                          required={'action': 'CLARIFY', 'question_key': 'day'})
        action = ClarificationRuleBased().select_action(obs)
        self.assertEqual((action['conversation'], action['question_key']), ('CLARIFY', 'day'))

    def test_answered_question_is_not_repeated(self):
        obs = observation([fact('pizza', day='WEEKEND')],
                          required={'action': 'CONDITIONAL_ANSWER', 'question_key': 'day'})
        obs['clarification_history'] = [{'question_key': 'day', 'status': 'UNRESOLVED'}]
        self.assertNotEqual(ClarificationRuleBased().select_action(obs)['conversation'], 'CLARIFY')

    def test_latest_simultaneous_cohort_retains_explicit_positive_change(self):
        obs = observation([fact('sushi'),
                           fact('sushi', 'DISLIKE', at='2026-09-01T00:00:00Z', mid='m2'),
                           fact('ramen', at='2026-09-01T00:00:00Z', mid='m3')])
        self.assertEqual(LastWriteWins().select_action(obs)['food'], 'ramen')

    def test_new_occasion_can_reopen_same_context_question(self):
        obs = observation([fact('pizza', day='WEEKEND')],
                          required={'action': 'CLARIFY', 'question_key': 'day'})
        obs['clarification_history'] = [{'question_key': 'day', 'status': 'RESOLVED',
                                         'at': '2026-01-01T00:00:00Z',
                                         'query_at': '2026-01-01T00:00:00Z'}]
        self.assertEqual(ClarificationRuleBased().select_action(obs)['conversation'], 'CLARIFY')

    def test_new_evidence_can_reopen_previous_preference_question(self):
        obs = observation([fact(), fact(polarity='DISLIKE', at='2026-09-01T00:00:00Z', mid='m2')],
                          required={'action': 'CLARIFY', 'question_key': 'current:pizza'})
        obs['clarification_history'] = [{'question_key': 'current:pizza', 'status': 'UNRESOLVED',
                                         'at': '2026-05-01T00:00:00Z'}]
        self.assertEqual(ClarificationRuleBased().select_action(obs)['conversation'], 'CLARIFY')

    def test_explicit_change_and_ambiguity_memory_operations(self):
        explicit = observation(incoming=[fact(kind='EXPLICIT_CHANGE')])
        ambiguous = observation(incoming=[fact(kind='AMBIGUOUS')])
        self.assertEqual(memory_operation(explicit), 'UPDATE_WITH_HISTORY')
        self.assertEqual(memory_operation(ambiguous), 'PRESERVE_CONFLICT')
        self.assertEqual(memory_operation(observation()), 'NOOP')

    def test_features_ignore_ids_seeds_and_query_wording(self):
        original = observation([fact()])
        modified = deepcopy(original)
        modified.update(user_id='secret-user', seed=978, scenario_id='gold-ANSWER',
                        query='Completamente otra redacción', truth='not allowed information')
        modified['historical_memory'][0].update(memory_id='newid', evidence_id='new-eid', user_id='newuser')
        self.assertEqual(feature_key(original), feature_key(modified))

    def test_features_invariant_to_physical_record_order(self):
        original = observation([fact('pizza'), fact('sopa', mid='m2')])
        modified = deepcopy(original)
        modified['historical_memory'].reverse()
        self.assertEqual(feature_key(original), feature_key(modified))

    def test_features_distinguish_temporal_validity(self):
        original = observation([fact()])
        changed = deepcopy(original)
        changed['historical_memory'][0]['valid_to'] = '2026-02-01T00:00:00Z'
        self.assertNotEqual(feature_key(original), feature_key(changed))

    def test_action_round_trip_preserves_food_question_operation(self):
        obs = observation([fact()])
        action = dict(conversation='CLARIFY', food=None, question_key='current:pizza',
                      memory_operation='PRESERVE_CONFLICT')
        self.assertEqual(_real_action(_canonical_action(action, obs), obs), action)
        self.assertIn(_canonical_action(action, obs), action_catalog(obs))

    def test_sft_and_q_share_representation_and_action_space(self):
        obs = observation([fact()])
        action = dict(conversation='ANSWER', food='pizza', question_key=None, memory_operation='NOOP')
        encoded = _canonical_action(action, obs)
        sft = SupervisedLongitudinal({feature_key(obs): {encoded: 7}})
        q = QLearningLongitudinal({feature_key(obs): {encoded: 3.0}})
        self.assertEqual(sft.select_action(obs), action)
        self.assertEqual(q.select_action(obs), action)
        self.assertEqual(SupervisedLongitudinal.from_dict(sft.to_dict()).select_action(obs), action)
        self.assertEqual(QLearningLongitudinal.from_dict(q.to_dict()).select_action(obs), action)
        self.assertEqual(sft.usage_summary()['learned_table_decisions'], 1)
        sft.select_action(observation())
        self.assertEqual(sft.usage_summary()['rule_fallback_decisions'], 1)
        sft.reset_usage()
        self.assertEqual(sft.usage_summary()['decisions'], 0)

    def test_planner_searches_public_hypotheses_without_mutating_observation(self):
        obs = observation([fact('pizza', day='WEEKEND'), fact('sopa', day='WEEKDAY', mid='m2')],
                          required={'action': 'CLARIFY', 'question_key': 'day'})
        before = deepcopy(obs)
        policy = PlanningBaseline()
        action = policy.select_action(obs)
        self.assertEqual(action['conversation'], 'CLARIFY')
        self.assertEqual(action['question_key'], 'day')
        self.assertIn('day', policy.last_plan['question_values'])
        self.assertFalse(policy.last_plan['uses_actual_future_events_or_private_responses'])
        self.assertEqual(obs, before)

    def test_planner_keeps_unresolved_reply_branch(self):
        obs = observation([fact(), fact(polarity='DISLIKE', mid='m2')],
                          required={'action': 'CLARIFY', 'question_key': 'current:pizza'})
        branches = PlanningBaseline()._branches(obs, 'current:pizza')
        self.assertEqual(len(branches), 3)
        self.assertTrue(branches[0]['reconciliation']['conflicts'])

    def test_gate_rejection_precedes_environment_or_output_creation(self):
        called = []
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'must-not-exist'
            with self.assertRaises(ValueError):
                train_q_learning([{}], gate_report={'gate_a': False},
                                 env_factory=lambda s: called.append(s), output_dir=output)
            self.assertFalse(called)
            self.assertFalse(output.exists())

    def test_gate_cannot_use_test_or_subthreshold_difference(self):
        gates = dict(gate_a=True, gate_b=True, gate_c=True,
                     gate_d=dict(status='PASS', validation_only=True, minimum_difference=.10,
                                 difference=.12, paired_ci=[.02, .2], reference_method='PLANNING_BASELINE',
                                 best_non_rl_method='SUPERVISED_LONGITUDINAL'))
        self.assertEqual(validate_training_gates(gates), gates)
        gates['gate_d']['validation_only'] = False
        with self.assertRaises(ValueError):
            validate_training_gates(gates)
        gates['gate_d']['validation_only'] = True
        gates['gate_d']['difference'] = .09
        with self.assertRaises(ValueError):
            validate_training_gates(gates)

    def test_multi_seed_report_serializes_without_policy_objects(self):
        gates = dict(gate_a=True, gate_b=True, gate_c=True,
                     gate_d=dict(status='PASS', validation_only=True, minimum_difference=.10,
                                 difference=.12, paired_ci=[.02, .2], reference_method='PLANNING_BASELINE',
                                 best_non_rl_method='SUPERVISED_LONGITUDINAL'))
        # This test stubs learning; it does not execute or report a training run.
        def fake_training(*args, **kwargs):
            return {'policy': QLearningLongitudinal(),
                    'training_metrics': {'validation_normalized_return': kwargs['seed'] / 1000},
                    'episode_log': []}
        with tempfile.TemporaryDirectory() as tmp:
            with patch('rl_longitudinal.train_rl.train_q_learning', side_effect=fake_training):
                result = train_three_seeds([], [], gate_report=gates, output_dir=Path(tmp) / 'stub')
        selected = result.pop('selected_policy')
        self.assertIsInstance(selected, QLearningLongitudinal)
        self.assertEqual(len(result['results']), 3)
        json.dumps(result)


if __name__ == '__main__':
    unittest.main()
