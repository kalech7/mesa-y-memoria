import unittest
from copy import deepcopy

from rl_feasibility.environment import ACTIONS, FoodPreferenceEnv, diagnostic_metrics, public_support
from rl_feasibility.scenario_generator import generate_scenarios


def case(category, seed=42):
    return next(s for s in generate_scenarios(12, seed) if s['category'] == category)


class EnvironmentContractTests(unittest.TestCase):
    def test_reset_replays_identical_public_state(self):
        env = FoodPreferenceEnv(case('CPC_COMPANY'))
        initial, info = env.reset(seed=1)
        after = env.step('CLARIFY_COMPANY')
        replay, _ = env.reset(seed=1)
        self.assertEqual(initial, replay)
        self.assertEqual(after, env.step('CLARIFY_COMPANY'))
        self.assertEqual(info, {})

    def test_terminal_requires_reset(self):
        env = FoodPreferenceEnv(case('CLEAR'))
        env.reset()
        _, _, terminal, truncated, info = env.step('ABSTAIN')
        self.assertTrue(terminal)
        self.assertFalse(truncated)
        self.assertEqual(info, {})
        with self.assertRaises(RuntimeError):
            env.step('ABSTAIN')

    def test_horizon_truncates_and_charges_terminal_failure(self):
        env = FoodPreferenceEnv(case('CPC_TWO_SLOT'), max_steps=1)
        env.reset()
        _, reward, terminated, truncated, _ = env.step('CLARIFY_COMPANY')
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertAlmostEqual(reward, -.45)
        self.assertFalse(diagnostic_metrics(env)['final_correct'])

    def test_two_question_budget_enforced(self):
        env = FoodPreferenceEnv(case('CPC_TWO_SLOT'))
        env.reset()
        env.step('CLARIFY_COMPANY')
        env.step('CLARIFY_MEAL')
        before = env.observation()
        with self.assertRaises(ValueError):
            env.step('CLARIFY_DAY')
        self.assertEqual(before, env.observation())

    def test_invalid_actions_do_not_mutate_episode(self):
        env = FoodPreferenceEnv(case('CLEAR'))
        obs, _ = env.reset()
        with self.assertRaises(ValueError):
            env.step('VERIFY')
        self.assertEqual(obs, env.observation())

    def test_outside_observation_mutation_is_isolated(self):
        env = FoodPreferenceEnv(case('CLEAR'))
        obs, _ = env.reset()
        obs['preferences'][0]['food_item'] = 'marisco'
        self.assertNotEqual(env.observation()['preferences'][0]['food_item'], 'marisco')

    def test_pipeline_extracts_from_text_instead_of_oracle_record(self):
        scenario = case('CLEAR')
        scenario['oracle_preferences'][0]['food_item'] = 'imaginary'
        env = FoodPreferenceEnv(scenario, condition='PIPELINE')
        obs, _ = env.reset()
        self.assertNotIn('imaginary', [p['food_item'] for p in obs['preferences']])
        self.assertTrue(obs['preferences'])

    def test_private_diagnostics_not_released_early(self):
        env = FoodPreferenceEnv(case('CLEAR'))
        env.reset()
        with self.assertRaises(ValueError):
            diagnostic_metrics(env)

    def test_generator_reproducible_split_users_disjoint(self):
        self.assertEqual(generate_scenarios(24, 42), generate_scenarios(24, 42))
        train = generate_scenarios(24, 42)
        test = generate_scenarios(24, 42, 'test')
        self.assertFalse({s['user_id'] for s in train} & {s['user_id'] for s in test})
        self.assertFalse({s['query'] for s in train} & {s['query'] for s in test})

    def test_future_evidence_is_not_observed(self):
        scenario = case('CLEAR')
        future = deepcopy(scenario['oracle_preferences'][0])
        future.update(preference_id='future', evidence_id='future', timestamp='2027-01-01T00:00:00Z')
        scenario['oracle_preferences'].append(future)
        scenario['events'].append({'id': 'future', 'text': 'Future secret event', 'timestamp': future['timestamp']})
        for condition in ('ORACLE_STRUCTURED', 'PIPELINE'):
            obs, _ = FoodPreferenceEnv(scenario, condition).reset()
            self.assertNotIn('Future secret event', str(obs))
            self.assertNotIn('future', [p['preference_id'] for p in obs['preferences']])


class EvidenceAndLatentBoundaryTests(unittest.TestCase):
    def test_pipeline_preserves_evidence_ids_and_timestamps(self):
        scenario = case('CLEAR')
        obs, _ = FoodPreferenceEnv(scenario, 'PIPELINE').reset()
        self.assertEqual(obs['preferences'][0]['evidence_id'], 'e0')
        self.assertEqual(obs['preferences'][0]['timestamp'], scenario['reference_time'])

    def test_evaluation_does_not_call_existing_heuristic(self):
        from unittest.mock import patch
        scenario = case('CLEAR')
        with patch('simulator.resolution.resolve_preferences', side_effect=AssertionError('reward circularity')):
            env = FoodPreferenceEnv(scenario)
            obs, _ = env.reset()
            env.step('ANSWER:' + public_support(obs)['supported_foods'][0])
            self.assertTrue(diagnostic_metrics(env)['final_correct'])

    def test_private_situation_changes_correctness_not_initial_observation(self):
        scenario = case('CPC_DAY')
        alternative = deepcopy(scenario)
        alternative['latent']['context']['day'] = ('WEEKEND' if scenario['latent']['context']['day'] == 'WEEKDAY' else 'WEEKDAY')
        foods = [p['food_item'] for p in scenario['oracle_preferences']]
        alternative['latent']['acceptable_foods'] = [next(f for f in foods if f not in scenario['latent']['acceptable_foods'])]
        a, b = FoodPreferenceEnv(scenario), FoodPreferenceEnv(alternative)
        self.assertEqual(a.reset()[0], b.reset()[0])
        action = 'ANSWER:' + scenario['latent']['acceptable_foods'][0]
        self.assertNotEqual(a.step(action)[1], b.step(action)[1])

    def test_historical_temporal_exception_is_not_applicable_today(self):
        scenario = case('CLEAR')
        alternative = next(f for f in ('ensalada', 'pizza', 'sopa', 'arroz') if f != scenario['oracle_preferences'][0]['food_item'])
        old = deepcopy(scenario['oracle_preferences'][0])
        old.update(preference_id='p_old', evidence_id='e_old', food_item=alternative, valid_to='2026-10-18')
        scenario['oracle_preferences'].append(old)
        scenario['events'].append({'id': 'e_old', 'text': 'Antes prefería otra opción.', 'timestamp': scenario['reference_time']})
        obs, _ = FoodPreferenceEnv(scenario).reset()
        self.assertIn('p_old', [p['preference_id'] for p in obs['preferences']])
        self.assertNotIn(alternative, public_support(obs)['supported_foods'])

    def test_future_valid_from_assertion_visible_but_not_applicable(self):
        scenario = case('CLEAR')
        alternative = next(f for f in ('ensalada', 'pizza', 'sopa', 'arroz') if f != scenario['oracle_preferences'][0]['food_item'])
        future = deepcopy(scenario['oracle_preferences'][0])
        future.update(preference_id='p_future', evidence_id='e_future', food_item=alternative, valid_from='2026-10-20')
        scenario['oracle_preferences'].append(future)
        scenario['events'].append({'id': 'e_future', 'text': 'Mañana tendré otra preferencia.', 'timestamp': scenario['reference_time']})
        obs, _ = FoodPreferenceEnv(scenario).reset()
        self.assertIn('p_future', [p['preference_id'] for p in obs['preferences']])
        self.assertNotIn(alternative, public_support(obs)['supported_foods'])


class ClosedObservationSchemaTests(unittest.TestCase):
    def test_private_fields_in_memory_rejected(self):
        for field in ('latent', 'expected', 'category', 'other_unapproved_field'):
            scenario = case('CLEAR')
            scenario['oracle_preferences'][0][field] = 'SECRET'
            with self.subTest(field=field), self.assertRaises(ValueError):
                FoodPreferenceEnv(scenario)

    def test_nested_private_context_rejected(self):
        scenario = case('CLEAR')
        scenario['oracle_preferences'][0]['context'] = {'place': {'latent': 'SECRET'}}
        with self.assertRaises(ValueError):
            FoodPreferenceEnv(scenario)

    def test_private_fields_cannot_be_injected_via_pipeline_events(self):
        scenario = case('CLEAR')
        scenario['events'][0]['context'] = {'expected': 'SECRET'}
        with self.assertRaises(ValueError):
            FoodPreferenceEnv(scenario, condition='PIPELINE')

    def test_invalid_future_value_cannot_be_declassified(self):
        scenario = case('CLEAR')
        scenario['response_map']['meal']['value'] = {'latent': 'SECRET'}
        with self.assertRaises(ValueError):
            FoodPreferenceEnv(scenario)


class EvidenceLinkTests(unittest.TestCase):
    def test_uncommunicated_oracle_record_never_becomes_observation(self):
        scenario = case('CLEAR')
        uncommunicated = deepcopy(scenario['oracle_preferences'][0])
        uncommunicated.update(preference_id='private_record', evidence_id='not_communicated')
        scenario['oracle_preferences'].append(uncommunicated)
        obs, _ = FoodPreferenceEnv(scenario).reset()
        self.assertNotIn('private_record', [p['preference_id'] for p in obs['preferences']])

    def test_original_and_reply_evidence_expose_provenance(self):
        env = FoodPreferenceEnv(case('CPC_DAY'))
        obs, _ = env.reset()
        self.assertEqual(obs['messages'][0]['evidence_id'], 'e0')
        self.assertEqual(obs['messages'][0]['source'], 'USER_STATEMENT')
        after, *_ = env.step('CLARIFY_DAY')
        self.assertEqual(after['messages'][-1]['evidence_id'], 'ec1')
        self.assertEqual(after['messages'][-1]['timestamp'], obs['reference_time'])


class TaskDenominatorTests(unittest.TestCase):
    def test_parser_failure_does_not_erase_required_clarification_denominator(self):
        from unittest.mock import patch
        from simulator.preferences import ParsedEvidence
        with patch('rl_feasibility.environment.extract_preferences', return_value=ParsedEvidence()):
            env = FoodPreferenceEnv(case('CPC_DAY'), condition='PIPELINE')
            obs, _ = env.reset()
        self.assertEqual(obs['preferences'], [])
        env.step('ABSTAIN')
        metrics = diagnostic_metrics(env)
        self.assertTrue(metrics['necessary_clarification_required'])
        self.assertTrue(metrics['necessary_clarification_omitted'])
