"""Real SQLite and public/action contracts for SPEC-LLM-LOCAL-001 (2026-10-23)."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from llm_local_001.environment import (
    LLMEnvironment, format_prompt, historical_rule_policy, parse_action,
    question_intent, run_episode, run_historical_episode,
)
from rl_small_pilot.environment_adapter import generate_splits


def scenarios(n=10):
    return generate_splits(n, n, n, seed=1701)


def scenario(family):
    return next(value for value in scenarios()['train'] if value['family'] == family)


def encoded(conversation='ANSWER', memory='NOOP', polarity='LIKE', ids=(), question=None, slot=None):
    result = {'conversation_action': conversation, 'memory_action': memory,
              'question': question, 'answer': {'polarity': polarity, 'text': 'Respuesta estructurada.'}
              if conversation == 'ANSWER' else None, 'evidence_ids': list(ids)}
    if slot is not None:
        result['question_slot'] = slot
    return json.dumps(result, ensure_ascii=False)


def rule_generator(prompt):
    return historical_rule_policy(json.JSONDecoder().raw_decode(prompt.split('OBSERVATION:\n', 1)[1])[0])


class PublicEvidenceContracts(unittest.TestCase):
    def test_private_labels_and_future_reply_cannot_change_prompt(self):
        original = scenario('RELEVANT_RESOLVABLE')
        changed = deepcopy(original)
        for session in changed['sessions']:
            session['gold'] = {'polarity': 'FORBID', 'needs_clarification': False,
                               'relevant_memory_ids': [], 'temporal': False, 'conflict': False}
        changed['sessions'][1]['responses']['preference']['text'] = 'PRIVATE FUTURE ANSWER'
        for representation in ('O1', 'O2'):
            envs = [LLMEnvironment(value, representation) for value in (original, changed)]
            try:
                prompts = [format_prompt(env.reset()) for env in envs]
                self.assertEqual(*prompts)
                visible = json.JSONDecoder().raw_decode(prompts[0].split('OBSERVATION:\n')[1])[0]
                def keys(value):
                    if isinstance(value, dict):
                        for key, child in value.items():
                            yield key
                            yield from keys(child)
                    elif isinstance(value, list):
                        for child in value:
                            yield from keys(child)
                forbidden = {'gold', 'kind', 'polarity', 'required_decision', 'reward',
                             'expected_polarity', 'family', 'responses', 'user_id'}
                self.assertFalse(forbidden.intersection(keys(visible)))
                self.assertNotIn('PRIVATE FUTURE ANSWER', prompts[0])
            finally:
                for env in envs:
                    env.close()

    def test_same_model_interface_changes_only_selected_visible_history(self):
        envs = [LLMEnvironment(scenario('TEMPORARY_EXCEPTION'), rep) for rep in ('O1', 'O2')]
        try:
            a, b = [env.reset() for env in envs]
            self.assertEqual(len([f for f in a['memory'] if f['food'] != 'maní']), 1)
            self.assertEqual(len([f for f in b['memory'] if f['food'] != 'maní']), 2)
            self.assertTrue(any(f['food'] == 'maní' for f in a['memory']))
            for key in ('query', 'query_at', 'context', 'remaining_clarifications'):
                self.assertEqual(a[key], b[key])
        finally:
            for env in envs:
                env.close()

    def test_logged_prompt_matches_actual_model_input_after_clarification(self):
        seen = []
        def generator(prompt):
            seen.append(prompt)
            return rule_generator(prompt)
        row = run_episode(scenario('RELEVANT_RESOLVABLE'), generator)
        self.assertEqual(seen, [generation['prompt'] for generation in row['generations']])
        after_clarification = json.JSONDecoder().raw_decode(seen[1].split('OBSERVATION:\n')[1])[0]
        self.assertIn('question', after_clarification['clarification_history'][0])
        self.assertEqual(row['format_errors'], 0)

    def test_o3_requires_identified_need(self):
        with self.assertRaises(ValueError):
            LLMEnvironment(scenario('EXPLICIT_CHANGE'), 'O3')


class GenerationExecutionContracts(unittest.TestCase):
    def test_invalid_generations_are_failed_decisions_and_preserved_in_denominator(self):
        row = run_episode(scenario('EXPLICIT_CHANGE'), lambda _: '```json\n{}\n```')
        self.assertTrue(row['terminated'])
        self.assertEqual(row['decisions'], 2)
        self.assertEqual(row['decision_errors'], 2)
        self.assertEqual(row['format_errors'], 2)
        self.assertEqual(row['return'], -2)
        self.assertTrue(all(step['format_error'] for step in row['steps']))
        self.assertTrue(all(step['executed_action']['conversation'] == 'NOOP' for step in row['steps']))

    def test_missing_evidence_invalid_id_duplicate_keys_and_nonjson_are_not_repaired(self):
        env = LLMEnvironment(scenario('EXPLICIT_CHANGE'))
        try:
            observation = env.reset()
            value = encoded(ids=['not_retrieved'])
            with self.assertRaises(ValueError):
                parse_action(value, observation)
            valid = encoded(ids=[observation['memory'][0]['evidence_id']])
            self.assertEqual(parse_action(valid, observation)['conversation_action'], 'ANSWER')
            invalid = valid[:-1] + ', "conversation_action":"CLARIFY"}'
            for raw in (invalid, valid + '\ncomment', '{}', 'null', 'NaN'):
                with self.assertRaises(ValueError):
                    parse_action(raw, observation)
        finally:
            env.close()

    def test_answer_without_citations_is_scored_unsupported_with_original_penalty(self):
        value = scenario('EXPLICIT_CHANGE')
        env = LLMEnvironment(value)
        try:
            env.reset()
            polarity = value['sessions'][1]['gold']['polarity']
            _, reward, _, _, _ = env.step(encoded(polarity=polarity))
            step = env.diagnostic_metrics()['steps'][0]
            self.assertFalse(step['correct'])
            self.assertTrue(step['unsupported_answer'])
            self.assertAlmostEqual(reward, -1.6)
        finally:
            env.close()

    def test_citations_cannot_cherry_pick_one_side_of_visible_conflict(self):
        value = scenario('RELEVANT_RESOLVABLE')
        env = LLMEnvironment(value)
        try:
            observation = env.reset()
            evidence = value['sessions'][1]['incoming'][0]
            _, reward, _, _, _ = env.step(encoded(polarity=evidence['polarity'], ids=[evidence['evidence_id']]))
            self.assertEqual(reward, -1.6)
            self.assertEqual(env.diagnostic_metrics()['conflicts_not_detected'], 1)
        finally:
            env.close()

    def test_irrelevant_question_cannot_obtain_scripted_reply_by_forging_slot(self):
        env = LLMEnvironment(scenario('RELEVANT_RESOLVABLE'))
        try:
            observation = env.reset()
            raw = encoded('CLARIFY', question='¿Cuál es tu edad?', slot='preference')
            next_observation, reward, done, _, info = env.step(raw)
            self.assertFalse(done)
            self.assertAlmostEqual(reward, -0.1)
            self.assertTrue(info['question_routing_error'])
            self.assertFalse(env._resolved_reply)
            self.assertFalse(any(f['source'] == 'CLARIFICATION' for f in next_observation['memory']))
            self.assertEqual(env.diagnostic_metrics()['steps'][0]['executed_action']['slot'], 'unspecified')
            self.assertEqual(env.diagnostic_metrics()['conflicts_not_detected'], 0)  # session not terminal yet
        finally:
            env.close()

    def test_question_router_routes_actual_semantics_over_declared_slot(self):
        self.assertEqual(question_intent('¿Ese cambio es solo para hoy o es permanente?', 'arroz'), 'preference')
        self.assertEqual(question_intent('¿Cuál es la fuente de esa información?', 'arroz'), 'source')
        self.assertEqual(question_intent('¿Para qué comida: desayuno, almuerzo o cena?', 'arroz'), 'meal')
        self.assertEqual(question_intent('¿Es un día laborable o fin de semana?', 'arroz'), 'day')
        self.assertEqual(question_intent('¿Me gusta programar?', 'arroz'), 'unspecified')
        self.assertEqual(question_intent('¿Cuál es tu preferencia sobre programación?', 'arroz'), 'unspecified')
        self.assertEqual(question_intent('¿Te gusta el arroz para el almuerzo?', 'arroz'), 'preference')

    def test_unnecessary_question_receives_no_positive_reward(self):
        env = LLMEnvironment(scenario('EXPLICIT_CHANGE'))
        try:
            env.reset()
            _, reward, _, _, _ = env.step(encoded('CLARIFY', question='¿Cambió tu preferencia?', slot='preference'))
            self.assertAlmostEqual(reward, -0.1)
            self.assertFalse(env.diagnostic_metrics()['steps'][0]['necessary_question'])
        finally:
            env.close()

    def test_unjustified_update_is_executed_penalized_and_can_corrupt_s3(self):
        env = LLMEnvironment(scenario('TEMPORARY_EXCEPTION'))
        try:
            observation = env.reset()
            answer = json.loads(historical_rule_policy(observation))
            answer['memory_action'] = 'UPDATE_WITH_HISTORY'
            next_observation, reward, _, _, _ = env.step(json.dumps(answer))
            self.assertAlmostEqual(reward, 0.5)
            # Its real consequences persist through SQLite close/reopen into S3.
            _, future_reward, done, _, _ = env.step(historical_rule_policy(next_observation))
            row = env.diagnostic_metrics()
            self.assertTrue(done)
            self.assertEqual(row['incorrect_updates'], 1)
            self.assertEqual(row['future_errors'], 1)
            self.assertLess(future_reward, 0)
        finally:
            env.close()

    def test_historical_database_is_preserved_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            db = Path(temporary) / 'trajectory.sqlite'
            row = run_historical_episode(scenario('EXPLICIT_CHANGE'), db_path=db)
            self.assertTrue(db.is_file())
            self.assertTrue(row['operations'])
            with self.assertRaises(FileExistsError):
                run_historical_episode(scenario('EXPLICIT_CHANGE'), db_path=db)


class HistoricalBaselineContracts(unittest.TestCase):
    def test_text_only_baseline_matches_native_without_degradation_in_all_families_and_splits(self):
        for split, values in scenarios().items():
            for value in values:
                with self.subTest(split=split, family=value['family'], direction=value['initial_direction']):
                    native = run_historical_episode(value)
                    textual = run_episode(value, rule_generator)
                    for key in ('return', 'future_accuracy', 'error_rate', 'necessary_asks', 'unnecessary_asks', 'incorrect_updates'):
                        self.assertEqual(native[key], textual[key], key)
                    self.assertEqual(textual['format_errors'], 0)
                    self.assertEqual(native['decision_errors'], 0)
                    self.assertIn('symbolic_interpreter_conflicts_detected', native)
                    self.assertEqual(native['conflicts_not_detected'], 0)


if __name__ == '__main__':
    unittest.main()
