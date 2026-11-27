"""Independent hand-written contracts for the deterministic resolution engine."""
from dataclasses import replace
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest
from memory.store import Store
from simulator.preferences import Preference, Restriction
from simulator.resolution import FoodPreferenceEngine, resolve_preferences


def pref(identifier='a', food='avena', polarity='LIKE', **kwargs):
    return Preference(preference_id=identifier, user_id='u', food_item=food,
                      polarity=polarity, evidence_id=identifier, timestamp='2026-01-01T09:00:00',
                      day_scope=kwargs.pop('day_scope', 'ANY'), meal_scope=kwargs.pop('meal_scope', 'ANY'), **kwargs)


def resolve(records, query='¿Qué prefiero en general?', restrictions=(), reference='2026-10-19'):
    return resolve_preferences(records, restrictions, query, reference, 'test', 'u')


class FoodResolutionTests(unittest.TestCase):
    def test_eight_day_meal_combinations(self):
        for day, query_day in [('WEEKDAY', 'lunes'), ('WEEKEND', 'sábado')]:
            for meal, name in [('BREAKFAST', 'desayuno'), ('LUNCH', 'almuerzo'),
                               ('DINNER', 'cena'), ('SNACK', 'merienda')]:
                with self.subTest(day=day, meal=meal):
                    selected = pref(day + meal, day_scope=day, meal_scope=meal)
                    other = pref('other', 'pizza', day_scope='WEEKEND' if day == 'WEEKDAY' else 'WEEKDAY', meal_scope=meal)
                    result = resolve([selected, other], f'¿Qué prefiero el {query_day} en el {name}?')
                    self.assertEqual(result['selected_action'], 'ANSWER')
                    self.assertEqual(result['selected_food_items'], ['avena'])
                    self.assertEqual(result['applicable_preference_ids'], [selected.preference_id])
                    self.assertEqual(result['retrieved_preference_ids'], sorted([selected.preference_id, 'other']))

    def test_unknown_is_not_any(self):
        result = resolve([pref(day_scope=None)])
        self.assertEqual(result['selected_action'], 'CLARIFY')
        self.assertIn('día', result['clarification_question'])

    def test_missing_meal_asks_specific_question(self):
        result = resolve([pref(meal_scope='LUNCH')], '¿Qué prefiero el lunes?')
        self.assertEqual(result['selected_action'], 'CLARIFY')
        self.assertIn('comida', result['clarification_question'])

    def test_day_switch_and_meal_switch(self):
        records = [pref('a', 'avena', day_scope='WEEKDAY', meal_scope='BREAKFAST'),
                   pref('b', 'pizza', day_scope='WEEKEND', meal_scope='BREAKFAST'),
                   pref('c', 'arroz', day_scope='WEEKDAY', meal_scope='LUNCH')]
        self.assertEqual(resolve(records, '¿Qué desayuno prefiero el lunes?')['selected_food_items'], ['avena'])
        self.assertEqual(resolve(records, '¿Qué desayuno prefiero el domingo?')['selected_food_items'], ['pizza'])
        self.assertEqual(resolve(records, '¿Qué almuerzo prefiero el lunes?')['selected_food_items'], ['arroz'])

    def test_same_food_opposite_polarity_preserves_conflict(self):
        result = resolve([pref('a'), pref('b', polarity='DISLIKE')])
        self.assertEqual(result['selected_action'], 'PRESERVE_CONFLICT')
        self.assertEqual(result['detected_conflict_type'], 'SAME_SCOPE')
        self.assertEqual(result['evidence_ids'], ['a', 'b'])
        self.assertEqual(result['selected_food_items'], [])

    def test_distinct_positive_foods_are_not_a_conflict(self):
        result = resolve([pref('a'), pref('b', 'arroz')])
        self.assertEqual(result['selected_action'], 'ANSWER')
        self.assertEqual(result['selected_food_items'], ['arroz', 'avena'])

    def test_observation_does_not_override_statement(self):
        records = [pref('a', polarity='DISLIKE'), pref('b', source='ORDER_HISTORY')]
        result = resolve(records)
        self.assertEqual(result['selected_action'], 'VERIFY')
        self.assertEqual(result['detected_conflict_type'], 'SCC')
        self.assertEqual(len(result['retrieved_preference_ids']), 2)

    def test_order_alone_requires_verification(self):
        self.assertEqual(resolve([pref(source='ORDER_HISTORY')])['selected_action'], 'VERIFY')

    def test_negative_is_not_positive(self):
        result = resolve([pref(polarity='DISLIKE')])
        self.assertEqual(result['selected_food_items'], [])
        self.assertEqual(result['evidence_ids'], ['a'])

    def test_explicit_update_keeps_history(self):
        old = pref()
        new = replace(pref('b', polarity='DISLIKE', explicit_update=True), timestamp='2026-02-01T09:00:00', valid_from='2026-02-01')
        result = resolve([old, new])
        self.assertEqual(result['selected_action'], 'ANSWER')
        self.assertEqual(result['applicable_preference_ids'], ['b'])
        self.assertEqual(result['retrieved_preference_ids'], ['a', 'b'])
        self.assertEqual(resolve([old, new], reference='2026-01-20')['selected_food_items'], ['avena'])

    def test_future_effective_update_does_not_supersede_early(self):
        new = replace(pref('b', polarity='DISLIKE', explicit_update=True,
                           valid_from='2026-11-01'), timestamp='2026-10-01')
        self.assertEqual(resolve([pref(), new], reference='2026-10-19')['selected_food_items'], ['avena'])
        self.assertEqual(resolve([pref(), new], reference='2026-11-02')['applicable_preference_ids'], ['b'])

    def test_explicit_supersedes_accepts_original_evidence_id(self):
        old = replace(pref('a'), evidence_id='original-message')
        new = replace(pref('b', food='fruta', explicit_update=True,
                           supersedes=('original-message',), valid_from='2026-11-01'), timestamp='2026-10-01')
        self.assertEqual(resolve([old, new], reference='2026-10-19')['selected_food_items'], ['avena'])
        self.assertEqual(resolve([old, new], reference='2026-11-02')['selected_food_items'], ['fruta'])
        self.assertEqual(resolve([old, new], reference='2026-11-02')['retrieved_preference_ids'], ['a', 'b'])

    def test_order_cannot_supersede_user_statement(self):
        observed = replace(pref('b', polarity='DISLIKE', source='ORDER_HISTORY',
                                supersedes=('a',)), timestamp='2026-02-01')
        result = resolve([pref(), observed])
        self.assertEqual(result['detected_conflict_type'], 'SCC')
        self.assertEqual(result['applicable_preference_ids'], ['a', 'b'])

    def test_partition_is_cpc_only_when_query_is_underdetermined(self):
        records = [pref('a', day_scope='WEEKDAY'), pref('b', 'pizza', day_scope='WEEKEND')]
        self.assertIsNone(resolve(records, '¿Qué prefiero el lunes?')['detected_conflict_type'])
        self.assertEqual(resolve(records, '¿Qué prefiero?')['detected_conflict_type'], 'CPC')

    def test_recency_without_explicit_update_cannot_resolve(self):
        newer = replace(pref('b', polarity='DISLIKE'), timestamp='2026-02-01')
        self.assertEqual(resolve([pref(), newer])['selected_action'], 'PRESERVE_CONFLICT')

    def test_boc_keeps_uncertainty(self):
        records = [pref(), replace(pref('b', polarity='DISLIKE'), timestamp='2026-02-01'),
                   replace(pref('c'), timestamp='2026-03-01')]
        result = resolve(records)
        self.assertEqual(result['detected_conflict_type'], 'BOC')
        self.assertEqual(result['selected_food_items'], [])

    def test_temporal_exception_does_not_change_habit(self):
        exception = replace(pref('b', polarity='DISLIKE', valid_from='2026-10-19', valid_to='2026-10-19', temporary=True), timestamp='2026-10-19')
        records = [pref(), exception]
        self.assertEqual(resolve(records, '¿Qué prefiero hoy?', reference='2026-10-19')['selected_food_items'], [])
        self.assertEqual(resolve(records, '¿Qué prefiero hoy?', reference='2026-10-20')['selected_food_items'], ['avena'])

    def test_partial_specificity_no_arbitrary_priority(self):
        records = [pref('a', day_scope='WEEKDAY'),
                   pref('b', polarity='DISLIKE', meal_scope='BREAKFAST')]
        result = resolve(records, '¿Qué desayuno prefiero el lunes?')
        self.assertEqual(result['selected_action'], 'PRESERVE_CONFLICT')

    def test_more_specific_scope_can_refine_general(self):
        records = [pref(), pref('b', polarity='DISLIKE', day_scope='WEEKEND', meal_scope='BREAKFAST')]
        self.assertEqual(resolve(records, '¿Qué desayuno prefiero el sábado?')['selected_food_items'], [])
        self.assertEqual(resolve(records, '¿Qué desayuno prefiero el lunes?')['selected_food_items'], ['avena'])

    def test_confirmed_restrictions_block_only_incompatible_items(self):
        restriction = Restriction('r', 'u', 'nueces', 'ALLERGY', day_scope='ANY', meal_scope='ANY', evidence_id='r')
        result = resolve([pref('a', 'nueces'), pref('b', 'fruta')], restrictions=[restriction])
        self.assertEqual(result['selected_food_items'], ['fruta'])
        self.assertIn('r', result['evidence_ids'])

    def test_uncertain_or_unscoped_restriction_never_assumes_safety(self):
        for restriction in [Restriction('r', 'u', 'avena', 'ALLERGY', certainty='UNCERTAIN', day_scope='ANY', meal_scope='ANY', evidence_id='r'),
                            Restriction('r', 'u', 'avena', 'ALLERGY', evidence_id='r')]:
            with self.subTest(restriction=restriction):
                result = resolve([pref()], restrictions=[restriction])
                self.assertEqual(result['selected_action'], 'VERIFY')
                self.assertEqual(result['selected_food_items'], [])

    def test_reordering_independent_records_preserves_semantics(self):
        records = [pref('a'), pref('b', 'fruta')]
        left, right = resolve(records), resolve(records[::-1])
        for key in left.keys() - {'execution_time_ms'}:
            self.assertEqual(left[key], right[key])

    def test_other_user_and_future_evidence_are_not_used(self):
        records = [pref(), replace(pref('b', 'pizza'), user_id='other'),
                   replace(pref('c', 'sopa'), timestamp='2027-01-01')]
        self.assertEqual(resolve(records)['retrieved_preference_ids'], ['a'])

    def test_pipeline_sqlite_reopen_and_irrelevant_message(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'memory.sqlite'
            store = Store(path, 'u')
            engine = FoodPreferenceEngine(store)
            event = {'message_id': 'e1', 'at': '2026-01-01T09:00:00', 'session_id': 's1',
                     'role': 'user', 'channel': 'chat', 'source': 'conversation',
                     'text': 'De lunes a viernes, para el almuerzo prefiero arroz.'}
            engine.ingest([event])
            initial = engine.decide('s', 'u', '¿Qué almuerzo prefiero el martes?', '2026-10-19')
            self.assertEqual(initial['selected_food_items'], ['arroz'])
            engine.ingest([{**event, 'message_id': 'e2', 'text': 'Hoy leí un libro.'}])
            later = engine.decide('s', 'u', '¿Qué almuerzo prefiero el martes?', '2026-10-19')
            self.assertEqual(initial['selected_food_items'], later['selected_food_items'])
            store.close()
            reopened = Store(path, 'u')
            self.assertEqual(FoodPreferenceEngine(reopened).decide('s', 'u', '¿Qué almuerzo prefiero el martes?', '2026-10-19')['selected_food_items'], ['arroz'])
            reopened.close()

    def test_future_intraday_evidence_is_unavailable(self):
        future = replace(pref(), timestamp='2026-10-19T18:00:00+00:00')
        result = resolve([future], reference='2026-10-19T12:00:00+00:00')
        self.assertEqual(result['selected_action'], 'ABSTAIN')
        self.assertEqual(result['retrieved_preference_ids'], [])

    def test_offset_equivalence_and_naive_utc_visibility(self):
        record = replace(pref(), timestamp='2026-10-19T18:00:00+05:00')
        self.assertEqual(resolve([record], reference='2026-10-19T13:00:00Z')['selected_food_items'], ['avena'])
        self.assertEqual(resolve([record], reference='2026-10-19T12:59:59Z')['selected_action'], 'ABSTAIN')
        self.assertEqual(resolve([record], reference='2026-10-19T13:00:00')['selected_food_items'], ['avena'])
        self.assertEqual(resolve([record], reference='2026-10-19')['selected_food_items'], ['avena'])

    def test_foreign_query_owner_rejected(self):
        with TemporaryDirectory() as directory:
            store = Store(Path(directory) / 'memory.sqlite', 'u')
            with self.assertRaises(ValueError):
                FoodPreferenceEngine(store).decide('s', 'foreign', '¿Qué prefiero?')
            store.close()

    def test_invalid_later_event_does_not_partially_ingest_batch(self):
        with TemporaryDirectory() as directory:
            store = Store(Path(directory) / 'memory.sqlite', 'u')
            valid = {'message_id': 'e1', 'at': '2026-01-01T09:00:00', 'session_id': 's1',
                     'role': 'user', 'channel': 'chat', 'source': 'USER_STATEMENT',
                     'text': 'En general prefiero arroz.'}
            with self.assertRaises(ValueError):
                FoodPreferenceEngine(store).ingest([valid, {**valid, 'message_id': 'e2', 'gold': {}}])
            self.assertEqual(store.all(), [])
            self.assertEqual(store.preferences(), [])
            store.close()

    def test_ingestion_private_metadata_rejected_before_mutation(self):
        with TemporaryDirectory() as directory:
            store = Store(Path(directory) / 'memory.sqlite', 'u')
            with self.assertRaises(ValueError):
                FoodPreferenceEngine(store).ingest([{'message_id': 'e', 'hidden_state': {'food': 'pizza'}}])
            self.assertEqual(store.all(), [])
            store.close()
