import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from rl_longitudinal.temporal_memory import TemporalFact, TemporalMemoryStore, instant
from rl_longitudinal.memory_actions import apply_memory_action

ANY = dict(day='ANY', meal='ANY', company='ANY', place='ANY')


def fact(mid='m1', food='pizza', polarity='LIKE', at='2026-01-01T12:00:00Z', **kw):
    values = dict(memory_id=mid,user_id='u',food=food,polarity=polarity,scope=dict(ANY),observed_at=at,recorded_at=at,
                  valid_from=at,evidence_id='e-'+mid)
    values.update(kw)
    return TemporalFact(**values)


class StoreFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.path=Path(self.temp.name)/'memory.sqlite'
        self.store=TemporalMemoryStore(self.path,'u')
    def tearDown(self):
        self.store.close(); self.temp.cleanup()


class TemporalMemoryTests(StoreFixture, unittest.TestCase):
    def test_utc_offsets_are_normalized(self):
        f=fact(at='2026-01-01T07:00:00-05:00')
        self.assertEqual(f.observed_at,'2026-01-01T12:00:00Z')
    def test_naive_timestamp_is_rejected(self):
        with self.assertRaises(ValueError): fact(at='2026-01-01T12:00:00')
    def test_recorded_at_not_earlier_than_observation(self):
        with self.assertRaises(ValueError): fact(recorded_at='2025-01-01T00:00:00Z')
    def test_unknown_scope_not_any(self):
        self.assertIsNone(fact(scope={'day':None}).scope['day'])
        self.assertEqual(fact().scope['day'],'ANY')
    def test_temporal_exception_requires_end(self):
        with self.assertRaises(ValueError): fact(kind='TEMPORARY_EXCEPTION')
    def test_immutable_idempotent_insert(self):
        f=fact(); self.store.add_fact(f); self.store.add_fact(f)
        self.assertEqual(len(self.store.facts()),1)
        with self.assertRaises(ValueError): self.store.add_fact(replace(f,polarity='DISLIKE'))
    def test_old_relevant_memory_is_retrieved(self):
        self.store.add_fact(fact()); self.store.add_fact(fact('m2',food='sopa',at='2026-09-01T00:00:00Z'))
        got=self.store.retrieve('2026-10-01T00:00:00Z')
        self.assertEqual({f.memory_id for f in got.facts},{'m1','m2'})
    def test_recording_today_does_not_start_validity_today(self):
        self.store.add_fact(fact(recorded_at='2026-09-01T00:00:00Z'))
        self.assertEqual(len(self.store.retrieve('2026-03-01',known_at='2026-10-01').facts),1)
        self.assertEqual(len(self.store.retrieve('2026-03-01',known_at='2026-08-01').facts),0)
    def test_future_evidence_is_not_exposed(self):
        self.store.add_fact(fact(at='2026-12-01T00:00:00Z'))
        self.assertFalse(self.store.retrieve('2026-01-02').facts)
    def test_memory_owner_is_enforced(self):
        with self.assertRaises(ValueError): self.store.add_fact(fact(user_id='other'))
    def test_retrieval_records_empty_result(self):
        result=self.store.retrieve('2026-02-01')
        self.assertEqual(result.facts,())
        self.assertTrue(self.store.retrieval_traces()[-1]['empty'])
    def test_explicit_update_links_old_record_without_deleting(self):
        self.store.add_fact(fact())
        self.store.add_fact(fact('m2',polarity='DISLIKE',at='2026-09-01T00:00:00Z',kind='EXPLICIT_CHANGE',supersedes=('m1',)),operation='UPDATE_WITH_HISTORY')
        self.assertEqual(len(self.store.facts()),2)
        op=self.store.operations()[-1]
        self.assertEqual(op['before'][0]['memory_id'],'m1')
        self.assertEqual(op['after']['memory_id'],'m2')
    def test_update_cannot_retire_other_food_or_context(self):
        self.store.add_fact(fact())
        with self.assertRaises(ValueError): self.store.add_fact(fact('m2',food='ramen',kind='EXPLICIT_CHANGE',supersedes=('m1',)),operation='UPDATE_WITH_HISTORY')
        with self.assertRaises(ValueError): self.store.add_fact(fact('m2',scope={**ANY,'day':'WEEKEND'},kind='EXPLICIT_CHANGE',supersedes=('m1',)),operation='UPDATE_WITH_HISTORY')
    def test_ambiguous_behavior_cannot_update_permanent_preference(self):
        with self.assertRaises(ValueError): apply_memory_action(self.store,'UPDATE_WITH_HISTORY',fact(kind='OBSERVED_BEHAVIOR',source='ORDER_HISTORY'))
    def test_delete_is_not_a_preference_resolution_action(self):
        with self.assertRaises(ValueError): apply_memory_action(self.store,'DELETE',fact())
    def test_mutating_input_scope_does_not_modify_persisted_fact(self):
        f=fact(); self.store.add_fact(f); f.scope['day']='WEEKEND'
        self.assertEqual(self.store.facts()[0].scope['day'],'ANY')
    def test_update_cannot_close_before_known_onset(self):
        self.store.add_fact(fact(at='2026-06-01T00:00:00Z'))
        with self.assertRaises(ValueError):
            self.store.add_fact(fact('m2',kind='EXPLICIT_CHANGE',supersedes=('m1',),at='2026-09-01T00:00:00Z',valid_from='2026-01-01'),operation='UPDATE_WITH_HISTORY')
    def test_supersession_is_not_possible_via_add_or_preserve(self):
        self.store.add_fact(fact())
        update=fact('m2',kind='EXPLICIT_CHANGE',supersedes=('m1',),at='2026-09-01T00:00:00Z')
        for operation in ('ADD','PRESERVE_CONFLICT'):
            with self.assertRaises(ValueError):self.store.add_fact(update,operation=operation)
        self.assertEqual(len(self.store.facts()),1)
    def test_order_history_cannot_masquerade_as_explicit_update(self):
        with self.assertRaises(ValueError):
            fact(kind='EXPLICIT_CHANGE',source='ORDER_HISTORY',supersedes=('old',))
    def test_memory_actions_consult_and_evaluate_history_before_update(self):
        self.store.add_fact(fact())
        apply_memory_action(self.store,'NOOP',at='2026-10-01',reason='Insufficient new evidence')
        trace=self.store.retrieval_traces()[-1]
        self.assertEqual(trace['retrieved_ids'],['m1'])
        self.assertEqual(trace['applicable_ids'],['m1'])
        self.assertEqual(trace['memory_operation'],'NOOP')
    def test_missing_timestamps_are_not_fabricated(self):
        with self.assertRaises(ValueError):TemporalFact('m','u','pizza','LIKE',evidence_id='e')
    def test_uncertain_clarification_cannot_commit_permanent_update(self):
        self.store.add_fact(fact())
        update=fact('m2',kind='CLARIFICATION',supersedes=('m1',),at='2026-09-01T00:00:00Z')
        with self.assertRaises(ValueError):
            self.store.record_clarification('current:pizza','2026-09-01','No sé',status='UNRESOLVED',resolved_fact=update)
        self.assertEqual(len(self.store.facts()),1)
