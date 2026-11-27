import unittest
from tests.test_temporal_memory import StoreFixture, fact
from rl_longitudinal.temporal_memory import TemporalMemoryStore
from rl_longitudinal.clarification_policy import decide


class MemoryPersistenceTests(StoreFixture, unittest.TestCase):
    def test_multi_session_memory_persistence(self):
        self.store.add_fact(fact(food='sushi'))
        self.store.close(); self.store=TemporalMemoryStore(self.path,'u')
        result=decide(self.store,'2026-10-01',{})
        self.assertEqual(result.foods,('sushi',))
        self.assertIn('m1',result.retrieved_ids)
    def test_future_session_reflects_memory_decisions(self):
        self.store.record_noop('2026-01-01','Did not save communicated fact')
        self.store.close(); self.store=TemporalMemoryStore(self.path,'u')
        self.assertEqual(decide(self.store,'2026-02-01',{}).action,'ABSTAIN')
        self.store.add_fact(fact(food='ramen'))
        self.store.close(); self.store=TemporalMemoryStore(self.path,'u')
        self.assertEqual(decide(self.store,'2026-03-01',{}).foods,('ramen',))
    def test_previous_question_refusal_survives_new_session(self):
        self.store.add_fact(fact());self.store.add_fact(fact('m2',polarity='DISLIKE'))
        self.store.record_clarification('current:pizza','2026-06-01','No quiero responder',status='REFUSED')
        self.store.close();self.store=TemporalMemoryStore(self.path,'u')
        self.assertEqual(decide(self.store,'2026-10-01',{}).action,'PRESERVE_CONFLICT')
    def test_old_legacy_tables_are_preserved(self):
        tables={r[0] for r in self.store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({'memories','operations','food_preferences','temporal_facts'} <= tables)
    def test_reopen_other_user_is_rejected(self):
        with self.assertRaises(ValueError):TemporalMemoryStore(self.path,'other')
