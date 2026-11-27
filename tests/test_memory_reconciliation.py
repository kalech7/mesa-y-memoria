import unittest
from dataclasses import replace
from rl_longitudinal.memory_reconciliation import reconcile
from tests.test_temporal_memory import fact, ANY


class MemoryReconciliationTests(unittest.TestCase):
    def result(self, facts, at='2026-10-01', context=None,target=None):
        return reconcile(facts,context=context or dict(day='WEEKDAY',meal='LUNCH',company='alone',place='home'),query_at=at,target_food=target)
    def test_last_write_does_not_override_context(self):
        old=fact(scope={**ANY,'day':'WEEKDAY'},food='ensalada')
        recent=fact('m2',food='pizza',at='2026-09-01T00:00:00Z',scope={**ANY,'day':'WEEKEND'})
        self.assertEqual(self.result([old,recent]).safe_foods,('ensalada',))
    def test_temporary_exception_preserves_stable_preference(self):
        stable=fact(food='vegetariana')
        exception=fact('m2',food='pollo',at='2026-06-10T00:00:00Z',kind='TEMPORARY_EXCEPTION',valid_to='2026-06-11')
        self.assertEqual(self.result([stable,exception],at='2026-06-20').safe_foods,('vegetariana',))
        self.assertEqual(exception.valid_to,'2026-06-11T00:00:00Z')
    def test_explicit_change_updates_current_validity(self):
        old=fact(food='sushi')
        change=fact('m2',food='sushi',polarity='DISLIKE',at='2026-05-20T00:00:00Z',kind='EXPLICIT_CHANGE',supersedes=('m1',))
        ramen=fact('m3',food='ramen',at='2026-05-20T00:00:00Z',kind='EXPLICIT_CHANGE')
        result=self.result([old,change,ramen])
        self.assertEqual(result.safe_foods,('ramen',))
        self.assertEqual(result.effective_intervals['m1']['valid_to'],'2026-05-20T00:00:00Z')
        self.assertIsNone(old.valid_to)
    def test_historical_query_uses_historical_validity(self):
        old=fact(food='sushi')
        change=fact('m2',food='sushi',polarity='DISLIKE',at='2026-05-20T00:00:00Z',kind='EXPLICIT_CHANGE',supersedes=('m1',))
        self.assertEqual(self.result([change,old],at='2026-03-01').safe_foods,('sushi',))
    def test_sources_do_not_establish_taste_by_recency(self):
        declared=fact(polarity='DISLIKE')
        order=fact('m2',at='2026-08-01T00:00:00Z',kind='OBSERVED_BEHAVIOR',source='ORDER_HISTORY')
        result=self.result([declared,order],target='pizza')
        self.assertEqual(result.safe_foods,())
        self.assertEqual(result.conflict_types,('SCC',))
    def test_behavior_alone_does_not_prove_liking(self):
        self.assertEqual(self.result([fact(kind='OBSERVED_BEHAVIOR',source='ORDER_HISTORY')]).safe_foods,())
    def test_unknown_scope_does_not_become_any(self):
        result=self.result([fact(scope={'day':None})])
        self.assertFalse(result.safe_foods)
        self.assertIn('pizza',result.uncertain_foods)
        self.assertNotIn('day',result.missing_slots)
    def test_physical_order_does_not_change_decision(self):
        values=[fact(),fact('m2',polarity='DISLIKE',at='2026-09-01T00:00:00Z')]
        self.assertEqual(self.result(values),self.result(list(reversed(values))))
    def test_irrelevant_recent_other_context_does_not_modify_decision(self):
        old=fact(scope={**ANY,'day':'WEEKDAY'})
        unrelated=fact('m2',at='2026-09-01T00:00:00Z',polarity='DISLIKE',scope={**ANY,'day':'WEEKEND'})
        self.assertEqual(self.result([old]).safe_foods,self.result([old,unrelated]).safe_foods)
    def test_older_relevant_evidence_changes_material_conflict(self):
        recent=fact('m2',at='2026-09-01T00:00:00Z')
        old=fact(polarity='DISLIKE')
        self.assertTrue(self.result([recent]).safe_foods)
        self.assertFalse(self.result([recent,old]).safe_foods)
    def test_restrictions_are_never_overridden_by_recency(self):
        allergy=fact(food='maní',polarity='FORBID',kind='RESTRICTION')
        desire=fact('m2',food='maní',at='2026-09-01T00:00:00Z')
        self.assertFalse(self.result([allergy,desire],target='maní').safe_foods)
    def test_safe_alternative_survives_irrelevant_conflict(self):
        values=[fact(),fact('m2',polarity='DISLIKE'),fact('m3',food='arroz')]
        self.assertEqual(self.result(values).safe_foods,('arroz',))
    def test_query_date_changes_active_exception(self):
        exception=fact(kind='TEMPORARY_EXCEPTION',valid_to='2026-01-02')
        self.assertEqual(self.result([exception],at='2026-01-01T23:59:59Z').safe_foods,('pizza',))
        self.assertFalse(self.result([exception],at='2026-01-02').safe_foods)
    def test_unknown_onset_does_not_license_prehistory(self):
        recent=fact(at='2026-09-01T00:00:00Z',valid_from=None)
        self.assertFalse(self.result([recent],at='2026-03-01').safe_foods)
