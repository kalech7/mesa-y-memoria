import unittest
from tests.test_temporal_memory import StoreFixture, fact, ANY
from rl_longitudinal.clarification_policy import decide


class ClarificationPolicyTests(StoreFixture, unittest.TestCase):
    def decide(self,**kw):
        return decide(self.store,'2026-10-01',dict(day='WEEKDAY',meal='LUNCH',company='alone',place='home'),**kw)
    def test_all_decisions_consult_memory(self):
        self.decide(); self.store.add_fact(fact()); self.decide()
        traces=self.store.retrieval_traces()
        self.assertEqual(len(traces),2)
        self.assertTrue(traces[0]['empty'])
        self.assertEqual(traces[1]['retrieved_ids'],['m1'])
        self.assertEqual(traces[1]['action'],'ANSWER')
    def test_relevant_ambiguity_triggers_clarify(self):
        self.store.add_fact(fact(kind='AMBIGUOUS',uncertainty='UNKNOWN'))
        self.assertEqual(self.decide().action,'CLARIFY')
    def test_relevant_conflict_triggers_clarify(self):
        self.store.add_fact(fact()); self.store.add_fact(fact('m2',polarity='DISLIKE'))
        answer=self.decide()
        self.assertEqual(answer.action,'CLARIFY'); self.assertIn('pizza',answer.question)
    def test_irrelevant_conflict_does_not_force_clarify(self):
        self.store.add_fact(fact()); self.store.add_fact(fact('m2',polarity='DISLIKE')); self.store.add_fact(fact('m3',food='arroz'))
        self.assertEqual(self.decide().foods,('arroz',))
    def test_clarification_answer_is_persisted(self):
        self.store.add_fact(fact()); self.store.add_fact(fact('m2',polarity='DISLIKE'))
        new=fact('m3',at='2026-09-01T00:00:00Z',kind='CLARIFICATION',supersedes=('m1','m2'))
        self.store.record_clarification('current:pizza','2026-09-01T00:00:00Z','Ahora sí me gusta',status='RESOLVED',resolved_fact=new)
        self.assertEqual(self.decide().action,'ANSWER')
        self.assertEqual(len(self.store.facts()),3)
    def test_unresolved_conflict_preserves_evidence(self):
        self.store.add_fact(fact()); self.store.add_fact(fact('m2',polarity='DISLIKE'))
        self.store.record_clarification('current:pizza','2026-09-01',response='No estoy seguro',status='UNRESOLVED')
        self.assertEqual(self.decide().action,'PRESERVE_CONFLICT')
        self.assertEqual(len(self.store.facts()),2)
    def test_previous_clarification_is_not_repeated(self):
        self.store.add_fact(fact()); self.store.add_fact(fact('m2',polarity='DISLIKE'))
        self.store.record_clarification('current:pizza','2026-09-01',response=None,status='REFUSED')
        self.assertNotEqual(self.decide().action,'CLARIFY')
    def test_unavailable_user_preserves_conflict(self):
        self.store.add_fact(fact()); self.store.add_fact(fact('m2',polarity='DISLIKE'))
        self.assertEqual(self.decide(user_available=False).action,'PRESERVE_CONFLICT')
    def test_no_unnecessary_question_after_explicit_update(self):
        self.store.add_fact(fact()); self.store.add_fact(fact('m2',polarity='DISLIKE',at='2026-05-20T00:00:00Z',kind='EXPLICIT_CHANGE',supersedes=('m1',)),operation='UPDATE_WITH_HISTORY')
        self.store.add_fact(fact('m3',food='ramen',at='2026-05-20T00:00:00Z'))
        self.assertEqual(self.decide().foods,('ramen',))
    def test_essential_context_omission_asks_specific_question(self):
        self.store.add_fact(fact(scope={**ANY,'day':'WEEKDAY'}))
        result=decide(self.store,'2026-10-01',{'meal':'LUNCH','company':'alone','place':'home'})
        self.assertEqual((result.action,result.question_key),('CLARIFY','day'))
    def test_restriction_protection_while_asking(self):
        self.store.add_fact(fact(food='maní',polarity='FORBID',kind='RESTRICTION'))
        self.store.add_fact(fact('m2',food='maní',at='2026-09-01T00:00:00Z'))
        result=self.decide(target_food='maní')
        self.assertEqual(result.action,'CLARIFY'); self.assertFalse(result.foods)
    def test_context_question_can_be_asked_on_different_occasion(self):
        self.store.add_fact(fact(scope={**ANY,'day':'WEEKDAY'}))
        self.store.record_clarification('day','2026-09-01','laborable',status='RESOLVED')
        result=decide(self.store,'2026-10-01',{})
        self.assertEqual(result.action,'CLARIFY')
    def test_new_relevant_evidence_may_reopen_old_question(self):
        self.store.add_fact(fact())
        self.store.record_clarification('current:pizza','2026-05-01','No sé',status='UNRESOLVED')
        self.store.add_fact(fact('m2',polarity='DISLIKE',at='2026-09-01T00:00:00Z'))
        self.assertEqual(self.decide().action,'CLARIFY')
    def test_behavior_without_declaration_requests_taste_not_assumes_it(self):
        self.store.add_fact(fact(kind='OBSERVED_BEHAVIOR',source='ORDER_HISTORY'))
        result=self.decide(target_food='pizza')
        self.assertEqual(result.action,'CLARIFY')
        self.assertEqual(result.question_key,'current:pizza')
    def test_unknown_preference_scope_does_not_reask_known_context(self):
        self.store.add_fact(fact(scope={**ANY,'company':None}))
        result=self.decide(target_food='pizza')
        self.assertEqual(result.action,'CLARIFY')
        self.assertEqual(result.question_key,'current:pizza')
    def test_answered_context_is_retrieved_not_merely_skipped(self):
        self.store.add_fact(fact(scope={**ANY,'day':'WEEKDAY'}))
        self.store.record_clarification('day','2026-10-01','Un día laboral',status='RESOLVED',value='WEEKDAY',query_at='2026-10-01')
        result=decide(self.store,'2026-10-01',{})
        self.assertEqual(result.action,'ANSWER')
        self.assertEqual(result.foods,('pizza',))
    def test_contradictory_context_answers_do_not_use_last_write(self):
        self.store.add_fact(fact(scope={**ANY,'day':'WEEKDAY'}))
        for value in ('WEEKDAY','WEEKEND'):
            self.store.record_clarification('day','2026-10-01',value,status='RESOLVED',value=value,query_at='2026-10-01')
        result=decide(self.store,'2026-10-01',{})
        self.assertNotEqual(result.action,'ANSWER')
    def test_unknown_stored_and_query_scope_never_becomes_universal(self):
        self.store.add_fact(fact(scope={'day':None,'meal':None,'company':None,'place':None}))
        result=decide(self.store,'2026-10-01',{})
        self.assertEqual(result.action,'CLARIFY')
        self.assertEqual(result.question_key,'current:pizza')
        self.assertFalse(result.foods)
