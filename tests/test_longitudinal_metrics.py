import tempfile
import unittest
from pathlib import Path

from rl_longitudinal.audit import audit_partitions, semantic_signature
from rl_longitudinal.baselines import ClarificationRuleBased, HistoricalRuleBased
from rl_longitudinal.evaluate import run_episode, gate_headroom
from rl_longitudinal.metrics import summarize, paired
from rl_longitudinal.scenario_generator import generate_episodes


class LongitudinalMetricTests(unittest.TestCase):
    def test_decisions_and_questions_have_distinct_denominators(self):
        case=generate_episodes(1,seed=918,split='test',families=['SOURCE_CHANGE'])[0]
        row=run_episode(ClarificationRuleBased(),case)
        result=summarize([row],10)
        self.assertEqual(result['decision_accuracy']['value'],1)
        self.assertEqual(result['necessary_clarification_recall']['value'],1)
        self.assertEqual(result['clarifications'],1)
        self.assertEqual(result['unnecessary_clarification_rate']['denominator'],1)
        self.assertGreater(result['historical_relevant_recall']['denominator'],0)

    def test_all_reference_facts_have_independent_temporal_labels(self):
        cases=generate_episodes(10,seed=987,split='test')
        rows=[run_episode(ClarificationRuleBased(),s) for s in cases]
        result=summarize(rows,10)
        self.assertEqual(result['temporal_applicability_accuracy']['value'],1)
        self.assertEqual(result['interpretation_accuracy']['value'],1)

    def test_no_questions_is_not_zero_error_rate(self):
        case=generate_episodes(1,seed=918,split='test',families=['SOURCE_CHANGE'])[0]
        row=run_episode(HistoricalRuleBased(),case)
        result=summarize([row],10)
        self.assertIsNone(result['unnecessary_clarification_rate']['value'])
        self.assertEqual(result['necessary_clarification_recall']['value'],0)

    def test_missing_persistence_reduces_historical_recall(self):
        case=generate_episodes(1,split='test',families=['STABLE_EXCEPTION'])[0]
        row=run_episode(ClarificationRuleBased(),case,no_persistence=True,enforce_invariants=False)
        self.assertLess(summarize([row],10)['historical_relevant_recall']['value'],1)
        self.assertTrue(row['sessions'][0]['correct'])
        self.assertFalse(row['sessions'][2]['correct'])

    def test_paired_comparison_rejects_duplicate_users(self):
        case=generate_episodes(1,split='test',families=['STABLE_EXCEPTION'])[0]
        row=run_episode(ClarificationRuleBased(),case)
        with self.assertRaises(ValueError):
            paired([row,row],[row,row],repetitions=10)

    def test_zero_headroom_does_not_authorize_training(self):
        case=generate_episodes(1,split='validation',families=['STABLE_EXCEPTION'])[0]
        row=run_episode(ClarificationRuleBased(),case)
        gate=gate_headroom({'PLANNING_BASELINE':[row],'SUPERVISED_LONGITUDINAL':[row]},10)
        self.assertEqual(gate['status'],'NO_MEANINGFUL_MARGIN')
        self.assertEqual(gate['paired_ci'],[0,0])

    def test_data_audit_detects_scripts_despite_new_ids_dates(self):
        train=generate_episodes(1,seed=987,split='train',families=['STABLE_EXCEPTION'])
        test=generate_episodes(1,seed=918,split='test',families=['STABLE_EXCEPTION'])
        report=audit_partitions({'train':train,'test':test})
        self.assertEqual(report['overlaps']['train/test']['users'],0)
        self.assertEqual(report['overlaps']['train/test']['native_states'],0)
        self.assertEqual(report['overlaps']['train/test']['abstract_scripts'],1)

    def test_conflict_recall_never_counts_unrelated_query_in_numerator(self):
        case=generate_episodes(1,split='test',families=['RESTRICTION'])[0]
        row=run_episode(ClarificationRuleBased(),case)
        metric=summarize([row],10)['relevant_conflict_recall']
        self.assertEqual(metric,{'numerator':1,'denominator':1,'value':1})

    def test_pipeline_lost_reply_stays_in_interpretation_denominator(self):
        case=generate_episodes(1,split='test',families=['SOURCE_CHANGE'])[0]
        row=run_episode(ClarificationRuleBased(),case)
        # Initial explicit statements + order + the communicated clarification.
        self.assertEqual(sum(s['interpretation_total'] for s in row['sessions']),4)

    def test_causal_control_and_treatment_use_identical_guards(self):
        from rl_longitudinal.validation import validate_environment
        case=generate_episodes(1,split='validation',families=['STABLE_EXCEPTION'])[0]
        with tempfile.TemporaryDirectory() as d:
            result=validate_environment([case],Path(d))
            self.assertTrue(result['all_causal_pairs_have_same_guards'])
            self.assertEqual(result['same_current_quality_but_future_effect'],1)

    def test_environment_accepts_three_and_ten_sessions(self):
        from copy import deepcopy
        from datetime import datetime,timedelta
        case=generate_episodes(1,split='test',families=['STABLE_EXCEPTION'])[0]
        for length in (3,10):
            variant=deepcopy(case)
            variant['sessions']=variant['sessions'][:3]
            while len(variant['sessions'])<length:
                session=deepcopy(variant['sessions'][-1])
                later=datetime.fromisoformat(session['at'].replace('Z','+00:00'))+timedelta(days=30)
                session['at']=session['query_at']=later.isoformat().replace('+00:00','Z')
                variant['sessions'].append(session)
            result=run_episode(ClarificationRuleBased(),variant)
            self.assertEqual(len(result['sessions']),length)
            self.assertTrue(all(s['correct'] for s in result['sessions']))


if __name__=='__main__':
    unittest.main()
