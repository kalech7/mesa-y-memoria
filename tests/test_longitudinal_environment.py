import tempfile
import unittest
from pathlib import Path

from rl_longitudinal.longitudinal_environment import LongitudinalEnv
from rl_longitudinal.scenario_generator import generate_episodes, FAMILIES
from rl_longitudinal.reward import evidence_grounded, independent_applicable


def scenario(family):
    return generate_episodes(1, split='test', families=[family])[0]


def action(observation):
    d = observation['required_decision']
    incoming = observation['incoming_evidence']
    op = ('UPDATE_WITH_HISTORY' if incoming and all(f['kind'] in ('EXPLICIT_CHANGE', 'CLARIFICATION') for f in incoming)
          else 'ADD' if incoming else 'NOOP')
    return dict(conversation=d['action'], food=d['foods'][0] if d['foods'] else None,
                question_key=d['question_key'], memory_operation=op)


def complete(env, observation):
    for _ in range(40):
        if observation.get('terminated'):
            return env.diagnostic_metrics()
        observation, _, _, _, _ = env.step(action(observation))
    raise AssertionError('Environment failed to terminate')


class LongitudinalEnvironmentTests(unittest.TestCase):
    def test_independent_reward_does_not_backdate_unknown_onset(self):
        f=scenario('STABLE_EXCEPTION')['sessions'][0]['incoming'][0]
        f['valid_from']=None
        self.assertFalse(independent_applicable(f,'2000-01-01T00:00:00Z',{}))

    def test_independent_unknown_scope_does_not_ground_missing_context(self):
        f=scenario('STABLE_EXCEPTION')['sessions'][0]['incoming'][0]
        f['scope']['company']=None
        self.assertFalse(independent_applicable(f,f['observed_at'],{}))
        self.assertFalse(evidence_grounded(f['food'],[f],f['observed_at'],{}))

    def test_independent_reward_temporary_same_food_exception(self):
        s=scenario('STABLE_EXCEPTION')
        stable=dict(s['sessions'][0]['incoming'][0]);stable['polarity']='DISLIKE'
        temporary=dict(s['sessions'][1]['incoming'][0]);temporary['food']=stable['food']
        self.assertTrue(evidence_grounded(stable['food'],[stable,temporary],s['sessions'][1]['query_at'],{}))
        self.assertFalse(evidence_grounded(stable['food'],[stable,temporary],s['sessions'][2]['query_at'],{}))

    def test_all_decisions_consult_memory(self):
        env = LongitudinalEnv(scenario('STABLE_EXCEPTION'))
        o, _ = env.reset()
        result = complete(env, o)
        traces = {row['trace_id'] for row in result['retrieval_traces']}
        self.assertTrue(all(step['trace_id'] in traces for step in result['steps']))
        self.assertTrue(result['retrieval_traces'][0]['empty'])
        env.close()

    def test_old_relevant_memory_is_retrieved(self):
        env = LongitudinalEnv(scenario('STABLE_EXCEPTION'))
        o, _ = env.reset(); result = complete(env, o)
        self.assertIn('m000', result['sessions'][2]['history_retrieved_ids'])
        self.assertEqual(result['sessions'][2]['action']['food'], 'ensalada')
        env.close()

    def test_temporary_exception_preserves_stable_preference(self):
        env = LongitudinalEnv(scenario('STABLE_EXCEPTION'))
        o, _ = env.reset(); result = complete(env, o)
        self.assertEqual(len(result['sessions'][-1]['memory_after']), 2)
        self.assertTrue(all(row['correct'] for row in result['sessions']))
        env.close()

    def test_multi_session_memory_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'memory.sqlite'
            env = LongitudinalEnv(scenario('EXPLICIT_CHANGE'), db_path=path)
            o, _ = env.reset(); before = env.store.conn
            o, *_ = env.step(action(o))
            self.assertIsNot(before, env.store.conn)
            self.assertTrue(path.is_file())
            self.assertEqual(o['historical_memory'][0]['food'], 'sushi')
            env.close()

    def test_historical_query_uses_historical_validity(self):
        env = LongitudinalEnv(scenario('EXPLICIT_CHANGE'))
        o, _ = env.reset(); result = complete(env, o)
        self.assertEqual([r['action']['food'] for r in result['sessions']], ['sushi','ramen','ramen','sushi'])
        self.assertTrue(all(r['correct'] for r in result['sessions']))
        env.close()

    def test_relevant_conflict_triggers_clarify(self):
        env = LongitudinalEnv(scenario('SOURCE_CHANGE'))
        o, _ = env.reset(); o, *_ = env.step(action(o))
        self.assertEqual(o['required_decision']['action'], 'CLARIFY')
        o, *_ = env.step(dict(conversation='ANSWER', food=o['target_food'], memory_operation='ADD'))
        self.assertEqual(env._steps[-1]['executed_action']['conversation'], 'CLARIFY')
        env.close()

    def test_clarification_answer_is_persisted(self):
        env = LongitudinalEnv(scenario('SOURCE_CHANGE'))
        o, _ = env.reset(); result = complete(env, o)
        self.assertTrue(any(f['memory_id'].startswith('clar_') for f in result['sessions'][2]['memory_after']))
        self.assertTrue(result['sessions'][2]['correct'])
        env.close()

    def test_unresolved_conflict_preserves_evidence(self):
        env = LongitudinalEnv(scenario('UNANSWERED'))
        o, _ = env.reset(); result = complete(env, o)
        self.assertEqual(len(result['sessions'][-1]['memory_after']), 2)
        self.assertEqual(result['sessions'][-1]['action']['conversation'], 'PRESERVE_CONFLICT')
        env.close()

    def test_uncertainty_after_reply_blocks_definitive_commitment(self):
        env = LongitudinalEnv(scenario('UNANSWERED'))
        o,_=env.reset();o,*_=env.step(action(o));o,*_=env.step(action(o))
        self.assertEqual(o['required_decision']['action'],'PRESERVE_CONFLICT')
        env.step(dict(conversation='ANSWER',food=o['target_food'],memory_operation='NOOP'))
        self.assertEqual(env._steps[-1]['executed_action']['conversation'],'PRESERVE_CONFLICT')
        self.assertIn('UNSUPPORTED_COMMITMENT',env._steps[-1]['shield_reasons'])
        env.close()

    def test_previous_clarification_is_not_repeated(self):
        env = LongitudinalEnv(scenario('UNANSWERED'))
        o, _ = env.reset(); result = complete(env, o)
        questions = [s for s in result['steps'] if s['executed_action']['conversation'] == 'CLARIFY']
        self.assertEqual(len(questions), 1)
        env.close()

    def test_irrelevant_conflict_does_not_force_clarify(self):
        env = LongitudinalEnv(scenario('IRRELEVANT_CONFLICT'))
        o, _ = env.reset(); result = complete(env, o)
        self.assertFalse(any(s['executed_action']['conversation'] == 'CLARIFY' for s in result['steps']))
        self.assertTrue(all(s['correct'] for s in result['sessions']))
        env.close()

    def test_future_session_reflects_memory_decisions(self):
        source = scenario('STABLE_EXCEPTION')
        good = LongitudinalEnv(source)
        o, _ = good.reset(); kept = complete(good, o)
        lost = LongitudinalEnv(source, no_persistence=True, enforce_invariants=False)
        o, _ = lost.reset(); dropped = complete(lost, o)
        self.assertTrue(kept['sessions'][2]['correct'])
        self.assertFalse(dropped['sessions'][2]['correct'])
        self.assertEqual(dropped['archive_evidence_ids'], ['e000','e001'])
        good.close(); lost.close()

    def test_noop_does_not_erase_current_observation(self):
        env = LongitudinalEnv(scenario('STABLE_EXCEPTION'), enforce_invariants=False)
        o, _ = env.reset()
        env.step(dict(conversation='ANSWER', food='ensalada', memory_operation='NOOP'))
        self.assertTrue(env.diagnostic_metrics()['sessions'][0]['correct'])
        self.assertEqual(env.store.facts(), [])
        env.close()

    def test_all_controlled_families(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                env = LongitudinalEnv(scenario(family))
                o, _ = env.reset(); result = complete(env, o)
                self.assertTrue(all(s['correct'] for s in result['sessions']), result['sessions'])
                env.close()

    def test_restrictions_are_never_overridden_by_recency(self):
        env = LongitudinalEnv(scenario('RESTRICTION'))
        o, _ = env.reset(); result = complete(env, o)
        self.assertFalse(any(s['restriction_violation'] for s in result['sessions']))
        self.assertFalse(any(s['action'].get('food') == 'maní' for s in result['sessions']))
        env.close()

    def test_blocked_unsafe_proposal_is_penalized_once(self):
        env = LongitudinalEnv(scenario('RESTRICTION'))
        o,_=env.reset();o,*_=env.step(action(o))
        _,reward,_,_,_=env.step(dict(conversation='ANSWER',food='maní',memory_operation='ADD'))
        step=env._steps[-1]
        self.assertEqual(step['executed_action']['conversation'],'CLARIFY')
        self.assertIsNone(step['executed_action'].get('food'))
        self.assertEqual(step['attempted_safety_penalty'],env.reward_config.unsafe_attempt_penalty)
        self.assertAlmostEqual(reward,-env.reward_config.clarification_cost-env.reward_config.unsafe_attempt_penalty)
        env.close()

    def test_noop_drop_is_shielded_in_conforming_policy(self):
        env = LongitudinalEnv(scenario('STABLE_EXCEPTION'))
        o, _ = env.reset()
        env.step(dict(conversation='ANSWER', food='ensalada', memory_operation='NOOP'))
        self.assertEqual(len(env.store.facts()), 1)
        self.assertIn('NOOP_WOULD_OMIT_NEW_EVIDENCE', env._steps[-1]['memory_violations'])
        env.close()

    def test_explicit_linked_change_wrong_add_is_shielded(self):
        env = LongitudinalEnv(scenario('EXPLICIT_CHANGE'))
        o, _ = env.reset(); o, *_ = env.step(action(o))
        a = action(o); a['memory_operation'] = 'ADD'
        o, *_ = env.step(a)
        self.assertIn('EXPLICIT_CHANGE_REQUIRES_HISTORICAL_UPDATE', env._steps[-1]['memory_violations'])
        self.assertEqual(env._steps[-1]['actual_memory_operations'][0]['executed'], 'UPDATE_WITH_HISTORY')
        result = complete(env, o)
        self.assertTrue(all(s['correct'] for s in result['sessions']))
        self.assertLess(result['sessions'][1]['proposed_correct_memory_operations'], result['sessions'][1]['correct_memory_operations'])
        env.close()

    def test_dataset_family_holdout(self):
        train = generate_episodes(80, split='train')
        shift = generate_episodes(20, split='shift')
        self.assertFalse({s['family'] for s in train} & {s['family'] for s in shift})
        self.assertEqual(train, generate_episodes(80, split='train'))


if __name__ == '__main__':
    unittest.main()
