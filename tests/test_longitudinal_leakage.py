from copy import deepcopy
import json
import unittest

from rl_longitudinal.longitudinal_environment import LongitudinalEnv
from rl_longitudinal.scenario_generator import generate_episodes


class LongitudinalLeakageTests(unittest.TestCase):
    def test_no_hidden_truth_leakage(self):
        s = generate_episodes(1, split='test')[0]
        env = LongitudinalEnv(s); o, info = env.reset()
        def walk(v):
            if isinstance(v, dict):
                for k, x in v.items():
                    self.assertNotIn(k, ('gold','family','seed','split','episode_id','latent','responses','acceptable_conversations'))
                    walk(x)
            elif isinstance(v, (list, tuple)):
                for x in v: walk(x)
        walk(o)
        self.assertEqual(info,{})
        self.assertNotIn(s['user_id'], json.dumps(o))
        env.close()

    def test_future_events_are_not_observed(self):
        s = generate_episodes(1, split='test')[0]
        env = LongitudinalEnv(s); o, _ = env.reset()
        self.assertNotIn('e001', json.dumps(o))
        env.close()

    def test_gold_mutation_does_not_change_observation(self):
        s = generate_episodes(1, split='test')[0]; changed = deepcopy(s)
        changed['sessions'][0]['gold']['foods'] = ['ramen']
        a, b = LongitudinalEnv(s), LongitudinalEnv(changed)
        oa, _ = a.reset(); ob, _ = b.reset()
        self.assertEqual(oa, ob)
        a.close(); b.close()

    def test_future_reply_mutation_does_not_change_observation(self):
        s = generate_episodes(1, split='test', families=['SOURCE_CHANGE'])[0]
        changed = deepcopy(s)
        changed['sessions'][1]['responses'] = {}
        a, b = LongitudinalEnv(s), LongitudinalEnv(changed)
        oa, _ = a.reset(); ob, _ = b.reset()
        self.assertEqual(oa, ob)
        a.close(); b.close()

    def test_observation_copy_cannot_mutate_environment(self):
        env = LongitudinalEnv(generate_episodes(1)[0]); o, _ = env.reset()
        o['incoming_evidence'][0]['food']='maní'
        self.assertEqual(env._observe()['incoming_evidence'][0]['food'],'ensalada')
        env.close()

    def test_future_fact_rejected(self):
        s = generate_episodes(1)[0]
        s['sessions'][0]['incoming'][0]['observed_at']='2099-01-01T00:00:00Z'
        with self.assertRaises(ValueError): LongitudinalEnv(s).reset()


if __name__ == '__main__':
    unittest.main()
