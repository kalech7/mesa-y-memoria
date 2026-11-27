"""Scientific contracts independent of model/dependency availability."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_local_001.artifacts import digest, write_json
from llm_local_001.data import freeze_data
from llm_local_001.metrics import paired, summarize
from llm_local_001.report import blocked_report
from llm_local_001.run import execute, parser


class DatasetContracts(unittest.TestCase):
    def test_disjoint_users_words_contexts_and_explicit_shared_abstract_support(self):
        with tempfile.TemporaryDirectory() as temporary:
            splits, audit = freeze_data(Path(temporary), (40, 20, 20), seed=43)
            for pair, result in audit['overlap'].items():
                self.assertFalse(result['user_overlap'], pair)
                self.assertFalse(result['context_overlap'], pair)
                self.assertFalse(result['preference_text_overlap'], pair)
                self.assertGreater(result['abstract_overlap_count'], 0, pair)
            self.assertEqual(audit['test_frozen_hash'], digest(splits['test']))
            self.assertFalse(audit['independent_human_sample'])
            for scenarios in splits.values():
                self.assertEqual(len({s['family'] for s in scenarios}), 5)
                self.assertTrue(all(len(s['sessions']) == 3 for s in scenarios))


class MetricsContracts(unittest.TestCase):
    def test_bootstrap_aggregates_complete_users_not_seeds_as_independent_people(self):
        def row(uid, ret):
            return {'user_id': uid, 'return': ret, 'future_accuracy': 1,
                    'error_rate': 0, 'steps': [{'terminal': True, 'correct': True}]}
        left = [row('a', 0), row('a', 2), row('b', 2), row('b', 4)]
        result = summarize(left, repetitions=100)
        self.assertEqual(result['users'], 2)
        self.assertEqual(result['metrics']['return']['n'], 2)
        self.assertEqual(result['metrics']['return']['mean'], 2)
        self.assertEqual(paired(left, [row('a', 0), row('b', 0)], 100)['return']['mean'], 2)
        with self.assertRaises(ValueError):
            paired(left, [row('a', 0)], 100)


class BlockedArtifactContracts(unittest.TestCase):
    def test_no_measurements_or_checkpoint_fabricated_when_no_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            _, dataset = freeze_data(out, (40, 20, 20))
            meta = {'created_at': '2026-10-23T00:00:00+00:00',
                    'source': 'unit_test', 'config_hash': 'test'}
            blocked_report(out, {'reasons': ['No causal local weights']}, dataset, meta)
            metrics = json.loads((out / 'training_metrics.json').read_text())
            self.assertIsNone(metrics['optimizer_updates'])
            self.assertEqual(json.loads((out / 'checkpoint_manifest.json').read_text())['checkpoints'], [])
            with (out / 'baseline_comparison.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({r['method'] for r in rows}, {'B0','B1','B2','B3','B4'})
            self.assertTrue(all(r['mean_return'] == '' and r['errors'] == '' for r in rows))
            self.assertTrue(all(r['status'] == 'NOT_RUN_BLOCKED' for r in rows))

    def test_cli_blocked_path_runs_without_loading_llm_or_calling_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'run'
            args = parser().parse_args(['--output', str(out), '--preflight-only', '--train-users', '40'])
            with patch('llm_local_001.run.run_preflight', return_value={
                    'status': 'blocked', 'reasons': ['No trainable local causal model']}):
                self.assertEqual(execute(args), 2)
            status = json.loads((out / 'run_status.json').read_text())
            self.assertFalse(status['test_evaluated'])
            self.assertEqual(status['llm_optimizer_updates'], 0)
            self.assertEqual(status['model_inference_calls'], 0)
            with self.assertRaises(FileExistsError):
                execute(args)


if __name__ == '__main__':
    unittest.main()
