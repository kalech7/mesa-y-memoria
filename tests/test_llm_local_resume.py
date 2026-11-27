"""Completed-stage reuse must preserve weights, budgets, and B1 ancestry."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from llm_local_001.experiment import clone_completed_training
from llm_local_001.model import artifact_hashes
from llm_local_001.training import TrainingConfig, TrainingResult


class CompletedStageReuseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.previous, self.output = self.root / 'old', self.root / 'new'
        self.output.mkdir()
        (self.output / 'stage_reuse_manifest.json').write_text('{}')
        self.config = TrainingConfig(updates=1, batch_size=2, seed=101)
        for name in ('model.py', 'training.py'):
            source = Path('llm_local_001') / name
            destination = self.previous / 'source_snapshot' / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    def stage(self, method):
        directory = self.previous / 'checkpoints/101' / method
        checkpoints = {}
        parameters = {'total': 10, 'trainable': 2, 'training_scope': 'lora_adapters_integrated_in_llm'}
        origin = None if method == 'B1' else str((self.previous / 'checkpoints/101/B1/selected').resolve())
        for label in ('initial', 'selected', 'last'):
            path = directory / label
            path.mkdir(parents=True)
            (path / 'adapter.bin').write_bytes(b'B1 weights')
            manifest = {'config': {'model_path': '/fixture-only'}, 'parameters': parameters,
                        'origin_checkpoint': origin, 'base_artifacts_sha256': {'base': 'fixture'},
                        'artifacts_sha256': artifact_hashes(path)}
            (path / 'local_checkpoint.json').write_text(json.dumps(manifest))
            checkpoints[label] = {'path': str(path.resolve()), 'manifest': manifest}
        history = [{'step': 0, 'tokens_processed': 0}, {'step': 1, 'tokens_processed': 20,
                    'rl_target_tokens_this_update': 20 if method == 'B3' else None}]
        result = TrainingResult('REINFORCE_EMA_EXACT_CONDITIONAL_KL' if method == 'B2' else 'SFT',
                    'fixture date', asdict(self.config), parameters, history, checkpoints,
                    0, 0.0, 20, 2, 1.0, 1, 0.1, origin)
        (directory / 'training_result.json').write_text(json.dumps(result.as_dict()))
        return directory

    def test_reuses_all_stages_preserving_checkpoint_bytes_and_remapping_ancestry(self):
        originals = {}
        for method in ('B1', 'B2', 'B3'):
            directory = self.stage(method)
            originals[method] = (directory / 'training_result.json').read_bytes()
        b1 = clone_completed_training(self.previous, self.output, 101, 'B1', self.config)
        b1_path = b1.checkpoints['selected']['path']
        for method in ('B2', 'B3'):
            result = clone_completed_training(self.previous, self.output, 101, method, self.config,
                    b1_path=b1_path, target_tokens=[20] if method == 'B3' else None)
            self.assertEqual(result.source_checkpoint, b1_path)
            old_manifest = self.previous / f'checkpoints/101/{method}/selected/local_checkpoint.json'
            new_manifest = Path(result.checkpoints['selected']['path']) / 'local_checkpoint.json'
            self.assertEqual(old_manifest.read_bytes(), new_manifest.read_bytes())
        for method, original in originals.items():
            self.assertEqual((self.previous / f'checkpoints/101/{method}/training_result.json').read_bytes(), original)
        record = json.loads((self.output / 'stage_reuse_manifest.json').read_text())
        self.assertEqual(set(record['reused_stages']), {'B1_seed101', 'B2_seed101', 'B3_seed101'})

    def test_missing_final_result_is_partial_and_not_reused(self):
        directory = self.stage('B1')
        (directory / 'training_result.json').unlink()
        self.assertIsNone(clone_completed_training(self.previous, self.output, 101, 'B1', self.config))
        self.assertFalse((self.output / 'checkpoints').exists())

    def test_rejects_changed_budget_or_checkpoint(self):
        directory = self.stage('B1')
        wrong_budget = TrainingConfig(updates=2, batch_size=2, seed=101)
        with self.assertRaisesRegex(ValueError, 'configuration'):
            clone_completed_training(self.previous, self.output, 101, 'B1', wrong_budget)
        (directory / 'last/adapter.bin').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'artifact mismatch'):
            clone_completed_training(self.previous, self.output, 101, 'B1', self.config)


if __name__ == '__main__':
    unittest.main()
