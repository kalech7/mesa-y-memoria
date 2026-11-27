"""Resource gates use measured capacity and actual requested LLM parameter scope."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from llm_local_001 import preflight


GIB = 1024 ** 3


def environment(peft=True, free_gpu=4 * GIB):
    return {'requested_executable': '/existing/python', 'packages': {'peft': 'existing' if peft else None},
            'torch': {'cuda_available': True, 'bf16_supported': True,
                      'devices': [{'free_memory_bytes': free_gpu}]}}


def candidate(weight_bytes=2 * GIB):
    return {'weight_bytes': weight_bytes, 'config': {'torch_dtype': 'bfloat16'}, 'weight_files': []}


class MemoryPlanContracts(unittest.TestCase):
    def test_prefers_lora_without_arbitrary_checkpoint_size_cap(self):
        plan = preflight._smoke_plan(candidate(), environment(), {'MemAvailable_bytes': 32 * GIB})
        self.assertEqual(plan['status'], 'planned')
        self.assertEqual(plan['device'], 'cuda')
        self.assertEqual(plan['precision'], 'bfloat16')
        self.assertEqual(plan['parameter_scope'], 'lora_adapters_integrated_in_llm')
        self.assertGreater(candidate()['weight_bytes'], 1_500_000_000)
        self.assertLess(plan['estimated_peak_bytes'], plan['available_memory_bytes'] * .8)
        full = preflight._smoke_plan(candidate(), environment(), {'MemAvailable_bytes': 32 * GIB}, 'full', 'cuda')
        self.assertEqual(full['status'], 'blocked')
        self.assertIn('not a measured OOM', full['reason'])

    def test_explicit_cuda_does_not_silently_fall_back_to_cpu(self):
        env = environment(free_gpu=128 * 1024 * 1024)
        required = preflight._smoke_plan(candidate(), env, {'MemAvailable_bytes': 32 * GIB}, 'lora', 'cuda')
        self.assertEqual(required['status'], 'blocked')
        automatic = preflight._smoke_plan(candidate(), env, {'MemAvailable_bytes': 32 * GIB}, 'lora')
        self.assertEqual(automatic['device'], 'cpu')
        self.assertEqual(automatic['precision'], 'float32')
        self.assertEqual(automatic['attempts'][0]['device'], 'cuda')

    def test_missing_peft_does_not_substitute_full_training_for_selected_lora(self):
        result = preflight._smoke_plan(candidate(), environment(peft=False), {'MemAvailable_bytes': 32 * GIB}, 'lora', 'cuda')
        self.assertEqual(result['status'], 'blocked')
        self.assertIn('PEFT', result['reason'])
        explicit_full = preflight._smoke_plan(candidate(1024), environment(peft=False), {'MemAvailable_bytes': 32 * GIB}, 'full', 'cpu')
        self.assertEqual(explicit_full['status'], 'planned')
        self.assertEqual(explicit_full['parameter_scope'], 'base_weights')

    def test_missing_measured_memory_never_passes(self):
        result = preflight._smoke_plan(candidate(), environment(free_gpu=0), {}, 'lora', 'cuda')
        self.assertEqual(result['status'], 'blocked')

    def test_safetensors_shapes_override_file_byte_estimate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.safetensors'
            header = json.dumps({'weight': {'dtype': 'BF16', 'shape': [100, 200], 'data_offsets': [0, 40000]},
                                 '__metadata__': {'format': 'pt'}}).encode()
            path.write_bytes(struct.pack('<Q', len(header)) + header)
            count, source = preflight._parameter_estimate({'weight_bytes': path.stat().st_size,
                                                         'weight_files': [{'path': str(path)}]})
            self.assertEqual(count, 20000)
            self.assertIn('tensor shapes', source)


class ModelAuditContracts(unittest.TestCase):
    def test_readme_without_license_does_not_pass_license_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'config.json').write_text(json.dumps({'architectures': ['GPT2LMHeadModel']}))
            (root / 'model.safetensors').write_bytes(b'weights')
            (root / 'tokenizer_config.json').write_text('{}')
            (root / 'README.md').write_text('# Model\nDescription only.\n')
            self.assertEqual(preflight._model_info(root)['license_evidence'], [])
            (root / 'README.md').write_text('---\nlicense: apache-2.0\n---\n')
            self.assertEqual(len(preflight._model_info(root)['license_evidence']), 1)

    def test_encoder_cannot_become_causal_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'config.json').write_text(json.dumps({'architectures': ['BertModel']}))
            (root / 'model.safetensors').write_bytes(b'weights')
            (root / 'tokenizer_config.json').write_text('{}')
            info = preflight._model_info(root)
            self.assertFalse(info['is_causal_candidate'])
            self.assertEqual(info['trainability'], 'not_a_complete_local_causal_llm')

    def test_no_causal_checkpoint_never_executes_model_smoke(self):
        def command(args, timeout=30):
            self.assertNotIn(preflight.MODEL_SMOKE, args)
            return {'returncode': 0, 'stdout': '', 'stderr': ''}
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(preflight, '_discover', return_value=([], [], {})), \
                 patch.object(preflight, '_ollama_inventory', return_value={'models': [], 'inference_calls': 0}), \
                 patch.object(preflight, '_run', side_effect=command):
                result = preflight.run_preflight(Path(temporary), training_mode='lora', device='cuda')
            self.assertEqual(result['status'], 'blocked')
            self.assertIsNone(result['selected_model'])
            artifact = json.loads((Path(temporary) / 'model_manifest.json').read_text())
            self.assertEqual(artifact['configuration']['training_mode'], 'lora')
            self.assertEqual(artifact['configuration']['device'], 'cuda')
            self.assertTrue(artifact['created_at_utc'])


if __name__ == '__main__':
    unittest.main()
