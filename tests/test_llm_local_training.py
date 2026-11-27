"""No model downloads: test scientific invariants independently of HF availability."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from llm_local_001.model import Generation, LocalPolicy, artifact_hashes
from llm_local_001.training import (
    EpisodeTrajectory, TrainingConfig, completion_label_mask,
    discounted_returns, update_baseline, train_sft, train_rl,
)


class TrainingContractTests(unittest.TestCase):
    def test_prompt_targets_are_masked(self):
        self.assertEqual(completion_label_mask([1, 2, 3], [4, 5]), [-100, -100, -100, 4, 5])

    def test_s2_return_includes_s3_consequence(self):
        self.assertEqual(discounted_returns([-0.1, 1.0]), [0.9, 1.0])
        self.assertEqual(discounted_returns([-0.1, 1.0], 0.5), [0.4, 1.0])

    def test_baseline_update_does_not_mutate_previous_value(self):
        baseline_used = 0.25
        later_baseline = update_baseline(baseline_used, [1.0, -1.0], 0.9)
        self.assertEqual(baseline_used, 0.25)
        self.assertAlmostEqual(later_baseline, 0.225)

    def test_every_action_requires_reward_even_invalid_json(self):
        invalid = Generation([1], [2], "invalid JSON")
        EpisodeTrajectory([invalid], [-1.0]).validate()
        with self.assertRaises(ValueError):
            EpisodeTrajectory([invalid], []).validate()

    def test_invalid_budget_fails_before_training(self):
        for config in (TrainingConfig(batch_size=0), TrainingConfig(kl_coefficient=-1),
                       TrainingConfig(baseline_decay=1.0), TrainingConfig(gamma=1.1)):
            with self.assertRaises(ValueError):
                config.validate()

    def test_checkpoint_tamper_detection_precedes_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            weights = path / "adapter_model.safetensors"
            weights.write_bytes(b"saved weights")
            manifest = {"artifacts_sha256": artifact_hashes(path)}
            (path / "local_checkpoint.json").write_text(json.dumps(manifest))
            weights.write_bytes(b"modified weights")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                LocalPolicy.from_checkpoint(path)

    def test_gguf_is_not_accepted_as_trainable_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text("{}")
            (path / "weights.gguf").write_bytes(b"GGUF")
            with self.assertRaisesRegex(ValueError, "GGUF"):
                LocalPolicy(str(path))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable; no dependency installation")
class CausalTokenGradientTests(unittest.TestCase):
    def test_selected_vocabulary_projection_matches_full_logits_and_gradients(self):
        import torch

        class SelectedLogits(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scores = torch.nn.Parameter(torch.randn(1, 5, 7))
                self.config = SimpleNamespace(max_position_embeddings=32)
                self.positions = None

            def forward(self, logits_to_keep=0, **kwargs):
                self.positions = logits_to_keep
                logits = self.scores if isinstance(logits_to_keep, int) else self.scores[:, logits_to_keep]
                return SimpleNamespace(logits=logits)

        class AdapterWrapper(torch.nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
                self.config = base.config

            def get_base_model(self):
                return self.base

            def forward(self, **kwargs):
                return self.base(**kwargs)

        policy = LocalPolicy.__new__(LocalPolicy)
        policy.torch = torch
        base = SelectedLogits()
        policy.model = AdapterWrapper(base)
        policy.config = SimpleNamespace(temperature=0.7)
        generation = Generation([1, 2, 3], [4, 5], "fixture")
        actual = policy.sequence_logits(generation, sampling_distribution=True)
        expected = base.scores[0, 2:4].float() / 0.7
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(base.positions.tolist(), [2, 3])
        actual.square().sum().backward()
        actual_grad = base.scores.grad.clone()
        base.scores.grad = None
        expected.square().sum().backward()
        torch.testing.assert_close(actual_grad, base.scores.grad, rtol=0, atol=0)

    def test_only_generated_token_logits_receive_gradients(self):
        import torch

        class TinyLogits(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scores = torch.nn.Parameter(torch.zeros(1, 5, 7))
                self.config = SimpleNamespace(max_position_embeddings=32)

            def forward(self, **kwargs):
                return SimpleNamespace(logits=self.scores)

        # Mathematical unit fixture only, never used as an experimental model.
        policy = LocalPolicy.__new__(LocalPolicy)
        policy.torch = torch
        policy.model = TinyLogits()
        policy.config = SimpleNamespace(temperature=1.0)
        generation = Generation([1, 2, 3], [4, 5], "malformed output still optimized")
        logp = policy.sequence_logprobs(generation)
        self.assertEqual(tuple(logp.shape), (2,))
        (-logp.sum()).backward()
        gradient = policy.model.scores.grad
        self.assertEqual(float(gradient[:, :2].abs().sum()), 0.0)
        self.assertGreater(float(gradient[:, 2:4].abs().sum()), 0.0)
        self.assertEqual(float(gradient[:, 4:].abs().sum()), 0.0)


@unittest.skipUnless(importlib.util.find_spec("transformers") and importlib.util.find_spec("tokenizers"),
                     "Optional existing HF dependencies unavailable; no installation or download")
class OfflineCausalIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("peft"), "Optional PEFT unavailable")
    def test_checkpointed_qwen_likelihood_preserves_eval_logits_and_gradients(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.set_num_threads(1)
        torch.manual_seed(13)
        policy = LocalPolicy.__new__(LocalPolicy)
        policy.torch = torch
        policy.config = SimpleNamespace(temperature=1.0, gradient_checkpointing=True)
        policy.training_enabled = True
        base = Qwen2ForCausalLM(Qwen2Config(vocab_size=31, hidden_size=16,
            intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=1, max_position_embeddings=128, attention_dropout=0.2))
        policy.model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=2,
            lora_alpha=4, lora_dropout=0.3, target_modules=["q_proj", "v_proj"]))
        policy._configure_training_memory()
        policy.model.eval()
        generated = Generation([1, 2, 3, 4], [5, 6, 7], "software fixture")
        parameters = [p for p in policy.model.parameters() if p.requires_grad]
        reference_logits = policy.sequence_logits(generated)
        reference_logits.square().sum().backward()
        reference_gradients = [p.grad.clone() for p in parameters]
        policy.model.zero_grad(set_to_none=True)
        rng_before = torch.get_rng_state().clone()
        with policy.likelihood_mode():
            self.assertTrue(base.model.training)
            self.assertFalse(base.model.layers[0].self_attn.training)
            self.assertTrue(all(not m.training for m in policy.model.modules()
                                if isinstance(m, torch.nn.Dropout)))
            actual_logits = policy.sequence_logits(generated)
            torch.testing.assert_close(actual_logits, reference_logits, rtol=0, atol=0)
            actual_logits.square().sum().backward()
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertTrue(all(not m.training for m in policy.model.modules()))
        for parameter, expected in zip(parameters, reference_gradients):
            torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)

    def test_real_hf_sft_rl_b3_checkpoints_without_downloads(self):
        """Random tiny LM is a software fixture, never a scientific baseline."""
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "synthetic_software_fixture"
            base.mkdir()
            vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
                     "prompt": 4, "good": 5, "bad": 6, "user": 7, "assistant": 8}
            tokenizer_backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
            tokenizer_backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_backend,
                bos_token="<bos>", eos_token="<eos>", unk_token="<unk>", pad_token="<pad>")
            tokenizer.save_pretrained(base)
            GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=32,
                n_embd=8, n_layer=1, n_head=1, bos_token_id=1, eos_token_id=2,
                pad_token_id=0, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)).save_pretrained(base)
            policy = LocalPolicy(str(base), device="cpu", precision="auto", lora=False,
                                 full_finetune=True, max_new_tokens=2)
            self.assertEqual(policy.config.precision, "float32")
            fingerprint = policy.base_fingerprint()
            counter = [0]

            def increasing_validation(_policy):
                counter[0] += 1
                return counter[0]

            config = TrainingConfig(updates=1, learning_rate=0.01, seed=7)
            demos = [{"prompt": "prompt", "completion": "good"}]
            b1 = train_sft(policy, demos, root / "B1", config, increasing_validation)
            self.assertGreater(b1.changed_parameters, 0)
            self.assertEqual(b1.best_step, 1)
            source = b1.checkpoints["selected"]["path"]
            b2policy = LocalPolicy.from_checkpoint(source, trainable=True)
            b3policy = LocalPolicy.from_checkpoint(source, trainable=True)
            reference = LocalPolicy.from_checkpoint(source, device="cpu")
            self.assertEqual(b2policy.base_fingerprint(), fingerprint)
            self.assertEqual(b3policy.base_fingerprint(), fingerprint)
            generated = b2policy.generate("prompt", sample=True)
            with torch.no_grad():
                self.assertTrue(torch.equal(b2policy.sequence_logits(generated), reference.sequence_logits(generated)))

            def rollout(current):
                return EpisodeTrajectory([current.generate("prompt", sample=True),
                                          current.generate("prompt", sample=True)], [-0.1, 1.0])

            b2 = train_rl(b2policy, rollout, root / "B2", config, lambda _: 0.0,
                          reference_policy=reference)
            self.assertGreater(b2.changed_parameters, 0)
            self.assertEqual(b2.best_step, 0)
            self.assertAlmostEqual(b2.history[1]["episode_return_mean"], 0.9)
            self.assertAlmostEqual(b2.history[1]["exact_conditional_kl_mean"], 0.0, places=6)
            self.assertTrue(all(p.grad is None for p in reference.model.parameters()))
            b3 = train_sft(b3policy, demos, root / "B3", config, lambda _: 0.0,
                           target_tokens_per_update=[b2.tokens_processed])
            self.assertGreaterEqual(b3.tokens_processed, b2.tokens_processed)
            self.assertLess(b3.history[1]["target_token_overshoot"], 3)
            self.assertEqual(b2.checkpoints["initial"]["manifest"]["artifacts_sha256"],
                             b3.checkpoints["initial"]["manifest"]["artifacts_sha256"])
            self.assertEqual(b2.source_checkpoint, b3.source_checkpoint)
            self.assertTrue(Path(b2.checkpoints["last"]["path"]).is_dir())


if __name__ == "__main__":
    unittest.main()
