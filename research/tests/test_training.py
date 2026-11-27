"""Actor masking, real prefixes, deferred credit and frozen-reference contracts."""
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_local_001.model import Generation
from research.src.policy.llm_controller import LLMController, InvalidOutput, validate_action
from research.src.training.sft import completion_label_mask, selection_key, train_sft
from research.src.training.episodic_grpo import (
    EpisodeRollout, PhaseSample, clipped_token_loss, group_advantages,
    technical_smoke, validate_group, inspect_candidate, QWEN_ARCHITECTURES,
)


def test_t20_only_actor_tokens_have_labels():
    assert completion_label_mask([11, 12, 13], [14, 15]) == [-100, -100, -100, 14, 15]
    phase = PhaseSample("select", Generation([11, 12], [13], "x"))
    assert phase.actor_mask == [0, 0, 1]


def test_t22_future_credit_only_changes_longitudinal_advantages():
    a = EpisodeRollout("root", "world", 1, [], 1., 0.)
    b = EpisodeRollout("root", "world", 2, [], 1., 1.)
    assert group_advantages([a.reward("short"), b.reward("short")]) == [0., 0.]
    long = group_advantages([a.reward("long"), b.reward("long")])
    assert long[0] < 0 < long[1]


def test_t25_zero_variation_has_zero_task_advantage():
    assert group_advantages([3., 3., 3., 3.]) == [0., 0., 0., 0.]


def test_t26_step_zero_wins_equal_checkpoint_scores():
    metrics = {"utility": 1., "unsupported_writes": 0., "questions": .2}
    assert selection_key(metrics, 0) > selection_key(metrics, 10)
    assert selection_key({**metrics, "unsupported_writes": .1}, 0) < selection_key(metrics, 10)


def make_rollouts(tmp_path):
    phases = [PhaseSample(phase, Generation([1, 2], [3], "{}"))
              for phase in ("select", "respond", "write")]
    return [EpisodeRollout("r", "w", seed, phases, 1., 0.,
                           memory_path=str(tmp_path / f"{seed}.sqlite")) for seed in range(4)]


def test_t23_rollout_database_and_seeds_must_be_distinct(tmp_path):
    rollouts = make_rollouts(tmp_path)
    validate_group(rollouts)
    rollouts[1].memory_path = rollouts[0].memory_path
    with pytest.raises(ValueError, match="SQLite"):
        validate_group(rollouts)


def test_t24_phases_remain_separate_and_ordered(tmp_path):
    rollouts = make_rollouts(tmp_path)
    rollouts[0].phases = rollouts[0].phases[:1]
    with pytest.raises(ValueError, match="prefixes"):
        validate_group(rollouts)


class FakeBackend:
    def __init__(self):
        self.config = SimpleNamespace(max_new_tokens=999)
        self.calls = []

    def encode_prompt(self, prompt):
        return [1] * 10

    def generate(self, prompt, sample=False):
        self.calls.append((prompt, sample, self.config.max_new_tokens))
        return Generation([1], [2], '{"action":"PROCEED","slot":null}')


def test_actor_rejects_private_keys_and_records_exact_prefix(tmp_path):
    (tmp_path / "select.txt").write_text("Select.")
    backend = FakeBackend()
    controller = LLMController(backend, tmp_path)
    observation = {"memory": [], "user_messages": [], "query": {}, "reuse_regime": "future"}
    action = controller.select(observation)
    assert action == {"action": "PROCEED", "slot": None}
    assert backend.calls[0][2] == 64
    assert backend.config.max_new_tokens == 999
    assert controller.phase_records[0]["prompt"] == backend.calls[0][0]
    with pytest.raises(ValueError, match="private"):
        controller.select({**observation, "gold_action": "ASK"})


def test_invalid_json_is_not_retried(tmp_path):
    (tmp_path / "select.txt").write_text("Select.")
    backend = FakeBackend()
    original = backend.generate

    def invalid(prompt, sample=False):
        result = original(prompt, sample)
        result.text = "not json"
        return result

    backend.generate = invalid
    controller = LLMController(backend, tmp_path)
    with pytest.raises(InvalidOutput):
        controller.select({"memory": []})
    assert len(backend.calls) == 1
    assert controller.phase_records[0]["retry_count"] == 0


def test_forced_ask_retains_one_actual_select_prefix_and_no_extra_call(tmp_path):
    (tmp_path / "select.txt").write_text("Select.")
    backend = FakeBackend()
    controller = LLMController(backend, tmp_path)
    with pytest.raises(InvalidOutput, match="C-ALWAYS"):
        controller.select_forced_ask({"memory": []})
    assert len(backend.calls) == 1
    assert len(controller.phase_records) == 1
    assert controller.phase_records[0]["phase"] == "select"
    assert "exactamente una pregunta" in controller.phase_records[0]["prompt"]


def test_b0p_prompt_variant_preserves_output_contract(tmp_path):
    (tmp_path / "select.txt").write_text("Select.")
    backend = FakeBackend()
    controller = LLMController(backend, tmp_path, prompt_variant="B0-P")
    assert controller.select({"memory": []}) == {"action": "PROCEED", "slot": None}
    assert "Comprueba internamente" in controller.phase_records[0]["prompt"]


def test_model_directory_name_cannot_promote_small_model_to_candidate(tmp_path):
    renamed = tmp_path / "Qwen2.5-3B-Instruct"
    renamed.mkdir()
    (renamed / "config.json").write_text(json.dumps({"model_type": "qwen2", "hidden_size": 16,
                                                     "_name_or_path": "Qwen/Qwen2.5-3B-Instruct"}))
    check = inspect_candidate(renamed)
    assert check["verified"] is False
    assert not check["architecture_matches"]
    report = technical_smoke(renamed, tmp_path / "rejected")
    assert report["load"] == "not_executed"
    assert report["g0_eligible"] is False


def test_architecture_alone_does_not_replace_a_pinned_manifest(tmp_path):
    config = {**QWEN_ARCHITECTURES["Qwen/Qwen2.5-1.5B-Instruct"],
              "model_type": "qwen2", "vocab_size": 151936,
              "tie_word_embeddings": True, "max_position_embeddings": 32768}
    (tmp_path / "config.json").write_text(json.dumps(config))
    check = inspect_candidate(tmp_path)
    assert check["architecture_matches"] == ["Qwen/Qwen2.5-1.5B-Instruct"]
    assert check["verified"] is False
    assert "manifest" in check["reasons"][0]


def test_frozen_reader_clears_earlier_s3_prefixes(tmp_path):
    from research.src.readers.frozen_llm import FrozenLLMReader
    (tmp_path / "reader.txt").write_text("Read.")
    backend = FakeBackend()
    backend.model = SimpleNamespace(parameters=lambda: iter(()))
    backend.training_enabled = False
    backend.generate = lambda prompt, sample=False: Generation([1], [2],
        '{"mode":"ANSWER","value":"pasta","branches":[],"evidence_ids":["m1"]}')
    reader = FrozenLLMReader(backend, tmp_path)
    assert reader.read([], {"attribute": "dinner_preference"})["value"] == "pasta"
    assert reader.controller.phase_records == []


def test_final_training_is_gated_before_loading_model(tmp_path):
    with pytest.raises(RuntimeError, match="TECHNICAL_LOCK_REQUIRED|GATE_REQUIRED"):
        train_sft(None, [], tmp_path, lambda _: 0.)


def test_real_environment_to_training_bridge_retains_phase_prefixes(tmp_path):
    from research.src.data.build_bank import build_bank
    from research.src.policy.rule_controller import RuleController
    from research.src.training.rollouts import make_rollout_collector
    teacher = RuleController()

    class VisibleBackend(FakeBackend):
        def generate(self, prompt, sample=False):
            observation = json.loads(prompt.split("\n\nOBSERVACION\n", 1)[1])
            phase = ("select" if prompt.startswith("Selecciona") else
                     "respond" if prompt.startswith("Responde") else "write")
            action = getattr(teacher, phase)(observation)
            return Generation([1, 2], [3, 4], json.dumps(action, ensure_ascii=False))

    episode = next(item for item in build_bank(roots=5) if item["scenario_family"] == "F4")
    collector = make_rollout_collector(tmp_path)
    rollout = collector(VisibleBackend(), episode, 101)
    assert Path(rollout.memory_path).is_file()
    assert [phase.phase for phase in rollout.phases] == ["select", "respond", "write"]
    assert rollout.split == "dev_smoke"
    assert rollout.questions == 1
    assert rollout.invalid_outputs == 0
    assert rollout.r2 == 1
    assert rollout.r3 == 1
    with pytest.raises(FileExistsError):
        collector(VisibleBackend(), episode, 101)


def test_grpo_clipping_kl_and_gradient():
    torch = pytest.importorskip("torch")
    new = torch.tensor([-.7, -.8], requires_grad=True)
    old = new.detach().clone()
    loss, kl = clipped_token_loss(new, old, old, 1.)
    assert float(kl.detach()) == 0.
    loss.backward()
    assert bool(torch.isfinite(new.grad).all())
    assert bool((new.grad < 0).all())
    # A positive-advantage action beyond the upper clipping bound gets no task gradient.
    new = torch.tensor([0.], requires_grad=True)
    old = torch.tensor([-1.])
    loss, _ = clipped_token_loss(new, old, new.detach(), 1., kl_coefficient=0.)
    loss.backward()
    assert float(new.grad[0]) == 0.


def test_t19_t21_real_tiny_lora_sft_and_grpo(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    torch.set_num_threads(1)
    model_dir = tmp_path / "tiny_qwen_fixture"
    model_dir.mkdir()
    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    vocab.update({f"word{i}": i for i in range(3, 32)})
    tokenizer = tokenizers.Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    hf_tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer,
        unk_token="<unk>", pad_token="<pad>", eos_token="<eos>", bos_token="<unk>")
    hf_tokenizer.save_pretrained(model_dir)
    config = transformers.Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=4096, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    transformers.Qwen2ForCausalLM(config).save_pretrained(model_dir)
    report = technical_smoke(model_dir, tmp_path / "smoke", allow_nonprotocol_model=True)
    assert report["status"] == "passed_nonprotocol_fixture", report.get("error")
    assert report["g0_eligible"] is False
    for phase in ("sft", "grpo"):
        result = report[phase]
        assert result["trainable_before"] != result["trainable_after"]
        assert result["frozen_before"] == result["frozen_after"]
        assert result["resource_counters"]["updates"] == 1
    assert report["grpo"]["reference_before"] == report["grpo"]["reference_after"]
    prefixes = [json.loads(line) for line in (tmp_path / "smoke/grpo/rollouts.jsonl").read_text().splitlines()]
    assert len(prefixes) == 4
    assert len(prefixes[0]["rollout"]["phases"]) == 3
    for phase, mask in zip(prefixes[0]["rollout"]["phases"], prefixes[0]["actor_masks"]):
        prefix_length = len(phase["generation"]["prompt_ids"])
        assert not any(mask[:prefix_length])
        assert all(mask[prefix_length:])


def test_effective_batch_groups_worlds_under_distinct_roots():
    from research.src.training.episodic_grpo import root_batch
    episodes = [{"root_id": str(root), "world": world} for root in range(8) for world in ("a", "b")]
    first, second = root_batch(episodes, 0, 8), root_batch(episodes, 1, 8)
    assert len({e["root_id"] for e in first}) == 8
    assert all(e["world"] == "a" for e in first)
    assert all(e["world"] == "b" for e in second)


def test_small_candidate_still_requires_pinned_untampered_artifacts(tmp_path):
    import hashlib
    from research.src.artifacts import canonical
    config = {**QWEN_ARCHITECTURES['Qwen/Qwen2.5-0.5B-Instruct'],
              'model_type': 'qwen2', 'vocab_size': 151936,
              'tie_word_embeddings': True, 'max_position_embeddings': 32768}
    (tmp_path / 'config.json').write_text(canonical(config))
    for name in ('tokenizer.json', 'tokenizer_config.json', 'model.safetensors'):
        (tmp_path / name).write_text('unit fixture, not real weights')
    files = {p.name: {'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in tmp_path.iterdir()}
    manifest = {'repo': 'Qwen/Qwen2.5-0.5B-Instruct', 'revision': '7' * 40, 'files': files}
    (tmp_path / 'download_manifest.json').write_text(canonical(manifest))
    assert inspect_candidate(tmp_path)['verified'] is True
    (tmp_path / 'model.safetensors').write_text('tampered')
    check = inspect_candidate(tmp_path)
    assert check['verified'] is False
    assert any('hash differs' in reason for reason in check['reasons'])


def test_hardware_profile_rejects_oversized_prefix_without_generation(tmp_path):
    from research.src.policy.llm_controller import MAX_INPUT_TOKENS
    (tmp_path / 'select.txt').write_text('Select.')
    backend = FakeBackend()
    backend.encode_prompt = lambda prompt: [1] * (MAX_INPUT_TOKENS + 1)
    with pytest.raises(ValueError, match='INPUT_TOKEN_LIMIT'):
        LLMController(backend, tmp_path).select({'memory': []})
    assert not backend.calls
