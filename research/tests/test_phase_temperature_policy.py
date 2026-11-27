"""CPU fixtures for per-phase sampling and matching likelihood distributions."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_local_001.model import Generation, LocalPolicy, artifact_hashes
from research.phase_temperature_policy import (
    DEFAULT_PHASE_TEMPERATURES, PHASE_MARKERS, PhaseTemperaturePolicy, verify_phase_profiles,
)
from research.src.policy.llm_controller import SpecLocalPolicy


class CharacterTokenizer:
    def decode(self, values, **kwargs):
        return ''.join(chr(value) for value in values)


@pytest.fixture
def policy_factory(monkeypatch):
    def init(self, model_path, **options):
        self.config = SimpleNamespace(model_path=str(model_path), temperature=options['temperature'])
        self.tokenizer = CharacterTokenizer()
        self.model = SimpleNamespace(eval=lambda: None)
    monkeypatch.setattr(SpecLocalPolicy, '__init__', init)
    return lambda **kwargs: PhaseTemperaturePolicy('/unused/cpu-fixture', **kwargs)


def prompt(policy, phase):
    return policy._instructions[phase] + '\n\nOBSERVACION\n{"user_messages":[]}'


@pytest.mark.parametrize('phase', list(DEFAULT_PHASE_TEMPERATURES))
def test_generate_uses_phase_temperature_and_preserves_output(policy_factory, monkeypatch, phase):
    policy = policy_factory()
    seen = []
    def generate(self, text, sample=False):
        seen.append((self.config.temperature, sample))
        return Generation(list(map(ord, text)), [1, 2], 'broken raw JSON', truncated=True)
    monkeypatch.setattr(LocalPolicy, 'generate', generate)
    output = policy.generate(prompt(policy, phase), sample=True)
    assert seen == [(DEFAULT_PHASE_TEMPERATURES[phase], True)]
    assert policy.config.temperature == 1.0
    assert output.text == 'broken raw JSON' and output.truncated
    assert output.completion_ids == [1, 2]
    assert output.metadata['phase_temperature'] == DEFAULT_PHASE_TEMPERATURES[phase]
    assert output.metadata['phase_profile_sha256'] == policy.phase_profile()['profile_sha256']


def test_greedy_uses_unchanged_global_base(policy_factory, monkeypatch):
    policy = policy_factory(temperature=.8)
    seen = []
    def generate(self, text, sample=False):
        seen.append((self.config.temperature, sample))
        return Generation(list(map(ord, text)), [1], '{}')
    monkeypatch.setattr(LocalPolicy, 'generate', generate)
    for phase in DEFAULT_PHASE_TEMPERATURES:
        output = policy.generate(prompt(policy, phase), sample=False)
        assert not output.metadata['phase_temperature_applied']
    assert seen == [(1.0, False)] * 3


def test_temperatures_restore_after_generation_and_likelihood_failure(policy_factory, monkeypatch):
    policy = policy_factory()
    def failure(*args, **kwargs):
        raise RuntimeError('backend failure')
    monkeypatch.setattr(LocalPolicy, 'generate', failure)
    with pytest.raises(RuntimeError):
        policy.generate(prompt(policy, 'select'), sample=True)
    assert policy.config.temperature == 1.0
    monkeypatch.setattr(LocalPolicy, 'sequence_logprobs', failure)
    generation = Generation(list(map(ord, prompt(policy, 'write'))), [1], '{}')
    with pytest.raises(RuntimeError):
        policy.sequence_logprobs(generation, sampling_distribution=True)
    assert policy.config.temperature == 1.0


def test_phase_classification_ignores_observation_and_metadata(policy_factory):
    policy = policy_factory()
    for phase in DEFAULT_PHASE_TEMPERATURES:
        text = prompt(policy, phase) + json.dumps(dict(PHASE_MARKERS))
        assert policy.phase_from_prompt(text) == phase
        generated = Generation(list(map(ord, '<chat user>\n' + text)), [1], '{}', metadata={'phase': 'WRONG'})
        assert policy.phase_from_generation(generated) == phase
    with pytest.raises(ValueError):
        policy.phase_from_prompt('OBSERVACION\n' + policy._instructions['select'])
    unknown = Generation(list(map(ord, 'Unknown instruction\nOBSERVACION\n' + PHASE_MARKERS['write'])), [1], '{}')
    with pytest.raises(ValueError):
        policy.phase_from_generation(unknown)


@pytest.mark.parametrize('values', [dict(select=1), dict(select=0, respond=.6, write=.35),
                                    dict(select=True, respond=.6, write=.35),
                                    dict(select=float('nan'), respond=.6, write=.35)])
def test_invalid_phase_profiles_fail_closed(policy_factory, values):
    with pytest.raises(ValueError):
        policy_factory(phase_temperatures=values)


def test_profiles_are_immutable_and_actor_reference_must_match(policy_factory):
    actor, reference = policy_factory(), policy_factory()
    assert verify_phase_profiles(actor, reference) == actor.phase_profile()
    with pytest.raises(TypeError):
        actor.phase_temperatures['select'] = .1
    returned = actor.phase_profile()
    returned['temperatures']['select'] = .1
    assert actor.phase_temperatures['select'] == 2.0
    different = policy_factory(phase_temperatures=dict(select=1, respond=.6, write=.35))
    with pytest.raises(ValueError, match='profiles differ'):
        verify_phase_profiles(actor, different)
    reference.config.temperature = .9
    with pytest.raises(ValueError, match='base temperature'):
        verify_phase_profiles(actor, reference)


@pytest.mark.parametrize('phase', list(DEFAULT_PHASE_TEMPERATURES))
def test_toy_actor_reference_likelihoods_use_same_phase_temperature(policy_factory, phase):
    torch = pytest.importorskip('torch')
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.tensor([.5, 1.1, -.7, .2]))
            self.config = SimpleNamespace(max_position_embeddings=20000)
        def forward(self, input_ids, attention_mask, use_cache=False):
            return SimpleNamespace(logits=self.scores[None, None, :].expand(1, input_ids.shape[1], -1))
    actor, reference = policy_factory(), policy_factory()
    for policy in (actor, reference):
        policy.torch, policy.model = torch, ToyModel()
    generated = Generation(list(map(ord, prompt(actor, phase))), [1, 2], '{}', metadata={'temperature': 99})
    expected = torch.log_softmax(actor.model.scores / DEFAULT_PHASE_TEMPERATURES[phase], -1)[[1, 2]]
    actual = actor.sequence_logprobs(generated, sampling_distribution=True)
    other = reference.sequence_logprobs(generated, sampling_distribution=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, other, rtol=0, atol=0)
    unscaled = actor.sequence_logprobs(generated, sampling_distribution=False)
    torch.testing.assert_close(unscaled, torch.log_softmax(actor.model.scores, -1)[[1, 2]], rtol=0, atol=0)
    actual.sum().backward()
    assert actor.model.scores.grad is not None
    assert actor.config.temperature == reference.config.temperature == 1.0


def test_from_checkpoint_preserves_subclass_and_profile_without_breaking_artifact_hash(policy_factory, monkeypatch, tmp_path):
    policy = policy_factory()
    def save(self, destination):
        Path(destination).mkdir(parents=True, exist_ok=True)
        (Path(destination) / 'weights.safetensors').write_bytes(b'CPU fixture')
        result = {'config': {'model_path': str(destination), 'temperature': .8},
                  'parameters': {'training_scope': 'base_weights'}, 'base_artifacts_sha256': {},
                  'artifacts_sha256': artifact_hashes(destination)}
        (Path(destination) / 'local_checkpoint.json').write_text(json.dumps(result))
        return result
    monkeypatch.setattr(LocalPolicy, 'save_checkpoint', save)
    manifest = policy.save_checkpoint(tmp_path / 'checkpoint')
    assert artifact_hashes(tmp_path / 'checkpoint') == manifest['artifacts_sha256']
    loaded = PhaseTemperaturePolicy.from_checkpoint(tmp_path / 'checkpoint', device='cpu')
    assert isinstance(loaded, PhaseTemperaturePolicy)
    assert loaded.config.temperature == 1.0
    assert loaded.phase_profile() == policy.phase_profile()
    with pytest.raises(ValueError, match='profile mismatch'):
        PhaseTemperaturePolicy.from_checkpoint(tmp_path / 'checkpoint', phase_temperatures=dict(select=1, respond=.6, write=.35))


def test_default_likelihood_remains_untempered_like_base(policy_factory, monkeypatch):
    policy = policy_factory()
    seen = []
    def likelihood(self, generated, *, sampling_distribution=False):
        seen.append((sampling_distribution, self.config.temperature))
        return [0.]
    monkeypatch.setattr(LocalPolicy, 'sequence_logprobs', likelihood)
    policy.sequence_logprobs(Generation([1], [1], 'unknown generic prompt'))
    assert seen == [(False, 1.0)]


@pytest.mark.parametrize('phase', list(DEFAULT_PHASE_TEMPERATURES))
def test_actual_qwen_tokenizer_identifies_uncached_reference_preamble_on_cpu(phase):
    transformers = pytest.importorskip('transformers')
    from research.phase_temperature_policy import PROMPT_ROOT
    parent = Path('research/runs/reward_review_20261104_002/warmup/optimizer/last/local_checkpoint.json')
    if not parent.is_file():
        pytest.skip('Local parent checkpoint unavailable; no downloads')
    model_root = json.loads(parent.read_text())['config']['model_path']
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_root, local_files_only=True, trust_remote_code=False)
    policy = PhaseTemperaturePolicy.__new__(PhaseTemperaturePolicy)
    policy.tokenizer, policy._phase_cache = tokenizer, {}
    text = (PROMPT_ROOT / f'{phase}.txt').read_text().rstrip() + '\n\nOBSERVACION\n{}'
    generated = Generation(policy.encode_prompt(text), [1], 'unused', metadata={'phase': 'wrong'})
    assert policy.phase_from_generation(generated) == phase
    prefix = tokenizer.decode(generated.prompt_ids[:128], skip_special_tokens=True)
    assert PHASE_MARKERS[phase] in prefix.split('OBSERVACION', 1)[0]
