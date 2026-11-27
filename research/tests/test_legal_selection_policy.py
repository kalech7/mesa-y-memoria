"""CPU categorical-policy math, interface and real-tokenizer contracts."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_local_001.model import Generation, LocalPolicy
from research.legal_selection_policy import ACTION_CATALOG, LegalSelectionPolicy
from research.phase_temperature_policy import verify_phase_profiles
from research.src.policy.llm_controller import SpecLocalPolicy, validate_action
from research.src.training.episodic_grpo import clipped_token_loss


class CharacterTokenizer:
    eos_token_id = 0
    chat_template = None
    def encode(self, text, **kwargs):
        return list(map(ord, text))
    def decode(self, values, skip_special_tokens=True):
        return ''.join(chr(value) for value in values if value != 0 or not skip_special_tokens)


@pytest.fixture
def toy(monkeypatch):
    torch = pytest.importorskip('torch')
    class Scores(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.tensor([.1, .3, -.1, .2, -.3]))
            self.config = SimpleNamespace(max_position_embeddings=20000)
    def init(self, model_path, **options):
        self.config = SimpleNamespace(temperature=options['temperature'], gradient_checkpointing=False)
        self.tokenizer, self.torch, self.model = CharacterTokenizer(), torch, Scores()
        self.training_enabled, self.calls = False, []
    def token_logprobs(self, generated, *, sampling_distribution=False):
        assert not sampling_distribution
        index = ACTION_CATALOG.index(generated.text)
        assert generated.completion_ids[-1] == self.tokenizer.eos_token_id
        self.calls.append((index, self.torch.is_grad_enabled()))
        return self.model.scores[index].expand(len(generated.completion_ids))
    monkeypatch.setattr(SpecLocalPolicy, '__init__', init)
    monkeypatch.setattr(SpecLocalPolicy, 'sequence_logprobs', token_logprobs)
    return lambda: LegalSelectionPolicy('/unused/cpu-fixture')


def prompt(policy, phase='select'):
    return policy._instructions[phase] + '\n\nOBSERVACION\n{"user_messages":[]}'


def test_catalog_is_exactly_five_legal_public_actions():
    parsed = [json.loads(text) for text in ACTION_CATALOG]
    assert parsed[0] == {'action': 'PROCEED', 'slot': None}
    assert {row['slot'] for row in parsed[1:]} == {'TEMPORAL_SCOPE', 'CURRENT_CONTEXT', 'VALID_UNTIL', 'GENERIC'}
    assert len(set(ACTION_CATALOG)) == 5
    for row in parsed:
        validate_action('select', row)


def test_categorical_probabilities_normalize_and_greedy_scores_exactly_five_candidates(toy):
    policy = toy()
    policy.model.train()
    output = policy.generate(prompt(policy), sample=False)
    assert policy.calls == [(index, False) for index in range(5)]
    assert output.text == ACTION_CATALOG[1]
    assert output.completion_ids == list(policy._completion_catalog()[1])
    assert policy.tokenizer.decode(output.completion_ids, skip_special_tokens=True) == output.text
    assert not output.truncated and policy.model.training
    probabilities = policy.torch.tensor(output.metadata['action_logprobs']).exp()
    assert float(probabilities.sum()) == pytest.approx(1.)
    assert output.metadata['likelihood_unit'] == 'action'
    assert output.metadata['chosen_action_logprob'] == pytest.approx(output.metadata['action_logprobs'][1])
    assert policy.config.temperature == 1.


def test_sampling_can_choose_every_action_without_filtering_or_repair(toy, monkeypatch):
    policy = toy()
    observed = []
    for choice in range(5):
        def sample(probabilities, count):
            assert count == 1 and probabilities.numel() == 5
            assert float(probabilities.sum()) == pytest.approx(1.)
            return policy.torch.tensor([choice])
        monkeypatch.setattr(policy.torch, 'multinomial', sample)
        output = policy.generate(prompt(policy), sample=True)
        observed.append(output.text)
        assert output.metadata['action_index'] == choice
        assert output.metadata['sampled'] and output.completion_ids[-1] == 0
    assert tuple(observed) == ACTION_CATALOG
    assert len(policy.calls) == 25


def test_actor_reference_action_likelihood_matches_and_mean_replication_has_categorical_gradient(toy):
    actor, reference = toy(), toy()
    assert verify_phase_profiles(actor, reference)
    generated = actor.generate(prompt(actor), sample=False)
    actor.calls.clear()
    actual = actor.sequence_logprobs(generated, sampling_distribution=True)
    other = reference.sequence_logprobs(generated, sampling_distribution=True)
    expected = actor.torch.log_softmax(actor.model.scores, dim=-1)[1]
    assert actual.numel() == len(generated.completion_ids)
    actor.torch.testing.assert_close(actual, expected.expand_as(actual), rtol=0, atol=0)
    actor.torch.testing.assert_close(actual, other, rtol=0, atol=0)
    assert actor.calls == [(index, True) for index in range(5)]
    actual.mean().backward()
    expected_gradient = -actor.torch.softmax(actor.model.scores.detach(), -1)
    expected_gradient[1] += 1
    actor.torch.testing.assert_close(actor.model.scores.grad, expected_gradient, rtol=1e-6, atol=1e-7)
    assert actor.config.temperature == reference.config.temperature == 1.


def test_historical_core_mean_is_exact_categorical_ratio_and_clipped_loss(toy):
    policy = toy()
    generated = policy.generate(prompt(policy), sample=False)
    old = policy.sequence_logprobs(generated, sampling_distribution=True).detach()
    assert policy.torch.equal((old - old).exp(), policy.torch.ones_like(old))
    with policy.torch.no_grad():
        policy.model.scores[1] += .4
    new = policy.sequence_logprobs(generated, sampling_distribution=True)
    reference = old.clone()
    loss, kl = clipped_token_loss(new, old, reference, .7)
    action_loss, action_kl = clipped_token_loss(new[:1], old[:1], reference[:1], .7)
    policy.torch.testing.assert_close(loss, action_loss, rtol=1e-6, atol=1e-7)
    policy.torch.testing.assert_close(kl, action_kl, rtol=1e-6, atol=1e-7)
    ratio = (new - old).exp()
    assert policy.torch.equal(ratio, ratio[:1].expand_as(ratio))


def test_unknown_completion_is_rejected_instead_of_repaired(toy):
    policy = toy()
    generation = Generation(policy.encode_prompt(prompt(policy)), [123, 125, 0], '{}')
    with pytest.raises(ValueError, match='canonical actions'):
        policy.sequence_logprobs(generation, sampling_distribution=True)
    assert policy.calls == []


def test_other_phases_delegate_with_original_token_temperatures(toy, monkeypatch):
    policy = toy()
    seen = []
    def generate(self, text, sample=False):
        seen.append(self.config.temperature)
        return Generation(self.encode_prompt(text), [1], 'unrepaired raw output')
    monkeypatch.setattr(LocalPolicy, 'generate', generate)
    for phase, temperature in (('respond', .6), ('write', .35)):
        output = policy.generate(prompt(policy, phase), sample=True)
        assert output.text == 'unrepaired raw output'
        assert output.metadata['likelihood_unit'] == 'token'
        assert seen[-1] == temperature
    assert policy.config.temperature == 1.


def test_profile_hash_catalog_and_temperatures_are_frozen(toy):
    policy = toy()
    profile = policy.phase_profile()
    assert profile['version'] == 'legal_selection_policy_v1'
    assert profile['temperatures'] == {'select': 1., 'respond': .6, 'write': .35}
    assert profile['selection']['catalog'] == list(ACTION_CATALOG)
    assert profile['selection']['temperature'] == 1.
    assert len(profile['selection']['catalog_sha256']) == len(profile['profile_sha256']) == 64
    with pytest.raises(ValueError, match='frozen'):
        policy._set_phase_temperatures({'select': 2., 'respond': .6, 'write': .35})


def test_actual_qwen_tokenizer_roundtrips_all_canonical_choices_including_eos_on_cpu():
    transformers = pytest.importorskip('transformers')
    parent = Path('research/runs/reward_review_20261104_002/warmup/optimizer/last/local_checkpoint.json')
    if not parent.is_file():
        pytest.skip('Local parent unavailable; no downloads')
    model_root = json.loads(parent.read_text())['config']['model_path']
    policy = LegalSelectionPolicy.__new__(LegalSelectionPolicy)
    policy.tokenizer = transformers.AutoTokenizer.from_pretrained(model_root, local_files_only=True, trust_remote_code=False)
    catalog = policy._completion_catalog()
    for text, ids in zip(ACTION_CATALOG, catalog):
        assert ids[-1] == policy.tokenizer.eos_token_id
        assert policy.tokenizer.decode(ids, skip_special_tokens=True) == text
        assert list(ids[:-1]) == policy.tokenizer.encode(text, add_special_tokens=False)
    assert len(set(catalog)) == 5


@pytest.mark.parametrize('with_saved_profile', [False, True])
def test_checkpoint_roundtrip_preserves_catalog_profile_and_subclass(toy, monkeypatch, tmp_path, with_saved_profile):
    from llm_local_001.model import artifact_hashes
    policy = toy()
    def save(self, destination):
        destination = Path(destination)
        destination.mkdir()
        (destination / 'weights.safetensors').write_bytes(b'CPU fixture')
        result = {'config': {'model_path': str(destination), 'temperature': .8},
                  'parameters': {'training_scope': 'base_weights'}, 'base_artifacts_sha256': {},
                  'artifacts_sha256': artifact_hashes(destination)}
        (destination / 'local_checkpoint.json').write_text(json.dumps(result))
        return result
    monkeypatch.setattr(LocalPolicy, 'save_checkpoint', save)
    target = tmp_path / 'checkpoint'
    manifest = policy.save_checkpoint(target)
    if not with_saved_profile:
        manifest.pop('phase_temperature_profile')
        (target / 'local_checkpoint.json').write_text(json.dumps(manifest))
    assert artifact_hashes(target) == manifest['artifacts_sha256']
    loaded = LegalSelectionPolicy.from_checkpoint(target, device='cpu')
    assert isinstance(loaded, LegalSelectionPolicy)
    assert loaded.config.temperature == 1.
    assert loaded.phase_profile() == policy.phase_profile()


def test_scoring_failure_restores_likelihood_mode_without_generating_an_action(toy, monkeypatch):
    policy = toy()
    policy.model.train()
    def failure(*args, **kwargs):
        raise RuntimeError('candidate scoring failed')
    monkeypatch.setattr(SpecLocalPolicy, 'sequence_logprobs', failure)
    with pytest.raises(RuntimeError, match='candidate scoring'):
        policy.generate(prompt(policy), sample=True)
    assert policy.model.training and policy.config.temperature == 1.


def test_tiny_actual_causal_likelihoods_produce_mean_normalized_categorical_distribution(toy, monkeypatch):
    policy = toy()
    torch = policy.torch
    class TinyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.linspace(-.2, .3, 128))
            self.config = SimpleNamespace(max_position_embeddings=20000)
        def forward(self, input_ids, attention_mask, use_cache=False):
            return SimpleNamespace(logits=self.scores[None, None, :].expand(1, input_ids.shape[1], -1))
    policy.model = TinyCausalLM()
    monkeypatch.setattr(SpecLocalPolicy, 'sequence_logprobs', LocalPolicy.sequence_logprobs)
    generated = policy.generate(prompt(policy), sample=False)
    base_logp = torch.log_softmax(policy.model.scores, -1)
    means = torch.stack([base_logp[list(ids)].mean() for ids in policy._completion_catalog()])
    expected = torch.log_softmax(means, -1)
    actual = policy.sequence_logprobs(generated, sampling_distribution=True)
    index = ACTION_CATALOG.index(generated.text)
    torch.testing.assert_close(actual, expected[index].expand_as(actual), rtol=1e-6, atol=1e-7)
    (-actual.mean()).backward()
    assert policy.model.scores.grad is not None and torch.isfinite(policy.model.scores.grad).all()
