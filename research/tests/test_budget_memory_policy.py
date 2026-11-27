"""CPU contracts for direct-load categorical memory manager and frozen reader."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_local_001.model import LocalPolicy, artifact_hashes
from research.budget_memory_policy import (
    BudgetMemoryPolicy, FrozenBudgetReader, MAX_PROMPT_TOKENS, canonical,
    verify_budget_profiles, verify_fresh_lora_zero_delta,
)
from research.src.policy.llm_controller import SpecLocalPolicy

ACTIONS = ['NOOP', 'STORE_SLOT_1', 'STORE_SLOT_2']


def observation():
    return {'decision_type': 'write', 'session_index': 0, 'session_id': 'S1',
            'date': '2026-01-01', 'exchange': [{'turn_id': 'u1', 'role': 'user', 'text': 'Prefiero agua.'}],
            'slots': [], 'questions_remaining': 1}


class CharacterTokenizer:
    eos_token_id = 0
    chat_template = None
    def encode(self, text, **kwargs):
        return list(map(ord, text))
    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(value) for value in ids if value != 0 or not skip_special_tokens)


@pytest.fixture
def factory(monkeypatch):
    torch = pytest.importorskip('torch')
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.linspace(-.2, .3, 128))
            self.config = SimpleNamespace(max_position_embeddings=2048)
            self.forward_calls = 0
        def forward(self, input_ids, attention_mask, use_cache=False):
            self.forward_calls += 1
            return SimpleNamespace(logits=self.scores[None, None, :].expand(1, input_ids.shape[1], -1))
    def init(self, model_path, **options):
        self.config = SimpleNamespace(model_path=str(model_path), temperature=options['temperature'],
                                      gradient_checkpointing=False, seed=101)
        self.torch, self.tokenizer, self.model = torch, CharacterTokenizer(), TinyModel()
        self.model.requires_grad_(options.get('lora', True))
        self.training_enabled, self.origin_checkpoint = False, None
    monkeypatch.setattr(SpecLocalPolicy, '__init__', init)
    return lambda **options: BudgetMemoryPolicy('/unused/cpu-fixture', **options)


def test_distribution_is_normalized_mean_actual_token_logprob_including_eos(factory):
    policy = factory()
    ids = policy.encode_prompt(policy.prompt(observation(), ACTIONS))
    means = policy.scores(ids, ACTIONS)
    logprobs = policy.distribution(ids, ACTIONS)
    base_logp = policy.torch.log_softmax(policy.model.scores, -1)
    expected_means = policy.torch.stack([base_logp[tokens].mean() for tokens in policy.candidate_token_ids(ACTIONS)])
    policy.torch.testing.assert_close(means, expected_means, rtol=1e-6, atol=1e-7)
    policy.torch.testing.assert_close(logprobs, policy.torch.log_softmax(expected_means, -1), rtol=1e-6, atol=1e-7)
    assert float(logprobs.exp().sum().detach()) == pytest.approx(1.)
    (-logprobs[1]).backward()
    assert policy.model.scores.grad is not None and policy.torch.isfinite(policy.model.scores.grad).all()


def test_decide_metadata_matches_differentiable_distribution_and_omits_ids_from_prompt(factory):
    policy = factory()
    chosen = policy.decide(observation(), ACTIONS, seed=101, sample=False, decision_id='EP-PRIVATE/101/S1/write/1')
    meta = chosen['metadata']
    assert set(chosen) == {'action', 'metadata'}
    assert chosen['action'] == ACTIONS[meta['action_index']]
    actual = policy.distribution(meta['prompt_ids'], ACTIONS)
    policy.torch.testing.assert_close(actual.detach().cpu(), policy.torch.tensor(meta['old_logprobs']), rtol=0, atol=0)
    assert 'EP-PRIVATE' not in meta['prompt'] and '101' not in meta['prompt']
    assert meta['candidate_token_ids'] == policy.candidate_token_ids(ACTIONS)
    assert meta['profile_sha256'] == policy.policy_profile()['profile_sha256']
    assert not meta['forced_action'] and meta['scores_evaluated']
    json.dumps(chosen, allow_nan=False)


def test_local_seed_reproducible_without_mutating_global_rng_and_ids_do_not_collide(factory):
    policy = factory()
    before = policy.torch.random.get_rng_state().clone()
    first = policy.decide(observation(), ACTIONS, seed=77, sample=True, decision_id='episode/77/S1/write/1')
    second = policy.decide(observation(), ACTIONS, seed=77, sample=True, decision_id='episode/77/S2/write/3')
    assert first['action'] == second['action']
    assert policy.torch.equal(before, policy.torch.random.get_rng_state())
    assert first['metadata']['decision_id'] != second['metadata']['decision_id']
    with pytest.raises(ValueError, match='unique'):
        policy.decide(observation(), ACTIONS, seed=77, sample=True, decision_id='episode/77/S1/write/1')


@pytest.mark.parametrize('sample', [False, True])
def test_single_proceed_is_exact_zero_no_forward_for_sampling_and_likelihood(factory, sample):
    policy = factory()
    obs = observation()
    obs.update(decision_type='ask', questions_remaining=0)
    result = policy.decide(obs, ['PROCEED'], seed=101, sample=sample, decision_id=f'forced/{sample}')
    meta = result['metadata']
    assert result['action'] == 'PROCEED' and meta['action_index'] == 0
    assert meta['forced_action'] and not meta['scores_evaluated']
    assert meta['old_logprobs'] == [0.] and meta['scores'] is None
    actual = policy.distribution(meta['prompt_ids'], ['PROCEED'])
    assert actual.tolist() == [0.] and not actual.requires_grad
    assert policy.model.forward_calls == 0
    assert meta['candidate_token_ids'] and meta['prompt_ids']


def test_prompt_and_candidate_limits_fail_without_truncation(factory):
    policy = factory()
    obs = observation()
    obs['exchange'][0]['text'] = 'x' * MAX_PROMPT_TOKENS
    with pytest.raises(ValueError, match='TOKEN_LIMIT'):
        policy.decide(obs, ACTIONS, seed=101, sample=False, decision_id='overflow')
    with pytest.raises(ValueError, match='Candidate token limit'):
        policy.candidate_token_ids(['x' * 80])
    assert policy.model.forward_calls == 0


def test_private_fields_and_nonpublic_reader_fields_are_rejected(factory):
    policy = factory()
    for obs in ({**observation(), 'truth': {}}, observation()):
        if 'truth' not in obs:
            obs['exchange'][0]['gold'] = 'hidden'
        with pytest.raises(ValueError):
            policy.prompt(obs, ACTIONS)
    reader = FrozenBudgetReader(factory(lora=False), {'drink_preference': ('agua', 'te')})
    with pytest.raises(ValueError):
        reader.answer({'slots': [], 'query': {}, 'answer_menu': [], 'history': []}, ['DEFER'], seed=1, decision_id='bad')


def test_frozen_reader_accepts_only_reopened_slots_query_and_global_menu(factory):
    backend = factory(lora=False)
    reader = FrozenBudgetReader(backend, {'drink_preference': ('agua', 'te')})
    candidates = ['agua', 'te', 'DEFER']
    obs = {'slots': [], 'query': {'attribute': 'drink_preference', 'date': '2026-01-01',
                                 'context': 'casa', 'session_id': 'S5'}, 'answer_menu': candidates}
    result = reader.answer(obs, candidates, seed=101, decision_id='reader/query1')
    assert result['answer'] in candidates and result['metadata']['kind'] == 'answer'
    assert not result['metadata']['sampled']
    assert all(not p.requires_grad for p in backend.model.parameters())
    with pytest.raises(ValueError, match='full public menu'):
        reader.answer({**obs, 'answer_menu': ['agua', 'DEFER']}, ['agua', 'DEFER'], seed=101, decision_id='reader/bad')
    with pytest.raises(ValueError, match='frozen'):
        FrozenBudgetReader(factory(), {'drink_preference': ('agua',)})


def test_reference_profile_and_logits_match_initial_actor(factory):
    actor, reference = factory(), factory(lora=False)
    assert verify_budget_profiles(actor, reference)
    ids = actor.encode_prompt(actor.prompt(observation(), ACTIONS))
    actor.torch.testing.assert_close(actor.distribution(ids, ACTIONS), reference.distribution(ids, ACTIONS), rtol=0, atol=0)
    profile = actor.policy_profile()
    profile['temperature'] = 99
    assert actor.policy_profile()['temperature'] == 1.
    reference.config.temperature = 2
    with pytest.raises(ValueError, match='temperature'):
        verify_budget_profiles(actor, reference)


def test_checkpoint_roundtrip_preserves_profile_without_artifact_hash_damage(factory, monkeypatch, tmp_path):
    policy = factory()
    def save(self, destination):
        destination = Path(destination)
        destination.mkdir()
        (destination / 'weights.safetensors').write_bytes(b'CPU fixture')
        result = {'config': {'model_path': str(destination), 'temperature': 1},
                  'parameters': {'training_scope': 'base_weights'}, 'base_artifacts_sha256': {},
                  'artifacts_sha256': artifact_hashes(destination)}
        (destination / 'local_checkpoint.json').write_text(json.dumps(result))
        return result
    monkeypatch.setattr(LocalPolicy, 'save_checkpoint', save)
    destination = tmp_path / 'checkpoint'
    manifest = policy.save_checkpoint(destination)
    assert artifact_hashes(destination) == manifest['artifacts_sha256']
    restored = BudgetMemoryPolicy.from_checkpoint(destination, device='cpu')
    assert isinstance(restored, BudgetMemoryPolicy) and restored.policy_profile() == policy.policy_profile()
    manifest['budget_memory_profile']['temperature'] = 99
    (destination / 'local_checkpoint.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='profile mismatch'):
        BudgetMemoryPolicy.from_checkpoint(destination, device='cpu')


def test_fresh_zero_lora_delta_guard_accepts_only_zero_b_and_frozen_base(factory):
    policy = factory()
    policy.training_enabled = True
    policy.model.scores.requires_grad_(False)
    policy.model.register_parameter('lora_B_fixture', policy.torch.nn.Parameter(policy.torch.zeros(2, 2)))
    assert verify_fresh_lora_zero_delta(policy)['initial_delta_is_zero']
    with policy.torch.no_grad():
        policy.model.lora_B_fixture[0, 0] = 1
    with pytest.raises(ValueError, match='exactly zero'):
        verify_fresh_lora_zero_delta(policy)


def test_cpu_tokenizer_direct_from_instruct_has_exact_candidates_without_project_checkpoint():
    transformers = pytest.importorskip('transformers')
    model_root = Path('models/Qwen2.5-0.5B-Instruct')
    if not model_root.is_dir():
        pytest.skip('Local Instruct unavailable; no downloads')
    policy = BudgetMemoryPolicy.__new__(BudgetMemoryPolicy)
    policy.tokenizer = transformers.AutoTokenizer.from_pretrained(model_root, local_files_only=True, trust_remote_code=False)
    policy._candidate_cache = {}
    for kind, values in [('action', ACTIONS), ('answer', ['agua', 'té de menta', 'DEFER'])]:
        for value, ids in zip(values, policy.candidate_token_ids(values, kind)):
            assert ids[-1] == policy.tokenizer.eos_token_id
            assert policy.tokenizer.decode(ids, skip_special_tokens=True) == canonical({kind: value})
    assert len(policy.encode_prompt(policy.prompt(observation(), ACTIONS))) <= MAX_PROMPT_TOKENS


def test_cpu_direct_load_and_fresh_lora_do_not_change_instruct_logits():
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')
    pytest.importorskip('peft')
    model_root = Path('models/Qwen2.5-0.5B-Instruct')
    if not model_root.is_dir():
        pytest.skip('Local Instruct unavailable; no downloads')
    policy = BudgetMemoryPolicy(str(model_root), device='cpu', precision='bfloat16', seed=101)
    assert policy.origin_checkpoint is None and not policy.training_enabled
    assert not any('lora_' in name for name, _ in policy.model.named_parameters())
    assert not any(parameter.requires_grad for parameter in policy.model.parameters())
    ids = policy.tokenizer.encode('Memoria del usuario.', add_special_tokens=False)
    tokens = torch.tensor([ids], device='cpu')
    with torch.no_grad():
        before = policy.model(input_ids=tokens, use_cache=False).logits.detach().clone()
    policy.enable_training()
    evidence = verify_fresh_lora_zero_delta(policy)
    assert evidence['zero_lora_b_factors'] > 0 and policy.config.lora_rank == 16
    with torch.no_grad():
        after = policy.model(input_ids=tokens, use_cache=False).logits.detach()
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert all(parameter.requires_grad == ('lora_' in name)
               for name, parameter in policy.model.named_parameters())
    del policy


def test_cpu_real_tokenizer_public_environment_prompts_fit_without_any_model_generation(tmp_path):
    transformers = pytest.importorskip('transformers')
    from research.budget_memory_data import FAMILIES, make_episode
    from research.budget_memory_env import run_episode
    model_root = Path('models/Qwen2.5-0.5B-Instruct')
    if not model_root.is_dir():
        pytest.skip('Local Instruct unavailable; no downloads')
    policy = BudgetMemoryPolicy.__new__(BudgetMemoryPolicy)
    policy.tokenizer = transformers.AutoTokenizer.from_pretrained(model_root, local_files_only=True, trust_remote_code=False)
    policy._candidate_cache = {}
    lengths = []
    def inspect(obs, candidates, kind):
        text = policy.prompt(obs, candidates, kind=kind)
        lengths.append(len(policy.encode_prompt(text)))
        assert lengths[-1] <= MAX_PROMPT_TOKENS
    class Manager:
        def decide(self, obs, candidates, **kwargs):
            inspect(obs, candidates, 'action')
            action = candidates[-1] if obs['decision_type'] == 'ask' else f"STORE_SLOT_{1 + obs['session_index'] % 2}"
            return {'action': action, 'metadata': {}}
    class Reader:
        def answer(self, obs, candidates, **kwargs):
            inspect(obs, candidates, 'answer')
            return {'answer': 'DEFER', 'metadata': {}}
    for index, family in enumerate(FAMILIES):
        episode = make_episode(family, index, 1234567)
        run_episode(Manager(), Reader(), episode, 101, tmp_path / str(index), sample=False)
    assert len(lengths) == 10 * len(FAMILIES)
