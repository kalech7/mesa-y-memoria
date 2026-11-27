"""Categorical LM policies over public actions for a bounded-memory POMDP.

No project SFT checkpoint is required. Candidate scores are mean causal-LM
completion log-probabilities including EOS; their softmax at T=1 defines one
categorical action distribution, not per-token action likelihoods. Training
callers must keep likelihood_mode() active through backward.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llm_local_001.model import Generation
from research.src.policy.llm_controller import SpecLocalPolicy

VERSION = 'budget_memory_policy_v1'
MAX_PROMPT_TOKENS = 1024
MAX_CANDIDATE_TOKENS = 64
TEMPERATURE = 1.0
MANAGER_INSTRUCTION = (
    'Gestiona dos slots de memoria para ayudar en consultas posteriores. '
    'Decide usando solo la observación y las acciones públicas. Preguntar es limitado y costoso. '
    'STORE copia literalmente el intercambio actual al slot elegido; reemplaza su contenido anterior. '
    'NOOP conserva ambos slots. Distingue preferencias del usuario de charla neutral y de propuestas del asistente. '
    'No inventes duración ni contexto. Elige exactamente una acción del catálogo.'
)
READER_INSTRUCTION = (
    'Responde la consulta usando solo los intercambios conservados en los slots. '
    'Respeta el atributo, la fecha, el contexto y las correcciones del usuario. '
    'Una propuesta o pregunta del asistente no confirma una preferencia. '
    'Elige un valor del menú público solo si está sustentado; si falta información elige DEFER.'
)
FORBIDDEN_KEYS = frozenset({
    'truth', 'gold', 'expected_memory', 'authorized_operations', 'latent_state',
    'future_queries', 'future_query', 'optimal_action', 'oracle', 'reward',
    'rewards', 'reward_components', 'world_id', 'episode_id', 'seed', 'family',
})
MANAGER_KEYS = frozenset({'decision_type', 'session_index', 'session_id', 'date',
                          'exchange', 'slots', 'questions_remaining'})
READER_KEYS = frozenset({'slots', 'query', 'answer_menu'})


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _assert_public(value):
    if isinstance(value, dict):
        if FORBIDDEN_KEYS.intersection(value):
            raise ValueError('Private or evaluator field in policy observation')
        for child in value.values():
            _assert_public(child)
    elif isinstance(value, list):
        for child in value:
            _assert_public(child)


class BudgetMemoryPolicy(SpecLocalPolicy):
    """A direct-load Instruct backend; enable_training adds only q/k/v/o LoRA."""

    def __init__(self, model_path, **options):
        options['temperature'] = TEMPERATURE
        if options.get('lora_rank', 16) != 16 or options.get('lora_dropout', 0.) != 0.:
            raise ValueError('Budget-memory LoRA requires rank16 and dropout0')
        super().__init__(model_path, **options)
        self._candidate_cache = {}
        self._used_decision_ids = set()

    def policy_profile(self):
        profile = {
            'version': VERSION, 'temperature': TEMPERATURE,
            'score_normalization': 'mean_completion_token_logprob_including_eos',
            'likelihood_unit': 'categorical_action',
            'single_candidate': 'logprob_zero_no_forward',
            'max_prompt_tokens': MAX_PROMPT_TOKENS, 'max_candidate_tokens': MAX_CANDIDATE_TOKENS,
            'manager_instruction_sha256': hashlib.sha256(MANAGER_INSTRUCTION.encode()).hexdigest(),
            'reader_instruction_sha256': hashlib.sha256(READER_INSTRUCTION.encode()).hexdigest(),
            'candidate_serialization': 'canonical_json_single_field_action_or_answer',
            'lora_rank': 16, 'lora_targets': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        }
        return {**profile, 'profile_sha256': digest(profile)}

    def candidate_token_ids(self, candidates, kind='action'):
        if kind not in {'action', 'answer'}:
            raise ValueError('Candidate kind must be action or answer')
        if (not isinstance(candidates, list) or not candidates
                or any(not isinstance(value, str) or not value for value in candidates)
                or len(set(candidates)) != len(candidates)):
            raise ValueError('Candidates must be unique nonempty public strings')
        key = (kind, tuple(candidates))
        if key not in self._candidate_cache:
            if self.tokenizer.eos_token_id is None:
                raise ValueError('Categorical policy requires EOS')
            rows = []
            for candidate in candidates:
                text = canonical({kind: candidate})
                ids = list(self.tokenizer.encode(text, add_special_tokens=False))
                if not ids or ids[-1] != self.tokenizer.eos_token_id:
                    ids.append(self.tokenizer.eos_token_id)
                if len(ids) > MAX_CANDIDATE_TOKENS:
                    raise ValueError('Candidate token limit exceeded; no truncation')
                if self.tokenizer.decode(ids, skip_special_tokens=True) != text:
                    raise ValueError('Canonical candidate does not roundtrip through tokenizer')
                rows.append(tuple(ids))
            if len(set(rows)) != len(rows):
                raise ValueError('Candidates must have distinct token sequences')
            self._candidate_cache[key] = tuple(rows)
        return [list(ids) for ids in self._candidate_cache[key]]

    def prompt(self, observation, candidates, *, kind='action'):
        if not isinstance(observation, dict):
            raise ValueError('Observation must be a public JSON object')
        allowed = MANAGER_KEYS if kind == 'action' else READER_KEYS if kind == 'answer' else None
        if allowed is None or set(observation) != allowed:
            raise ValueError('Observation does not match its public manager/reader contract')
        _assert_public(observation)
        self.candidate_token_ids(candidates, kind)
        instruction = MANAGER_INSTRUCTION if kind == 'action' else READER_INSTRUCTION
        return instruction + '\nOBSERVACION\n' + canonical(observation) + '\nCATALOGO\n' + canonical(candidates)

    @staticmethod
    def _validate_prompt_ids(prompt_ids):
        if (not isinstance(prompt_ids, list) or not prompt_ids
                or len(prompt_ids) > MAX_PROMPT_TOKENS
                or any(type(value) is not int or value < 0 for value in prompt_ids)):
            raise ValueError('Invalid prompt IDs or input token limit exceeded; no truncation')

    def scores(self, prompt_ids, candidates, kind='action'):
        """Differentiable candidate means; no semantic parsing, masks or repair."""
        self._validate_prompt_ids(prompt_ids)
        values = []
        for candidate, ids in zip(candidates, self.candidate_token_ids(candidates, kind)):
            generated = Generation(list(prompt_ids), ids, canonical({kind: candidate}))
            token_logprobs = SpecLocalPolicy.sequence_logprobs(self, generated, sampling_distribution=False)
            values.append(token_logprobs.mean())
        return self.torch.stack(values)

    def distribution(self, prompt_ids, candidates, kind='action'):
        """One normalized log-probability per action, with autograd intact."""
        self._validate_prompt_ids(prompt_ids)
        self.candidate_token_ids(candidates, kind)
        if len(candidates) == 1:
            return self.torch.zeros(1, device=self.device)
        return self.torch.log_softmax(self.scores(prompt_ids, candidates, kind) / TEMPERATURE, dim=-1)

    def _choose(self, observation, candidates, *, kind, seed, sample, decision_id):
        if not isinstance(decision_id, str) or not decision_id or decision_id in self._used_decision_ids:
            raise ValueError('A unique nonempty decision_id is required; no implicit retries')
        if type(seed) is not int or seed < 0 or type(sample) is not bool:
            raise ValueError('A nonnegative integer seed and boolean sample are required')
        prompt = self.prompt(observation, candidates, kind=kind)
        prompt_ids = self.encode_prompt(prompt)
        if len(prompt_ids) > MAX_PROMPT_TOKENS:
            raise ValueError('INPUT_TOKEN_LIMIT: no silent truncation')
        with self.torch.no_grad(), self.likelihood_mode():
            forced = len(candidates) == 1
            scores = None if forced else self.scores(prompt_ids, candidates, kind)
            logprobs = (self.torch.zeros(1, device=self.device) if forced else
                        self.torch.log_softmax(scores / TEMPERATURE, dim=-1))
            if not bool(self.torch.isfinite(logprobs).all()):
                raise RuntimeError('NONFINITE_CATEGORICAL_DISTRIBUTION')
            if sample:
                # Local generator: sampling one decision does not reset the
                # simulator RNG or the actor RNG when the reader is called.
                generator = self.torch.Generator(device=logprobs.device).manual_seed(seed)
                selected = int(self.torch.multinomial(logprobs.exp(), 1, generator=generator).item())
            else:
                selected = int(logprobs.argmax().item())
        self._used_decision_ids.add(decision_id)
        profile = self.policy_profile()
        metadata = {
            'decision_id': decision_id, 'seed': seed, 'sampled': sample, 'kind': kind,
            'prompt': prompt, 'prompt_ids': list(prompt_ids),
            'candidate_token_ids': self.candidate_token_ids(candidates, kind),
            'candidates': list(candidates), 'action_index': selected,
            'old_logprobs': logprobs.detach().cpu().tolist(),
            'scores': None if forced else scores.detach().cpu().tolist(),
            'forced_action': forced, 'scores_evaluated': not forced,
            'chosen_logprob': float(logprobs[selected].detach().cpu()),
            'profile_sha256': profile['profile_sha256'], 'policy_version': VERSION,
            'likelihood_unit': 'categorical_action', 'temperature': TEMPERATURE,
            'score_normalization': profile['score_normalization'],
            'observation_sha256': digest(observation), 'catalog_sha256': digest({kind: candidates}),
        }
        return {kind: candidates[selected], 'metadata': metadata}

    def decide(self, observation, candidates, *, seed, sample, decision_id):
        return self._choose(observation, candidates, kind='action', seed=seed, sample=sample, decision_id=decision_id)

    def save_checkpoint(self, destination):
        manifest = super().save_checkpoint(destination)
        manifest['budget_memory_profile'] = self.policy_profile()
        # This manifest is excluded from LocalPolicy.artifact_hashes.
        (Path(destination) / 'local_checkpoint.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        return manifest

    @classmethod
    def from_checkpoint(cls, checkpoint, *, trainable=False, device=None):
        manifest = json.loads((Path(checkpoint) / 'local_checkpoint.json').read_text())
        obj = super().from_checkpoint(checkpoint, trainable=trainable, device=device)
        if manifest.get('budget_memory_profile') != obj.policy_profile():
            raise ValueError('Budget-memory checkpoint profile mismatch')
        return obj


class FrozenBudgetReader:
    """A separate frozen Instruct reader; full dialogue/private state are rejected."""

    def __init__(self, backend, value_menus=None):
        if (getattr(backend, 'training_enabled', False) or hasattr(backend.model, 'peft_config')
                or any(parameter.requires_grad for parameter in backend.model.parameters())):
            raise ValueError('Reader must be frozen and have no controller adapters')
        if value_menus is None:
            from research.budget_memory_env import VALUE_MENUS
            value_menus = VALUE_MENUS
        self.backend = backend
        self.value_menus = {attribute: tuple(values) for attribute, values in value_menus.items()}

    @classmethod
    def load(cls, model_path, *, value_menus=None, **options):
        options['lora'] = False
        return cls(BudgetMemoryPolicy(model_path, **options), value_menus=value_menus)

    def answer(self, observation, candidates, *, seed, decision_id):
        if not isinstance(observation, dict) or set(observation) != READER_KEYS:
            raise ValueError('Reader accepts only slots, query and global answer_menu')
        if (not isinstance(observation['query'], dict)
                or set(observation['query']) != {'attribute', 'date', 'context', 'session_id'}):
            raise ValueError('Reader query has nonpublic or missing fields')
        attribute = observation['query']['attribute']
        expected = list(self.value_menus[attribute]) + ['DEFER']
        if candidates != expected or observation['answer_menu'] != expected:
            raise ValueError('Reader requires the constant full public menu for its attribute')
        return self.backend._choose(observation, candidates, kind='answer', seed=seed, sample=False,
                                    decision_id=decision_id)


def verify_budget_profiles(actor, reference):
    if actor.config.temperature != TEMPERATURE or reference.config.temperature != TEMPERATURE:
        raise ValueError('Actor/reference base temperature mismatch')
    if actor.policy_profile() != reference.policy_profile():
        raise ValueError('Actor/reference categorical profiles differ')
    return actor.policy_profile()


def verify_fresh_lora_zero_delta(policy):
    """Check the zero-B LoRA initialization contract, without a fake SFT origin.

    This is a structural zero-delta check. A separate numerical logits comparison
    is still appropriate in the preflight for the actual backbone/precision.
    """
    if policy.origin_checkpoint is not None or not policy.training_enabled:
        raise ValueError('Expected a fresh direct-load trainable LoRA actor')
    factors = [(name, parameter) for name, parameter in policy.model.named_parameters() if 'lora_B' in name]
    if not factors or any(bool(parameter.detach().ne(0).any()) for _, parameter in factors):
        raise ValueError('Fresh LoRA B must be exactly zero')
    frozen_base = [parameter for name, parameter in policy.model.named_parameters() if 'lora_' not in name]
    if any(parameter.requires_grad for parameter in frozen_base):
        raise ValueError('Fresh actor must keep all base parameters frozen')
    return {'origin_checkpoint': None, 'zero_lora_b_factors': len(factors),
            'initial_delta_is_zero': True, 'new_sft_updates': 0}
