"""A hierarchical RL controller on fixed provider-LLM contextual features.

Only the FP32 MLP head is trainable. The Instruct transformer is a frozen feature
encoder and its vocabulary projection is never called. This is controller RL,
not LoRA/SFT or fine-tuning of the language model. Joint sampling is exact;
greedy evaluation follows gate argmax then conditional argmax (negative ties).
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from research.budget_memory_policy import MANAGER_KEYS, _assert_public, canonical, digest
from research.src.policy.llm_controller import SpecLocalPolicy

VERSION = 'decision_memory_policy_v1'
ASK_ACTIONS = ('PROCEED', 'ASK_TEMPORAL_SCOPE', 'ASK_CURRENT_CONTEXT')
WRITE_ACTIONS = ('NOOP', 'STORE_SLOT_1', 'STORE_SLOT_2')
MAX_PROMPT_TOKENS = 1024
STRUCTURAL_FEATURES = ('phase_ask', 'phase_write', 'questions_remaining', 'session_index_over_3',
                       'slot1_empty', 'slot2_empty', 'slot1_age_sessions_over_3', 'slot2_age_sessions_over_3')
INSTRUCTION = (
    'Gestiona dos slots de memoria para ayudar en consultas posteriores. '
    'Hay un presupuesto total de una pregunta y cuatro sesiones. '
    'PROCEED continúa; ASK_TEMPORAL_SCOPE pregunta por duración; ASK_CURRENT_CONTEXT pregunta por lugar. '
    'NOOP conserva los slots; STORE_SLOT_1 y STORE_SLOT_2 copian todo el intercambio actual al slot indicado, reemplazando su contenido. '
    'Observa únicamente los datos públicos siguientes.'
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _catalogue(candidates):
    if (not isinstance(candidates, list) or any(not isinstance(value, str) for value in candidates)
            or len(set(candidates)) != len(candidates)):
        raise ValueError('Candidates must be a unique public catalogue')
    if candidates == ['PROCEED']:
        return 'ask', ASK_ACTIONS, True
    if len(candidates) == 3 and set(candidates) == set(ASK_ACTIONS):
        return 'ask', ASK_ACTIONS, False
    if len(candidates) == 3 and set(candidates) == set(WRITE_ACTIONS):
        return 'write', WRITE_ACTIONS, False
    raise ValueError('Expected the complete ASK or WRITE catalogue, or budget-forced PROCEED')


def render_observation(observation):
    if not isinstance(observation, dict) or set(observation) != MANAGER_KEYS:
        raise ValueError('Manager observation violates its public contract')
    _assert_public(observation)
    phase = observation['decision_type']
    if phase not in {'ask', 'write'}:
        raise ValueError('Unknown public decision type')
    lines = [INSTRUCTION, f"Decisión: {phase}. Sesión: {observation['session_id']}; fecha: {observation['date']}.",
             f"Índice de sesión: {observation['session_index']}; preguntas restantes: {observation['questions_remaining']}.",
             'INTERCAMBIO ACTUAL:']
    for turn in observation['exchange']:
        lines.append(f"{turn['role']} [{turn['turn_id']}]: {turn['text']}")
    for slot in observation['slots']:
        lines.append(f"SLOT {slot['slot_id']}; origen {slot['source_session']}; fecha {slot['stored_at']}:")
        for turn in slot['exchange']:
            lines.append(f"{turn['role']} [{turn['turn_id']}]: {turn['text']}")
    lines.append('ACCIONES PÚBLICAS: ' + canonical(list(ASK_ACTIONS if phase == 'ask' else WRITE_ACTIONS)))
    return '\n'.join(lines)


def structural_features(observation):
    """Mechanical public capacity/time fields; no keyword or preference rules."""
    phase, index, remaining = (observation[key] for key in ('decision_type', 'session_index', 'questions_remaining'))
    if phase not in {'ask', 'write'} or type(index) is not int or not 0 <= index <= 3 or (type(remaining) is not int or remaining not in (0, 1)):
        raise ValueError('Invalid public phase, session or question budget')
    slots = {slot['slot_id']: slot for slot in observation['slots']}
    if len(observation['slots']) != 2 or set(slots) != {1, 2}:
        raise ValueError('Exactly two public memory slots required')
    empty, ages = [], []
    for number in (1, 2):
        slot = slots[number]
        empty.append(float(not slot['exchange']))
        source = slot['source_session']
        if source is None:
            ages.append(0.)
        elif type(source) is int and 0 <= source <= index:
            ages.append((index - source) / 3.)
        else:
            raise ValueError('Invalid public slot source session')
    return [float(phase == 'ask'), float(phase == 'write'), float(remaining), index / 3., *empty, *ages]


def joint_distribution(logits, candidates):
    """Gate and conditional factors, aligned to the caller's full catalogue."""
    import torch
    family, canonical_actions, forced = _catalogue(candidates)
    if logits.ndim != 1 or logits.numel() != 8 or not bool(torch.isfinite(logits).all()):
        raise ValueError('The hierarchical head must return exactly eight logits')
    offset = 0 if family == 'ask' else 4
    gate = logits[offset:offset+2].log_softmax(-1)
    conditional = logits[offset+2:offset+4].log_softmax(-1)
    if forced:
        joint = logits[:1] * 0.  # Exact forced logp zero, without a semantic mask.
    else:
        canonical_joint = torch.stack((gate[0], gate[1]+conditional[0], gate[1]+conditional[1]))
        joint = torch.stack([canonical_joint[canonical_actions.index(action)] for action in candidates])
    return {'joint_logprobs': joint, 'gate_logprobs': gate, 'conditional_logprobs': conditional,
            'family': family, 'canonical_actions': canonical_actions, 'forced_action': forced}


class DecisionMemoryPolicy:
    """MLP controller; .model/.head are small, .encoder_backend stays frozen."""

    def __init__(self, model_path=None, *, device='cpu', precision='bfloat16', seed=101,
                 hidden_dim=64, trainable=False, encoder_backend=None):
        if hidden_dim != 64:
            raise ValueError('This prospective architecture is fixed at 64 hidden units')
        self.encoder_backend = encoder_backend or SpecLocalPolicy(
            model_path, device=device, precision=precision, lora=False, seed=seed,
            gradient_checkpointing=False)
        encoder = self.encoder_backend
        if (getattr(encoder, 'origin_checkpoint', None) is not None or hasattr(encoder.model, 'peft_config')
                or any(parameter.requires_grad for parameter in encoder.model.parameters())):
            raise ValueError('Encoder must be a direct-load frozen provider model without adapters')
        self.torch = encoder.torch
        self.encoder_hidden_size = int(encoder.model.config.hidden_size)
        self.feature_size = self.encoder_hidden_size + len(STRUCTURAL_FEATURES)
        self.hidden_dim, self.seed = hidden_dim, seed
        self.config = SimpleNamespace(model_path=str(Path(encoder.config.model_path).resolve()),
                                      temperature=1., seed=seed, lora=False)
        # Initialize on CPU under a saved CPU RNG state; no simulator/CUDA RNG reset.
        with self.torch.random.fork_rng(devices=[]):
            self.torch.random.default_generator.manual_seed(seed)
            self.head = self.torch.nn.Sequential(self.torch.nn.Linear(self.feature_size, hidden_dim),
                                                self.torch.nn.Tanh(), self.torch.nn.Linear(hidden_dim, 8))
            self.torch.nn.init.zeros_(self.head[-1].weight)
            self.torch.nn.init.zeros_(self.head[-1].bias)
        self.head = self.head.float().to(device)
        self.model = self.head
        self.training_enabled = bool(trainable)
        self.head.requires_grad_(self.training_enabled)
        self.origin_checkpoint = None
        self._feature_cache, self._used_decision_ids = {}, set()
        self._base_identity = {'model_path': self.config.model_path,
                               'model_type': encoder.model.config.model_type,
                               'hidden_size': self.encoder_hidden_size,
                               'precision': encoder.config.precision,
                               'artifacts_sha256': encoder.base_fingerprint()}

    @property
    def device(self):
        return next(self.head.parameters()).device

    def enable_training(self):
        self.training_enabled = True
        self.head.requires_grad_(True)

    @contextmanager
    def likelihood_mode(self):
        previous = self.head.training
        self.head.eval()
        try:
            yield
        finally:
            self.head.train(previous)

    def policy_profile(self):
        profile = {'version': VERSION, 'trainable_scope': 'contextual_RL_head_only; provider_LLM_frozen',
                   'encoder': deepcopy(self._base_identity), 'encoder_body': 'encoder_backend.model.base_model',
                   'feature_pooling': 'last_prompt_token_then_fixed_layer_norm_float32',
                   'feature_size': self.feature_size, 'structural_features': list(STRUCTURAL_FEATURES),
                   'head': {'hidden_dim': self.hidden_dim, 'activation': 'tanh', 'output_logits': 8, 'dtype': 'float32',
                            'initial_output': 'all zeros; gates .5, conditionals .5'},
                   'temperature': 1., 'sampling': 'exact joint categorical gate-times-conditional',
                   'greedy': 'hierarchical gate argmax then conditional argmax; negative gate wins ties',
                   'catalogues': {'ask': list(ASK_ACTIONS), 'write': list(WRITE_ACTIONS)},
                   'max_prompt_tokens': MAX_PROMPT_TOKENS,
                   'render_sha256': hashlib.sha256(inspect.getsource(render_observation).encode()).hexdigest(),
                   'instruction_sha256': hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
                   'structural_sha256': hashlib.sha256(inspect.getsource(structural_features).encode()).hexdigest(),
                   'joint_distribution_sha256': hashlib.sha256(inspect.getsource(joint_distribution).encode()).hexdigest(),
                   'feature_implementation_sha256': hashlib.sha256(inspect.getsource(self.feature).encode()).hexdigest(),
                   'decoder_implementation_sha256': hashlib.sha256(inspect.getsource(self.decide).encode()).hexdigest()}
        return {**profile, 'profile_sha256': digest(profile)}

    def feature(self, observation):
        prompt = render_observation(observation)
        public = structural_features(observation)
        key = hashlib.sha256(prompt.encode()).hexdigest()
        prompt_ids = self.encoder_backend.encode_prompt(prompt)
        if not prompt_ids or len(prompt_ids) > MAX_PROMPT_TOKENS:
            raise ValueError('INPUT_TOKEN_LIMIT: no truncation')
        cached = key in self._feature_cache
        if not cached:
            encoder = self.encoder_backend
            encoder.model.eval()
            tokens = self.torch.tensor([prompt_ids], device=encoder.device)
            with self.torch.no_grad():
                hidden = encoder.model.base_model(input_ids=tokens, attention_mask=self.torch.ones_like(tokens),
                                                   use_cache=False, return_dict=True).last_hidden_state[0, -1].float()
                normalized = self.torch.nn.functional.layer_norm(hidden, (self.encoder_hidden_size,))
            if normalized.numel() != self.encoder_hidden_size or not bool(self.torch.isfinite(normalized).all()):
                raise ValueError('Invalid frozen encoder feature')
            self._feature_cache[key] = tuple(normalized.cpu().tolist())
        vector = list(self._feature_cache[key]) + public
        return {'feature_vector': vector, 'prompt': prompt, 'prompt_ids': list(prompt_ids),
                'prompt_sha256': key, 'feature_sha256': digest(vector), 'encoder_cache_hit': cached}

    def _head_logits(self, features, head=None):
        head = self.head if head is None else head
        parameter = next(head.parameters())
        vector = self.torch.as_tensor(features, dtype=self.torch.float32, device=parameter.device).detach()
        if vector.ndim != 1 or vector.numel() != self.feature_size or not bool(self.torch.isfinite(vector).all()):
            raise ValueError('Stored public feature vector has wrong dimension or nonfinite values')
        return head(vector)

    def hierarchy(self, features, candidates, *, head=None):
        return joint_distribution(self._head_logits(features, head), candidates)

    def distribution(self, features, candidates, *, head=None):
        return self.hierarchy(features, candidates, head=head)['joint_logprobs']

    def decide(self, observation, candidates, *, seed, sample, decision_id):
        family, actions, forced = _catalogue(candidates)
        if family != observation.get('decision_type'):
            raise ValueError('Public catalogue does not match public decision type')
        if family == 'ask' and forced != (observation.get('questions_remaining') == 0):
            raise ValueError('Only an exhausted public question budget permits singleton PROCEED')
        if not isinstance(decision_id, str) or not decision_id or decision_id in self._used_decision_ids:
            raise ValueError('Every session/decision needs a unique nonempty identity')
        if type(seed) is not int or seed < 0 or type(sample) is not bool:
            raise ValueError('Invalid decision seed or sampling flag')
        record = self.feature(observation)
        with self.torch.no_grad(), self.likelihood_mode():
            hierarchy = self.hierarchy(record['feature_vector'], candidates)
            logprobs = hierarchy['joint_logprobs']
            if forced:
                index = 0
            elif sample:
                generator = self.torch.Generator(device=logprobs.device).manual_seed(seed)
                index = int(self.torch.multinomial(logprobs.exp(), 1, generator=generator).item())
            else:
                # Joint MAP would prefer NOOP until p(STORE) exceeds ~2/3 when
                # slots are uncertain. Evaluate the explicit gate decision first.
                gate = int(hierarchy['gate_logprobs'].argmax().item())
                branch = int(hierarchy['conditional_logprobs'].argmax().item())
                action = actions[0] if gate == 0 else actions[1+branch]
                index = candidates.index(action)
        chosen = candidates[index]
        positive = chosen != actions[0]
        gate_index = int(positive)
        conditional_index = actions.index(chosen)-1 if positive else None
        metadata = {**record, 'decision_id': decision_id, 'seed': seed, 'sampled': sample,
                    'candidates': list(candidates), 'action_index': index,
                    'old_logprobs': logprobs.cpu().tolist(), 'probabilities': logprobs.exp().cpu().tolist(),
                    'gate_logprobs': hierarchy['gate_logprobs'].cpu().tolist(),
                    'conditional_logprobs': hierarchy['conditional_logprobs'].cpu().tolist(),
                    'chosen_gate_logprob': 0. if forced else float(hierarchy['gate_logprobs'][gate_index].cpu()),
                    'chosen_conditional_logprob': float(hierarchy['conditional_logprobs'][conditional_index].cpu()) if positive else 0.,
                    'forced_action': forced, 'policy_version': VERSION,
                    'profile_sha256': self.policy_profile()['profile_sha256'],
                    'likelihood_unit': 'hierarchical_joint_action', 'controller_only_RL': True}
        self._used_decision_ids.add(decision_id)
        return {'action': chosen, 'metadata': metadata}

    def credit_parameter_fingerprint(self):
        """Small controller-only snapshot for counterfactual replay immutability."""
        hasher = hashlib.sha256()
        for name, value in sorted(self.head.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            hasher.update(name.encode())
            hasher.update(str(tensor.dtype).encode())
            hasher.update(str(tuple(tensor.shape)).encode())
            hasher.update(tensor.view(self.torch.uint8).numpy().tobytes())
        return hasher.hexdigest()

    def save_checkpoint(self, destination):
        from safetensors.torch import save_file
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=False)
        head_file = target/'head.safetensors'
        save_file({name: value.detach().cpu().contiguous() for name,value in self.head.state_dict().items()}, str(head_file))
        manifest = {'version': VERSION, 'created_without_SFT': True, 'origin_checkpoint': self.origin_checkpoint,
                    'seed': self.seed, 'hidden_dim': self.hidden_dim, 'base_identity': deepcopy(self._base_identity),
                    'profile': self.policy_profile(), 'head_sha256': _sha(head_file)}
        (target/'decision_memory_checkpoint.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        return manifest

    @classmethod
    def from_checkpoint(cls, checkpoint, *, device='cpu', trainable=False, encoder_backend=None):
        from safetensors.torch import load_file
        path = Path(checkpoint)
        manifest = json.loads((path/'decision_memory_checkpoint.json').read_text())
        if manifest['version'] != VERSION or _sha(path/'head.safetensors') != manifest['head_sha256']:
            raise ValueError('Controller checkpoint version or head checksum mismatch')
        identity = manifest['base_identity']
        obj = cls(identity['model_path'], device=device, precision=identity['precision'], seed=manifest['seed'],
                  hidden_dim=manifest['hidden_dim'], trainable=trainable, encoder_backend=encoder_backend)
        if obj._base_identity != identity or obj.policy_profile() != manifest['profile']:
            raise ValueError('Frozen encoder or feature/render profile changed')
        obj.head.load_state_dict(load_file(str(path/'head.safetensors'), device=str(obj.device)), strict=True)
        obj.origin_checkpoint = str(path.resolve())
        return obj
