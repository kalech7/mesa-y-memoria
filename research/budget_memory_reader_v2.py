"""Frozen readable reader selected by interface development on TRAIN only.

The manager, global answer menus, canonical candidate JSON and mean-token+EOS
scoring are inherited unchanged. Only the reader instruction/observation render
changes. The two generic in-context examples tested during calibration are not
included. Profile/checkpoint identity binds both instruction and render source.
"""
from __future__ import annotations

import hashlib
import inspect
from types import MappingProxyType

from research.budget_memory_policy import BudgetMemoryPolicy, FrozenBudgetReader, canonical, digest

VERSION = 'budget_memory_readable_reader_v1'
TOPICS = MappingProxyType({'breakfast_preference': 'desayuno', 'dinner_preference': 'cena', 'drink_preference': 'bebida'})
READER_INSTRUCTION = (
    'Responde exclusivamente con un objeto JSON de la forma {"answer":"valor"}. '
    'El valor debe ser una opción exacta del menú completo, o DEFER. '
    'Contesta únicamente sobre el atributo consultado, usando afirmaciones del usuario. '
    'Aplica su última corrección, su contexto y su vigencia. '
    'Una sugerencia del asistente no confirma un hecho. '
    'Si no se puede saber la preferencia en esa fecha y lugar, responde {"answer":"DEFER"}. '
    'No elijas una opción solo porque aparezca en el menú.'
)


def render_reader_observation(obs):
    """Render every slot and turn verbatim, in supplied order, without extraction."""
    lines = ['CONVERSACIONES CONSERVADAS (sin interpretar ni resumir):']
    for slot in obs['slots']:
        lines.append(f"Memoria {slot['slot_id']}; fecha {slot['stored_at']}; sesión de origen {slot['source_session']}:")
        for turn in slot['exchange']:
            role = 'USUARIO' if turn['role'] == 'user' else 'ASISTENTE'
            lines.append(f"{role} [{turn['turn_id']}]: {turn['text']}")
    query = obs['query']
    lines.extend(['CONSULTA ACTUAL:', f"Tema: {TOPICS[query['attribute']]}. Fecha: {query['date']}. "
                  f"Lugar: {query['context']}. Sesión: {query['session_id']}.",
                  f"¿Qué prefiere el usuario para {TOPICS[query['attribute']]} en esa fecha y lugar?",
                  'MENÚ PÚBLICO COMPLETO:'])
    lines.append(canonical(obs['answer_menu']))
    return '\n'.join(lines)


def reader_template_profile():
    rendering = inspect.getsource(render_reader_observation)
    return {
        'reader_interface_version': VERSION,
        'reader_instruction_sha256': hashlib.sha256(READER_INSTRUCTION.encode()).hexdigest(),
        'reader_render_sha256': hashlib.sha256(rendering.encode()).hexdigest(),
        'reader_template_sha256': digest({'instruction': READER_INSTRUCTION, 'topics': dict(TOPICS), 'render': rendering}),
    }


class ReadableBudgetPolicy(BudgetMemoryPolicy):
    """Reader-only presentation override; inherited action scoring is untouched."""

    def prompt(self, observation, candidates, *, kind='action'):
        original = super().prompt(observation, candidates, kind=kind)
        return (READER_INSTRUCTION + '\n' + render_reader_observation(observation)) if kind == 'answer' else original

    def policy_profile(self):
        profile = super().policy_profile()
        profile.pop('profile_sha256')
        profile.update(reader_template_profile())
        return {**profile, 'profile_sha256': digest(profile)}


class FrozenReadableBudgetReader(FrozenBudgetReader):
    @classmethod
    def load(cls, model_path, *, value_menus=None, **options):
        options['lora'] = False
        return cls(ReadableBudgetPolicy(model_path, **options), value_menus=value_menus)
