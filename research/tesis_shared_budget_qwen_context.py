"""Public, non-authoritative evidence annotations for Qwen's shared-budget prompt."""
from __future__ import annotations

import re
from unicodedata import normalize

from research.tesis_interactive_env import public_evidence


PROMPT_FORMAT = "public-context-v2"
ANNOTATION_TURN_ID = "assistant_public_context_v2"
_LIMITED = re.compile(
    r"\b(?:prisa|poco tiempo|sin tiempo|apenas podre|no puedo quedarme|"
    r"no tengo (?:tiempo|(?:un )?rato)|me (?:estan llamando|llaman|tengo que ir)|"
    r"prestar atencion|(?:guarde?|guardar|soltar|suelto) el telefono|"
    r"(?:cierro|cerrar) el chat|(?:entro|entrando) a una? (?:cita|reunion))\b"
)
_AVAILABLE = re.compile(
    r"\b(?:tengo (?:un |algo de )?(?:rato|ratito|momento|minuto|tiempo)(?: libre)?|"
    r"me puedo quedar|me quedo pendiente|puedo (?:leer|contestar|responder)|"
    r"me queda tiempo|sigo (?:aqui|en el chat))\b"
)
_LATER = re.compile(r"\b(?:manana|pasado manana|pr[oó]xim[oa]s?|mas tarde|despues)\b")
_UNCERTAIN = re.compile(
    r"\b(?:depende(?: de)?|en el aire|puede cambiar|no se|no tengo claro|"
    r"me falta saber|me confirmen|por confirmar|cuadramos el horario)\b"
)
_ROUTINE = re.compile(
    r"\b(?:rutina|de costumbre|como siempre|habitual|encaminad[ao]|"
    r"todo (?:listo|organizado|resuelto))\b"
)
_MISSING = {
    "TEMPORAL_SCOPE": "alcance de la preferencia",
    "VALID_UNTIL": "fecha de fin de la preferencia",
    "CURRENT_CONTEXT": "contexto de la consulta",
    "GENERIC": "otro dato del diálogo",
}
_UNKNOWN = "sin indicio claro"


def _plain(text):
    return "".join(letter for letter in normalize("NFKD", text.casefold())
                   if not 0x300 <= ord(letter) <= 0x36F)


def _contrast(text, *, first, second, first_label, second_label):
    first_match, second_match = bool(first.search(text)), bool(second.search(text))
    if first_match == second_match:
        return _UNKNOWN
    return first_label if first_match else second_label


def _text_cues(messages):
    user_text = _plain(" ".join(turn["text"] for turn in messages if turn["role"] == "user"))
    later = " ".join(part for part in re.split(r"[.!?;\n]+", user_text) if _LATER.search(part))
    return (
        _contrast(user_text, first=_LIMITED, second=_AVAILABLE, first_label="limitada",
                  second_label="disponible"),
        _contrast(later, first=_UNCERTAIN, second=_ROUTINE, first_label="incierto",
                  second_label="previsto como rutina"),
    )


def with_public_context(observation):
    """Return a prompt-only observation enriched from the visible dialogue.

    Treat the note as a parser estimate. Scope can be ambiguous even when the
    current meal is clear. Availability and later-plan cues remain unknown
    when unsupported or contradictory. Repeated calls replace the earlier
    note, allowing text augmentations to be re-annotated safely.
    """
    messages = [turn for turn in observation["user_messages"]
                if not (turn.get("turn_id") == ANNOTATION_TURN_ID and turn["role"] == "assistant")]
    visible = {**observation, "user_messages": messages}
    evidence = public_evidence(visible)
    availability, later = _text_cues(messages)
    pending = _MISSING.get(evidence["missing_slot"], "ninguno detectado")
    current = "clara" if evidence["current_resolved"] else "dudosa"
    scope = "ambiguo" if evidence["scope_ambiguous"] else "sin ambigüedad detectada"
    text = (
        f"Resumen automático del diálogo de esta visita. Visita {observation['visit']}, "
        f"preguntas disponibles {observation['questions_remaining']}. "
        f"Elección actual {current}. Alcance temporal {scope}. "
        f"Dato pendiente detectado: {pending}. Disponibilidad para responder ahora: "
        f"{availability}. Plan de una visita posterior: {later}."
    )
    return {**visible, "user_messages": [
        *messages, {"role": "assistant", "turn_id": ANNOTATION_TURN_ID, "text": text},
    ]}
