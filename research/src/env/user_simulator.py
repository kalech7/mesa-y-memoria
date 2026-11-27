"""A deterministic response table; no method, reward or correctness arguments."""
from copy import deepcopy
import hashlib

from .imperfections import normalize_config, uniform, supported_context_fragment

QUESTIONS = {
    "TEMPORAL_SCOPE": "¿Lo dices solo para esta ocasión o como preferencia habitual?",
    "CURRENT_CONTEXT": "¿En qué contexto quieres que aplique la recomendación?",
    "VALID_UNTIL": "¿Hasta qué fecha se aplica esa excepción?",
    "GENERIC": "¿Puedes darme más detalles?",
}
NON_RESPONSE = {"turn_id": "u3", "text": "No puedo precisar ese dato ahora."}


def response_availability(value):
    """Audit delivery status from public response content, never latent truth."""
    text = " ".join(value["text"].casefold().split())
    neutral = text.startswith(("no tengo más detalles", "no puedo precisar", "no sé precisar",
                               "sobre ese punto no tengo información adicional", "no sé si", "no se si",
                               "no estoy seguro", "no estoy segura", "no he decidido",
                               "tal vez", "quizá", "quizas", "quizás", "podría", "puede que"))
    if neutral:
        return "unavailable"
    explicit = value.get("availability")
    if explicit is not None and explicit not in {"available", "unavailable"}:
        raise ValueError("INVALID_RESPONSE_AVAILABILITY")
    return explicit or "available"


def nonresponse_uniform(root_id, slot):
    """Shared nested assignment: method, branch order and rate never enter it."""
    return int.from_bytes(hashlib.sha256(f"7331:{root_id}:{slot}".encode()).digest()[:8], "big") / 2**64


class UserSimulator:
    def __init__(self, replies, *, root_id="", nonresponse_rate=0.0, imperfections=None):
        if not 0 <= nonresponse_rate <= 1:
            raise ValueError("INVALID_NONRESPONSE_RATE")
        self._replies = deepcopy(replies)
        self._root_id = root_id
        self.nonresponse_rate = nonresponse_rate
        self.imperfections = normalize_config(imperfections)
        self.last_response_status = "not_asked"
        self.last_reply_audit = None

    def reply(self, slot):
        if slot not in QUESTIONS:
            raise ValueError("INVALID_SLOT")
        draw = nonresponse_uniform(self._root_id, slot)
        if draw < self.nonresponse_rate:
            self.last_response_status = "unavailable"
            self.last_reply_audit = {"requested_slot": slot, "delivery": "unavailable",
                                     "reason": "legacy_nonresponse"}
            return deepcopy(NON_RESPONSE)
        value = self._replies.get(slot, NON_RESPONSE)
        if isinstance(value, str):
            value = {"turn_id": "u3", "text": value}
        if self.imperfections.active:
            draw = uniform(self.imperfections, self._root_id, "reply", slot)
            config = self.imperfections
            if draw < config.unavailable_reply_rate:
                self.last_response_status = "unavailable"
                self.last_reply_audit = {"requested_slot": slot, "delivery": "unavailable",
                                         "reason": "configured_unavailable"}
                return {"turn_id": value["turn_id"], "text": NON_RESPONSE["text"]}
            if draw < config.unavailable_reply_rate + config.partial_reply_rate:
                fragment, source = supported_context_fragment(self._replies, slot, value)
                self.last_response_status = "partial" if fragment else "unavailable"
                self.last_reply_audit = {"requested_slot": slot, "delivery": self.last_response_status,
                                         "reason": "supported_context_only" if fragment else "no_safe_partial_fragment",
                                         "source_slot": source}
                return {"turn_id": value["turn_id"], "text": fragment or NON_RESPONSE["text"]}
        # Availability belongs to the simulator audit, not an inferred hidden
        # answer. Neutral entries in the existing deterministic table are explicit.
        self.last_response_status = response_availability(value)
        self.last_reply_audit = {"requested_slot": slot, "delivery": self.last_response_status,
                                 "reason": "original_table"}
        # Even a malicious response fixture cannot put hidden keys into a tool result.
        return {"turn_id": value["turn_id"], "text": value["text"]}

    def clear(self):
        self._replies.clear()
        self._root_id = ""
        self.last_response_status = "cleared"
        self.last_reply_audit = None
