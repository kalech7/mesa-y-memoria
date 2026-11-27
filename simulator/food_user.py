"""A private synthetic user, isolated from the preference decision engine.

Only explicitly requested available slots are communicated. No implicit preference
or global scope is supplied. This is a controlled protocol, not a general dialogue
model and not a probabilistic information-gain estimator.
"""
import re
from agents.policies import normalize


class SyntheticFoodUser:
    def __init__(self, private_profile):
        self._answers = dict(private_profile.get("clarification_responses", {}))
        self._communicated = set()

    def respond(self, question):
        text = normalize(question)
        requested = []
        for slot, pattern in (
            ("day_scope", r"dia|semana|laboral|laborable"),
            ("meal_scope", r"tipo de comida|desayuno|almuerzo|cena|merienda"),
            ("source", r"fuente|pedido|quien|otra persona"),
            ("stability", r"estable|oscil|cambi|actualmente"),
        ):
            if re.search(pattern, text) and slot in self._answers:
                requested.append(slot)
        if not requested:
            return {"text": "No tengo ese dato disponible; precisa qué información necesitas.",
                    "communicated_slots": [], "new_information": False}
        answers = [self._answers[slot]["text"] for slot in requested]
        new = bool(set(requested) - self._communicated)
        self._communicated.update(requested)
        return {"text": " ".join(answers), "communicated_slots": requested, "new_information": new}
