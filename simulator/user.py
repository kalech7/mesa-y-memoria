"""Respondedor conservador por intenciones, con estado privado.

Cobertura lingüística limitada: lo no reconocido devuelve uncertain. No se atribuye
validez semántica humana a estas reglas ni se usa el juez para abrir el estado privado.
"""
import re
from dataclasses import dataclass
from agents.policies import normalize
from simulator.preferences import parse_context


@dataclass
class Reply:
    text: str
    question_relevance: str
    information_gain: float | None
    resolved: bool
    slot: str
    duplicate: bool = False
    applicability: str = "none"


def interpret_question(question, context_query, topic_aliases):
    text = normalize(question)
    # La pregunta exacta manda: mencionar un ingrediente no convierte presupuesto en preferencia.
    if re.search(r"\b(hermanas?|hermanos?|amigos?|amigas?|pareja|vecinos?|vecinas?|hijos?|hijas?|madre|padre|familia|invitados?)\b", text):
        return "other_person"
    if re.search(r"\b(alergia|alergico|medico|medica|diagnostico)\b", text):
        return "medical_unknown"
    for pattern, slot in [(r"presupuesto|cuanto.*(?:gastar|costar)|euros|dolares", "budget"),
                          (r"cuanto tiempo|minutos|horario", "time"),
                          (r"que.*(?:tienes|queda).*?(?:casa|nevera|despensa)|ingredientes disponibles", "ingredients")]:
        if re.search(pattern, text):
            return slot
    if "?" not in question and "¿" not in question:
        return "unknown"
    decisive = re.search(r"prefier|preferencia|quieres|deseas|inclu|evit|compro|comprar|tomar|comer|ponemos|ponerte|mantene|confirm|gusta|cocina", text)
    if not decisive:
        return "unknown"
    # Detectar temas ajenos incluso en preguntas que incluyen 'prefieres'.
    target = any(normalize(alias) in text for alias in topic_aliases)
    if not target:
        from agents.policies import FOODS
        if any(food in text for food in FOODS):
            return "other_topic"
        # Solo aceptar elipsis claras y locales; nunca '¿algo más?' ni una frase genérica.
        contextual = any(normalize(alias) in normalize(context_query) for alias in topic_aliases)
        if not contextual or not re.search(r"ahora|actual|esas comidas|para responder|sobre", text):
            return "unknown"
    asked = parse_context(question)
    requested = parse_context(context_query)
    if asked.meal_scope not in (None, "ANY") and requested.meal_scope not in (None, "ANY") and asked.meal_scope != requested.meal_scope:
        return "other_scope"
    if asked.weekdays and requested.weekdays and not set(asked.weekdays) & set(requested.weekdays):
        return "other_scope"
    if asked.unknown_conditions:
        return "unknown"
    return "preference"


class UserSimulator:
    def __init__(self, hidden_state, preference_already_communicated=False):
        self._state = hidden_state
        self.seen = set()
        self.resolved_slots = set()
        self.preference_already_communicated = preference_already_communicated

    def respond(self, question, context_query):
        normalized = normalize(question).strip()
        slot = interpret_question(question, context_query, self._state["topic_aliases"])
        asked_context = parse_context(question)
        occasion = bool(re.search(r"esta vez|esta ocasion|esta cena|esta comida|esta semana|\bhoy\b|para la cena del", normalized))
        # Una respuesta a un día concreto nunca establece un hábito transversal.
        # El estado legacy no contiene preferencias condicionadas por día; no
        # revelarlo si la pregunta introduce un ámbito que aquel no representa.
        state_context = parse_context(self._state.get("scope", ""))
        if slot == "preference" and asked_context.weekdays and not state_context.weekdays:
            return Reply("No tengo una preferencia comunicada específicamente para ese día.", "uncertain", None, False,
                         "unknown_scope", False, "none")
        if asked_context.weekdays:
            occasion = True
        applicability = "occasion" if occasion else "current_scope"
        resolved_key = (slot, applicability)
        duplicate = normalized in self.seen or resolved_key in self.resolved_slots
        self.seen.add(normalized)
        if slot in self._state["available_slots"]:
            return Reply(self._state["available_slots"][slot], "irrelevant", 0.0, False, slot, duplicate)
        if slot != "preference":
            return Reply("No sé qué dato concreto de mis comidas necesitas confirmar. ¿Puedes precisar la pregunta?", "uncertain", None, False, slot, duplicate)
        if self._state["mode"] == "uncertain":
            return Reply("Todavía no lo tengo claro; dame alternativas para poder decidir más adelante.", "relevant", 0.0, False, slot, duplicate)
        answer = (f"Para esta ocasión, prefiero {self._state['preference']}." if occasion else
                  f"Actualmente, para {self._state['scope']} prefiero {self._state['preference']}.")
        self.resolved_slots.add(resolved_key)
        return Reply(answer, "relevant", 0.0 if duplicate or self.preference_already_communicated else 1.0,
                     not duplicate, slot, duplicate, applicability)
