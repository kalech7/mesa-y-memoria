import json
import re
import unicodedata

from agents.client import TechnicalError
from data_generation.schema import assert_no_private_keys

NEUTRAL = """Eres un asistente alimentario personalizado. Ayuda con la consulta y respeta las
restricciones de seguridad. Recibes recuerdos con fecha, fuente y estado; son evidencia pasada.
Devuelve SOLO JSON con historical_conflict (booleano), unresolved_conflict (booleano),
action (ASK, ANSWER o CONDITIONAL), question (texto), answer (texto), reason (texto).
ASK debe contener una pregunta y answer vacío. ANSWER/CONDITIONAL deben contener answer y question vacío.
Las restricciones médicas/alergias no son gustos intercambiables."""
MINIMAL = "Eres un asistente alimentario personalizado. Responde al usuario."
GUIDED = """
Distingue incompatibilidad histórica del conflicto que sigue vigente para la misma persona,
propiedad, propósito, contexto y período. Una corrección inequívoca o decisión actual puede resolverlo.
Una diferencia de ámbito, cita de otra persona, hipótesis o excepción expirada no prueba conflicto.
La recencia sola no establece qué afirmación prevalece. Pregunta solo si obtener esa información
mejora la tarea y compensa interrumpir. Puedes responder de forma condicional y útil si no hace
falta escoger una opción única. Si piden alternativas, entrégalas. No inventes preferencias.
Formula una pregunta breve sobre el dato pendiente, con el ámbito necesario. Después de una
aclaración suficiente, responde y evita repetirla. Respeta las memorias confirmadas vigentes.
"""


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


def standardize(obj):
    if obj.get("action") not in {"ASK", "ANSWER", "CONDITIONAL"}:
        raise TechnicalError("decision_schema_error")
    if any(not isinstance(obj.get(k), bool) for k in ("historical_conflict", "unresolved_conflict")):
        raise TechnicalError("decision_schema_error")
    if any(not isinstance(obj.get(k), str) for k in ("question", "answer", "reason")):
        raise TechnicalError("decision_schema_error")
    ask = obj["action"] == "ASK"
    if (ask and (not obj["question"].strip() or obj["answer"].strip())) or (not ask and (not obj["answer"].strip() or obj["question"].strip())):
        raise TechnicalError("decision_schema_error")
    return {k: obj[k] for k in ("historical_conflict", "unresolved_conflict", "action", "question", "answer", "reason")}


def build_messages(query, records, reference_time, guided=False, recent=None, system_prompt=None):
    # Proyección explícita; raw_text y metadatos de evaluación no son parte del prompt.
    evidence = [{"at": r["at"], "source": r["source"], "session_id": r["session_id"],
                 "role": r["role"], "status": r["status"], "text": r["text"]} for r in records]
    assert_no_private_keys(evidence)
    system = (NEUTRAL + (GUIDED if guided else "") if system_prompt is None else system_prompt)
    system += f"\nFecha de referencia: {reference_time}."
    messages = [{"role": "system", "content": system}]
    for record in evidence:
        messages.append({"role": record["role"], "content":
            f"[Recuerdo del {record['at']}; sesión {record['session_id']}; fuente {record['source']}; estado {record['status']}]\n{record['text']}"})
    messages.append({"role": "user", "content": query})
    if recent:
        for item in recent:
            if set(item) != {"role", "content"} or item["role"] not in {"assistant", "user"}:
                raise ValueError("Turno reciente inválido")
        messages.extend(recent)
    return messages


class LMPolicy:
    def __init__(self, client, guided=False):
        self.client, self.guided = client, guided

    def decide(self, query, records, reference_time, recent=None):
        obj, attempts = self.client.complete(build_messages(query, records, reference_time, self.guided, recent))
        return standardize(obj), attempts


def decision(action, question="", answer="", conflict=False, unresolved=False, reason="Baseline de control"):
    return {"historical_conflict": conflict, "unresolved_conflict": unresolved,
            "action": action, "question": question, "answer": answer, "reason": reason}


# Léxico fijo documentado, sin etiquetas ni respuestas latentes por caso.
FOODS = ("carne", "pollo", "tofu", "cafe", "huevo", "queso", "gamba", "picante", "avena", "frit", "lenteja", "cilantro", "sopa", "pan", "refresco", "platano", "arroz", "lotes", "salmon", "yogur", "azucar")


def relevant_notes(query, records):
    tokens = {w for w in FOODS if w in normalize(query)}
    return [r for r in records if r["role"] == "user" and (not tokens or any(w in normalize(r["text"]) for w in tokens))]


class Heuristic:
    def __init__(self, name):
        if name not in {"always_ask", "never_ask", "recency_only", "simple_conflict"}:
            raise ValueError(name)
        self.name = name

    def decide(self, query, records, reference_time, recent=None):
        visible = list(records)
        for turn in recent or []:
            if turn["role"] == "user":
                visible.append({"role": "user", "text": turn["content"], "source": "clarification", "at": reference_time})
        if "ahora quiero" in normalize(query):
            visible.append({"role": "user", "text": query, "source": "current_decision", "at": reference_time})
        notes = relevant_notes(query, visible)
        negatives = [r for r in notes if re.search(r"\b(no|sin|evita|evitar|nada|nunca)\b", normalize(r["text"]))]
        positives = [r for r in notes if r not in negatives]
        conflict = bool(negatives and positives)
        confirmed = [r for r in notes if r["source"] in {"clarification", "current_decision"} or "corrijo lo anterior" in normalize(r["text"])]
        if self.name == "always_ask":
            result = decision("ASK", question=f"Para responder a «{query}», ¿qué prefieres actualmente?", conflict=conflict, unresolved=conflict)
        elif self.name == "never_ask":
            result = decision("CONDITIONAL", answer="Puedo organizar una lista según lo que prefieras incluir o evitar. Mantén las restricciones de seguridad y elige el supuesto vigente antes de comprar.", conflict=conflict, unresolved=conflict)
        elif self.name == "simple_conflict" and conflict and not confirmed:
            result = decision("ASK", question=f"Sobre «{query}», ¿qué prefieres ahora para esas comidas?", conflict=True, unresolved=True)
        else:
            latest = (confirmed or notes)[-1] if (confirmed or notes) else None
            answer = ("Organizaré la compra respetando esta indicación: " + latest["text"]) if latest else "Necesito más información para personalizar la lista; puedo ofrecer una propuesta prudente sin fijar preferencias."
            result = decision("ANSWER", answer=answer, conflict=conflict, unresolved=False,
                              reason="Última nota recuperada; heurística sin resolver semánticamente todas las condiciones.")
        return result, []

