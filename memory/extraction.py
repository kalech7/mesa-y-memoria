import re
import unicodedata
from agents.client import TechnicalError

EXTRACTION = """Extrae memorias del mensaje pasado sin inferir el futuro.
Conserva literalmente negaciones, persona, condiciones, ámbito, fechas y citas de terceros.
No reconcilies hechos incompatibles. No conviertas planes hipotéticos ni citas en preferencias propias.
Responde SOLO JSON {"memories": ["texto fiel", ...]}. Puedes devolver una lista vacía."""


def negation_loss(raw, fragments):
    # Alarma superficial, no prueba de equivalencia semántica.
    normalize = lambda s: "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")
    cues = r"\b(no|sin|nunca|nada|evita|evitar|quita|quites)\b"
    return bool(re.search(cues, normalize(raw))) and not bool(re.search(cues, normalize(" ".join(fragments))))


class Extractor:
    def __init__(self, client):
        self.client = client

    def extract(self, message, prior_context=""):
        obj, trace = self.client.complete([{"role": "system", "content": EXTRACTION}, {"role": "user", "content":
            f"Fecha: {message['at']}\nFuente: {message['source']}\nContexto anterior: {prior_context}\nMensaje: {message['text']}"}])
        values = obj.get("memories")
        if not isinstance(values, list) or not all(isinstance(value, str) and value.strip() for value in values):
            raise TechnicalError("extraction_schema_error", trace)
        return values, {"negation_loss_alarm": negation_loss(message["text"], values), "dropped": not values}
