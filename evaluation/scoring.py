import re
from agents.policies import normalize
from agents.client import TechnicalError

JUDGE_PROTOCOL = """Juez auxiliar de fidelidad, NO árbitro definitivo ni evaluador independiente.
Recibirás evidencia privada exclusivamente para evaluar. No generes respuestas para el agente.
Evalúa si la respuesta cumple la consulta y respeta la preferencia comunicada y su ámbito.
No basta con citar palabras ni dar dos alternativas cuando se exige una decisión conocida.
Devuelve SOLO JSON {"verdict":"pass|fail|uncertain", "task_success":"pass|fail|uncertain", "reason":"texto"}.
Si falta evidencia o la respuesta es ambigua, usa uncertain."""


def conservative_fidelity(answer, state):
    """Solo casos inequívocos; el resto requiere juez auxiliar/revisión humana.

    No computa 'éxito' por repetir el gold. Regla limitada documentada, distinta
    de la interpretación del simulador y nunca usada para liberar su estado.
    """
    text = normalize(answer)
    expected = normalize(state["preference"])
    other = normalize(state["option_b"] if state["preference"] == state["option_a"] else state["option_a"])
    if any(x in text for x in ("si prefieres", "en ambos", "dos alternativas", "no se", "elige el supuesto")):
        return "uncertain"
    if expected in text and other in text:
        return "uncertain"
    # Solo afirmaciones directas, no citas ni condiciones.
    if '«' in answer or '"' in answer or "indicacion:" in text or "dijiste" in text:
        return "uncertain"
    direct = re.search(r"(?:te recomiendo|te propongo|preparare|compra|haremos|incluire|evitare|seguire con)\s+(.+)", text)
    if direct and expected in direct.group(1) and other not in text:
        return "pass"
    if direct and other in direct.group(1) and expected not in text:
        return "fail"
    return "uncertain"


def score_answer(answer, state, query, judge=None):
    if judge is None:
        return {"fidelity": conservative_fidelity(answer, state), "task_success": "uncertain",
                "source": "conservative_text_rule", "human_validated": False}
    import json
    obj, _ = judge.complete([{"role": "system", "content": JUDGE_PROTOCOL}, {"role": "user", "content":
        json.dumps({"query": query, "scope": state["scope"], "confirmed_preference": state["preference"], "response": answer}, ensure_ascii=False)}])
    if obj.get("verdict") not in {"pass", "fail", "uncertain"} or obj.get("task_success") not in {"pass", "fail", "uncertain"}:
        raise TechnicalError("judge_schema_error")
    return {"fidelity": obj["verdict"], "task_success": obj["task_success"], "source": "auxiliary_same_family_possible",
            "reason": str(obj.get("reason", "")), "human_validated": False}

