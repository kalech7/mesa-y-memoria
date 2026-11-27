"""Generador de candidatos sintéticos; sus etiquetas NO equivalen a anotación humana.

Los IDs representan biografías generadas separadamente, no sujetos humanos observados.
El holdout comparte vocabulario/plantillas: esta amenaza se declara en el manifiesto.
"""
import calendar
import copy
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from original.scenarios import TOPICS, FILLERS
from data_generation.schema import digest, validate_dataset

ALIASES = {
    "carne": ["carne", "pollo", "vegetarian"], "tofu": ["tofu"],
    "cafe": ["café", "cafe", "cafeína", "cafeina", "descafeinado"],
    "huevos": ["huevo"], "queso": ["queso"], "gambas": ["gamba", "marisco"],
    "picante": ["picante", "pique", "suave"], "avena": ["avena"],
    "fritos": ["frit", "freír", "freir", "freidora"], "lentejas": ["lenteja"],
    "cilantro": ["cilantro"], "sopas": ["sopa", "crema", "cuchara"],
    "pan": ["pan", "integral", "blanco"], "refresco": ["refresco", "agua"],
    "platano": ["plátano", "platano", "batido"], "arroz": ["arroz"],
    "preparacion": ["lotes", "domingo", "cada noche", "cocinar"],
    "pescado": ["salmón", "salmon", "pescado"], "yogur": ["yogur"],
    "postre": ["azúcar", "azucar", "postre", "endulzar"],
}
SETTINGS = [
    "desde que trabajo por turnos", "porque comparto piso y compro por mi cuenta",
    "con el horario nuevo de la biblioteca", "ahora que llevo la comida al taller",
    "cuando vuelvo de dar clases", "mientras preparo las oposiciones",
    "desde que me mudé cerca del mercado", "al organizarme entre trabajo y estudios",
    "porque la cocina de casa es pequeña", "cuando regreso de los ensayos",
    "con las jornadas partidas de la tienda", "para no improvisar después de entrenar",
    "desde que teletrabajo tres días", "al coordinar la compra con mis compañeros",
    "con las clases de la tarde", "porque hago la compra los miércoles",
    "para aprovechar las verduras del huerto", "cuando tengo guardia en recepción",
    "desde que uso una cocina compartida", "con los viajes en autobús al trabajo",
]
WRAPPERS = [
    "Una cosa, {setting}: {statement}",
    "Se me olvidó comentarte esto. {statement} El resto de la compra lo miro luego.",
    "Sobre lo que estuvimos viendo el otro día: {statement}",
    "Hoy lo estaba pensando mientras hacía la lista. {statement}",
    "Apúntame esto cuando puedas: {statement} A veces se me olvida decirlo.",
    "Por cierto, {statement} Me viene bien tenerlo apuntado, {setting}.",
]
ASSISTANTS = [
    "Gracias, lo tendré presente cuando organicemos la compra.",
    "Podemos retomarlo cuando prepares el próximo menú.",
    "Entendido. ¿Quieres que continuemos con el resto de la lista?",
    "Lo anoto con el contexto que me acabas de dar.",
]
VARIANTS = ("open", "open_single", "current", "correction", "alternatives", "single", "scope", "source", "temporary", "hypothetical", "uncertain", "consistent")


def shift_months(date, months):
    value = date.year * 12 + date.month - 1 + months
    year, month = divmod(value, 12)
    return date.replace(year=year, month=month + 1, day=min(date.day, calendar.monthrange(year, month + 1)[1]))


def make_dataset(seed=42, dev_people=20, test_people=80):
    public, gold = [], []
    for index in range(dev_people + test_people):
        split = "dev" if index < dev_people else "test"
        rng = random.Random(f"biography-v2:{seed}:{split}:{index}")
        person = digest(["person-v2", seed, split, index])[:20]
        topic, scope, option_a, option_b, original_a, original_b, query, _, followup = TOPICS[index % len(TOPICS)]
        setting = rng.choice(SETTINGS)
        reference = datetime(2026, 10, rng.randint(12, 25), 12, tzinfo=timezone.utc)
        # Fechas congeladas; no depende de la fecha del equipo.
        start = shift_months(reference, -rng.randint(4, 17))
        session_count = rng.randint(9, 11)
        span = (reference - start).days - 7
        positions = sorted(rng.sample(range(2, session_count - 1), 2))
        first, second = positions
        base = []
        fillers = rng.sample(FILLERS, session_count - 2)
        filler_iter = iter(fillers)
        a_text = rng.choice(WRAPPERS).format(setting=setting, statement=original_a)
        b_text = rng.choice(WRAPPERS).format(setting=setting, statement=original_b)
        for j in range(session_count):
            at = start + timedelta(days=round(j * span / (session_count - 1)), minutes=rng.randint(0, 180))
            text = a_text if j == first else b_text if j == second else next(filler_iter)
            session = digest([person, "session", j])[:16]
            for role, content, delta in [("user", text, 0), ("assistant", rng.choice(ASSISTANTS), 1)]:
                base.append({"message_id": digest([session, role])[:20], "session_id": session,
                    "at": (at + timedelta(minutes=delta)).isoformat(), "role": role,
                    "channel": rng.choice(["chat", "voz", "movil"]), "source": "conversation",
                    "text": content, "context": "Organización personal de comidas y compras."})
        first_id, second_id = base[first * 2]["message_id"], base[second * 2]["message_id"]
        truth = rng.choice([option_a, option_b])
        for variant in VARIANTS:
            history = copy.deepcopy(base)
            current_query = query
            historical, unresolved = True, True
            kind, actions = "unresolved_conflict", ["ASK", "CONDITIONAL"]
            expected, mode = truth, "certain"
            rationale = "Evidencia incompatible, pero recencia por sí sola no prueba vigencia. Se admite una respuesta condicional útil."
            if variant == "open_single":
                current_query = query + " Necesito una única lista que siga mi preferencia actual, no dos alternativas; si falta algún dato decisivo, pregúntamelo."
                actions = ["ASK"]
                rationale = "La tarea exige elegir la preferencia actual y no admite bifurcar la lista; falta evidencia para escoger."
            elif variant == "current":
                current_query = f"Para {scope}, ahora quiero {truth}. {query}"
                kind, actions, unresolved = "historical_conflict", ["ANSWER"], False
                rationale = "La consulta comunica una decisión actual para el ámbito relevante."
            elif variant == "correction":
                message = copy.deepcopy(history[-2])
                message.update(message_id=digest([person, "correction"])[:20], session_id=digest([person, "correction-session"])[:16],
                    at=(reference - timedelta(days=2)).isoformat(),
                    text=f"Corrijo lo anterior sobre {scope}: dejo sin efecto las dos notas previas. De ahora en adelante prefiero {truth}.")
                history.append(message)
                kind, actions, unresolved = "correction", ["ANSWER"], False
                rationale = "Corrección explícita y vigente del propio usuario."
            elif variant == "alternatives":
                current_query = f"Para {scope}, dame ideas en los dos supuestos que comentamos. No hace falta escoger uno todavía."
                actions = ["CONDITIONAL"]
                rationale = "El usuario pide alternativas y no requiere resolver la preferencia."
            elif variant == "single":
                history = [m for m in history if m["session_id"] != base[second * 2]["session_id"]]
                historical, unresolved, expected, kind, actions = False, False, option_a, "no_conflict", ["ANSWER"]
                rationale = "Existe una sola declaración relevante."
            elif variant == "scope":
                for m in history:
                    if m["message_id"] == second_id:
                        m["text"] = f"Solo para la cena de mi club, que preparo para otras personas: {original_b} Esto no describe mis comidas personales."
                historical, unresolved, expected, kind, actions = False, False, option_a, "scope_mismatch", ["ANSWER"]
                rationale = "Las notas describen destinatarios y propósitos diferentes."
            elif variant == "source":
                for m in history:
                    if m["message_id"] == second_id:
                        m["text"] = f"El formulario importado dice sobre mí: «{original_b}». Lo rellenó otra persona; no he confirmado esa información."
                        m["source"] = "user_quoting_imported_form"
                historical, unresolved, kind, actions = False, True, "source_conflict", ["ASK", "CONDITIONAL", "ANSWER"]
                expected = option_a
                rationale = "La fuente externa no confirmada no tiene la misma autoridad que la declaración personal; conservar cautela."
            elif variant == "temporary":
                for m in history:
                    if m["message_id"] == second_id:
                        m["text"] = f"Solo durante esta semana de viaje: {original_b} Después vuelvo a mi rutina anterior."
                historical, unresolved, expected, kind, actions = False, False, option_a, "scope_mismatch", ["ANSWER"]
                rationale = "La excepción temporal ya terminó en la fecha de referencia."
            elif variant == "hypothetical":
                for m in history:
                    if m["message_id"] == second_id:
                        m["text"] = f"Estoy imaginando qué diría un personaje de una receta: «{original_b}». Es un ejemplo inventado, no mi preferencia."
                historical, unresolved, expected, kind, actions = False, False, option_a, "scope_mismatch", ["ANSWER"]
                rationale = "La segunda declaración es hipotética y no atribuible al usuario."
            elif variant == "uncertain":
                kind, mode, actions = "unknown", "uncertain", ["ASK", "CONDITIONAL"]
                rationale = "La persona aún no puede elegir; preguntar no garantiza resolver la incertidumbre."
            elif variant == "consistent":
                for m in history:
                    if m["message_id"] == second_id:
                        m["text"] = f"Sigo con lo que te comenté: {original_a}"
                historical, unresolved, expected, kind, actions = False, False, option_a, "no_conflict", ["ANSWER"]
                rationale = "Dos notas redundantes, semánticamente consistentes."
            case_id = digest(["case-v2", person, variant])[:24]
            # Distractores futuros públicos, sin preferencia confirmada incrustada.
            future = [{**copy.deepcopy(base[0]), "message_id": digest([case_id, "future-noise"])[:20],
                "session_id": digest([case_id, "future-session"])[:16],
                "at": (reference + timedelta(days=1)).isoformat(),
                "text": "Vi ofertas en el mercado y estoy reorganizando la lista; todavía queda comida del domingo."}]
            case = {"case_id": case_id, "person_id": person, "split": split, "synthetic": True,
                "reference_time": reference.isoformat(), "history": history, "query": current_query,
                "followup": followup, "future_distractors": future}
            evidence_ids = [first_id] + ([] if variant == "single" else [second_id])
            evidence = [{"message_id": m["message_id"], "quote": m["text"]} for m in history if m["message_id"] in evidence_ids]
            public.append(case)
            gold.append({"case_id": case_id, "person_id": person, "topic_id": topic, "variant": variant,
                "kind": kind, "scope": scope, "historical_conflict": historical, "unresolved_conflict": unresolved,
                "acceptable_actions": actions, "evidence": evidence, "rationale": rationale,
                "annotation": {"status": "provisional", "reviewer": "", "validated": False},
                "hidden_state": {"preference": expected, "option_a": option_a, "option_b": option_b,
                    "scope": scope, "topic_aliases": ALIASES[topic], "mode": mode,
                    "available_slots": {"budget": "Todavía no he fijado un presupuesto.", "time": "Tengo unos veinte minutos.",
                                        "ingredients": "No tengo la lista de ingredientes a mano."}}})
    validate_dataset(public, gold)
    return public, gold


def write_dataset(root, public, gold):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    private = root / "private"
    private.mkdir(mode=0o700)
    (root / "casos_agent.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in public), encoding="utf8")
    path = private / "casos_gold.jsonl"
    with path.open("x", encoding="utf8") as stream:
        stream.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in gold)
    path.chmod(0o600)
    manifest = {"seeded_synthetic": True, "version": "biography-v2", "public_sha256": digest(public),
        "private_sha256": digest(gold), "people": validate_dataset(public, gold), "cases": len(public),
        "variants_per_person": len(VARIANTS), "human_validated": sum(g["annotation"]["validated"] for g in gold),
        "independence_caveat": "Biografías sintéticas distintas, vocabulario y plantillas compartidos. No demuestran independencia ni realismo humano."}
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    return manifest
