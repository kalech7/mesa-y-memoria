"""Hand-authored Spanish holdout for the three-visit budget experiment."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
import json
from pathlib import Path

from research.src.artifacts import digest, file_hash, write_json
from research.tesis_bank import _remember
from research.tesis_interactive_env import _fact, _prior, _turn, _update_truth


ROOT = Path(__file__).resolve().parent / "data/tesis_extended_heldout_v1"
VERSION = "tesis_extended_heldout_v1"
DAY = "2038-06-"
D = "dinner_preference"
B = "breakfast_preference"
R = "drink_preference"


def day(number):
    return DAY + f"{number:02d}"


@dataclass(frozen=True)
class Preference:
    value: str
    kind: str = "ongoing"
    start: int | None = None
    through: int | None = None
    context: str | None = None


def pref(value, *, kind="ongoing", start=None, through=None, context=None):
    return Preference(value, kind, start, through, context)


def prior(attribute, value, text, *, start=1, kind="ongoing", through=None, context=None):
    return attribute, pref(value, start=start, kind=kind, through=through, context=context), text


def scope_branch(preference, reply):
    return preference, reply


def visit(attribute, when, text, question, answer, *, stored=None, write=None, fact=None,
          context=None, alternatives=(), slot=None, style="direct", partial=None):
    if isinstance(answer, str):
        answer = ("ANSWER", answer)
    return {
        "attribute": attribute, "day": when, "text": text, "question": question, "response": answer,
        "retrieval": stored, "new_write": write, "fact": fact or write, "context": context,
        "alternatives": alternatives, "slot": slot, "style": style, "partial": partial,
    }


S, O, I, C = "session", "ongoing", "interval", "contextual"
DINNER, BREAKFAST, DRINK = "sopa de zapallo", "humita de choclo", "agua de cedrón"


def ambiguity(value, *, first=None, second=None):
    return (
        scope_branch(pref(value, kind=S), first or "Es solo para esta sesión. Luego vuelvo a lo anterior."),
        scope_branch(pref(value, kind=O), second or "Sí, cámbialo a mi preferencia habitual para las próximas veces."),
    )


def common_case(identity, family, history, visits, probes, rationale):
    return {"id": identity, "family": family, "prior": history, "visits": visits,
            "probes": probes, "rationale": rationale}


CASES = (
    common_case("res-01", "partial_and_competing", (
        prior(D, "locro de papa", "Mi cena habitual es locro de papa."),
        prior(B, "pan de yuca", "Por lo general desayuno pan de yuca."),
        prior(R, "agua mineral", "Suelo tomar agua mineral."),
    ), (
        visit(D, 7, "Me van a llamar a caja de un momento a otro. Para la cena, prefiero sopa de zapallo. "
              "Mañana sabré si también me toca organizar los desayunos.",
              "¿Qué dejo listo para la cena de hoy?", DINNER, stored="locro de papa",
              alternatives=ambiguity(DINNER), slot="TEMPORAL_SCOPE", style="refused"),
        visit(B, 8, "El turno cambió y me asignaron el desayuno. Para el desayuno, prefiero humita de choclo.",
              "¿Qué sirvo de desayuno ahora?", BREAKFAST, stored="pan de yuca",
              alternatives=ambiguity(BREAKFAST, second="Pon humita de choclo como mi desayuno habitual."),
              slot="TEMPORAL_SCOPE", style="partial",
              partial="Sé qué me apetece ahora. Me falta confirmar qué haré los demás días."),
        visit(R, 10, "La bebida quedó igual. No necesito cambiar lo que tomo normalmente.",
              "¿Qué bebida era la mía de siempre?", "agua mineral", stored="agua mineral"),
    ), ((D, 14, None), (B, 14, None), (B, 20, None)),
        "Los dos cambios pueden usarse hoy. El primero tropieza con poca disponibilidad. "
        "El segundo puede requerir las dos aclaraciones y afecta dos consultas posteriores."),
    common_case("res-02", "ready_then_delayed", (
        prior(B, "papaya picada", "Anota que normalmente desayuno papaya picada."),
        prior(D, "menestra de fréjol", "Lo habitual para mi cena es menestra de fréjol."),
        prior(R, "jugo de guayaba", "El jugo de guayaba es mi bebida habitual."),
    ), (
        visit(B, 7, "Tengo unos minutos con el chat abierto. Para el desayuno, prefiero avena con canela.",
              "¿Qué preparo para desayunar hoy?", "avena con canela", stored="papaya picada",
              alternatives=ambiguity("avena con canela"), slot="TEMPORAL_SCOPE"),
        visit(D, 8, "Al final sí voy a la reunión y saldré tarde. Para la cena, prefiero verduras al wok.",
              "¿Qué ceno esta noche?", "verduras al wok", stored="menestra de fréjol",
              alternatives=ambiguity("verduras al wok"), slot="TEMPORAL_SCOPE", style="partial",
              partial="Para esta noche está decidido. Para más adelante todavía no puedo fijarlo."),
        visit(R, 10, "Volví a comer en casa. De bebida seguimos como antes.",
              "¿Qué bebida suelo escoger?", "jugo de guayaba", stored="jugo de guayaba"),
    ), ((B, 14, None), (B, 20, None), (D, 14, None)),
        "Una pregunta temprana puede rendir más. La respuesta de la segunda visita requiere un seguimiento "
        "y la batería de consultas reutiliza más veces el primer atributo."),
    common_case("res-03", "context_switch", (
        prior(R, "agua mineral", "Cuando trabajo en la oficina, normalmente tomo agua mineral.",
              kind=C, context="oficina"),
        prior(R, "infusión de menta", "En casa mi bebida de siempre es infusión de menta.",
              kind=C, context="casa"),
    ), (
        visit(R, 7, "Para las jornadas de oficina cambia mi bebida habitual a agua de cedrón. En casa sigue lo de siempre.",
              "¿Qué bebida pongo en mi mesa de la oficina?", DRINK, stored="agua mineral", context="oficina",
              write=pref(DRINK, kind=C, start=7, context="oficina")),
        visit(R, 8, "Cerraron el edificio por mantenimiento. Hoy estoy trabajando desde casa.",
              "¿Qué bebida me toca aquí en casa?", "infusión de menta", stored="infusión de menta", context="casa"),
        visit(R, 10, "Ya abrieron la oficina y volví a mi puesto. Estoy revisando el menú que anotamos.",
              "¿Qué dejo de bebida para la oficina?", DRINK, stored=DRINK, context="oficina"),
    ), ((R, 14, "oficina"), (R, 14, "casa"), (R, 16, "oficina")),
        "La escritura de la primera visita solo sustituye el valor en la oficina. Volver a casa o a la oficina "
        "cambia qué memoria se recupera sin autorizar otra escritura."),
    common_case("res-04", "inclusive_interval", (
        prior(D, "pescado al horno", "Mi opción habitual para cenar es pescado al horno."),
    ), (
        visit(D, 7, "Por la obra no puedo usar el horno. Desde hoy hasta el día 9 inclusive, prefiero locro de papa para la cena.",
              "¿Qué ceno en estos días de obra?", "locro de papa", stored="pescado al horno",
              write=pref("locro de papa", kind=I, start=7, through=9)),
        visit(D, 9, "Todavía están los obreros. Es el último día del cambio temporal.",
              "¿Qué tengo anotado para cenar hoy?", "locro de papa", stored="locro de papa"),
        visit(D, 10, "Ya puedo volver a usar el horno. Quiero recuperar la cena de siempre.",
              "¿Qué prefería normalmente para la cena?", "pescado al horno", stored="pescado al horno"),
    ), ((D, 10, None), (D, 11, None), (D, 14, None)),
        "El día 9 se incluye y el día 10 queda fuera. La excepción caduca y queda vigente el valor habitual."),
    common_case("res-05", "third_person", (
        prior(B, "tortilla de verde", "Mi desayuno habitual es tortilla de verde."),
        prior(R, "café pasado", "Yo normalmente tomo café pasado."),
    ), (
        visit(B, 7, "Mi hermano quiere pan de centeno en su desayuno; me pidió que lo apunte en su lista. "
              "Yo sigo con el mismo desayuno.",
              "¿Qué desayuno tengo anotado para mí?", "tortilla de verde", stored="tortilla de verde"),
        visit(R, 8, "Viene mi vecina y ella siempre toma té de jazmín. Esa bebida es para ella.",
              "¿Y cuál es mi bebida habitual?", "café pasado", stored="café pasado"),
        visit(B, 10, "Mi hermano desayunará fuera al final. De mi desayuno no he cambiado nada.",
              "¿Qué preparo para mi desayuno de siempre?", "tortilla de verde", stored="tortilla de verde"),
    ), ((B, 14, None), (R, 14, None), (B, 16, None)),
        "Las dos preferencias nuevas corresponden a terceros. El almacén personal del usuario conserva las suyas."),
    common_case("res-06", "explicit_correction", (
        prior(D, "mote con queso", "Normalmente ceno mote con queso."),
        prior(R, "limonada con hierbabuena", "Para beber, suelo pedir limonada con hierbabuena."),
        prior(R, "agua de coco", "Hasta el día 6 inclusive, quiero agua de coco.", start=4, kind=I, through=6),
    ), (
        visit(D, 7, "Quita el mote con queso de mi cena de siempre. Desde hoy dejo estofado de garbanzos como mi cena habitual.",
              "¿Qué anoto como cena de siempre?", "estofado de garbanzos", stored="mote con queso",
              write=pref("estofado de garbanzos", start=7)),
        visit(R, 8, "La excepción del agua de coco era hasta el día 6. Hoy quiero usar la ficha de siempre.",
              "¿Cuál es mi bebida habitual?", "limonada con hierbabuena", stored="limonada con hierbabuena"),
        visit(D, 10, "Hoy viene mi prima y ella cenará mote con queso. Para mí sigue valiendo mi cambio.",
              "¿Cuál es mi cena habitual ahora?", "estofado de garbanzos", stored="estofado de garbanzos"),
    ), ((D, 14, None), (R, 14, None), (D, 16, None)),
        "La rectificación sustituye el habitual desde su fecha. Una excepción expirada y una mención a "
        "la prima no autorizan cambios nuevos."),
    common_case("res-07", "future_start", (
        prior(B, "tostadas con tomate", "Mi desayuno habitual es tostadas con tomate."),
    ), (
        visit(B, 7, "El curso arranca el 9 de junio. Desde el día 9 en adelante quiero tamal de maíz como desayuno habitual. "
              "Hoy mantén el desayuno anterior.",
              "¿Qué me preparo para desayunar hoy?", "tostadas con tomate", stored="tostadas con tomate",
              write=pref("tamal de maíz", start=9)),
        visit(B, 8, "Aún no empieza el curso. Hoy seguimos con el desayuno de antes.",
              "¿Qué desayuno me toca hoy?", "tostadas con tomate", stored="tostadas con tomate"),
        visit(B, 10, "Ya empezó el curso y me estoy ajustando al horario nuevo.",
              "¿Qué habíamos dejado como desayuno desde el día 9?", "tamal de maíz", stored="tamal de maíz"),
    ), ((B, 14, None), (B, 16, None), (B, 17, None)),
        "La autorización de escritura llega el día 7. El valor nuevo entra en vigor el 9 y no debe sustituir "
        "el recuerdo aplicable los días 7 y 8."),
    common_case("res-08", "partial_followup", (
        prior(R, "jugo de maracuyá", "Lo habitual para mí es jugo de maracuyá."),
        prior(B, "maduros asados", "De desayuno suelo pedir maduros asados."),
        prior(D, "sopa de zapallo", "Mi cena habitual es sopa de zapallo."),
    ), (
        visit(R, 7, "Puedo seguir aquí un momento. Para la bebida, prefiero jugo de tomate de árbol.",
              "¿Qué bebida sirvo ahora?", "jugo de tomate de árbol", stored="jugo de maracuyá",
              alternatives=ambiguity("jugo de tomate de árbol"), slot="TEMPORAL_SCOPE", style="partial",
              partial="Seguro lo quiero para hoy. El alcance más allá de hoy sigue por confirmar."),
        visit(B, 8, "Hoy tengo todo el desayuno resuelto. Mantén mi desayuno habitual.",
              "¿Qué desayuno tenía guardado?", "maduros asados", stored="maduros asados"),
        visit(D, 10, "Resulta que cenaré en casa. Para la cena, prefiero tacos de hongos.",
              "¿Qué ceno hoy?", "tacos de hongos", stored="sopa de zapallo",
              alternatives=ambiguity("tacos de hongos"), slot="TEMPORAL_SCOPE"),
    ), ((R, 14, None), (R, 16, None), (D, 14, None)),
        "La respuesta parcial permite servir la bebida actual. El presupuesto compite entre resolver la "
        "vigencia de esa bebida y aclarar la cena sobrevenida."),
    common_case("res-09", "conditional_context", (
        prior(R, "horchata lojana", "En casa suelo tomar horchata lojana.", kind=C, context="casa"),
        prior(R, "té de jazmín", "En la oficina suelo tomar té de jazmín.", kind=C, context="oficina"),
    ), (
        visit(R, 7, "Todavía no sé si voy a estar en casa o en la oficina. En ambos sitios mantengo la bebida que ya anotamos.",
              "¿Qué bebida me corresponde en el sitio donde termine hoy?",
              ("CONDITIONALIZE", ["horchata lojana", "té de jazmín"])),
        visit(R, 8, "Cancelaron el desplazamiento. Hoy estoy en casa.",
              "¿Cuál era mi bebida de casa?", "horchata lojana", stored="horchata lojana", context="casa"),
        visit(R, 10, "Ahora sí vine a la oficina, pero las preferencias siguen igual.",
              "¿Cuál era mi bebida de oficina?", "té de jazmín", stored="té de jazmín", context="oficina"),
    ), ((R, 14, "casa"), (R, 14, "oficina"), (R, 17, "casa")),
        "Con el lugar abierto, una respuesta condicionada conserva las dos preferencias, y aplazar es "
        "prudente. El lugar explícito en visitas siguientes selecciona un recuerdo aplicable."),
    common_case("res-10", "late_plan_change", (
        prior(B, "huevos con espinaca", "Suelo desayunar huevos con espinaca."),
        prior(R, "agua mineral", "La bebida de siempre es agua mineral."),
        prior(D, "arroz con lentejas", "Para mi cena habitual elijo arroz con lentejas."),
    ), (
        visit(B, 7, "Ya me están esperando. Para el desayuno, prefiero maduros asados. "
              "Creía que mañana comería fuera, pero todavía no tengo el itinerario.",
              "¿Qué preparo para el desayuno de hoy?", "maduros asados", stored="huevos con espinaca",
              alternatives=ambiguity("maduros asados"), slot="TEMPORAL_SCOPE", style="refused"),
        visit(R, 8, "Cambió el itinerario y comeré aquí. Para la bebida, prefiero agua de coco.",
              "¿Qué bebida pongo hoy?", "agua de coco", stored="agua mineral",
              alternatives=ambiguity("agua de coco"), slot="TEMPORAL_SCOPE"),
        visit(D, 10, "También cancelaron la comida nocturna. Para la cena, prefiero tortilla de yuca.",
              "¿Qué dejo de cena ahora?", "tortilla de yuca", stored="arroz con lentejas",
              alternatives=ambiguity("tortilla de yuca"), slot="TEMPORAL_SCOPE"),
    ), ((B, 14, None), (R, 14, None), (D, 14, None)),
        "Una aclaración fallida al comienzo consume presupuesto útil para dos cambios posteriores."),
    common_case("res-11", "scoped_exception", (
        prior(D, "ensalada de quinua", "Mi cena habitual en cualquier sitio es ensalada de quinua."),
        prior(D, "choclo con queso", "Cuando trabajo en la oficina, mi cena habitual es choclo con queso.",
              kind=C, context="oficina"),
    ), (
        visit(D, 7, "En la oficina durante la capacitación, desde hoy hasta el día 9 inclusive, quiero cenar verduras al wok.",
              "¿Qué dejo para la cena de la oficina hoy?", "verduras al wok", stored="choclo con queso", context="oficina",
              write=pref("verduras al wok", kind=I, start=7, through=9, context="oficina")),
        visit(D, 8, "No hubo sesión presencial y hoy estoy en casa. La capacitación sigue mañana.",
              "¿Qué cenaba normalmente en casa?", "ensalada de quinua", stored="ensalada de quinua", context="casa"),
        visit(D, 10, "La capacitación terminó. Ya volví a la oficina.",
              "¿Qué cena corresponde a un día normal de oficina?", "choclo con queso",
              stored="choclo con queso", context="oficina"),
    ), ((D, 14, "oficina"), (D, 14, "casa"), (D, 17, "oficina")),
        "La excepción solo rige en la oficina y hasta el día 9. Cambiar de lugar a mitad del intervalo "
        "y volver después del cierre no prolonga ni globaliza el alcance."),
    common_case("res-12", "session_only", (
        prior(B, "yogur con mango", "Habitualmente desayuno yogur con mango."),
        prior(D, "pescado al horno", "Para cenar normalmente preparo pescado al horno."),
    ), (
        visit(B, 7, "Me regalaron pan de centeno y lo aprovecharé. Para el desayuno, prefiero pan de centeno, "
              "solo para esta sesión. Mi desayuno habitual queda igual.",
              "¿Qué desayuno preparo en esta visita?", "pan de centeno", stored="yogur con mango",
              write=pref("pan de centeno", kind=S, start=7)),
        visit(B, 8, "Se acabó el pan del regalo. Quiero retomar el desayuno que tenía guardado.",
              "¿Cuál era mi desayuno de siempre?", "yogur con mango", stored="yogur con mango"),
        visit(D, 10, "Para la cena tampoco tengo novedades. Sigo con lo habitual.",
              "¿Qué cena tengo anotada?", "pescado al horno", stored="pescado al horno"),
    ), ((B, 14, None), (D, 14, None), (B, 17, None)),
        "Una elección explícitamente de sesión se puede registrar con ese alcance. Las visitas siguientes "
        "solo usan la preferencia persistente."),
)


def _query(attribute, when, session, context=None):
    return {"attribute": attribute, "date": day(when), "session_id": session, "context": context}


def _expiry(preference):
    return (date.fromisoformat(day(preference.through)) + timedelta(days=1)).isoformat() if preference.through else None


def _native_fact(attribute, preference, when, session):
    return _fact(attribute, preference.value, preference.kind, day(preference.start or when),
                 context=preference.context, expiry=_expiry(preference),
                 session_id=session if preference.kind == S else None)


def _labeled_scope(attribute, preference, when, session=None):
    return {
        "attribute": attribute, "value": preference.value, "scope": preference.kind,
        "valid_from": day(preference.start or when),
        "valid_through_inclusive": day(preference.through) if preference.through else None,
        "context": preference.context,
        "session_id": session if preference.kind == S else None,
    }


def _retrieval(available, query, label, conditional):
    if conditional:
        values = set(conditional)
        return [item for item in available if item["attribute"] == query["attribute"] and item["value"] in values]
    if label is None:
        return []
    value = _update_truth([
        _native_fact(item["attribute"], Preference(
            item["value"], item["scope"], int(item["valid_from"][-2:]),
            int(item["valid_through_inclusive"][-2:]) if item["valid_through_inclusive"] else None,
            item["context"],
        ), int(query["date"][-2:]), item["session_id"])
        for item in available
    ], query)
    if value != label:
        raise ValueError(f"Incompatible authored retrieval {label!r} for {query}. Actual {value!r}.")
    # These roots maintain at most one matching record for a chosen value.
    compatible = [item for item in available
                  if item["attribute"] == query["attribute"] and item["value"] == label
                  and item["valid_from"] <= query["date"]
                  and (item["valid_through_inclusive"] is None or item["valid_through_inclusive"] >= query["date"])
                  and (item["context"] is None or item["context"] == query["context"])
                  and (item["scope"] != S or item["session_id"] == query["session_id"])]
    return compatible[-1:]


def _public(case):
    return {
        "id": case["id"], "question_budget": 2,
        "prior_user_evidence": [
            {"date": day(preference.start), "text": text,
             "context": preference.context}
            for _, preference, text in case["prior"]
        ],
        "visits": [
            {"visit": i, "session_id": f"S{i + 1}", "date": day(item["day"]), "context": item["context"],
             "attribute": item["attribute"], "user_message": item["text"], "question": item["question"]}
            for i, item in enumerate(case["visits"], 1)
        ],
        "future_queries": [_query(attribute, when, "S5", context)
                           for attribute, when, context in case["probes"]],
    }


def _gold(case):
    available = [_labeled_scope(attribute, preference, preference.start)
                 for attribute, preference, _ in case["prior"]]
    gold = []
    for index, item in enumerate(case["visits"], 1):
        session = f"S{index + 1}"
        attribute, when, context = item["attribute"], item["day"], item["context"]
        query = _query(attribute, when, session, context)
        mode, value = item["response"]
        retrieves = _retrieval(available, query, item["retrieval"], value if mode == "CONDITIONALIZE" else None)
        writes = ([_labeled_scope(attribute, item["new_write"], when, session)] if item["new_write"] else [])
        gold.append({"visit": index, "response": {"mode": mode, "value": value},
                     "new_write": writes, "retrieves": retrieves})
        available.extend(writes)
    return {"id": case["id"], "family": case["family"], "visits": gold,
            "author_adjudication": case["rationale"],
            "question_opportunities": [
                {"visit": index, "slot": item["slot"], "style": item["style"],
                 "partial_reply": item["partial"],
                 "world_replies": [reply for _, reply in item["alternatives"]]}
                for index, item in enumerate(case["visits"], 1) if item["alternatives"]
            ]}


def _worlds(case):
    memory, historical = [], {}
    for attribute, preference, text in case["prior"]:
        _remember(memory, historical, attribute, preference.value, day(preference.start), text,
                  kind=preference.kind, end=_expiry(preference), context=preference.context)
    count = 2 if any(item["alternatives"] for item in case["visits"]) else 1
    root = case["id"]
    for world in range(count):
        facts = [_prior(record) for record in memory]
        visits = []
        for index, item in enumerate(case["visits"], 1):
            attribute, when, context = item["attribute"], item["day"], item["context"]
            session = f"S{index + 1}"
            alternatives = tuple(_native_fact(attribute, pref_value, when, session)
                                 for pref_value, _ in item["alternatives"])
            factual = item["fact"]
            reply = None
            if alternatives:
                factual, reply = item["alternatives"][world]
            native = _native_fact(attribute, factual, when, session) if factual else None
            if native:
                facts.append(native)
            mode, value = item["response"]
            visits.append({
                "query": _query(attribute, when, session, context),
                "messages": [_turn(index, "situacion", "user", item["text"]),
                             _turn(index, "consulta", "user", item["question"])],
                "expected": value if mode == "ANSWER" else None,
                "fact": native, "alternatives": list(alternatives) or None, "slot": item["slot"],
                "style": item["style"], "answer": reply, "strategy_cue": None,
                "partial_answer": item["partial"],
            })
        future = []
        for index, (attribute, when, context) in enumerate(case["probes"], 1):
            query = _query(attribute, when, "S5", context)
            future.append({"kind": f"reuse_{index}", "query": query, "expected": _update_truth(facts, query)})
        yield {
            "root_id": root, "world_id": f"{root}:w{world + 1}", "realization_id": "authored",
            "split": "reserved", "scenario_family": case["family"], "recipe_id": root,
            "question_budget": 2, "world_probability": 1.0 / count, "initial_memory": memory,
            "prior_sessions": list(historical.values()), "impatient": False,
            "visits": visits, "future_queries": future, "second_visit_kind": "independent_ambiguity",
        }


REVIEW_INSTRUCTIONS = [
    "Evalúa las visitas en orden. Las visitas posteriores no son evidencia disponible para una decisión anterior.",
    "En cada visita asume que se aplicaron antes solo las escrituras justificadas por sus mensajes. No inventes "
    "una respuesta opcional de aclaración.",
    "En response indica ANSWER con un valor, CONDITIONALIZE con una lista ordenada de alternativas o DEFER con null.",
    "En new_write incluye solo cambios autorizados por la visita actual. Una preferencia de otra persona no modifica "
    "la memoria del usuario. Incluye una excepción de sesión si el texto la expresa de forma explícita.",
    "En retrieves incluye los recuerdos previos aplicables a la consulta, incluso si el mensaje actual justifica "
    "responder con otro valor. Si la consulta es condicional por lugar, conserva los dos recuerdos con sus contextos.",
    "Cada recuerdo o escritura lleva attribute, value, scope, valid_from, valid_through_inclusive, context, session_id. "
    "Usa las fechas ISO del caso, null para el último día de un habitual y para contextos y sesiones que no aplican.",
    "Los alcances son ongoing, contextual, interval y session. Un alcance de intervalo puede tener contexto adicional. "
    "En sesión, indica el session_id; la fecha sola no representa la vigencia.",
    "Usa solo los archivos PUBLIC.json y REVIEW_TEMPLATE.json. Pon reviewer_id único y una justificación breve "
    "si dudas de una fecha, contexto, atribución o respuesta.",
]


def freeze(folder=ROOT):
    public = [_public(case) for case in CASES]
    gold = [_gold(case) for case in CASES]
    worlds = [world for case in CASES for world in _worlds(case)]
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "MANIFEST.json"
    if manifest.exists():
        raise FileExistsError("Create a new reserve version before changing sealed artifacts")
    write_json(folder / "PUBLIC.json", {"version": VERSION, "rubric": REVIEW_INSTRUCTIONS, "cases": public})
    write_json(folder / "REVIEW_TEMPLATE.json", {"reviewer_id": "", "cases": [
        {"id": item["id"], "visits": [
            {"visit": visit["visit"], "response": {"mode": None, "value": None},
             "new_write": [], "retrieves": [], "rationale": ""}
            for visit in item["visits"]]}
        for item in public
    ]})
    write_json(folder / "PRIVATE_GOLD.json", {"version": VERSION, "cases": gold})
    (folder / "PRIVATE_WORLDS.jsonl").write_text(
        "".join(json.dumps(world, ensure_ascii=False) + "\n" for world in worlds), encoding="utf-8")
    artifacts = ["PUBLIC.json", "REVIEW_TEMPLATE.json", "PRIVATE_GOLD.json", "PRIVATE_WORLDS.jsonl"]
    value = {
        "version": VERSION, "status": "frozen_for_blind_review", "split": "reserved",
        "authorship": "separate AI subagent, manual writing without development generator or its wording",
        "real_user_data": False, "human_reviews": 0, "roots": len(public), "worlds": len(worlds),
        "visits_per_root": 3, "public_digest": digest(public),
        "artifacts": {name: file_hash(folder / name) for name in artifacts},
    }
    write_json(manifest, value)
    return value


def verify(folder=ROOT):
    manifest = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["version"] != VERSION or manifest["split"] != "reserved":
        raise ValueError("Unexpected bank version or split")
    for filename, checksum in manifest["artifacts"].items():
        if file_hash(folder / filename) != checksum:
            raise ValueError(f"Frozen artifact changed: {filename}")
    return manifest


def load_executable(folder=ROOT):
    """Supply executable sealed roots. Call only after the training selection is final."""
    manifest = verify(folder)
    rows = [json.loads(line) for line in (folder / "PRIVATE_WORLDS.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != manifest["worlds"] or any(row["split"] != "reserved" for row in rows):
        raise ValueError("Invalid executable reserve population")
    return rows


def load_gold(folder=ROOT):
    verify(folder)
    return json.loads((folder / "PRIVATE_GOLD.json").read_text(encoding="utf-8"))["cases"]


def _scope_record(record):
    scope = record["scope"]
    end = scope["valid_until_exclusive"]
    return {
        "attribute": record["attribute"], "value": record["value"], "scope": scope["kind"],
        "valid_from": scope["valid_from"],
        "valid_through_inclusive": (date.fromisoformat(end) - timedelta(days=1)).isoformat() if end else None,
        "context": scope["context"], "session_id": scope["session_id"],
    }


def _normalize_response(outcome):
    mode = outcome["mode"]
    value = ([branch["value"] for branch in outcome["branches"]] if mode == "CONDITIONALIZE"
             else outcome["value"])
    return mode, tuple(sorted(value)) if isinstance(value, list) else value


def semantic_suite(folder=ROOT):
    """Measure no-question current, write and retrieval against authored labels."""
    from research.src.readers.symbolic import read
    from research.tesis_extended_env import execute

    authors = {row["id"]: row["visits"] for row in load_gold(folder)}
    worlds = {}
    for row in load_executable(folder):
        worlds.setdefault(row["root_id"], row)
    if set(worlds) != set(authors):
        raise ValueError("Executable roots do not match the authored labels")
    cases = []
    for root, story in worlds.items():
        trace = execute(story, ("PROCEED",) * 3)
        visits = []
        for measured, gold in zip(trace["visits"], authors[root], strict=True):
            actual = measured["outcome"]
            read_ids = set(read(measured["memory_before"], measured["query"])["evidence_ids"])
            retrieves = [_scope_record(record) for record in measured["memory_before"]
                         if record["memory_id"] in read_ids]
            writes = [_scope_record(operation["record"]) for operation in measured["executed_operations"]
                      if "record" in operation]
            expected_mode = gold["response"]["mode"]
            answer = gold["response"]["value"]
            reference = expected_mode, tuple(sorted(answer)) if isinstance(answer, list) else answer
            same = lambda first, second: sorted(map(digest, first)) == sorted(map(digest, second))
            visits.append({
                "visit": gold["visit"],
                "response_exact": _normalize_response(actual) == reference,
                "strict_supported": bool(measured["current_supported"]),
                "new_write_exact": same(writes, gold["new_write"]),
                "retrieval_exact": same(retrieves, gold["retrieves"]),
                "actual_response": actual, "actual_new_write": writes,
                "actual_retrieval": retrieves, "authored": gold,
            })
        cases.append({"root_id": root, "family": story["scenario_family"], "visits": visits})
    rows = [visit for case in cases for visit in case["visits"]]
    return {
        "protocol": "semantic_no_question_replay", "roots": len(cases), "visits": len(rows),
        "matches": {name: sum(row[name] for row in rows)
                    for name in ("response_exact", "strict_supported", "new_write_exact", "retrieval_exact")},
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "semantic"))
    parser.add_argument("--folder", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "semantic":
        if args.output is None:
            parser.error("The semantic suite requires an output path outside the frozen packet")
        if args.output.resolve().is_relative_to(args.folder.resolve()):
            parser.error("The semantic suite must not write inside the frozen packet")
        result = semantic_suite(args.folder)
        write_json(args.output, result)
        result = {key: value for key, value in result.items() if key != "cases"}
    else:
        result = freeze(args.folder) if args.action == "freeze" else verify(args.folder)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
