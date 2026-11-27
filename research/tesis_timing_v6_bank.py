"""Paired Spanish stories about spending one clarification now or in a later visit.

A pair has the same present meal requests and the same latent preference worlds.
Only the public calendar hint changes at visit one. The calendar is informative
but fallible: in one fifth of the probability mass the announced plan changes.
The two scopes are independent of that hint and of one another. All replicas of
a root therefore have the same first two user utterances until an action buys a
clarification. Reusing the same worlds under budgets one and two isolates cost.
"""
from __future__ import annotations

from datetime import date, timedelta
from itertools import product
import random

from research.src.artifacts import digest


VERSION = "tesis_timing_paired_v6_1"
PACKETS = {"train": 3, "validation": 3, "reserved": 8}
PAIRS_PER_PACKET = 10
WORLD_COUNT = 8
WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes")
FIRST_THURSDAY = {"train": date(2034, 3, 2), "validation": date(2035, 1, 4), "reserved": date(2037, 1, 1)}
ARTICLES = {
    "dinner_preference": ("la cena", "de la cena"),
    "breakfast_preference": ("el desayuno", "del desayuno"),
    "drink_preference": ("la bebida", "de la bebida"),
}

# The corpora differ in meal vocabulary and sentence templates. Repetitions of
# a wording within one corpus are measured as such, not independent authors.
def _train_words():
    return {
        "dinner": ("sopa", "locro de papa", "menestra de fréjol", "pescado al horno",
                   "arroz con lentejas", "tortilla de verduras", "pasta con tomate",
                   "crema de zapallo", "pollo con arroz", "guiso de garbanzos"),
        "breakfast": ("avena con manzana", "yogur con granola", "pan con queso",
                      "bolón de verde", "humita", "tostadas con huevo"),
        "drink": ("jugo de mora", "agua mineral", "té de menta", "horchata lojana",
                  "café con leche", "limonada"),
        "opening": ("Hoy quiero {food} para cenar.", "Para la cena de hoy me provoca {food}.",
                    "Esta noche se me antoja {food}.", "Ponme {food} para la cena de hoy."),
        "second": ("Hoy me apetece {food} para {meal}.", "Para {meal} de hoy elijo {food}.",
                   "Anota {food} para {meal} de esta vez."),
        "cue": (
            "Si sale como lo he planeado, repito {many} durante cinco días. Para {few} solo tengo una fecha.",
            "Solo tengo una casilla para {few}. Para {many} estoy armando el menú de toda la semana.",
            "Puede cambiar el horario. De momento tengo {many} de lunes a viernes y {few} una sola vez.",
            "Puse {few} en una fecha suelta. Todavía me falta decidir {many} para varios días.",
            "En mi calendario hay cinco espacios para {many} y un único espacio para {few}.",
            "El encargo {of_many} se repite cada día de trabajo. El {of_few} quedó para una sola fecha.",
        ),
        "question": ("¿Lo dices solo para hoy o puedo recordarlo como tu preferencia habitual?",
                     "¿Esa elección vale únicamente para hoy o también para otras ocasiones?"),
        "ongoing": ("Sí, {food} es mi elección habitual para {meal}. Puedes recordarlo.",
                    "Para {meal}, {food} es lo habitual. Guárdalo para otras veces."),
        "session": ("Es solo por hoy. Para {meal} no he elegido nada para los próximos días.",
                    "Solo por hoy, por favor. El resto de las veces prefiero decidir después."),
        "third": ("Ya tengo el calendario definitivo. Quedaron cinco fechas para {many} y una para {few}.",
                  "Cerré el menú. Voy a necesitar {many} cinco días y {few} en uno."),
    }


def _validation_words():
    return {
        "dinner": ("sopa de cebada", "ceviche de palmito", "arroz con choclo", "lasaña de verduras",
                   "tallarines con albahaca", "estofado de res", "pastel de acelga", "sango de verde",
                   "tortitas de quinua", "crema de coliflor"),
        "breakfast": ("mote pillo", "arepa de maíz", "yogur con uvillas", "batido de avena",
                      "pan de yuca", "huevos con tomate"),
        "drink": ("infusión de cedrón", "jugo de guayaba", "agua con pepino",
                  "té negro", "colada de piña", "batido de coco"),
        "opening": ("Me gustaría cenar {food} hoy.", "Para esta noche elijo {food}.",
                    "Apúntame {food} como cena de hoy.", "Hoy me quedo con {food} para la cena."),
        "second": ("Para esta vez me quedo con {food} para {meal}.",
                   "Mi pedido para {meal} de hoy es {food}.",
                   "Para {meal} me gustaría {food} en esta ocasión."),
        "cue": (
            "De momento calculo {few} para una vez. Necesito organizar {many} para los cinco días laborables.",
            "Pienso preparar {many} varios días seguidos, mientras que {few} aparece una sola vez en la lista.",
            "En principio repito {many} toda la semana. Para {few} solo he reservado el {single_weekday}.",
            "El borrador {of_few} lleva una casilla. El borrador {of_many} lleva cinco casillas para días distintos.",
            "Salvo que cambie el encargo, {many} vuelve cada día. Con {few} bastará una vez.",
        ),
        "question": ("¿Es una elección para esta vez o quieres que la recuerde en próximas visitas?",
                     "¿Lo tomo como algo de hoy o como una preferencia que sigue vigente?"),
        "ongoing": ("Puedes guardar {food} como mi opción habitual para {meal}.",
                    "Quiero que recuerdes {food} para {meal}. Es una preferencia habitual."),
        "session": ("Cuenta solo por hoy. Después puedo elegir otra cosa.",
                    "Es solo por hoy. Decidiré {meal} de los días siguientes cuando llegue el momento."),
        "third": ("Acaban de confirmar los pedidos. Serán cinco de {many_without_article} y uno de {few_without_article}.",
                  "Ahora sí tengo las fechas. Debo organizar {many} cinco veces y {few} en una ocasión."),
    }


def _reserved_words():
    return {
        "dinner": ("guiso de chochos", "sopa de quinua", "pescado al vapor", "arroz con hongos",
                   "tigrillo de verde", "ensalada tibia de habas", "crema de zanahoria", "ñoquis de zapallo",
                   "pastel de choclo", "tortilla de brócoli", "lentejas con maduro", "puré de camote"),
        "breakfast": ("tamales de mote", "pan de centeno", "avena con pera", "tortilla de maíz",
                      "yogur con frutilla", "muesli con cacao", "empanada de viento", "quimbolito"),
        "drink": ("jugo de naranjilla", "agua de hierbaluisa", "té de jamaica", "café pasado",
                  "batido de taxo", "leche de almendra", "chocolate con canela", "agua de coco"),
        "opening": ("Esta vez voy a cenar {food}.", "Para la cena de esta noche, {food} me viene bien.",
                    "Hoy en la cena se me antojó {food}.", "Si todavía puedo elegir la cena de hoy, tomo {food}.",
                    "Esta noche quisiera que me sirvan {food}."),
        "second": ("He elegido {food} para {meal} de esta visita.",
                   "Para {meal}, esta vez quisiera {food}.",
                   "Si me preguntas por {meal} de hoy, me quedo con {food}.",
                   "Hoy voy a pedir {food} para {meal}."),
        "cue": (
            "Por lo que veo en la agenda, {many} ocupa las cinco jornadas. Para {few} hay un solo turno.",
            "Para {few} únicamente tengo un compromiso. La previsión {of_many} cubre toda mi semana laboral.",
            "Si se mantiene el pedido, me toca resolver {many} día tras día. Para {few} me apuntaron solo el {single_weekday}.",
            "Me dejaron un pedido suelto para {few}. En cambio, quieren {many} durante varios días seguidos.",
            "Mi borrador semanal deja la fila {of_many} llena de lunes a viernes y la fila {of_few} con una sola marca.",
            "Solo aparece {few} una vez en el reparto que me pasaron. Para {many} me tocarían cinco fechas distintas.",
            "Todo indica que voy a repetir {many} en cinco jornadas. El turno {of_few} parece ser único.",
            "Queda pendiente ratificar la lista. Para {few} calculé una fecha, para {many} calculé una por cada día laborable.",
        ),
        "question": ("¿Quieres que lo recuerde para otros días o se queda en esta ocasión?",
                     "¿Guardo esta elección como costumbre o solo la tomo para hoy?"),
        "ongoing": ("Déjalo anotado para otras visitas. Mi elección habitual para {meal} es {food}.",
                    "Es mi preferencia habitual, puedes recordar {food} para {meal}."),
        "session": ("Me sirve solo por hoy. No tengo una elección estable para después.",
                    "Solo por hoy. Para {meal} de la próxima vez todavía no me he decidido."),
        "third": ("Me llegó la agenda aprobada. Lleva cinco pedidos para {many} y uno para {few}.",
                  "Se acabaron los cambios. El reparto definitivo tiene {many} en cinco días y {few} en uno."),
    }


_WORDS = {"train": _train_words, "validation": _validation_words, "reserved": _reserved_words}


def _rng(*values):
    return random.Random(digest([VERSION, *values]))


def _id(prefix, *values):
    return prefix + digest([VERSION, *values])[:18]


def _agenda_words(many, few):
    most, least = ARTICLES[many], ARTICLES[few]
    return {"many": most[0], "few": least[0], "of_many": most[1], "of_few": least[1],
            "many_without_article": most[0].split(" ", 1)[1], "few_without_article": least[0].split(" ", 1)[1]}


def _recipes(words, split):
    meals = [("breakfast_preference", food) for food in words["breakfast"]]
    meals.extend(("drink_preference", food) for food in words["drink"])
    available = list(product(words["dinner"], meals))
    _rng("recipes", split).shuffle(available)
    return available


def _visit(date_value, attr, value, utterance, scope, wording, index, question_index):
    return {"date": date_value.isoformat(), "attribute": attr, "value": value, "utterance": utterance,
            "question": wording["question"][question_index % len(wording["question"])],
            "reply": wording[scope][index % len(wording[scope])].format(food=value, meal=ARTICLES[attr][0]),
            "scope": scope}


def _queries(day_three, many, few, food_values, scopes, single_day):
    rows = []
    for offset in range(1, 6):
        for attr in ((many, few) if offset == single_day else (many,)):
            rows.append({"date": (day_three + timedelta(days=offset)).isoformat(), "attribute": attr,
                         "expected_value": food_values[attr] if scopes[attr] == "ongoing" else None})
    return rows


def _pair(split, packet_id, serial, recipe, words):
    rng = _rng("public_wording", split, serial)
    dinner, (second_attribute, second_food) = recipe
    first_attribute = "dinner_preference"
    food_values = {first_attribute: dinner, second_attribute: second_food}
    opening = words["opening"][rng.randrange(len(words["opening"]))].format(food=dinner)
    second = words["second"][rng.randrange(len(words["second"]))].format(
        food=second_food, meal=ARTICLES[second_attribute][0])
    cue_index = serial % len(words["cue"])
    # A Thursday opening and Sunday confirmation put the five future services
    # on actual Monday–Friday dates. All worlds of a pair share these dates.
    first_date = FIRST_THURSDAY[split] + timedelta(weeks=serial)
    second_date = first_date + timedelta(days=1)
    last_date = second_date + timedelta(days=2)
    single_day = 1 + rng.randrange(5)
    reply_indices = [rng.randrange(2) for _ in range(2)]
    question_indices = [rng.randrange(2) for _ in range(2)]
    third_index = rng.randrange(len(words["third"]))
    pair_id = _id("p", split, serial)
    for variant in ("ask_now", "save"):
        high, low = ((first_attribute, second_attribute) if variant == "ask_now"
                     else (second_attribute, first_attribute))
        cue = words["cue"][cue_index].format(**_agenda_words(high, low), single_weekday=WEEKDAYS[single_day - 1])
        root = {"split": split, "packet_id": packet_id, "pair_id": pair_id,
                "root_id": _id("r", split, serial, variant), "variant": variant, "opening": opening, "cue": cue}
        worlds = []
        for predicted, scope_one, scope_two in product((True, False), ("ongoing", "session"), ("ongoing", "session")):
            many, few = (high, low) if predicted else (low, high)
            scopes = {first_attribute: scope_one, second_attribute: scope_two}
            visits = [_visit(first_date, first_attribute, dinner, f"{opening} {cue}", scope_one, words,
                             reply_indices[0], question_indices[0]),
                      _visit(second_date, second_attribute, second_food, second, scope_two, words,
                             reply_indices[1], question_indices[1]),
                      {"date": last_date.isoformat(),
                       "utterance": words["third"][third_index].format(**_agenda_words(many, few))}]
            worlds.append({**root, "world_id": _id("w", split, serial, variant, predicted, scope_one, scope_two),
                           "mass": (.8 if predicted else .2) / 4, "visits": visits,
                           "future_queries": _queries(last_date, many, few, food_values, scopes, single_day)})
        yield {**root, "worlds": worlds}


def build_bank(split):
    """Load only one split. Ten pairs share a packet for a 10/20 cost comparison."""
    if split not in _WORDS:
        raise ValueError(f"Unknown timing bank split: {split}")
    words = _WORDS[split]()
    recipes = _recipes(words, split)
    roots = []
    for packet in range(PACKETS[split]):
        packet_id = _id("b", split, packet)
        group = []
        for i in range(PAIRS_PER_PACKET):
            serial = packet * PAIRS_PER_PACKET + i
            group.extend(_pair(split, packet_id, serial, recipes[serial], words))
        # Independent of the variants' labels and world probabilities.
        _rng("packet_order", split, packet).shuffle(group)
        roots.extend(group)
    return roots


def build_worlds(split):
    """Deterministic packet order, then root order, then all eight compatible worlds."""
    return [world for root in build_bank(split) for world in root["worlds"]]


def split_metadata(split):
    if split not in PACKETS:
        raise ValueError(f"Unknown timing bank split: {split}")
    words = _WORDS[split]()
    return {"version": VERSION, "split": split, "packets": PACKETS[split],
            "pairs": PACKETS[split] * PAIRS_PER_PACKET, "roots": 2 * PACKETS[split] * PAIRS_PER_PACKET,
            "worlds": 2 * PACKETS[split] * PAIRS_PER_PACKET * WORLD_COUNT,
            "agenda_templates": len(words["cue"]), "opening_templates": len(words["opening"]),
            "second_visit_templates": len(words["second"]), "independent_human_authors": 0,
            "simulated_agenda_reliability": .8, "scope_probabilities": {"ongoing": .5, "session": .5}}


# Precomputed from complete canonical world lists. A training process can seal
# these identifiers in its plan without generating or inspecting reserved rows.
FROZEN_WORLD_SHA256 = {
    "train": "2b8226745c3d6b27ab9cd89533d7d6b1495da21679b955038db0528a34f8b2d3",
    "validation": "71225541a847a049608450efd0ddb9828c5399743f40c22f8c033930189dd110",
    "reserved": "1bf52cafbe6e4c43d9080074ce0ab58c842a0aacf49f4536d72b48fca545ed78",
}


def verify_worlds(split, worlds):
    """Check the world order, texts, responses and private probabilities."""
    if split not in FROZEN_WORLD_SHA256 or digest(worlds) != FROZEN_WORLD_SHA256[split]:
        raise ValueError(f"Timing bank contents differ from the sealed {split!r} worlds")
