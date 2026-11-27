"""Synthetic TRAIN and validation worlds for three-session planning in Spanish."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, timedelta
from math import fsum, isclose
import random

from research.src.artifacts import canonical, digest
from research.tesis_bank import FOOD, _remember
from research.tesis_interactive_env import _fact, _prior, _turn, _update_truth


VERSION = "tesis_extended_bank_v1"
DEFAULT_SEED = 582941
DEFAULT_ROOTS_PER_FAMILY = 3
FIRST_DIRECT = {"ready": .86, "busy": .29}
ACTIVATION = {"regular": .22, "contingent": .86}
WORDING = {
    "train": {
        "intro": "Estoy anotando el menú antes de pasar por el mercado.",
        "ready": "Tengo un rato para seguir hablando.",
        "busy": "Voy con prisa y enseguida tengo que mirar la agenda.",
        "regular": "Para pasado mañana espero seguir la rutina.",
        "contingent": "Lo de pasado mañana depende de cómo se organice el encuentro.",
        "second": "Vuelvo a preparar otra parte del menú.",
        "confirmed": "Confirmaron el encuentro. La comida de mañana sigue en el aire.",
        "canceled": "Cancelaron el encuentro. Para mañana vuelvo a la comida de costumbre.",
        "third": "Ya me toca decidir qué usar.",
        "pref": "Para {meal}, prefiero {value}.",
        "interval": "Estos días, para {meal}, prefiero {value}. Hasta el {end} inclusive.",
        "question": "¿Qué preparo hoy para {meal}?",
        "session": "Es para esta ocasión. Para las próximas sigue lo de siempre.",
        "ongoing": "Déjalo como preferencia habitual para las próximas veces.",
        "old": "Mi preferencia habitual para {meal} es {value}.",
        "expired": "Esa semana, para {meal}, preferí {value}. Después volví a mi rutina.",
    },
    "validation": {
        "intro": "Reviso las compras en una libreta antes de salir de casa.",
        "ready": "Puedo leer el chat otro momento antes de salir.",
        "busy": "Estoy entrando a una cita y quizá guarde el teléfono.",
        "regular": "La comida de pasado mañana ya quedó más o menos encaminada.",
        "contingent": "Para pasado mañana el menú puede cambiar hasta que cuadremos el horario.",
        "second": "Volví a cuadrar lo siguiente de la lista.",
        "confirmed": "El encuentro sigue en pie. Parte del plan de mañana está por confirmar.",
        "canceled": "Suspendieron la reunión. Mañana quedará como día de rutina.",
        "third": "Ahora puedo cerrar esta parte de la lista.",
        "pref": "Sobre {meal}, prefiero {value}.",
        "interval": "Sobre {meal}, prefiero {value}. Durante unos días, hasta el {end} inclusive.",
        "question": "¿Qué anoto para {meal} de hoy?",
        "session": "Cuenta solo esta sesión. En otro momento mantenemos la elección anterior.",
        "ongoing": "Apúntalo como preferencia habitual desde hoy.",
        "old": "Como opción habitual para {meal} acostumbro {value}.",
        "expired": "Durante aquellos días mi elección de {meal} fue {value}; luego volví a la anterior.",
    },
}


def _query(attribute, when, session):
    return {"attribute": attribute, "date": when.isoformat(), "session_id": session, "context": None}


def _messages(visit, opening, statement, query):
    return [
        _turn(visit, "apertura", "user", opening),
        _turn(visit, "pauta", "assistant", "Dime qué quieres usar para el menú."),
        _turn(visit, "eleccion", "user", statement),
        _turn(visit, "consulta", "user", query),
    ]


def _visit(index, query, value, opening, wording, scope, *, style="direct", cue=None, end=None):
    attribute = query["attribute"]
    when = query["date"]
    meal = FOOD[attribute][0]
    if end is None:
        alternatives = [
            _fact(attribute, value, "session", when, session_id=query["session_id"]),
            _fact(attribute, value, "ongoing", when),
        ]
        fact = alternatives[1] if scope == "ongoing" else alternatives[0]
        statement = wording["pref"].format(meal=meal, value=value)
    else:
        alternatives = None
        expiry = (end + timedelta(days=1)).isoformat()
        fact = _fact(attribute, value, "interval", when, expiry=expiry)
        statement = wording["interval"].format(meal=meal, value=value, end=end.isoformat())
    return {
        "query": query,
        "messages": _messages(index, opening, statement, wording["question"].format(meal=meal)),
        "expected": value, "fact": fact, "alternatives": alternatives,
        "slot": "TEMPORAL_SCOPE" if alternatives else None,
        "style": style, "answer": wording[scope] if alternatives else None, "strategy_cue": cue,
    }


def _root(split, serial, seed, availability, prospect, recipes):
    wording = WORDING[split]
    family = f"{availability}_{prospect}"
    rng = random.Random(digest([VERSION, seed, split, family, serial]))
    attributes = rng.sample(list(FOOD), len(FOOD))
    while True:
        pairs = [rng.sample(FOOD[attribute][1], 2) for attribute in attributes]
        recipe = digest([attributes, pairs])
        if recipe not in recipes:
            recipes.add(recipe)
            break
    first_day = date(2044, 3, 1) + timedelta(days=7 * serial)
    days = [first_day + timedelta(days=i) for i in range(3)]
    prior, stale, stale_expiry = [first_day - timedelta(days=n) for n in (24, 17, 9)]
    queries = [_query(attr, day, f"S{i}") for i, (attr, day) in enumerate(zip(attributes, days), 2)]
    initial, sessions = [], {}
    for attribute, (old, _) in zip(attributes, pairs):
        _remember(initial, sessions, attribute, old, prior.isoformat(),
                  wording["old"].format(meal=FOOD[attribute][0], value=old))
    second_attr = attributes[1]
    expired = rng.choice([value for value in FOOD[second_attr][1] if value not in pairs[1]])
    _remember(initial, sessions, second_attr, expired, stale.isoformat(),
              wording["expired"].format(meal=FOOD[second_attr][0], value=expired),
              kind="interval", end=stale_expiry.isoformat())

    first_opening = " ".join(wording[key] for key in ("intro", availability, prospect))
    first_direct = FIRST_DIRECT[availability] + rng.uniform(-.025, .025)
    activation = ACTIVATION[prospect] + rng.uniform(-.05, .05)
    styles = {"direct": first_direct, "partial": 1 - first_direct}
    schedule = (
        ("canceled", "interval", 1 - activation),
        ("confirmed", "session", activation / 2),
        ("confirmed", "ongoing", activation / 2),
    )
    last_valid = days[2] + timedelta(days=1 + serial % 2)
    later = first_day + timedelta(days=10)
    probes = [
        ("visit_one_reuse", _query(attributes[0], later, "S5")),
        ("visit_two_reuse", _query(attributes[1], later, "S5")),
        ("last_day_of_interval_or_preference_reuse", _query(attributes[2], last_valid, "S5")),
        ("post_expiry_or_preference_reuse", _query(attributes[2], last_valid + timedelta(days=1), "S6")),
    ]

    rows = []
    for style, style_mass in styles.items():
        for first_scope in ("session", "ongoing"):
            first = _visit(1, queries[0], pairs[0][1], first_opening, wording, first_scope,
                           style=style, cue=family)
            root_id = "tx" + digest([initial, queries[0], first["messages"]])[:24]
            for second_scope in ("session", "ongoing"):
                for event, third_scope, schedule_mass in schedule:
                    second = _visit(2, queries[1], pairs[1][1], wording["second"] + " " + wording[event],
                                    wording, second_scope, cue=f"plan_{event}")
                    interval = last_valid if event == "canceled" else None
                    third = _visit(3, queries[2], pairs[2][1], wording["third"], wording, third_scope,
                                   cue=f"plan_{event}", end=interval)
                    facts = [_prior(record) for record in initial] + [v["fact"] for v in (first, second, third)]
                    outcomes = [{"kind": kind, "query": query, "expected": _update_truth(facts, query)}
                                for kind, query in probes]
                    world_id = "tw" + digest([root_id, style, first_scope, second_scope, third_scope])[:24]
                    rows.append({
                        "root_id": root_id, "world_id": world_id, "split": split,
                        "realization_id": "v0", "scenario_family": family, "recipe_id": recipe,
                        "world_probability": style_mass * schedule_mass / 4, "question_budget": 2,
                        "initial_memory": initial, "prior_sessions": list(sessions.values()),
                        "visits": [first, second, third], "future_queries": outcomes,
                        "plan_change": event, "second_visit_kind": "independent_ambiguity",
                        "impatient": False,
                    })
    return rows


def group_roots(rows):
    roots = defaultdict(list)
    for row in rows:
        roots[row["root_id"]].append(row)
    return dict(roots)


def audit_development(splits):
    if set(splits) != set(WORDING):
        raise ValueError("Training generation must contain exactly TRAIN and validation")
    roots_seen, recipes_seen, world_ids, texts_seen = set(), set(), set(), set()
    report = {}
    for split, rows in splits.items():
        roots = group_roots(rows)
        families = Counter()
        openings = set()
        for root_id, worlds in roots.items():
            first = worlds[0]
            recipe = first["recipe_id"]
            family = first["scenario_family"]
            if root_id in roots_seen or recipe in recipes_seen:
                raise ValueError("Duplicate root or recipe across development")
            roots_seen.add(root_id)
            recipes_seen.add(recipe)
            families[family] += 1

            def visible(row):
                visit = row["visits"][0]
                return canonical([row["initial_memory"], visit["query"], visit["messages"], visit["strategy_cue"]])

            public = visible(first)
            openings.add(first["visits"][0]["messages"][0]["text"])
            if len(worlds) != 24 or not isclose(fsum(world["world_probability"] for world in worlds), 1.):
                raise ValueError(f"Incomplete latent alternatives for {root_id}")
            for world in worlds:
                if (visible(world) != public or world["split"] != split or world["recipe_id"] != recipe
                        or world["world_id"] in world_ids or world["world_probability"] <= 0):
                    raise ValueError(f"Invalid world or public prefix for {root_id}")
                world_ids.add(world["world_id"])
            if len({world["plan_change"] for world in worlds}) != 2:
                raise ValueError(f"Missing a plausible change of plan for {root_id}")
        if texts_seen & openings:
            raise ValueError("Opening text overlaps across roles")
        texts_seen |= openings
        report[split] = {"roots": len(roots), "worlds": len(rows), "roots_per_family": dict(families)}
    return report


def build_development(seed=DEFAULT_SEED, roots_per_family=DEFAULT_ROOTS_PER_FAMILY):
    if type(roots_per_family) is not int or not 1 <= roots_per_family <= 24:
        raise ValueError("Use between 1 and 24 roots per family")
    splits, recipes = {}, set()
    serial = 0
    for split in WORDING:
        rows = []
        for availability in FIRST_DIRECT:
            for prospect in ACTIVATION:
                for _ in range(roots_per_family):
                    rows.extend(_root(split, serial, seed, availability, prospect, recipes))
                    serial += 1
        splits[split] = rows
    audit_development(splits)
    return splits
