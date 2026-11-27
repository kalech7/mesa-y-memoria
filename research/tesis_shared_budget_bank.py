"""Paired Spanish development stories for competing cross-session clarifications."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import date, timedelta
from math import fsum, isclose
import random

from research.src.artifacts import canonical, digest
from research.tesis_bank import FOOD, _remember
from research.tesis_interactive_env import _fact, _prior, _turn, _update_truth


VERSION = "tesis_shared_budget_development_v1"
DEFAULT_SEED = 691407
DEFAULT_ROOTS_PER_REGIME = 4
FIRST_REPLY_RATES = {"ready": 0.9, "busy": 0.25}
SECOND_AMBIGUITY_RATES = {"regular": 0.2, "contingent": 0.8}
WORDING = {
    "train": {
        "ready": "Me puedo quedar un momento leyendo el chat.",
        "busy": "Estoy en la fila y tengo que prestar atención cuando me llamen.",
        "regular": "Mañana espero repetir mi rutina de comida.",
        "contingent": "La comida de mañana depende de cómo me salga un encargo.",
        "opening": "Quiero apuntar los platos antes de salir.",
        "focus": "Para {meal}, prefiero {value}.",
        "request": "Con eso, ¿qué dejo listo para {meal}?",
        "next": "Volví para organizar otra parte de la comida.",
        "same": "En cuanto a {meal}, mantengo la elección de siempre.",
        "temporary": "Para {meal}, prefiero {value}. Es para esta ocasión.",
        "session": "Es para esta ocasión. La próxima vez sigo con la opción de siempre.",
        "ongoing": "Déjalo como mi preferencia habitual para las próximas veces.",
        "previous": "Mi preferencia habitual para {meal} es {value}.",
        "expired": "Esa semana escogí {value} para {meal} y después volví a la opción de siempre.",
    },
    "validation": {
        "ready": "Todavía tengo batería y puedo leer un mensaje más.",
        "busy": "Voy entrando a una cita y quizá guarde el teléfono enseguida.",
        "regular": "La próxima comida está encaminada como de costumbre.",
        "contingent": "La próxima comida sigue en el aire mientras cuadramos el horario.",
        "opening": "Antes de cerrar la lista quiero comprobar una cosa.",
        "focus": "Sobre {meal}, prefiero {value}.",
        "request": "¿Qué anoto en la lista para {meal}, entonces?",
        "next": "Ya pasó un día y me queda otra elección pendiente.",
        "same": "Para {meal} puedes usar el gusto que ya conoces.",
        "temporary": "Sobre {meal}, prefiero {value}. Solo en esta conversación.",
        "session": "Cuenta solo esta sesión. Para otro día mantén lo que te había dicho antes.",
        "ongoing": "Apúntalo como preferencia habitual desde hoy.",
        "previous": "Como opción de {meal} para cada semana suelo escoger {value}.",
        "expired": "Durante aquel viaje mi elección de {meal} fue {value}; quedó atrás al volver.",
    },
}


def group_roots(rows):
    """Group latent realizations by the identical starting dialogue."""
    roots = defaultdict(list)
    for row in rows:
        roots[row["root_id"]].append(row)
    return dict(roots)


def public_cue_key(observation):
    """Read the compact cue that accompanies the public dialogue in the environment."""
    return observation.get("strategy_cue")


def _messages(index, opening, preference, request):
    return [
        _turn(index, "inicio", "user", opening),
        _turn(index, "eco", "assistant", "Cuéntame qué te apetece."),
        _turn(index, "eleccion", "user", preference),
        _turn(index, "consulta", "user", request),
    ]


def _query(attribute, when, session):
    return {"attribute": attribute, "date": when.isoformat(), "session_id": session, "context": None}


def _ambiguous_visit(query, messages, value, scope, wording, *, style="direct", cue=None):
    attribute, when = query["attribute"], query["date"]
    alternatives = [
        _fact(attribute, value, "session", when, session_id=query["session_id"]),
        _fact(attribute, value, "ongoing", when),
    ]
    fact = alternatives[0 if scope == "session" else 1]
    return {"query": query, "messages": messages, "expected": value, "fact": fact,
            "alternatives": alternatives, "slot": "TEMPORAL_SCOPE", "style": style,
            "answer": wording[scope], "strategy_cue": cue}


def _root(split, serial, seed, availability, next_meal, seen_recipes):
    rng = random.Random(digest([VERSION, seed, split, serial, availability, next_meal]))
    wording = WORDING[split]
    early, late, unrelated = rng.sample(list(FOOD), len(FOOD))
    while True:
        old, new = rng.sample(FOOD[early][1], 2)
        late_old, late_new, expired = rng.sample(FOOD[late][1], 3)
        identity = digest([early, old, new, late, late_old, late_new])
        if identity not in seen_recipes:
            seen_recipes.add(identity)
            break
    unrelated_value = rng.choice(FOOD[unrelated][1])
    same_day = date(2041, 2, 1) + timedelta(days=serial * 2)
    previous_day = (same_day - timedelta(days=21)).isoformat()
    next_day = same_day + timedelta(days=1)
    probe_day = next_day + timedelta(days=5)
    memory, history = [], {}
    for attribute, value in ((early, old), (late, late_old), (unrelated, unrelated_value)):
        _remember(memory, history, attribute, value, previous_day,
                  wording["previous"].format(meal=FOOD[attribute][0], value=value))
    _remember(memory, history, late, expired, (same_day - timedelta(days=10)).isoformat(),
              wording["expired"].format(meal=FOOD[late][0], value=expired), kind="interval",
              end=(same_day - timedelta(days=4)).isoformat())

    first_query = _query(early, same_day, "S2")
    second_query = _query(late, next_day, "S3")
    cue = f"{availability}_{next_meal}"
    early_meal, late_meal = FOOD[early][0], FOOD[late][0]
    first_messages = _messages(1, " ".join(wording[key] for key in ("opening", availability, next_meal)),
                               wording["focus"].format(meal=early_meal, value=new),
                               wording["request"].format(meal=early_meal))
    first_public = [memory, first_query, first_messages, cue]
    root = "sb" + digest(first_public)[:24]
    probes = [
        ("first_preference_reuse", _query(early, probe_day, "S4")),
        ("second_preference_reuse", _query(late, probe_day, "S4")),
        ("unrelated_preference", _query(unrelated, probe_day, "S4")),
    ]
    reply_rate = FIRST_REPLY_RATES[availability]
    ambiguity_rate = SECOND_AMBIGUITY_RATES[next_meal]
    first_scope_mass = {"session": 0.5, "ongoing": 0.5}
    reply_mass = {"direct": reply_rate, "refused": 1 - reply_rate}
    second_mass = {"routine": 1 - ambiguity_rate, "session": ambiguity_rate / 2,
                   "ongoing": ambiguity_rate / 2}
    rows = []
    for early_scope, early_mass in first_scope_mass.items():
        for style, style_mass in reply_mass.items():
            first = _ambiguous_visit(first_query, first_messages, new, early_scope, wording,
                                     style=style, cue=cue)
            for late_scope, late_mass in second_mass.items():
                routine = late_scope == "routine"
                temporary = routine and serial % 2 == 1
                statement_key = "temporary" if temporary else "same" if routine else "focus"
                statement = wording[statement_key].format(meal=late_meal, value=late_new)
                second_messages = _messages(2, wording["next"], statement, wording["request"].format(meal=late_meal))
                if routine:
                    fact = (_fact(late, late_new, "session", second_query["date"], session_id="S3")
                            if temporary else None)
                    second = {"query": second_query, "messages": second_messages,
                              "expected": late_new if temporary else late_old,
                              "fact": fact, "alternatives": None, "slot": None, "style": "direct",
                              "answer": None, "strategy_cue": None}
                else:
                    second = _ambiguous_visit(second_query, second_messages, late_new, late_scope, wording)
                facts = [_prior(record) for record in memory]
                facts.extend(visit["fact"] for visit in (first, second) if visit["fact"] is not None)
                future = [{"kind": kind, "query": probe, "expected": _update_truth(facts, probe)}
                          for kind, probe in probes]
                rows.append(deepcopy({
                    "root_id": root,
                    "world_id": "sw" + digest([root, early_scope, style, late_scope])[:24],
                    "realization_id": "v0", "split": split,
                    "scenario_family": cue, "recipe_id": identity, "question_budget": 1,
                    "world_probability": early_mass * style_mass * late_mass,
                    "initial_memory": memory, "prior_sessions": list(history.values()),
                    "impatient": False, "visits": [first, second], "future_queries": future,
                    "second_visit_kind": "temporary" if temporary else "routine" if routine else "independent_ambiguity",
                }))
    return rows


def audit_development(splits):
    """Check provenance of public cues, root isolation and latent probability mass."""
    if set(splits) != set(WORDING):
        raise ValueError("Solo se admiten TRAIN y validación en el piloto CPU.")
    seen_roots, seen_recipes, seen_worlds, seen_openings = set(), set(), set(), set()
    report = {}
    for split, rows in splits.items():
        counts = defaultdict(int)
        groups = group_roots(rows)
        openings = set()
        for root, worlds in groups.items():
            reference = worlds[0]
            first = reference["visits"][0]
            recipe = reference["recipe_id"]
            cue = first["strategy_cue"]
            availability, next_meal = cue.split("_")
            opening = first["messages"][0]["text"]
            openings.add(opening)
            if root in seen_roots or recipe in seen_recipes:
                raise ValueError(f"Raíz o combinación de platos repetida en {split}: {root}")
            seen_roots.add(root)
            seen_recipes.add(recipe)
            counts[cue] += 1
            public = canonical([reference["initial_memory"], first["query"], first["messages"], cue])
            if (len(worlds) != 12 or not all(WORDING[split][key] in opening for key in (availability, next_meal))
                    or not isclose(fsum(world["world_probability"] for world in worlds), 1.0)):
                raise ValueError(f"Mundos, masa o indicios públicos inconsistentes en {root}")
            for world in worlds:
                visit = world["visits"][0]
                actual = canonical([world["initial_memory"], visit["query"], visit["messages"], visit["strategy_cue"]])
                if (world["split"] != split or world["recipe_id"] != recipe or actual != public
                        or world["world_id"] in seen_worlds or world["world_probability"] <= 0):
                    raise ValueError(f"Prefijo, pertenencia o identidad de mundo inválido en {root}")
                seen_worlds.add(world["world_id"])
            reply_mass = fsum(w["world_probability"] for w in worlds if w["visits"][0]["style"] == "direct")
            late_mass = fsum(w["world_probability"] for w in worlds if w["visits"][1]["slot"] is not None)
            if not (isclose(reply_mass, FIRST_REPLY_RATES[availability])
                    and isclose(late_mass, SECOND_AMBIGUITY_RATES[next_meal])):
                raise ValueError(f"Calibración por indicio inconsistente en {root}")
        if seen_openings & openings:
            raise ValueError("Redacciones de apertura compartidas entre particiones.")
        seen_openings |= openings
        report[split] = {"roots": len(groups), "worlds": len(rows), "roots_per_cue": dict(counts)}
    return report


def build_development(seed=DEFAULT_SEED, roots_per_regime=DEFAULT_ROOTS_PER_REGIME):
    """Return controlled TRAIN and validation data. The human reserve is separate."""
    if type(roots_per_regime) is not int or not 1 <= roots_per_regime <= 24:
        raise ValueError("Se requieren entre 1 y 24 raíces por régimen.")
    splits, seen_recipes = {}, set()
    serial = 0
    for split in WORDING:
        rows = []
        for availability in FIRST_REPLY_RATES:
            for next_meal in SECOND_AMBIGUITY_RATES:
                for _ in range(roots_per_regime):
                    rows.extend(_root(split, serial, seed, availability, next_meal, seen_recipes))
                    serial += 1
        splits[split] = rows
    audit_development(splits)
    return splits
