"""Disjoint synthetic users for development of three-visit policies."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, timedelta
from math import fsum, isclose
import random

from research.src.artifacts import canonical, digest
from research.tesis_bank import FOOD, _remember
from research.tesis_interactive_env import _fact, _prior, _turn, _update_truth


VERSION = "tesis_profiles_development_v1"
DEFAULT_SEED = 419583
# Style, probability of a changed plan, uncertainty in visit 2, type of visit 3.
PROFILES = {
    "direct_routine": ("direct", .18, True, "scope"),
    "direct_unstable": ("direct", .82, True, "context"),
    "partial_routine": ("partial", .17, False, "scope"),
    "partial_unstable": ("partial", .85, True, "context"),
    "deferred_recontact": ("deferred", .63, True, "scope"),
    "refused_leaves": ("refused", .75, True, "context"),
}

WORDS = {
    "train": {
        "intro": ("Me toca ordenar la lista del mercado.", "Voy preparando una nota para cocinar.",
                  "Estoy repasando las opciones del menú.", "Anoto lo que me toca comprar."),
        "direct": ("Tengo un rato para seguir con esto.", "Me quedo pendiente del chat."),
        "partial": ("Tengo que mirar mi agenda antes de decidir los detalles.", "Voy con prisa ahora."),
        "deferred": ("Ahora tengo poco tiempo. Te respondo cuando vuelva.", "Voy con prisa. Te confirmo más tarde."),
        "refused": ("Voy con prisa y enseguida cierro el chat.", "Me están llamando y suelto el teléfono."),
        "calm": ("Pasado mañana sigo la comida de costumbre.", "Para pasado mañana espero seguir la rutina."),
        "uncertain": ("Lo de pasado mañana sigue en el aire.", "Pasado mañana depende de cómo cambie el horario."),
        "changed": "Cambiaron el encuentro. Lo de mañana está en el aire.",
        "canceled": "Se canceló el encuentro. Mañana vuelvo a la rutina.",
        "neutral": "Cierro esta parte de la lista.",
        "third_context": "Con el cambio de plan ya tengo otra ubicación. Ese lugar cambia lo que me sirve. ¿Qué elijo para {meal}?",
        "scope": ("Para {meal}, prefiero {value}.", "Si hablamos de {meal}, prefiero {value}."),
        "correction": "Rectifico. Para {meal}, prefiero {value}.",
        "temporary": "Para {meal}, prefiero {value}. Solo para esta ocasión.",
        "interval": "Para {meal}, prefiero {value}. Hasta el {end} inclusive.",
        "part": "Por ahora me sirve. Déjame mirar esto antes de precisar el alcance.",
        "session": "Solo para esta ocasión. En las demás sigue la elección de siempre.",
        "ongoing": "Déjalo como preferencia habitual desde hoy.",
        "ask": "¿Qué pongo hoy para {meal}?",
        "reported": "Mi hermana dijo: «Para {meal}, prefiero {value}». Ella está preparando su plato.",
        "late": "Ya volví. Para {meal}, prefiero {value} de forma habitual.",
        "old": "Mi preferencia habitual para {meal} es {value}.",
        "office": "Para {meal}, cuando estoy en la oficina, mi opción es {value}.",
        "old_interval": "Durante unos días escogí {value} para {meal} y después volví a mi rutina.",
    },
    "validation": {
        "intro": ("Volví a mirar mi cuaderno de compras.", "Me falta completar la libreta con el menú.",
                  "Me avisaron que prepare una lista corta.", "Estoy revisando lo que se sirve en casa."),
        "direct": ("Todavía puedo leer un mensaje más.", "Puedo contestar antes de salir."),
        "partial": ("Me queda poco tiempo hasta que revise el calendario.", "No tengo un rato para quedarme."),
        "deferred": ("Entro a una cita. Te digo más tarde cuando retome la conversación.",
                     "Voy a guardar el teléfono. Te escribo cuando pueda."),
        "refused": ("En cualquier momento voy a cerrar el chat.", "Ya tengo que soltar el teléfono."),
        "calm": ("La comida de pasado mañana parece encaminada.", "Pasado mañana mantengo la pauta de costumbre."),
        "uncertain": ("Para pasado mañana hay cosas por confirmar.", "El plan de pasado mañana puede cambiar."),
        "changed": "Me movieron el compromiso. Parte del plan de mañana está por confirmar.",
        "canceled": "Suspendieron el compromiso. La comida de mañana queda como de costumbre.",
        "neutral": "Volví a mirar la última parte de la lista.",
        "third_context": "El lugar final es distinto y en cada sitio tengo una opción. ¿Qué recomiendas para {meal}?",
        "scope": ("Sobre {meal}, prefiero {value}.", "En cuanto a {meal}, prefiero {value}."),
        "correction": "Cambio el dato que acabo de decir. En cuanto a {meal}, prefiero {value}.",
        "temporary": "Sobre {meal}, prefiero {value}. Úsalo solo en esta conversación.",
        "interval": "En cuanto a {meal}, prefiero {value}. Vale hasta el {end} inclusive.",
        "part": "Por ahora sí. Tengo que mirar la agenda para decir algo más concreto.",
        "session": "Cuenta solo esta sesión. Después mantenemos la alternativa anterior.",
        "ongoing": "Regístralo como mi preferencia habitual para el futuro.",
        "ask": "¿Qué elección hago para {meal} en esta visita?",
        "reported": "Mi sobrina escribió: «Para {meal}, prefiero {value}». Esa fue su elección.",
        "late": "Retomo el chat. Sobre {meal}, prefiero {value} de forma habitual.",
        "old": "Como opción habitual para {meal}, suelo escoger {value}.",
        "office": "Mi opción de {meal} en la oficina es {value}.",
        "old_interval": "En aquel viaje elegí {value} para {meal}. La excepción quedó atrás.",
    },
}


def _query(attribute, when, number, context=None):
    return {"attribute": attribute, "date": when.isoformat(), "session_id": f"S{number}",
            "context": context}


def _scope_fact(query, value, scope, end=None):
    return _fact(query["attribute"], value, scope, query["date"],
                 session_id=query["session_id"] if scope == "session" else None,
                 expiry=(end + timedelta(days=1)).isoformat() if end else None)


def _turns(index, opening, rows, question):
    texts = [opening, *rows, question]
    return [_turn(index, f"u{i}", "user", text) for i, text in enumerate(texts)]


def _preference_visit(index, query, values, wording, opening, scope, *,
                      wording_index=0, style="direct", end=None, overwrite=False,
                      known_session=False, distractor=False):
    _, chosen, other = values
    meal = FOOD[query["attribute"]][0]
    rows = []
    if distractor:
        rows.append(wording["reported"].format(meal=meal, value=other))
    if overwrite:
        rows.append(wording["scope"][wording_index].format(meal=meal, value=other))
    alternatives = None
    slot = None
    if end is not None:
        final = wording["interval"].format(meal=meal, value=chosen, end=end.isoformat())
    elif known_session:
        final = wording["temporary"].format(meal=meal, value=chosen)
    else:
        final = wording["correction" if overwrite else "scope"]
        final = (final if isinstance(final, str) else final[wording_index]).format(meal=meal, value=chosen)
        alternatives = [_scope_fact(query, chosen, kind) for kind in ("session", "ongoing")]
        slot = "TEMPORAL_SCOPE"
    rows.append(final)
    return {
        "query": query, "messages": _turns(index, opening, rows, wording["ask"].format(meal=meal)),
        "expected": chosen, "fact": _scope_fact(query, chosen, scope, end),
        "alternatives": alternatives, "slot": slot, "style": style,
        "answer": wording[scope] if slot else None, "partial_answer": wording["part"], "strategy_cue": None,
    }


def _context_visit(query, values, location, wording):
    old, _, office = values
    meal = FOOD[query["attribute"]][0]
    return {
        "query": query, "messages": _turns(3, wording["neutral"], [],
                                           wording["third_context"].format(meal=meal)),
        "expected": office if location == "oficina" else old, "fact": None, "alternatives": None,
        "slot": "CURRENT_CONTEXT", "style": "direct",
        "answer": "Estoy en la oficina." if location == "oficina" else "Estoy en casa.",
        "strategy_cue": None,
    }


def _root(split, family, index, serial, seed, seen_recipes):
    words = WORDS[split]
    style, changed_rate, middle_ambiguous, late_kind = PROFILES[family]
    rng = random.Random(digest([VERSION, seed, split, family, index]))
    date_offset = 5 * serial + rng.randrange(4)
    first_day = date(2045, 9, 2) + timedelta(days=date_offset)
    days = [first_day + timedelta(days=i) for i in range(3)]
    attributes = rng.sample(list(FOOD), len(FOOD))
    while True:
        values = [rng.sample(FOOD[attr][1], 3) for attr in attributes]
        recipe = digest([attributes, [entry[:2] for entry in values]])
        if recipe not in seen_recipes:
            seen_recipes.add(recipe)
            break
    query_one, query_two, query_three = [_query(attr, day, i + 2)
                                         for i, (attr, day) in enumerate(zip(attributes, days))]
    budget = 2 if style == "partial" or index % 2 == 0 else 1
    changed_rate += rng.uniform(-.04, .04)
    start = (first_day - timedelta(days=25)).isoformat()
    earlier = (first_day - timedelta(days=9)).isoformat()
    expiry = (first_day - timedelta(days=4)).isoformat()
    initial, sessions = [], {}
    for attr, (old, _, _) in zip(attributes, values):
        _remember(initial, sessions, attr, old, start, words["old"].format(meal=FOOD[attr][0], value=old))
    if late_kind == "context":
        _remember(initial, sessions, attributes[2], values[2][2], start,
                  words["office"].format(meal=FOOD[attributes[2]][0], value=values[2][2]),
                  kind="contextual", context="oficina")
    _remember(initial, sessions, attributes[1], values[1][2], earlier,
              words["old_interval"].format(meal=FOOD[attributes[1]][0], value=values[1][2]),
              kind="interval", end=expiry)

    prospect = "calm" if changed_rate < .35 else "uncertain"
    position = index % 2
    opening = " ".join((words["intro"][index % len(words["intro"])], words[style][position],
                        words[prospect][position]))
    late_states = ("casa", "oficina") if late_kind == "context" else ("session", "ongoing")
    portfolio = (
        ("canceled", "interval", 1 - changed_rate),
        ("changed", late_states[0], changed_rate / 2),
        ("changed", late_states[1], changed_rate / 2),
    )
    mid_states = (("session", .5), ("ongoing", .5)) if middle_ambiguous else (("session", 1.),)
    last = days[2] + timedelta(days=1 + index % 2)
    after = first_day + timedelta(days=10)
    probes = [
        ("first_scope_reuse", _query(attributes[0], after, 5)),
        ("corrected_preference_reuse", _query(attributes[1], after, 5)),
        ("third_last_valid_day", _query(attributes[2], last, 5, "casa")),
        ("third_post_expiry_or_context", _query(attributes[2], last + timedelta(days=1), 6, "oficina")),
    ]
    rows = []
    for scope_one in ("session", "ongoing"):
        for scope_two, middle_mass in mid_states:
            for plan, late_truth, chance in portfolio:
                first = _preference_visit(1, query_one, values[0], words, opening, scope_one,
                                          wording_index=position, style=style, distractor=index % 2 == 0)
                if style == "deferred":
                    returning_choice = values[0][0 if scope_one == "session" else 1]
                    meal = FOOD[attributes[0]][0]
                    clarified = words["late"].format(meal=meal, value=returning_choice)
                    if scope_one == "session":
                        clarified = ("La otra elección fue solo para aquella ocasión. " + clarified)
                    first["recontact"] = {"visit": 2 if plan == "canceled" else 3, "message": clarified}
                # The final declaration in the second visit corrects an earlier one.
                second = _preference_visit(2, query_two, values[1], words, words[plan], scope_two,
                                           wording_index=position, overwrite=True,
                                           known_session=not middle_ambiguous,
                                           distractor=index % 2 == 1)
                if plan == "changed" and late_kind == "context":
                    third = _context_visit(query_three, values[2], late_truth, words)
                else:
                    query = {**query_three, "context": "casa"} if plan == "canceled" else query_three
                    third = _preference_visit(3, query, values[2], words, words["neutral"], late_truth,
                                              wording_index=position, end=last if plan == "canceled" else None)
                visits = [first, second, third]
                facts = [_prior(record) for record in initial] + [visit["fact"] for visit in visits if visit["fact"]]
                future = [{"kind": kind, "query": probe, "expected": _update_truth(facts, probe)}
                          for kind, probe in probes]
                root = "vp" + digest([initial, first["messages"], query_one, budget])[:24]
                world = "vw" + digest([root, scope_one, scope_two, plan, late_truth])[:24]
                rows.append({
                    "root_id": root, "profile_id": "user_" + root[2:], "world_id": world,
                    "realization_id": "v0", "split": split, "scenario_family": family,
                    "recipe_id": recipe, "question_budget": budget,
                    "world_probability": .5 * middle_mass * chance,
                    "initial_memory": initial, "prior_sessions": list(sessions.values()),
                    "visits": visits, "future_queries": future, "plan_change": plan,
                    "second_visit_kind": "independent_ambiguity", "impatient": False,
                })
    return rows


def group_roots(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["root_id"]].append(row)
    return dict(grouped)


def _visible_prefix(story):
    visit = story["visits"][0]
    return canonical([visit["messages"], visit["query"], story["initial_memory"], story["question_budget"]])


def audit_development(splits):
    if set(splits) != set(WORDS):
        raise ValueError("Profile development consists only of TRAIN and validation")
    seen_roots, seen_worlds, seen_recipes, seen_profiles, seen_openings = set(), set(), set(), set(), set()
    report = {}
    for split, rows in splits.items():
        roots = group_roots(rows)
        families, budgets, styles = Counter(), Counter(), Counter()
        openings = set()
        for root, worlds in roots.items():
            first = worlds[0]
            recipe, profile = first["recipe_id"], first["profile_id"]
            if root in seen_roots or recipe in seen_recipes or profile in seen_profiles:
                raise ValueError("Root, recipe or profile identity is reused")
            seen_roots.add(root)
            seen_recipes.add(recipe)
            seen_profiles.add(profile)
            families[first["scenario_family"]] += 1
            budgets[first["question_budget"]] += 1
            styles[first["visits"][0]["style"]] += 1
            openings.add(first["visits"][0]["messages"][0]["text"])
            visible = _visible_prefix(first)
            if (len(worlds) not in (6, 12)
                    or not isclose(fsum(world["world_probability"] for world in worlds), 1.)
                    or len({world["plan_change"] for world in worlds}) != 2):
                raise ValueError(f"Incomplete latent distribution at {root}")
            for world in worlds:
                if (world["split"] != split or world["profile_id"] != profile
                        or world["recipe_id"] != recipe or world["world_id"] in seen_worlds
                        or world["world_probability"] <= 0 or _visible_prefix(world) != visible):
                    raise ValueError(f"Invalid world or starting observation at {root}")
                seen_worlds.add(world["world_id"])
        if openings & seen_openings:
            raise ValueError("Opening text is shared across partitions")
        seen_openings |= openings
        report[split] = {"profiles": len(roots), "worlds": len(rows), "families": dict(families),
                         "question_budgets": dict(budgets), "reply_styles": dict(styles)}
    return report


def build_development(seed=DEFAULT_SEED, roots_per_family=4, validation_roots_per_family=2):
    sizes = {"train": roots_per_family, "validation": validation_roots_per_family}
    if any(type(size) is not int or not 1 <= size <= 24 for size in sizes.values()):
        raise ValueError("Profile groups use between 1 and 24 roots per family")
    splits, recipes, serial = {}, set(), 0
    for split, size in sizes.items():
        rows = []
        for family in PROFILES:
            for index in range(size):
                rows.extend(_root(split, family, index, serial, seed, recipes))
                serial += 1
        splits[split] = rows
    audit_development(splits)
    return splits
