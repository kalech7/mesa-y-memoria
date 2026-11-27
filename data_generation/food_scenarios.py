"""Independent, intention-authored fixtures for SPEC-001.

The generator never imports extraction, memory, retrieval or resolution code. Gold
is assembled alongside authored intentions, not computed from simulator outputs.
These are controlled Spanish templates, not a validated model of human behavior.
Splits reserve wording families and users; categories intentionally cross splits
so all twenty requirements remain evaluable. Test wording is not a claim of a
blind external holdout: maintainers can inspect this source.
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter

GOLD_VERSION = "v2"
GOLD_VERSION_REASON = "CPC labels unresolved contextual ambiguity; resolved coexistence is not a conflict. S20 daily scope is now explicit in visible evidence."

CATEGORIES = tuple(f"S{i:02d}" for i in range(1, 21))
MEALS = {"BREAKFAST": "desayuno", "LUNCH": "almuerzo", "DINNER": "cena", "SNACK": "merienda"}
COMBINATIONS = tuple((day, meal) for day in ("WEEKDAY", "WEEKEND") for meal in MEALS)
FOODS = ("avena", "arroz", "sopa", "fruta", "pizza", "ensalada", "tostadas", "tofu", "lentejas", "huevos", "pan", "queso")
PUBLIC_KEYS = {"scenario_id", "user_id", "category", "split", "family", "reference_time", "events", "query"}


def _id(seed, category, index):
    return hashlib.sha256(f"food-spec001:{seed}:{category}:{index}".encode()).hexdigest()[:20]


def _split(index):
    # 6/3/3 per category with 12 rows. No user is reused across scenarios.
    return "dev" if index % 12 < 6 else "validation" if index % 12 < 9 else "test"


def make_food_dataset(seed=42, instances=240):
    """Return disjoint public, gold and private projections (>=20 rows).

    Category counts differ by at most one when ``instances`` is not a multiple
    of twenty. The standard 240 rows give twelve rows per requirement. Seed
    controls food substitutions and food/context associations, not gold inference.
    """
    if not isinstance(instances, int) or instances < 20:
        raise ValueError("instances must be an integer >= 20")
    rng = random.Random(seed)
    public, gold, private = [], [], []
    for ordinal in range(instances):
        category = CATEGORIES[ordinal % 20]
        index = ordinal // 20
        split = _split(index)
        family = {"dev": "direct", "validation": "conversational", "test": "reformulated"}[split]
        sid = _id(seed, category, index)
        uid = f"user-{sid}"
        foods = rng.sample(list(FOODS), len(FOODS))
        first, second = foods[:2]
        events = []

        def event(text, source="USER_STATEMENT", at=None, **fields):
            message_id = f"{sid}-e{len(events) + 1}"
            events.append({"message_id": message_id, "session_id": f"{sid}-session-{len(events) + 1}",
                           "at": at or f"2026-10-{len(events) + 1:02d}T09:00:00+00:00",
                           "role": "user", "channel": "chat" if source == "USER_STATEMENT" else "orders",
                           "source": source, "text": text, **fields})
            return message_id

        def like(food, scope="En general", negative=False):
            forms = {"dev": f"{scope}, {'no me gusta' if negative else 'me gusta'} {food}.",
                     "validation": f"{scope}, {'no me agrada' if negative else 'prefiero'} {food}.",
                     "test": f"{scope}, {'me disgusta' if negative else 'me encanta'} {food}."}
            return forms[split]

        def query(scope=""):
            return {"dev": f"¿Qué prefiero comer{scope}?", "validation": f"¿Qué comida me gusta{scope}?",
                    "test": f"¿Qué alimento prefiero{scope}?"}[split]

        context = {"day_scope": None, "meal_scope": None}
        selected, expected, relevant, forbidden = [], [], [], []
        actions, conflict, needs = ["ANSWER"], None, False
        rationale = "La evidencia explícita es aplicable a la consulta."
        slots = {}
        q = query()

        if category == "S01":
            expected = [event(like(first))]
            selected = [first]
        elif category in {"S02", "S03"}:
            weekday = category == "S02"
            day_scope = "WEEKDAY" if weekday else "WEEKEND"
            scope = "De lunes a viernes" if weekday else "Los fines de semana"
            eid = event(like(first, scope))
            # Include an out-of-scope query to verify R1, without inventing a default.
            inside = index % 4 != 3
            target = weekday if inside else not weekday
            q = query(" el martes" if target else " el domingo")
            context["day_scope"] = "WEEKDAY" if target else "WEEKEND"
            relevant = [eid]
            if inside:
                expected, selected = [eid], [first]
            else:
                actions = ["ABSTAIN"]
                rationale = "La única preferencia comunicada tiene un día incompatible; no hay dato aplicable."
        elif category in {"S04", "S05", "S06", "S07"}:
            meal = tuple(MEALS)[int(category[1:]) - 4]
            other = tuple(MEALS)[(int(category[1:]) - 3) % 4]
            expected = [event(like(first, f"En el {MEALS[meal]}"))]
            relevant = expected + [event(like(second, f"En el {MEALS[other]}"))]
            q = query(f" para el {MEALS[meal]}")
            context["meal_scope"] = meal
            selected = [first]
        elif category == "S08":
            # Every profile carries all eight combinations, never a single cue.
            target = COMBINATIONS[index % 8]
            for offset, (day, meal) in enumerate(COMBINATIONS):
                scope = f"{'Entre semana' if day == 'WEEKDAY' else 'Los fines de semana'}, para el {MEALS[meal]}"
                eid = event(like(foods[offset], scope))
                relevant.append(eid)
                if (day, meal) == target:
                    expected, selected = [eid], [foods[offset]]
            context = {"day_scope": target[0], "meal_scope": target[1]}
            q = query(f" el {'martes' if target[0] == 'WEEKDAY' else 'sábado'} para el {MEALS[target[1]]}")
        elif category in {"S09", "S14"}:
            a = event(like(first, "Entre semana"))
            b = event(like(second, "Los fines de semana"))
            relevant, conflict = [a, b], "CPC"
            if category == "S09":
                target = index % 2 == 0
                context["day_scope"] = "WEEKDAY" if target else "WEEKEND"
                q = query(" el lunes" if target else " el domingo")
                expected, selected = ([a], [first]) if target else ([b], [second])
                conflict = None
                rationale = "Las preferencias coexisten en contextos distintos; el día conocido selecciona una sin conflicto no resuelto."
            else:
                expected, actions, needs = [a, b], ["CLARIFY"], True
                rationale = "Falta el día; preguntar discrimina preferencias válidas coexistentes."
                slots = {"day_scope": {"value": "WEEKDAY", "text": "Me refiero a un día entre semana."}}
        elif category == "S10":
            expected = [event(like(first)), event(like(first, negative=True))]
            actions, conflict = ["PRESERVE_CONFLICT", "CLARIFY"], "SAME_SCOPE"
            rationale = "El mismo alimento tiene polaridades opuestas sin rectificación explícita."
        elif category == "S11":
            a = event(like("pollo"))
            b = event("Corrijo lo anterior: desde el 2026-10-15 prefiero lentejas en lugar de pollo para cualquier comida y cualquier día.",
                      at="2026-10-15T09:00:00+00:00", supersedes=[a])
            expected, relevant, selected, forbidden = [b], [a, b], ["lentejas"], ["pollo"]
            rationale = "La corrección es explícita, tiene inicio de vigencia y conserva el evento anterior."
        elif category == "S12":
            expected = [event("En general, me gusta carne."), event("En general, no me gusta carne."),
                        event("Mi preferencia oscila: a veces me gusta carne y otras veces no me gusta carne; no tengo una preferencia estable.")]
            actions, conflict = ["PRESERVE_CONFLICT", "CLARIFY", "CONDITIONAL_ANSWER"], "BOC"
            rationale = "La trayectoria y la inestabilidad declarada no justifican una preferencia definitiva."
            slots = {"stability": {"value": "UNKNOWN", "text": "No tengo una preferencia estable; hoy tampoco la he decidido."}}
        elif category == "S13":
            expected = [event("En general, no me gusta carne."), event("Pedí carne para cualquier comida y cualquier día.", source="ORDER_HISTORY")]
            actions, conflict = ["VERIFY", "PRESERVE_CONFLICT", "CLARIFY"], "SCC"
            rationale = "Un pedido no invalida una declaración: ambas fuentes deben conservarse."
            slots = {"source": {"value": "THIRD_PARTY_ORDER", "text": "Ese pedido fue para otra persona."}}
        elif category == "S15":
            a = event(like(first, "En el almuerzo"))
            b = event(like(second, "En la merienda"))
            expected, actions, conflict, needs = [a, b], ["CLARIFY"], "CPC", True
            rationale = "Falta el tipo de comida necesario para elegir entre las dos preferencias."
            slots = {"meal_scope": {"value": "LUNCH", "text": "Me refiero al almuerzo."}}
        elif category == "S16":
            expected = [event(like(first, negative=True))]
            forbidden = [first]
            rationale = "La respuesta puede indicar el rechazo; DISLIKE nunca debe convertirse en selección positiva."
        elif category == "S17":
            expected = [event("Tengo alergia a las nueces."), event("En general, me gustan nueces y fruta.")]
            selected, forbidden = ["fruta"], ["nueces"]
            rationale = "La restricción explícita bloquea nueces aunque exista gusto positivo; fruta permanece disponible."
        elif category == "S18":
            expected, selected = [event(like(first))], [first]
            event("El autobús sale a las ocho; mi carpeta de trabajo es azul.")
            rationale = "El dato distractor no expresa preferencias y no altera la selección."
        elif category == "S19":
            expected, selected = [event(like(first, "Para la merienda"))], [first]
            context["meal_scope"] = "SNACK"
            q = query(" para la merienda")
        elif category == "S20":
            a = event("Todos los días, en la cena, me gusta sopa.")
            b = event("Solo el 2026-10-19 para la cena prefiero pizza en lugar de sopa.", at="2026-10-16T09:00:00+00:00")
            day = (18, 19, 20)[index % 3]
            q = query(f" para la cena del 2026-10-{day}")
            context = {"day_scope": "WEEKEND" if day == 18 else "WEEKDAY", "meal_scope": "DINNER", "date": f"2026-10-{day}"}
            expected, relevant = ([b], [a, b]) if day == 19 else ([a], [a, b])
            selected, forbidden = (["pizza"], ["sopa"]) if day == 19 else (["sopa"], [])
            rationale = "La excepción tiene una sola fecha; antes y después sigue aplicándose la preferencia habitual."
        if not relevant:
            relevant = list(expected)
        case = {"scenario_id": sid, "user_id": uid, "category": category, "split": split, "family": family,
                "reference_time": "2026-10-25T12:00:00+00:00", "events": events, "query": q}
        annotation = {"scenario_id": sid, "category": category, "expected_context": context,
                      "expected_food_items": sorted(selected), "acceptable_actions": actions,
                      "expected_evidence_ids": expected, "relevant_evidence_ids": relevant,
                      "expected_conflict_type": conflict, "requires_clarification": needs,
                      "rationale": rationale, "forbidden_food_items": forbidden}
        # Authored latent intentions are independent of parser and decision output.
        def intention(eid, food, polarity="LIKE", day=None, meal=None, **extra):
            return {"evidence_id": eid, "food_item": food, "polarity": polarity,
                    "day_scope": day, "meal_scope": meal, **extra}

        eids = [row["message_id"] for row in events]
        intentions, constraints = [], []
        if category in {"S01", "S18"}:
            intentions = [intention(eids[0], first, day="ANY", meal="ANY")]
        elif category in {"S02", "S03"}:
            intentions = [intention(eids[0], first, day="WEEKDAY" if category == "S02" else "WEEKEND")]
        elif category in {"S04", "S05", "S06", "S07"}:
            intentions = [intention(eids[0], first, meal=meal), intention(eids[1], second, meal=other)]
        elif category == "S08":
            intentions = [intention(eids[offset], foods[offset], day=day, meal=meal)
                          for offset, (day, meal) in enumerate(COMBINATIONS)]
        elif category in {"S09", "S14"}:
            intentions = [intention(eids[0], first, day="WEEKDAY"), intention(eids[1], second, day="WEEKEND")]
        elif category == "S10":
            intentions = [intention(eids[0], first, day="ANY", meal="ANY"),
                          intention(eids[1], first, "DISLIKE", day="ANY", meal="ANY")]
        elif category == "S11":
            intentions = [intention(eids[0], "pollo", day="ANY", meal="ANY", valid_to="2026-10-15"),
                          intention(eids[1], "lentejas", day="ANY", meal="ANY", valid_from="2026-10-15", supersedes=[eids[0]])]
        elif category == "S12":
            intentions = [intention(eids[0], "carne", day="ANY", meal="ANY"),
                          intention(eids[1], "carne", "DISLIKE", day="ANY", meal="ANY"),
                          intention(eids[2], "carne", "NEUTRAL", stability="OSCILLATING")]
        elif category == "S13":
            intentions = [intention(eids[0], "carne", "DISLIKE", day="ANY", meal="ANY", source="USER_STATEMENT"),
                          intention(eids[1], "carne", day="ANY", meal="ANY", source="ORDER_HISTORY")]
        elif category == "S15":
            intentions = [intention(eids[0], first, meal="LUNCH"), intention(eids[1], second, meal="SNACK")]
        elif category == "S16":
            intentions = [intention(eids[0], first, "DISLIKE", day="ANY", meal="ANY")]
        elif category == "S17":
            constraints = [{"evidence_id": eids[0], "food_item": "nueces", "type": "ALLERGY", "certainty": "EXPLICIT"}]
            intentions = [intention(eids[1], food, day="ANY", meal="ANY") for food in ("nueces", "fruta")]
        elif category == "S19":
            intentions = [intention(eids[0], first, meal="SNACK")]
        elif category == "S20":
            intentions = [intention(eids[0], "sopa", day="ANY", meal="DINNER"),
                          intention(eids[1], "pizza", meal="DINNER", valid_from="2026-10-19", valid_to="2026-10-19", exception=True)]
        if category in {"S14", "S15"}:
            annotation["clarification_followup"] = {
                "query_suffix": " el martes" if category == "S14" else " para el almuerzo",
                "expected_food_items": [first], "expected_evidence_ids": [eids[0]], "acceptable_actions": ["ANSWER"]}
        state = {"scenario_id": sid, "true_state": {"user_id": uid, "profile": {"synthetic": True, "seed": seed, "profile_index": ordinal},
                                                    "contextual_preferences": intentions, "restrictions": constraints},
                 "uncommunicated": {name: answer["value"] for name, answer in slots.items()},
                 "clarification_responses": slots}
        public.append(case)
        gold.append(annotation)
        private.append(state)
    validate_food_dataset(public, gold, private)
    return public, gold, private


def validate_food_dataset(public, gold, private):
    """Validate provenance and privacy independently of inference semantics."""
    if not len(public) == len(gold) == len(private):
        raise ValueError("Projection lengths differ")
    ids, user_splits = set(), {}
    forbidden = {"true_state", "uncommunicated", "clarification_responses", "expected_context", "expected_food_items", "acceptable_actions", "rationale"}

    def check_public(value):
        if isinstance(value, dict):
            if set(value) & forbidden:
                raise ValueError("Private field in public projection")
            for child in value.values():
                check_public(child)
        elif isinstance(value, list):
            for child in value:
                check_public(child)

    for case, expected, hidden in zip(public, gold, private):
        sid = case["scenario_id"]
        if set(case) != PUBLIC_KEYS or sid in ids or sid != expected["scenario_id"] or sid != hidden["scenario_id"]:
            raise ValueError("Invalid or duplicate scenario IDs/schema")
        ids.add(sid)
        check_public(case)
        uid, split = case["user_id"], case["split"]
        if uid in user_splits and user_splits[uid] != split:
            raise ValueError("User crosses splits")
        user_splits[uid] = split
        evidence = {event["message_id"] for event in case["events"]}
        if not set(expected["expected_evidence_ids"]) <= evidence or not set(expected["relevant_evidence_ids"]) <= evidence:
            raise ValueError("Untraceable evidence")
    return {"instances": len(public), "categories": dict(sorted(Counter(row["category"] for row in public).items())),
            "splits": dict(sorted(Counter(row["split"] for row in public).items()))}
