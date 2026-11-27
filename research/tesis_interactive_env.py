"""Two interactive visits and delayed reuse through the native memory environment."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date, timedelta
from itertools import product
import json
from pathlib import Path
import random
import re
from tempfile import TemporaryDirectory
from functools import lru_cache

from research import tesis_bank
from research.src.artifacts import canonical, digest, file_hash, write_json
from research.src.env.episode import EpisodeEnvironment
from research.src.env.user_simulator import QUESTIONS
from research.src.memory.executor import validate_record
from research.src.policy.rule_controller import RuleController, evidence_model
from research.src.readers.symbolic import read
from research.src.scoring.evidence import audit_operation, score_response


VERSION = "tesis_interactive_v1"
DEFAULT_SEED = 683915
ACTIONS = ("PROCEED", *QUESTIONS)
MAX_QUESTIONS = 2
BANK = Path(__file__).resolve().parents[1] / "research/data/tesis_bank_v1"
NOOP = [{"op": "NOOP"}]
REUSE = "Habrá nuevas conversaciones. Conserva solo las preferencias y excepciones que la persona haya confirmado."

DISCOURSE = {
    "train": (
        ("Estoy mirando qué hay en la despensa.", "Hoy hago una compra rápida.", "Voy a dejar algo preparado."),
        "Claro, dime qué tienes en mente.",
        ("¿Qué me recomiendas para {meal}?", "¿Qué dejo preparado para {meal}?"),
        ("Perdón, ayer iba de un lado a otro.", "Vuelvo con lo de la comida."),
        ("Para esta ocasión, prefiero {value}. Solo esta ocasión.", "Prefiero {value}. Es solo esta ocasión."),
        ("Para hoy elijo {value}.", "Esta vez tengo ganas de {value}."),
    ),
    "validation": (
        ("Me cambiaron un plan y tengo que improvisar.", "Voy a organizar el menú antes de salir.", "Tengo un momento para planear la comida."),
        "Cuéntame qué te sirve hoy.",
        ("Con eso, ¿qué preparo para {meal}?", "¿Qué pongo en la lista para {meal}?"),
        ("Hola de nuevo. Ya es otro día.", "Me quedé pensando en lo del menú."),
        ("Solo esta ocasión, prefiero {value}.", "Para este rato, prefiero {value}. Solo esta ocasión."),
        ("Ahora quiero {value}.", "Tengo en mente {value} para hoy."),
    ),
    "evaluation": (
        ("Estoy sacando cosas del refrigerador.", "Se me hizo tarde para organizar la comida.", "Antes de pasar por la tienda, te cuento algo."),
        "Te leo. ¿Qué te apetece?",
        ("Entonces, ¿qué dejarías para {meal}?", "¿Con qué te quedarías para {meal}?"),
        ("Buenas, hoy retomé esto.", "Otra vez me toca decidir qué comer."),
        ("Prefiero {value}. Que sea solo esta ocasión.", "Solo esta ocasión me viene bien: prefiero {value}."),
        ("Hoy me tira más {value}.", "Para comer ahora elijo {value}."),
    ),
}
REPLY = {
    "train": {
        "partial": "Es por ahora. Dame un segundo, que quiero mirar cómo me organizo esta semana.",
        "refused": "No lo tengo decidido. Mejor dejarlo ahí por ahora.",
        "other": "Eso no es lo que me está cambiando la elección.",
        "repeat": "Ya te dije lo que tenía claro.",
        "context_partial": "Todavía estoy de camino; déjame mirar adónde voy.",
        "expiry_partial": "Sí termina esta semana; tengo que mirar la fecha.",
        "session": "Solo esta ocasión. Para después sigue lo de siempre.",
        "ongoing": "Déjalo como preferencia habitual. Lo voy a seguir eligiendo.",
        "interval": "Miré la agenda: hasta el {end} inclusive.",
    },
    "validation": {
        "partial": "Por ahora. A ver, dame un segundo y miro qué tengo pensado para los demás días.",
        "refused": "No puedo concretarlo ahora. Mejor dejémoslo por hoy.",
        "other": "Ese detalle no tiene que ver con el cambio que te conté.",
        "repeat": "Eso ya te lo aclaré antes.",
        "context_partial": "Estoy esperando que me confirmen el sitio. Ya mismo puedo mirar.",
        "expiry_partial": "No dura indefinidamente. Dame un momento para ver el último día.",
        "session": "Solo esta ocasión, nada más. Mantén mi opción de siempre para otro día.",
        "ongoing": "Sí, tómalo como mi preferencia habitual desde ahora.",
        "interval": "Ya lo vi, es hasta el {end} inclusive.",
    },
    "evaluation": {
        "partial": "Por ahora, sí. En un rato puedo decirte si esto se queda para más adelante.",
        "refused": "No quiero fijar eso todavía. Déjalo pendiente, por favor.",
        "other": "Por ese lado no cambia nada.",
        "repeat": "Eso ya quedó dicho, ¿te parece si seguimos?",
        "context_partial": "Aún no llego. Déjame ver el mensaje y te digo el sitio.",
        "expiry_partial": "Es por unos días. Espera que revise el calendario para decirte la fecha.",
        "session": "Es solo esta ocasión. La próxima vez vuelvo a lo de siempre.",
        "ongoing": "Quiero que lo anotes como mi preferencia habitual para después también.",
        "interval": "Aquí está: vale hasta el {end} inclusive.",
    },
}


def _turn(visit, index, role, text):
    return {"turn_id": f"v{visit}_{index}", "role": role, "text": text}


def _fact(attribute, value, kind, start, *, context=None, expiry=None, session_id=None):
    return {"attribute": attribute, "value": value, "scope": {"kind": kind,
            "valid_from": start, "valid_until_exclusive": expiry, "context": context,
            "session_id": session_id}}


def _prior(record):
    return {key: deepcopy(record[key]) for key in ("attribute", "value", "scope")}


def _update_truth(facts, query):
    """Authoring oracle independent of SQLite, writes, the parser and reader."""
    candidates = []
    for i, fact in enumerate(facts):
        scope = fact["scope"]
        when = query["date"]
        if fact["attribute"] != query["attribute"] or (scope["valid_from"] and when < scope["valid_from"]):
            continue
        if scope["valid_until_exclusive"] and when >= scope["valid_until_exclusive"]:
            continue
        if scope["kind"] == "session" and scope["session_id"] != query["session_id"]:
            continue
        if scope["context"] is not None and scope["context"] != query["context"]:
            continue
        priority = {"ongoing": 0, "contextual": 1, "interval": 2, "session": 4}[scope["kind"]]
        candidates.append((priority, i, fact["value"]))
    return max(candidates)[2] if candidates else None


def _memory_history(native, rng):
    memory = deepcopy(native["initial_memory"])
    history = deepcopy(native["prior_sessions"])
    attr = native["current_query"]["attribute"]
    start = date.fromisoformat(native["current_query"]["date"])
    older, expired = ((start - timedelta(days=d)).isoformat() for d in (21, 17))
    for other, (meal, foods) in tesis_bank.FOOD.items():
        if other == attr:
            continue
        existing = {item["value"] for item in memory if item["attribute"] == other}
        grouped = {s["date"]: s for s in history}
        if not existing:
            chosen = rng.choice(foods)
            tesis_bank._remember(memory, grouped, other, chosen, older,
                                 f"Normalmente prefiero {chosen} para {meal}.")
            existing = {item["value"] for item in memory if item["attribute"] == other}
        value = rng.choice([food for food in foods if food not in existing])
        tesis_bank._remember(memory, grouped, other, value, older,
                            f"Esa semana elegí {value} para {meal}; después volví a lo habitual.",
                            kind="interval", end=expired)
        history = list(grouped.values())
    return memory, history


def _story(native, seed):
    rng = random.Random(digest([VERSION, seed, native["root_id"]]))
    split = native["split"]
    opening, assistant, queries, revisit, temporary, choices = DISCOURSE[split]
    reply = REPLY[split]
    family = native["scenario_family"]
    truth = native["truth"]
    attr = native["current_query"]["attribute"]
    meal, foods = tesis_bank.FOOD[attr]
    first = {**native["current_query"], "session_id": "S2"}
    d1 = date.fromisoformat(first["date"])
    d2 = d1 + timedelta(days=rng.choice((0, 1, 2, 3)))
    second = {**first, "date": d2.isoformat(), "session_id": "S3", "context": rng.choice(("casa", "oficina"))}
    memory, history = _memory_history(native, rng)
    style = rng.choice(("direct", "direct", "partial", "partial", "refused"))
    impatient = rng.randrange(4) == 0
    style_hint = "Tengo muy poco tiempo." if impatient else "Tengo un minuto para hablar."
    value = truth.get("new_value")
    slot = truth["decisive_slot"]
    fact = _fact(attr, value, truth["scope"]["kind"], first["date"],
                 expiry=truth["scope"]["valid_until_exclusive"], session_id=truth["scope"]["session_id"],
                 context=truth["scope"]["context"]) if truth["scope"] else None
    alternatives = None
    if family == "F0":
        statement = rng.choice(("Sigue valiendo la opción de siempre.", "Hoy me sirve lo que suelo pedir."))
    elif family == "F1":
        statement = rng.choice((f"Ya le cogí el gusto a otra cosa. Prefiero {value} de forma habitual.",
                                f"Cambié mi elección. Desde ahora prefiero {value} de forma habitual."))
    elif family == "F2":
        end = (date.fromisoformat(fact["scope"]["valid_until_exclusive"]) - timedelta(days=1)).isoformat()
        if rng.randrange(2):
            slot = "VALID_UNTIL"
            statement = f"Estos días prefiero {value}. Es por unos días, luego vuelvo a lo de antes."
            alternatives = [fact, {**fact, "scope": {**fact["scope"], "valid_until_exclusive":
                (d1 + timedelta(days=14)).isoformat()}}]
        else:
            statement = f"Estos días prefiero {value}. Hasta el {end} inclusive. Después vuelvo a lo de siempre."
    elif family == "F3":
        if first["context"] is None:
            statement = rng.choice(("En casa y en el trabajo tengo opciones diferentes. Todavía falta el lugar.",
                                    "La elección depende de dónde esté. Aún no te dije el lugar."))
        else:
            place = "casa" if first["context"] == "casa" else "la oficina"
            statement = f"Para esta comida estoy en {place}."
    else:
        statement = rng.choice((f"Para {meal}, prefiero {value}. Hoy me apetece variar un poco.",
                                f"Me cambiaron las ganas. Para {meal}, prefiero {value}."))
        alternatives = [_fact(attr, value, "session", first["date"], session_id="S2"),
                        _fact(attr, value, "ongoing", first["date"])]

    first_messages = [_turn(1, "inicio", "user", rng.choice(opening) + " " + style_hint),
                      _turn(1, "eco", "assistant", assistant), _turn(1, "dato", "user", statement),
                      _turn(1, "consulta", "user", rng.choice(queries).format(meal=meal))]
    answer = None
    if slot == "CURRENT_CONTEXT":
        place = "casa" if truth["current_context"] == "casa" else "la oficina"
        answer = f"Ya me confirmaron. Estoy en {place}."
    if slot == "VALID_UNTIL":
        answer = reply["interval"].format(end=end)
    if slot == "TEMPORAL_SCOPE":
        answer = reply[fact["scope"]["kind"]]
    first_visit = {"query": first, "messages": first_messages, "expected": truth["current_value"],
                   "fact": fact, "alternatives": alternatives, "slot": slot, "style": style, "answer": answer}

    # A later explicit choice can mask the current reward while the earlier
    # clarification still affects the eventual default. Other visits ask to reuse it.
    second_kind = rng.choice(("reuse", "temporary", "temporary"))
    if family != "F4" and rng.randrange(3) == 0:
        second_kind = "revision"
    old = truth["initial_value"]
    selected = rng.choice([food for food in foods if food not in (old, value)])
    second_fact = second_alt = second_slot = second_answer = None
    second_intro = rng.choice(revisit)
    if d1 == d2:
        second_intro = {"train": "Hola de nuevo, vuelvo un rato más tarde.",
                        "validation": "Volví después de hacer la compra.",
                        "evaluation": "Hola, ya estoy de vuelta de la tienda."}[split]
    if second_kind == "reuse":
        second_statement = rng.choice(("¿Te acuerdas de qué suelo elegir?", "Quiero seguir con lo que ya te conté."))
    elif second_kind == "temporary":
        second_statement = rng.choice(temporary).format(value=selected)
        second_fact = _fact(attr, selected, "session", second["date"], session_id="S3")
    else:
        second_statement = f"Tengo otra opción. Prefiero {selected}. " + rng.choice(choices).format(value=selected)
        second_slot = "TEMPORAL_SCOPE"
        second_alt = [_fact(attr, selected, "session", second["date"], session_id="S3"),
                      _fact(attr, selected, "ongoing", second["date"])]
        second_fact = rng.choice(second_alt)
        second_answer = reply[second_fact["scope"]["kind"]]
    second_messages = [_turn(2, "inicio", "user", second_intro),
                       _turn(2, "eco", "assistant", "Claro, cuéntame qué necesitas para hoy."),
                       _turn(2, "dato", "user", second_statement),
                       _turn(2, "consulta", "user", rng.choice(queries).format(meal=meal))]
    facts = [_prior(r) for r in memory]
    if fact is not None:
        facts.append(fact)
    if second_fact is not None:
        facts.append(second_fact)
    second_visit = {"query": second, "messages": second_messages, "expected": _update_truth(facts, second),
                    "fact": second_fact, "alternatives": second_alt, "slot": second_slot,
                    "style": rng.choice(("direct", "partial", "refused")), "answer": second_answer}
    later = {**second, "date": (d2 + timedelta(days=8)).isoformat(), "session_id": "S4"}
    other_attr = rng.choice([a for a in tesis_bank.FOOD if a != attr])
    probes = [("reuse", later), ("unrelated_preference", {**later, "attribute": other_attr})]
    if family == "F2":
        # Check both sides of the date boundary, after closing the first visit.
        expiry = date.fromisoformat(fact["scope"]["valid_until_exclusive"])
        if d2 < expiry:
            probes.append(("last_day_of_interval", {**later, "date": (expiry - timedelta(days=1)).isoformat()}))
        if d2 <= expiry:
            probes.append(("first_day_after_interval", {**later, "date": expiry.isoformat()}))
    public_identity = {"memory": memory, "messages": first_messages, "query": first}
    root = "it" + digest(public_identity)[:24]
    world = "iw" + digest([root, native["latent_world_id"], second_fact])[:24]
    return {"root_id": root, "world_id": world, "realization_id": "v0", "split": split,
            "scenario_family": family, "second_visit_kind": second_kind, "impatient": impatient,
            "initial_memory": memory, "prior_sessions": history, "visits": [first_visit, second_visit],
            "future_queries": [{"kind": kind, "query": probe, "expected": _update_truth(facts, probe)} for kind, probe in probes],
            "template": f"{split}:{family}", "origin_root": native["root_id"],
            "recipe_id": digest([family, attr, truth["initial_value"], value, second_kind, selected])}


def build_bank(seed=DEFAULT_SEED, sizes=None):
    """New words and interaction recipes over the same food-preference domain."""
    native = tesis_bank.build_splits(sizes, seed=seed)
    return {split: [_story(row, seed) for row in rows] for split, rows in native.items()}


_REFUSAL = re.compile(r"mejor dejarlo ahí|mejor dejémoslo|déjalo pendiente|ya me tengo que ir|no puedo aclarar nada", re.I)
_PARTIAL = re.compile(r"por ahora|déjame (?:mirar|ver)|(?:tengo que|dame un momento para) (?:mirar|ver)|revise el calendario|puedo mirar", re.I)
_EXPIRY = re.compile(r"(?:por unos días|por unos dias|esta semana|unos días|unos dias)", re.I)


def _native(observation):
    return {key: observation[key] for key in ("query", "memory", "user_messages", "reuse_regime")}


def public_evidence(observation):
    """Summarize visible cues. Contains no access to the simulator or authored answers."""
    parsed = evidence_model(_native(observation))
    users = [m for m in observation["user_messages"] if m["role"] == "user"]
    text = " ".join(m["text"] for m in users)
    replies = [m["text"] for m in users if m.get("in_reply_to", "").startswith("a_tool_")]
    last = replies[-1] if replies else ""
    slot = parsed["missing_slot"]
    if slot == "TEMPORAL_SCOPE" and _EXPIRY.search(text) and _EXPIRY.search(" ".join(m["text"] for m in users if not m.get("in_reply_to"))):
        slot = "VALID_UNTIL"
    unavailable = _REFUSAL.search(last) or any("no puedo aclarar nada" in m["text"].casefold() for m in users)
    state = "unavailable" if unavailable else "partial" if _PARTIAL.search(last) and slot else "resolved" if replies and not slot else "initial"
    values = set(parsed["current_values"])
    return {"missing_slot": slot, "clarification_state": state,
            "current_resolved": len(values) == 1 and None not in values,
            "scope_ambiguous": parsed["scope_ambiguous"]}


def reference_action(observation):
    """Existing visible rule with a bounded follow-up for partial replies."""
    parsed = public_evidence(observation)
    slot = parsed["missing_slot"]
    if not observation["questions_remaining"] or parsed["clarification_state"] == "unavailable" or slot is None:
        return "PROCEED"
    if slot not in observation["asked_slots"] or parsed["clarification_state"] == "partial":
        return slot
    return "PROCEED"


class _VisitEnvironment(EpisodeEnvironment):
    """Extend delivery while retaining the native observation, store and executor."""

    def deliver(self, action, text):
        if self.question_count >= MAX_QUESTIONS:
            raise ValueError("QUESTION_BUDGET_EXCEEDED")
        self.question_count += 1
        question_id = f"a_tool_{self._current_query['session_id']}_{self.question_count}"
        prompt = QUESTIONS[action] if self.question_count == 1 else "Para asegurarme: " + QUESTIONS[action]
        event = {"turn_id": question_id, "role": "assistant", "text": prompt,
                 "about_attribute": self._current_query["attribute"]}
        reply = {"turn_id": f"r_{self._current_query['session_id']}_{self.question_count}", "role": "user",
                 "text": text, "in_reply_to": question_id, "about_attribute": event["about_attribute"]}
        self._messages.extend((event, reply))
        return deepcopy(event), deepcopy(reply)


@lru_cache(maxsize=8192)
def _decision_observation(text):
    """Reuse pure public inference without caching trajectories or hidden truth."""
    visible = json.loads(text)
    controller = RuleController()
    answer = controller.respond(visible)
    # Only confirmed operations can alter persisted memory. The evaluator
    # tracks unresolved changes separately to distinguish luck from evidence.
    proposed = [op for op in controller.write(visible)["operations"]
                if op.get("record", {}).get("status") == "confirmed"]
    approved = [op for op in proposed if audit_operation(op, visible)]
    return answer, approved or NOOP, len(proposed) - len(approved)


def _possible_values(knowledge, query):
    return {_update_truth(assignment, query) for assignment in product(*knowledge)}


def _address(query, visit, evidence):
    """An explicit delivery or authored fixed context resolves the address."""
    if query["context"] is None and visit["slot"] == "CURRENT_CONTEXT" and evidence:
        return {**query, "context": "oficina" if "oficina" in visit["answer"].casefold() else "casa"}
    return query


class InteractiveEpisode:
    """Policy sees a conversational prefix. Terminal scoring uses authored worlds."""

    def __init__(self, story, db_path=":memory:"):
        self.story = deepcopy(story)
        self._temporary = TemporaryDirectory(prefix="tesis-interactive-") if str(db_path) == ":memory:" else None
        self.db_path = Path(self._temporary.name) / "memory.sqlite" if self._temporary else Path(db_path)
        if self.db_path.exists():
            raise FileExistsError(f"Use a fresh rollout path: {self.db_path}")
        self._visits = []
        self._actions = []
        self._knowledge = [(_prior(r),) for r in story["initial_memory"]]
        self._future = None
        self._env = None
        self._open(0, self.story["initial_memory"])

    def _open(self, index, records):
        self._visit = self.story["visits"][index]
        self._events = []
        self._resolved = False
        self._attempts = Counter()
        messages = deepcopy(self._visit["messages"])
        if index and self.story["impatient"] and self._visits[-1]["failed_questions"]:
            messages.insert(2, _turn(2, "prisa", "user", "Voy de salida y hoy no puedo aclarar nada más."))
        self._env = _VisitEnvironment({"root_id": self.story["root_id"], "initial_memory": records,
                                      "user_messages": messages, "current_query": self._visit["query"],
                                      "future_query": self.story["future_queries"][0]["query"]}, self.db_path)
        self._opening = messages

    @property
    def done(self):
        return self._future is not None

    def observe(self):
        if self.done:
            raise RuntimeError("EPISODE_COMPLETE")
        view = self._env.observe("select")
        return {**view, "reuse_regime": REUSE, "visit": len(self._visits) + 1,
                "questions_remaining": MAX_QUESTIONS - len(self._events),
                "asked_slots": [event["action"] for event in self._events]}

    @property
    def legal_actions(self):
        if self.done:
            return ()
        view = self.observe()
        if not view["questions_remaining"] or any(_REFUSAL.search(m["text"]) for m in view["user_messages"] if m["role"] == "user"):
            return ("PROCEED",)
        return ACTIONS

    def _ask(self, action):
        visit = self._visit
        wording = REPLY[self.story["split"]]
        self._attempts[action] += 1
        target = visit["slot"]
        if self._resolved or self._attempts[action] > 1 and action != target:
            text, delivery = wording["repeat"], "redundant"
        elif action != target:
            if self.story["impatient"] and (self._events or target is None):
                text, delivery = "Ya me tengo que ir. Sigamos con lo que tienes.", "unavailable"
            else:
                text, delivery = wording["other"], "irrelevant"
        elif visit["style"] == "refused":
            text, delivery = wording["refused"], "unavailable"
        elif visit["style"] == "partial" and self._attempts[action] == 1:
            key = {"CURRENT_CONTEXT": "context_partial", "VALID_UNTIL": "expiry_partial"}.get(target, "partial")
            text, delivery = wording[key], "partial"
        else:
            text, delivery = visit["answer"], "resolved"
            self._resolved = True
        question, reply = self._env.deliver(action, text)
        self._events.append({"action": action, "question": question, "reply": reply, "delivery": delivery,
                             "informative": delivery == "resolved"})

    def _commit(self):
        visible = _native(self.observe())
        answer, operations, omitted = _decision_observation(canonical(visible))
        write = self._env.apply_memory_operations(operations)
        executed = write["executed_operations"]
        visit = self._visit
        if visit["fact"] is not None:
            self._knowledge.append(tuple(visit["alternatives"] if visit["alternatives"] and not self._resolved else [visit["fact"]]))
        current = score_response(answer, visible, visit["expected"])
        # Authored semantic ambiguity from earlier visits is relevant on a
        # reuse request, even if the reader happens to guess the old default.
        if len(self._visits) and self.story["second_visit_kind"] == "reuse":
            supported = current["success"] and _possible_values(self._knowledge, visit["query"]) == {answer["value"]}
        else:
            supported = current["success"]
        if visit["slot"] == "CURRENT_CONTEXT" and not self._resolved:
            supported = False
        self._env.close_session()
        record = {"query": deepcopy(visible["query"]), "effective_query": _address(visit["query"], visit, self._resolved),
                  "opening_messages": deepcopy(self._opening), "events": deepcopy(self._events),
                  "expected": visit["expected"], "outcome": deepcopy(answer),
                  "current_supported": int(supported), "current_correct": current["factual"],
                  "unsupported_commitments": int(answer["mode"] == "ANSWER" and not supported),
                  "questions": len(self._events), "invalid_outputs": int(not write["valid"] or not current["valid"]),
                  "unsupported_writes": 0, "omitted_confirmed_updates": omitted + int(not write["valid"]),
                  "informative_questions": sum(event["informative"] for event in self._events),
                  "failed_questions": sum(event["delivery"] in ("unavailable", "irrelevant", "redundant") for event in self._events),
                  "redundant_questions": sum(event["delivery"] in ("irrelevant", "redundant") for event in self._events),
                  "executed_operations": executed, "memory_before": visible["memory"],
                  "memory_after": self._env.private_trace["final_memory"]}
        self._visits.append(record)
        if len(self._visits) == 1:
            self._open(1, record["memory_after"])
            return
        self._final = self._env.reopen_for_reuse()["memory"]
        self._future = []
        for probe in self.story["future_queries"]:
            query, expected = probe["query"], probe["expected"]
            response = read(self._final, query)
            answerable = response["mode"] == "ANSWER"
            correct = answerable and response["value"] == expected
            supported = correct and _possible_values(self._knowledge, query) == {expected}
            self._future.append({"kind": probe["kind"], "query": query, "response": response, "expected": expected,
                                 "correct": int(correct), "supported": int(supported),
                                 "unsupported_commitment": int(answerable and not supported)})

    def step(self, action):
        if action not in self.legal_actions:
            raise ValueError(f"Illegal SELECT action: {action}")
        self._actions.append(action)
        if action == "PROCEED":
            self._commit()
        else:
            self._ask(action)
        return (None, True) if self.done else (self.observe(), False)

    def summary(self):
        if not self.done:
            raise RuntimeError("EPISODE_INCOMPLETE")
        return deepcopy({**{key: self.story[key] for key in ("root_id", "world_id", "split", "scenario_family", "second_visit_kind")},
                         "visits": self._visits, "future": self._future, "final_memory": self._final,
                         "actions": self._actions, "questions": sum(visit["questions"] for visit in self._visits)})

    def close(self):
        if self._env is not None:
            self._env.dispose()
        if self._temporary is not None:
            self._temporary.cleanup()
