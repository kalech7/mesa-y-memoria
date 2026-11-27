"""Public Spanish language adapter for the profile environment.

This version preserves original turns as the source of every persisted quote.
Canonical wording is an intermediate parser representation. Neither parsing nor
the action mask receives authored facts, answers, profile names or future visits.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import date
from functools import lru_cache
import json
import re

from research.src.artifacts import canonical
from research.src.policy.rule_controller import RuleController, evidence_model as native_evidence
from research.src.scoring.evidence import audit_operation
from research.tesis_extended_env import evaluate
from research.tesis_interactive_env import MAX_QUESTIONS, _native, _possible_values, public_evidence as native_public_evidence
from research.tesis_profiles_env import ProfilesEpisode, _visible_timing, outcome
from research.tesis_shared_budget_env import ACTIONS, QUESTION_ACTIONS
from research.tesis_shared_budget_planner import Decision, WorldTree
from research.tesis_shared_budget_qwen_context import _plain


VERSION = "tesis_profiles_parser_v5"
MONTHS = {name: index for index, name in enumerate((
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
    "septiembre", "octubre", "noviembre", "diciembre"), 1)}
MEALS = {"dinner_preference": "la cena", "breakfast_preference": "el desayuno",
         "drink_preference": "la bebida"}
ATTRIBUTE = re.compile(r"\b(cena(?:r|s)?|desayun(?:o|os|ar)|bebidas?|beber)\b", re.I)
QUESTION = re.compile(r"¿[^?]*\?|[^.!?¿]*\?", re.S)
QUOTED = re.compile(r"«[^»]*»|“[^”]*”|\"[^\"]*\"")
THIRD_PERSON = re.compile(
    r"\b(?:mi (?:herman[oa]|pareja|madre|padre|sobrin[oa]|amig[oa]|hij[oa])|"
    r"(?:para|a) (?:ella|ellos|ellas)|para el mismo|(?:ella|el) (?:lo |la )?(?:pidio|quiere|prefiere)|"
    r"me (?:dijeron|escribieron|pidieron)|(?:dijo|escribio|comento|pidio|prefiere) para (?:si|ella|el))\b"
)
PREPARATION = re.compile(
    r"^(?:helar(?:la|lo)?|enfriar(?:la|lo)?|calentar(?:la|lo)?|hervir(?:la|lo)?|"
    r"servir(?:la|lo)?|(?:tomar|beber|comer)(?:la|lo|las|los)|"
    r"tomar\s+(?:fri[ao]|caliente)|preparar(?:la|lo)?)\b"
)
PENDING = re.compile(r"\b(?:dame|dejame|necesito|tengo que|falta|antes de).{0,65}"
                     r"(?:momento|segundo|mirar|revisar|confirmar|precisar|agenda|calendario)\b")
REFUSAL = re.compile(r"\b(?:mejor (?:dejarlo|dejemoslo)|dejalo pendiente|"
                     r"no (?:puedo|quiero|voy a) (?:precisar|aclarar|concretar|contestar)|"
                     r"no (?:lo )?(?:se|tengo decidido)|sigamos con lo que)\b")


def _timing(observation):
    cues = _visible_timing(observation)
    text = _plain(" ".join(turn["text"] for turn in observation["user_messages"]
                            if turn.get("role") == "user"))
    limited = bool(re.search(r"\b(?:se me va el (?:bus|tren)|no voy a poder seguir|"
                             r"estoy (?:pagando|conduciendo|manejando)|tengo que (?:salir|irme)|"
                             r"me llaman de|voy de salida)\b", text))
    available = bool(re.search(r"\b(?:ya termine lo urgente|tengo (?:un hueco|unos minutos)|"
                               r"puedo seguir en el chat|sigo leyendo|estoy libre|sin apuro)\b", text))
    if cues["availability"] == "sin indicio claro" and limited != available:
        cues["availability"] = "limitada" if limited else "disponible"
    cues["expressed_return"] = cues["expressed_return"] or bool(re.search(
        r"\b(?:cuando (?:vuelva|retome|reabra)|en el (?:proximo|siguiente) chat|"
        r"al (?:volver|reabrir)|luego (?:te|podre) (?:confirmo|contesto|respondo|revisar))\b", text))
    return cues


def _attribute(text, default):
    found = ATTRIBUTE.search(_plain(text))
    if found is None:
        return default
    word = found.group(1)
    return "breakfast_preference" if word.startswith("desayun") else (
        "dinner_preference" if word.startswith("cena") else "drink_preference")


def _date_words(text, when):
    """Resolve explicit Spanish calendar expressions at their visible date."""
    anchor = date.fromisoformat(when)
    months = "|".join(MONTHS)
    pattern = rf"\b(\d{{1,2}}) de ({months})(?: de (\d{{4}}))?\b"
    def absolute(match):
        try:
            resolved = date(int(match.group(3) or anchor.year), MONTHS[_plain(match.group(2))],
                            int(match.group(1)))
            prefix = _plain(text[:match.start()].rsplit(".", 1)[-1])
            if match.group(3) is None and resolved < anchor and re.search(r"\b(?:empieza|comienza|empezara|comenzara)\b", prefix):
                resolved = resolved.replace(year=anchor.year + 1)
            return resolved.isoformat()
        except ValueError:
            return match.group()
    text = re.sub(pattern, absolute, text, flags=re.I)
    def month_day(match):
        try:
            return "hasta el " + anchor.replace(day=int(match.group(1))).isoformat()
        except ValueError:
            return match.group()
    text = re.sub(r"\bhasta (?:el )?(?:d[ií]a )?(\d{1,2})(?=\s+inclusive\b)", month_day, text, flags=re.I)
    return text


def _scope_words(text):
    text = re.sub(r"\b(?:[uú]nicamente|solamente)\b", "solo", text, flags=re.I)
    text = re.sub(r"\b(?:esta vez|por esta vez|para esta vez|por esa sola sesi[oó]n|"
                  r"para aquella sesi[oó]n|para esta sesi[oó]n solamente|"
                  r"solo (?:va|cuenta|corresponde) para esta sesi[oó]n)\b",
                  "solo para esta sesión", text, flags=re.I)
    text = re.sub(r"\b(?:como mi (?:cena|bebida|desayuno) habitual|"
                  r"(?:mi |la )?(?:cena|bebida|desayuno) habitual|"
                  r"como (?:mi )?opci[oó]n habitual|como h[aá]bito|para las pr[oó]ximas veces)\b",
                  "de forma habitual", text, flags=re.I)
    return text


def _safe_clauses(text):
    text = QUESTION.sub(" ", text)
    # The first person inside reported speech belongs to its quoted speaker.
    text = QUOTED.sub(" ", text)
    pieces = re.split(r"[.!?;\n]+|\b(?:pero|en cambio|mientras que|y ahora)\b", text, flags=re.I)
    safe = []
    for piece in pieces:
        piece = piece.strip(" ,:")
        if not piece:
            continue
        plain = _plain(piece)
        own = re.search(r"\b(?:para mi|yo (?:prefiero|quiero)|mi (?:cena|bebida|desayuno))\b", plain)
        if THIRD_PERSON.search(plain) and not own:
            continue
        if re.search(r"\b(?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)\s+(?:me )?(?:pidió|quiere|prefiere|dijo|escribió)\b", piece) and not own:
            continue
        prep = re.search(r"\b(?:prefiero|quiero)\s+(.+)", plain)
        if prep and PREPARATION.search(prep.group(1)):
            continue
        safe.append(piece)
    return safe


def _assertion(clause, default):
    """Return a canonical positive assertion, or preserve an unresolved issue."""
    attribute = _attribute(clause, default)
    patterns = (
        r"\bprefiero\s+(.+)",
        r"\b(?:me (?:apetece|gusta)|se me antoja|me quedo con|elijo|escojo)\s+(.+)",
        r"\bme gustar[ií]a\s+(.+)",
        r"\b(?:quiero|anota|guarda|registra|conserva|pon|deja)\s+(.+?)(?=\s+como\s+(?:mi\s+)?(?:cena|bebida|desayuno|preferencia|opci[oó]n)|$)",
        r"\bmi\s+(?:nuevo\s+)?(?:cena|bebida|desayuno|opci[oó]n|preferencia)\s+habitual\s+(?:es|ser[aá]|queda(?:r[aá])?\s+como)\s+(.+)",
        r"\bcambi[eé]\s+mi\s+(?:elecci[oó]n|opci[oó]n|preferencia)\s+habitual(?:\s+para\s+(?:cenar|desayunar|beber))?\s+a\s+(.+)",
    )
    for pattern in patterns:
        match = re.search(pattern, clause, re.I)
        if match is None:
            continue
        value = match.group(1).strip(" ,:")
        if PREPARATION.search(_plain(value)):
            return None
        if re.match(r"(?:lo (?:que|habitual|anterior|de siempre|de antes)|"
                    r"la (?:opci[oó]n|bebida|cena|ficha|pauta|rutina|rectificaci[oó]n|correcci[oó]n|elecci[oó]n)|"
                    r"el (?:dato|plato|h[aá]bito)|mi (?:habitual|propia)|"
                    r"(?:volver a|seguir|repasar|mantener|conservar)\b)", value, re.I):
            return None
        if re.search(r"\b(?:no|nunca|jam[aá]s|quiz[aá]s?|tal vez|puede que|podr[ií]a)\b.{0,30}(?:prefiero|quiero|elijo|apetece|gusta|gustar[ií]a|antoja|quedo|escojo)\b", clause, re.I):
            return f"Para {MEALS[attribute]}, no prefiero {value}."
        if re.search(r"\bme gustar[ií]a\b", clause, re.I):
            return f"Para {MEALS[attribute]}, quizá prefiero {value}."
        value = re.sub(r"\s+(?:como|de)\s+(?:mi\s+)?(?:bebida|desayuno|cena)\b.*$", "", value, flags=re.I)
        value = re.sub(r"\s+para\s+(?:la cena|el desayuno|la bebida|cenar|desayunar|beber)\b", "", value, flags=re.I)
        if not value:
            return None
        text = f"Para {MEALS[attribute]}, prefiero {value}."
        plain = _plain(clause)
        if re.search(r"\b(?:habitual|habitualmente|normalmente|siempre|suelo|todos los dias)\b", plain) and not re.search(r"\b(?:por ahora|por unos dias|hasta|esta sesion|esta ocasion)\b", plain):
            text += " De forma habitual."
        context = re.search(r"\b(?:en|para|cuando estoy en)\s+(?:la\s+)?(casa|oficina)\b", plain)
        if context:
            text = text.replace("prefiero ", f"en {context.group(1)}, prefiero ", 1)
        return text
    return None


def _normalize_turn(turn, query):
    raw = _date_words(turn["text"], turn.get("date") or query["date"])
    clauses = _safe_clauses(raw)
    declarations = []
    topic = query["attribute"]
    for index, clause in enumerate(clauses):
        topic = _attribute(clause, topic)
        statement = _assertion(clause, topic)
        if statement:
            declarations.append((index, statement))
    normalized = []
    for position, (chosen, statement) in enumerate(declarations):
        text = statement
        left = 0 if position == 0 else chosen
        right = declarations[position + 1][0] if position + 1 < len(declarations) else len(clauses)
        local_clauses = clauses[left:right]
        local_text = ". ".join(local_clauses)
        # Scope can precede the first assertion, e.g. a bounded workshop.
        # Later declarations have their own scope and remain chronological.
        scope = []
        for clause in local_clauses:
            clause = _scope_words(clause)
            for pattern in (r"\b(?:desde|hasta) (?:el )?\d{4}-\d{2}-\d{2}(?:\s+inclusive)?",
                            r"\bdurante (?:\d+|un|dos|tres) d[ií]as?",
                            r"\b(?:solo )?(?:en|para) esta (?:sesi[oó]n|ocasi[oó]n|conversaci[oó]n)",
                            r"\b(?:por unos d[ií]as|por ahora|de ahora en adelante|a partir de ahora)\b"):
                scope.extend(match.group() for match in re.finditer(pattern, clause, re.I))
        future = re.search(r"\b(?:empieza|comienza|empezar[aá]|comenzar[aá]|desde|a partir de|a partir del) (?:el )?(\d{4}-\d{2}-\d{2})", local_text, re.I)
        if future:
            scope.append("desde el " + future.group(1))
        restriction = re.search(r"\b(?:solo|[uú]nicamente).*?\b(?:en|para) (?:la )?(casa|oficina)\b", local_text, re.I)
        if restriction:
            scope.append("se aplica en " + restriction.group(1).lower())
        text += " " + ". ".join(dict.fromkeys(scope))
        identifier = turn["turn_id"] if len(declarations) == 1 else f"{turn['turn_id']}#v5_{position}"
        normalized.append({**turn, "turn_id": identifier, "text": _scope_words(text).strip()})
    if not normalized:
        normalized.append({**turn, "text": _scope_words(". ".join(clauses)).strip()
                           or "Sin una nueva declaración personal."})
    # Explicit current location and preference restrictions remain distinct.
    locations = []
    for clause in clauses:
        plain = _plain(clause)
        if re.search(r"\b(?:no|si|quiza|tal vez|podria)\b", plain):
            continue
        match = re.search(r"\b(?:estoy en|me quedo en|hoy (?:trabajo|estare) en|(?:hare|trabajo) desde|"
                          r"vuelvo a|toca volver (?:al escritorio de|a)|me asignaron el escritorio de)\s+"
                          r"(?:la )?(casa|oficina)\b", plain)
        if match:
            locations.append(match.group(1))
    if locations:
        normalized[-1]["text"] += f". Estoy en {locations[-1]}."
    return normalized


@lru_cache(maxsize=16384)
def _normalized(text):
    observation = json.loads(text)
    observation["user_messages"] = [message for turn in observation["user_messages"] for message in
        (_normalize_turn(turn, observation["query"]) if turn.get("role", "user") == "user" else [turn])]
    return observation


def normalized_observation(observation):
    """Expose the auditable parser representation using only native public keys."""
    return deepcopy(_normalized(canonical(_native(observation))))


def evidence_model(observation):
    return native_evidence(normalized_observation(observation))


def public_evidence(observation):
    parsed = native_public_evidence(normalized_observation(observation))
    replies = [turn for turn in observation["user_messages"] if turn.get("role") == "user"
               and turn.get("in_reply_to", "").startswith("a_tool_")]
    if replies:
        last = _plain(replies[-1]["text"])
        if REFUSAL.search(last):
            parsed["clarification_state"] = "unavailable"
        elif parsed["missing_slot"] and PENDING.search(last):
            parsed["clarification_state"] = "partial"
    return parsed


def public_features(observation, legal=()):
    parsed = public_evidence(observation)
    return {**_timing(observation), "visit": observation["visit"],
            "questions_remaining": observation["questions_remaining"],
            "clarification_state": parsed["clarification_state"],
            "current_resolved": parsed["current_resolved"], "missing_slot": parsed["missing_slot"],
            "legal_actions": tuple(legal)}


def rule_action(observation, legal):
    """Ask for visible unresolved evidence, including the generic fallback."""
    parsed = public_evidence(observation)
    slot = parsed["missing_slot"]
    return slot if slot in legal and parsed["clarification_state"] != "unavailable" else "PROCEED"


@lru_cache(maxsize=16384)
def _fixed_decision(text):
    raw = json.loads(text)
    parsed = normalized_observation(raw)
    controller = RuleController()
    response = controller.respond(parsed)
    originals = {turn["turn_id"]: turn["text"] for turn in raw["user_messages"]}
    candidates = []
    for operation in controller.write(parsed)["operations"]:
        if operation.get("record", {}).get("status") != "confirmed":
            continue
        approved = audit_operation(operation, parsed)
        operation = deepcopy(operation)
        references = list(dict.fromkeys(_original_id(reference, originals)
                                        for reference in operation["record"]["source_turn_ids"]))
        operation["record"]["source_turn_ids"] = references
        operation["record"]["evidence_quotes"] = [{"turn_id": reference, "text": originals[reference]}
                                                    for reference in references]
        candidates.append((operation, approved))
    return response, tuple(candidates)


def _original_id(identifier, originals):
    return identifier if identifier in originals else re.sub(r"#v5_\d+$", "", identifier)


class ProfilesEpisodeV5(ProfilesEpisode):
    """The V4 episode with the public parser adapter at its normal entry points."""

    def observe(self):
        visible = super().observe()
        timing = _timing(visible)
        visible["strategy_cue"] = tuple(timing[key] for key in (
            "availability", "later_plan", "expressed_return", "observed_recontact"))
        return visible

    @property
    def legal_actions(self):
        if self.done:
            return ()
        visible = self.observe()
        parsed = public_evidence(visible)
        _, candidates = _fixed_decision(canonical(_native(visible)))
        closing = ("PROCEED",)
        if any(approved and operation["record"]["scope"]["kind"] == "session"
               for operation, approved in candidates):
            closing += ("PROVISIONAL",)
        if not parsed["current_resolved"]:
            closing += ("DEFER",)
        slot = parsed["missing_slot"]
        if (not visible["questions_remaining"] or len(self._events) >= MAX_QUESTIONS
                or slot not in QUESTION_ACTIONS or parsed["clarification_state"] == "unavailable"
                or slot in visible["asked_slots"] and parsed["clarification_state"] != "partial"):
            return closing
        return tuple(action for action in ACTIONS if action in (*closing, slot))

    def _finish_visit(self, action):
        raw = _native(self.observe())
        parsed = normalized_observation(raw)
        response, candidates = _fixed_decision(canonical(raw))
        response = (deepcopy(response) if action != "DEFER" else
                    {"mode": "DEFER", "value": None, "branches": [], "evidence_ids": []})
        proposed = self._propose_writes(deepcopy(candidates), raw) if action == "PROCEED" else ()
        accepted = [(operation, approved) for operation, approved in proposed
                    if approved or self.verifier == "off"]
        written = self._env.apply_memory_operations([deepcopy(op) for op, _ in accepted] or [{"op": "NOOP"}])
        trace = self._visit_trace(action, parsed, response, proposed, accepted, written)
        original_ids = {turn["turn_id"] for turn in raw["user_messages"]}
        original_ids.update(record["memory_id"] for record in raw["memory"])
        trace["outcome"]["evidence_ids"] = list(dict.fromkeys(
            _original_id(identifier, original_ids) for identifier in trace["outcome"]["evidence_ids"]))
        if self._visit["fact"] is None and trace["current_supported"]:
            if _possible_values(self._knowledge, trace["effective_query"]) != {response["value"]}:
                trace["current_supported"] = 0
                trace["unsupported_commitments"] = 1
        self._visits.append(trace)
        if len(self._visits) < len(self.story["visits"]):
            self._env.dispose()
            self._open(len(self._visits), trace["memory_after"])
        else:
            self._finish_episode()


def execute(story, prefix, *, episode_class=ProfilesEpisodeV5, **options):
    from research.tesis_profiles_env import execute as replay
    return replay(story, prefix, episode_class=episode_class, **options)


def simulate(story, prefix, *, episode_class=ProfilesEpisodeV5, **options):
    result = execute(story, prefix, episode_class=episode_class, **options)
    return result if isinstance(result, Decision) else outcome(result)


def enumerate_world(story):
    pending, decisions, outcomes, steps = [()], {}, {}, 0
    while pending:
        prefix = pending.pop()
        result = simulate(story, prefix)
        steps += len(prefix)
        if isinstance(result, Decision):
            decisions[prefix] = result
            pending.extend(prefix + (action,) for action in result.legal)
        else:
            outcomes[prefix] = result
    return WorldTree(story, decisions, outcomes, steps)


def coverage_diagnostic(rows):
    """Compare parser openings in executable worlds, after earlier PROCEEDs.

    Authored slots are consulted here as diagnostic annotations after replay.
    They never enter either parser, controller or legality calculation.
    """
    from research.tesis_profiles_env import execute as execute_v4, public_features as features_v4
    grouped = defaultdict(list)
    for story in rows:
        for index, visit in enumerate(story["visits"]):
            result = {"world_id": story["world_id"], "expected_slot": visit.get("slot")}
            for name, replay, features in (("v4", execute_v4, features_v4), ("v5", execute, public_features)):
                node = replay(story, ("PROCEED",) * index, question_budget=2)
                offered = [action for action in node.legal if action in QUESTION_ACTIONS]
                data = {"offered": offered, "parser_slot": features(node.observation)["missing_slot"]}
                slot = visit.get("slot")
                if slot in offered and visit.get("style") == "partial":
                    following = replay(story, ("PROCEED",) * index + (slot,), question_budget=2)
                    data["partial_followup_available"] = slot in following.legal
                result[name] = data
            grouped[(story["root_id"], index + 1)].append(result)
    details = [{"root_id": root, "visit": visit, "worlds": worlds}
               for (root, visit), worlds in sorted(grouped.items())]
    summary = {}
    for name in ("v4", "v5"):
        opportunities = [row for row in details if any(world["expected_slot"] for world in row["worlds"])]
        summary[name] = {
            "opportunities": len(opportunities),
            "opened_in_all_worlds": sum(all(world["expected_slot"] in world[name]["offered"]
                                             for world in row["worlds"]) for row in opportunities),
            "unexpected_openings": sum(any(world[name]["offered"] and world["expected_slot"] is None
                                               for world in row["worlds"]) for row in details),
            "partial_followups": sum(any(world[name].get("partial_followup_available", False)
                                           for world in row["worlds"]) for row in details),
        }
    return {"version": VERSION, "roots": len({row["root_id"] for row in rows}),
            "visits": len(details), "summary": summary, "details": details}
