"""B4: explicit public grammar, operating on exactly the actor's observation.

This is a conventional baseline for the development language, not a hidden-state
oracle. Unsupported language remains uncertain. The evidence solver also defines
the small finite hypothesis space used by the development evaluator.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re


def _scope(kind, when, *, end=None, context=None, session=None):
    return {"kind": kind, "session_id": session, "context": context,
            "valid_from": when, "valid_until_exclusive": end}


def _text_messages(observation):
    return [m for m in observation.get("user_messages", [])
            if isinstance(m, dict) and isinstance(m.get("text"), str)
            and m.get("role", "user") == "user"]


def _positive_wording(text):
    # This small explicit construction expresses continuation, not a negative
    # preference. Evidence quotes always retain the original user wording.
    return re.sub(r"^(\s*)no he dejado de preferir\b", r"\1prefiero", text, flags=re.I)


def _definite_reply(message):
    from .dialogue import personal_wording
    from ..env.user_simulator import response_availability
    if response_availability({"text": message["text"]}) != "available":
        return False
    text = personal_wording(message["text"])
    # A standalone "No," can reject the preceding alternative while its
    # following clause still states a definite scope.
    text = re.sub(r"^\s*no\s*,\s*", "", text, flags=re.I)
    return not re.search(r"\b(?:no|nunca|jamás|quizá|quizás|quizas|tal vez|podría|puede que)\b",
                         text, re.I)


def _preference_language_issue_single(observation):
    """Fail closed at unsupported polarity/temporal language boundaries.

    The schema represents positive preferences; it cannot persist an exclusion
    as if it were a positive food choice. A missing start date is not 'today'.
    """
    messages = _text_messages(observation)
    if not messages:
        return None
    from .dialogue import personal_wording
    text = _positive_wording(personal_wording(messages[0]["text"])).split("¿", 1)[0]
    negative = (re.search(r"\b(?:no|nunca|jamás)\s+(?:\w+\s+){0,4}(?:prefiero|preferir|preferencia)\b", text, re.I)
                or re.search(r"\bprefiero\s+(?:no|nunca|jamás)\b", text, re.I))
    if negative:
        return "unsupported_negative_preference"
    if re.search(r"\bprefiero\b", text, re.I) and re.search(r"\b(?:no|nunca|jamás|quizá|quizás|quizas|tal vez|podría|puede que)\b", text, re.I):
        return "unresolved_preference_assertion"
    if re.search(r"\b(?:después|luego|más adelante)\b.*\bprefiero\b", text, re.I):
        return "unanchored_preference_start"
    return None


def preference_language_issue(observation):
    from .dialogue import is_dialogue, focus
    if is_dialogue(observation):
        observation = focus(observation)[0]
    return _preference_language_issue_single(observation)


def _new_fact_single(observation):
    from .dialogue import personal_wording, preference_value, temporal_scope

    messages = _text_messages(observation)
    if not messages:
        return None, None, None
    first = messages[0]
    text = _positive_wording(personal_wording(first["text"])).strip()
    informative_replies = [m for m in messages[1:] if _definite_reply(m)]
    later = " ".join(m["text"] for m in informative_replies)
    combined = text + " " + later
    query = observation.get("query", {})
    context = query.get("context")
    if context is None:
        found_context = re.search(r"(?:estoy en|ceno en|hoy en)\s+(?:la\s+)?(casa|oficina)", combined, re.I)
        if found_context:
            context = found_context.group(1).lower()
    issue = preference_language_issue(observation)
    if issue in {"unsupported_negative_preference", "unresolved_preference_assertion"}:
        return None, context, "GENERIC"
    match = re.search(r"prefiero\s+([^.!?]+?)(?=\s+de forma habitual|[.!?]|$)", text, re.I)
    alternate = re.search(r"preferencia habitual desde hoy es\s+([^.!?]+)", text, re.I)
    if not match and not alternate:
        return None, context, None
    value = preference_value((match or alternate).group(1).strip())
    # A place in the preference clause restricts the preference. It does not
    # disclose the user's current place, which may still require a question.
    assertion = text.split("¿", 1)[0]
    preference_sentence = next((part for part in assertion.split(".") if re.search(r"prefiero|preferencia habitual desde hoy es", part, re.I)), assertion)
    restriction = re.search(r"\ben\s+(?:la\s+)?(casa|oficina)\b", preference_sentence, re.I)
    preference_context = restriction.group(1).lower() if restriction else None
    if preference_context:
        value = re.sub(r"\s+en\s+(?:la\s+)?(?:casa|oficina)\s*$", "", value, flags=re.I).strip()
    # An explicitly named meal in the assertion identifies its attribute. The
    # current question can concern another attribute (negative-control task).
    # Infer this from the statement's visible words, never from scenario labels.
    statement = text.split(".", 1)[0]
    attribute = query.get("attribute", "dinner_preference")
    if re.search(r"\b(?:cena|cenar)\b", statement, re.I):
        attribute = "dinner_preference"
    elif re.search(r"\b(?:bebida|beber)\b", statement, re.I):
        attribute = "drink_preference"
    elif re.search(r"\b(?:desayuno|desayunar)\b", statement, re.I):
        attribute = "breakfast_preference"
    if not value:
        return None, context, "GENERIC"
    record_scope, missing, scope_sources = temporal_scope([first, *informative_replies], query, preference_context)
    if record_scope is None:
        return None, context, missing
    if preference_context and (record_scope["kind"] in {"ongoing", "interval"} or
                               (record_scope["kind"] == "unknown" and not scope_sources
                                and not re.search(r"\bahora\b", preference_sentence, re.I))):
        record_scope["kind"] = "contextual"
    if issue == "unanchored_preference_start":
        record_scope = _scope("unknown", None, context=preference_context)
    sources = [first]
    sources.extend(message for message in scope_sources if message is not first)
    record = {"attribute": attribute, "value": value,
              "scope": record_scope, "status": "unresolved" if record_scope["kind"] == "unknown" else "confirmed",
              "source_turn_ids": [m["turn_id"] for m in sources],
              "evidence_quotes": [{"turn_id": m["turn_id"], "text": m["text"]} for m in sources]}
    return record, context, "TEMPORAL_SCOPE" if record_scope["kind"] == "unknown" else None


def _new_fact(observation):
    from .dialogue import is_dialogue, parse_dialogue
    if is_dialogue(observation):
        return parse_dialogue(observation, _new_fact_single)
    return _new_fact_single(observation)


def _apply_authorized(memory, operations):
    result = deepcopy(memory)
    for index, operation in enumerate(operations):
        if operation["op"] == "NOOP":
            continue
        record = deepcopy(operation["record"])
        record["memory_id"] = "observed_u2" if len(operations) == 1 else f"observed_u2_{index}"
        record["recorded_at"] = record["scope"].get("valid_from") or ""
        record["supersedes"] = list(operation.get("target_ids", []))
        result.append(record)
    return result


def _authorized_operations_single(observation):
    """Canonical faithful writes justified by this visible public grammar."""
    fact, _, _ = _new_fact(observation)
    if fact is None or preference_language_issue(observation):
        return [{"op": "NOOP"}]
    attr = fact["attribute"]
    # Do not rewrite an equivalent record on every encounter.
    if any(all(item.get(k) == fact.get(k) for k in ("attribute", "value", "scope", "status"))
           for item in observation.get("memory", [])):
        return [{"op": "NOOP"}]
    kind = fact["scope"]["kind"]
    targets = [r["memory_id"] for r in observation.get("memory", [])
               if r.get("attribute") == attr and r.get("scope", {}).get("kind") == kind
               and r.get("scope", {}).get("context") == fact["scope"].get("context")
               and r.get("scope", {}).get("valid_until_exclusive") is None]
    replaces = kind in {"ongoing", "contextual"} and fact["scope"].get("valid_until_exclusive") is None
    operation = "MARK_UNRESOLVED" if kind == "unknown" else "SUPERSEDE" if replaces and targets else "ADD"
    payload = {"op": operation, "record": fact}
    if operation == "SUPERSEDE":
        payload["target_ids"] = targets
    return [payload]


def authorized_operations(observation):
    from .dialogue import is_dialogue, declared_attributes
    if not is_dialogue(observation):
        return _authorized_operations_single(observation)
    operations = []
    # A conversation may contain two independent preference updates. The
    # declared atomic WRITE budget remains two; the queried attribute comes first.
    for attribute in declared_attributes(observation)[:2]:
        target = deepcopy(observation)
        target["query"]["attribute"] = attribute
        operations.extend(op for op in _authorized_operations_single(target) if op["op"] != "NOOP")
    return operations or [{"op": "NOOP"}]


def expected_memory(observation):
    """Observable evidence retained faithfully; no hidden-world label is used."""
    return _apply_authorized(observation.get("memory", []), authorized_operations(observation))


def _read(memory, query):
    from research.src.readers.symbolic import read
    return read(memory, query)


def compatible_worlds(observation, query=None):
    """Finite possible answers after the evidence actually visible to the actor.

    The optional query can be historical/future while relative dates in the user
    evidence remain anchored at observation.query (the actual acquisition date).
    Unknown temporal updates branch into session and ongoing hypotheses.
    """
    effective_query = deepcopy(query or observation.get("query", {}))
    fact, context, missing_slot = _new_fact(observation)
    if preference_language_issue(observation) or (fact is None and missing_slot is not None):
        return [{"value": None, "evidence_ids": [m["turn_id"] for m in _text_messages(observation)]}]
    if query is None and context is not None:
        effective_query["context"] = context
    memory = expected_memory(observation)
    # A current utterance establishes today's chosen value even if future
    # applicability is unresolved. It does not authorize an ongoing write.
    from ..readers.symbolic import applicability
    if (fact and query is None and fact["attribute"] == effective_query.get("attribute")
            and fact["scope"]["kind"] == "unknown"
            and applicability(fact, effective_query) is not False
            and (fact["scope"].get("context") is None or effective_query.get("context") is not None)):
        return [{"value": fact["value"], "evidence_ids": list(fact["source_turn_ids"])}]
    contextual = [r for r in memory if r.get("attribute") == effective_query.get("attribute")
                  and r.get("scope", {}).get("kind") == "contextual"]
    queries = [effective_query]
    if contextual and effective_query.get("context") is None:
        contexts = {json.dumps(r["scope"]["context"], sort_keys=True): r["scope"]["context"] for r in contextual}
        queries = [{**effective_query, "context": contexts[key]} for key in sorted(contexts)]
        # The listed places are not an exhaustive statement of where the user
        # may be. Preserve the unresolved query as well (including a general
        # fallback outside a documented context).
        queries.append(effective_query)
    candidates = []
    # Only a newly received utterance has a known acquisition session. A
    # persisted unknown-duration record must not be re-anchored to whichever
    # query happens to read it (especially the new S3 session).
    unknown = [r for r in memory if r.get("scope", {}).get("kind") == "unknown"
               and r.get("memory_id", "").startswith("observed_u2")]
    alternatives = [memory]
    if unknown:
        alternatives = []
        from itertools import product
        unknown_ids = [r["memory_id"] for r in unknown]
        for choices in product((False, True), repeat=len(unknown_ids)):
            choices = dict(zip(unknown_ids, choices))
            alternative = deepcopy(memory)
            for record in alternative:
                if record.get("scope", {}).get("kind") != "unknown" or record.get("memory_id") not in choices:
                    continue
                record["status"] = "confirmed"
                if choices[record["memory_id"]]:
                    record["scope"]["kind"] = "contextual" if record["scope"].get("context") is not None else "ongoing"
                    record["supersedes"] = [r["memory_id"] for r in alternative
                        if r["memory_id"] != record["memory_id"] and r.get("attribute") == record["attribute"]
                        and r.get("scope", {}).get("kind") == record["scope"]["kind"]
                        and r.get("scope", {}).get("context") == record["scope"].get("context")]
                else:
                    record["scope"]["kind"] = "session"
                    record["scope"]["session_id"] = observation.get("query", {}).get("session_id", "S2")
            alternatives.append(alternative)
    for alternate in alternatives:
        for q in queries:
            response = _read(alternate, q)
            sources = []
            for mid in response.get("evidence_ids", []):
                if mid.startswith("observed_u2"):
                    sources.extend(next(r for r in alternate if r["memory_id"] == mid)["source_turn_ids"])
                else:
                    sources.append(mid)
            sources = list(dict.fromkeys(sources))
            if response.get("mode") == "ANSWER":
                candidates.append({"value": response["value"], "evidence_ids": sources})
            else:
                values = [b.get("value") for b in response.get("branches", [])]
                candidates.extend({"value": value, "evidence_ids": sources} for value in values)
                if not values:
                    candidates.append({"value": None, "evidence_ids": sources})
    unique = {}
    for candidate in candidates:
        unique.setdefault(candidate["value"], candidate)
    return list(unique.values())


def evidence_model(observation):
    fact, context, slot = _new_fact(observation)
    attribute = observation.get("query", {}).get("attribute")
    if fact is not None and fact["attribute"] != attribute:
        slot = None
    new_context_rule = bool(fact and fact["attribute"] == attribute and fact["scope"].get("context") is not None)
    if slot is None and context is None and (new_context_rule or any(r.get("scope", {}).get("context") is not None
                                               and r.get("attribute") == attribute
                                               for r in observation.get("memory", []))):
        slot = "CURRENT_CONTEXT"
    return {"records": expected_memory(observation),
            "current_values": [w["value"] for w in compatible_worlds(observation)],
            "missing_slot": slot, "user_context": context, "new_record": fact,
            "scope_ambiguous": bool(fact and fact["scope"]["kind"] == "unknown"),
            "language_issue": preference_language_issue(observation)}


class RuleController:
    method = "B4"

    def select(self, observation):
        model = evidence_model(observation)
        return {"action": "ASK", "slot": model["missing_slot"]} if model["missing_slot"] else {"action": "PROCEED", "slot": None}

    def select_forced_ask(self, observation):
        """C-ALWAYS diagnostic: use the same visible variable selection rule."""
        return {"action": "ASK", "slot": evidence_model(observation)["missing_slot"] or "GENERIC"}

    def respond(self, observation):
        worlds = compatible_worlds(observation)
        values = {w["value"] for w in worlds}
        sources = sorted({source for w in worlds for source in w["evidence_ids"]})
        if len(values) == 1 and None not in values:
            return {"mode": "ANSWER", "value": next(iter(values)), "branches": [], "evidence_ids": sources}
        if values - {None}:
            interpreted = _read(expected_memory(observation), observation["query"])
            branches = interpreted.get("branches", [])
            if {b["value"] for b in branches} != values - {None}:
                branches = [{"condition": {"compatible_value": value}, "value": value}
                            for value in sorted(values - {None})]
            return {"mode": "CONDITIONALIZE", "value": None,
                    "branches": branches, "evidence_ids": sources}
        return {"mode": "DEFER", "value": None, "branches": [], "evidence_ids": sources}

    def write(self, observation):
        return {"operations": authorized_operations(observation)}
