"""Public, chronological evidence for explicitly role-tagged conversations.

Assistant turns are context/questions, never preference evidence. This remains
a declared finite grammar, not a claim to interpret arbitrary conversation.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import re


SCOPE_WORDING = (
    ("ambiguous_end", r"\bhasta ma[ñn]ana\b(?!\s*,?\s*inclusive\b)"),
    ("absolute_end", r"\bhasta\s+(?:el\s+)?(\d{4}-\d{2}-\d{2})\s*,?\s*inclusive\b"),
    ("inclusive_tomorrow", r"\bhasta ma[ñn]ana\s*,?\s*inclusive\b"),
    ("duration", r"\bdurante\s+(\d{1,4}|un|uno|dos|tres)\s+d[ií]as?\b"),
    ("absolute_day", r"\b(?:s[oó]lo\s+)?(?:para|por)\s+el\s+(\d{4}-\d{2}-\d{2})\b"),
    ("today", r"\b(?:(?:s[oó]lo\s+)?(?:para|por)\s+hoy|s[oó]lo\s+hoy"
     r"|(?:s[oó]lo\s+)?(?:para\s+)?esta noche)\b"),
    ("tomorrow", r"\b(?:(?:s[oó]lo\s+)?para\s+ma[ñn]ana|s[oó]lo\s+ma[ñn]ana)\b"),
    ("session", r"\b(?:(?:s[oó]lo\s+)?(?:en|para)\s+esta\s+(?:conversaci[oó]n|sesi[oó]n|ocasi[oó]n)"
     r"|s[oó]lo\s+esta\s+(?:conversaci[oó]n|sesi[oó]n|ocasi[oó]n))\b"),
    ("ongoing", r"\b(?:de forma habitual|preferencia habitual|desde (?:ahora|hoy|ma[ñn]ana)|de ahora en adelante"
     r"|a partir de (?:ahora|hoy))\b"),
    ("unknown", r"\b(?:s[oó]lo por ahora|por ahora|ahora mismo)\b"),
)
ABSOLUTE_START = re.compile(r"\bdesde el\s+(\d{4}-\d{2}-\d{2})\b", re.I)
RELATIVE_START = re.compile(r"\bdesde ma[ñn]ana\b", re.I)
PREFERENCE_TODAY = re.compile(r"\b(hoy)\s+prefiero\b|\bprefiero\b[^.!?;]+?\b(hoy)\s*(?=[.!?;]|$)", re.I)
MEAL_SUFFIX = re.compile(
    r"\s+para\s+(?:cenar|desayunar|beber|la cena|el desayuno)(?:\s+(?:hoy|ma[ñn]ana|esta noche))?$", re.I,
)
DURATION_WORDS = {"un": 1, "uno": 1, "dos": 2, "tres": 3}
REPORTED_QUOTE = re.compile(
    r"\b(?:dijo|dice|dec[ií]a|escribi[oó]|coment[oó]|cont[oó]|respondi[oó]|repiti[oó])\s*"
    r"(?:que\s*)?(?::\s*)?(?:«[^»]*»|“[^”]*”|\"[^\"]*\")",
    re.I,
)


def personal_wording(text):
    """Leave out directly quoted speech attributed to someone else."""
    return REPORTED_QUOTE.sub(" ", text)


def is_dialogue(observation):
    return any("role" in m for m in observation.get("user_messages", []))


def users(observation):
    return [m for m in observation.get("user_messages", [])
            if m.get("role", "user") == "user" and isinstance(m.get("text"), str)]


def attribute_in(text, default=None):
    # Prefer the meal clause containing the assertion over a later question.
    assertion = personal_wording(text).split("¿", 1)[0]
    matches = []
    for attribute, pattern in (("dinner_preference", r"\bcena(?:r|s)?\b"),
                               ("drink_preference", r"\b(?:bebida(?:s)?|beber)\b"),
                               ("breakfast_preference", r"\bdesayun(?:o|os|ar)\b")):
        matches.extend((m.start(), attribute) for m in re.finditer(pattern, assertion, re.I))
    return sorted(matches)[-1][1] if matches else default


def preference_turn(message):
    text = personal_wording(message["text"]).strip()
    if text.startswith("¿"):
        # A user's question is not a declaration of preference.
        return False
    return bool(re.search(r"\bprefiero\b|\bpreferencia habitual desde hoy es\b", text.split("¿", 1)[0], re.I))


def declarations(observation):
    default = observation.get("query", {}).get("attribute", "dinner_preference")
    return [(m, attribute_in(m["text"], default)) for m in users(observation) if preference_turn(m)]


def declared_attributes(observation):
    ordered = list(dict.fromkeys(attribute for _, attribute in reversed(declarations(observation))))
    query_attribute = observation.get("query", {}).get("attribute")
    if query_attribute in ordered:
        ordered.remove(query_attribute)
        ordered.insert(0, query_attribute)
    return ordered


def current_context(observation):
    context = observation.get("query", {}).get("context")
    for message in users(observation):
        text = personal_wording(message["text"])
        if re.search(r"\b(?:no|quizá|quizás|tal vez)\b", text, re.I):
            continue
        matches = list(re.finditer(r"\b(?:estoy en|ceno en|hoy en)\s+(?:la\s+)?(casa|oficina)\b", text, re.I))
        if matches:
            context = matches[-1].group(1).lower()
    return context


def focus(observation):
    """Choose the final declaration for the queried attribute, without gold."""
    messages = users(observation)
    entries = declarations(observation)
    query_attribute = observation.get("query", {}).get("attribute")
    relevant = [(m, a) for m, a in entries if a == query_attribute]
    if not entries:
        focused = deepcopy(observation)
        focused["user_messages"] = deepcopy(messages)
        return focused, None, []
    chosen, attribute = (relevant or entries)[-1]
    following, last_attribute = [], None
    chosen_seen = False
    for message in messages:
        if preference_turn(message):
            last_attribute = attribute_in(message["text"], query_attribute)
            if message["turn_id"] == chosen["turn_id"]:
                chosen_seen = True
            continue
        if not chosen_seen:
            continue
        explicit = message.get("about_attribute") or attribute_in(message["text"])
        if explicit is None and message.get("in_reply_to", "").startswith("a_tool_"):
            explicit = query_attribute
        if explicit == attribute or (explicit is None and last_attribute == attribute):
            following.append(message)
    focused = deepcopy(observation)
    focused["user_messages"] = deepcopy([chosen, *following])
    focused["query"]["attribute"] = attribute
    context = current_context(observation)
    if context is not None:
        focused["query"]["context"] = context
    return focused, chosen, following


def _scope_events(message):
    text = personal_wording(message["text"]).split("¿", 1)[0]
    events = sorted((match.start(), kind, match.group(1) if match.lastindex else None)
                    for kind, pattern in SCOPE_WORDING for match in re.finditer(pattern, text, re.I))
    if not events and (match := PREFERENCE_TODAY.search(text)):
        return [(match.start(1 if match.group(1) else 2), "today", None)]
    return events


def preference_value(text):
    """Remove trailing scope wording from the positive food expression."""
    prefix = "prefiero "
    events = _scope_events({"text": prefix + text})
    if events:
        text = text[:events[0][0] - len(prefix)]
    if match := ABSOLUTE_START.search(text):
        text = text[:match.start()]
    return MEAL_SUFFIX.sub("", text.rstrip(" ,;:")).rstrip(" ,;:")


def temporal_scope(messages, query, context=None):
    """Resolve scope from chronological statements and the dates they were made."""
    first = messages[0]
    when = first.get("date") or query.get("date")
    scope = {"kind": "unknown", "session_id": None, "context": context,
             "valid_from": when, "valid_until_exclusive": None}
    source = None
    start_source = None
    start_value = None
    event = None
    for message in messages:
        statement = personal_wording(message["text"]).split("¿", 1)[0]
        if match := ABSOLUTE_START.search(statement):
            start_value, start_source = match.group(1), message
        elif RELATIVE_START.search(statement):
            start_value, start_source = "tomorrow", message
        events = _scope_events(message)
        if events:
            _, kind, value = events[-1]
            event = kind, value
            source = message
    explicit_start = None
    if start_value:
        try:
            if start_value == "tomorrow":
                anchor = start_source.get("date") or query.get("date")
                explicit_start = (date.fromisoformat(anchor) + timedelta(days=1)).isoformat()
            else:
                explicit_start = date.fromisoformat(start_value).isoformat()
            scope["valid_from"] = explicit_start
        except (TypeError, ValueError, OverflowError):
            return None, "GENERIC", []
    if event is None:
        return scope, "TEMPORAL_SCOPE", [start_source] if start_source else []
    kind, value = event
    if kind == "ambiguous_end":
        return None, "VALID_UNTIL", []
    if kind == "session":
        scope["kind"] = "session"
        scope["session_id"] = query.get("session_id", "S2")
    elif kind in {"ongoing", "unknown"}:
        scope["kind"] = kind
        if kind == "ongoing" and not explicit_start:
            scope["valid_from"] = source.get("date") or query.get("date")
    else:
        relative_day = source.get("date") or query.get("date")
        try:
            start = date.fromisoformat(explicit_start or when)
            if kind == "absolute_end":
                end = date.fromisoformat(value) + timedelta(days=1)
            elif kind == "absolute_day":
                start = date.fromisoformat(value)
                end = start + timedelta(days=1)
                if explicit_start and explicit_start != start.isoformat():
                    return None, "VALID_UNTIL", []
            else:
                reference = date.fromisoformat(relative_day)
                if kind in {"today", "tomorrow"}:
                    start = reference + timedelta(days=int(kind == "tomorrow"))
                    end = start + timedelta(days=1)
                    if explicit_start and explicit_start != start.isoformat():
                        return None, "VALID_UNTIL", []
                elif kind == "inclusive_tomorrow":
                    end = reference + timedelta(days=2)
                else:
                    start = date.fromisoformat(explicit_start) if explicit_start else reference
                    days = DURATION_WORDS.get(value.lower())
                    end = start + timedelta(days=int(value) if days is None else days)
        except (TypeError, ValueError, OverflowError):
            return None, "VALID_UNTIL", []
        if not start < end:
            return None, "VALID_UNTIL", []
        scope.update(kind="interval", valid_from=start.isoformat(), valid_until_exclusive=end.isoformat())
    sources = [message for message in messages if message is source or message is start_source]
    return scope, "TEMPORAL_SCOPE" if kind == "unknown" else None, sources


def parse_dialogue(observation, parse_single):
    focused, chosen, following = focus(observation)
    context = current_context(observation)
    if chosen is None:
        return None, context, None
    fact, _, missing = parse_single(focused)
    if fact is None:
        return None, context, missing
    from .rule_controller import _definite_reply
    definite = [chosen, *(m for m in following if _definite_reply(m))]
    restriction = fact["scope"].get("context")
    restriction_source = None
    for message in definite:
        # Current location is not an enduring restriction of the preference.
        pattern = r"\b(?:solo en|se aplica en|aplica en)\s+(?:la\s+)?(casa|oficina)\b"
        matches = list(re.finditer(pattern, personal_wording(message["text"]), re.I))
        if matches:
            restriction = matches[-1].group(1).lower()
            restriction_source = message
    scope = fact["scope"]
    scope["context"] = restriction
    if restriction is not None and (scope["kind"] in {"ongoing", "interval"} or
                                    (scope["kind"] == "unknown" and not re.search(r"\bahora\b", chosen["text"], re.I)
                                     and not any(kind == "unknown" for message in definite
                                                 for _, kind, _ in _scope_events(message)))):
        scope["kind"] = "contextual"
    fact["status"] = "unresolved" if fact["scope"]["kind"] == "unknown" else "confirmed"
    source_ids = set(fact["source_turn_ids"])
    if restriction_source:
        source_ids.add(restriction_source["turn_id"])
    source_messages = [message for message in definite if message["turn_id"] in source_ids]
    fact["source_turn_ids"] = [m["turn_id"] for m in source_messages]
    fact["evidence_quotes"] = [{"turn_id": m["turn_id"], "text": m["text"]} for m in source_messages]
    return fact, context, "TEMPORAL_SCOPE" if fact["scope"]["kind"] == "unknown" else None
