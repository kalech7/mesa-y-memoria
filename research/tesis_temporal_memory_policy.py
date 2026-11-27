"""Common public action space for the two trained memory actors."""
from __future__ import annotations

from datetime import date, timedelta
import json
import re

from research.src.env.user_simulator import QUESTIONS
from research.src.policy.dialogue import attribute_in, current_context, personal_wording
from research.src.readers.symbolic import applicability
from web_demo.recommendations import meal_date


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MEALS = {"dinner_preference": "cena", "breakfast_preference": "desayuno", "drink_preference": "bebida"}
SCOPE = {"session": "sesión", "interval": "intervalo", "contextual": "contextual", "ongoing": "habitual",
         "unknown": "pendiente"}
NOOP = {"operations": [{"op": "NOOP"}]}
ASK_SLOT = {"TEMPORAL_SCOPE": "duración del cambio", "VALID_UNTIL": "fecha final",
            "CURRENT_CONTEXT": "lugar aplicable", "GENERIC": "datos del cambio"}
RECALL = re.compile(r"\b(?:qu[eé] (?:\w+ ){0,4}prefiero|qu[eé] recuerdas|recuerdas (?:qu[eé]|cu[aá]l)|"
                    r"te acuerdas|mi (?:gusto|preferencia) habitual|qu[eé] ten[ií]as guardado)\b", re.I)
# Extract the mentioned value and provenance. Scope, permit/noop and reply are model actions.
VALUE = re.compile(
    r"\b(?:preferencia habitual desde hoy es|me quedo con|prefiero|pref[eé]rir|me (?:suele )?apetece(?:r)?"
    r"|me gusta m[aá]s|elijo|inclin[aá]ndome por|(?:mi|la) (?:nueva )?preferencia (?:para \w+ )?"
    r"(?:habitual )?(?:es|ser[aá]|sea)|(?:mi )?elecci[oó]n de|cambia a|gusto estable de"
    r"|(?:reemplazar|cambiar) (?:el gusto|la preferencia) [^.!?¿;]+? por|cambi[eé] (?:mi\s+)?preferencia (?:a|por)"
    r"|guarda(?:r)?)\s+([^.!?¿;]+)", re.I,
)
VALUE_END = re.compile(
    r"\s*(?:,|\b(?:s[oó]lo|[úu]nicamente|nada m[aá]s)\b|\b(?:para|por)\s+(?:hoy|ma[ñn]ana|el \d{4}|esta (?:sesión|conversación))"
    r"|\b(?:para\s+(?:cenar|desayunar|beber|la cena|el desayuno)|por lo que queda|de hoy|de aqu[ií] en adelante"
    r"|vale|[uú]ltimamente|de forma habitual|habitualmente|desde|hasta|durante|en (?:la )?(?:casa|oficina))\b)", re.I)
PLACE = re.compile(r"\b(casa|oficina)\b", re.I)
ABSOLUTE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
DAYS = re.compile(r"\b(?:por|durante)\s+(\d+|un|uno|dos|tres|cuatro|cinco|seis|siete)\s+d[ií]as?\b", re.I)
WORD_DAYS = {word: i for i, word in enumerate(("un", "dos", "tres", "cuatro", "cinco", "seis", "siete"), 1)}
WORD_DAYS["uno"] = 1
ANSWERING = "¿Es solo para esta conversación, por hoy, por otro plazo o un cambio habitual?"


def mentions_preference(text):
    return bool(VALUE.search(text.split("¿", 1)[0]))


def scope(kind, start, end=None, context=None, session=None):
    return {"kind": kind, "valid_from": start, "valid_until_exclusive": end, "context": context,
            "session_id": session}


def observation_for(messages, memory, day, session, default, *, recall=False):
    latest = messages[-1]["text"]
    question = latest.rsplit("¿", 1)[-1]
    attr = attribute_in(personal_wording(question), attribute_in(personal_wording(latest), default))
    query = {"attribute": attr, "date": (meal_date(latest, day) or day) if recall else day,
             "session_id": session, "context": None}
    query["context"] = current_context({"user_messages": [m for m in messages if m.get("date") == day],
                                       "query": query})
    return {"memory": memory, "user_messages": messages, "query": query}


def _dates(messages, now):
    today = date.fromisoformat(now)
    tomorrow = today + timedelta(days=1)
    intervals = [(today, tomorrow), (tomorrow, tomorrow + timedelta(days=1))]
    territory = " ".join(turn["text"] for turn in messages)
    for match in DAYS.finditer(territory):
        value = match[1].lower()
        count = int(value) if value.isdecimal() else WORD_DAYS[value]
        if 0 < count < 366:
            intervals.append((today, today + timedelta(days=count)))
    if re.search(r"\bma[ñn]ana\b", territory, re.I):
        intervals.append((today, tomorrow + timedelta(days=1)))
    explicit = set()
    for value in ABSOLUTE.findall(territory):
        try:
            explicit.add(date.fromisoformat(value))
        except ValueError:
            continue
    explicit = sorted(explicit)[:3]
    for end in explicit:
        intervals.append((end, end + timedelta(days=1)))
        if end >= today:
            intervals.append((today, end + timedelta(days=1)))
        if end > today:
            intervals.append((today, end))
    if len(explicit) > 1:
        intervals.append((explicit[0], explicit[-1] + timedelta(days=1)))
    return list(dict.fromkeys(intervals)), explicit


def write_actions(observation):
    query = observation["query"]
    messages = observation["user_messages"]
    actions = [NOOP]
    # Keep the original words in every proposed citation, including a rival
    # interpretation of reported speech when the clause contains a value.
    declaration = None
    for index, message in reversed(list(enumerate(messages))):
        if message.get("role", "user") != "user":
            continue
        if match := VALUE.search(message["text"].split("¿", 1)[0]):
            value = VALUE_END.split(match[1], maxsplit=1)[0].strip(" \"“”«»")
            if value:
                declaration = (index, message, value)
                break
    if declaration is None:
        return actions
    index, message, value = declaration
    statement = next((part for part in re.split(r"[.!?;¿]", message["text"]) if VALUE.search(part)), message["text"])
    attr = attribute_in(personal_wording(statement), query["attribute"])
    tail = messages[index:]
    now = tail[-1].get("date") or query["date"]
    intervals, explicit = _dates(tail, now)
    locations = list(dict.fromkeys(place.casefold() for turn in tail for place in PLACE.findall(turn["text"])))
    actions.extend({"slot": slot} for slot in ("TEMPORAL_SCOPE", "VALID_UNTIL"))
    if locations:
        actions.append({"slot": "CURRENT_CONTEXT"})
    choices = [
        scope("session", now, session=query["session_id"]),
        scope("ongoing", now),
        *(scope("interval", start.isoformat(), stop.isoformat()) for start, stop in intervals),
    ]
    if re.search(r"\bma[ñn]ana\b", " ".join(turn["text"] for turn in tail), re.I):
        choices.append(scope("ongoing", (date.fromisoformat(now) + timedelta(days=1)).isoformat()))
    choices.extend(scope("ongoing", day.isoformat()) for day in explicit)
    for location in locations[:1]:
        choices.append(scope("contextual", now, context=location))
        choices.extend(scope("contextual", start.isoformat(), end.isoformat(), location) for start, end in intervals[:3])
    sources = [message]
    if tail[-1] is not message and tail[-1].get("role", "user") == "user":
        sources.append(tail[-1])
    for option in choices[:len(LETTERS) - len(actions)]:
        targets = [r["memory_id"] for r in observation["memory"]
                   if r["attribute"] == attr and r["scope"]["kind"] == option["kind"]
                   and r["scope"].get("context") == option["context"]
                   and r["scope"].get("valid_until_exclusive") is None]
        targets = targets if option["kind"] in ("ongoing", "contextual") and not option["valid_until_exclusive"] else []
        record = {"attribute": attr, "value": value, "scope": option, "status": "confirmed",
                  "source_turn_ids": [m["turn_id"] for m in sources],
                  "evidence_quotes": [{"turn_id": m["turn_id"], "text": m["text"]} for m in sources]}
        action = {"op": "SUPERSEDE" if targets else "ADD", "record": record}
        if targets:
            action["target_ids"] = targets
        actions.append({"operations": [action]})
    return actions


def response(mode="DEFER", value=None, branches=None, evidence=None):
    return {"mode": mode, "value": value, "branches": branches or [], "evidence_ids": evidence or []}


def recall_actions(observation):
    memory = observation["memory"]
    options = [response()]
    relevant = [r for r in memory if r["attribute"] == observation["query"]["attribute"]]
    selected = [*relevant, *(r for r in memory if r not in relevant)][-20:]
    possibilities = {}
    for record in selected:
        value = record["value"]
        options.append(response("ANSWER", value, evidence=[record["memory_id"]]))
        if record in relevant and applicability(record, observation["query"]) is not False:
            context = record["scope"].get("context")
            condition = context or "según el alcance pendiente"
            possibilities[value] = (condition, record["memory_id"])
    if len(possibilities) > 1:
        branches = [{"value": value, "condition": condition} for value, (condition, _) in possibilities.items()]
        options.append(response("CONDITIONALIZE", branches=branches,
                                evidence=[mid for _, mid in possibilities.values()]))
    return options


def _date_label(day, when):
    if not day:
        return "—"
    offset = (date.fromisoformat(day) - date.fromisoformat(when)).days
    return {0: "hoy", 1: "mañana", 2: "pasado mañana", -1: "ayer"}.get(offset, day)


def _scope_label(item, query):
    start, end = (_date_label(item[key], query["date"]) for key in ("valid_from", "valid_until_exclusive"))
    kind = SCOPE[item["kind"]]
    note = item.get("context") or item.get("session_id") or ""
    return f"{kind} {note} [{start}, {end})"


def prompt(phase, observation, options):
    query = observation["query"]
    text = "\n".join(f"{turn['turn_id']}: {turn['text']}" for turn in observation["user_messages"][-4:])
    records = "\n".join(f"{record['memory_id']}: {MEALS.get(record['attribute'], record['attribute'])} "
                        f"{record['value']} {_scope_label(record['scope'], query)}"
                        f"{' sustituye ' + ','.join(record['supersedes']) if record.get('supersedes') else ''}"
                        for record in observation["memory"][-10:]) or "vacía"
    query_label = f"{MEALS[query['attribute']]} | {query['date']} | {query['session_id']} | {query['context'] or 'lugar sin indicar'}"
    if phase == "write":
        instruction = ("Decide qué guardar de lo dicho por el usuario. Una ocasión de chat difiere de un día de calendario. "
                       "El final de un intervalo es exclusivo. Conserva cambios confirmados y su evidencia. "
                       "Pregunta si falta la duración o la fecha y evita registrar lo que dijo otra persona.")
    else:
        instruction = ("Recupera de SQLite la preferencia vigente para la consulta. Respeta fecha, sesión, comida, contexto "
                       "y sustituciones. El final del intervalo es exclusivo. Si no hay evidencia, aplaza. "
                       "Si el lugar o alcance pendiente cambia la respuesta, condiciona o aplaza.")
    labels = "\n".join(f"{letter}. {_action_label(phase, action, query)}"
                       for letter, action in zip(LETTERS, options))
    return (f"{instruction} Devuelve solo la letra.\nConsulta: {query_label}\n"
            f"Diálogo:\n{text}\nSQLite:\n{records}\n{labels}\nLetra:")


def _action_label(phase, action, query):
    if phase == "recall":
        mode = action["mode"]
        if mode == "ANSWER":
            return f"Responder {action['value']} ({','.join(action['evidence_ids'])})"
        if mode == "CONDITIONALIZE":
            return "Condicionar: " + " / ".join(f"{b['value']} si {b['condition']}" for b in action["branches"])
        return "Aplazar por falta de evidencia"
    if "slot" in action:
        return "Preguntar " + ASK_SLOT[action["slot"]]
    labels = []
    for op in action["operations"]:
        if op["op"] == "NOOP":
            labels.append("Conservar memoria")
            continue
        record = op["record"]
        food = MEALS.get(record["attribute"], record["attribute"])
        targets = ",".join(op.get("target_ids", []))
        citations = ",".join(q["turn_id"] for q in record["evidence_quotes"])
        labels.append(f"{'Sustituir ' + targets if targets else 'Guardar'} {food}: {record['value']} "
                      f"{_scope_label(record['scope'], query)} ({citations})")
    return " / ".join(labels)


def question(slot):
    return ANSWERING if slot == "TEMPORAL_SCOPE" else QUESTIONS[slot]


def option_tokens(tokenizer):
    tokens = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(ids) != 1 for ids in tokens) or len({tuple(ids) for ids in tokens}) != len(tokens):
        raise ValueError("Memory options must use distinct single tokens.")
    return [ids[0] for ids in tokens]


def action_json(action):
    return json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
