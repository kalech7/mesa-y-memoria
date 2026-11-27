"""Public chat wording adaptations, with every citation kept in the user's words."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import re

from research.src.policy.rule_controller import RuleController, compatible_worlds, expected_memory
from research.src.policy.dialogue import preference_turn
from research.src.readers.symbolic import read
from research.src.scoring.evidence import audit_operation
from research.tesis_profiles_parser_v5 import _original_id, normalized_observation, public_evidence


ATTRIBUTES = {"cenar": ("la cena", "dinner_preference"),
              "desayunar": ("el desayuno", "breakfast_preference"),
              "beber": ("la bebida", "drink_preference")}
NAMED_OTHER = re.compile(r"\b[Pp]ara\s+(?!m[ií]\b)([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)\s*,?\s*(?:yo\s+)?(?:prefiero|quiero|anota|guarda)\b")
OTHER_RECIPIENT = re.compile(r"\bpara\s+(?:mi|tu|su)\s+(?:herman[oa]|pareja|madre|padre|sobrin[oa]|"
                             r"amig[oa]|hij[oa]|abuel[oa]|prim[oa])s?\b", re.I)
INSTRUCTION_PREFIX = re.compile(
    r"^\s*(?:anota|apunta|registra|guarda|corrige|actualiza)\s+"
    r"(?:(?:una|esta|la|mi)\s+)?(?:nueva\s+)?"
    r"(?:rectificaci[oó]n|correcci[oó]n|actualizaci[oó]n|modificaci[oó]n|cambio)"
    r"(?:\s+(?:para\s+m[ií]|en\s+mi\s+ficha))?\s*:\s*"
    r"(?=mi\s+(?:cena|bebida|desayuno)\s+habitual(?:\s+ahora)?\s+es\s+)",
    re.I,
)

# Interpret a scope-only reply only when it answers a previous question.
# The writer still cites the untouched original turn, including the user's food.
SCOPE_REPLIES = (
    (r"(?:s[ií][, ]+)?(?:(?:gu[aá]rdal[ao]|recu[eé]rdal[ao]|an[oó]tal[ao])\s+)?"
     r"(?:como\s+)?(?:(?:mi\s+)?preferencia\s+)?habitual", "De forma habitual."),
    (r"(?:s[oó]lo\s+)?(?:por|para|en)\s+esta\s+visita", "Solo para esta sesión."),
)


def public_representation(raw):
    observation = deepcopy(raw)
    adaptations = []
    for turn in observation["user_messages"]:
        if turn["role"] != "user":
            continue
        text = turn["text"]
        if turn.get("in_reply_to"):
            for pattern, representation in SCOPE_REPLIES:
                if re.fullmatch(pattern, text.strip(" .!"), re.I):
                    text = representation
                    adaptations.append({"turn_id": turn["turn_id"], "kind": "explicit_scope_reply"})
                    break
        pieces = re.split(r"(?<=[.!?])\s+", text)
        rewritten = []
        for clause in pieces:
            if NAMED_OTHER.search(clause) or OTHER_RECIPIENT.search(clause):
                adaptations.append({"turn_id": turn["turn_id"], "kind": "named_third_party_not_a_personal_update"})
                continue
            instruction = INSTRUCTION_PREFIX.match(clause)
            if instruction:
                clause = clause[instruction.end():]
                clause = re.sub(r"^(mi\s+(?:cena|bebida|desayuno)\s+habitual)\s+ahora\s+es\b",
                                r"\1 es", clause, count=1, flags=re.I)
                adaptations.append({"turn_id": turn["turn_id"],
                                    "kind": "instruction_prefix_before_explicit_preference",
                                    "prefix": instruction.group().strip()})
            routine = re.search(r"\bsuelo\s+(cenar|desayunar|beber)\s+(.+?)[.!]?$", clause, re.I)
            if routine:
                title = ATTRIBUTES[routine.group(1).lower()][0]
                clause = f"Para {title}, prefiero {routine.group(2).rstrip('.!')} de forma habitual."
                adaptations.append({"turn_id": turn["turn_id"], "kind": "routine_wording"})
            today = re.search(r"\b(?:(?:s[oó]lo\s+)?(?:por|para)\s+hoy|hoy\s+prefiero)\b", clause, re.I)
            if today and re.search(r"\bprefiero\b", clause, re.I):
                when = turn.get("date") or raw["query"]["date"]
                clause = re.sub(r"\b(?:s[oó]lo\s+)?(?:por|para)\s+hoy\b", "", clause, flags=re.I)
                clause = re.sub(r"\bhoy\s+prefiero\b", "prefiero", clause, flags=re.I)
                clause = clause.strip(" .")
                clause += f". Desde el {when}. Hasta el {when} inclusive."
                adaptations.append({"turn_id": turn["turn_id"], "kind": "explicit_today_calendar_scope"})
            rewritten.append(clause)
        turn["text"] = " ".join(rewritten) or "Sin una nueva declaración personal."
    return observation, adaptations


def decision(raw):
    represented, adaptations = public_representation(raw)
    parsed = normalized_observation(represented)
    controller = RuleController()
    response = controller.respond(parsed)
    originals = {turn["turn_id"]: turn["text"] for turn in raw["user_messages"]}
    proposals = list(controller.write(parsed)["operations"])
    validated_prefix = None
    if len(proposals) == 1 and proposals[0].get("record", {}).get("scope", {}).get("kind") in ("session", "interval"):
        selected = proposals[0]["record"]
        declarations = [index for index, turn in enumerate(parsed["user_messages"])
                        if turn["role"] == "user" and preference_turn(turn)]
        for boundary in reversed(declarations[1:]):
            prefix = {**parsed, "user_messages": parsed["user_messages"][:boundary]}
            earlier = controller.write(prefix)["operations"]
            compatible = [item for item in earlier if item.get("record", {}).get("attribute") == selected["attribute"]
                          and item["record"]["scope"]["kind"] in ("ongoing", "contextual")
                          and item["record"]["status"] == "confirmed" and audit_operation(item, prefix)]
            if compatible:
                proposals = [compatible[-1], *proposals]
                validated_prefix = prefix
                adaptations.append({"turn_id": _original_id(parsed["user_messages"][boundary]["turn_id"], originals),
                                    "kind": "retain_habit_before_explicit_exception"})
                break
    candidates = []
    for item in proposals:
        if item.get("record", {}).get("status") != "confirmed":
            continue
        operation = deepcopy(item)
        justified = audit_operation(operation, parsed)
        if validated_prefix is not None and item is proposals[0] and not justified:
            justified = audit_operation(operation, validated_prefix)
        references = list(dict.fromkeys(
            _original_id(identifier, originals) for identifier in operation["record"]["source_turn_ids"]))
        operation["record"]["source_turn_ids"] = references
        operation["record"]["evidence_quotes"] = [
            {"turn_id": identifier, "text": originals[identifier]} for identifier in references]
        candidates.append((operation, justified))
    return response, tuple(candidates), parsed, adaptations


def evidence(raw):
    represented, _ = public_representation(raw)
    return public_evidence(represented)


def query_date(text, today):
    explicit = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if explicit:
        try:
            return date.fromisoformat(explicit.group(1)).isoformat()
        except ValueError as error:
            raise ValueError("La fecha de la consulta no existe en el calendario.") from error
    for pattern, offset in ((r"\bpasado ma[ñn]ana\b", 2), (r"\banteayer\b", -2),
                            (r"\bma[ñn]ana\b", 1), (r"\bayer\b", -1)):
        if re.search(pattern, text, re.I):
            return (date.fromisoformat(today) + timedelta(days=offset)).isoformat()
    return today


def query_attribute(text, default):
    for pattern, attribute in (
        (r"\b(?:desayun(?:o|ar|aba)|desayunos)\b", "breakfast_preference"),
        (r"\b(?:bebidas?|beber|bebo|tomar)\b", "drink_preference"),
        (r"\b(?:cena(?:s|r)?|ceno|cenaba)\b", "dinner_preference"),
    ):
        if re.search(pattern, text, re.I):
            return attribute
    return default


def query_context(text, fallback):
    found = re.search(r"\b(?:en|para) (?:la )?(casa|oficina)\b", text, re.I)
    return found.group(1).lower() if found else fallback


def respond_to_query(parsed, query):
    if parsed["query"] == query:
        return RuleController().respond(parsed)
    worlds = compatible_worlds(parsed, query=query)
    values = {world["value"] for world in worlds}
    references = sorted({identifier for world in worlds for identifier in world["evidence_ids"]})
    if len(values) == 1 and None not in values:
        return {"mode": "ANSWER", "value": next(iter(values)), "branches": [], "evidence_ids": references}
    valid = values - {None}
    if valid:
        interpreted = read(expected_memory(parsed), query)
        branches = interpreted.get("branches", [])
        if {branch["value"] for branch in branches} != valid:
            branches = [{"condition": f"si sigue siendo aplicable {value}", "value": value} for value in sorted(valid)]
        return {"mode": "CONDITIONALIZE", "value": None, "branches": branches, "evidence_ids": references}
    return {"mode": "DEFER", "value": None, "branches": [], "evidence_ids": references}
