"""Fixed reader. No dialogue, simulator, evaluator or hidden-world access."""
from __future__ import annotations


def _context_matches(expected, actual):
    if isinstance(expected, dict) and isinstance(actual, dict):
        return all(actual.get(key) == value for key, value in expected.items())
    return expected == actual


def applicability(record, query):
    """True: applicable; None: applicability depends on missing context/scope."""
    if record["attribute"] != query["attribute"]:
        return False
    scope = record["scope"]
    at = query["date"]
    if scope.get("valid_from") and at < scope["valid_from"]:
        return False
    if scope.get("valid_until_exclusive") and at >= scope["valid_until_exclusive"]:
        return False
    if scope["kind"] == "session" and scope.get("session_id") != query.get("session_id"):
        return False
    # Context and temporal/session limits are simultaneous constraints. A
    # context-bearing interval or unresolved record must not become global.
    if scope.get("context") is not None:
        if query.get("context") is None:
            return None
        if not _context_matches(scope["context"], query.get("context")):
            return False
    if scope["kind"] == "unknown":
        return None
    return True


def scope_specificity(record):
    scope = record["scope"]
    if scope["kind"] == "session":
        return 4
    if scope.get("valid_until_exclusive") is not None:
        return 3 if scope.get("context") is not None else 2
    return 1 if scope.get("context") is not None else 0


def read(memory, query):
    applicable = [(r, applicability(r, query)) for r in memory]
    applicable = [(r, a) for r, a in applicable if a is not False]
    # Edges close the old record only inside the new record's declared scope.
    suppressed = {mid for r, a in applicable if a is True for mid in r.get("supersedes", [])}
    applicable = [(r, a) for r, a in applicable if r["memory_id"] not in suppressed]
    unknown = [(r, a) for r, a in applicable if a is None or r.get("status") == "unresolved"]
    definite = [r for r, a in applicable if a is True and r.get("status") != "unresolved"]
    if definite:
        maximum = max(scope_specificity(r) for r in definite)
        definite = [r for r in definite if scope_specificity(r) == maximum]
    values = {r["value"] for r in definite}
    uncertainty_changes_value = any(r["value"] not in values for r, _ in unknown)
    if len(values) == 1 and not uncertainty_changes_value:
        return {"mode": "ANSWER", "value": next(iter(values)), "branches": [],
                "evidence_ids": [r["memory_id"] for r in definite]}
    alternatives = sorted(values | {r["value"] for r, _ in unknown})
    branches = []
    for value in alternatives:
        sources = [r for r, _ in applicable if r["value"] == value]
        condition = sources[0]["scope"].get("context") if len(sources) == 1 else None
        if condition is None and any(r.get("scope", {}).get("kind") == "unknown" for r, _ in unknown):
            pending_value = any(r["value"] == value for r, _ in unknown)
            condition = ("si el cambio pendiente sigue siendo aplicable" if pending_value
                         else "si el cambio pendiente ya no es aplicable")
        branches.append({"condition": condition or "según el alcance pendiente", "value": value})
    return {"mode": "CONDITIONALIZE" if len(alternatives) > 1 else "DEFER", "value": None,
            "branches": branches if len(alternatives) > 1 else [],
            "evidence_ids": [r["memory_id"] for r, _ in applicable]}


class SymbolicReader:
    reader_id = "fixed_symbolic"

    def read(self, memory, query):
        return read(memory, query)

    __call__ = read
