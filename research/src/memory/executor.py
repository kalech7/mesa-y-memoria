"""Structural validation only: false but well-formed writes really execute."""
from __future__ import annotations

from copy import deepcopy
from datetime import date

from .store import canonical


KINDS = {"session", "interval", "contextual", "ongoing", "unknown"}
OPERATIONS = {"NOOP", "ADD", "SUPERSEDE", "MARK_UNRESOLVED"}


def _string(value, field):
    if not isinstance(value, str) or not value:
        raise ValueError(f"INVALID_{field}")


def _date(value, field):
    if value is not None:
        _string(value, field)
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError(f"INVALID_{field}")


def validate_record(record, *, initial=False):
    if not isinstance(record, dict):
        raise ValueError("INVALID_RECORD")
    required = {"attribute", "value", "scope", "status", "source_turn_ids", "evidence_quotes"}
    optional = {"supersedes", "recorded_at"}
    if initial:
        required.add("memory_id")
    if not required <= record.keys() or set(record) - required - optional:
        raise ValueError("INVALID_RECORD_FIELDS")
    if initial:
        _string(record["memory_id"], "MEMORY_ID")
    _string(record["attribute"], "ATTRIBUTE")
    _string(record["value"], "VALUE")
    scope = record["scope"]
    scope_fields = {"kind", "session_id", "context", "valid_from", "valid_until_exclusive"}
    if not isinstance(scope, dict) or set(scope) != scope_fields or scope["kind"] not in KINDS:
        raise ValueError("INVALID_SCOPE")
    if scope["session_id"] is not None:
        _string(scope["session_id"], "SESSION_ID")
    if scope["context"] is not None and not isinstance(scope["context"], (str, dict)):
        raise ValueError("INVALID_CONTEXT")
    _date(scope["valid_from"], "VALID_FROM")
    _date(scope["valid_until_exclusive"], "VALID_UNTIL")
    if scope["valid_from"] and scope["valid_until_exclusive"] and scope["valid_from"] >= scope["valid_until_exclusive"]:
        raise ValueError("EMPTY_INTERVAL")
    if scope["kind"] == "session" and not scope["session_id"]:
        raise ValueError("SESSION_ID_REQUIRED")
    if scope["kind"] == "interval" and not (scope["valid_from"] and scope["valid_until_exclusive"]):
        raise ValueError("INTERVAL_DATES_REQUIRED")
    if scope["kind"] == "contextual" and scope["context"] is None:
        raise ValueError("CONTEXT_REQUIRED")
    if record["status"] not in {"confirmed", "unresolved"}:
        raise ValueError("INVALID_STATUS")
    sources = record["source_turn_ids"]
    if not isinstance(sources, list) or not sources or any(not isinstance(v, str) or not v for v in sources):
        raise ValueError("INVALID_SOURCES")
    quotes = record["evidence_quotes"]
    if not isinstance(quotes, list) or not quotes:
        raise ValueError("INVALID_QUOTES")
    for quote in quotes:
        if not isinstance(quote, dict) or set(quote) != {"turn_id", "text"}:
            raise ValueError("INVALID_QUOTE")
        _string(quote["text"], "QUOTE_TEXT")
        if quote["turn_id"] not in sources:
            raise ValueError("QUOTE_SOURCE_MISMATCH")
    if "recorded_at" in record:
        _string(record["recorded_at"], "RECORDED_AT")
    if "supersedes" in record and (not isinstance(record["supersedes"], list) or any(not isinstance(v, str) for v in record["supersedes"])):
        raise ValueError("INVALID_SUPERSEDES")
    canonical(record)


class Executor:
    def __init__(self, store):
        self.store = store

    def execute(self, operations, recorded_at="2026-01-12"):
        try:
            if not isinstance(operations, list) or not 1 <= len(operations) <= 2:
                raise ValueError("WRITE_OPERATION_LIMIT")
            existing = {r["memory_id"] for r in self.store.list_records()}
            for operation in operations:
                if not isinstance(operation, dict) or operation.get("op") not in OPERATIONS:
                    raise ValueError("INVALID_OPERATION")
                op = operation["op"]
                if op == "NOOP":
                    if len(operations) != 1 or set(operation) != {"op"}:
                        raise ValueError("NOOP_MUST_BE_ONLY_OPERATION")
                    continue
                if set(operation) - {"op", "record", "target_ids"} or "record" not in operation:
                    raise ValueError("INVALID_OPERATION_FIELDS")
                validate_record(operation["record"])
                targets = operation.get("target_ids", [])
                if not isinstance(targets, list) or any(not isinstance(v, str) for v in targets) or not set(targets) <= existing:
                    raise ValueError("UNKNOWN_TARGET")
                if op == "SUPERSEDE" and not targets:
                    raise ValueError("SUPERSEDE_TARGET_REQUIRED")
                if op != "SUPERSEDE" and targets:
                    raise ValueError("TARGETS_REQUIRE_SUPERSEDE")
                if operation["record"].get("supersedes", []) != targets:
                    # The actor may omit supersedes; explicit contradictory edges are invalid.
                    if "supersedes" in operation["record"]:
                        raise ValueError("SUPERSEDES_TARGET_MISMATCH")
                if op == "MARK_UNRESOLVED" and (operation["record"]["scope"]["kind"] != "unknown" or operation["record"]["status"] != "unresolved"):
                    raise ValueError("UNRESOLVED_SCOPE_REQUIRED")
            executed = []
            with self.store.conn:
                for operation in operations:
                    applied = deepcopy(operation)
                    if operation["op"] != "NOOP":
                        record = deepcopy(operation["record"])
                        record["memory_id"] = self.store.next_id()
                        record["recorded_at"] = recorded_at
                        record["supersedes"] = list(operation.get("target_ids", []))
                        self.store.conn.execute("INSERT INTO cm_records(memory_id,payload) VALUES (?,?)",
                                                (record["memory_id"], canonical(record)))
                        applied["record"] = record
                    executed.append(applied)
                self.store.conn.execute("INSERT INTO cm_executions(payload) VALUES (?)", (canonical(executed),))
            return {"valid": True, "executed_operations": executed, "error": None}
        except (ValueError, TypeError, KeyError) as exc:
            return {"valid": False, "executed_operations": [], "error": str(exc)}


MemoryExecutor = Executor
