"""Filter WRITE batches with public evidence and stage them using the native executor."""
from __future__ import annotations

from contextlib import closing
from copy import deepcopy
from itertools import combinations

from research.src.artifacts import canonical
from research.src.memory.executor import Executor
from research.src.memory.store import Store
from research.src.policy.llm_controller import InvalidOutput, OBSERVATION_KEYS
from research.src.policy.rule_controller import authorized_operations, expected_memory
from research.src.readers.symbolic import read
from research.src.scoring.evidence import audit_operation, probe_queries


NOOP = {"op": "NOOP"}
MODES = ("veto", "repair")


def _stage(observation, operations):
    """Use a disposable SQLite database with the exact production WRITE schema."""
    with closing(Store(":memory:")) as store:
        store.initialize(observation["memory"])
        result = Executor(store).execute(operations or [NOOP], recorded_at=observation["query"]["date"])
        return store.list_records() if result["valid"] else None, result["error"]


def write_effect(operation):
    """Stable write identity independent of executor IDs and citation ordering."""
    record = operation["record"]
    return canonical([record[key] for key in ("attribute", "value", "scope", "status")]
                     + [sorted(set(operation.get("target_ids", [])))])


def operation_issue(observation, operation):
    """Classify a single proposed operation before considering its batch."""
    _, error = _stage(observation, [operation])
    if error is not None:
        return "invalid_structure:" + error
    if not audit_operation(operation, observation):
        return "unsupported_evidence"
    return None


def _value(memory, query):
    answer = read(memory, query)
    return answer["value"] if answer["mode"] == "ANSWER" else None


def _probes(observation, reference, operations):
    records = [operation["record"] for operation in operations]
    attrs = sorted({record["attribute"] for record in records})
    probes = []
    for query in probe_queries(observation, *reference, *records):
        for attribute in attrs:
            candidate = {**query, "attribute": attribute}
            before, expected = (_value(memory, candidate) for memory in (observation["memory"], reference))
            probes.append((candidate, before, expected))
    return probes


def _compatible_batch(observation, operations, probes):
    memory, error = _stage(observation, operations)
    if error is not None:
        return None
    values = []
    for query, before, expected in probes:
        after = _value(memory, query)
        if before is not None and before == expected and after != before:
            return None
        if after is not None and after not in (before, expected):
            return None
        values.append(after)
    return values


def _improves(previous, candidate, probes):
    """Add a confirmed operation only if it gains an answer without losing support."""
    gained = False
    for before, after, (_, _, expected) in zip(previous, candidate, probes):
        if expected is None:
            continue
        if before == expected and after != expected:
            return False
        if before != expected and after == expected:
            gained = True
    return gained


def plan_write(observation, proposed, *, mode="repair"):
    """Return a validated WRITE action and diagnostics for the live writer or replay.

    The planner receives only the native public WRITE observation and proposed action.
    It stages alternatives in disposable SQLite, retains a maximum safe subset in
    original order and optionally fills omissions with confirmed visible evidence.
    """
    if mode not in MODES:
        raise ValueError("Unknown WRITE guard mode")
    if (not isinstance(observation, dict) or set(observation) - OBSERVATION_KEYS
            or not {"memory", "user_messages", "query"} <= observation.keys()):
        raise ValueError("Expected a public WRITE observation")
    submitted = proposed.get("operations") if isinstance(proposed, dict) and set(proposed) == {"operations"} else None
    input_error = not isinstance(submitted, list) or not 1 <= len(submitted) <= 2
    proposal_kind = "malformed" if input_error else "noop" if submitted == [NOOP] else "writes"
    rejected, eligible, seen = [], [], set()
    reasons = ["malformed_write_action"] if input_error else []
    if not input_error:
        for index, operation in enumerate(submitted):
            error = operation_issue(observation, operation)
            if error is None:
                if operation == NOOP:
                    if len(submitted) != 1:
                        rejected.append({"index": index, "reason": "noop_in_batch"})
                    continue
                signature = write_effect(operation)
                if signature in seen:
                    error = "duplicate_operation"
                else:
                    seen.add(signature)
            if error is None:
                eligible.append((index, operation))
            else:
                rejected.append({"index": index, "reason": error})

    alternatives = []
    if mode == "repair":
        for operation in authorized_operations(observation):
            record = operation.get("record", {})
            if record.get("status") == "confirmed" and operation_issue(observation, operation) is None:
                alternatives.append(operation)

    reference = expected_memory(observation)
    probes = _probes(observation, reference, [operation for _, operation in eligible] + alternatives)
    kept, values = (), None
    for count in range(len(eligible), -1, -1):
        for subset in combinations(eligible, count):
            values = _compatible_batch(observation, [operation for _, operation in subset], probes)
            if values is not None:
                kept = subset
                break
        if values is not None:
            break
    if values is None:
        raise RuntimeError("Could not stage unchanged WRITE observation")
    kept_indices = [index for index, _ in kept]
    rejected.extend({"index": index, "reason": "batch_changes_unsupported_answer"}
                    for index, _ in eligible if index not in kept_indices)
    operations = [operation for _, operation in kept]
    recovered = []
    for operation in alternatives:
        if len(operations) == 2 or any(write_effect(item) == write_effect(operation) for item in operations):
            continue
        staged = _compatible_batch(observation, [*operations, operation], probes)
        if staged is not None and _improves(values, staged, probes):
            recovered.append(operation)
            operations.append(operation)
            values = staged
    if recovered:
        status = "repaired"
        reasons.append("recovered_confirmed_update")
    elif rejected or input_error:
        status = "partially_kept" if kept else "vetoed"
    else:
        status = "kept" if kept else "noop"
    reasons.extend(item["reason"] for item in rejected)
    return {"mode": mode, "status": status, "proposal_kind": proposal_kind, "operations": deepcopy(operations or [NOOP]),
            "accepted_indices": kept_indices, "recovered_operations": deepcopy(recovered),
            "rejected": sorted(rejected, key=lambda item: item["index"]), "reasons": reasons,
            "public_probes": len(probes), "input_error": input_error}


class GuardedWriter:
    """Use a WRITE guard with an existing controller and retain the proposed output."""

    def __init__(self, controller, *, mode="repair"):
        if mode not in MODES:
            raise ValueError("Unknown WRITE guard mode")
        self.controller = controller
        self.mode = mode
        self.last_write = None

    def __getattr__(self, name):
        return getattr(self.controller, name)

    def select(self, observation):
        return self.controller.select(observation)

    def respond(self, observation):
        return self.controller.respond(observation)

    def write(self, observation):
        proposed, error = None, None
        try:
            proposed = self.controller.write(deepcopy(observation))
        except InvalidOutput as exc:
            error = str(exc)
        decision = plan_write(observation, proposed, mode=self.mode)
        self.last_write = {**decision, "proposed": deepcopy(proposed), "model_error": error}
        return {"operations": decision["operations"]}
