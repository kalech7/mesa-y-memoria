"""Reward learned memory actions after their effects in the existing SQLite environment."""
from __future__ import annotations

from collections import Counter
from contextlib import closing
from dataclasses import asdict, dataclass
from statistics import mean

from research.src.memory.executor import Executor
from research.src.memory.store import Store, canonical
from research.src.readers.symbolic import read
from research.src.scoring.evidence import response_valid


@dataclass(frozen=True)
class Weights:
    write: float = 1.0
    future: float = 1.5
    joint: float = 0.5
    unsupported_write: float = 2.0
    missing_write: float = 0.6
    harmful_future: float = 0.9
    malformed: float = 2.5
    recall: float = 1.8
    grounded_recall: float = 0.7
    wrong_recall: float = 1.4
    unsupported_recall: float = 1.1
    ask: float = 2.2
    redundant_ask: float = 1.35
    wrong_slot: float = 1.5


WEIGHTS = Weights()
REWARD = asdict(WEIGHTS)


def _change(record):
    return canonical([record["attribute"], record["value"], record["scope"], record["status"],
                      sorted(record.get("supersedes", []))])


def _projection(result):
    branches = result.get("branches", ())
    return (result["mode"], result.get("value"),
            frozenset(branch["value"] if isinstance(branch, dict) else branch for branch in branches))


def _cited_sources(record):
    return frozenset(record.get("source_turn_ids", []))


def _supported_citations(record, messages, allowed):
    refs = _cited_sources(record)
    quotes = record.get("evidence_quotes", [])
    return (bool(refs) and refs == {quote["turn_id"] for quote in quotes}
            and all(quote["text"] == messages.get(quote["turn_id"]) for quote in quotes)
            and (not allowed or refs in allowed))


def write_reward(observation, operations, expected_writes, probes, *, weights=WEIGHTS):
    """Execute candidate writes and compare native state plus dated future queries."""
    before = observation["memory"]
    visible = {message["turn_id"]: message["text"] for message in observation["user_messages"]
               if message.get("role", "user") == "user"}
    expected = Counter(_change(record) for record in expected_writes)
    allowed = {}
    for record in expected_writes:
        if refs := _cited_sources(record):
            allowed.setdefault(_change(record), set()).add(refs)
    with closing(Store(":memory:")) as store:
        store.initialize(before)
        executed = Executor(store).execute(operations, recorded_at=observation["query"]["date"])
        after = store.list_records()
    previous_ids = {record["memory_id"] for record in before}
    additions = [record for record in after if record["memory_id"] not in previous_ids]
    supported = Counter()
    unsupported_writes = 0
    for record in additions:
        effect = _change(record)
        if supported[effect] < expected[effect] and _supported_citations(record, visible, allowed.get(effect)):
            supported[effect] += 1
        else:
            unsupported_writes += 1
    missing_writes = (expected - supported).total()
    exact = bool(executed["valid"] and not missing_writes and not unsupported_writes)
    hits, harmful = [], []
    for probe in probes:
        query, wanted = probe["query"], _projection(probe["expected"])
        old, actual = (read(memory, query) for memory in (before, after))
        hits.append(_projection(actual) == wanted)
        harmful.append(_projection(old) == wanted and not hits[-1])
    future = mean(hits) if hits else float(exact)
    damage = mean(harmful) if harmful else 0.0
    joint = exact and future == 1.0
    if not executed["valid"]:
        total = -weights.malformed
    else:
        total = (weights.write * exact + weights.future * future + weights.joint * joint
                 - weights.unsupported_write * unsupported_writes - weights.missing_write * missing_writes
                 - weights.harmful_future * damage)
    return {"total": total, "valid": executed["valid"], "exact_write": exact, "future": future,
            "joint": joint, "unsupported_writes": unsupported_writes, "missing_writes": missing_writes,
            "harmful_future": damage, "matched_probes": sum(hits), "probes": len(hits)}


def _branches(response):
    return sorted(canonical(branch) for branch in response["branches"])


def recall_reward(memory, query, response, expected, *, weights=WEIGHTS):
    """Grade the actor's answer and citations against records available to it."""
    if not response_valid(response):
        return {"total": -weights.malformed, "valid": False, "correct_recall": False, "grounded": False}
    actual = _projection(response)
    wanted = _projection(expected)
    projection = read(memory, query)
    correct = actual == wanted
    if correct and response["mode"] == "CONDITIONALIZE":
        correct = _branches(response) == _branches(expected)
    cited = frozenset(response["evidence_ids"])
    valid_evidence = frozenset(projection["evidence_ids"])
    cited_rows = {record["memory_id"]: record for record in memory if record["memory_id"] in cited}
    grounded = cited <= valid_evidence and len(cited_rows) == len(cited)
    if response["mode"] == "ANSWER":
        grounded = grounded and bool(cited_rows) and all(record["value"] == response["value"]
                                                       for record in cited_rows.values())
    if response["mode"] == "CONDITIONALIZE":
        grounded = grounded and {record["value"] for record in cited_rows.values()} == actual[2]
    if len(response["evidence_ids"]) != len(cited):
        grounded = False
    total = ((weights.recall + weights.grounded_recall) * (correct and grounded)
             - weights.wrong_recall * (not correct) - weights.unsupported_recall * (not grounded))
    return {"total": total, "valid": True, "correct_recall": correct, "grounded": grounded}


def action_reward(row, action, *, weights=WEIGHTS):
    """Reward one candidate in a group sharing the same visible observation."""
    expected = row["actions"][row["correct_index"]]
    if "slot" in action or "slot" in expected:
        if "slot" in action and action.get("slot") == expected.get("slot"):
            return {"total": weights.ask, "correct_ask": True}
        penalty = weights.wrong_slot if "slot" in expected else weights.redundant_ask
        return {"total": -penalty, "correct_ask": False}
    observation = row["observation"]
    if row["phase"] == "write":
        expected_writes = [{**op["record"], "supersedes": op.get("target_ids", [])}
                           for op in expected["operations"] if op["op"] != "NOOP"]
        return write_reward(observation, action.get("operations"), expected_writes, row.get("probes", ()),
                            weights=weights)
    if row["phase"] != "recall":
        raise ValueError(f"Unknown temporal-memory phase: {row['phase']}")
    return recall_reward(observation["memory"], observation["query"], action, expected, weights=weights)


def score_action(row, action):
    return action_reward(row, action)["total"]
