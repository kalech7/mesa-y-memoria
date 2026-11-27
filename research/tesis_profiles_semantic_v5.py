"""Compare no-question parser replays with separately authored semantic labels."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import re

from research.src.artifacts import canonical
from research.src.readers.symbolic import read
from research.tesis_extended_heldout import _normalize_response, _scope_record
from research.tesis_profiles_env import execute as execute_v4
from research.tesis_profiles_parser_v5 import execute as execute_v5


MATCHES = ("response_exact", "strict_supported", "new_write_exact", "retrieval_exact")
REPORTED_SPEECH = re.compile(
    r"[«»\"“”]|\b(?:herman[oa]|sobrin[oa]|prim[oa]|vecin[oa]|para (?:él|ella|ellos|ellas))\b",
    re.IGNORECASE,
)
SCOPE_FIELDS = ("scope", "valid_from", "valid_through_inclusive", "context", "session_id")


def _record_differences(actual, expected):
    """Compare full multisets while preserving duplicate and scope discrepancies."""
    actual_counts = Counter(canonical(record) for record in actual)
    expected_counts = Counter(canonical(record) for record in expected)
    extra = [json.loads(record) for record, count in (actual_counts - expected_counts).items()
             for _ in range(count)]
    missing = [json.loads(record) for record, count in (expected_counts - actual_counts).items()
               for _ in range(count)]
    scope = []
    for record in actual:
        alternatives = [reference for reference in expected
                        if (record["attribute"], record["value"]) ==
                        (reference["attribute"], reference["value"])]
        if alternatives and record not in alternatives:
            scope.append({
                "actual": record,
                "expected_alternatives": alternatives,
                "differing_fields": [field for field in SCOPE_FIELDS
                                     if all(record[field] != reference[field] for reference in alternatives)],
            })
    return {"extra": extra, "missing": missing, "scope": scope}


def _citation_audit(measured):
    """Check provenance references and exact quotation text, without a semantic judge."""
    visible = {turn["turn_id"]: turn["text"] for turn in measured["opening_messages"]
               if turn.get("role") == "user"}
    ids = set(visible)
    for record in measured["memory_before"]:
        ids.add(record["memory_id"])
        for quote in record.get("evidence_quotes", ()):
            visible.setdefault(quote["turn_id"], quote["text"])
    response = measured["outcome"]
    cited = response.get("evidence_ids", ())
    unknown = sorted(set(cited) - ids)
    invalid_writes = []
    for operation in measured["executed_operations"]:
        record = operation.get("record")
        if record is None:
            continue
        references = record.get("source_turn_ids", ())
        quotes = record.get("evidence_quotes", ())
        if (not references or set(references) != {quote["turn_id"] for quote in quotes}
                or any(visible.get(quote["turn_id"]) != quote["text"] for quote in quotes)):
            invalid_writes.append(record["memory_id"])
    return {
        "unknown_response_ids": unknown,
        "answer_without_citation": response["mode"] == "ANSWER" and not cited,
        "invalid_write_quote_ids": invalid_writes,
        "scope": "references and verbatim quotations only",
    }


def _compare_visit(measured, expected):
    response = measured["outcome"]
    read_ids = set(read(measured["memory_before"], measured["query"])["evidence_ids"])
    retrieved = [_scope_record(record) for record in measured["memory_before"]
                 if record["memory_id"] in read_ids]
    writes = [_scope_record(operation["record"]) for operation in measured["executed_operations"]
              if "record" in operation]
    wanted = expected["response"]
    value = wanted["value"]
    reference = wanted["mode"], tuple(sorted(value)) if isinstance(value, list) else value
    write_diff = _record_differences(writes, expected["new_write"])
    retrieval_diff = _record_differences(retrieved, expected["retrieves"])
    text = " ".join(turn["text"] for turn in measured["opening_messages"] if turn.get("role") == "user")
    response_exact = _normalize_response(response) == reference
    annotated_values = {(record["attribute"], record["value"]) for record in expected["new_write"]}
    extra_values = [record for record in write_diff["extra"]
                    if (record["attribute"], record["value"]) not in annotated_values]
    citations = _citation_audit(measured)
    issues = {
        "response_mismatch_with_reported_speech": not response_exact and bool(REPORTED_SPEECH.search(text)),
        "unannotated_write_value_with_reported_speech": bool(extra_values) and bool(REPORTED_SPEECH.search(text)),
        "write_scope_mismatch": bool(write_diff["scope"]),
        "retrieval_scope_mismatch": bool(retrieval_diff["scope"]),
        "extra_write": bool(write_diff["extra"]),
        "missing_write": bool(write_diff["missing"]),
        "extra_retrieval": bool(retrieval_diff["extra"]),
        "missing_retrieval": bool(retrieval_diff["missing"]),
        "invalid_citation_reference": bool(citations["unknown_response_ids"] or citations["answer_without_citation"]),
        "invalid_write_quote": bool(citations["invalid_write_quote_ids"]),
    }
    return {
        "visit": expected["visit"],
        "response_exact": response_exact,
        "strict_supported": bool(measured["current_supported"]),
        "new_write_exact": not write_diff["extra"] and not write_diff["missing"],
        "retrieval_exact": not retrieval_diff["extra"] and not retrieval_diff["missing"],
        "actual_response": response,
        "actual_new_write": writes,
        "actual_retrieval": retrieved,
        "authored": expected,
        "write_differences": write_diff,
        "retrieval_differences": retrieval_diff,
        "citation_audit": citations,
        "audit_flags": issues,
        "trace": {name: measured[name] for name in
                  ("query", "opening_messages", "memory_before", "executed_operations")},
    }


def _validate(stories, gold):
    grouped = defaultdict(list)
    for story in stories:
        grouped[story["root_id"]].append(story)
    authors = {case["id"]: case["visits"] for case in gold}
    if len(authors) != len(gold) or not grouped or set(grouped) != set(authors):
        raise ValueError("Each executable root needs exactly one corresponding semantic annotation")
    for root, worlds in grouped.items():
        first = worlds[0]
        opening = canonical((first["initial_memory"], [(visit["query"], visit["messages"])
                                                      for visit in first["visits"]]))
        expected = authors[root]
        if [visit["visit"] for visit in expected] != list(range(1, len(first["visits"]) + 1)):
            raise ValueError(f"Semantic visits must match the ordered executable visits for {root}")
        for story in worlds:
            public = canonical((story["initial_memory"], [(visit["query"], visit["messages"])
                                                        for visit in story["visits"]]))
            if public != opening:
                raise ValueError(f"No-question public observations vary across latent worlds for {root}")
    return grouped, authors


def semantic_comparison(stories, gold):
    """Compare V4/V5 against independent labels without issuing any questions.

    Inputs are executable story dictionaries and a list of annotated cases,
    matching the reserve loaders' return values. The caller verifies and locks
    the selected run before loading its reserve. This function has no file I/O.
    One replay per root suffices because the no-question observations are
    required to be identical across that root's latent worlds. Semantic exact
    matches use the independently annotated fields, while strict_supported is
    an additional simulator result. Reported-speech flags identify candidates
    for attribution review and do not assert the cause of a mismatch.
    """
    stories, gold = tuple(stories), tuple(gold)
    grouped, authors = _validate(stories, gold)
    parsers = {}
    for name, execute in (("v4", execute_v4), ("v5", execute_v5)):
        cases = []
        for root, worlds in grouped.items():
            story = worlds[0]
            actions = ("PROCEED",) * len(story["visits"])
            trace = execute(story, actions)
            if trace["questions"] or any(visit["events"] for visit in trace["visits"]):
                raise ValueError("The semantic replay must not deliver any optional clarification")
            visits = [_compare_visit(measured, expected)
                      for measured, expected in zip(trace["visits"], authors[root], strict=True)]
            cases.append({"root_id": root, "family": story["scenario_family"],
                          "actions": list(actions), "latent_worlds": len(worlds), "visits": visits})
        rows = [visit for case in cases for visit in case["visits"]]
        parsers[name] = {
            "roots": len(cases), "visits": len(rows),
            "matches": {metric: sum(visit[metric] for visit in rows) for metric in MATCHES},
            "complete_semantic_visits": sum(all(visit[metric] for metric in
                                                ("response_exact", "new_write_exact", "retrieval_exact"))
                                             for visit in rows),
            "audit_counts": {flag: sum(visit["audit_flags"][flag] for visit in rows)
                             for flag in rows[0]["audit_flags"]},
            "cases": cases,
        }
    return {
        "protocol": "paired_semantic_no_question_v4_v5",
        "roots": len(grouped), "visits": parsers["v4"]["visits"],
        "parsers": parsers,
        "delta_matches_v5_minus_v4": {metric: parsers["v5"]["matches"][metric] - parsers["v4"]["matches"][metric]
                                      for metric in MATCHES},
        "audit_interpretation": "Reported-speech flags mark attribution review candidates; citation checks verify references and exact quotation text.",
    }
