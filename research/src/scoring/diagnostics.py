"""Evaluator-only secondary diagnostics with explicit finite-scope denominators.

Coverage of the declared reuse query and synthetic scope probes are reported by
their actual names. They do not stand in for an unannotated global fact inventory
or for naturally occurring scope opportunities.
"""
from __future__ import annotations

from copy import deepcopy

from ..policy.rule_controller import compatible_worlds, expected_memory
from ..readers.symbolic import read, applicability
from .evidence import audit_operation, probe_queries


def _values(observation, query=None):
    return {w["value"] for w in compatible_worlds(observation, query=query)}


def _faithful(record, initial, observation):
    original = {r["memory_id"]: r for r in initial}
    if record["memory_id"] in original:
        return record == original[record["memory_id"]]
    return audit_operation({"op": "SUPERSEDE" if record.get("supersedes") else "ADD",
                            "record": record, "target_ids": record.get("supersedes", [])}, observation)


def episode_diagnostics(trace, metrics):
    """Return rates alongside numerators, denominators and identification limits.

    The trace is private evaluator data, never passed to actor tools. Acquisition
    is judged against evidence available before/after the actual emitted reply;
    a reply suppressed from WRITE in I1 is still physically acquired.
    """
    episode = trace.get("episode", {})
    observations = trace.get("observations", {})
    prior = observations.get("select")
    interaction = trace.get("interaction_observation")
    final = trace.get("final_memory")
    future = episode.get("future_query")
    if not all(isinstance(v, dict) for v in (prior, interaction, future)) or not isinstance(final, list):
        return {"diagnostics_status": "not_identifiable", "diagnostics_reason": "Incomplete private trace"}
    decision = trace.get("parsed_actions", {}).get("select", {})
    asked = int(metrics.get("questions", decision.get("action") == "ASK"))
    slot = decision.get("slot") if asked else None
    truth = episode.get("truth", {})
    known_sufficiency = "decisive_slot" in truth
    sufficient = truth.get("decisive_slot") is None
    result = {
        "diagnostics_status": "executed",
        "unnecessary_questions_numerator": int(bool(asked and sufficient)) if known_sufficiency else None,
        "unnecessary_questions_denominator": int(sufficient) if known_sufficiency else None,
        "unnecessary_question_rate": float(bool(asked)) if known_sufficiency and sufficient else None,
        "unnecessary_questions_definition": "Questions on preregistered initially sufficient episodes / initially sufficient episodes",
        "unnecessary_questions_reason": None if known_sufficiency else "No preregistered sufficiency annotation",
        "acquisition_questions_denominator": asked,
    }
    after = deepcopy(prior)
    reply = trace.get("user_reply")
    if asked and isinstance(reply, dict):
        after["user_messages"].append(deepcopy(reply))
    before_now, after_now = _values(prior), _values(after)
    before_later, after_later = _values(prior, future), _values(after, future)
    current_gain = asked and len(before_now) > 1 and len(after_now) == 1 and None not in after_now
    future_gain = asked and len(before_later) > 1 and len(after_later) == 1 and None not in after_later
    relevant = known_sufficiency and slot == truth.get("decisive_slot") and slot is not None
    informative = bool(asked and reply and (after_now != before_now or after_later != before_later))
    need_now = len(before_now) != 1 or None in before_now
    need_future = len(before_later) != 1 or None in before_later
    need_any = need_now or need_future
    # Evaluation-only enumeration: never expose simulated replies/valid slots
    # in the actor's legal action set or its input observation.
    valid_slots = []
    for candidate_slot, candidate_reply in episode.get("replies", {}).items():
        if isinstance(candidate_reply, str):
            candidate_reply = {"turn_id": "u3", "text": candidate_reply}
        candidate_obs = deepcopy(prior)
        candidate_obs["user_messages"].append(deepcopy(candidate_reply))
        now_values, later_values = _values(candidate_obs), _values(candidate_obs, future)
        if ((need_now and len(now_values) == 1 and None not in now_values)
                or (need_future and len(later_values) == 1 and None not in later_values)):
            valid_slots.append(candidate_slot)
    resolvable = bool(valid_slots)
    result.update(need_now=int(need_now), need_future=int(need_future), need_any=int(need_any),
                  resolvable_by_available_slots=int(resolvable), valid_slots=sorted(valid_slots),
                  ask_requested=int(decision.get("action") == "ASK"), ask_emitted=int(decision.get("action") == "ASK"),
                  ask_delivered=asked, slot=slot,
                  user_response_received=int(bool(asked and reply and trace.get("user_response_status") != "unavailable")),
                  response_informative=int(informative),
                  necessary_omission_numerator=int(need_any and resolvable and not asked),
                  necessary_omission_denominator=int(need_any and resolvable),
                  unnecessary_ask_numerator=int(asked and not need_any), unnecessary_ask_denominator=int(not need_any),
                  slot_accuracy_numerator=int(asked and need_any and slot in valid_slots),
                  slot_accuracy_denominator=int(asked and need_any),
                  complete_acquisition_numerator=int(need_any and asked and slot in valid_slots and informative),
                  complete_acquisition_denominator=int(need_any))
    result.update({"useful_acquisition_current_numerator": int(bool(relevant and informative and current_gain)),
                   "useful_acquisition_future_numerator": int(bool(relevant and informative and future_gain)),
                   "useful_acquisition_numerator": int(bool(relevant and informative and (current_gain or future_gain))),
                   "useful_acquisition_current_rate": float(bool(relevant and informative and current_gain)) if asked else None,
                   "useful_acquisition_future_rate": float(bool(relevant and informative and future_gain)) if asked else None,
                   "useful_acquisition_definition": "Decisive-slot questions reducing compatible values to a justified singleton / actual questions"})
    if not known_sufficiency:
        for key in ("useful_acquisition_current_numerator", "useful_acquisition_future_numerator",
                    "useful_acquisition_numerator", "useful_acquisition_current_rate", "useful_acquisition_future_rate"):
            result[key] = None
        result["useful_acquisition_reason"] = "No preregistered decisive-variable annotation"
    # Facts used by the faithful symbolic reference for the declared S3 query.
    # Counting a new paraphrase of that same fact must not add a denominator.
    reference = expected_memory(interaction)
    wanted = read(reference, future)
    reference_ids = set(wanted.get("evidence_ids", [])) if wanted["mode"] == "ANSWER" else set()
    facts = {}
    for record in reference:
        if record["memory_id"] in reference_ids:
            key = (record["attribute"], record["value"], repr(record["scope"]))
            facts[key] = record
    write_observation = observations.get("write", interaction)
    fidelity = {r["memory_id"]: _faithful(r, interaction["memory"], write_observation) for r in final}
    actual = read(final, future)
    used = set(actual.get("evidence_ids", []))
    covered = 0
    for record in facts.values():
        if actual["mode"] == "ANSWER" and actual["value"] == record["value"] and any(
                r["memory_id"] in used and fidelity[r["memory_id"]]
                and r["attribute"] == record["attribute"] and r["value"] == record["value"]
                and applicability(r, future) is True for r in final):
            covered += 1
    result.update({"memory_coverage_at_reuse_numerator": covered,
                   "memory_coverage_at_reuse_denominator": len(facts),
                   "memory_coverage_at_reuse": covered / len(facts) if facts else None,
                   "memory_coverage_at_reuse_reason": None if facts else "No uniquely decisive observed fact for the declared S3 query",
                   "memory_coverage_definition": "Semantically available, provenance-audited decisive reference facts for declared S3 query",
                   "memory_coverage": None,
                   "memory_coverage_reason": "A complete independently annotated decisive-fact inventory is not available"})
    # A scope error is identified only where a committed value contradicts the
    # evidence-compatible values. Unknown scope alone is not a scope violation.
    scoped = any(r["scope"]["kind"] in {"interval", "session", "contextual"}
                 for r in [*reference, *final] if r["attribute"] == future["attribute"])
    probes = probe_queries(interaction, *reference, *final) if scoped else []
    temporal_errors = contextual_errors = commitments = wrong_commitments = 0
    for query in probes:
        response = read(final, query)
        if response["mode"] != "ANSWER":
            continue
        commitments += 1
        if response["value"] not in _values(interaction, query):
            wrong_commitments += 1
            if query.get("context") != interaction["query"].get("context"):
                contextual_errors += 1
            if (query["date"] != interaction["query"]["date"]
                    or query.get("session_id") != interaction["query"].get("session_id")):
                temporal_errors += 1
    result.update({"scope_probe_wrong_commitments": wrong_commitments,
                   "scope_probe_temporal_errors": temporal_errors,
                   "scope_probe_contextual_errors": contextual_errors,
                   "scope_probe_queries_denominator": len(probes),
                   "scope_probe_commitments_denominator": commitments,
                   "scope_probe_error_rate": wrong_commitments / len(probes) if probes else None,
                   "scope_probe_definition": "Wrong commitments over a synthetic calendar/context probe grid; diagnostic, not natural scope-opportunity rate",
                   "scope_probe_axis_definition": "Temporal/context counts describe varied probe axes, may overlap, and do not identify causal attribution",
                   "scope_error_rate": None,
                   "scope_error_reason": "No separately preregistered inventory of natural scope opportunities"})
    return result
