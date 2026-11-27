"""Evaluator-only evidence audit. Latent factual success is kept separate."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta

EVALUATOR_VERSION = "evidence_v6_day_session_scope"

def response_valid(action):
    if not isinstance(action, dict) or set(action) != {"mode", "value", "branches", "evidence_ids"}:
        return False
    if action["mode"] not in ("ANSWER", "DEFER", "CONDITIONALIZE"):
        return False
    if not isinstance(action["branches"], list) or not isinstance(action["evidence_ids"], list):
        return False
    if not all(isinstance(i, str) for i in action["evidence_ids"]):
        return False
    if action["mode"] == "ANSWER":
        return isinstance(action["value"], str) and bool(action["value"]) and not action["branches"]
    if action["value"] is not None:
        return False
    if action["mode"] == "DEFER":
        return not action["branches"]
    return bool(action["branches"]) and all(isinstance(b, dict) and set(b) == {"condition", "value"}
                                             and isinstance(b["condition"], (str, dict))
                                             and isinstance(b["value"], str) for b in action["branches"])


def score_response(response, observation, factual_value, *, query=None, faithful=True):
    from ..policy.rule_controller import compatible_worlds
    if not response_valid(response):
        return {"success": 0, "factual": 0, "justified": False, "unsupported_commitment": 0,
                "utility": -1, "valid": False}
    worlds = compatible_worlds(observation, query=query)
    values = {world.get("value") for world in worlds}
    visible_ids = {m["memory_id"] for m in observation.get("memory", [])}
    visible_ids.update(u["turn_id"] for u in observation.get("user_messages", []))
    cited = set(response["evidence_ids"])
    support_ids = {source for world in worlds for source in world.get("evidence_ids", [])
                   if world.get("value") == response.get("value") or response["mode"] != "ANSWER"}
    citations_ok = bool(cited) and cited <= visible_ids and bool(cited & support_ids)
    answer = response["mode"] == "ANSWER"
    factual = int(answer and response["value"] == factual_value)
    justified = answer and len(values) == 1 and None not in values and response["value"] in values and citations_ok and faithful
    success = int(factual and justified)
    if answer:
        utility = 1 if success else -1
    elif response["mode"] == "DEFER":
        utility = 0
    else:
        # A conditional statement is admissible only for visible alternatives;
        # it cannot collect free neutral reward for invented food values.
        branch_values = {b["value"] for b in response["branches"]}
        from .conditions import condition_errors
        errors = condition_errors(response, observation, query=query)
        utility = 0 if branch_values <= (values - {None}) and citations_ok and not errors else -1
    return {"success": success, "factual": factual, "justified": bool(justified),
            "unsupported_commitment": int(answer and not justified), "utility": utility,
            "valid": True, "admissible_values": sorted(v for v in values if v is not None)}


def probe_queries(observation, *additional_records):
    """Public calendar and context probes, independent of future latent truth."""
    query = deepcopy(observation["query"])
    day = date.fromisoformat(query["date"][:10])
    records = list(observation.get("memory", [])) + list(additional_records)
    days = {day + timedelta(days=offset) for offset in (-30, -1, 0, 1, 7, 31, 366)}
    contexts = [query.get("context"), "casa", "oficina", None]
    for record in records:
        scope = record.get("scope", {})
        if scope.get("context") is not None and scope["context"] not in contexts:
            contexts.append(scope["context"])
        for field in ("valid_from", "valid_until_exclusive"):
            if scope.get(field):
                boundary = date.fromisoformat(scope[field])
                for offset in (-1, 0, 1):
                    try:
                        days.add(boundary + timedelta(days=offset))
                    except OverflowError:
                        pass
    probes = []
    for at in sorted(days):
        for context in contexts:
            # Probe both the acquisition session and another session, including
            # at the same calendar date: session scope is not a day-long rule.
            for session in {query.get("session_id"), "probe_session"}:
                probes.append({**query, "date": at.isoformat(), "session_id": session,
                               "context": context})
    return probes


def audit_operation(operation, observation):
    """Compare scope semantics and provenance, allowing equivalent representations.

    The executor intentionally never calls this function. False but well formed
    records are persisted unchanged and are penalized here afterwards.
    """
    from ..policy.rule_controller import expected_memory, compatible_worlds
    from ..readers.symbolic import read
    if operation.get("op") == "NOOP":
        return True
    record = operation.get("record", {})
    sources = {m["turn_id"]: m["text"] for m in observation.get("user_messages", []) if m.get("role", "user") == "user"}
    for memory in observation.get("memory", []):
        for quote in memory.get("evidence_quotes", []):
            sources.setdefault(quote["turn_id"], quote["text"])
    refs = record.get("source_turn_ids", [])
    quotes = record.get("evidence_quotes", [])
    if not refs or not quotes or not set(refs) <= sources.keys():
        return False
    if {q.get("turn_id") for q in quotes} != set(refs):
        return False
    if any(not q.get("text") or q["text"] not in sources.get(q.get("turn_id"), "") for q in quotes):
        return False
    # The cited turns themselves must entail the write; a correct value elsewhere
    # in the observation cannot validate an unrelated quote as its provenance.
    citation_observation = deepcopy(observation)
    citation_observation["user_messages"] = [message for message in observation.get("user_messages", [])
                                              if message["turn_id"] in refs]
    citation_observation["memory"] = [memory for memory in observation.get("memory", [])
                                      if set(memory.get("source_turn_ids", [])) <= set(refs)]
    cited_reference = expected_memory(citation_observation)
    if not any(m.get("attribute") == record.get("attribute") and m.get("value") == record.get("value")
               for m in cited_reference):
        return False
    reference = expected_memory(observation)
    same_attribute = [m for m in reference if m["attribute"] == record.get("attribute")]
    if not any(m["value"] == record.get("value") for m in same_attribute):
        return False
    candidate = {**deepcopy(record), "memory_id": "candidate", "supersedes": operation.get("target_ids", [])}
    if operation.get("op") == "MARK_UNRESOLVED" and candidate.get("scope", {}).get("kind") != "unknown":
        return False
    # A newly created record may not introduce claims stronger than the visible
    # evidence on any probe. Missing coverage is measured separately.
    if candidate.get("scope", {}).get("kind") == "unknown":
        # An unresolved marker still asserts its dates/context. It cannot smuggle
        # arbitrary scope metadata past probes by never committing to a value.
        if not any(m.get("scope") == candidate["scope"] and m.get("value") == candidate.get("value")
                   for m in same_attribute if m.get("scope", {}).get("kind") == "unknown"):
            return False
    # A general preference can be true while a specific exception wins the
    # actual recommendation. Audit the assertion against cited supporting
    # records, and its operational consequences against the complete memory.
    # Comparing candidate in isolation to a full-world answer confuses these.
    from ..readers.symbolic import applicability
    resulting_memory = deepcopy(observation.get("memory", [])) + [candidate]
    supporters = [item for item in cited_reference
                  if item.get("attribute") == candidate.get("attribute")
                  and item.get("value") == candidate.get("value")
                  and item.get("status") == "confirmed"]
    for query in probe_queries(observation, candidate, *reference, *cited_reference):
        query = {**query, "attribute": candidate.get("attribute")}
        if applicability(candidate, query) is True:
            if not any(applicability(item, query) is True for item in supporters):
                return False
        result = read(resulting_memory, query)
        if result.get("mode") == "ANSWER":
            admissible = {w.get("value") for w in compatible_worlds(observation, query=query)}
            if admissible != {result["value"]}:
                return False
    if operation.get("op") == "SUPERSEDE":
        targets = set(operation.get("target_ids", []))
        original = {item["memory_id"]: item for item in observation.get("memory", [])}
        if not targets or not targets <= original.keys():
            return False
        if any(original[mid]["attribute"] != candidate["attribute"] for mid in targets):
            return False
        # A new general preference does not revoke a separately documented
        # contextual/temporary exception, even if their current values coincide.
        # Preserve that relationship rather than silently erasing its scope.
        from ..readers.symbolic import scope_specificity
        if any(scope_specificity(original[mid]) > scope_specificity(candidate)
               for mid in targets):
            return False
    return True


class LegacyEvaluator:
    """Historical scorer for explicit audit/reproduction only, never training."""
    def score_current(self, episode_trace):
        return score_response(episode_trace["response"], episode_trace["observations"]["respond"],
                              episode_trace["episode"]["truth"]["current_value"])

    def score_memory(self, episode_trace):
        observation = episode_trace["observations"]["write"]
        operations = episode_trace.get("executed_operations", [])
        checks = [audit_operation(op, observation) for op in operations if op.get("op") != "NOOP"]
        return {"unsupported_operations": sum(not value for value in checks), "operations": len(checks),
                "write_precision": sum(checks) / len(checks) if checks else None,
                "operation_faithfulness": checks}

    def score_future(self, episode_trace):
        from ..readers.symbolic import read
        memory = episode_trace["final_memory"]
        query = episode_trace["episode"]["future_query"]
        response = episode_trace["future_response"]
        if not response_valid(response):
            return {"success": 0, "factual": 0, "justified": False,
                    "unsupported_commitment": 0, "utility": -1, "valid": False}
        full = episode_trace["interaction_observation"]
        # A reader's internal consistency does not establish fidelity. Compare
        # its value against ALL evidence the user actually supplied before S2
        # closed, including statements omitted from the persisted memory.
        observed = {**full, "memory": full["memory"]}
        score = score_response(response, observed, episode_trace["episode"]["truth"]["future_value"], query=query)
        used = set(response.get("evidence_ids", []))
        relevant = [m for m in memory if m.get("memory_id") in used]
        full_ids = {m["memory_id"] for m in full["memory"]}
        # Re-evaluate citations as actual persisted record IDs, then separately
        # require their user provenance; original S1 IDs remain trustworthy.
        if response_valid(response) and response["mode"] == "ANSWER":
            from ..policy.rule_controller import compatible_worlds
            values = {w.get("value") for w in compatible_worlds(observed, query=query)}
            faithful = bool(used) and used <= {m["memory_id"] for m in memory}
            for record in relevant:
                if record["memory_id"] not in full_ids:
                    op = {"op": "SUPERSEDE" if record.get("supersedes") else "ADD", "record": record,
                          "target_ids": record.get("supersedes", [])}
                    # Acquisition and writing have different information sets
                    # under I1. A lucky unseen scope may be factually right but
                    # is not a supported write.
                    faithful = faithful and audit_operation(op, episode_trace["observations"]["write"])
            # Citations must also support the selected value in the actual
            # persisted state; citing an unrelated or expired record is not
            # provenance for a correct guess.
            from ..readers.symbolic import applicability
            faithful = faithful and any(record.get("value") == response["value"]
                                        and applicability(record, query) is True
                                        for record in relevant)
            score["justified"] = bool(faithful and values == {response["value"]})
            score["success"] = int(bool(score["factual"]) and score["justified"])
            score["unsupported_commitment"] = int(not score["justified"])
            score["utility"] = 1 if score["success"] else -1
        return score


def persisted_provenance(episode_trace, response):
    """Validate citations in their S3 namespace, with S2 write authorization.

    Original IDs cannot launder mutated records. New IDs require a faithful
    operation from the evidence actually delivered to this writer, including I1.
    This helper never consults latent truth or an actor's future prediction.
    """
    from ..readers.symbolic import applicability
    memory = episode_trace["final_memory"]
    query = episode_trace["episode"]["future_query"]
    full = episode_trace["interaction_observation"]
    original = {r["memory_id"]: r for r in full["memory"]}
    stored = {r["memory_id"]: r for r in memory}
    cited = set(response["evidence_ids"])
    errors = []
    if not cited:
        errors.append("missing_citations")
    if len(stored) != len(memory):
        errors.append("duplicate_persisted_ids")
    if not cited <= stored.keys():
        errors.append("citation_not_in_persisted_memory")
    relevant = []
    for mid in sorted(cited):
        record = stored.get(mid)
        if record is None:
            continue
        if mid in original:
            if record != original[mid]:
                errors.append("mutated_initial_record:" + mid)
        else:
            operation = {"op": "SUPERSEDE" if record.get("supersedes") else "ADD",
                         "record": record, "target_ids": record.get("supersedes", [])}
            if not audit_operation(operation, episode_trace["observations"]["write"]):
                errors.append("unsupported_record:" + mid)
        if record["attribute"] == query["attribute"] and applicability(record, query) is not False:
            relevant.append(record)
    return {"valid": not errors, "errors": errors, "relevant_records": relevant}


class Evaluator(LegacyEvaluator):
    """Production evaluator shared by rollouts, final evaluation and diagnostics."""
    version = EVALUATOR_VERSION

    def score_future(self, episode_trace):
        response = episode_trace["future_response"]
        if not response_valid(response):
            return {"success": 0, "factual": 0, "justified": False,
                    "unsupported_commitment": 0, "utility": -1, "valid": False,
                    "response_category": "invalid", "evaluator_version": self.version}
        full = episode_trace["interaction_observation"]
        query = episode_trace["episode"]["future_query"]
        score = score_response(response, full, episode_trace["episode"]["truth"]["future_value"], query=query)
        score["evaluator_version"] = self.version
        if response["mode"] == "DEFER":
            score["response_category"] = "faithful_deferral"
            return score
        from ..policy.rule_controller import compatible_worlds
        from ..readers.symbolic import applicability
        admissible = {w["value"] for w in compatible_worlds(full, query=query)}
        provenance = persisted_provenance(episode_trace, response)
        score["provenance_valid"] = provenance["valid"]
        score["provenance_errors"] = provenance["errors"]
        relevant = provenance["relevant_records"]
        if response["mode"] == "ANSWER":
            supported = (provenance["valid"] and admissible == {response["value"]}
                and any(r["value"] == response["value"] and applicability(r, query) is True for r in relevant))
            score.update(justified=bool(supported), success=int(bool(score["factual"]) and supported),
                         unsupported_commitment=int(not supported))
            score["utility"] = 1 if score["success"] else -1
            score["response_category"] = "correct_supported" if score["success"] else "incorrect_or_unsupported"
        else:
            claimed = {b["value"] for b in response["branches"]}
            represented = {r["value"] for r in relevant}
            from .conditions import condition_errors
            errors = condition_errors(response, full, query=query, records=relevant)
            faithful = bool(provenance["valid"] and claimed and claimed <= (admissible - {None})
                            and claimed <= represented and not errors)
            score.update(utility=0 if faithful else -1, conditional_memory_provenance_valid=faithful,
                         conditional_conditions_valid=not errors, conditional_condition_errors=errors,
                         response_category="faithful_conditional" if faithful else "unsupported_conditional")
        return score
