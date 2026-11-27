"""Read-only, evaluator-only diagnosis of archived SPEC-RL-003 PIPELINE failures.

No function in this module is imported by a policy. Author annotations are used
only to audit extraction after the archived actions have already been taken.
The exact logged observations are retained, including the legacy decision hint;
they must never be used as SPEC-RL-004 training input.
"""
from collections import Counter, defaultdict
from datetime import datetime
import argparse
import hashlib
import json
from pathlib import Path


CAUSES = {
    "1": "Recuperación de memoria",
    "2": "Extracción lingüística",
    "3": "Interpretación temporal",
    "4": "Detección de conflictos",
    "5": "Selección de aclaración",
    "6": "Actualización de memoria",
    "7": "Selección de respuesta final",
    "8": "Corrección automática realizada por el contrato",
}
FIELDS = ("polarity", "kind", "scope", "observed_at", "recorded_at",
          "valid_from", "valid_to", "source", "uncertainty")


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rank(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _identity(fact):
    return fact["evidence_id"], fact["food"]


def _normalized(field, value):
    if field.endswith("_at") or field in ("valid_from", "valid_to"):
        if value:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    return value


def _communicated_reference(scenario, row, session_index):
    """Only already communicated events and actually elicited replies, no future."""
    facts, events = [], []
    for index, session in enumerate(scenario["sessions"][:session_index + 1]):
        facts.extend(session["incoming"])
        events.extend(session.get("events", []))
        for step in row["steps"]:
            if step["session_index"] != index or "response" not in step:
                continue
            key = step["executed_action"].get("question_key")
            response = session.get("responses", {}).get(key, {})
            # Reply annotations become evaluator evidence only after elicitation.
            fact = response.get("fact")
            if fact:
                fact = dict(fact, memory_id="clar_" + fact["memory_id"],
                            evidence_id="clar_" + fact["evidence_id"])
                facts.append(fact)
            events.append({"at": step["at"], "text": step["response"],
                           "source": "clarification", "question_key": key,
                           "message_id": fact["evidence_id"] if fact else None})
    return facts, events


def diagnose_case(row, session, scenario, oracle_row, source):
    index = session["session_index"]
    pairs = [(step, obs) for step, obs in zip(row["steps"], row["policy_observations"])
             if step["session_index"] == index]
    if len(pairs) != len(session["steps"]):
        raise ValueError("Missing exact archived policy observations")
    terminal_step, observation = pairs[-1]
    facts = observation["historical_memory"] + observation["incoming_evidence"]
    public = {_identity(f): f for f in facts}
    reference, events = _communicated_reference(scenario, row, index)
    discrepancies = []
    for expected in reference:
        actual = public.get(_identity(expected))
        if actual is None:
            discrepancies.append({"evidence_id": expected["evidence_id"], "food": expected["food"],
                                  "field": "fact", "expected": "communicated author annotation",
                                  "observed": "absent from policy fact index",
                                  "origin": "extraction_or_prior_memory_omission"})
            continue
        for field in FIELDS:
            if _normalized(field, expected.get(field)) != _normalized(field, actual.get(field)):
                discrepancies.append({"evidence_id": expected["evidence_id"], "food": expected["food"],
                                      "field": field, "expected": expected.get(field),
                                      "observed": actual.get(field), "origin": "extraction_or_adapter"})
    consulted = set(terminal_step["consulted_memory_ids"])
    target = observation.get("target_food")
    relevant_available = [f for f in facts if not target or f["food"] == target or f["kind"] == "RESTRICTION"]
    not_retrieved = [f["memory_id"] for f in relevant_available if f["memory_id"] not in consulted]
    contract = [dict(step_index=j, shield_reasons=s["shield_reasons"],
                     memory_violations=s["memory_violations"], proposed=s["proposed_action"],
                     executed=s["executed_action"], actual_memory_operations=s["actual_memory_operations"])
                for j, (s, _) in enumerate(pairs) if s["shield_reasons"] or s["memory_violations"]]
    bad_updates = any("UNJUSTIFIED_STABLE_UPDATE" in s["memory_violations"] for s, _ in pairs)
    causes = []
    if not_retrieved:
        causes.append("1")
    if discrepancies:
        causes.append("2")
    if bad_updates:
        causes.append("6")
    # Downstream symptoms are not automatically independent causes. A final
    # failure with corrupted facts does not establish faulty final selection.
    classifications = {}
    for number, name in CAUSES.items():
        classifications[number] = {"name": name,
            "status": "probable_origin" if number in causes else
                      "observed_intervention" if number == "8" and contract else
                      "downstream_symptom_not_isolated" if number == "3" and session["temporal_correct"] < session["temporal_total"] else
                      "not_established"}
    reasons = []
    if discrepancies:
        reasons.append("La extracción/adapter difiere de anotaciones de evidencia ya comunicada; no se usó preferencia latente.")
    if not_retrieved:
        reasons.append("Hechos pertinentes presentes en la observación no figuran en la recuperación terminal.")
    if session["temporal_correct"] < session["temporal_total"]:
        reasons.append("La métrica temporal falla; puede ser consecuencia de polaridad/tipo/alcance extraídos incorrectamente.")
    oracle_session = next(s for s in oracle_row["sessions"] if s["session_index"] == index)
    reasons.append("La sesión emparejada ORACLE_STRUCTURED acierta." if oracle_session["correct"] else
                   "La sesión emparejada ORACLE_STRUCTURED también falla.")
    return {
        "case_id": f'{row["episode_id"]}:session_{index}', "user_id": row["user_id"],
        "family": row["family"], "session_index": index, "policy": row["policy_name"],
        "condition": "PIPELINE", "historical_source": source,
        "original_input": {"query": session["query"], "query_at": session["query_at"],
                           "current_session_events": scenario["sessions"][index].get("events", [])},
        "available_memories": {"stored_facts_before_session": session["memory_before"],
                               "visible_facts_at_terminal_decision": facts,
                               "already_communicated_raw_evidence": events},
        "retrieved_memories": [f for f in facts if f["memory_id"] in consulted],
        "retrieved_memory_ids": sorted(consulted),
        "missing_relevant_available_memory_ids": not_retrieved,
        "policy_observation": observation, "observation_status": "exact_archived_not_reconstructed",
        "decision_trace": [{"policy_observation": o, "proposed_action": s["proposed_action"],
                            "executed_action": s["executed_action"], "reward": s["reward"],
                            "actual_memory_operations": s["actual_memory_operations"],
                            "shield_reasons": s["shield_reasons"], "memory_violations": s["memory_violations"],
                            "response": s.get("response"), "correct": s.get("correct")}
                           for s, o in pairs],
        "proposed_action": terminal_step["proposed_action"], "executed_action": terminal_step["executed_action"],
        "reward": terminal_step["reward"], "session_return": session["return"],
        "final_result": {"correct": session["correct"], "grounded": session["grounded"],
                         "expected_evaluator_only": session["expected"],
                         "matched_oracle_correct": oracle_session["correct"],
                         "matched_oracle_action": oracle_session["action"]},
        "extraction_discrepancies": discrepancies,
        "probable_origin_categories": causes,
        "origin_multiplicity": "single_probable_origin" if len(causes) == 1 else
                               "multiple_probable_origins" if len(causes) > 1 else "undetermined",
        "classification": classifications, "contract_interventions": contract,
        "explanation": reasons,
        "causal_limit": "Diagnóstico retrospectivo probable; el contraste Oracle cambia toda la extracción y no identifica el efecto causal de cada campo.",
        "policy_error_given_correct_information": "not_established" if discrepancies else "requires_adjudication",
    }


def run_diagnosis(source="reports/rl_longitudinal/final_matched", output="reports/rl_observability/diagnosis",
                  *, sample_size=120, seed=4004, policy="CLARIFICATION_RULE_BASED"):
    source, output = Path(source), Path(output)
    pipeline_file, oracle_file = source / "pipeline" / f"{policy}.jsonl", source / "test" / f"{policy}.jsonl"
    private_file = source / "evaluator_private/test.json"
    rows = load_jsonl(pipeline_file)
    # Keep only the matched terminal results; loading every Oracle observation
    # would duplicate large archived histories without adding diagnostic value.
    oracle = {}
    with oracle_file.open() as stream:
        for line in stream:
            r = json.loads(line)
            oracle[r["episode_id"]] = {"sessions": [
                {k: s[k] for k in ("session_index", "correct", "action")}
                for s in r["sessions"]]}
    scenarios = {r["episode_id"]: r for r in json.loads(private_file.read_text())}
    by_family = defaultdict(list)
    for row in rows:
        if any(not s["correct"] for s in row["sessions"]):
            by_family[row["family"]].append(row)
    families = sorted(by_family)
    if sample_size < 100:
        raise ValueError("SPEC-RL-004 requires at least 100 failed decisions")
    # Equal family allocation; one failure per user before any repeated user.
    selected = []
    for family_index, family in enumerate(families):
        count = sample_size // len(families) + int(family_index < sample_size % len(families))
        candidates = sorted(by_family[family], key=lambda r: _rank(seed, r["user_id"]))
        if len(candidates) < count:
            raise ValueError("Insufficient distinct users for balanced failure sample")
        for index, row in enumerate(candidates[:count]):
            failures = sorted([s for s in row["sessions"] if not s["correct"]], key=lambda s: s["session_index"])
            session = failures[index % len(failures)]
            selected.append(diagnose_case(row, session, scenarios[row["episode_id"]], oracle[row["episode_id"]], str(pipeline_file)))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "failed_decisions.jsonl").open("w") as stream:
        for case in selected:
            stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    contract_summary = {}
    for file in sorted((source / "pipeline").glob("*.jsonl")):
        counts, shields, corrections = Counter(), Counter(), Counter()
        with file.open() as stream:
            for line in stream:
                archived = json.loads(line)
                counts.update(users=1, sessions=len(archived["sessions"]),
                              failed_sessions=sum(not s["correct"] for s in archived["sessions"]),
                              proposed_decisions=len(archived["steps"]))
                for step in archived["steps"]:
                    counts["contract_intervened_decisions"] += bool(step["shield_reasons"] or step["memory_violations"])
                    shields.update(step["shield_reasons"])
                    corrections.update(step["memory_violations"])
        contract_summary[file.stem] = dict(counts, shield_reasons=dict(shields),
                                          memory_correction_reasons=dict(corrections))
    summary = {
        "sample_size": len(selected), "unique_users": len({c["user_id"] for c in selected}),
        "seed": seed, "policy": policy, "selection": "equal families, hash-ranked users, alternating failed session positions, one per user",
        "population_failed_sessions": sum(not s["correct"] for r in rows for s in r["sessions"]),
        "population_users": len(rows), "population_sessions": sum(len(r["sessions"]) for r in rows),
        "by_family": dict(Counter(c["family"] for c in selected)),
        "by_session_index": dict(Counter(c["session_index"] for c in selected)),
        "probable_origins": {CAUSES[k]: sum(k in c["probable_origin_categories"] for c in selected) for k in CAUSES},
        "origin_multiplicity": {label: sum(c["origin_multiplicity"] == label for c in selected)
                                for label in ("single_probable_origin", "multiple_probable_origins", "undetermined")},
        "extraction_discrepancy_fields": dict(Counter(d["field"] for c in selected for d in c["extraction_discrepancies"])),
        "temporal_symptom_cases": sum(c["classification"]["3"]["status"] == "downstream_symptom_not_isolated" for c in selected),
        "matched_oracle_correct_cases": sum(c["final_result"]["matched_oracle_correct"] for c in selected),
        "all_pipeline_methods_contract_audit": contract_summary,
        "source_sha256": {str(p): sha256(p) for p in (pipeline_file, oracle_file, private_file, Path(__file__))},
        "limitations": ["Synthetic generator annotations, no independent human adjudication.",
                        "Stratified failure-only sample is descriptive, not a population error-rate estimate.",
                        "Multiple field discrepancies from one extractor are one probable origin, not multiple independent causes.",
                        "Temporal/conflict/question/final errors can be downstream of corrupted extraction; no RL correction claim.",
                        "Legacy required_decision is retained only in this offline diagnostic archive and forbidden in new policy inputs.",
                        "Archive evidence IDs logged as consulted do not establish that a policy consumed their raw text."]}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="reports/rl_longitudinal/final_matched")
    parser.add_argument("--output", default="reports/rl_observability/diagnosis")
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=4004)
    args = parser.parse_args()
    result = run_diagnosis(args.source, args.output, sample_size=args.sample_size, seed=args.seed)
    print(json.dumps({k: result[k] for k in ("sample_size", "unique_users", "probable_origins", "matched_oracle_correct_cases")}, ensure_ascii=False))
