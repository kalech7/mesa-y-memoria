import collections
import random
import statistics


def rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def boolean_detection(rows, prediction, gold):
    valid = [r for r in rows if isinstance(r.get(prediction), bool)]
    tp = sum(r[prediction] and r[gold] for r in valid)
    fp = sum(r[prediction] and not r[gold] for r in valid)
    fn = sum(not r[prediction] and r[gold] for r in valid)
    return {"precision": rate(tp, tp + fp), "recall": rate(tp, tp + fn),
            "evaluable": len(valid), "not_evaluable": len(rows) - len(valid)}


def grouped_bootstrap(rows, value_key, seed=42, repeats=1000):
    groups = collections.defaultdict(list)
    for row in rows:
        if isinstance(row.get(value_key), (bool, int, float)):
            groups[row["person_id"]].append(float(row[value_key]))
    if len(groups) < 2:
        return {"low": None, "high": None, "clusters": len(groups), "reason": "Menos de dos historias evaluables"}
    rng, keys = random.Random(seed), sorted(groups)
    values = []
    for _ in range(repeats):
        sample = [value for key in rng.choices(keys, k=len(keys)) for value in groups[key]]
        values.append(statistics.mean(sample))
    values.sort()
    return {"low": values[int(.025 * repeats)], "high": values[min(repeats - 1, int(.975 * repeats))],
            "clusters": len(groups), "repeats": repeats, "confidence": .95,
            "caveat": "Condicional al generador sintético; no valida independencia de plantillas."}


def summarize_group(rows, seed):
    valid = [r for r in rows if "initial_action" in r]
    required = [r for r in valid if r["acceptable_actions"] == ["ASK"]]
    forbidden = [r for r in valid if "ASK" not in r["acceptable_actions"]]
    asked = [r for r in valid if r["initial_action"] == "ASK"]
    questions = [q for r in rows for q in r.get("questions", [])]
    eligible = [r for r in rows if r["unresolved_conflict"] and r["kind"] != "unknown"]
    clarified = [r for r in rows if r.get("clarification_resolved")]
    pair = [r for r in rows if r.get("retrieval_pair_recall") is not None]
    conditional = [r for r in valid if r["initial_action"] == "CONDITIONAL"]
    tokens = [t for r in rows for t in r.get("api_calls", [])]
    extraction = [flag for r in rows for flag in r.get("extraction_audit", [])]
    return {
        "n_attempted": len(rows), "histories": len({r["person_id"] for r in rows}), "initial_evaluable": len(valid),
        "initial_not_evaluable": len(rows) - len(valid),
        "technical_errors": sum(bool(r.get("technical_errors")) for r in rows),
        "technical_error_events": sum(len(r.get("technical_errors", [])) for r in rows),
        "action_accuracy": rate(sum(r["action_correct"] for r in valid), len(valid)),
        "action_accuracy_ci": grouped_bootstrap(valid, "action_correct", seed),
        "completion_adjusted_action_success": rate(sum(r["action_correct"] for r in valid), len(rows)),
        "historical_conflict": boolean_detection(rows, "initial_historical_conflict", "historical_conflict"),
        "unresolved_conflict": boolean_detection(rows, "initial_unresolved_conflict", "unresolved_conflict"),
        "ASK_recall_required": rate(sum(r["initial_action"] == "ASK" for r in required), len(required)),
        "ASK_false_positive_rate": rate(sum(r["initial_action"] == "ASK" for r in forbidden), len(forbidden)),
        "CONDITIONAL_appropriateness": rate(sum("CONDITIONAL" in r["acceptable_actions"] for r in conditional), len(conditional)),
        "question_utility_auto": rate(sum(q["resolved"] and (q["information_gain"] or 0) > 0 and not q["duplicate"] for q in questions), len(questions)),
        "question_utility_human": rate(0, 0),
        "repeated_questions": sum(q["duplicate"] for q in questions),
        "clarification_resolved": rate(len(clarified), len(asked)),
        "after_clarification_fidelity": rate(sum(r.get("after_fidelity") == "pass" for r in clarified), len(clarified)),
        "after_clarification_uncertain": sum(r.get("after_fidelity") in {None, "uncertain"} for r in clarified),
        "after_clarification_task_success": rate(sum(r.get("after_task_success") == "pass" for r in clarified), len(clarified)),
        "confirmation_persisted": rate(sum(r.get("confirmation_memory_saved", False) for r in clarified), len(clarified)),
        "followup_memory_consistency": rate(sum(r.get("followup_fidelity") == "pass" for r in clarified), len(clarified)),
        "followup_uncertain_or_missing": sum(r.get("followup_fidelity") in {None, "uncertain"} for r in clarified),
        "unnecessary_ask_after_clarification": rate(sum(r.get("followup_action") == "ASK" for r in clarified), len(clarified)),
        "end_to_end_success_all_eligible": rate(sum(r.get("end_to_end_success", False) for r in eligible), len(eligible)),
        "end_to_end_uncertain_or_missing": sum(
            r.get("after_task_success") in {None, "uncertain"} or r.get("followup_task_success") in {None, "uncertain"} for r in eligible),
        "end_to_end_note": "Éxitos confirmados/total elegible como cota conservadora; uncertain/missing NO equivale a fallo del agente.",
        "retrieval_pair_recall": rate(sum(r["retrieval_pair_recall"] for r in pair), len(pair)),
        "extraction_negation_loss_alarm": rate(sum(x["negation_loss_alarm"] for x in extraction), len(extraction)),
        "extraction_fidelity_human": rate(0, 0),
        "completion_tokens": sum((t.get("usage") or {}).get("completion_tokens", 0) or 0 for t in tokens),
        "prompt_tokens": sum((t.get("usage") or {}).get("prompt_tokens", 0) or 0 for t in tokens),
        "reasoning_tokens_reported": sum(t.get("reasoning_tokens") or 0 for t in tokens),
        "reasoning_tokens_missing_calls": sum(t.get("reasoning_tokens") is None for t in tokens),
        "latency_seconds_total": sum(t.get("latency_seconds", 0) for t in tokens),
        "api_cost_by_component": {client: {
            "calls": len([t for t in tokens if t.get("client") == client]),
            "completion_tokens": sum((t.get("usage") or {}).get("completion_tokens", 0) or 0 for t in tokens if t.get("client") == client),
            "latency_seconds": sum(t.get("latency_seconds", 0) for t in tokens if t.get("client") == client)}
            for client in sorted({t.get("client", "unknown") for t in tokens})},
        "finish_reasons": dict(collections.Counter(str(t.get("finish_reason")) for t in tokens)),
        "parse_errors": sum(e["kind"] in {"parse_error", "decision_schema_error"} for r in rows for e in r.get("technical_errors", [])),
        "cost_money": None, "cost_note": "No se proporcionaron tarifas; no se inventa coste monetario.",
    }


def summarize(rows, seed=42):
    names = sorted({r["condition"] for r in rows})
    groups = {name: [r for r in rows if r["condition"] == name] for name in names}
    output = {"provisional_automatic": True, "not_training_evidence": True,
              "conditions": {name: summarize_group(group, seed) for name, group in groups.items()},
              "by_kind": {name: {kind: summarize_group([r for r in group if r["kind"] == kind], seed)
                                      for kind in sorted({r["kind"] for r in group})} for name, group in groups.items()},
              "paired": {}}
    output["by_length"] = {name: {label: summarize_group([r for r in group if (r["history_messages"] <= 18) == short], seed)
        for label, short in [("at_most_18_messages", True), ("more_than_18_messages", False)]}
        for name, group in groups.items()}
    output["by_noise"] = {name: {str(noise): summarize_group([r for r in group if r["future_noise"] == noise], seed)
        for noise in sorted({r["future_noise"] for r in group})} for name, group in groups.items()}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a = {r["case_id"]: r for r in groups[left] if "action_correct" in r}
            b = {r["case_id"]: r for r in groups[right] if "action_correct" in r}
            diffs = [{"person_id": a[k]["person_id"], "delta": int(b[k]["action_correct"]) - int(a[k]["action_correct"])} for k in sorted(a.keys() & b.keys())]
            output["paired"][left + "__vs__" + right] = {
                "n": len(diffs), "right_minus_left": statistics.mean(r["delta"] for r in diffs) if diffs else None,
                "cluster_bootstrap_ci": grouped_bootstrap(diffs, "delta", seed)}
    return output
