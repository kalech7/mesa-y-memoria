"""SPEC-001 local evaluation. Declarative expectations never enter the engine.

Wilson intervals describe the finite synthetic benchmark, not human generalization.
The generator and scorer use no extraction/resolution method to compute gold.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from data_generation.food_scenarios import make_food_dataset, validate_food_dataset, GOLD_VERSION, GOLD_VERSION_REASON
from memory.store import Store
from simulator.resolution import FoodPreferenceEngine
from simulator.food_user import SyntheticFoodUser

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def rate(successes, total):
    result = {"numerator": successes, "denominator": total, "value": successes / total if total else None,
              "wilson_95": None}
    if total:
        z = 1.959963984540054
        p, d = successes / total, 1 + z * z / total
        center = (p + z * z / (2 * total)) / d
        margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / d
        result["wilson_95"] = [max(0., center - margin), min(1., center + margin)]
    return result


def score_output(case, expected, result, preferences, restrictions):
    """Compare structures to authored expectations, not generated natural language."""
    visible = {event["message_id"] for event in case["events"]}
    pref_origins = {p.preference_id: p.evidence_id for p in preferences}
    restriction_origins = {r.restriction_id: r.evidence_id for r in restrictions}
    retrieved = {pref_origins[i] for i in result.get("retrieved_preference_ids", []) if i in pref_origins}
    retrieved |= {restriction_origins[i] for i in result.get("retrieved_restriction_ids", []) if i in restriction_origins}
    relevant = set(expected["relevant_evidence_ids"])
    cited = set(result.get("evidence_ids", []))
    required = set(expected["expected_evidence_ids"])
    selected = set(result.get("selected_food_items", []))
    desired = set(expected["expected_food_items"])
    context = result.get("resolved_context", {})
    contextual = all(context.get(key) == value for key, value in expected["expected_context"].items())
    meal = context.get("meal_scope") == expected["expected_context"].get("meal_scope")
    action = result.get("selected_action") in expected["acceptable_actions"]
    conflict = result.get("detected_conflict_type") == expected["expected_conflict_type"]
    foods = selected == desired
    evidence = required <= cited and cited <= visible
    # This is fidelity to authored evidence/food sets, not semantic human judgment.
    fidelity = evidence and selected <= desired
    forbidden = selected & set(expected.get("forbidden_food_items", []))
    issues = []
    for name, correct in (("context", contextual), ("meal", meal), ("action", action),
                          ("foods", foods), ("evidence", evidence), ("conflict", result.get("detected_conflict_type") == expected["expected_conflict_type"])):
        if not correct:
            issues.append(name)
    if forbidden:
        issues.append("forbidden_food")
    return {"context_correct": contextual, "meal_correct": meal, "action_correct": action,
            "foods_correct": foods, "evidence_correct": evidence, "fidelity": fidelity,
            "decision_correct": contextual and action and foods and evidence and conflict and not forbidden,
            "conflict_correct": conflict,
            "retrieval_hits": len(retrieved & relevant), "retrieved_count": len(retrieved),
            "relevant_count": len(relevant), "retrieved_evidence_ids": sorted(retrieved),
            "forbidden_selections": sorted(forbidden),
            "restriction_violation": bool(selected & set(expected["restriction_food_items"])) if "restriction_food_items" in expected else
                bool(forbidden) if case["category"] == "S17" else None,
            "clarification_required": expected["requires_clarification"],
            "clarification_forbidden": "CLARIFY" not in expected["acceptable_actions"],
            "asked": result.get("selected_action") == "CLARIFY", "issues": issues}


def summarize(rows):
    valid = [r for r in rows if not r.get("technical_error")]
    metrics = {field: rate(sum(r["scores"][key] for r in valid), len(valid)) for field, key in (
        ("context_accuracy", "context_correct"),
        ("action_accuracy", "action_correct"), ("food_accuracy", "foods_correct"),
        ("structured_decision_accuracy", "decision_correct"), ("evidence_fidelity", "fidelity"),
        ("conflict_accuracy", "conflict_correct"))}
    meals = [r for r in valid if r["expected"]["expected_context"].get("meal_scope") not in (None, "ANY")]
    unspecified = [r for r in valid if r["expected"]["expected_context"].get("meal_scope") is None]
    metrics["meal_accuracy"] = rate(sum(r["scores"]["meal_correct"] for r in meals), len(meals))
    metrics["unknown_meal_preservation"] = rate(sum(r["scores"]["meal_correct"] for r in unspecified), len(unspecified))
    metrics["retrieval_precision_micro"] = rate(sum(r["scores"]["retrieval_hits"] for r in valid), sum(r["scores"]["retrieved_count"] for r in valid))
    metrics["retrieval_recall_micro"] = rate(sum(r["scores"]["retrieval_hits"] for r in valid), sum(r["scores"]["relevant_count"] for r in valid))
    required = [r for r in valid if r["scores"]["clarification_required"]]
    forbidden = [r for r in valid if r["scores"]["clarification_forbidden"]]
    restricted = [r for r in valid if r["scores"]["restriction_violation"] is not None]
    metrics["necessary_clarification_rate"] = rate(sum(r["scores"]["asked"] for r in required), len(required))
    metrics["unnecessary_clarification_rate"] = rate(sum(r["scores"]["asked"] for r in forbidden), len(forbidden))
    metrics["restriction_violation_rate"] = rate(sum(r["scores"]["restriction_violation"] for r in restricted), len(restricted))
    confusion = Counter((r["expected"]["expected_conflict_type"] or "NONE", r["result"].get("detected_conflict_type") or "NONE") for r in valid)
    metrics["conflict_confusion"] = {f"{left} -> {right}": count for (left, right), count in sorted(confusion.items())}
    positives = [r for r in valid if r["expected"]["expected_conflict_type"]]
    predicted = [r for r in valid if r["result"].get("detected_conflict_type")]
    tp = sum(r["scores"]["conflict_correct"] for r in predicted if r["expected"]["expected_conflict_type"])
    metrics["conflict_precision"] = rate(tp, len(predicted))
    metrics["conflict_recall"] = rate(tp, len(positives))
    followups = [r["clarification_followup"] for r in valid if "clarification_followup" in r]
    metrics["clarification_followup_success"] = rate(sum(r["correct"] for r in followups), len(followups))
    metrics["attempted"] = len(rows)
    metrics["evaluable"] = len(valid)
    metrics["technical_errors"] = len(rows) - len(valid)
    metrics["failures"] = sum(bool(r["scores"]["issues"]) for r in valid)
    latencies = sorted(r["result"]["execution_time_ms"] for r in valid)
    metrics["latency_ms"] = {"n": len(latencies), "mean": statistics.mean(latencies) if latencies else None,
                             "p50": statistics.median(latencies) if latencies else None,
                             "p95": latencies[min(len(latencies)-1, math.ceil(.95 * len(latencies))-1)] if latencies else None,
                             "scope": "pure resolution/context filtering; excludes SQLite reads, extraction, ingest and scoring"}
    return metrics


def grouped(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: summarize(value) for key, value in sorted(groups.items())}


def _json(path, value, private=False):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    if private:
        path.chmod(0o600)


def _jsonl(path, values, private=False):
    path.write_text("".join(json.dumps(v, ensure_ascii=False, allow_nan=False) + "\n" for v in values), encoding="utf8")
    if private:
        path.chmod(0o600)


def evaluate_case(case):
    """No annotations/category/split/family/private data cross this boundary."""
    with_store = Store(":memory:", case["user_id"])
    try:
        engine = FoodPreferenceEngine(with_store)
        engine.ingest(case["events"])
        result = engine.decide(case["scenario_id"], case["user_id"], case["query"], case["reference_time"])
        return result, with_store.preferences(), with_store.restrictions()
    finally:
        with_store.close()


def evaluate_clarification(case, private, expected):
    """User-private replies reach the engine only after a targeted question."""
    store = Store(":memory:", case["user_id"])
    try:
        engine = FoodPreferenceEngine(store)
        engine.ingest(case["events"])
        first = engine.decide(case["scenario_id"], case["user_id"], case["query"], case["reference_time"])
        if first["selected_action"] != "CLARIFY":
            return {"correct": False, "reason": "No clarification was requested"}
        reply = SyntheticFoodUser(private).respond(first["clarification_question"])
        event = {"message_id": case["scenario_id"] + "-clarification", "session_id": "clarification",
                 "at": case["reference_time"], "role": "user", "channel": "chat", "source": "clarification", "text": reply["text"]}
        engine.ingest([event])
        visible_query = case["query"] + " " + reply["text"]
        result = engine.decide(case["scenario_id"], case["user_id"], visible_query, case["reference_time"])
        return {"correct": (result["selected_action"] in expected["acceptable_actions"]
                            and result["selected_food_items"] == expected["expected_food_items"]
                            and set(expected["expected_evidence_ids"]) <= set(result["evidence_ids"])),
                "question": first["clarification_question"], "reply": reply, "result": result}
    finally:
        store.close()


def run_evaluation(output, seeds=(42, 43, 44), instances=240, run_tests=True, challenge_path=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    test_info = {"executed": False, "returncode": None, "tests": None, "successful": None}
    if run_tests:
        import re
        completed = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
                                   cwd=ROOT, capture_output=True, text=True)
        log = completed.stdout + completed.stderr
        (output / "unittest.log").write_text(log, encoding="utf8")
        count = re.search(r"Ran (\d+) tests? in", log)
        test_info = {"executed": True, "returncode": completed.returncode,
                     "tests": int(count.group(1)) if count else 0,
                     "successful": completed.returncode == 0 and bool(count) and int(count.group(1)) > 0}
    rows, dataset_manifest, projections = [], [], []
    for seed in seeds:
        public, gold, private = make_food_dataset(seed, instances)
        dataset_manifest.append({"seed": seed, **validate_food_dataset(public, gold, private),
                                 "public_hash": digest(public), "gold_hash": digest(gold), "private_hash": digest(private)})
        path = output / f"seed_{seed}"
        path.mkdir()
        (path / "private").mkdir(mode=0o700)
        _jsonl(path / "scenarios.jsonl", public)
        _jsonl(path / "private" / "expectations.jsonl", gold, True)
        _jsonl(path / "private" / "users.jsonl", private, True)
        for case, expected, hidden in zip(public, gold, private):
            row = {"scenario_id": case["scenario_id"], "seed": seed, "user_id": case["user_id"],
                   "category": case["category"], "split": case["split"], "family": case["family"],
                   "context_group": f"{expected['expected_context'].get('day_scope')}/{expected['expected_context'].get('meal_scope')}",
                   "conflict_group": expected["expected_conflict_type"] or "NONE", "input": case, "expected": expected}
            try:
                result, preferences, restrictions = evaluate_case(case)
                row.update(result=result, preferences=[asdict(p) for p in preferences], restrictions=[asdict(r) for r in restrictions],
                           scores=score_output(case, expected, result, preferences, restrictions))
                if "clarification_followup" in expected:
                    row["clarification_followup"] = evaluate_clarification(case, hidden, expected["clarification_followup"])
            except Exception as exc:
                # No silent exclusion or conversion into ABSTAIN.
                row["technical_error"] = {"type": type(exc).__name__, "message": str(exc)}
            rows.append(row)
        projections.extend(public)
    challenge_rows, instrument_review = [], {}
    if challenge_path:
        challenge_data = json.loads(Path(challenge_path).read_text(encoding="utf8"))
        review_path = Path(challenge_path).with_name("challenge_review.json")
        if review_path.exists():
            instrument_review = json.loads(review_path.read_text(encoding="utf8"))
            if instrument_review.get("original_fixture_sha256") != hashlib.sha256(Path(challenge_path).read_bytes()).hexdigest():
                raise ValueError("La revisión no corresponde al hash del desafío")
        challenge_items = challenge_data if isinstance(challenge_data, list) else challenge_data["scenarios"]
        for item in challenge_items:
            case, expected = item["case"], item["expected"]
            # Additional typed scoring metadata is an independent review of the
            # instrument, never a changed expected action/food after seeing output.
            expected = dict(expected)
            if case["scenario_id"] in instrument_review.get("restriction_scenarios", []):
                expected["restriction_food_items"] = list(expected["forbidden_food_items"])
            row = {"scenario_id": case["scenario_id"], "seed": "challenge", "category": case["category"],
                   "split": "challenge", "family": case["family"], "input": case, "expected": expected,
                   "context_group": f"{expected['expected_context'].get('day_scope')}/{expected['expected_context'].get('meal_scope')}",
                   "conflict_group": expected["expected_conflict_type"] or "NONE"}
            try:
                result, prefs, restrictions = evaluate_case(case)
                row.update(result=result, preferences=[asdict(p) for p in prefs], restrictions=[asdict(r) for r in restrictions],
                           scores=score_output(case, expected, result, prefs, restrictions))
            except Exception as exc:
                row["technical_error"] = {"type": type(exc).__name__, "message": str(exc)}
            challenge_rows.append(row)
        _json(output / "challenge_fixture.json", challenge_data)
        _json(output / "challenge_instrument_review.json", instrument_review)
        _jsonl(output / "challenge_results.jsonl", challenge_rows)
    _jsonl(output / "results.jsonl", rows)
    failures = []
    for row in rows + challenge_rows:
        if row.get("technical_error") or row["scores"]["issues"]:
            failures.append({"scenario_id": row["scenario_id"], "seed": row["seed"], "category": row["category"],
                             "input": row["input"], "expected": row["expected"], "obtained": row.get("result"),
                             "issues": row.get("scores", {}).get("issues", []), "technical_error": row.get("technical_error"),
                             "responsible_component": "extractor/context/resolver" if not row.get("technical_error") else "pipeline",
                             "possible_cause": "Compare extracted records and applicability with authored expectations; no automatic relabeling.",
                             "recommended_correction": "Inspect evidence and scope; add a justified regression, preserving original failed run."})
    _jsonl(output / "failures.jsonl", failures)
    summary = summarize(rows)
    by_split, by_seed = grouped(rows, "split"), grouped(rows, "seed")
    thresholds = {"context_accuracy": .95, "meal_accuracy": .95, "structured_decision_accuracy": .90, "evidence_fidelity": .95}
    def passes(metrics):
        return (metrics["technical_errors"] == 0
                and metrics["evaluable"] > 0
                and all(metrics[key]["denominator"] == 0 or metrics[key]["value"] >= value for key, value in thresholds.items())
                and (metrics["restriction_violation_rate"]["denominator"] == 0 or metrics["restriction_violation_rate"]["value"] == 0)
                and metrics["conflict_accuracy"]["value"] >= .90)
    # Canonical deterministic examples are one dev case/category/seed, additional
    # combinatorial/metamorphic contracts live in unittest; no hidden test bypass.
    first_per_category = {}
    for row in rows:
        first_per_category.setdefault((row["seed"], row["category"]), row)
    canonical = list(first_per_category.values())
    gate_a = None if not run_tests else bool(test_info["successful"] and len({r["category"] for r in canonical}) == 20
        and all(not r.get("technical_error") and not r["scores"]["issues"] for r in canonical)
        and summary["restriction_violation_rate"]["value"] == 0)
    gate_b = (passes(summary) and all(passes(m) for m in by_split.values())
              and all(passes(m) for m in grouped(rows, "category").values())
              and all(passes(m) for m in grouped(rows, "context_group").values()))
    # A retained wording family is held separately, but still finite known grammar.
    gate_c_controlled = (all(passes(m) for m in by_seed.values()) and "test" in by_split and passes(by_split["test"])
              and len(seeds) >= 3 and len({r["family"] for r in rows}) >= 3)
    challenge_summary = summarize(challenge_rows) if challenge_rows else None
    gate_c = gate_c_controlled and (passes(challenge_summary) if challenge_summary else False)
    verdict = ("FUNCIONAL Y VALIDADO EN PRUEBAS SINTÉTICAS" if gate_a and gate_b and gate_c and not failures else
               "FUNCIONAL CON LIMITACIONES" if summary["evaluable"] and summary["technical_errors"] == 0 else
               "PARCIALMENTE FUNCIONAL" if summary["evaluable"] else "NO FUNCIONAL")
    if gate_a is None and not failures:
        verdict = "EVIDENCIA INSUFICIENTE"
    sources = sorted(p for folder in ("simulator", "memory", "data_generation", "evaluation", "tests", "agents", "scripts") for p in (ROOT / folder).glob("*.py"))
    # Only the imported topic catalogue is needed from original/. The legacy
    # runner may contain private infrastructure configuration and is not copied.
    sources.append(ROOT / "original" / "scenarios.py")
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    import shutil
    for path in sources:
        target = output / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    calendar_inconsistent = set(instrument_review.get("calendar_inconsistent_scenarios", []))
    sensitivity = summarize([r for r in challenge_rows if r["scenario_id"] not in calendar_inconsistent]) if challenge_rows else None
    report = {"spec": "SPEC-001", "verdict": verdict, "gates": {"A": gate_a, "B": gate_b, "C": gate_c},
              "tests": test_info, "global": summary, "by_category": grouped(rows, "category"), "by_context": grouped(rows, "context_group"),
              "by_conflict": grouped(rows, "conflict_group"), "by_split": by_split, "by_seed": by_seed,
              "controlled_robustness": gate_c_controlled, "challenge": challenge_summary,
              "challenge_by_category": grouped(challenge_rows, "category") if challenge_rows else {},
              "challenge_hash": digest(challenge_data) if challenge_path else None,
              "challenge_file_sha256": hashlib.sha256(Path(challenge_path).read_bytes()).hexdigest() if challenge_path else None,
              "challenge_instrument_review": {
                  "posthoc": True, "human_validated": False,
                  "review": instrument_review,
                  "sensitivity_excluding_calendar_inconsistency": sensitivity},
              "gold_version": GOLD_VERSION, "gold_version_reason": GOLD_VERSION_REASON,
              "eight_combination_coverage": dict(sorted(Counter(r["context_group"] for r in rows if r["category"] == "S08").items())),
              "thresholds": thresholds, "datasets": dataset_manifest, "python": platform.python_version(),
              "source_sha256": hashes, "elapsed_seconds": time.perf_counter() - started,
              "network_calls": 0, "training_performed": False,
              "limitations": ["Controlled finite Spanish grammar/catalog; not general NLP or human validation.",
                              "Wording families/users disjoint, but categories/grammar shared and source inspectable; not blind external generalization.",
                              "Wilson intervals conditional on synthetic instances; shared templates invalidate population inference.",
                              "Evidence fidelity is structural and expectation-based, not human semantic quality.",
                              "Retrieval is exhaustive over stored food records; this does not validate BGE or LLM extraction."]}
    _json(output / "metrics.json", report)
    lines = ["# Validación sintética SPEC-001", "", f"Dictamen: **{verdict}**.", "",
             f"Gates: A={gate_a}; B={gate_b}; C={gate_c}.",
             f"Instancias controladas: {len(rows)}; desafío adicional: {len(challenge_rows)}. Fallos totales: {len(failures)}. Errores técnicos controlados: {summary['technical_errors']}.",
             f"Pruebas: {test_info}. Tiempo total: {report['elapsed_seconds']:.3f} s.", "",
             "## Métricas globales", "", "| Métrica | Numerador/denominador | Valor | IC Wilson 95% |", "|---|---:|---:|---|"]
    for key, metric in summary.items():
        if isinstance(metric, dict) and "denominator" in metric:
            lines.append(f"| {key} | {metric['numerator']}/{metric['denominator']} | {metric['value']} | {metric['wilson_95']} |")
    lines += ["", "## Resultados por categoría", "", "| Categoría | N | Decisión correcta | Fallos |", "|---|---:|---:|---:|"]
    for category, metrics in report["by_category"].items():
        m = metrics["structured_decision_accuracy"]
        lines.append(f"| {category} | {metrics['attempted']} | {m['numerator']}/{m['denominator']} | {metrics['failures']} |")
    for title, groups in (("contexto", report["by_context"]), ("conflicto", report["by_conflict"]), ("partición", report["by_split"])):
        lines += ["", f"## Resultados por {title}", "", "| Grupo | N | Decisión correcta | Fallos |", "|---|---:|---:|---:|"]
        for name, metrics in groups.items():
            m = metrics["structured_decision_accuracy"]
            lines.append(f"| {name} | {metrics['attempted']} | {m['numerator']}/{m['denominator']} | {metrics['failures']} |")
    lines += ["", "Los detalles por contexto, conflicto, semilla y partición están en metrics.json.",
              "Cada fallo conserva entrada, esperado, obtenido y diagnóstico inicial en failures.jsonl.", "", "## Límites", ""]
    if challenge_summary:
        lines += [f"Desafío independiente: {challenge_summary['structured_decision_accuracy']}; fallos: {challenge_summary['failures']}.", ""]
    lines += [f"- {limit}" for limit in report["limitations"]]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--instances", type=int, default=320,
                        help="320 per seed gives 16/category and two repetitions of each S08 combination")
    parser.add_argument("--skip-tests", action="store_true", help="Gate A stays non-evaluable")
    parser.add_argument("--challenge", type=Path, default=ROOT / "tests/fixtures/food_preferences/challenge.json")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or args.instances < 200:
        parser.error("Unique seeds and >=200 instances per seed required")
    report = run_evaluation(args.output, args.seeds, args.instances, not args.skip_tests,
                            args.challenge if args.challenge.exists() else None)
    print(json.dumps({"verdict": report["verdict"], "gates": report["gates"], "cases": report["global"]["attempted"],
                      "failures": report["global"]["failures"], "technical_errors": report["global"]["technical_errors"]}, ensure_ascii=False))
    return 0 if all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
