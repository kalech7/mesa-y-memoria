"""Reconstruye tablas exploratorias sin cambiar etiquetas ni resultados originales."""
import argparse
import copy
import csv
import json
from pathlib import Path
from data_generation.schema import digest
from evaluation.metrics import summarize
from evaluation.runner import load_jsonl


def snapshot_hash(path):
    sources = sorted(p for folder in ("agents", "memory", "simulator", "data_generation", "evaluation", "tests")
                     for p in (path / folder).glob("*.py"))
    return digest({str(p.relative_to(path)): digest(p.read_text()) for p in sources})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    all_rows, audit, conditions, public_hashes = [], [], {}, set()
    for run in args.run:
        rows = load_jsonl(run / "per_case.jsonl")
        manifest = json.loads((run / "manifest.json").read_text())
        public_hashes.add(manifest["configuration"]["public_hash"])
        if len(public_hashes) > 1:
            raise ValueError("No combinar datasets públicos diferentes")
        for condition in {r["condition"] for r in rows}:
            if condition in conditions:
                raise ValueError("No combinar una condición de ejecuciones distintas")
            conditions[condition] = str(run)
        for row in rows:
            row["run_directory"] = str(run)
        all_rows.extend(rows)
        registered = sum(len(r.get("api_calls", [])) for r in rows)
        trace_paths = [run / "agent_trace.jsonl", run / "private" / "judge_trace.jsonl"]
        traces = [t for path in trace_paths if path.exists() for t in load_jsonl(path)]
        audit.append({"run": str(run), "completed_records": len(rows),
            "rows_with_technical_error": sum(bool(r.get("technical_errors")) for r in rows),
            "calls_in_completed_rows": registered, "trace_calls_including_incomplete_cases": len(traces),
            "completion_tokens_all_traces": sum((t.get("usage") or {}).get("completion_tokens", 0) or 0 for t in traces),
            "latency_seconds_all_traces": sum(t.get("latency_seconds", 0) for t in traces),
            "snapshot_hash_verified": snapshot_hash(run / "source_snapshot") == manifest["configuration"]["source_hash"],
            "public_hash": manifest["configuration"]["public_hash"], "source_hash": manifest["configuration"]["source_hash"]})
    metrics = summarize(all_rows)
    # Sensibilidad post hoc: el revisor consideró útil ASK en un control de nota antigua.
    # Ampliar ASK como aceptable es una hipótesis de sensibilidad, NO un gold validado.
    sensitivity = copy.deepcopy(all_rows)
    for row in sensitivity:
        if row["case_id"] == "24a356e4fa1942a2e5e4a534":
            row["acceptable_actions"] = sorted(set(row["acceptable_actions"]) | {"ASK"})
            if "initial_action" in row:
                row["action_correct"] = row["initial_action"] in row["acceptable_actions"]
    results = {"not_scientific_training_evidence": True, "audit": audit,
        "original_provisional_gold": metrics,
        "sensitivity_allow_staleness_confirmation": summarize(sensitivity),
        "sensitivity_caveat": "Revisión parcial posterior de utilidad; no valida todas las acciones ni toda la taxonomía."}
    (args.output / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf8")
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf8") as stream:
        fields = ["condition", "attempted", "evaluable", "action_correct", "action_denominator", "technical_errors",
                  "ASK_recall_numerator", "ASK_recall_denominator", "ASK_FPR_numerator", "ASK_FPR_denominator"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for condition, m in metrics["conditions"].items():
            writer.writerow({"condition": condition, "attempted": m["n_attempted"], "evaluable": m["initial_evaluable"],
                "action_correct": m["action_accuracy"]["numerator"], "action_denominator": m["action_accuracy"]["denominator"],
                "technical_errors": m["technical_errors"], "ASK_recall_numerator": m["ASK_recall_required"]["numerator"],
                "ASK_recall_denominator": m["ASK_recall_required"]["denominator"], "ASK_FPR_numerator": m["ASK_false_positive_rate"]["numerator"],
                "ASK_FPR_denominator": m["ASK_false_positive_rate"]["denominator"]})
    with (args.output / "per_case.jsonl").open("w", encoding="utf8") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(all_rows), "snapshot_hashes_verified": all(a["snapshot_hash_verified"] for a in audit)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
