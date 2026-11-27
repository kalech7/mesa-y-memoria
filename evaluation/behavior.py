"""Describe conducta inicial observada; no evalúa corrección ni utiliza gold."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def describe(rows):
    grouped = defaultdict(list)
    observations = []
    for row in rows:
        action = row.get("initial_action")
        observation = {
            "case_id": row["case_id"], "condition": row["condition"],
            "action_declared": action,
            "historical_conflict_declared": row.get("initial_historical_conflict"),
            "unresolved_conflict_declared": row.get("initial_unresolved_conflict"),
            "question": row.get("initial_question", ""),
            "answer": row.get("initial_answer", ""),
            "explanation": row.get("initial_reason", ""),
            "technical_errors": row.get("technical_errors", []),
        }
        observations.append(observation)
        grouped[row["condition"]].append(observation)
    summary = {"purpose": "descriptive_initial_behavior_only", "new_model_calls": 0,
               "correctness_evaluated": False, "human_semantic_action_review": "pending",
               "conditions": {}}
    for condition, group in grouped.items():
        valid = [r for r in group if r["action_declared"] in {"ASK", "ANSWER", "CONDITIONAL"}]
        counts = Counter(r["action_declared"] for r in valid)
        summary["conditions"][condition] = {
            "records": len(group), "initial_evaluable": len(valid),
            "initial_missing": len(group) - len(valid),
            "records_with_technical_errors": sum(bool(r["technical_errors"]) for r in group),
            "actions_declared": {a: counts[a] for a in ("ASK", "ANSWER", "CONDITIONAL")},
            "ASK_rate": counts["ASK"] / len(valid) if valid else None,
            "note": "Frecuencia de conducta, no accuracy ni tasa de errores. Revisar contenido además de etiqueta JSON.",
        }
    return observations, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in (args.run / "per_case.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
    observations, summary = describe(rows)
    summary["source_run"] = str(args.run)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "behavior.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    with (args.output / "observations.jsonl").open("w", encoding="utf8") as stream:
        for row in observations:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output / "observations.csv").open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(observations[0]) if observations else ["case_id"])
        writer.writeheader()
        writer.writerows(observations)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
