"""Importa revisión humana explícita en un dataset NUEVO; conserva las propuestas."""
import argparse
import copy
import csv
from pathlib import Path
from data_generation.build import write_dataset
from data_generation.schema import validate_dataset
from evaluation.runner import load_jsonl

KINDS = {"unresolved_conflict", "historical_conflict", "correction", "scope_mismatch", "source_conflict", "unknown", "no_conflict"}


def apply_reviews(public, gold, reviews):
    result = copy.deepcopy(gold)
    by_id = {g["case_id"]: g for g in result}
    seen = set()
    for row in reviews:
        if row.get("validated", "").strip().lower() not in {"true", "1", "si", "sí"}:
            continue
        case_id = row["case_id"]
        if case_id not in by_id or case_id in seen:
            raise ValueError("Caso inexistente o anotación duplicada")
        seen.add(case_id)
        if not row["reviewer"].strip() or not row["notes"].strip():
            raise ValueError("Revisión validada exige revisor y justificación")
        if row["historical_conflict"].lower() not in {"true", "false"} or row["unresolved_conflict"].lower() not in {"true", "false"}:
            raise ValueError("Booleanos de revisión deben ser true/false")
        actions = sorted(set(x.strip() for x in row["acceptable_actions"].split("|")))
        if not actions or not set(actions) <= {"ASK", "ANSWER", "CONDITIONAL"} or row["kind"] not in KINDS:
            raise ValueError("Etiquetas de revisión inválidas")
        g = by_id[case_id]
        g["initial_proposal"] = {k: g[k] for k in ("historical_conflict", "unresolved_conflict", "kind", "acceptable_actions", "rationale")}
        g.update(historical_conflict=row["historical_conflict"].lower() == "true", unresolved_conflict=row["unresolved_conflict"].lower() == "true",
                 kind=row["kind"], acceptable_actions=actions, rationale=row["notes"],
                 annotation={"validated": True, "status": "human_reviewed", "reviewer": row["reviewer"].strip()})
    validate_dataset(public, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    public = load_jsonl(args.dataset / "casos_agent.jsonl")
    gold = load_jsonl(args.dataset / "private" / "casos_gold.jsonl")
    with args.review.open(encoding="utf8", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    result = apply_reviews(public, gold, reviews)
    print(write_dataset(args.output, public, result))


if __name__ == "__main__":
    main()
