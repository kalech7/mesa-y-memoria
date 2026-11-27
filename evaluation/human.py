"""Agrega anotaciones humanas externas; conserva uncertain y desacuerdos."""
import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

DOMAINS = {"question_relevance_human": {"relevant", "irrelevant", "uncertain"},
           "information_gain_human": {"high", "low", "none", "uncertain"},
           "question_useful_human": {"true", "false", "uncertain"}}
for field in ("answer_grounded_human", "task_success_human", "after_grounded_human", "after_task_success_human", "followup_consistency_human", "followup_task_success_human"):
    DOMAINS[field] = {"pass", "fail", "uncertain"}


def aggregate(reviews, key):
    lookup = {r["review_id"]: r for r in key}
    grouped, seen, ratings = defaultdict(list), set(), defaultdict(dict)
    for row in reviews:
        if not row.get("reviewer", "").strip():
            continue
        identity = (row["review_id"], row["reviewer"].strip())
        if identity in seen or row["review_id"] not in lookup:
            raise ValueError("Revisión duplicada o ID desconocido")
        seen.add(identity)
        for field, allowed in DOMAINS.items():
            value = row.get(field, "").strip().lower()
            if value and value not in allowed:
                raise ValueError(f"Valor de anotación inválido: {field}")
            if value:
                ratings[(row["review_id"], field)][identity[1]] = value
        grouped[lookup[row["review_id"]]["condition"]].append(row)
    result = {"human_rating_rows": len(seen), "conditions": {}, "agreement": {}}
    for condition, rows in grouped.items():
        result["conditions"][condition] = {field: dict(Counter(r.get(field, "").strip().lower() or "missing" for r in rows)) for field in DOMAINS}
    for field in DOMAINS:
        pairs = [pair for (review_id, name), reviewers in ratings.items() if name == field
                 for pair in itertools.combinations(sorted(reviewers.items()), 2)]
        agreement = sum(left[1] == right[1] for left, right in pairs)
        result["agreement"][field] = {"paired_ratings": len(pairs), "agreements": agreement,
            "disagreements": len(pairs) - agreement, "agreement_rate": agreement / len(pairs) if pairs else None,
            "note": "Acuerdo descriptivo; conservar adjudicación y no tratar ratings como historias independientes."}
    result["pending"] = not bool(seen)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, action="append", required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reviews = []
    for path in args.review:
        with path.open(encoding="utf8", newline="") as stream:
            reviews.extend(csv.DictReader(stream))
    result = aggregate(reviews, json.loads(args.key.read_text()))
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(f"Ratings humanos: {result['human_rating_rows']}")


if __name__ == "__main__":
    main()
