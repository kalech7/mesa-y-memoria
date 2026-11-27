"""Compare the independent, blinded reviews with the author's sealed labels."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path

from research.src.artifacts import canonical, file_hash, write_json
from research.tesis_extended_heldout import load_gold


FIELDS = ("response", "new_write", "retrieves")
RECORD_FIELDS = {"attribute", "value", "scope", "valid_from", "valid_through_inclusive", "context", "session_id"}


def normalized(field, value):
    if field == "response":
        answer = value["value"]
        if isinstance(answer, list):
            answer = sorted(item.strip().casefold() for item in answer)
        elif isinstance(answer, str):
            answer = answer.strip().casefold()
        return (value["mode"], answer)
    return sorted(canonical({
        name: raw.strip().casefold() if isinstance(raw, str) else raw
        for name, raw in item.items() if name in RECORD_FIELDS
    }) for item in value)


def read_review(path, expected):
    document = json.loads(path.read_text(encoding="utf-8"))
    name = document["reviewer_id"].strip()
    data = {}
    for case in document["cases"]:
        for row in case["visits"]:
            key = (case["id"], row["visit"])
            if key in data:
                raise ValueError(f"Duplicate review visit {key}")
            mode = row["response"]["mode"]
            answer = row["response"]["value"]
            if ((mode == "ANSWER" and isinstance(answer, str) and answer.strip())
                    or (mode == "DEFER" and answer is None)
                    or (mode == "CONDITIONALIZE" and isinstance(answer, list) and len(answer) > 1
                        and all(isinstance(item, str) and item.strip() for item in answer))):
                pass
            else:
                raise ValueError(f"Invalid reviewed response at {key}")
            for field in ("new_write", "retrieves"):
                for item in row[field]:
                    if not RECORD_FIELDS.issubset(item):
                        raise ValueError(f"Incomplete {field} at {key}")
                    if item["scope"] not in {"ongoing", "session", "interval", "contextual"}:
                        raise ValueError(f"Invalid scope at {key}")
                    for date_field in ("valid_from", "valid_through_inclusive"):
                        value = item[date_field]
                        if value is not None and date.fromisoformat(value).isoformat() != value:
                            raise ValueError(f"Non-ISO date at {key}")
            data[key] = row
    if not name or set(data) != expected:
        raise ValueError(f"Incomplete blind review: {path}")
    return name, data


def compare(folder, review_a, review_b):
    authored = {(case["id"], row["visit"]): row for case in load_gold(folder) for row in case["visits"]}
    first, a = read_review(review_a, set(authored))
    second, b = read_review(review_b, set(authored))
    if first == second:
        raise ValueError("Two distinct blinded reviewers are required")
    agreement, differences = Counter(), []
    for key, author in authored.items():
        for field in FIELDS:
            values = {label: row[field] for label, row in (("author", author), ("a", a[key]), ("b", b[key]))}
            gold, left, right = (normalized(field, values[label]) for label in ("author", "a", "b"))
            agreement[f"reviewers_agree_{field}"] += left == right
            if gold != left or gold != right:
                differences.append({
                    "root_id": key[0], "visit": key[1], "field": field, "values": values,
                    "notes": {"a": a[key].get("rationale"), "b": b[key].get("rationale")},
                })
    return {
        "review_method": "two_independent_ai_reviews_of_common_public_packet",
        "reviewers": [first, second], "human_reviews": 0, "visits": len(authored),
        "manifest_sha256": file_hash(folder / "MANIFEST.json"),
        "files_sha256": [file_hash(review_a), file_hash(review_b)],
        "exact_agreement_after_case_normalization": dict(agreement),
        "differences_to_adjudicate": differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--a", required=True, type=Path)
    parser.add_argument("--b", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_json(args.output, compare(args.folder, args.a, args.b))


if __name__ == "__main__":
    main()
