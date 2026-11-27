"""Score the two independent AI readings of the frozen 100-story packets."""
import argparse
from collections import Counter
import json
from pathlib import Path

from research.review_packets import BANK, verify_materials
from research.src.artifacts import file_hash
from research.tesis_bank_review import _read_decisions


FIELDS = ("information_sufficient", "need_future", "decisive_slot", "admissible_answers", "writable_scope")


def agreement(pairs):
    """Observed agreement and unweighted Cohen's kappa for categorical judgments."""
    n = len(pairs)
    yes = sum(a == b for a, b in pairs)
    left, right = (Counter(row[i] for row in pairs) for i in (0, 1))
    chance_numerator = sum(count * right[label] for label, count in left.items())
    denominator = n * n - chance_numerator
    return {"agree": yes, "n": n, "kappa": (yes * n - chance_numerator) / denominator if denominator else None}


def category(value):
    return tuple(sorted(value)) if isinstance(value, list) else value


def score(folder):
    manifest = verify_materials(BANK)
    keys = json.loads((BANK / "review/private_key.json").read_bytes())
    reviewers, hashes, ids = [], {}, []
    for side in ("A", "B"):
        before, after = folder / f"{side}_antes.json", folder / f"{side}_final.json"
        rows = json.loads(before.read_bytes())
        initial = {row["audit_id"]: row for row in rows}
        final, reviewer = _read_decisions(after, keys)
        if len(rows) != len(initial) or set(initial) != set(keys) or set(final) != set(keys):
            raise ValueError(f"Cobertura incompleta o casos duplicados para {side}")
        for case, row in initial.items():
            if row["both_plausible"] is not None or not row["rationale"].strip() or row["reviewer_id"] != reviewer:
                raise ValueError(f"Primera lectura inválida en {case}")
            for field in FIELDS:
                value = row[field]
                if field == "admissible_answers":
                    value = [item.strip().casefold() for item in value]
                if category(value) != category(final[case][field]):
                    raise ValueError(f"La segunda lectura cambió {case}/{field}")
        reviewers.append(final)
        ids.append(reviewer)
        hashes.update({path.name: file_hash(path) for path in (before, after)})
    if ids[0] == ids[1]:
        raise ValueError("Los dos revisores deben tener identificadores distintos")
    a, b = reviewers
    metrics, differences = {}, {}
    for field in (*FIELDS, "both_plausible"):
        cases = [case for case in keys if field != "both_plausible" or keys[case]["has_pair"]]
        metrics[field] = agreement([(category(a[c][field]), category(b[c][field])) for c in cases])
        for case in cases:
            if category(a[case][field]) != category(b[case][field]):
                differences.setdefault(case, []).append(field)
    core = ("information_sufficient", "decisive_slot", "need_future")
    joint = sum(all(a[c][f] == b[c][f] for f in core) for c in keys)
    reference = {reviewer: {field: sum(category(rows[c][field]) == category(keys[c]["literal"][field])
                                       for c in keys) for field in FIELDS} for reviewer, rows in zip(ids, reviewers)}
    return {"review_type": "AI", "human_decisions_supplied": 0, "bank_manifest_sha256": file_hash(BANK / "manifest.json"),
            "source_sha256": hashes, "reviewers": ids, "cases": manifest["review"]["roots"],
            "initial_joint_agreement": {"agree": joint, "n": len(keys), "fields": list(core)},
            "agreement": metrics, "differences": differences, "matches_synthetic_key_out_of_100": reference}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    print(json.dumps(score(parser.parse_args().folder), ensure_ascii=False, indent=2))
