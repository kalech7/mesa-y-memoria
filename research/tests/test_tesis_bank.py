from collections import Counter
from copy import deepcopy
import json

import pytest

from research import tesis_bank
from research.src.artifacts import file_hash, write_json
from research.src.data.semantic_partition_v2 import SemanticSplitLeakage
from research.tesis_bank_review import prepare_review, reconcile_review


def test_counterfactuals_and_split_deduplication():
    splits = tesis_bank.build_splits()
    assert splits == tesis_bank.build_splits()
    report = tesis_bank.audit_isolation(splits)
    assert report["templates_and_recipe_ids_disjoint"]
    assert report["paired_public_prefixes"]["evaluation:F4"] == 10
    for split, rows in splits.items():
        roots = {row["root_id"]: row["scenario_family"] for row in rows}
        assert set(Counter(roots.values()).values()) == {tesis_bank.DEFAULT_SIZES[split] // 5}
    contaminated = deepcopy(splits)
    duplicated = deepcopy(splits["train"][0])
    duplicated["root_id"] = "different-id-same-observation"
    contaminated["evaluation"].append(duplicated)
    with pytest.raises(SemanticSplitLeakage):
        tesis_bank.audit_isolation(contaminated)


def test_blinded_material_and_disagreement_reconciliation(tmp_path):
    splits = tesis_bank.build_splits()
    packet_root = tmp_path / "review"
    info = prepare_review([*splits["train"], *splits["validation"]], packet_root, seed=13)
    assert info["roots"] == 100 and info["human_decisions_supplied"] == 0
    manifest = {"review": info}
    write_json(tmp_path / "manifest.json", manifest)
    cases = [json.loads((packet_root / f"reviewer_{side}/01_casos.json").read_text())["cases"]
             for side in ("a", "b")]
    assert [row["audit_id"] for row in cases[0]] != [row["audit_id"] for row in cases[1]]
    assert all(set(row) == {"audit_id", "memory", "conversation", "query", "reuse_regime"} for row in cases[0])
    blank_path = packet_root / "reviewer_a/03_respuestas_en_blanco.json"
    with pytest.raises(ValueError, match="identificador de la persona"):
        reconcile_review(tmp_path, manifest, blank_path, blank_path, tmp_path / "blank.json")

    # Synthetic review records exist only inside this test directory.
    keys = json.loads((packet_root / "private_key.json").read_text())
    for side in ("a", "b"):
        write_json(tmp_path / f"fixture_{side}.json", [
            {"audit_id": aid, "reviewer_id": f"test_fixture_{side}", **key["literal"],
             "both_plausible": True if key["has_pair"] else None, "rationale": "test fixture"}
            for aid, key in keys.items()
        ])
    a, b = (tmp_path / f"fixture_{side}.json" for side in ("a", "b"))
    b_rows = json.loads(b.read_text())
    b_rows[0]["admissible_answers"] = ["un valor diferente"]
    b.unlink()
    write_json(b, b_rows)
    report = reconcile_review(tmp_path, manifest, a, b, tmp_path / "pending.json")
    assert report["status"] == "pending_human_audit"
    assert report["unresolved_audit_ids"] == [b_rows[0]["audit_id"]]
    adjudication = tmp_path / "adjudication.json"
    write_json(adjudication, [{"audit_id": b_rows[0]["audit_id"], "decision": "keep_as_written",
                               "rationale": "test fixture adjudication"}])
    report = reconcile_review(tmp_path, manifest, a, b, tmp_path / "reconciled.json", adjudications=adjudication)
    assert report["status"] == "passed" and report["roots"] == 100
    assert report["source_sha256"]["review_a"] == file_hash(a)
