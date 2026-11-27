import csv
from copy import deepcopy
import json
from pathlib import Path

from research.src.artifacts import canonical
from research.src.closure_data import CLAIM_IDS, _annotate, prepare_development, validate_protocol_plan
from research.src.data.build_bank import build_bank


def _bank(path, split, seed, *, composition=False):
    episodes = build_bank(roots=5, seed=seed)
    for e in episodes:
        e["split"] = split
        # Independent content rather than just changing the root IDs.
        e["user_messages"][0]["text"] += " Versión " + str(seed) + "."
        if composition:
            record = deepcopy(e["initial_memory"][0])
            record.update(memory_id="extra", attribute="lunch_preference", value="verduras",
                          evidence_quotes=[{"turn_id": "extra", "text": "En el almuerzo prefiero verduras."}])
            e["initial_memory"].append(record)
    annotated = [_annotate(e) for e in episodes]
    path.write_text("".join(canonical(e) + "\n" for e in annotated))
    return annotated


def _plan(tmp_path):
    parent = tmp_path / "historical"
    parent.mkdir()
    historical = parent / "old_test.jsonl"
    _bank(historical, "test_id", 11)
    (parent / "context.json").write_text(json.dumps({"paths": {"test_id": str(historical)}}))
    paths = {}
    for key, split, seed in (("train_bank", "train", 1), ("validation_bank", "validation", 2),
                             ("test_id_bank", "test_id", 3), ("test_composition_bank", "test_composition", 4)):
        path = tmp_path / (key + ".jsonl")
        _bank(path, split, seed, composition=key == "test_composition_bank")
        paths[key] = str(path)
    precision = tmp_path / "precision.json"
    precision.write_text(json.dumps({"status": "executed", "source": "development", "estimated_half_width": .03}))
    return {**paths, "training_seeds": [101, 202, 303], "claims": list(CLAIM_IDS), "family_alpha": .05,
            "individual_confidence": 1 - .05 / 11, "harm_margin": .02,
            "selection_validity_guards": {"utility_tolerance": 1e-6,
                "validity_guards": {"invalid_output_rate": .05, "unsupported_writes": .05}},
            "budgets": {"training_seconds_per_arm_seed": 3600, "validation_seconds_per_arm_seed": 300,
                        "save_seconds_per_arm_seed": 60, "max_new_llm_calls": 600},
            "precision_plan": {"source": "development", "method": "paired_root_bootstrap",
                               "artifact_path": str(precision), "minimum_relevant_effect": .05,
                               "target_half_width": .03, "planned_test_id_roots": 5,
                               "planned_test_composition_roots": 5}}, parent


def test_protocol_audit_requires_actual_development_precision_before_opening_test(tmp_path):
    plan, parent = _plan(tmp_path)
    del plan["precision_plan"]
    result = validate_protocol_plan(plan, historical_root=parent)
    assert result["status"] == "failed"
    assert result["audit"]["test_banks_opened"] is False
    assert str(Path(plan["test_id_bank"]).resolve()) not in result["artifacts_sha256"]


def test_protocol_locks_real_disjoint_banks_but_retains_temporal_limit(tmp_path):
    plan, parent = _plan(tmp_path)
    result = validate_protocol_plan(plan, historical_root=parent)
    assert result["status"] == "passed", result["errors"]
    assert result["audit"]["test_banks_opened"] is True
    assert result["audit"]["confirmatory_temporal_provenance"] is False
    assert result["audit"]["human_audit"] == "not_executed"
    assert result["artifacts_sha256"][str(Path(plan["test_composition_bank"]).resolve())]


def test_new_seed_of_same_generator_is_rejected_as_composition_holdout(tmp_path):
    plan, parent = _plan(tmp_path)
    _bank(Path(plan["test_composition_bank"]), "test_composition", 99)
    result = validate_protocol_plan(plan, historical_root=parent)
    assert result["status"] == "failed"
    assert any("COMPOSITION_NOT_HELD_OUT mechanism_signature" in e for e in result["errors"])


def test_changing_root_ids_cannot_disguise_historical_test_reuse(tmp_path):
    plan, parent = _plan(tmp_path)
    rows = [json.loads(line) for line in (parent / "old_test.jsonl").read_text().splitlines()]
    for row in rows:
        row["root_id"] += "_renamed"
    Path(plan["test_id_bank"]).write_text("".join(canonical(e) + "\n" for e in rows))
    result = validate_protocol_plan(plan, historical_root=parent)
    assert any("HISTORICAL_TEST_REUSE" in e for e in result["errors"])


def test_root_cannot_cross_partitions_and_missing_metadata_blocks_lock(tmp_path):
    plan, parent = _plan(tmp_path)
    rows = [json.loads(line) for line in Path(plan["train_bank"]).read_text().splitlines()]
    for row in rows:
        row["split"] = "validation"
        del row["template_id"]
    Path(plan["validation_bank"]).write_text("".join(canonical(e) + "\n" for e in rows))
    result = validate_protocol_plan(plan, historical_root=parent)
    assert any("ROOT_SPLIT_LEAKAGE" in e for e in result["errors"])
    assert any("missing required metadata" in e for e in result["errors"])


def test_only_planned_claim_family_and_training_seeds_are_accepted(tmp_path):
    plan, parent = _plan(tmp_path)
    plan["claims"] = ["arbitrary_" + str(i) for i in range(11)]
    plan["training_seeds"] = [101]
    result = validate_protocol_plan(plan, historical_root=parent)
    assert result["status"] == "failed"
    assert result["audit"]["test_banks_opened"] is False
    assert any("training_seeds" in e for e in result["errors"])


def test_development_selection_keeps_all_variants_and_blind_reviews_unfilled(tmp_path):
    parent, output = tmp_path / "parent", tmp_path / "closure"
    parent.mkdir()
    episodes = build_bank(roots=40, seed=42, realizations=2)
    for e in episodes:
        e["split"] = "validation"
    path = parent / "validation.jsonl"
    path.write_text("".join(canonical(e) + "\n" for e in episodes))
    (parent / "context.json").write_text(json.dumps({"paths": {"validation": str(path)}}))
    result = prepare_development(parent, output, roots=30)
    assert result["status"] == "passed"
    assert result["human_review_roots"] == 30
    selected = set(result["selected_root_ids"])
    exported = [json.loads(line) for line in (output / "DEVELOPMENT_V4.jsonl").read_text().splitlines()]
    assert len(exported) == sum(e["root_id"] in selected for e in episodes)
    assert all(e["template_id"].startswith("derived_surface_") for e in exported)
    assert result["human_audit"] == "not_executed"
    reviews = list(csv.DictReader((output / "HUMAN_REVIEW_V4.csv").open()))
    assert len(reviews) == 30
    assert all(not row["reviewer_id"] and not row["need_now"] for row in reviews)
    assert all("root_id" not in row["visible_variants"] and '"truth"' not in row["visible_variants"] for row in reviews)
    assert (output / "private/HUMAN_REVIEW_KEY_V4.json").exists()
