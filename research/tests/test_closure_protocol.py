from pathlib import Path
from unittest.mock import patch

import pytest

from research.src.artifacts import digest, file_hash, write_json
from research.src.protocol import GateError, assert_split_allowed
from research.src.closure_protocol import (checked_stage, execution_context, record_stage,
    require_training_authorization, verify_locked_protocol, selected_checkpoints)
from research.closure_v4 import main


def new_run(tmp_path):
    write_json(tmp_path / "MANIFIESTO_V4.json", {"protocol_version": "closure_v4", "stages": {}})
    return tmp_path


def stage(run, name, result=None):
    evidence = run / (name + "_evidence.json")
    write_json(evidence, {"measured": [1, 2, 3]})
    record_stage(run, name, result or {"status": "passed"}, [evidence])
    return evidence


def authorize(run):
    for name in ("inventory", "test-environment", "test-training"):
        stage(run, name)
    stage(run, "diagnose", {"status": "partial", "critical_checks_passed": True})
    write_json(run / "PILOT_PLAN.json", {"max_updates_per_branch": 5})
    from research.src.artifacts import load_config
    manifest = load_config(run / "MANIFIESTO_V4.json")
    manifest["pilot_plan_sha256"] = file_hash(run / "PILOT_PLAN.json")
    write_json(run / "MANIFIESTO_V4.json", manifest)


def test_no_labels_without_evidence_release_training(tmp_path):
    run = new_run(tmp_path)
    with pytest.raises(GateError, match="inventory_REQUIRED"):
        require_training_authorization(run)


def test_tampered_fixture_evidence_blocks_training(tmp_path):
    run = new_run(tmp_path)
    authorize(run)
    require_training_authorization(run)
    (run / "test-environment_evidence.json").write_text("{}")
    with pytest.raises(GateError, match="EVIDENCE_CHANGED"):
        require_training_authorization(run)


def test_changed_sources_invalidate_stage(tmp_path):
    run = new_run(tmp_path)
    stage(run, "inventory")
    with patch("research.src.closure_protocol.source_hashes", return_value={"changed": "abc"}):
        with pytest.raises(GateError, match="SOURCE_CHANGED"):
            checked_stage(run, "inventory")


def test_sources_changed_during_stage_cannot_be_sealed(tmp_path):
    run = new_run(tmp_path)
    from research.src.artifacts import load_config
    manifest = load_config(run / "MANIFIESTO_V4.json")
    manifest["code_snapshot_sha256"] = "different_snapshot"
    write_json(run / "MANIFIESTO_V4.json", manifest)
    with pytest.raises(GateError, match="SOURCE_CHANGED_DURING_RUN"):
        stage(run, "inventory")


def test_blocked_secondary_does_not_release_critical_diagnosis(tmp_path):
    run = new_run(tmp_path)
    for name in ("inventory", "test-environment", "test-training"):
        stage(run, name)
    stage(run, "diagnose", {"status": "partial", "critical_checks_passed": False})
    with pytest.raises(GateError, match="CRITICAL_DIAGNOSTICS"):
        require_training_authorization(run)


def test_closure_training_cannot_read_test(tmp_path):
    run = new_run(tmp_path)
    authorize(run)
    with execution_context(run, "training"):
        assert_split_allowed("train", training=True)
        with pytest.raises(GateError, match="CLOSED_TEST"):
            assert_split_allowed("test_id")
        with pytest.raises(GateError, match="train partition"):
            assert_split_allowed("validation", training=True)


def test_missing_lock_rejects_before_models_or_opening(tmp_path):
    run = new_run(tmp_path)
    assert main(["evaluate-locked-test", "--output", str(run)]) == 2
    assert not (run / "TEST_OPENING_V4.json").exists()
    with pytest.raises(GateError, match="LOCKED_PROTOCOL_REQUIRED"):
        selected_checkpoints(run)


def test_corrupted_lock_rejected(tmp_path):
    run = new_run(tmp_path)
    write_json(run / "PROTOCOLO_BLOQUEADO.json", {"status": "locked", "sha256": "invented"})
    with pytest.raises(GateError, match="PROTOCOL_CHANGED"):
        verify_locked_protocol(run)


def test_history_cannot_be_overwritten():
    from research.src.closure_protocol import ROOT
    with pytest.raises(GateError, match="HISTORICAL"):
        record_stage(ROOT / "runs/full_exploratory_001", "fake", {"status": "passed"})
