import json
from pathlib import Path

import pytest

from research.src.artifacts import file_hash
from research.weak_sft_qa import _archived_paths, _check_archived_stages, _paired


def test_archived_gate_checks_original_artifact_hashes(tmp_path):
    plan = tmp_path / "PILOT_PLAN.json"
    plan.write_text("{}")
    stages = tmp_path / "stages"
    stages.mkdir()
    manifest = {"pilot_plan_sha256": file_hash(plan), "code_snapshot_sha256": "original_snapshot", "stages": {}}
    for name in ("inventory", "test-environment", "test-training", "diagnose"):
        path = stages / f"{name}.json"
        path.write_text(json.dumps({
            "stage": name,
            "code_snapshot_sha256": "original_snapshot",
            "artifacts_sha256": {str(plan): file_hash(plan)},
            "result": {"status": "passed", "critical_checks_passed": True},
        }))
        manifest["stages"][name] = {"path": str(path), "sha256": file_hash(path)}
    _check_archived_stages(tmp_path, manifest, Path)

    (stages / "inventory.json").write_text("{}")
    with pytest.raises(ValueError, match="Archived stage changed: inventory"):
        _check_archived_stages(tmp_path, manifest, Path)


def test_archived_paths_follow_the_supplied_run_and_repo(tmp_path):
    run = tmp_path / "copied_run"
    old_repo = Path("/old/location")
    old_run = old_repo / "research/runs/example"
    old_parent = Path("/historical/checkout/research/runs/parent")
    plan = {"runner_path": str(old_repo / "research/weak_sft_experiment.py"), "parent": {"path": str(old_parent)}}
    manifest = {"stages": {"inventory": {"path": str(old_run / "stages/inventory.json")}}}
    resolve = _archived_paths(run, plan, manifest)
    repo = Path(__file__).resolve().parents[2]

    assert resolve(old_run / "SELECTION.json") == run / "SELECTION.json"
    assert resolve(old_repo / "research/data/bank.jsonl") == repo / "research/data/bank.jsonl"
    assert resolve(old_parent / "local_checkpoint.json") == repo / "research/runs/parent/local_checkpoint.json"


def test_paired_contrast_weights_families_equally():
    families = {"a": "F0", "b": "F1", "c": "F1"}
    baseline = {
        "families": families,
        "by_root": {root: {"J_primary": 0} for root in families},
        "metrics": {"J_primary": 0},
    }
    candidate = {
        "families": families,
        "by_root": {"a": {"J_primary": 1}, "b": {"J_primary": 0}, "c": {"J_primary": 0}},
        "metrics": {"J_primary": 0.5},
    }

    assert _paired(candidate, baseline) == {
        "delta_J_pp": 50, "delta_J_pp_by_family": {"F0": 100, "F1": 0}, "improved": 1, "tied": 2, "worse": 0,
    }
