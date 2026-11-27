"""Long-run orchestration contracts without model loading or test opening."""
import json
from pathlib import Path

import pytest

from research.src.artifacts import canonical, file_hash
from research.src.full_campaign import FullCampaign, semantic_config, power_report_for_lock, verify_intervention_invariants


def campaign(tmp_path):
    root = tmp_path / "research"
    destination = root / "runs/probe"
    destination.mkdir(parents=True)
    return FullCampaign({"root": str(root), "destination": str(destination),
        "config": {"execution_mode": "exploratory", "runtime": {"evaluation_batch_size": 2},
                   "sample": {"test_id_locked_roots": None}},
        "paths": {}, "model_path": str(root / "model"), "research_status": "exploratory"})


def test_sample_size_lock_does_not_change_semantic_campaign_identity():
    before = {"sample": {"test_id_locked_roots": None}, "reward": {"future": 1}}
    after = {"sample": {"test_id_locked_roots": 300}, "reward": {"future": 1}}
    assert semantic_config(before) == semantic_config(after)
    after["reward"]["future"] = 0
    assert semantic_config(before) != semantic_config(after)


def test_completed_unit_reuse_verifies_hash_and_never_calls_model(tmp_path):
    execution = campaign(tmp_path)
    execution.active_stage = "probe"
    source = execution.trace_unit("unit", lambda: iter([{"seed": 101, "metrics": {"FJS": 0}}]),
                                  inputs={"weights": "frozen"}, expected_rows=1)
    def prohibited():
        raise AssertionError("Completed unit must not generate again")
    assert execution.trace_unit("unit", prohibited, inputs={"weights": "frozen"}, expected_rows=1) == source
    source.write_text('{"changed":true}\n')
    with pytest.raises(RuntimeError, match="UNIT_CHANGED"):
        execution.trace_unit("unit", prohibited, inputs={"weights": "frozen"}, expected_rows=1)


def test_partial_trace_is_preserved_and_never_silently_overwritten(tmp_path):
    execution = campaign(tmp_path)
    execution.active_stage = "probe"
    def interrupted():
        yield {"seed": 101}
        raise RuntimeError("infrastructure failure")
    with pytest.raises(RuntimeError, match="infrastructure"):
        execution.trace_unit("partial", interrupted, inputs={"seed": 101})
    first = execution.unit_path("partial").read_text()
    with pytest.raises(RuntimeError, match="EXPLICIT_RECOVERY"):
        execution.trace_unit("partial", lambda: iter([]), inputs={"seed": 101})
    assert execution.unit_path("partial").read_text() == first


def test_baseline_clone_records_reuse_without_another_independent_training(tmp_path):
    execution = campaign(tmp_path)
    execution.active_stage = "probe"
    source = execution.trace_unit("base101", lambda: iter([{"seed": 101, "method": "B0",
        "metrics": {"FJS": 0}, "resource_counters": {"generated_tokens": 30}}]), inputs={}, expected_rows=1)
    target = execution.clone_unit("base202", source, 202)
    row = json.loads(target.read_text())
    assert row["seed"] == 202
    assert row["cache_reuse"]["not_independent_training_run"] is True
    assert row["cache_reuse"]["source_trace_sha256"] == file_hash(source)
    assert row["resource_counters"]["generated_tokens"] == 0


def test_power_bridge_uses_real_worst_contrast_scenarios_and_retains_limits():
    native = {"precision_limited": True, "contrasts": [
        {"status": "executed", "scenarios": [{"candidate_total_id_roots": 300,
            "variance_multiplier": 1.5, "estimated_power": .6}]},
        {"status": "executed", "scenarios": [{"candidate_total_id_roots": 300,
            "variance_multiplier": 1.5, "estimated_power": .4}]}]}
    report = power_report_for_lock(native)
    assert report["status"] == "planning_only"
    assert report["scenarios"][0]["estimated_power"] == .4
    assert report["precision_limited"] is True
    assert report["native_five_contrast_report"] == native


def test_intervention_checks_prompt_bytes_as_well_as_observation_bytes(tmp_path):
    paths = {}
    for arm in ("I0", "I1", "I2"):
        response = {"visible_reply": arm != "I0"}
        write = {"visible_reply": arm == "I2"}
        row = {"root_id": "r", "latent_world_id": "w", "realization_id": "v",
               "metrics": {"C2": 1, "FJS": int(arm == "I2")},
               "rewards_by_component": {"R_long": 1 + int(arm == "I2")},
               "parsed_actions": {"respond": {"mode": "ANSWER", "value": "pasta"}},
               "trace": {"observations": {"respond": response, "write": write},
                         "phase_records": [{"phase": phase, "prompt": canonical(observation),
                            "generation": {"prompt_ids": [int(observation["visible_reply"])]}}
                            for phase, observation in (("respond", response), ("write", write))]}}
        path = tmp_path / (arm + ".jsonl")
        path.write_text(canonical(row) + "\n")
        paths[arm] = path
    result = verify_intervention_invariants(paths)
    assert result["status"] == "passed"
    assert result["effects"][0]["Delta_persistencia"] == 1
    bad = json.loads(paths["I2"].read_text())
    bad["trace"]["phase_records"][0]["prompt"] += " LEAK"
    paths["I2"].write_text(canonical(bad) + "\n")
    with pytest.raises(RuntimeError, match="PROMPT_BYTES_DIFFER"):
        verify_intervention_invariants(paths)


def test_full_dispatch_reaches_exploratory_completion_without_status_collision(tmp_path, monkeypatch):
    from research.src import full_campaign, prepare
    execution = campaign(tmp_path)
    data = execution.root / "data"
    data.mkdir()
    for split in ("train", "validation"):
        path = data / (split + ".jsonl")
        path.write_text('{"stub":true}\n')
        execution.paths[split] = path
    manifests = execution.root / "manifests"
    manifests.mkdir()
    (manifests / "selected_checkpoints.json").write_text('{"stub":true}')
    parent = execution.destination / "parent.json"
    parent.write_text('{"stub":true}')
    secondary = execution.destination / "secondary.json"
    secondary.write_text('{"stub":true}')
    marks = []
    monkeypatch.setattr(full_campaign, "require_gate", lambda *args: None)
    monkeypatch.setattr(prepare, "mark_gate", lambda gate, *args, **kwargs: marks.append(gate))
    empty = lambda: {"status": "executed", "artifacts": [], "traces": []}
    monkeypatch.setattr(execution, "mechanism", empty)
    monkeypatch.setattr(execution, "baselines", empty)
    monkeypatch.setattr(execution, "sft", lambda: {**empty(), "campaign": str(parent)})
    monkeypatch.setattr(execution, "release_g3", lambda *args: empty())
    monkeypatch.setattr(execution, "continuations", lambda *args: {**empty(), "campaign": str(parent)})
    monkeypatch.setattr(execution, "selection_and_g4", lambda *args: {**empty(), "secondary_plan": str(secondary)})
    monkeypatch.setattr(execution, "closed_test", empty)
    monkeypatch.setattr(execution, "interventions", lambda *args: empty())
    monkeypatch.setattr(execution, "common_reader_experiment", lambda *args, **kwargs: empty())
    monkeypatch.setattr(execution, "robustness", lambda *args: empty())
    monkeypatch.setattr(execution, "reports", lambda *args: {**empty(), "scheduled_rows": 0})
    result = execution.run()
    assert result["status"] == "executed"
    assert result["human_audit_status"] == "not_executed"
    assert marks == ["G5"]
    assert (execution.destination / "completion.json").is_file()
