"""V4 reexport keeps intervention identity and missing evidence explicit."""
import csv
import json
from pathlib import Path

import pytest

from research.src.artifacts import file_hash
from research.src.closure_export import flatten_episode, reexport_results


def _episode(rate=None):
    return {"experiment": "E5", "split": "test_id", "method": "B1", "seed": 101,
            "root_id": "root", "latent_world_id": "world", "realization_id": "v0",
            "scenario_family": "F4", "observation_condition": "O2", "intervention": None,
            "reader_id": "fixed_symbolic", "extraction_track": "REF", "adapter_hash": "weight_hash",
            "nonresponse_rate": rate, "technical_status": "executed",
            "metrics": {"C2": 1, "FJS": 0, "questions": 1, "invalid_outputs": 2,
                        "unsupported_operations": 3, "unsupported_write_episode": 1,
                        "useful_acquisition_numerator": 0, "acquisition_questions_denominator": 1},
            "rewards_by_component": {"R_long": -2., "R_short": -1., "components": {"current": 1}},
            "trace": {"rollout_seed": 0, "initial_memory": [], "final_memory": [],
                      "executed_operations": [], "episode": {"truth": {"secret": "HIDDEN"}},
                      "observations": {"write": {"user_messages": [{"text": "No puedo responder"}]}}}}


def _source(parent, rows, *, expected=None):
    traces = parent / "traces"
    reports = parent / "reports"
    traces.mkdir(parents=True)
    reports.mkdir()
    source = traces / "episodes.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (reports / "ANALISIS.json").write_text(json.dumps({"source_traces": [
        {"path": "/old/machine/traces/episodes.jsonl", "sha256": expected or file_hash(source)}]}))
    return source


def test_v4_delivery_metrics_survive_flattening():
    episode = _episode()
    episode["metrics"].update(ask_requested=1, ask_emitted=1, ask_delivered=1,
                              user_response_received=1, response_informative=0)
    flat = flatten_episode(episode, "closure_test")
    assert flat["ask_delivered"] == 1
    assert flat["response_informative"] == 0


def test_nonresponse_conditions_survive_reexport_without_deduplication(tmp_path):
    parent, output = tmp_path / "parent", tmp_path / "closure"
    source = _source(parent, [_episode(.2), _episode(.4), _episode(None), _episode(0)])
    original_hash = file_hash(source)
    result = reexport_results(parent, output, "closure_test")
    assert result["status"] == "passed"
    assert result["rows"] == 4
    assert result["duplicate_primary_keys"] == 0
    assert result["legacy_collision_keys"] == 1
    assert result["legacy_collision_rows"] == 4
    with (output / "RESULTADOS_V4.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {r["nonresponse_rate"] for r in rows} == {"0.2", "0.4", "null", "0"}
    assert len({r["condition_id"] for r in rows}) == 4
    assert all(r["J_primary"] == "0" for r in rows)
    assert all(r["unsupported_write_count"] == "3" for r in rows)
    assert all(r["acquisition_questions_denominator"] == "1" for r in rows)
    assert all(r["simulator_version"] == "null" for r in rows)
    assert file_hash(source) == original_hash
    memory_text = (output / "MEMORIAS_AUDITADAS.jsonl").read_text()
    assert "HIDDEN" not in memory_text
    assert '"before":[]' in memory_text
    assert '"semantic_support_status":"not_reaudited_by_exporter"' in memory_text


def test_legitimate_replicates_have_distinct_condition_ids():
    a, b = _episode(), _episode()
    a["stochastic_replica_id"], b["stochastic_replica_id"] = "r1", "r2"
    assert flatten_episode(a, "run")["condition_id"] != flatten_episode(b, "run")["condition_id"]


def test_duplicate_episode_is_failed_and_both_rows_are_kept(tmp_path):
    parent = tmp_path / "parent"
    _source(parent, [_episode(.2), _episode(.2)])
    result = reexport_results(parent, tmp_path / "closure", "r")
    assert result["status"] == "failed"
    assert result["rows"] == 2
    assert result["duplicate_primary_keys"] == 1
    assert result["duplicate_primary_rows"] == 2


def test_corrupted_source_hash_is_reported(tmp_path):
    parent = tmp_path / "parent"
    _source(parent, [_episode(.2)], expected="incorrect")
    result = reexport_results(parent, tmp_path / "closure", "r")
    assert result["status"] == "failed"
    assert result["sources"][0]["verified"] is False


def test_missing_sources_and_budget_do_not_manufacture_success(tmp_path):
    parent, output = tmp_path / "parent", tmp_path / "closure"
    parent.mkdir()
    result = reexport_results(parent, output, "r")
    assert result["status"] == "blocked_missing_evidence"
    assert result["rows"] == 0
    assert result["supplementary"]["budgets"]["status"] == "blocked_missing_evidence"
    assert not (output / "PRESUPUESTOS_V4.csv").exists()


def test_output_cannot_overwrite_history_or_previous_export(tmp_path):
    parent = tmp_path / "parent"
    _source(parent, [_episode()])
    with pytest.raises(ValueError):
        reexport_results(parent, parent / "closure", "r")
    output = tmp_path / "closure"
    reexport_results(parent, output, "r")
    with pytest.raises(FileExistsError):
        reexport_results(parent, output, "r")


def test_absent_scores_stay_null_even_on_technical_failure():
    row = _episode()
    row["technical_status"] = "failed"
    row["metrics"] = {}
    flat = flatten_episode(row, "r")
    assert flat["J_primary"] is None
    assert flat["current_error"] is None
    assert flat["invalid_output_count"] is None


def test_memory_null_boundary_has_unambiguous_semantics(tmp_path):
    row = _episode()
    row["trace"]["final_memory"] = [{"memory_id": "m1", "scope": {"kind": "ongoing", "valid_until_exclusive": None}},
                                      {"memory_id": "m2", "scope": {"kind": "unknown", "valid_until_exclusive": None}}]
    parent, output = tmp_path / "parent", tmp_path / "closure"
    _source(parent, [row])
    reexport_results(parent, output, "r")
    memory = json.loads((output / "MEMORIAS_AUDITADAS.jsonl").read_text())
    assert memory["after"][0]["valid_until_semantics"] == "until_superseded"
    assert memory["after"][1]["valid_until_semantics"] == "unknown"


def test_budget_and_curve_records_are_explicitly_historical(tmp_path):
    parent, output = tmp_path / "parent", tmp_path / "closure"
    _source(parent, [_episode()])
    (parent / "reports/PRESUPUESTOS.csv").write_text("method,seed,elapsed_seconds\nB1,101,2.5\n")
    (parent / "reports/CURVAS_ENTRENAMIENTO.csv").write_text("method,seed,step,loss\nB1,101,1,0.5\n")
    result = reexport_results(parent, output, "r")
    assert result["supplementary"]["budgets"]["rows"] == 1
    assert "historical_report_not_new_training" in (output / "CURVAS_RL_V4.csv").read_text()
