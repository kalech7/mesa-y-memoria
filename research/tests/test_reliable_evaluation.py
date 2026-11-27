from copy import deepcopy

from research.src.closure_environment import _episode, _record, run_environment_checks
from research.src.evaluation.runner import run_episode
from research.src.policy.rule_controller import RuleController
from research.src.scoring.evidence import Evaluator, LegacyEvaluator, EVALUATOR_VERSION
from research.reliable_evaluation import summarize_sampling, verify_manifest, decision
import json
import pytest
from research.src.artifacts import file_hash


def ambiguous():
    return run_episode(_episode(), RuleController(), method="C-NEVER")


def test_production_runner_uses_corrected_future_return_without_rescoring_script():
    row = ambiguous()
    assert row["rewards_by_component"]["components"]["future"] == 0
    assert row["metrics"]["FJS"] == 0
    assert row["rewards_by_component"]["R_long"] == 1
    assert row["rewards_by_component"]["R_short"] == 1
    assert row["evaluator_version"] == EVALUATOR_VERSION


def test_answer_cannot_launder_a_mutated_initial_record_under_its_old_id():
    row = run_episode(_episode(), RuleController())
    trace = deepcopy(row["trace"])
    mid = trace["initial_memory"][0]["memory_id"]
    trace["final_memory"][0]["value"] = "pasta"
    trace["future_response"]["evidence_ids"] = [mid]
    assert LegacyEvaluator().score_future(trace)["success"] == 1
    score = Evaluator().score_future(trace)
    assert score["factual"] == 1 and score["success"] == 0
    assert score["utility"] == -1
    assert "mutated_initial_record:" + mid in score["provenance_errors"]


def test_renaming_new_memory_id_preserves_conditional_score():
    trace = ambiguous()["trace"]
    original = Evaluator().score_future(trace)
    mid = trace["final_memory"][-1]["memory_id"]
    trace["final_memory"][-1]["memory_id"] = "arbitrary_persisted_id"
    trace["future_response"]["evidence_ids"] = ["arbitrary_persisted_id" if i == mid else i for i in trace["future_response"]["evidence_ids"]]
    assert Evaluator().score_future(trace) == original


def test_turn_id_is_not_accepted_as_memory_citation_by_accident():
    trace = ambiguous()["trace"]
    trace["future_response"]["evidence_ids"] = ["u2"]
    score = Evaluator().score_future(trace)
    assert score["utility"] == -1
    assert "citation_not_in_persisted_memory" in score["provenance_errors"]


def test_expired_record_cannot_support_conditional_branch():
    trace = ambiguous()["trace"]
    new = trace["final_memory"][-1]
    new["scope"].update(kind="interval", valid_until_exclusive="2026-01-13")
    assert Evaluator().score_future(trace)["utility"] == -1


def test_reader_correct_value_does_not_hide_wrong_source_quote():
    episode = _episode()
    episode["user_messages"][0]["text"] = "Desde hoy prefiero pasta de forma habitual. ¿Qué ceno hoy?"
    row = run_episode(episode, RuleController(), intervention="I2", forced_slot="TEMPORAL_SCOPE")
    trace = deepcopy(row["trace"])
    operation = trace["executed_operations"][0]
    operation["record"]["evidence_quotes"][-1]["text"] = episode["user_messages"][0]["text"]
    trace["final_memory"][-1]["evidence_quotes"] = deepcopy(operation["record"]["evidence_quotes"])
    assert Evaluator().score_memory(trace)["unsupported_operations"] == 1
    score = Evaluator().score_future(trace)
    assert score["factual"] == 1 and score["success"] == 0
    assert score["provenance_valid"] is False


def test_closure_gate_executes_new_namespace_and_provenance_fixtures():
    result = run_environment_checks()
    assert result["status"] == "passed"
    names = {r["id"] for r in result["checks"]}
    assert {"EVA-S3-PERSISTED-CONDITIONAL", "EVA-S3-IMMUTABLE-INITIAL", "REW-S3-SHORT-INVARIANCE"} <= names


def test_sampling_uses_fixed_action_values_and_keeps_invalid_select_cost():
    state = {"observable_state_id": "s", "action_values": {
        "PROCEED": {"R_long": 1., "J_primary": 0},
        "ASK:TEMPORAL_SCOPE": {"R_long": 1.9, "J_primary": 1}}}
    rows = [{"observable_state_id": "s", "action_id": action}
            for action in ["PROCEED", "ASK:TEMPORAL_SCOPE", "INVALID"]]
    result = summarize_sampling(rows, {"states": [state]})
    assert result["draws"] == 3 and result["invalid_selects"] == 1
    assert result["draws_below_best_long_action"] == 2
    assert abs(result["state_uniform_mean_selection_regret"] - 2/3) < 1e-9


def test_missing_or_tampered_evidence_never_releases_evaluation(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}")
    manifest = {"status": "executed", "training": False, "artifacts_sha256": {evidence.name: file_hash(evidence)}}
    (tmp_path / "completion.json").write_text(json.dumps(manifest))
    verify_manifest(tmp_path)
    evidence.write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="Changed source artifact"):
        verify_manifest(tmp_path)


def test_no_observed_gap_is_not_declared_population_equivalence():
    result = decision({"class_counts": {}}, {"draws_below_best_long_action": 0})
    assert result["recommendation"] == "do_not_continue_GRPO_on_current_bank"
    assert "population equivalence" in result["cannot_conclude"]
    assert result["learning_efficacy"] == "not_executed"
