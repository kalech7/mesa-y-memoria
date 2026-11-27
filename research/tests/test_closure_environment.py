"""Regression tests for demonstrated V4 infrastructure failures."""
from copy import deepcopy

import pytest

from research.src.closure_environment import (
    _episode, _query, _record, _scope, infrastructure_fixtures,
    run_diagnostics, run_environment_checks, run_model_diagnostics,
)
from research.src.evaluation.runner import run_episode
from research.src.env.user_simulator import NON_RESPONSE, UserSimulator, nonresponse_uniform
from research.src.policy.rule_controller import RuleController, compatible_worlds


def test_independent_environment_fixtures_pass_without_models():
    result = run_environment_checks()
    assert result["fixture_count"] == 30
    assert len(infrastructure_fixtures()) == 30
    assert result["status"] == "passed", [c for c in result["checks"] if c["status"] != "passed"]


def test_persisted_unknown_scope_is_not_reanchored_to_new_session():
    unknown = _record("unknown", "pasta", scope=_scope("unknown", start="2026-01-12"))
    observation = {"memory": [_record(), unknown], "user_messages": [],
                   "query": _query(date="2026-01-19", session="S3")}
    assert {w["value"] for w in compatible_worlds(observation)} == {"pasta", "sopa"}


def test_c_always_executes_with_broken_model_selector():
    class Broken(RuleController):
        def select(self, obs):
            return {"action": "broken"}
        def select_forced_ask(self, obs):
            return {"action": "broken"}
    result = run_episode(_episode(), Broken(), method="C-ALWAYS")
    assert result["metrics"]["questions"] == result["metrics"]["ask_delivered"] == 1
    assert result["control_contract_version"] == "closure_v4"
    assert result["parsed_actions"]["select"] == {"action": "ASK", "slot": "TEMPORAL_SCOPE"}
    assert result["trace"]["user_response_status"] == "available"


def test_simulator_nested_noresponse_assignment_keeps_availability_separate():
    responses = []
    for rate in (.2, .4):
        simulator = UserSimulator(_episode()["replies"], root_id="fixed", nonresponse_rate=rate)
        reply = simulator.reply("TEMPORAL_SCOPE")
        unavailable = nonresponse_uniform("fixed", "TEMPORAL_SCOPE") < rate
        assert (reply == NON_RESPONSE) == unavailable
        assert simulator.last_response_status == ("unavailable" if unavailable else "available")
        assert set(reply) == {"turn_id", "text"}
        responses.append(unavailable)
    assert not responses[0] or responses[1]


def test_diagnostic_is_explicit_about_rule_fixture_not_b1(tmp_path):
    result = run_diagnostics(tmp_path / "missing_history", tmp_path / "output")
    assert result["status"] == "partial"
    assert result["gate_status"] == "blocked_missing_evidence"
    assert result["new_llm_calls"] == 0
    assert result["historical_cases"] == 0
    assert "not_B1" in (tmp_path / "output" / "DIAGNOSTICO_MEMORIAS.jsonl").read_text()
    assert result["fixture_matrix"]["D2"]["FJS"] == 1
    assert result["fixture_matrix"]["D3"]["FJS"] == 0
    assert any("No frozen B1/LLM-reader calls" in item for item in result["limitations"])


def test_e6_zero_question_uses_exact_e4_input_parser_score(monkeypatch):
    from research.src.evaluation import secondary
    from research.src.readers.symbolic import read
    row = run_episode(_episode(), RuleController())
    seen = []
    def generate(backend, phase, observations, **kwargs):
        seen.append((phase, deepcopy(observations)))
        return [{"parsed_action": read(o["memory"], o["query"])} for o in observations]
    monkeypatch.setattr(secondary, "generate_batch", generate)
    e4 = list(secondary.reread_frozen([row], object()))[0]
    e6 = list(secondary.reuse_with_interaction([row], object(), max_questions_s3=0))[0]
    assert seen[0] == seen[1]
    assert e4["secondary_reader_score"] == e6["secondary_reader_score"]
    assert e6["metrics"]["s3_questions"] == 0
    with pytest.raises(ValueError, match="BUDGET"):
        list(secondary.reuse_with_interaction([row], object(), max_questions_s3=2))


def test_model_diagnostic_d4_changes_only_writer_and_budgets_calls(tmp_path):
    import json
    from types import SimpleNamespace
    from llm_local_001.model import Generation
    class Backend:
        config = SimpleNamespace(max_new_tokens=64)
        def encode_prompt(self, prompt):
            return [1]
        def generate(self, prompt, **kwargs):
            observation = json.loads(prompt.split("OBSERVACION\n", 1)[1])
            phase = "select" if prompt.startswith("Selecciona") else "respond" if prompt.startswith("Responde") else "write"
            # Deliberately omit a useful write: D4 must improve it while keeping
            # the exact generated D2 answer and acquired evidence.
            action = {"operations": [{"op": "NOOP"}]} if phase == "write" else getattr(RuleController(), phase)(observation)
            return Generation([1], [2], json.dumps(action), truncated=False)
    result = run_model_diagnostics(tmp_path / "parent", tmp_path / "executed", episodes=[_episode()], backend=Backend())
    assert result["status"] == "executed" and result["critical_checks_passed"]
    assert result["new_llm_calls"] == 9
    assert result["summary"]["D2"]["FJS"] == 0
    assert result["summary"]["D4"]["FJS"] == 1
    rows = [json.loads(line) for line in (tmp_path / "executed" / "DIAGNOSTICO_MODELO_TRACES.jsonl").read_text().splitlines()]
    d2, d4 = rows[2], rows[4]
    assert d2["trace"]["response"] == d4["trace"]["response"]
    assert d2["trace"]["observations"]["write"] == d4["trace"]["observations"]["write"]
    blocked = run_model_diagnostics(tmp_path / "parent", tmp_path / "budget", episodes=[_episode()], max_llm_calls=8)
    assert blocked["status"] == "not_executed_budget"
    assert blocked["new_llm_calls"] == 0
