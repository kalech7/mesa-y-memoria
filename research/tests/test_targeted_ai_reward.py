"""CPU contracts for safety tiers, target credit and AI feedback integrity."""
from copy import deepcopy
import json
from threading import Thread
import time

import pytest

from research.src.artifacts import write_json
from research.src.evaluation.runner import run_episode
from research.src.policy.rule_controller import RuleController
from research.targeted_ai_reward import (
    FLAGS, JUDGE, judge_payload, parse_verdict, request_feedback, reward,
    rewards_from_components, target_write_opportunity,
)
from research.tests.test_competence_evaluation import _episode, _record


def verdict(**changes):
    return {**{flag: True for flag in FLAGS}, "feedback": "Se revisaron las evidencias.", **changes}


def trajectory(tmp_path, *, ambiguous=False, controller=None):
    episode = _episode(ambiguous=ambiguous, world="ongoing", split="dev_smoke")
    return run_episode(episode, controller or RuleController(), db_path=tmp_path / "trajectory.sqlite")


def test_any_invalid_phase_ranks_below_safe_wrong_response(tmp_path):
    row = trajectory(tmp_path)
    wrong = deepcopy(row)
    wrong["trace"]["response"]["value"] = "valor incorrecto"
    safe = reward(wrong, verdict(response_supported=False))
    assert safe["tier"] == 0 and not safe["unsafe"]
    for phase in ("select", "respond", "write"):
        damaged = deepcopy(row)
        damaged["trace"]["errors"] = [{"phase": phase, "kind": "invalid_output"}]
        unsafe = reward(damaged, verdict())
        assert unsafe["tier"] == -2 and unsafe["unsafe"]
        assert all(unsafe["phase_rewards"][key] < safe["phase_rewards"][key] for key in safe["phase_rewards"])


def test_joint_tier_dominates_all_safe_nonjoint_phase_auxiliaries():
    base = {
        "tier": 3, "costs": .10, "ask_cost_waiver": 0.,
        "informative_question": False, "questions": 1, "current_ok": True,
        "benefit": -1, "target_update_needed": True, "target_memory_complete": False,
        "global_memory_integrity": True, "necessary_omission": True,
    }
    low_joint = rewards_from_components(base)
    assert low_joint["phase_auxiliary"]["write"] == -.30
    for tier in (0, 1):
        high_other = rewards_from_components({
            **base, "tier": tier, "costs": 0., "questions": 0, "informative_question": True,
            "benefit": 1, "target_memory_complete": True, "necessary_omission": False,
        })
        assert all(low_joint["phase_rewards"][phase] > high_other["phase_rewards"][phase]
                   for phase in low_joint["phase_rewards"])


def test_relevant_omission_is_penalized_but_uncertain_future_is_not_invented(tmp_path):
    class NoWriter(RuleController):
        def write(self, observation):
            return {"operations": [{"op": "NOOP"}]}
    row = trajectory(tmp_path, ambiguous=True, controller=NoWriter())
    score = reward(row, verdict(target_memory_complete=False, future_supported=False))
    assert score["program_current"] == 1 and score["program_future"] == 0
    assert score["target_update_needed"] and score["necessary_omission"]
    assert score["phase_auxiliary"]["write"] == -.15


def test_irrelevant_faithful_write_gets_no_positive_write_auxiliary(tmp_path):
    episode = _episode(split="dev_smoke")
    drink = _record("m2", "agua", "2026-01-01")
    drink["attribute"] = "drink_preference"
    drink["evidence_quotes"][0]["text"] = "Para beber prefiero agua de forma habitual."
    episode["initial_memory"].append(drink)
    episode["user_messages"] = [
        {"role": "user", "turn_id": "u1", "text": "Para beber prefiero té de menta de forma habitual."},
        {"role": "user", "turn_id": "u2", "text": "¿Qué ceno hoy?"},
    ]
    row = run_episode(episode, RuleController(), db_path=tmp_path / "irrelevant.sqlite")
    assert any(operation["op"] != "NOOP" for operation in row["trace"]["executed_operations"])
    row["trace"]["response"] = {"mode": "ANSWER", "value": "té de menta", "branches": [], "evidence_ids": ["u1"]}
    score = reward(row, verdict(response_supported=False))
    assert score["tier"] == 0 and not score["target_update_needed"]
    assert score["phase_auxiliary"]["write"] == 0 and score["episodic_reward"] == 0


def test_opportunity_depends_on_public_observation_and_query_without_gold(tmp_path):
    row = trajectory(tmp_path, ambiguous=True)
    original = target_write_opportunity(row["trace"])
    changed = deepcopy(row["trace"])
    changed["episode"]["truth"] = {"hidden": "unavailable"}
    changed["episode"]["opportunity"] = {"hidden": "unavailable"}
    assert original and target_write_opportunity(changed) is original


def test_global_integrity_veto_and_mean_phase_transport_match(tmp_path):
    row = trajectory(tmp_path, ambiguous=True)
    good = reward(row, verdict())
    bad = reward(row, verdict(global_memory_integrity=False))
    assert good["program_joint"] == 1 and good["joint"] and good["tier"] == 3
    assert bad["program_joint"] == 1 and not bad["joint"] and bad["tier"] == -2
    assert bad["benefit"] == 0
    for result in (good, bad):
        assert result["episodic_reward"] == pytest.approx(sum(result["phase_rewards"].values()) / 3)
        assert result["optimizer_r2"] + result["optimizer_r3"] - result["costs"] == pytest.approx(result["episodic_reward"])
        assert all(abs(value) <= .300000000001 for value in result["phase_auxiliary"].values())


def test_payload_retains_global_memory_and_does_not_reveal_rule_metrics(tmp_path):
    row = trajectory(tmp_path)
    record = _record("m99", "agua", "2026-01-01")
    record["attribute"] = "drink_preference"
    row["trace"]["initial_memory"].append(record)
    row["trace"]["final_memory"].append(deepcopy(record))
    payload = judge_payload(row)
    assert any(r["memory_id"] == "m99" for r in payload["memoria_inicial"])
    changed = deepcopy(row)
    changed["metrics"] = {"automatic_reward": "MUST_NOT_REACH_JUDGE"}
    assert judge_payload(changed) == payload
    assert "MUST_NOT_REACH_JUDGE" not in json.dumps(payload)


def test_feedback_flags_fail_closed():
    with pytest.raises(ValueError, match="five flags"):
        parse_verdict({"feedback": "Incomplete"})
    with pytest.raises(ValueError, match="JSON booleans"):
        parse_verdict(verdict(response_supported=1))


def test_atomic_request_requires_exact_hash_and_judge(tmp_path):
    def answer():
        request = tmp_path / "judge_queue/group.request.json"
        while not request.exists():
            time.sleep(.01)
        response = {"judge": JUDGE, "request_sha256": "wrong", "verdicts": {"case": verdict()}}
        write_json(request.with_name("group.response.json"), response)
    worker = Thread(target=answer)
    worker.start()
    with pytest.raises(ValueError, match="request hash"):
        request_feedback(tmp_path, "group", [{"id": "case", "evidence": {"raw": "observable"}}], timeout=2)
    worker.join()
