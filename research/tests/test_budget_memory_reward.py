"""CPU tests of utility, evidence isolation and atomic real-judge transport."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
from threading import Thread
import time

import pytest

from research.budget_memory_reward import (
    JUDGE, FLAGS, _atomic_json, judge_payload, parse_verdict, request_feedback,
    reward, validate_response, write_response,
)
from research.src.artifacts import file_hash


def trace():
    return {
        "episode_id": "ep1", "dialogue": [
            {"role": "user", "text": "En casa prefiero sopa; en oficina, arroz.",
             "turn_id": "u1", "date": "2030-01-01", "session_id": "S1"},
        ], "queries": [{"query_id": "q1", "query": {
            "attribute": "dinner", "date": "2030-01-02", "context": "casa", "session_id": "S2"},
            "response": {"mode": "ANSWER", "value": "sopa"}, "accepted_by_user": True,
            "memory_supported": True}],
        "final_slots": [
            {"slot_id": 1, "exchange": [{"role": "user", "text": "En casa prefiero sopa; en oficina, arroz.", "turn_id": "u1"}],
             "source_session": 0, "stored_at": "2030-01-01"},
            {"slot_id": 2, "exchange": [], "source_session": None, "stored_at": None}],
        "ask_count": 0, "write_count": 0,
    }


def verdict(**changes):
    return {"queries": {"q1": {"response_supported": True, "scope_supported": True,
                                "feedback": "Sopa está autorizada en casa.", **changes}}}


@pytest.mark.parametrize("accepted,memory,support,scope", tuple(product((False, True), repeat=4)))
def test_acceptance_memory_guard_and_both_support_flags_are_required(accepted, memory, support, scope):
    row = trace(); row["queries"][0]["accepted_by_user"] = accepted
    row["queries"][0]["memory_supported"] = memory
    score = reward(row, verdict(response_supported=support, scope_supported=scope))
    assert score["episodic_reward"] == (1 if accepted and memory and support and scope else -1)


@pytest.mark.parametrize("accepted,support,scope", tuple(product((False, True), repeat=3)))
def test_defer_is_always_zero_before_real_action_costs(accepted, support, scope):
    row = trace(); row["queries"][0].update(accepted_by_user=accepted, response={"mode": "DEFER", "value": None})
    row.update(ask_count=1, write_count=2)
    score = reward(row, verdict(response_supported=support, scope_supported=scope))
    assert score["query_utilities"] == [0]
    assert score["episodic_reward"] == pytest.approx(-.14)


def test_queries_are_averaged_once_and_costs_are_per_episode():
    row = trace(); second = deepcopy(row["queries"][0]); second.update(query_id="q2", accepted_by_user=False)
    row["queries"].append(second); row.update(ask_count=2, write_count=3)
    labels = verdict(); labels["queries"]["q2"] = deepcopy(labels["queries"]["q1"])
    score = reward(row, labels)
    assert score["query_utilities"] == [1, -1]
    assert score["mean_utility"] == 0
    assert score["episodic_reward"] == pytest.approx(-.26)


def test_malformed_answer_never_earns_acceptance_reward():
    row = trace(); row["queries"][0]["response"]["mode"] = "invented_mode"
    assert reward(row, verdict())["episodic_reward"] == -1


@pytest.mark.parametrize("name,value", [("ask_count", True), ("write_count", -.1), ("write_count", -1), ("ask_count", None)])
def test_costs_fail_closed_on_invalid_environment_measurements(name, value):
    row = trace(); row[name] = value
    with pytest.raises(ValueError): reward(row, verdict())


def test_acceptance_measurement_is_not_truthy_coerced():
    row = trace(); row["queries"][0]["accepted_by_user"] = 1
    with pytest.raises(ValueError, match="boolean"): reward(row, verdict())


def test_payload_is_invariant_to_all_hidden_or_automatic_fields():
    row = trace(); original = judge_payload(row)
    row.update(hidden_user_truth="LEAK", gold="LEAK", ideal_actions="LEAK", slots_final="LEAK",
               metrics={"success": "LEAK"}, public_judge_payload="LEAK")
    row["queries"][0].update(accepted_by_user=False, publicly_supported="LEAK", compat_world_count="LEAK",
                              memory_supported=False, reader_metadata={"gold": "LEAK"})
    row["queries"][0]["query"]["truth"] = "LEAK"
    row["queries"][0]["response"]["expected_value"] = "LEAK"
    row["dialogue"][0]["gold"] = "LEAK"
    row["final_slots"][0]["automatic_support"] = "LEAK"
    row["final_slots"][0]["exchange"][0]["hidden_truth"] = "LEAK"
    assert judge_payload(row) == original
    assert "LEAK" not in json.dumps(judge_payload(row))
    assert judge_payload({"trace": row}) == original


def test_full_dialogue_retains_chronology_corrections_and_context():
    row = trace(); row["dialogue"].append({"role": "user", "text": "Corrijo: en casa prefiero pasta.",
                                         "turn_id": "u2", "date": "2030-01-03", "context": "casa"})
    assert judge_payload(row)["dialogue"] == row["dialogue"]
    projected = judge_payload(row); projected["dialogue"][0]["text"] = "mutation"
    assert row["dialogue"][0]["text"] != "mutation"


@pytest.mark.parametrize("bad", [
    {}, {"queries": {}}, {"queries": {"q1": {"response_supported": 1, "scope_supported": True, "feedback": "x"}}},
    {"queries": {"q1": {"response_supported": True, "scope_supported": True, "feedback": ""}}},
    {"queries": {"q1": {"response_supported": True, "scope_supported": True, "feedback": "x", "gold": 1}}},
])
def test_exact_typed_verdict_schema(bad):
    with pytest.raises(ValueError): parse_verdict(bad)


def test_query_ids_must_match_without_missing_or_extra_labels():
    with pytest.raises(ValueError, match="IDs"): parse_verdict(verdict(), ["q2"])
    row = trace(); row["queries"].append(deepcopy(row["queries"][0]))
    with pytest.raises(ValueError, match="Duplicate"): reward(row, verdict())


def _request(tmp_path):
    path = tmp_path / "judge_queue/g.request.json"
    _atomic_json(path, {"samples": [{"id": "ep1", "evidence": judge_payload(trace())}]})
    return path


def test_atomic_write_never_overwrites_existing_artifacts(tmp_path):
    path = tmp_path / "sample.json"; _atomic_json(path, {"before": 1})
    original = path.read_bytes()
    with pytest.raises(FileExistsError): _atomic_json(path, {"after": 2})
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_judge_writer_and_hash_bound_reply_roundtrip(tmp_path):
    request = _request(tmp_path)
    path = write_response(request, {"ep1": verdict()})
    assert validate_response(request, json.loads(path.read_text())) == {"ep1": verdict()}
    with pytest.raises(FileExistsError): write_response(request, {"ep1": verdict()})


@pytest.mark.parametrize("fault", ["judge", "hash", "episode_id", "query_id", "extra_field"])
def test_response_mismatch_fails_closed(tmp_path, fault):
    request = _request(tmp_path)
    response = {"judge": JUDGE, "request_sha256": file_hash(request), "verdicts": {"ep1": verdict()}}
    if fault == "judge": response["judge"] = "another judge"
    if fault == "hash": response["request_sha256"] = "0" * 64
    if fault == "episode_id": response["verdicts"]["extra"] = verdict()
    if fault == "query_id": response["verdicts"]["ep1"]["queries"]["extra"] = verdict()["queries"]["q1"]
    if fault == "extra_field": response["hidden"] = True
    with pytest.raises(ValueError): validate_response(request, response)


def test_request_waits_for_real_reply_and_has_no_default_positive_feedback(tmp_path):
    errors = []
    def judge():
        try:
            request = tmp_path / "judge_queue/g.request.json"
            deadline = time.monotonic() + 3
            while not request.exists() and time.monotonic() < deadline: time.sleep(.01)
            write_response(request, {"ep1": verdict(response_supported=False)})
        except Exception as error: errors.append(error)
    worker = Thread(target=judge); worker.start()
    labels = request_feedback(tmp_path, "g", [{"id": "ep1", "evidence": judge_payload(trace())}], timeout=3)
    worker.join()
    assert not errors
    assert labels["ep1"]["queries"]["q1"]["response_supported"] is False
    assert reward(trace(), labels["ep1"])["episodic_reward"] == -1


def test_timeout_preserves_request_and_produces_no_automatic_verdict(tmp_path):
    with pytest.raises(TimeoutError):
        request_feedback(tmp_path, "g", [{"id": "ep1", "evidence": judge_payload(trace())}], timeout=.01)
    assert (tmp_path / "judge_queue/g.request.json").exists()
    assert not (tmp_path / "judge_queue/g.response.json").exists()


def test_hidden_fields_cannot_be_smuggled_into_queue(tmp_path):
    evidence = judge_payload(trace()); evidence["queries"][0]["accepted_by_user"] = True
    with pytest.raises(ValueError, match="nonpublic"):
        request_feedback(tmp_path, "g", [{"id": "ep1", "evidence": evidence}], timeout=.01)


def test_reusing_sample_id_in_one_batch_is_rejected(tmp_path):
    sample = {"id": "ep1", "evidence": judge_payload(trace())}
    with pytest.raises(ValueError, match="Duplicate"):
        request_feedback(tmp_path, "g", [sample, deepcopy(sample)], timeout=.01)


def test_lucky_correct_guess_with_empty_slots_is_rejected_even_if_judge_misses_it():
    row = trace()
    for slot in row["final_slots"]: slot["exchange"] = []
    row["queries"][0]["memory_supported"] = False
    assert row["queries"][0]["accepted_by_user"] is True
    assert reward(row, verdict())["episodic_reward"] == -1
    assert judge_payload(row)["dialogue"]  # Complete dialogue cannot replace missing slots.


def test_memory_guard_is_required_and_exactly_boolean():
    row = trace(); row["queries"][0]["memory_supported"] = 1
    with pytest.raises(ValueError, match="memory_supported"): reward(row, verdict())
    del row["queries"][0]["memory_supported"]
    with pytest.raises(ValueError, match="memory_supported"): reward(row, verdict())


def test_context_exception_preserves_general_value_outside_scope_in_real_environment(tmp_path):
    """Regression for the earlier judge's false global-deletion assumption.

    Exercise the actual new SQLite store and private user semantics, not a
    mocked SUPERSEDE graph. Literal slot replacement has no implicit edges.
    """
    from research.budget_memory_env import BudgetMemoryStore, resolve_world
    store = BudgetMemoryStore(tmp_path / "context.sqlite")
    general = [{"role": "user", "text": "Habitualmente bebo café con leche.", "turn_id": "u1"}]
    contextual = [{"role": "user", "text": "En casa prefiero té de menta.", "turn_id": "u2"}]
    store.store(2, general, 0, "2030-01-01")
    before = store.snapshot()[1]
    store.store(1, contextual, 1, "2030-01-02")
    persisted = BudgetMemoryStore(store.path).snapshot()
    assert persisted[1] == before
    assert persisted[0]["exchange"] == contextual
    assert persisted[1]["exchange"] == general
    world = {"facts": [
        {"attribute": "drink_preference", "value": "café con leche", "valid_from": "2030-01-01",
         "valid_until_exclusive": None, "scope_kind": "ongoing", "sequence": 0},
        {"attribute": "drink_preference", "value": "té de menta", "valid_from": "2030-01-02",
         "valid_until_exclusive": None, "scope_kind": "contextual", "context": "casa", "sequence": 1},
    ]}
    query = {"attribute": "drink_preference", "date": "2030-01-03", "session_id": "S3"}
    assert resolve_world(world, {**query, "context": "casa"}) == "té de menta"
    assert resolve_world(world, {**query, "context": "oficina"}) == "café con leche"


def test_interval_expiry_and_correction_use_actual_user_semantics():
    from research.budget_memory_env import resolve_world
    base = dict(attribute="drink_preference", valid_until_exclusive=None, scope_kind="ongoing")
    world = {"facts": [
        {**base, "value": "agua", "valid_from": "2030-01-01", "sequence": 0},
        {**base, "value": "té de menta", "valid_from": "2030-01-02", "sequence": 1,
         "valid_until_exclusive": "2030-01-04"},
    ]}
    query = dict(attribute="drink_preference", context=None, session_id="S3")
    assert resolve_world(world, {**query, "date": "2030-01-03"}) == "té de menta"
    assert resolve_world(world, {**query, "date": "2030-01-04"}) == "agua"
    world["facts"].append({**base, "value": "café con leche", "valid_from": "2030-01-05", "sequence": 2})
    assert resolve_world(world, {**query, "date": "2030-01-05"}) == "café con leche"
