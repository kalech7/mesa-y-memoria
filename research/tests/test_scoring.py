"""Independent adversarial checks of justification and persisted provenance."""
from copy import deepcopy

from research.src.data.build_bank import build_bank
from research.src.env.episode import EpisodeEnvironment
from research.src.policy.rule_controller import RuleController
from research.src.readers.symbolic import read
from research.src.scoring.evidence import Evaluator, audit_operation, response_valid, score_response


def case(family, *, world=None):
    return next(episode for episode in build_bank() if episode["scenario_family"] == family
                and (family != "F4" or episode["truth"].get("decisive_slot") == "TEMPORAL_SCOPE")
                and (world is None or episode["truth"]["scope_kind"] == world))


def observation(episode, *, reply=False, condition="O2"):
    memory = deepcopy(episode["initial_memory"])
    if condition == "O1":
        memory = [max((m for m in memory if m["attribute"] == episode["current_query"]["attribute"]),
                      key=lambda m: m["recorded_at"])]
    messages = deepcopy(episode["user_messages"])
    if reply:
        messages.append(deepcopy(episode["replies"][episode["truth"]["decisive_slot"]]))
    return {"memory": memory, "query": deepcopy(episode["current_query"]), "user_messages": messages,
            "reuse_regime": "later_same_attribute_without_questions"}


def answer(value, ids):
    return {"mode": "ANSWER", "value": value, "branches": [], "evidence_ids": ids}


def trace_after(env, episode, response, interaction):
    env.close_session()
    future = env.reopen_for_reuse()
    trace = deepcopy(env.private_trace)
    trace.update(episode=deepcopy(episode), response=response,
                 interaction_observation=interaction, future_response=read(**future))
    return trace


def test_T18_lucky_context_is_not_justified():
    episode = next(e for e in build_bank() if e["scenario_family"] == "F3" and e["current_query"]["context"] is None)
    obs = observation(episode)
    correct_value = episode["truth"]["current_value"]
    guess = answer(correct_value, [obs["memory"][0]["memory_id"]])
    score = score_response(guess, obs, correct_value)
    assert score["factual"] == 1
    assert score["success"] == 0 and not score["justified"]
    assert score["utility"] == -1


def test_T18_O1_cannot_use_omitted_history():
    episode = case("F0")
    obs = observation(episode, condition="O1")
    guess = answer(episode["truth"]["current_value"], [obs["memory"][0]["memory_id"]])
    score = score_response(guess, obs, episode["truth"]["current_value"])
    assert score["factual"] == 1
    assert score["success"] == 0


def test_noop_does_not_make_old_memory_sufficient(tmp_path):
    episode = case("F4", world="session")
    env = EpisodeEnvironment(episode, tmp_path / "a.db")
    obs = env.observe("respond")
    current = RuleController().respond(obs)
    env.observe("write")
    env.apply_memory_operations([{"op": "NOOP"}])
    trace = trace_after(env, episode, current, obs)
    assert trace["future_response"]["value"] == episode["truth"]["future_value"]
    # The forgotten ambiguous declaration still prevents justified future use:
    # the world could equally have meant an ongoing change.
    score = Evaluator().score_future(trace)
    assert score["factual"] == 1 and score["success"] == 0
    env.dispose()


def test_I1_lucky_write_does_not_receive_justified_success(tmp_path):
    episode = case("F4", world="ongoing")
    env = EpisodeEnvironment(episode, tmp_path / "a.db")
    env.ask("TEMPORAL_SCOPE")
    respond = env.observe("respond", intervention="I1")
    write = env.observe("write", intervention="I1")
    # Deliberately give the actor a lucky operation constructed from withheld
    # evidence. It remains a real persisted write, and must be penalized.
    lucky_op = RuleController().write(respond)["operations"][0]
    assert not audit_operation(lucky_op, write)
    assert env.apply_memory_operations([lucky_op])["valid"]
    trace = trace_after(env, episode, RuleController().respond(respond), respond)
    assert trace["future_response"]["value"] == episode["truth"]["future_value"]
    assert Evaluator().score_memory(trace)["unsupported_operations"] == 1
    assert Evaluator().score_future(trace)["success"] == 0
    env.dispose()


def test_I2_faithful_persisted_write_succeeds(tmp_path):
    episode = case("F4", world="ongoing")
    env = EpisodeEnvironment(episode, tmp_path / "a.db")
    env.ask("TEMPORAL_SCOPE")
    respond = env.observe("respond", intervention="I2")
    write = env.observe("write", intervention="I2")
    op = RuleController().write(write)["operations"][0]
    assert audit_operation(op, write)
    env.apply_memory_operations([op])
    trace = trace_after(env, episode, RuleController().respond(respond), respond)
    assert Evaluator().score_memory(trace)["unsupported_operations"] == 0
    assert Evaluator().score_future(trace)["success"] == 1
    env.dispose()


def test_one_false_operation_has_one_penalty():
    episode = case("F1")
    obs = observation(episode)
    op = RuleController().write(obs)["operations"][0]
    op["record"]["value"] = "inexistente"
    op["record"]["scope"]["valid_from"] = "2030-01-01"
    op["record"]["evidence_quotes"][0]["text"] = "cita inventada"
    trace = {"observations": {"write": obs}, "executed_operations": [op]}
    score = Evaluator().score_memory(trace)
    assert score["unsupported_operations"] == 1
    assert score["operations"] == 1 and score["write_precision"] == 0


def test_scope_boundary_outside_fixed_probe_grid_is_checked():
    episode = case("F4", world="session")
    obs = observation(episode, reply=True)
    op = RuleController().write(obs)["operations"][0]
    op["record"]["scope"].update(kind="interval", session_id=None,
                                  valid_from="2035-05-01", valid_until_exclusive="2035-05-02")
    assert not audit_operation(op, obs)


def test_unknown_scope_does_not_smuggle_false_dates():
    episode = case("F4")
    obs = observation(episode)
    op = RuleController().write(obs)["operations"][0]
    assert audit_operation(op, obs)
    op["record"]["scope"]["valid_until_exclusive"] = "2035-05-01"
    assert not audit_operation(op, obs)


def test_response_parser_rejects_extra_keys_and_bad_branches():
    valid = answer("pasta", ["u2"])
    assert response_valid(valid)
    assert not response_valid({**valid, "latent_scope": "ongoing"})
    assert not response_valid({**valid, "branches": [{"condition": "hoy", "value": "pasta"}]})
    assert not response_valid({**valid, "mode": "DEFER", "value": "pasta"})
    assert not response_valid({**valid, "mode": "CONDITIONALIZE", "value": None,
                               "branches": [{"condition": "hoy", "value": "pasta", "gold": True}]})


def test_correct_current_value_requires_relevant_citation():
    episode = case("F1")
    obs = observation(episode)
    response = answer(episode["truth"]["current_value"], [obs["memory"][0]["memory_id"]])
    assert score_response(response, obs, episode["truth"]["current_value"])["success"] == 0
    response["evidence_ids"] = ["u2"]
    assert score_response(response, obs, episode["truth"]["current_value"])["success"] == 1


def test_write_cannot_cite_unrelated_true_quote():
    episode = case("F1")
    obs = observation(episode)
    op = RuleController().write(obs)["operations"][0]
    stale = obs["memory"][0]
    op["record"]["source_turn_ids"] = deepcopy(stale["source_turn_ids"])
    op["record"]["evidence_quotes"] = deepcopy(stale["evidence_quotes"])
    assert not audit_operation(op, obs)


def test_ongoing_write_cannot_omit_essential_scope_source():
    episode = case("F4", world="ongoing")
    obs = observation(episode, reply=True)
    op = RuleController().write(obs)["operations"][0]
    op["record"]["source_turn_ids"] = ["u2"]
    op["record"]["evidence_quotes"] = [q for q in op["record"]["evidence_quotes"] if q["turn_id"] == "u2"]
    assert not audit_operation(op, obs)


def test_invalid_future_output_counts_as_failure_without_crashing():
    episode = case("F0")
    trace = {"episode": episode, "final_memory": episode["initial_memory"], "future_response": None}
    result = Evaluator().score_future(trace)
    assert result["success"] == 0 and result["utility"] == -1 and not result["valid"]


def test_general_ongoing_write_is_faithful_beneath_contextual_exception():
    episode = case("F1")
    obs = observation(episode)
    exception = deepcopy(obs["memory"][0])
    exception.update(memory_id="context_exception", value="plato_contextual")
    exception["scope"].update(kind="contextual", context="oficina")
    exception["source_turn_ids"] = ["s1_exception"]
    exception["evidence_quotes"] = [{"turn_id": "s1_exception", "text": "En la oficina prefiero plato_contextual."}]
    obs["memory"].append(exception)
    operation = RuleController().write(obs)["operations"][0]
    assert operation["record"]["scope"]["kind"] == "ongoing"
    assert audit_operation(operation, obs)


def test_general_supersede_cannot_suppress_specific_exception():
    episode = case("F1")
    obs = observation(episode)
    exception = deepcopy(obs["memory"][0])
    exception.update(memory_id="context_exception", value="plato_contextual")
    exception["scope"].update(kind="contextual", context="oficina")
    exception["source_turn_ids"] = ["s1_exception"]
    exception["evidence_quotes"] = [{"turn_id": "s1_exception", "text": "En la oficina prefiero plato_contextual."}]
    obs["memory"].append(exception)
    operation = RuleController().write(obs)["operations"][0]
    operation["target_ids"].append("context_exception")
    assert not audit_operation(operation, obs)
