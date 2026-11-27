"""Literal regressions for the synthetic-data audit's four semantic defects."""
from copy import deepcopy

import pytest

from research.src.closure_environment import _episode, _record, _query, _scope
from research.src.env.user_simulator import UserSimulator
from research.src.evaluation.runner import run_episode
from research.src.policy.rule_controller import RuleController, _new_fact, authorized_operations, compatible_worlds
from research.src.readers.symbolic import applicability, read
from research.src.scoring.evidence import Evaluator, audit_operation, score_response, EVALUATOR_VERSION


def observation(text, *, context=None):
    return {"memory": [_record()], "query": _query(context=context),
            "user_messages": [{"turn_id": "u2", "text": text}],
            "reuse_regime": "later_same_attribute_without_questions"}


@pytest.mark.parametrize("text", [
    "Desde hoy no prefiero pasta de forma habitual. ¿Qué ceno hoy?",
    "Nunca prefiero pasta. ¿Qué ceno hoy?",
    "No es cierto que prefiero pasta. ¿Qué ceno hoy?",
    "Prefiero no comer pasta. ¿Qué ceno hoy?",
])
def test_negation_never_authorizes_a_positive_preference_or_a_lucky_answer(text):
    obs = observation(text)
    assert _new_fact(obs)[0] is None
    assert authorized_operations(obs) == [{"op": "NOOP"}]
    assert RuleController().respond(obs)["mode"] == "DEFER"
    lucky = {"mode": "ANSWER", "value": "pasta", "branches": [], "evidence_ids": ["u2"]}
    assert score_response(lucky, obs, "pasta")["utility"] == -1


def test_explicit_double_negative_continuation_retains_original_quote():
    text = "No he dejado de preferir pasta de forma habitual. ¿Qué ceno hoy?"
    obs = observation(text)
    operation = authorized_operations(obs)[0]
    assert operation["record"]["value"] == "pasta"
    assert operation["record"]["evidence_quotes"] == [{"turn_id": "u2", "text": text}]
    assert audit_operation(operation, obs)


def test_unanchored_later_does_not_become_a_present_preference_or_dated_write():
    obs = observation("Después prefiero pasta. ¿Qué ceno hoy?")
    fact = _new_fact(obs)[0]
    assert fact["scope"]["kind"] == "unknown"
    assert fact["scope"]["valid_from"] is None
    assert fact["scope"]["valid_until_exclusive"] is None
    assert authorized_operations(obs) == [{"op": "NOOP"}]
    assert RuleController().respond(obs)["mode"] == "DEFER"


@pytest.mark.parametrize("prefix", ["Solo en casa", "Como excepción en casa"])
def test_context_and_end_date_survive_together_in_the_reference_write(prefix):
    obs = observation(f"{prefix} hasta el 2026-01-13 inclusive prefiero pasta. ¿Qué ceno hoy?", context="casa")
    operation = authorized_operations(obs)[0]
    scope = operation["record"]["scope"]
    assert scope == _scope("contextual", start="2026-01-12", end="2026-01-14", context="casa")
    assert audit_operation(operation, obs)
    stored = {**operation["record"], "memory_id": "new", "supersedes": []}
    memory = obs["memory"] + [stored]
    assert read(memory, _query(date="2026-01-13", context="casa"))["value"] == "pasta"
    assert read(memory, _query(date="2026-01-13", context="oficina"))["value"] == "sopa"
    assert read(memory, _query(date="2026-01-14", context="casa"))["value"] == "sopa"


def test_context_restriction_is_not_a_disclosure_of_the_current_location():
    obs = observation("Solo en casa hasta el 2026-01-13 inclusive prefiero pasta. ¿Qué ceno hoy?")
    values = {w["value"] for w in compatible_worlds(obs)}
    assert values == {"sopa", "pasta"}
    assert RuleController().respond(obs)["mode"] == "CONDITIONALIZE"
    assert RuleController().select(obs) == {"action": "ASK", "slot": "CURRENT_CONTEXT"}


@pytest.mark.parametrize("context,expected", [("casa", "pasta"), ("oficina", "sopa")])
def test_current_answer_respects_a_new_preference_context(context, expected):
    obs = observation("Solo en casa hasta el 2026-01-13 inclusive prefiero pasta. ¿Qué ceno hoy?", context=context)
    assert RuleController().respond(obs)["value"] == expected


@pytest.mark.parametrize("date,expected", [("2026-01-11", "sopa"), ("2026-01-12", "pasta"), ("2026-01-14", "sopa")])
def test_current_answer_respects_explicit_start_and_end(date, expected):
    obs = observation("Desde el 2026-01-12 hasta el 2026-01-13 inclusive prefiero pasta. ¿Qué ceno hoy?")
    obs["query"]["date"] = date
    assert RuleController().respond(obs)["value"] == expected


def test_bounded_context_exception_overrides_context_rule_only_inside_its_window():
    general_home = _record("home", "arroz", scope=_scope("contextual", context="casa"))
    exception = _record("short", "pasta", scope=_scope("contextual", start="2026-01-12", end="2026-01-14", context="casa"))
    assert read([general_home, exception], _query(context="casa"))["value"] == "pasta"
    assert read([general_home, exception], _query(date="2026-01-14", context="casa"))["value"] == "arroz"


def test_new_context_rule_does_not_supersede_a_bounded_exception():
    obs = observation("En casa prefiero pasta de forma habitual. ¿Qué ceno hoy?", context="casa")
    old = _record("home", "arroz", scope=_scope("contextual", context="casa"))
    short = _record("short", "lentejas", scope=_scope("contextual", start="2026-01-12", end="2026-01-14", context="casa"))
    obs["memory"] = [old, short]
    operation = authorized_operations(obs)[0]
    assert operation["target_ids"] == ["home"]
    assert audit_operation(operation, obs)
    assert RuleController().respond(obs)["value"] == "lentejas"


def contextual_trace():
    ep = _episode()
    ep["initial_memory"] = [_record("home", "pasta", scope=_scope("contextual", context="casa")),
                            _record("office", "arroz", scope=_scope("contextual", context="oficina"))]
    ep["user_messages"] = [{"turn_id": "u2", "text": "¿Qué me recomiendas para cenar hoy?"}]
    ep["truth"].update(current_value="pasta", future_value="pasta", decisive_slot="CURRENT_CONTEXT")
    return run_episode(ep, RuleController(), method="C-NEVER")["trace"]


@pytest.mark.parametrize("phase", ["current", "future"])
def test_swapping_only_a_conditional_place_is_penalized_even_with_valid_values_and_quotes(phase):
    trace = contextual_trace()
    response = trace["response" if phase == "current" else "future_response"]
    assert {b["value"]: b["condition"] for b in response["branches"]} == {"pasta": "casa", "arroz": "oficina"}
    evaluator = Evaluator()
    method = evaluator.score_current if phase == "current" else evaluator.score_future
    assert method(trace)["utility"] == 0
    next(b for b in response["branches"] if b["value"] == "pasta")["condition"] = "oficina"
    assert method(trace)["utility"] == -1
    assert method(trace)["success"] == 0


def test_false_conditional_premise_cannot_hide_behind_an_existing_unrelated_source():
    trace = contextual_trace()
    trace["future_response"]["branches"][0]["condition"] = {"context": "casa"}
    # Alphabetical first value is arroz, whose office preference is not for casa.
    assert trace["future_response"]["branches"][0]["value"] == "arroz"
    score = Evaluator().score_future(trace)
    assert score["utility"] == -1 and score["conditional_conditions_valid"] is False


def test_superseded_or_more_specific_preferences_are_used_to_check_branch_conditions():
    trace = contextual_trace()
    # A general pasta rule must not justify pasta in the office where rice wins.
    trace["interaction_observation"]["memory"].append(_record("general", "pasta"))
    next(b for b in trace["future_response"]["branches"] if b["value"] == "pasta")["condition"] = "oficina"
    assert Evaluator().score_future(trace)["utility"] == -1


@pytest.mark.parametrize("kind", ["interval", "session", "unknown"])
def test_context_is_enforced_for_every_scope_kind(kind):
    scope = _scope(kind, start="2026-01-12", end="2026-01-14" if kind == "interval" else None,
                   context="casa", session="S2" if kind == "session" else None)
    record = _record("new", "pasta", scope=scope)
    assert applicability(record, _query(context="oficina")) is False


def test_missing_dictionary_context_is_uncertain_without_crashing():
    obs = observation("¿Qué me recomiendas para cenar hoy?")
    obs["memory"] = [_record("home", "pasta", scope=_scope("contextual", context={"place": "casa"}))]
    assert RuleController().respond(obs)["mode"] != "ANSWER"


@pytest.mark.parametrize("availability", [None, "available", "unavailable"])
def test_existing_ood_neutral_is_unavailable_even_if_metadata_claims_otherwise(availability):
    reply = {"turn_id": "u3", "text": "Sobre ese punto no tengo información adicional."}
    if availability is not None:
        reply["availability"] = availability
    simulator = UserSimulator({"GENERIC": reply})
    assert simulator.reply("GENERIC") == {"turn_id": "u3", "text": reply["text"]}
    assert simulator.last_response_status == "unavailable"


def test_neutral_response_has_no_acquisition_or_fabricated_ongoing_scope():
    ep = _episode()
    ep["replies"]["TEMPORAL_SCOPE"]["text"] = "Sobre ese punto no tengo información adicional."
    row = run_episode(ep, RuleController(), intervention="I2", forced_slot="TEMPORAL_SCOPE")
    assert row["metrics"]["user_response_received"] == 0
    assert row["metrics"]["response_informative"] == 0
    assert row["executed_operations"][0]["record"]["scope"]["kind"] == "unknown"
    assert row["metrics"]["FJS"] == 0


@pytest.mark.parametrize("text", [
    "No puedo precisar si es una preferencia habitual.",
    "No sé si es una preferencia habitual.",
    "Sobre ese punto no tengo información adicional: podría ser habitual.",
    "No estoy seguro de si solo para hoy.",
    "No puedo precisar hasta el 2026-01-15 inclusive.",
    "No es una preferencia habitual.",
    "Tal vez sea una preferencia habitual.",
])
def test_uncertain_or_negative_reply_keywords_do_not_resolve_scope(text):
    ep = _episode()
    ep["replies"]["TEMPORAL_SCOPE"] = {"turn_id": "u3", "text": text}
    row = run_episode(ep, RuleController(), intervention="I2", forced_slot="TEMPORAL_SCOPE")
    record = row["executed_operations"][0]["record"]
    assert record["scope"]["kind"] == "unknown"
    assert record["source_turn_ids"] == ["u2"]
    assert row["metrics"]["response_informative"] == 0
    assert row["metrics"]["FJS"] == 0
    assert row["rewards_by_component"]["components"]["future"] == 0


def test_denial_of_a_double_negative_does_not_become_an_affirmative_assertion():
    obs = observation("Es falso que no he dejado de preferir pasta de forma habitual. ¿Qué ceno hoy?")
    assert RuleController().respond(obs)["mode"] == "DEFER"
    assert authorized_operations(obs) == [{"op": "NOOP"}]


def test_repaired_evaluator_is_explicitly_versioned():
    assert EVALUATOR_VERSION == "evidence_v6_day_session_scope"
