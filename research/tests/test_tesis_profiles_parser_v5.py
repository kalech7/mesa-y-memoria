from copy import deepcopy

import pytest

from research.src.policy.rule_controller import evidence_model as native_evidence
from research.tesis_interactive_env import _native
from research.tesis_profiles_bank import build_development
from research.tesis_profiles_env import execute as execute_v4
from research.tesis_profiles_parser_v5 import (
    _fixed_decision, evidence_model, execute, normalized_observation, public_features, rule_action,
)
from research.src.artifacts import canonical


def observation(*texts):
    return {
        "query": {"attribute": "dinner_preference", "date": "2038-06-11", "session_id": "S2", "context": None},
        "memory": [], "reuse_regime": "Habrá otra consulta.", "visit": 1,
        "questions_remaining": 2, "asked_slots": [],
        "user_messages": [{"turn_id": f"u{index}", "role": "user", "text": text}
                          for index, text in enumerate(texts)],
    }


@pytest.mark.parametrize("text", [
    "¿Me ayudas con la lista? Para la cena, prefiero arroz con lentejas.",
    "No tengo claro el resto de la semana. Para la cena, prefiero arroz con lentejas.",
    "Para mi cena me apetece arroz con lentejas.",
    "Para la cena, me quedo con arroz con lentejas.",
])
def test_personal_choice_survives_opening_questions_and_unrelated_uncertainty(text):
    parsed = evidence_model(observation(text))
    assert parsed["new_record"]["value"] == "arroz con lentejas"
    assert parsed["missing_slot"] == "TEMPORAL_SCOPE"
    assert parsed["current_values"] == ["arroz con lentejas"]


@pytest.mark.parametrize("text", [
    "Ya no me apetece arroz con lentejas.",
    "Quizá me quedo con arroz con lentejas.",
    "Nunca prefiero arroz con lentejas.",
    "Me gustaría arroz con lentejas.",
])
def test_negation_and_conjecture_do_not_become_confirmed_positive_preferences(text):
    parsed = evidence_model(observation(text))
    assert parsed["new_record"] is None
    assert parsed["missing_slot"] == "GENERIC"


def test_preparation_and_quoted_or_attributed_preferences_do_not_change_food():
    parsed = evidence_model(observation(
        "Mi sobrina escribió «para la cena prefiero arroz». Para mi cena prefiero calentarla.",
        "¿Qué prefiero yo para la cena?",
    ))
    assert parsed["new_record"] is None
    assert parsed["missing_slot"] is None


@pytest.mark.parametrize("text", ["Quiero la pauta de siempre.", "Para mí conserva la rectificación que hice.",
                                  "Quiero lo habitual.", "Deja mi propia lista como estaba."])
def test_requests_to_reuse_memory_are_not_new_food_values(text):
    assert evidence_model(observation(text))["new_record"] is None


def test_correction_does_not_inherit_superseded_scope_and_attributes_are_independent():
    visible = observation(
        "Para la cena prefiero arroz de forma habitual. Rectifico, para la cena prefiero sopa.",
        "Para la bebida prefiero agua solo para esta sesión.",
    )
    parsed = evidence_model(visible)
    assert parsed["new_record"]["value"] == "sopa"
    assert parsed["new_record"]["scope"]["kind"] == "unknown"
    # A second attribute in the same raw turn also retains its own scope.
    combined = observation("Para la cena prefiero arroz de forma habitual. Para la bebida prefiero agua solo para esta sesión.")
    _, candidates = _fixed_decision(canonical(_native(combined)))
    records = {op["record"]["attribute"]: (op["record"], approved) for op, approved in candidates}
    assert set(records) == {"dinner_preference", "drink_preference"}
    assert records["dinner_preference"][0]["scope"]["kind"] == "ongoing"
    assert records["drink_preference"][0]["scope"]["kind"] == "session"
    assert all(approved for _, approved in records.values())
    assert all(record["source_turn_ids"] == ["u0"] for record, _ in records.values())
    assert all(record["evidence_quotes"] == [{"turn_id": "u0", "text": combined["user_messages"][0]["text"]}]
               for record, _ in records.values())


def test_explicit_future_start_and_inclusive_end_resolve_from_visible_calendar():
    future = evidence_model(observation(
        "El horario nuevo empieza el 14 de junio. A partir de ese día mi cena habitual será sopa de zapallo."
    ))["new_record"]
    assert future["scope"]["valid_from"] == "2038-06-14"
    bounded = evidence_model(observation(
        "Desde hoy hasta el 13 de junio inclusive quiero agua mineral como bebida."
    ))["new_record"]
    assert bounded["attribute"] == "drink_preference"
    assert bounded["value"] == "agua mineral"
    assert bounded["scope"]["valid_until_exclusive"] == "2038-06-14"
    next_year = observation("El horario nuevo empieza el 5 de enero. Mi cena habitual será arroz.")
    next_year["query"]["date"] = "2038-12-28"
    assert evidence_model(next_year)["new_record"]["scope"]["valid_from"] == "2039-01-05"


def test_deictic_context_restriction_does_not_become_general_preference():
    parsed = evidence_model(observation(
        "Cambia solo la bebida para la oficina. Allí deja agua mineral de ahora en adelante."
    ))
    assert parsed["new_record"]["attribute"] == "drink_preference"
    assert parsed["new_record"]["scope"]["context"] == "oficina"
    assert parsed["new_record"]["scope"]["kind"] == "contextual"
    assert parsed["user_context"] is None


def test_public_features_ignore_authored_labels_and_latent_facts():
    visible = observation("Voy con prisa. Para la cena prefiero sopa.")
    poisoned = {**visible, "strategy_cue": "PRIVATE", "scenario_family": "PRIVATE",
                "fact": {"value": "PRIVATE"}, "answer": "PRIVATE", "future_queries": ["PRIVATE"]}
    assert public_features(visible) == public_features(poisoned)


def test_training_first_visit_language_preserves_native_preference_semantics():
    rows = build_development(roots_per_family=1, validation_roots_per_family=1)["train"]
    for story in rows:
        node = execute_v4(story, ())
        native, parsed = native_evidence(_native(node.observation)), evidence_model(node.observation)
        for field in ("attribute", "value", "scope", "status"):
            assert parsed["new_record"][field] == native["new_record"][field]


def test_paraphrase_runs_through_sqlite_with_original_provenance():
    story = deepcopy(next(story for story in build_development()["train"]
                          if story["scenario_family"] == "direct_routine"
                          and story["visits"][0]["fact"]["scope"]["kind"] == "ongoing"
                          and story["visits"][1]["fact"]["scope"]["kind"] == "ongoing"
                          and story["plan_change"] == "canceled"))
    statement = next(turn for turn in story["visits"][0]["messages"]
                     if "prefiero" in turn["text"] and "hermana" not in turn["text"])
    statement["text"] = "¿Me ayudas? " + statement["text"].replace("prefiero", "me apetece")
    path = ()
    while hasattr(node := execute(story, path), "legal"):
        path += (rule_action(node.observation, node.legal),)
    assert node["future"][0]["supported"] == 1
    record = next(op["record"] for op in node["visits"][0]["executed_operations"] if "record" in op)
    assert record["evidence_quotes"][0] == {"turn_id": statement["turn_id"], "text": statement["text"]}
    assert "#v5_" not in repr(node["visits"][0]["outcome"])
