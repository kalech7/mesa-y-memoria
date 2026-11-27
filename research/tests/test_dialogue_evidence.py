from copy import deepcopy
from contextlib import closing

import pytest

from research.src.closure_environment import _episode, _record, _query, _scope
from research.src.env.episode import EpisodeEnvironment
from research.src.memory.executor import Executor
from research.src.memory.store import Store
from research.src.policy.rule_controller import RuleController, authorized_operations, compatible_worlds, evidence_model, expected_memory
from research.src.readers.symbolic import read
from research.src.scoring.evidence import audit_operation
from research.src.data.semantic_partition_v2 import assert_semantic_split_isolation_v2, SemanticSplitLeakage
from research.src.data.semantic_partition_v3 import semantic_episode_id_v3
from research.tesis_memory_guard import plan_write


def observation(*messages, attribute='dinner_preference'):
    return {'memory': [_record('m1', 'arroz'), _record('m2', 'agua', attribute='drink_preference')],
            'query': _query(attribute=attribute), 'reuse_regime': 'later',
            'user_messages': [{'turn_id': f'u{i}', 'role': 'user', 'text': text} for i, text in enumerate(messages, 1)]}


def test_assistant_preference_question_is_not_user_evidence():
    obs = observation('Hola.', '¿Qué me recomiendas para cenar?')
    obs['user_messages'].insert(1, {'turn_id': 'a1', 'role': 'assistant', 'text': 'Para la cena prefiero sopa de forma habitual.'})
    assert authorized_operations(obs) == [{'op': 'NOOP'}]
    assert RuleController().respond(obs)['value'] == 'arroz'


def test_latest_correction_wins_over_first_preference_and_greeting():
    obs = observation('Hola.', 'Para la cena prefiero pasta de forma habitual.',
                      'Corrijo: para la cena prefiero sopa de forma habitual.', '¿Qué ceno?')
    assert RuleController().respond(obs)['value'] == 'sopa'
    op = authorized_operations(obs)[0]
    assert op['op'] == 'SUPERSEDE'
    assert op['record']['source_turn_ids'] == ['u3']
    assert audit_operation(op, obs)


def test_later_scope_correction_replaces_prior_ongoing_scope():
    obs = observation('Para la cena prefiero sopa de forma habitual.',
                      'Corrijo el plazo: hasta el 2026-01-15 inclusive.', '¿Qué ceno?')
    record = evidence_model(obs)['new_record']
    assert record['scope']['kind'] == 'interval'
    assert record['scope']['valid_until_exclusive'] == '2026-01-16'
    assert record['source_turn_ids'] == ['u1', 'u2']
    assert {w['value'] for w in compatible_worlds(obs, query=_query(date='2026-01-16', session='S3'))} == {'arroz'}
    assert audit_operation(authorized_operations(obs)[0], obs)


@pytest.mark.parametrize(("wording", "scope", "expected"), [
    ("Solo por hoy", _scope("interval", start="2026-01-12", end="2026-01-13"), ("sopa", "sopa", "arroz", "arroz")),
    ("Solo en esta conversación", _scope("session", start="2026-01-12", session="S2"), ("sopa", "arroz", "arroz", "arroz")),
    ("Esta noche", _scope("interval", start="2026-01-12", end="2026-01-13"), ("sopa", "sopa", "arroz", "arroz")),
    ("Hoy", _scope("interval", start="2026-01-12", end="2026-01-13"), ("sopa", "sopa", "arroz", "arroz")),
    ("Solo para mañana", _scope("interval", start="2026-01-13", end="2026-01-14"), ("arroz", "arroz", "sopa", "arroz")),
    ("Solo para el 2026-01-13", _scope("interval", start="2026-01-13", end="2026-01-14"), ("arroz", "arroz", "sopa", "arroz")),
    ("Hasta mañana inclusive", _scope("interval", start="2026-01-12", end="2026-01-14"), ("sopa", "sopa", "sopa", "arroz")),
    ("Durante dos días", _scope("interval", start="2026-01-12", end="2026-01-14"), ("sopa", "sopa", "sopa", "arroz")),
    ("Desde ahora", _scope(start="2026-01-12"), ("sopa", "sopa", "sopa", "sopa")),
    ("Desde mañana", _scope(start="2026-01-13"), ("arroz", "arroz", "sopa", "sopa")),
])
def test_explicit_calendar_and_session_scopes_survive_reopening_sqlite(tmp_path, wording, scope, expected):
    visible = observation(f"Prefiero sopa {wording.lower()}.")
    operation = authorized_operations(visible)[0]
    record = operation["record"]
    assert record["value"] == "sopa"
    assert record["scope"] == scope
    assert record["source_turn_ids"] == ["u1"]
    assert audit_operation(operation, visible)
    guard = plan_write(visible, {"operations": [{"op": "NOOP"}]})
    assert guard["operations"] == [operation]
    path = tmp_path / "temporal.sqlite"
    with closing(Store(path)) as memory:
        memory.initialize(visible["memory"])
        assert Executor(memory).execute(guard["operations"], recorded_at="2026-01-12")["valid"]
    with closing(Store(path)) as memory:
        queries = (_query(), _query(session="S3"), _query(date="2026-01-13", session="S4"),
                   _query(date="2026-01-14", session="S5"))
        assert [read(memory.list_records(), query)["value"] for query in queries] == list(expected)


def test_relative_day_is_anchored_to_the_visible_dated_scope_reply_and_keeps_the_quote():
    visible = observation("Prefiero sopa.", "Solo por hoy.")
    visible["user_messages"][0]["date"] = "2026-01-11"
    visible["user_messages"][1]["date"] = "2026-01-12"
    visible["query"]["date"] = "2026-01-14"
    operation = authorized_operations(visible)[0]
    assert operation["record"]["scope"] == _scope("interval", start="2026-01-12", end="2026-01-13")
    assert operation["record"]["source_turn_ids"] == ["u1", "u2"]
    assert operation["record"]["evidence_quotes"][-1] == {"turn_id": "u2", "text": "Solo por hoy."}
    assert audit_operation(operation, visible)


def test_no_comma_can_reject_the_previous_alternative_and_confirm_today():
    confirmed = observation("Prefiero sopa.", "No, solo por hoy.")
    assert evidence_model(confirmed)["new_record"]["scope"] == _scope(
        "interval", start="2026-01-12", end="2026-01-13")
    for reply in ("No solo por hoy.", "No, no sé si solo por hoy."):
        unresolved = observation("Prefiero sopa.", reply)
        assert evidence_model(unresolved)["missing_slot"] == "TEMPORAL_SCOPE"


@pytest.mark.parametrize("wording", ("Hasta mañana.", "Durará durante 0 días.", "Hasta el 2026-02-30 inclusive."))
def test_an_ambiguous_or_invalid_boundary_asks_and_does_not_authorize_a_write(wording):
    visible = observation("Prefiero sopa.", wording)
    assert RuleController().select(visible) == {"action": "ASK", "slot": "VALID_UNTIL"}
    assert authorized_operations(visible) == [{"op": "NOOP"}]


def test_a_date_after_an_ambiguous_boundary_resolves_with_the_latest_evidence():
    visible = observation("Prefiero sopa.", "Hasta mañana.", "Confirmo: hasta el 2026-01-15 inclusive.")
    record = authorized_operations(visible)[0]["record"]
    assert record["scope"] == _scope("interval", start="2026-01-12", end="2026-01-16")
    assert record["source_turn_ids"] == ["u1", "u3"]


@pytest.mark.parametrize("quotation", ("«", "“", "\""))
def test_third_party_quotation_is_not_the_users_preference(quotation):
    end = {"«": "»", "“": "”", "\"": "\""}[quotation]
    other = f"Mi compañera dijo {quotation}para cenar prefiero pollo de forma habitual{end}."
    visible = observation(other, "¿Qué prefiero yo para cenar?")
    assert authorized_operations(visible) == [{"op": "NOOP"}]
    assert RuleController().respond(visible)["value"] == "arroz"
    own = observation(other + " Para la cena yo prefiero pasta solo por hoy.")
    record = authorized_operations(own)[0]["record"]
    assert record["value"] == "pasta"
    assert record["scope"] == _scope("interval", start="2026-01-12", end="2026-01-13")
    assert audit_operation(authorized_operations(own)[0], own)


def test_reported_place_and_meal_do_not_reassign_an_own_preference():
    visible = observation("Yo prefiero sopa solo por hoy. Mi compañera dijo «estoy en casa: para beber prefiero té».")
    assert evidence_model(visible)["user_context"] is None
    assert evidence_model(visible)["new_record"]["attribute"] == "dinner_preference"


def test_other_attribute_scope_does_not_disambiguate_dinner():
    obs = observation('Ahora, para la cena prefiero sopa.', 'Para beber prefiero café.',
                      'Como preferencia habitual desde ahora.', '¿Qué ceno?')
    assert evidence_model(obs)['missing_slot'] == 'TEMPORAL_SCOPE'
    operations = authorized_operations(obs)
    assert len(operations) == 2
    by_attr = {op['record']['attribute']: op['record'] for op in operations}
    assert by_attr['dinner_preference']['scope']['kind'] == 'unknown'
    assert by_attr['drink_preference']['scope']['kind'] == 'ongoing'
    assert len({r['memory_id'] for r in expected_memory(obs)}) == len(expected_memory(obs))


def test_context_location_and_context_restriction_are_distinct():
    obs = observation('Hoy estoy en casa.', 'Para la cena prefiero sopa.',
                      'Esa preferencia se aplica en la oficina.', '¿Qué ceno?')
    model = evidence_model(obs)
    assert model['user_context'] == 'casa'
    assert model['new_record']['scope']['context'] == 'oficina'
    assert RuleController().respond(obs)['value'] == 'arroz'


def test_two_unknown_attributes_branch_without_identity_collision():
    obs = observation('Para la cena ahora prefiero sopa.', 'Para beber ahora prefiero café.', '¿Qué ceno?')
    future = _query(date='2026-01-19', session='S3')
    assert {r['value'] for r in compatible_worlds(obs, query=future)} == {'arroz', 'sopa'}
    future['attribute'] = 'drink_preference'
    assert {r['value'] for r in compatible_worlds(obs, query=future)} == {'agua', 'café'}


def test_tool_clarification_does_not_apply_to_another_attribute():
    obs = observation('Para la cena ahora prefiero sopa.', 'Para beber ahora prefiero café.', '¿Qué ceno?')
    obs['user_messages'].append({'turn_id': 'u4', 'role': 'user', 'text': 'Como preferencia habitual desde ahora.',
        'in_reply_to': 'a_tool_1', 'about_attribute': 'dinner_preference'})
    records = {op['record']['attribute']: op['record'] for op in authorized_operations(obs)}
    assert records['dinner_preference']['scope']['kind'] == 'ongoing'
    assert records['drink_preference']['scope']['kind'] == 'unknown'


def test_assistant_quote_cannot_authorize_write():
    obs = observation('¿Qué ceno?')
    obs['user_messages'].append({'turn_id': 'a1', 'role': 'assistant', 'text': 'Para la cena prefiero sopa de forma habitual.'})
    forged = {'op': 'ADD', 'record': {**_record(value='sopa'), 'source_turn_ids': ['a1'],
              'evidence_quotes': [{'turn_id': 'a1', 'text': obs['user_messages'][-1]['text']}]}}
    assert not audit_operation(forged, obs)


def test_role_default_and_explicit_user_cannot_bypass_split_guard():
    first = _episode()
    second = deepcopy(first)
    second['root_id'] = 'renamed'
    second['user_messages'][0]['role'] = 'user'
    assert semantic_episode_id_v3(first) == semantic_episode_id_v3(second)
    with pytest.raises(SemanticSplitLeakage):
        assert_semantic_split_isolation_v2({'train': [first], 'validation': [second]})


def test_dialogue_tool_question_is_visible_and_I1_withholds_only_reply(tmp_path):
    episode = _episode()
    episode['user_messages'] = [{'turn_id': 'u1', 'role': 'user', 'text': 'Ahora, para la cena prefiero pasta.'},
                                {'turn_id': 'u2', 'role': 'user', 'text': '¿Qué ceno?'}]
    env = EpisodeEnvironment(episode, tmp_path / 'dialogue.sqlite')
    try:
        reply = env.ask('TEMPORAL_SCOPE')
        assert reply['role'] == 'user'
        assert reply['in_reply_to'] == 'a_tool_1'
        i1, i2 = env.observe('write', intervention='I1'), env.observe('write', intervention='I2')
        assert i1['user_messages'][-1]['role'] == 'assistant'
        assert i2['user_messages'][:-1] == i1['user_messages']
        assert evidence_model(i1)['new_record']['scope']['kind'] == 'unknown'
        assert evidence_model(i2)['new_record']['scope']['kind'] == 'ongoing'
    finally:
        env.dispose()
