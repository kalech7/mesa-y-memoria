"""CPU contract tests: no language model, privileged policy or learned judge."""
import json
import re
import sqlite3

import pytest

from research.budget_memory_data import FAMILIES, build_bank, build_dataset, make_episode
from research.budget_memory_env import (
    ASK_ACTIONS, WRITE_ACTIONS, BudgetMemoryStore, check_token_budget,
    full_history_reader_observation, reader_candidates, resolve_world, run_episode, score_trace, score_response,
)


class ScriptedManager:
    def __init__(self, ask_at=None, ask_action="ASK_TEMPORAL_SCOPE", writes=None):
        self.ask_at = ask_at
        self.ask_action = ask_action
        self.writes = writes or {}
        self.observations = []

    def decide(self, observation, candidates, **kwargs):
        self.observations.append(json.loads(json.dumps(observation)))
        index = observation["session_index"]
        if observation["decision_type"] == "ask":
            action = self.ask_action if index == self.ask_at else "PROCEED"
        else:
            action = self.writes.get(index, "NOOP")
        return {"action": action, "metadata": {"prompt_ids": [1, 2],
                "candidate_token_ids": [[i] for i in range(len(candidates))],
                "action_index": candidates.index(action), "old_logprobs": [-1.] * len(candidates)}}


class LiteralTestReader:
    """A deliberately tiny fixture reader for unambiguous literal preference text.

    This is test infrastructure, not a production baseline or an oracle: it sees
    only the same raw slots/query that the frozen LM will receive.
    """
    def __init__(self):
        self.observations = []

    def answer(self, observation, candidates, **kwargs):
        self.observations.append(observation)
        attr = observation["query"]["attribute"]
        word = {"breakfast_preference": "desayuno", "dinner_preference": "cena",
                "drink_preference": "beber"}[attr]
        options = []
        for slot in observation["slots"]:
            text = " ".join(t["text"] for t in slot["exchange"])
            if word in text:
                pattern = "|".join(re.escape(v) for v in sorted(candidates[:-1], key=len, reverse=True))
                values = re.findall(pattern, text)
                if "Como preferencia habitual desde ahora." in text:
                    options.extend(values[-1:])
                elif "de forma habitual" in text:
                    options.extend(values)
        return {"answer": options[-1] if options else "DEFER", "metadata": {"frozen": True}}


def _temporal_with_available_reply():
    for index in range(1000):
        episode = make_episode("temporal", index, 91)
        facts = episode["private"]["worlds"][0]["facts"]
        if ([f["sequence"] for f in facts] == [0, 1, 2] and
            episode["private"]["worlds"][0]["replies"]["2"]["ASK_TEMPORAL_SCOPE"]["text"] != "Ahora no te sé decir."):
            episode["private"]["actual_world_id"] = episode["private"]["worlds"][1]["world_id"]
            return episode
    raise AssertionError("No available fixed test episode")


def test_balanced_fresh_user_world_disjoint_bank():
    a, b = build_bank(), build_bank()
    assert a == b
    users, worlds, roots = [], [], []
    for role, count in (("train", 60), ("validation", 15), ("evaluation", 25)):
        rows = a[role]
        assert len(rows) == count
        assert {f: sum(r["family"] == f for r in rows) for f in FAMILIES} == {f: count // 5 for f in FAMILIES}
        users.extend(r["user_id"] for r in rows)
        roots.extend(r["root_id"] for r in rows)
        worlds.extend(w["world_id"] for r in rows for w in r["private"]["worlds"])
    assert len(set(users)) == len(users) == len(set(roots))
    assert len(set(worlds)) == len(worlds)


def test_dataset_manifest_and_no_overwrite(tmp_path):
    manifest = build_dataset(tmp_path, 81, 82)
    assert manifest["historical_data_read"] is False
    assert manifest["model_calls"] == manifest["scoring_calls"] == 0
    assert len((tmp_path / "train.jsonl").read_text().splitlines()) == 60
    with pytest.raises(FileExistsError):
        build_dataset(tmp_path, 81, 82)


def test_sql_replacement_and_reopening_really_remove_old_slot(tmp_path):
    path = tmp_path / "memory.sqlite"
    store = BudgetMemoryStore(path)
    old = [{"turn_id": "old", "role": "user", "text": "Dato expulsado"}]
    new = [{"turn_id": "new", "role": "user", "text": "Dato actual"}]
    store.store(1, old, 0, "2035-01-01")
    store.store(2, old, 0, "2035-01-01")
    store.store(1, new, 1, "2035-01-02")
    reopened = BudgetMemoryStore(path).snapshot()
    assert reopened[0]["exchange"] == new and reopened[1]["exchange"] == old
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM slots").fetchone()[0] == 2
        assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [("slots",)]
    with pytest.raises(ValueError):
        store.store(3, new, 1, "2035-01-02")


def test_history_is_deleted_from_observation_and_reader(tmp_path):
    episode = make_episode("retention", 1, 71)
    manager, reader = ScriptedManager(), LiteralTestReader()
    trace = run_episode(manager, reader, episode, 12, tmp_path)
    old_text = episode["sessions"][0]["exchange"][0]["text"]
    for observation in manager.observations[2:]:
        assert old_text not in json.dumps(observation, ensure_ascii=False)
    for observation in reader.observations:
        assert set(observation) == {"slots", "query", "answer_menu"}
        assert all(not slot["exchange"] for slot in observation["slots"])
    assert len(trace["decisions"]) == 8
    assert old_text in json.dumps(trace["dialogue"], ensure_ascii=False)
    assert trace["metrics"]["net_return"] == 0


def test_budget_catalogues_and_full_metadata_preserved(tmp_path):
    manager = ScriptedManager(ask_at=0)
    trace = run_episode(manager, LiteralTestReader(), make_episode("retention", 2, 71), 15, tmp_path)
    assert trace["ask_count"] == 1
    assert trace["decisions"][0]["candidates"] == list(ASK_ACTIONS)
    assert all(d["candidates"] == ["PROCEED"] for d in trace["decisions"][2::2])
    assert all(d["candidates"] == list(WRITE_ACTIONS) for d in trace["decisions"][1::2])
    for decision in trace["decisions"]:
        assert decision["metadata"] == decision["policy_output"]["metadata"]
        assert "candidate_token_ids" in decision["metadata"]
    assert len({d["decision_id"] for d in trace["decisions"]}) == 8


def test_reader_gets_reopened_two_slots_not_privileged_history(tmp_path):
    episode = make_episode("retention", 3, 71)
    fact_positions = [i for i, s in enumerate(episode["sessions"]) if "de forma habitual" in s["exchange"][-1]["text"]]
    manager = ScriptedManager(writes={p: f"STORE_SLOT_{i+1}" for i, p in enumerate(fact_positions)})
    reader = LiteralTestReader()
    trace = run_episode(manager, reader, episode, 16, tmp_path)
    assert trace["persistence"]["reopened_before_reader"]
    assert trace["metrics"]["joint_success"] == 1
    assert trace["metrics"]["net_return"] == pytest.approx(.96)
    for observation in reader.observations:
        assert len(observation["slots"]) == 2
        serialized = json.dumps(observation)
        assert not any(key in serialized for key in ("actual_world", "facts", "private", "dialogue"))


def test_temporal_ask_eliminates_worlds_and_enables_supported_future(tmp_path):
    episode = _temporal_with_available_reply()
    writes = {0: "STORE_SLOT_1", 1: "STORE_SLOT_2", 2: "STORE_SLOT_1"}
    noask = run_episode(ScriptedManager(writes=writes), LiteralTestReader(), episode, 1, tmp_path / "noask")
    ask = run_episode(ScriptedManager(ask_at=2, writes=writes), LiteralTestReader(), episode, 1, tmp_path / "ask")
    assert all(q["compatible_world_count"] == 3 for q in noask["queries"])
    assert all(q["compatible_world_count"] == 1 for q in ask["queries"])
    assert ask["metrics"]["joint_success"] == 1
    assert ask["metrics"]["net_return"] > noask["metrics"]["net_return"]


def test_lucky_guess_after_ambiguity_is_not_supported(tmp_path):
    episode = _temporal_with_available_reply()
    actual = next(w for w in episode["private"]["worlds"] if w["world_id"] == episode["private"]["actual_world_id"])
    # Fixed test-only guess; this callable is never a production reader.
    def lucky(observation, candidates, **kwargs):
        return resolve_world(actual, observation["query"])
    trace = run_episode(ScriptedManager(), lucky, episode, 5, tmp_path)
    ambiguous = [q for q in trace["queries"] if not q["publicly_supported"]]
    assert len(ambiguous) == 1 and ambiguous[0]["accepted_by_user"]
    assert ambiguous[0]["deterministic_utility"] == -1


def test_same_world_assignment_across_rollout_seeds(tmp_path):
    episode = _temporal_with_available_reply()
    a = run_episode(ScriptedManager(ask_at=2), LiteralTestReader(), episode, 1, tmp_path / "a")
    b = run_episode(ScriptedManager(ask_at=2), LiteralTestReader(), episode, 999, tmp_path / "b")
    assert a["dialogue"] == b["dialogue"]
    assert [q["accepted_by_user"] for q in a["queries"]] == [q["accepted_by_user"] for q in b["queries"]]


def test_public_actor_and_judge_payloads_exclude_solution_and_future(tmp_path):
    episode = make_episode("context", 4, 71)
    trace = run_episode(ScriptedManager(ask_at=2, ask_action="ASK_CURRENT_CONTEXT"),
                        LiteralTestReader(), episode, 5, tmp_path)
    forbidden = {"private", "facts", "worlds", "actual_world_id", "accepted_by_user", "publicly_supported", "memory_supported"}
    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(v) for v in value)) if value else set()
        return set()
    for d in trace["decisions"]:
        assert not keys(d["observation"]) & (forbidden | {"query", "queries", "family", "root_id", "episode_id"})
    assert not keys(trace["public_judge_payload"]) & forbidden


def test_full_history_is_explicit_higher_budget_diagnostic(tmp_path):
    trace = run_episode(ScriptedManager(ask_at=2), LiteralTestReader(),
                        _temporal_with_available_reply(), 7, tmp_path)
    observation = full_history_reader_observation(trace, trace["queries"][0])
    assert len(observation["slots"]) == 4
    assert sum(len(s["exchange"]) for s in observation["slots"]) == len(trace["dialogue"])
    assert observation["answer_menu"] == reader_candidates(observation["query"]["attribute"])


def test_token_preflight_never_silently_truncates():
    class CharacterTokenizer:
        def encode(self, value, **kwargs):
            return list(value)
    with pytest.raises(ValueError, match="exceeds"):
        check_token_budget([make_episode("retention", 1, 1)], CharacterTokenizer(), limit=2)


def test_fail_closed_for_invalid_action_and_overwriting_trace(tmp_path):
    episode = make_episode("retention", 7, 71)
    with pytest.raises(ValueError, match="outside"):
        run_episode(lambda *a, **kw: "CORRECT_ACTION", LiteralTestReader(), episode, 1, tmp_path / "bad")
    run_episode(ScriptedManager(), LiteralTestReader(), episode, 1, tmp_path / "ok")
    with pytest.raises(FileExistsError):
        run_episode(ScriptedManager(), LiteralTestReader(), episode, 1, tmp_path / "ok")


def test_score_trace_uses_support_as_well_as_acceptance():
    trace = {"queries": [{"response": {"mode": "ANSWER"}, "accepted_by_user": True,
                           "publicly_supported": False, "memory_supported": False},
                          {"response": {"mode": "DEFER"}, "accepted_by_user": False,
                           "publicly_supported": False, "memory_supported": False}], "ask_count": 1, "write_count": 2}
    score = score_trace(trace)
    assert score["utility_mean"] == -.5
    assert score["net_return"] == pytest.approx(-.64)
    assert score["joint_success"] == 0


def test_empty_slots_cannot_score_a_lucky_correct_answer(tmp_path):
    episode = make_episode("retention", 3, 71)
    actual = next(w for w in episode["private"]["worlds"] if w["world_id"] == episode["private"]["actual_world_id"])
    def lucky(observation, candidates, **kwargs):
        return resolve_world(actual, observation["query"])
    trace = run_episode(ScriptedManager(), lucky, episode, 1, tmp_path)
    assert all(q["accepted_by_user"] and q["publicly_supported"] and not q["memory_supported"] for q in trace["queries"])
    assert trace["metrics"]["net_return"] == -1


def test_current_literal_support_passes_but_forged_text_fails(tmp_path):
    episode = make_episode("retention", 3, 71)
    trace = run_episode(ScriptedManager(), LiteralTestReader(), episode, 1, tmp_path)
    full = full_history_reader_observation(trace, trace["queries"][0])
    actual = episode["private"]["worlds"][0]
    query = trace["queries"][0]["query"]
    answer = resolve_world(actual, query)
    assert score_response(episode, trace["dialogue"], full["slots"], query, answer)["memory_supported"]
    for slot in full["slots"]:
        for turn in slot["exchange"]:
            turn["text"] += " texto inventado"
    assert not score_response(episode, trace["dialogue"], full["slots"], query, answer)["memory_supported"]


def test_corrected_or_expired_fact_is_not_a_current_support(tmp_path):
    episode = make_episode("correction", 2, 71)
    trace = run_episode(ScriptedManager(), LiteralTestReader(), episode, 1, tmp_path)
    actual = episode["private"]["worlds"][0]
    original, _, correction = actual["facts"]
    query = next(q for q in episode["queries"] if q["attribute"] == original["attribute"])
    old_turn = next(t for t in trace["dialogue"] if t["turn_id"] == original["support_sets"][0][0])
    slots = [{"exchange": [old_turn]}]
    flags = score_response(episode, trace["dialogue"], slots, query, correction["value"])
    assert flags["accepted_by_user"] and flags["publicly_supported"] and not flags["memory_supported"]
    wrong = score_response(episode, trace["dialogue"], slots, query, original["value"])
    assert not wrong["accepted_by_user"] and not wrong["memory_supported"]


def test_assistant_proposal_cannot_sustain_a_preference(tmp_path):
    episode = make_episode("retention", 3, 71)
    trace = run_episode(ScriptedManager(), LiteralTestReader(), episode, 1, tmp_path)
    query = episode["queries"][0]
    answer = resolve_world(episode["private"]["worlds"][0], query)
    # Even an exact real user text with its role changed is invalid provenance.
    source = next(t.copy() for t in trace["dialogue"] if answer in t["text"] and t["role"] == "user")
    source["role"] = "assistant"
    assert not score_response(episode, trace["dialogue"], [{"exchange": [source]}], query, answer)["memory_supported"]


def test_uninformative_or_wrong_slot_reply_does_not_prove_temporal_scope(tmp_path):
    episode = _temporal_with_available_reply()
    trace = run_episode(ScriptedManager(ask_at=2, ask_action="ASK_CURRENT_CONTEXT"), LiteralTestReader(), episode, 1, tmp_path)
    full = full_history_reader_observation(trace, episode["queries"][0])
    actual = episode["private"]["worlds"][1]
    query = next(q for q in episode["queries"] if q["attribute"] == actual["facts"][-1]["attribute"])
    answer = resolve_world(actual, query)
    assert not score_response(episode, trace["dialogue"], full["slots"], query, answer)["memory_supported"]


def test_namespace_is_identity_only_and_never_changes_public_behavior(tmp_path):
    episode = _temporal_with_available_reply()
    a = run_episode(ScriptedManager(ask_at=2), LiteralTestReader(), episode, 1, tmp_path / "a", namespace="methodA")
    b = run_episode(ScriptedManager(ask_at=2), LiteralTestReader(), episode, 1, tmp_path / "b", namespace="methodB")
    assert [d["decision_id"] for d in a["decisions"]] != [d["decision_id"] for d in b["decisions"]]
    assert [d["observation"] for d in a["decisions"]] == [d["observation"] for d in b["decisions"]]
    assert [d["seed"] for d in a["decisions"]] == [d["seed"] for d in b["decisions"]]
    assert a["dialogue"] == b["dialogue"] and a["metrics"] == b["metrics"]


def test_order_is_variable_and_dependencies_stay_coherent():
    bank = build_bank()
    neutral_positions = set()
    relevant_positions = set()
    for rows in bank.values():
        for episode in rows:
            for index, session in enumerate(episode["sessions"]):
                assert session["exchange"][-1]["turn_id"] == f"s{index+1}u1"
                if any(text in session["exchange"][-1]["text"] for text in ("paseo", "música", "foto", "tarde", "acompañarme")):
                    neutral_positions.add(index)
            facts = episode["private"]["worlds"][0]["facts"]
            relevant_positions.update(f["sequence"] for f in facts[:2])
            if episode["family"] in ("correction", "temporal", "context"):
                assert facts[0]["sequence"] < facts[-1]["sequence"]
    assert relevant_positions == neutral_positions == {0, 1, 2, 3}


def test_empty_support_sets_are_rejected():
    from research.budget_memory_env import validate_episode
    episode = make_episode("retention", 1, 1)
    episode["private"]["worlds"][0]["facts"][0]["support_sets"] = [[]]
    with pytest.raises(ValueError, match="nonempty"):
        validate_episode(episode)


def test_expired_interval_and_unknown_reply_are_not_current_memory_evidence(tmp_path):
    episode = _temporal_with_available_reply()
    episode["private"]["actual_world_id"] = episode["private"]["worlds"][0]["world_id"]
    interval = episode["private"]["worlds"][0]["facts"][-1]
    query = next(q.copy() for q in episode["queries"] if q["attribute"] == interval["attribute"])
    query["date"] = interval["valid_until_exclusive"]
    trace = run_episode(ScriptedManager(ask_at=2), LiteralTestReader(), episode, 1, tmp_path)
    full = full_history_reader_observation(trace, query)
    flags = score_response(episode, trace["dialogue"], full["slots"], query, interval["value"])
    assert not flags["accepted_by_user"] and not flags["memory_supported"]


def test_enumerated_public_states_cover_actual_trajectory(tmp_path):
    from research.budget_memory_env import canonical, iter_public_observations
    episode = _temporal_with_available_reply()
    trace = run_episode(ScriptedManager(ask_at=2, writes={0:'STORE_SLOT_1',1:'STORE_SLOT_2',2:'STORE_SLOT_1'}),
                        LiteralTestReader(), episode, 1, tmp_path)
    public = {(kind, canonical(obs), tuple(candidates)) for kind, obs, candidates in iter_public_observations(episode)}
    for decision in trace["decisions"]:
        assert ('action', canonical(decision["observation"]), tuple(decision["candidates"])) in public
    for query in trace["queries"]:
        observation = query["reader_observation"]
        assert ('answer',canonical(observation),tuple(observation['answer_menu'])) in public


def test_temporal_prior_is_genuinely_ambiguous_inside_and_outside_interval():
    from datetime import date, timedelta
    episode = _temporal_with_available_reply()
    worlds = episode['private']['worlds']
    assert {w['facts'][-1]['scope_kind'] for w in worlds} == {'session','interval','ongoing'}
    query = next(q.copy() for q in episode['queries'] if q['attribute']==worlds[0]['facts'][-1]['attribute'])
    until = worlds[0]['facts'][-1]['valid_until_exclusive']
    for day in ((date.fromisoformat(until)-timedelta(days=1)).isoformat(),until):
        query['date']=day
        assert len({resolve_world(w,query) for w in worlds}) == 2
