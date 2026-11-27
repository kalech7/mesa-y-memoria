import pytest

from research.src.artifacts import canonical
from research.src.memory.store import Store
from research.tesis_interactive_env import InteractiveEpisode, build_bank, reference_action


@pytest.fixture(scope="module")
def stories():
    return build_bank()["train"]


def play(story, directory, actions=()):
    env = InteractiveEpisode(story, directory)
    try:
        prefixes = [canonical(env.observe())]
        for action in actions:
            env.step(action)
            prefixes.append(None if env.done else canonical(env.observe()))
        while not env.done:
            env.step(reference_action(env.observe()))
        return prefixes, env.summary()
    finally:
        env.close()


def test_two_questions_reveal_world_but_do_not_change_today(stories, tmp_path):
    anchor = next(row for row in stories if row["scenario_family"] == "F4" and row["visits"][0]["style"] == "partial"
                  and row["second_visit_kind"] == "temporary")
    worlds = [row for row in stories if row["root_id"] == anchor["root_id"]]
    cases = []
    for i, story in enumerate(worlds):
        prefix, full = play(story, tmp_path / f"full{i}.sqlite", ["TEMPORAL_SCOPE", "TEMPORAL_SCOPE", "PROCEED"])
        _, short = play(story, tmp_path / f"short{i}.sqlite", ["TEMPORAL_SCOPE", "PROCEED"])
        assert [v["current_supported"] for v in full["visits"]] == [1, 1]
        assert [v["current_supported"] for v in short["visits"]] == [1, 1]
        assert [event["delivery"] for event in full["visits"][0]["events"]] == ["partial", "resolved"]
        assert short["future"][0]["supported"] == 0
        assert full["future"][0]["supported"] == 1
        assert full["future"][1]["kind"] == "unrelated_preference"
        assert full["future"][1]["supported"] == short["future"][1]["supported"] == 1
        assert full["visits"][1]["memory_before"] == full["visits"][0]["memory_after"]
        database = Store(tmp_path / f"full{i}.sqlite")
        try:
            assert database.list_records() == full["final_memory"]
        finally:
            database.close()
        cases.append((prefix, full["future"][0]["expected"]))
    left, right = cases
    assert left[0][:2] == right[0][:2]
    assert left[0][2] != right[0][2]
    assert left[1] != right[1]


def test_refusal_and_budget_are_public(stories, tmp_path):
    story = next(row for row in stories if row["scenario_family"] == "F4" and row["visits"][0]["style"] == "refused")
    env = InteractiveEpisode(story, tmp_path / "refusal.sqlite")
    try:
        assert "CURRENT_CONTEXT" in env.legal_actions
        env.step("TEMPORAL_SCOPE")
        assert env.legal_actions == ("PROCEED",)
        assert env.observe()["questions_remaining"] == 1
        assert "refused" not in canonical(env.observe())
        assert reference_action(env.observe()) == "PROCEED"
        with pytest.raises(ValueError, match="Illegal SELECT"):
            env.step("TEMPORAL_SCOPE")
    finally:
        env.close()


def test_session_only_expires_even_if_second_visit_is_the_same_day(stories, tmp_path):
    story = next(row for row in stories if row["scenario_family"] == "F4"
                 and row["visits"][0]["fact"]["scope"]["kind"] == "session"
                 and row["visits"][0]["style"] != "refused" and row["second_visit_kind"] == "reuse"
                 and row["visits"][0]["query"]["date"] == row["visits"][1]["query"]["date"])
    _, output = play(story, tmp_path / "same_day.sqlite")
    first, second = output["visits"]
    assert first["query"]["date"] == second["query"]["date"]
    assert first["query"]["session_id"] != second["query"]["session_id"]
    assert first["outcome"]["value"] != second["outcome"]["value"]
    assert [first["current_supported"], second["current_supported"]] == [1, 1]


def test_exclusive_expiry_reactivates_prior_preference(stories, tmp_path):
    story = next(row for row in stories if row["scenario_family"] == "F2" and row["visits"][0]["slot"] is None
                 and {"last_day_of_interval", "first_day_after_interval"} <= {q["kind"] for q in row["future_queries"]}
                 and row["second_visit_kind"] != "revision")
    _, output = play(story, tmp_path / "interval.sqlite")
    observations = {q["kind"]: q for q in output["future"]}
    assert observations["last_day_of_interval"]["response"]["value"] != observations["first_day_after_interval"]["response"]["value"]
    assert all(q["supported"] for q in observations.values())


def test_irrelevant_question_can_shut_down_later_clarification(stories, tmp_path):
    story = next(row for row in stories if row["impatient"] and row["visits"][0]["slot"] is not None
                 and row["visits"][1]["slot"] is not None)
    initial_choices = []
    for name, initial_actions in (("silent", ("PROCEED",)), ("wrong", ("GENERIC", "PROCEED"))):
        env = InteractiveEpisode(story, tmp_path / (name + ".sqlite"))
        try:
            for action in initial_actions:
                env.step(action)
            observation = env.observe()
            assert observation["visit"] == 2
            initial_choices.append((observation, env.legal_actions))
        finally:
            env.close()
    silent, wrong = initial_choices
    assert silent[0]["memory"] == wrong[0]["memory"]
    assert any("no puedo aclarar nada más" in m["text"] for m in wrong[0]["user_messages"])
    assert silent[1] != wrong[1] == ("PROCEED",)
