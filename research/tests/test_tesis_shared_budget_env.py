from copy import deepcopy

import pytest

from research.src.artifacts import canonical
from research.tesis_interactive_env import InteractiveEpisode, build_bank
from research import tesis_shared_budget_env as shared_budget
from research.tesis_shared_budget_env import (
    CLOSE_ACTIONS,
    SharedBudgetEpisode,
    decision_state,
    rule_action,
)


@pytest.fixture(scope="module")
def stories():
    return build_bank()["train"]


def test_one_question_is_shared_between_visits(stories):
    story = next(row for row in stories if row["visits"][0]["slot"] == "TEMPORAL_SCOPE"
                 and row["visits"][0]["style"] == "direct")
    spent = SharedBudgetEpisode(story)
    saved = SharedBudgetEpisode(story)
    try:
        assert rule_action(spent.observe(), spent.legal_actions) == "TEMPORAL_SCOPE"
        spent.step("TEMPORAL_SCOPE")
        assert spent.observe()["questions_remaining"] == 0
        assert spent.legal_actions == CLOSE_ACTIONS
        second, done = spent.step("PROCEED")
        assert not done and second["visit"] == 2
        assert second["questions_remaining"] == 0
        assert second["asked_slots"] == []
        assert spent.legal_actions == CLOSE_ACTIONS
        with pytest.raises(ValueError, match="Illegal shared-budget action"):
            spent.step("TEMPORAL_SCOPE")
        spent.step("PROCEED")
        assert spent.summary()["questions"] == 1

        second, done = saved.step("PROCEED")
        assert not done and second["questions_remaining"] == 1
        assert "TEMPORAL_SCOPE" in saved.legal_actions
        saved.step("TEMPORAL_SCOPE")
        saved.step("PROCEED")
        assert saved.summary()["questions"] == 1
    finally:
        spent.close()
        saved.close()


def test_provisional_response_preserves_memory_and_defer_withholds_answer(stories):
    story = next(row for row in stories if row["scenario_family"] == "F1")
    traces = {}
    for action in CLOSE_ACTIONS:
        env = SharedBudgetEpisode(story)
        try:
            second, _ = env.step(action)
            assert second["questions_remaining"] == 1
            env.step("PROCEED")
            traces[action] = env.summary()["visits"][0]
        finally:
            env.close()
    assert traces["PROCEED"]["memory_after"] != traces["PROCEED"]["memory_before"]
    provisional = traces["PROVISIONAL"]
    assert provisional["outcome"] == traces["PROCEED"]["outcome"]
    assert provisional["current_supported"] == 1
    assert provisional["session_only_recommendation"]
    assert provisional["memory_after"] == provisional["memory_before"]
    deferred = traces["DEFER"]
    assert deferred["outcome"]["mode"] == "DEFER"
    assert deferred["current_supported"] == 0
    assert deferred["memory_after"] == deferred["memory_before"]


def test_stock_commit_outcomes_and_public_ambiguity_are_preserved(stories):
    target = next(row for row in stories if row["visits"][0]["slot"] == "TEMPORAL_SCOPE"
                  and row["visits"][0]["style"] == "direct")
    twins = [deepcopy(row) for row in stories if row["root_id"] == target["root_id"]]
    assert len(twins) == 2
    prefixes, replies = [], []
    for story in twins:
        story["visits"][0]["strategy_cue"] = "ready_regular"
        shared = SharedBudgetEpisode(story)
        stock = InteractiveEpisode(story)
        try:
            prefixes.append(canonical(shared.observe()))
            assert decision_state(shared.observe())["strategy_cue"] == "ready_regular"
            for action in ("TEMPORAL_SCOPE", "PROCEED", "PROCEED"):
                shared.step(action)
                stock.step(action)
                if action == "TEMPORAL_SCOPE":
                    replies.append(canonical(shared.observe()))
            extended, original = shared.summary(), stock.summary()
            assert extended["final_memory"] == original["final_memory"]
            assert [v["outcome"] for v in extended["visits"]] == [v["outcome"] for v in original["visits"]]
            assert [q["supported"] for q in extended["future"]] == [q["supported"] for q in original["future"]]
            assert all(not query["expired_reuse"] for query in extended["future"])
        finally:
            shared.close()
            stock.close()
    assert prefixes[0] == prefixes[1]
    assert replies[0] != replies[1]


def test_verifier_ablation_replays_same_writer_proposal(stories, monkeypatch):
    story = next(row for row in stories if row["scenario_family"] == "F1")
    first_session = story["visits"][0]["query"]["session_id"]
    write = shared_budget.RuleController.write

    def wrong_citation(controller, observation):
        result = write(controller, observation)
        if observation["query"]["session_id"] == first_session:
            wrong = observation["user_messages"][0]
            for operation in result["operations"]:
                if "record" in operation:
                    operation["record"].update(
                        source_turn_ids=[wrong["turn_id"]],
                        evidence_quotes=[{"turn_id": wrong["turn_id"], "text": wrong["text"]}],
                    )
        return result

    monkeypatch.setattr(shared_budget.RuleController, "write", wrong_citation)
    shared_budget._fixed_decision.cache_clear()
    traces = {}
    try:
        for mode in ("veto", "off"):
            env = SharedBudgetEpisode(story, verifier=mode)
            try:
                env.step("PROCEED")
                env.step("PROCEED")
                traces[mode] = env.summary()["visits"][0]
            finally:
                env.close()
    finally:
        shared_budget._fixed_decision.cache_clear()
    assert traces["veto"]["outcome"] == traces["off"]["outcome"]
    assert traces["veto"]["rejected_proposals"] == 1
    assert traces["veto"]["unsupported_writes"] == 0
    assert traces["veto"]["memory_before"] == traces["veto"]["memory_after"]
    assert traces["off"]["rejected_proposals"] == 0
    assert traces["off"]["unsupported_writes"] == 1
    assert traces["off"]["memory_before"] != traces["off"]["memory_after"]
