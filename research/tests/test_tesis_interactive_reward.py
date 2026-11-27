from copy import deepcopy

import pytest

from research.tesis_interactive_reward import RewardConfig, aggregate, metrics, paired_group, reward


def trace(*, future=1, questions=0, commitment=0):
    return {"root_id": "r", "world_id": "w", "split": "train", "family": "duration",
            "questions": questions, "visits": [{"current_supported": int(not commitment), "questions": questions,
                                                "unsupported_commitments": commitment},
                                               {"current_supported": 1, "questions": 0}],
            "future": [{"supported": future, "correct": 1}]}


def test_delayed_outcome_changes_preferred_action_when_current_is_identical():
    proceed, ask = trace(future=0), trace(questions=2)
    group = paired_group([proceed, ask])
    assert group["future_varies_with_current_fixed"]
    assert group["preferred_trajectories"] == {"longitudinal": [1], "immediate": [0]}
    assert group["horizon_changes_preference"]
    assert reward(ask)["total"] - reward(ask, horizon="immediate")["total"] == pytest.approx(1.5)
    assert metrics(proceed)["future_correct"] == 1 and metrics(proceed)["future_supported"] == 0


def test_redundant_or_unanswered_questions_cost_the_same_and_do_not_collect_bonus():
    clean = trace()
    unproductive = trace(questions=1)
    unproductive["visits"][0].update(redundant_questions=1, failed_questions=1)
    informative = trace(questions=1)
    informative["visits"][0]["informative_questions"] = 1
    assert reward(clean)["total"] - reward(unproductive)["total"] == pytest.approx(.08)
    assert reward(informative) == reward(unproductive)
    assert paired_group([clean, unproductive])["only_penalties_vary"]


def test_invalid_unsupported_writes_and_unsupported_commitments_pay_actual_counts():
    invalid = trace(commitment=1)
    invalid["visits"][0].update(invalid_outputs=2, unsupported_writes=1, omitted_confirmed_updates=1)
    result = reward(invalid)
    assert result["total"] == pytest.approx(2.0 - .50 - .75 - .35)
    assert metrics(invalid)["strict_joint"] == 0
    omitted = trace()
    omitted["visits"][0]["omitted_confirmed_updates"] = 1
    assert reward(omitted, horizon="immediate") == reward(trace(), horizon="immediate")


def test_worlds_and_families_are_weighted_independently_of_realizations():
    zero1 = {**trace(future=0), "realization_id": "a"}
    zero2 = {**trace(future=0), "realization_id": "b"}
    one = {**trace(), "world_id": "alternate"}
    separate_root = {**trace(), "root_id": "different", "family": "stable"}
    result = aggregate([zero1, zero2, one, separate_root])
    assert result["episodes"] == 4
    assert result["roots"] == 2
    assert result["by_root"]["r"]["future_supported"] == .5
    assert result["macro"]["future_supported"] == .75
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate([zero1, deepcopy(zero1)])


def test_reward_rejects_impossible_or_incomplete_traces():
    invalid = trace()
    invalid["future"][0]["correct"] = 0
    with pytest.raises(ValueError, match="factually correct"):
        reward(invalid)
    with pytest.raises(ValueError, match="complete trace"):
        reward({**trace(), "visits": []})
    with pytest.raises(ValueError, match="Unknown reward horizon"):
        reward(trace(), horizon="reward-by-label")
    with pytest.raises(ValueError, match="coefficient"):
        RewardConfig(question_cost=float("nan"))
    with pytest.raises(ValueError, match="share an initial root"):
        paired_group([trace(), {**trace(), "root_id": "another"}])


def test_unsupported_guess_never_beats_deferring_with_the_same_future():
    defer, guess = trace(future=0), trace(future=0, commitment=1)
    for row in (defer, guess):
        row["visits"][0]["current_supported"] = 0
    comparison = paired_group([defer, guess])
    assert comparison["preferred_trajectories"] == {"longitudinal": [0], "immediate": [0]}
    assert reward(guess)["total"] == pytest.approx(reward(defer)["total"] - .35)


def test_exhausting_question_budget_without_utility_strictly_reduces_return():
    start = trace(questions=0)
    all_questions = trace(questions=2)
    all_questions["visits"][0]["failed_questions"] = 2
    assert reward(start)["total"] - reward(all_questions)["total"] == pytest.approx(.16)
    all_questions["visits"][0]["failed_questions"] = 3
    with pytest.raises(ValueError, match="failed_questions"):
        reward(all_questions)
