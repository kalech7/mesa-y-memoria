from copy import deepcopy

import pytest

from research.tesis_profiles_learning import IMMEDIATE, RL, SFT, fit_from_trees, load_policies
from research.tesis_shared_budget_planner import Decision, WorldTree


def features(observation, legal):
    return observation["visible_reply"]


def fixed(observation, legal):
    return "PASS" if "PASS" in legal else legal[0]


def world(urgent, *, split="train"):
    def decision(prefix, reply, legal):
        return Decision({"visit": len(prefix) + 1, "action_history": prefix,
                         "visible_reply": reply}, legal)

    def outcome(longitudinal, immediate):
        return {"longitudinal": longitudinal, "immediate": immediate}

    choices = {
        (): decision((), "opening", ("ASK_NOW", "WAIT")),
        ("ASK_NOW",): decision(("ASK_NOW",), "spent", ("PASS",)),
        ("WAIT",): decision(("WAIT",), "urgent" if urgent else "routine", ("PASS", "ASK_LATER")),
    }
    outcomes = {}
    for prefix, longitudinal, immediate in (
        (("ASK_NOW", "PASS"), .70, .90),
        (("WAIT", "PASS"), .45 if urgent else .80, .80),
        (("WAIT", "ASK_LATER"), .95 if urgent else .60, .80),
    ):
        signal = "urgent" if urgent else "routine"
        third = f"third_visit_after_{signal}" if prefix[0] == "WAIT" else "third_visit_after_ask"
        choices[prefix] = decision(prefix, third, ("FINISH",))
        outcomes[(*prefix, "FINISH")] = outcome(longitudinal, immediate)
    return WorldTree({"root_id": "same-public-root", "world_id": str(urgent),
                      "world_probability": .75 if urgent else .25, "split": split},
                     choices, outcomes, 12)


def test_matched_logged_controllers_distinguish_fixed_continuation_and_bellman(tmp_path):
    urgent, routine = world(True), world(False)
    study = fit_from_trees([urgent, routine], public_features=features, fixed_policy=fixed)
    start = urgent.decisions[()]
    assert study.policies[SFT](start.observation, start.legal) == "ASK_NOW"
    assert study.policies[RL](start.observation, start.legal) == "WAIT"
    assert study.policies[IMMEDIATE](start.observation, start.legal) == "ASK_NOW"
    assert study.policies[RL].action_values(start.observation, start.legal)["WAIT"] == pytest.approx(.9125)
    assert study.policies[SFT].action_values(start.observation, start.legal)["WAIT"] == pytest.approx(.5375)

    info = study.telemetry
    assert info["visits_in_train"] == [1, 2, 3]
    assert info["stochastic_state_action_pairs"] >= 1
    assert info["transition_rows"] == sum(len(node.legal) for item in (urgent, routine)
                                           for node in item.decisions.values())
    assert list(info["arms"].values()).count(info["arms"][RL]) == 3

    artifact = tmp_path / "trained.json"
    study.save(artifact)
    loaded = load_policies(artifact, public_features=features, fallback=fixed)[RL]
    assert loaded.action_values(start.observation, start.legal) == study.policies[RL].action_values(
        start.observation, start.legal)
    novel = {**deepcopy(start.observation), "visible_reply": "unseen"}
    assert loaded.action_values(novel, start.legal) is None
    assert loaded(novel, start.legal) == "ASK_NOW"
    assert loaded.coverage(reset=True)["fallback_decision_visits"] == 1
    assert loaded.coverage()["decision_visits"] == 0


def test_reserved_trees_cannot_fit_public_controllers():
    with pytest.raises(ValueError, match="TRAIN"):
        fit_from_trees([world(False, split="reserved")], public_features=features, fixed_policy=fixed)
