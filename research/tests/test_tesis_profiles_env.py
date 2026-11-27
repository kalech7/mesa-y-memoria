from copy import deepcopy

import pytest

from research.tesis_profiles_bank import audit_development, build_development, group_roots
from research.tesis_profiles_env import execute, public_features, rule_action, simulate


@pytest.fixture(scope="module")
def bank():
    return build_development()


def test_profiles_are_disjoint_and_actor_timing_ignores_private_labels(bank):
    audit = audit_development(bank)
    assert audit["train"]["profiles"] == 24
    assert audit["validation"]["profiles"] == 12
    assert set(audit["train"]["reply_styles"]) == {"partial", "direct", "deferred", "refused"}
    assert set(audit["validation"]["question_budgets"]) == {1, 2}
    worlds = next(iter(group_roots(bank["validation"]).values()))
    node = execute(worlds[0], ())
    assert all(execute(world, ()) == node for world in worlds)
    poisoned = {**node.observation, "strategy_cue": "PRIVATE", "profile_id": "PRIVATE",
                "future_queries": [{"expected": "PRIVATE"}]}
    assert public_features(node.observation, node.legal) == public_features(poisoned, node.legal)


@pytest.mark.parametrize("plan,returning", [("canceled", 2), ("changed", 3)])
def test_deferred_reply_requires_asking_and_arrives_only_in_later_visit(bank, plan, returning):
    story = next(row for row in bank["validation"]
                 if row["scenario_family"] == "deferred_recontact" and row["plan_change"] == plan
                 and row["visits"][0]["fact"]["scope"]["kind"] == "ongoing")
    ask = "TEMPORAL_SCOPE"
    first = execute(story, (ask,))
    assert "TEMPORAL_SCOPE" not in first.legal
    assert not public_features(first.observation)["observed_recontact"]
    for visit in range(2, 4):
        asked = execute(story, (ask,) + ("PROCEED",) * (visit - 1))
        unasked = execute(story, ("PROCEED",) * (visit - 1))
        assert not public_features(unasked.observation)["observed_recontact"]
        assert public_features(asked.observation)["observed_recontact"] == (visit == returning)
    path = (ask,)
    while hasattr(node := execute(story, path), "legal"):
        path += (rule_action(node.observation, node.legal),)
    assert len(node["recontacts"]) == 1
    assert node["future"][0]["supported"] == 1
    assert simulate(story, path)["delivered_recontacts"] == 1
    assert execute(story, ("PROCEED",) * 3)["future"][0]["supported"] == 0


def test_corrected_preference_is_retained_and_a_quoted_third_party_is_not(bank):
    story = next(row for row in bank["validation"]
                 if row["scenario_family"] == "direct_routine" and row["plan_change"] == "canceled"
                 and row["question_budget"] == 2 and row["visits"][1]["fact"]["scope"]["kind"] == "ongoing")
    path = ("TEMPORAL_SCOPE", "PROCEED", "TEMPORAL_SCOPE", "PROCEED", "PROCEED")
    trace = execute(story, path)
    assert all(visit["current_supported"] for visit in trace["visits"])
    writes = [operation["record"] for visit in trace["visits"] for operation in visit["executed_operations"]
              if "record" in operation]
    assert [record["value"] for record in writes] == [visit["fact"]["value"] for visit in story["visits"]]
    assert all("sobrina" not in quote["text"] for record in writes for quote in record["evidence_quotes"])


def test_location_query_becomes_supported_only_after_a_location_reply(bank):
    story = next(row for row in bank["validation"]
                 if row["scenario_family"] == "direct_unstable" and row["plan_change"] == "changed"
                 and "oficina" in row["visits"][2]["answer"])
    step = execute(story, ("PROCEED", "PROCEED"))
    assert "CURRENT_CONTEXT" in step.legal
    blocked = execute(story, ("PROCEED",) * 3)
    addressed = execute(story, ("PROCEED", "PROCEED", "CURRENT_CONTEXT", "PROCEED"))
    assert blocked["visits"][2]["current_supported"] == 0
    assert addressed["visits"][2]["current_supported"] == 1
    assert addressed["visits"][2]["effective_query"]["context"] == "oficina"


def test_backward_recontact_is_rejected(bank):
    story = deepcopy(next(row for row in bank["validation"] if row["scenario_family"] == "deferred_recontact"))
    story["visits"][0]["recontact"]["visit"] = 1
    with pytest.raises(ValueError, match="later visit"):
        execute(story, ("TEMPORAL_SCOPE",))
