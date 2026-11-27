from copy import deepcopy

import pytest

from research.tesis_extended_bank import audit_development, build_development, group_roots
from research.tesis_extended_env import enumerate_world, evaluate, execute, simulate
from research.tesis_interactive_env import public_evidence


@pytest.fixture(scope="module")
def bank():
    return build_development(roots_per_family=1)


def test_root_keeps_latent_plan_and_partial_reply_private(bank):
    audit = audit_development(bank)
    assert audit["train"]["roots"] == audit["validation"]["roots"] == 4
    assert audit["train"]["worlds"] == audit["validation"]["worlds"] == 96
    worlds = next(iter(group_roots(bank["validation"]).values()))
    assert {world["plan_change"] for world in worlds} == {"confirmed", "canceled"}
    assert {world["visits"][0]["style"] for world in worlds} == {"direct", "partial"}
    observations = [execute(world, ()) for world in worlds]
    assert all(node == observations[0] for node in observations)


def test_partial_reply_leaves_a_choice_to_spend_or_reserve(bank):
    partial = next(row for row in bank["validation"]
                   if row["visits"][0]["style"] == "partial" and row["plan_change"] == "confirmed")
    ask = "TEMPORAL_SCOPE"
    observed = execute(partial, (ask,))
    assert observed.observation["questions_remaining"] == 1
    assert public_evidence(observed.observation)["clarification_state"] == "partial"
    assert ask in observed.legal
    follow_up = execute(partial, (ask, ask))
    assert follow_up.observation["questions_remaining"] == 0
    assert ask not in follow_up.legal
    with pytest.raises(ValueError, match="Illegal"):
        execute(partial, (ask, ask, "PROCEED", ask))

    continued = execute(partial, (ask, "PROCEED", "PROCEED"))
    assert continued.observation["visit"] == 3
    assert continued.observation["questions_remaining"] == 1
    assert ask in continued.legal
    outcome = simulate(partial, (ask, "PROCEED", "PROCEED", ask, "PROCEED"))
    assert outcome["current_supported"] == 1
    assert outcome["questions"] == 2
    assert outcome["questions_third"] == 1
    assert outcome["partial_replies"] == 1
    assert outcome["partial_followups"] == 0
    repeated = simulate(partial, (ask, ask, "PROCEED", "PROCEED", "PROCEED"))
    assert repeated["partial_followups"] == 1


def test_cancellation_changes_later_actions_and_preserves_expiry_boundary(bank):
    worlds = next(iter(group_roots(bank["validation"]).values()))
    canceled = next(row for row in worlds if row["plan_change"] == "canceled")
    confirmed = next(row for row in worlds if row["plan_change"] == "confirmed")
    assert execute(canceled, ("PROCEED",)) != execute(confirmed, ("PROCEED",))
    assert "TEMPORAL_SCOPE" in execute(confirmed, ("PROCEED", "PROCEED")).legal
    assert "TEMPORAL_SCOPE" not in execute(canceled, ("PROCEED", "PROCEED")).legal
    trace = execute(canceled, ("PROCEED", "PROCEED", "PROCEED"))
    write = trace["visits"][-1]["executed_operations"][0]
    assert write["record"]["scope"]["kind"] == "interval"
    last_day, expired = trace["future"][-2:]
    assert last_day["response"]["value"] == write["record"]["value"]
    assert expired["response"]["value"] != last_day["response"]["value"]
    assert last_day["supported"] == expired["supported"] == 1


def test_runtime_rejects_incompatible_shared_budget(bank):
    world = deepcopy(bank["train"][0])
    world["question_budget"] = 3
    with pytest.raises(ValueError, match="shared questions"):
        execute(world, ())


def test_question_removal_marks_an_incompatible_following_action(bank):
    row = next(world for world in bank["validation"]
               if world["visits"][0]["style"] == "direct"
               and world["visits"][0]["fact"]["scope"]["kind"] == "session")
    tree = enumerate_world({**row, "world_probability": 1.})

    def policy(observation, legal):
        if not observation["action_history"]:
            return "TEMPORAL_SCOPE"
        return "PROVISIONAL" if "PROVISIONAL" in legal else "PROCEED"

    metrics = evaluate([tree], policy, method="diagnostic")[0]["metrics"]
    assert metrics["questions"] == 1
    assert metrics["counterfactual_evaluable_questions"] == 0
