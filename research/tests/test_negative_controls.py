from copy import deepcopy

from research.src.data.build_bank import build_bank
from research.src.data.negative_controls import unaffected_attribute_controls
from research.src.policy.rule_controller import RuleController, expected_memory
from research.src.readers.symbolic import read
from research.src.scoring.evidence import audit_operation


def observe(episode):
    return {"memory": deepcopy(episode["initial_memory"]), "user_messages": deepcopy(episode["user_messages"]),
            "query": deepcopy(episode["current_query"]), "reuse_regime": "later_same_attribute_without_questions"}


def test_rule_answers_unaffected_drink_and_preserves_dinner_uncertainty():
    controls = unaffected_attribute_controls(build_bank(), roots=20)
    assert len({e["root_id"] for e in controls}) == 20
    for episode in controls:
        observation = observe(episode)
        policy = RuleController()
        assert policy.select(observation) == {"action": "PROCEED", "slot": None}
        assert policy.respond(observation)["value"] == "agua"
        operation = policy.write(observation)["operations"][0]
        assert operation["op"] == "MARK_UNRESOLVED"
        assert operation["record"]["attribute"] == "dinner_preference"
        assert audit_operation(operation, observation)
        assert read(expected_memory(observation), episode["future_query"])["value"] == "agua"


def test_clarifying_dinner_cannot_change_present_or_future_drink():
    episode = unaffected_attribute_controls(build_bank(), roots=20)[0]
    observation = observe(episode)
    policy = RuleController()
    before = policy.respond(observation)
    observation["user_messages"].append(deepcopy(episode["replies"]["TEMPORAL_SCOPE"]))
    assert policy.respond(observation)["value"] == before["value"] == "agua"
    assert read(expected_memory(observation), episode["future_query"])["value"] == "agua"
    operation = policy.write(observation)["operations"][0]
    assert operation["record"]["attribute"] == "dinner_preference"
    assert audit_operation(operation, observation)


def test_controls_keep_parent_groups_and_do_not_mutate_original_bank():
    original = build_bank()
    before = deepcopy(original)
    controls = unaffected_attribute_controls(original, roots=20)
    parent_ids = {(e["root_id"], e["latent_world_id"], e["realization_id"], e["split"]) for e in original}
    assert original == before
    for episode in controls:
        parent_realization = episode["realization_id"].removesuffix(":unaffected_drink")
        assert (episode["root_id"], episode["latent_world_id"], parent_realization, episode["split"]) in parent_ids
        assert episode["opportunity"]["eligible"] is False
        assert episode["opportunity"]["negative_control"] is True
        assert episode["opportunity"]["scope_slot"] == "TEMPORAL_SCOPE"
        assert episode["authoring"]["independent"] is False
