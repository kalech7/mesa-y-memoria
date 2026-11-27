"""Contract tests for controlled delivery faults, without inference or gold repair."""
from copy import deepcopy
import json

import pytest

from research.src.env.episode import EpisodeEnvironment
from research.src.env.imperfections import ImperfectionConfig, normalize_config, uniform
from research.src.env.user_simulator import UserSimulator, NON_RESPONSE, nonresponse_uniform
from research.src.policy.llm_controller import LLMController
from research.tests.test_environment import episode, record


def context_episode():
    value = episode()
    value["replies"]["TEMPORAL_SCOPE"]["text"] = "Como preferencia habitual desde ahora."
    value["replies"]["CURRENT_CONTEXT"] = {"turn_id": "u3", "text": "Estoy en casa."}
    value["current_query"]["context"] = "casa"
    return value


def test_zero_config_preserves_observations_replies_traces_and_persistence(tmp_path):
    plain = EpisodeEnvironment(episode(), tmp_path / "plain.db")
    zero = EpisodeEnvironment(episode(), tmp_path / "zero.db", imperfections={"seed": 87})
    try:
        assert plain.observe("select") == zero.observe("select")
        assert plain.ask("TEMPORAL_SCOPE") == zero.ask("TEMPORAL_SCOPE")
        for phase in ("select", "respond", "write"):
            assert plain.observe(phase) == zero.observe(phase)
        assert plain.private_trace == zero.private_trace
        plain.close_session()
        zero.close_session()
        assert plain.reopen_for_reuse() == zero.reopen_for_reuse()
        assert plain.store.hash() == zero.store.hash()
    finally:
        plain.dispose()
        zero.dispose()


def test_memory_fault_changes_delivery_not_sqlite_or_truth(tmp_path):
    original = episode()
    before = deepcopy(original)
    env = EpisodeEnvironment(original, tmp_path / "memory.db", imperfections={"memory_omission_rate": 1})
    try:
        assert env.observe("select")["memory"] == []
        assert env.store.list_records() == before["initial_memory"]
        assert env.private_trace["initial_memory"] == before["initial_memory"]
        assert env.private_trace["imperfections"]["retrieval"]["select:O2"]["omitted_ids"] == ["m1"]
        assert original == before
        env.close_session()
        assert env.reopen_for_reuse()["memory"] == before["initial_memory"]
        assert env.private_trace["final_memory"] == before["initial_memory"]
    finally:
        env.dispose()


def test_masks_shared_across_phases_interventions_and_policy_rollout_seeds(tmp_path):
    value = context_episode()
    value["initial_memory"] = [record(mid=f"m{i}") for i in range(40)]
    cfg = ImperfectionConfig(seed=513, memory_omission_rate=.4, partial_reply_rate=.5)
    a = EpisodeEnvironment(value, tmp_path / "a.db", imperfections=cfg)
    b = EpisodeEnvironment(value, tmp_path / "b.db", imperfections=cfg.to_dict())
    try:
        a.reset(rollout_seed=101)
        b.reset(rollout_seed=303)
        ids = lambda obs: [r["memory_id"] for r in obs["memory"]]
        expected = ids(a.observe("select"))
        assert 0 < len(expected) < 40
        assert ids(b.observe("select")) == expected
        assert a.ask("TEMPORAL_SCOPE") == b.ask("TEMPORAL_SCOPE")
        for phase in ("select", "respond", "write"):
            for intervention in (None, "I0", "I1", "I2"):
                assert ids(a.observe(phase, intervention=intervention)) == expected
        assert a.observe("respond", intervention="I1") == a.observe("respond", intervention="I2")
        assert a.observe("write", intervention="I0") == a.observe("write", intervention="I1")
    finally:
        a.dispose()
        b.dispose()


def test_masks_nested_by_rate_and_seed_changes_assignment():
    low = ImperfectionConfig(seed=13, memory_omission_rate=.2)
    high = ImperfectionConfig(seed=13, memory_omission_rate=.6)
    other = ImperfectionConfig(seed=29, memory_omission_rate=.2)
    draws = [uniform(low, "root", "s2_memory", str(i)) for i in range(100)]
    assert draws == [uniform(high, "root", "s2_memory", str(i)) for i in range(100)]
    assert draws != [uniform(other, "root", "s2_memory", str(i)) for i in range(100)]
    assert all(not (draw < .2) or draw < .6 for draw in draws)


def test_supported_partial_reveals_context_but_withholds_duration(tmp_path):
    value = context_episode()
    original = deepcopy(value)
    env = EpisodeEnvironment(value, tmp_path / "partial.db", imperfections={"partial_reply_rate": 1})
    try:
        observation = env.observe("select")
        assert "Estoy en casa" not in json.dumps(observation, ensure_ascii=False)
        reply = env.ask("TEMPORAL_SCOPE")
        assert reply == {"turn_id": "u3", "text": "Estoy en casa."}
        assert reply["text"] == original["replies"]["CURRENT_CONTEXT"]["text"]
        assert env.private_trace["user_response_status"] == "partial"
        assert env.private_trace["imperfections"]["reply"]["source_slot"] == "CURRENT_CONTEXT"
        assert "habitual" not in json.dumps(env.observe("respond"), ensure_ascii=False)
        assert value == original
    finally:
        env.dispose()


def test_partial_does_not_cut_condition_into_false_permanent_preference():
    replies = {"TEMPORAL_SCOPE": {"turn_id": "u3", "text": "Prefiero pasta; solo para hoy."}}
    sim = UserSimulator(replies, root_id="r", imperfections={"partial_reply_rate": 1})
    assert sim.reply("TEMPORAL_SCOPE")["text"] == NON_RESPONSE["text"]
    assert sim.last_response_status == "unavailable"
    assert sim.last_reply_audit["reason"] == "no_safe_partial_fragment"


def test_partial_request_does_not_relabel_full_context_reply_as_partial():
    replies = {"CURRENT_CONTEXT": {"turn_id": "u3", "text": "Estoy en casa."}}
    sim = UserSimulator(replies, root_id="r", imperfections={"partial_reply_rate": 1})
    assert sim.reply("CURRENT_CONTEXT")["text"] == NON_RESPONSE["text"]
    assert sim.last_response_status == "unavailable"


@pytest.mark.parametrize("source", [
    {"turn_id": "u3", "text": "Estoy en casa.", "availability": "unavailable"},
    {"turn_id": "u3", "text": "No estoy seguro. Estoy en casa."},
    "No sé si estoy seguro. Estoy en casa.",
])
def test_partial_cannot_promote_unavailable_or_uncertain_context(source):
    replies = {"TEMPORAL_SCOPE": {"turn_id": "u3", "text": "Solo para hoy."},
               "CURRENT_CONTEXT": source}
    original = deepcopy(replies)
    sim = UserSimulator(replies, root_id="r", imperfections={"partial_reply_rate": 1})
    assert sim.reply("TEMPORAL_SCOPE")["text"] == NON_RESPONSE["text"]
    assert sim.last_response_status == "unavailable"
    assert sim.last_reply_audit["reason"] == "no_safe_partial_fragment"
    assert replies == original


def test_unavailable_preserves_original_reply_namespace_and_is_repeatable():
    sim = UserSimulator({"TEMPORAL_SCOPE": {"turn_id": "unique_user_reply", "text": "Solo para hoy."}},
                        root_id="r", imperfections={"unavailable_reply_rate": 1})
    assert sim.reply("TEMPORAL_SCOPE") == {"turn_id": "unique_user_reply", "text": NON_RESPONSE["text"]}
    assert sim.reply("TEMPORAL_SCOPE") == {"turn_id": "unique_user_reply", "text": NON_RESPONSE["text"]}
    assert sim.last_reply_audit["reason"] == "configured_unavailable"


def test_legacy_nonresponse_assignment_unchanged_with_new_faults():
    root = next(str(i) for i in range(100) if nonresponse_uniform(str(i), "TEMPORAL_SCOPE") < .4)
    sim = UserSimulator(context_episode()["replies"], root_id=root, nonresponse_rate=.4,
                        imperfections={"partial_reply_rate": 1})
    assert sim.reply("TEMPORAL_SCOPE") == NON_RESPONSE
    assert sim.last_reply_audit["reason"] == "legacy_nonresponse"


def test_fault_audit_and_hidden_world_never_appear_in_actor_prompt(tmp_path):
    value = context_episode()
    env = EpisodeEnvironment(value, tmp_path / "private.db", imperfections={"memory_omission_rate": 1,
                                                                              "partial_reply_rate": 1})
    other_value = deepcopy(value)
    other_value["truth"] = {"future_value": "HIDDEN_WORLD_B"}
    other_value["latent_world_id"] = "HIDDEN_WORLD_B"
    other = EpisodeEnvironment(other_value, tmp_path / "other.db", imperfections=env.imperfections)
    try:
        obs = env.observe("select")
        assert obs == other.observe("select")
        prompt = LLMController(object()).prompt("select", obs)
        assert set(obs) == {"memory", "query", "user_messages", "reuse_regime"}
        for private in ("omitted_ids", "partial_reply_rate", "HIDDEN_WORLD_B", "source_slot", "imperfections"):
            assert private not in prompt
    finally:
        env.dispose()
        other.dispose()


@pytest.mark.parametrize("config", [
    {"seed": True}, {"seed": 1.2}, {"memory_omission_rate": -1}, {"partial_reply_rate": float("nan")},
    {"partial_reply_rate": True}, {"unavailable_reply_rate": 1.1},
    {"partial_reply_rate": .6, "unavailable_reply_rate": .6}, {"gold": "leak"},
])
def test_invalid_configs_rejected(config):
    with pytest.raises(ValueError):
        normalize_config(config)
