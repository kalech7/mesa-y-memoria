from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from research.src.data.build_bank import build_bank, teacher_trajectories, import_independent_bank
from research.src.data.audit_splits import actor_prefix, audit_splits, audit_human_reviews, prepare_human_audit
from research.src.policy.rule_controller import RuleController, compatible_worlds, expected_memory
from research.src.analysis.paired_statistics import (
    aggregate_rows, binomial_interval, paired_bootstrap, select_checkpoint,
)
from research.src.readers.symbolic import read


def obs(episode, reply=False):
    value = {"memory": deepcopy(episode["initial_memory"]),
             "user_messages": deepcopy(episode["user_messages"]),
             "query": deepcopy(episode["current_query"]), "reuse_regime": "later_without_questions"}
    if reply:
        slot = episode["truth"]["decisive_slot"]
        if slot:
            value["user_messages"].append(deepcopy(episode["replies"][slot]))
    return value


def test_t01_root_balance_and_leakage():
    episodes = build_bank()
    report = audit_splits({"dev_smoke": episodes})
    assert report["partitions"]["dev_smoke"]["roots"] == 40
    assert set(report["partitions"]["dev_smoke"]["families"].values()) == {8}
    clone = deepcopy(episodes[0])
    clone["split"] = "train"
    with pytest.raises(ValueError, match="ROOT_SPLIT_LEAKAGE"):
        audit_splits({"dev_smoke": episodes, "train": [clone]})
    with pytest.raises(ValueError, match="INDEPENDENT_AUTHORING_REQUIRED"):
        build_bank("test_ood")


def test_t02_t03_counterfactual_prefix_and_selective_revelation():
    episodes = [e for e in build_bank() if e.get("opportunity", {}).get("eligible")]
    roots = {e["root_id"] for e in episodes}
    for root in roots:
        pair = [e for e in episodes if e["root_id"] == root]
        assert len(pair) == 2
        assert actor_prefix(pair[0]) == actor_prefix(pair[1])
        assert pair[0]["replies"]["TEMPORAL_SCOPE"] != pair[1]["replies"]["TEMPORAL_SCOPE"]
        assert pair[0]["replies"]["GENERIC"] == pair[1]["replies"]["GENERIC"]


def test_t06_teacher_never_uses_latent_scope_before_reply():
    episodes = [e for e in build_bank() if e.get("opportunity", {}).get("eligible")]
    teacher = RuleController()
    for episode in episodes:
        observation = obs(episode)
        assert teacher.select(observation) == {"action": "ASK", "slot": "TEMPORAL_SCOPE"}
        assert teacher.respond(observation)["value"] == episode["truth"]["current_value"]
        operation = teacher.write(observation)["operations"][0]
        assert operation["op"] == "MARK_UNRESOLVED"
        assert operation["record"]["scope"]["kind"] == "unknown"
        assert len({w["value"] for w in compatible_worlds(observation, episode["future_query"])}) == 2
    demos = list(teacher_trajectories(episodes[:2]))
    assert {d.get("continuation") for d in demos} >= {"normal", "irrelevant", "nonresponse"}


def test_observable_solver_covers_five_families_and_preserves_scopes():
    policy = RuleController()
    for episode in build_bank():
        observation = obs(episode, reply=True)
        assert policy.respond(observation)["value"] == episode["truth"]["current_value"]
        future = read(expected_memory(observation), episode["future_query"])
        assert future["mode"] == "ANSWER", (episode["scenario_family"], future)
        assert future["value"] == episode["truth"]["future_value"]
        if episode["scenario_family"] == "F3":
            assert policy.write(observation) == {"operations": [{"op": "NOOP"}]}
        if episode["scenario_family"] == "F2":
            assert policy.write(observation)["operations"][0]["op"] == "ADD"
        if episode["scenario_family"] == "F4" and not episode["opportunity"]["eligible"]:
            assert policy.select(observation) == {"action": "PROCEED", "slot": None}


def test_o1_loss_is_uncertainty_not_hidden_reconstruction():
    episode = next(e for e in build_bank() if e["scenario_family"] == "F0")
    observation = obs(episode)
    target = [r for r in observation["memory"] if r["attribute"] == "dinner_preference"]
    observation["memory"] = [max(target, key=lambda r: r["recorded_at"])]
    assert RuleController().respond(observation)["mode"] == "DEFER"


def test_t07_missing_human_audit_is_pending():
    assert audit_human_reviews([])["status"] == "pending_human_audit"
    packet = prepare_human_audit(build_bank())
    assert packet["status"] == "pending_human_audit"
    assert packet["reviewer_packets"] == {}


def test_independent_import_does_not_certify_authorship_or_semantics(tmp_path):
    # A test fixture exercising provenance validation is not external evidence.
    episode = deepcopy(build_bank()[0])
    episode["split"] = "test_external"
    path = tmp_path / "external_fixture.json"
    path.write_text(json.dumps([episode]))
    with pytest.raises(ValueError, match="INDEPENDENT_AUTHORSHIP_ATTESTATION_REQUIRED"):
        import_independent_bank(path, "test_external")
    episode["authoring"] = {"independent": True, "author_id": "fixture_author"}
    path.write_text(json.dumps([episode]))
    imported = import_independent_bank(path, "test_external", generator_author_id="different_fixture_author")
    assert imported["audit"]["semantic_audit_status"] == "pending_independent_semantic_audit"
    assert imported["audit"]["human_audit_status"] == "pending_human_audit"


def test_generated_episodes_and_teacher_outputs_match_json_schemas():
    schema_lib = pytest.importorskip("jsonschema")
    root = Path(__file__).resolve().parents[1]
    episode_schema = json.loads((root / "schemas" / "episode.schema.json").read_text())
    action_schema = json.loads((root / "schemas" / "action.schema.json").read_text())
    validator = schema_lib.Draft202012Validator
    episodes = build_bank()
    for episode in episodes:
        validator(episode_schema, format_checker=schema_lib.FormatChecker()).validate(episode)
    for demonstration in teacher_trajectories(episodes):
        validator(action_schema, format_checker=schema_lib.FormatChecker()).validate(demonstration["action"])
        assert json.loads(demonstration["completion"]) == demonstration["action"]
        assert demonstration["split"] == "dev_smoke"


def _rows():
    return [{"method": method, "seed": seed, "root_id": str(root), "scenario_family": "F" + str(root % 5),
             "condition": "O2", "latent_world_id": "w1", "realization_id": "v1",
             "future_justified_success": float(method == "B2-L")}
            for method in ("B1", "B2-L") for seed in (101, 202, 303) for root in range(10)]


def test_t29_duplicate_realizations_do_not_add_independent_roots():
    rows = _rows()
    plain = paired_bootstrap(rows, "B2-L", "B1", resamples=40)
    duplicated = paired_bootstrap(rows + rows, "B2-L", "B1", resamples=40)
    assert plain == duplicated
    assert plain["independent_roots"] == 10
    assert plain["estimate"] == 1
    assert len(aggregate_rows(rows + rows)) == 60
    with pytest.raises(ValueError, match="PAIRED_ROOTS_REQUIRED"):
        paired_bootstrap(rows[:-1], "B2-L", "B1", resamples=5)


def test_rare_event_exact_limits_and_step_zero():
    interval = binomial_interval(0, 100)
    assert interval["interval"][0] == 0
    assert 0.03 < interval["interval"][1] < 0.04
    assert math.isclose(interval["interval"][1], 1 - .025 ** .01)
    checkpoints = [{"step": 0, "utility": 1}, {"step": 10, "utility": 1}]
    assert select_checkpoint(checkpoints)["step"] == 0
    with pytest.raises(ValueError, match="STEP_ZERO_REQUIRED"):
        select_checkpoint(checkpoints[1:])
