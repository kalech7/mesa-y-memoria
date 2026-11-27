from collections import defaultdict
from copy import deepcopy
from datetime import date
import hashlib
import json

import pytest

from research.src.data.development_v2 import build_development_v2, prepare_development_v2, STRATA
from research.src.data.semantic_partition_v2 import assert_semantic_split_isolation_v2, semantic_episode_id_v2


def test_new_bank_is_development_only_with_complete_paired_paraphrases():
    episodes = build_development_v2()
    roots = defaultdict(list)
    for episode in episodes:
        roots[episode["root_id"]].append(episode)
        assert episode["split"] == "validation"
        assert episode["data_role"] == "prospective_development_only"
        assert episode["authoring"]["human_authored"] is False
        assert episode["authoring"]["selection_uses_B1_results"] is False
        assert set(episode["replies"]) == {"TEMPORAL_SCOPE", "CURRENT_CONTEXT", "VALID_UNTIL", "GENERIC"}
        assert episode["current_query"]["attribute"] == episode["future_query"]["attribute"]
        assert 4 <= len(episode["initial_memory"]) <= 8
    assert len(roots) == 50 and len(episodes) == 120
    for variants in roots.values():
        worlds = defaultdict(list)
        for e in variants: worlds[e["latent_world_id"]].append(e)
        for pair in worlds.values():
            assert {e["realization_id"] for e in pair} == {"v0", "v1"}
            assert pair[0]["user_messages"][0]["text"] != pair[1]["user_messages"][0]["text"]
            for field in ("truth", "initial_memory", "current_query", "future_query", "replies"):
                assert pair[0][field] == pair[1][field]
    assert {e["authoring"]["stratum"] for e in episodes} == set(STRATA)


def test_seed_changes_order_not_content_or_root_identity():
    a, b = build_development_v2(seed=1), build_development_v2(seed=2)
    assert [e["root_id"] for e in a] != [e["root_id"] for e in b]
    assert {semantic_episode_id_v2(e) for e in a} == {semantic_episode_id_v2(e) for e in b}
    assert {e["root_id"] for e in a} == {e["root_id"] for e in b}


def test_literal_gold_covers_inclusive_last_day_exclusive_boundary_and_after():
    temporal = [e for e in build_development_v2() if e["authoring"]["stratum"] == "temporal_boundaries"]
    assert {e["authoring"]["boundary"] for e in temporal} == {"inside", "exclusive_end", "after"}
    for e in temporal:
        future, end = e["future_query"]["date"], e["truth"]["scope"]["valid_until_exclusive"]
        if e["authoring"]["boundary"] == "inside":
            assert future < end and e["truth"]["future_value"] == e["truth"]["new_value"]
        elif e["authoring"]["boundary"] == "exclusive_end":
            assert future == end and e["truth"]["future_value"] == e["truth"]["initial_value"]
        else:
            assert future > end and e["truth"]["future_value"] == e["truth"]["initial_value"]


def test_context_expiry_and_foreign_attribute_are_real_compositions():
    episodes = build_development_v2()
    contextual = [e for e in episodes if e["authoring"]["stratum"] == "new_context_interval"]
    assert {e["future_query"]["context"] for e in contextual} == {"casa", "oficina"}
    assert all(e["truth"]["scope"]["context"] == "casa" and e["truth"]["scope"]["valid_until_exclusive"] for e in contextual)
    for e in episodes:
        if e["authoring"]["stratum"] == "unaffected_future_attribute":
            assert e["opportunity"]["affected_attribute"] == "dinner_preference"
            assert e["future_query"]["attribute"] == "drink_preference"
            assert e["truth"]["current_value"] == e["truth"]["future_value"] == "agua"
        if e["authoring"]["stratum"] == "successive_supersessions":
            records = {r["memory_id"]:r for r in e["initial_memory"]}
            depth2 = [r for r in records.values() if r["supersedes"] and records[r["supersedes"][0]]["supersedes"]]
            assert len(depth2) == 1


def test_ambiguous_worlds_have_identical_public_input_and_distinct_resolving_replies():
    groups = defaultdict(list)
    for e in build_development_v2():
        if e["authoring"]["stratum"] == "ambiguous_temporal_boundary":groups[(e["root_id"],e["realization_id"])].append(e)
    for pair in groups.values():
        assert len(pair) == 2
        for field in ("initial_memory", "user_messages", "current_query", "future_query"):
            assert pair[0][field] == pair[1][field]
        assert pair[0]["truth"]["future_value"] != pair[1]["truth"]["future_value"]
        assert pair[0]["replies"]["TEMPORAL_SCOPE"] != pair[1]["replies"]["TEMPORAL_SCOPE"]


def test_prepare_snapshots_before_write_and_refuses_overwriting(tmp_path):
    history = tmp_path/"history";history.mkdir()
    # A different fixture identity and utterance; only snapshot behavior is tested.
    old = deepcopy(build_development_v2()[0])
    old["root_id"] = "historical_snapshot_fixture_root"
    old["user_messages"][0]["text"] = "Un caso anterior independiente."
    for split in ("dev_smoke", "train", "validation", "test_id", "test_ood", "test_external"):
        (history/f"{split}.jsonl").write_text(json.dumps(old)+"\n")
    before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in history.iterdir()}
    destination, report = tmp_path/"new_data", tmp_path/"new_report"
    result = prepare_development_v2(destination, report, historical_directory=history)
    assert result["roots"] == 50 and result["episodes"] == 120
    assert result["historical_bank_hashes_before"] == result["historical_bank_hashes_after"]
    for name, expected in before.items():
        assert hashlib.sha256((history/name).read_bytes()).hexdigest() == expected
        assert hashlib.sha256((report/"data_source_snapshot"/name).read_bytes()).hexdigest() == expected
    episodes = [json.loads(line) for line in (destination/"validation.jsonl").open()]
    assert_semantic_split_isolation_v2({"validation": episodes}, historical_splits={"old": [old]})
    with pytest.raises(FileExistsError):
        prepare_development_v2(destination, report, historical_directory=history)
