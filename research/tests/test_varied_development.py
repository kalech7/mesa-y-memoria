from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json

import pytest

from research.varied_development import (
    build_varied_development, diversity_plan, prepare_varied_development,
    MEMORY_COUNTS, STRATA, DISHES, DRINKS, BREAKFASTS,
)
from research.src.data.semantic_partition_v3 import semantic_episode_id_v3


def by_root(rows):
    groups = defaultdict(list)
    for row in rows:groups[row["root_id"]].append(row)
    return groups


def test_fifty_roots_sixty_world_cases_with_explicit_development_provenance():
    rows = build_varied_development();groups = by_root(rows)
    assert len(rows) == 60 and len(groups) == 50
    assert Counter(v[0]["authoring"]["stratum"] for v in groups.values()) == {s:10 for s in STRATA}
    for row in rows:
        assert row["split"] == "validation"
        assert row["data_role"] == "prospective_development_only"
        assert row["realization_id"] == "v0"
        assert row["authoring"]["provided_history_model_generated"] is False
        assert row["authoring"]["model_results_used_for_design"] is False
        assert row["authoring"]["principal_queries_evaluated"] == 1
        assert row["current_query"]["attribute"] == row["future_query"]["attribute"]


def test_memory_count_distribution_and_short_local_ids():
    roots = {rid:variants[0] for rid,variants in by_root(build_varied_development()).items()}
    assert Counter(len(r["initial_memory"]) for r in roots.values()) == Counter(MEMORY_COUNTS)
    for row in roots.values():
        assert [m["memory_id"] for m in row["initial_memory"]] == [f"m{i+1}" for i in range(len(row["initial_memory"]))]


def test_real_prior_user_statements_support_every_initial_memory_quote():
    prior_counts = set()
    for row in build_varied_development():
        sessions = row["prior_sessions"];prior_counts.add(len(sessions))
        assert 1 <= len(sessions) <= 3
        sources = {m["turn_id"]:m for s in sessions for m in s["messages"]}
        assert len(sources) == len(row["initial_memory"])
        for memory in row["initial_memory"]:
            assert memory["recorded_at"] < row["current_query"]["date"]
            for quote in memory["evidence_quotes"]:
                assert quote["turn_id"] in memory["source_turn_ids"]
                assert sources[quote["turn_id"]]["role"] == "user"
                assert quote["text"] == sources[quote["turn_id"]]["text"]
    assert prior_counts == {1,2,3}


def test_dialogue_counts_roles_main_query_and_new_reply_identity():
    lengths, question_positions = set(), set()
    for row in build_varied_development():
        messages=row["user_messages"];lengths.add(len(messages))
        assert 2 <= len(messages) <= 8
        assert messages[-1]["role"] == "user" and "?" in messages[-1]["text"]
        ids=[m["turn_id"] for m in messages];assert len(ids)==len(set(ids))
        assert all(m["role"] in {"user","assistant"} for m in messages)
        for reply in row["replies"].values():assert reply["turn_id"] not in ids
        first_fact=next(i for i,m in enumerate(messages) if m["role"]=="user" and "prefiero" in m["text"].lower())
        for i,m in enumerate(messages):
            if m["role"]=="assistant" and "?" in m["text"]:
                question_positions.add("before_fact" if i<first_fact else "after_fact")
    assert lengths==set(range(2,9))
    assert question_positions=={"before_fact","after_fact"}


def test_all_twenty_four_dishes_and_beverage_breakfast_vocab_appear_as_user_evidence():
    plan=diversity_plan(build_varied_development())
    assert len(plan["dishes_observed"])>=24
    assert len(plan["drinks_observed"])==len(DRINKS)
    assert len(plan["breakfasts_observed"])==len(BREAKFASTS)
    assert all(" " in value for value in DISHES+DRINKS+BREAKFASTS)
    assert plan["query_attributes_by_root"]=={"dinner_preference":30,"drink_preference":10,"breakfast_preference":10}


def test_supplied_assistant_proposals_are_not_gold_user_preferences():
    tested=0
    for row in build_varied_development():
        if row["authoring"]["stratum"]!="unrelated_attribute_control":continue
        assert row["truth"]["current_value"]==row["initial_memory"][0]["value"]
        assert row["truth"]["future_value"]==row["initial_memory"][0]["value"]
        for m in row["user_messages"]:
            if m["role"]=="assistant" and "prefiero" in m["text"]:
                assert row["truth"]["current_value"] not in m["text"]
                tested+=1
    assert tested>=5


def test_literal_correction_and_last_scope_recipes():
    rows=build_varied_development();boundaries=set();corrections=0
    for row in rows:
        kind=row["authoring"]["stratum"];truth=row["truth"]
        if kind=="ongoing_correction":
            messages=[m["text"] for m in row["user_messages"] if m["role"]=="user"]
            corrected=next(t for t in messages if t.startswith("Corrijo:"))
            assert truth["new_value"] in corrected
            assert truth["current_value"]==truth["future_value"]==truth["new_value"]
            assert messages.index(corrected)<messages.index("Como preferencia habitual desde ahora.")
            corrections+=1
        if kind=="separate_interval":
            end=truth["scope"]["valid_until_exclusive"];future=row["future_query"]["date"]
            boundaries.add(row["authoring"]["boundary"])
            expected=truth["new_value"] if future<end else truth["initial_value"]
            assert truth["future_value"]==expected
            inclusive=(date.fromisoformat(end)-timedelta(days=1)).isoformat()
            assert any(inclusive in m["text"] for m in row["user_messages"] if m["role"]=="user")
    assert corrections==10 and boundaries=={"inside","exclusive_end","after"}


def test_ambiguous_worlds_have_identical_public_history_but_different_future_truth():
    pairs=[v for v in by_root(build_varied_development()).values() if v[0]["authoring"]["stratum"]=="ambiguous_duration"]
    assert len(pairs)==10
    for pair in pairs:
        assert len(pair)==2
        for key in ("initial_memory","prior_sessions","user_messages","current_query","future_query","observable_state_id"):
            assert pair[0][key]==pair[1][key]
        assert pair[0]["truth"]["current_value"]==pair[1]["truth"]["current_value"]
        assert pair[0]["truth"]["future_value"]!=pair[1]["truth"]["future_value"]
        assert pair[0]["replies"]["TEMPORAL_SCOPE"]!=pair[1]["replies"]["TEMPORAL_SCOPE"]


def test_seed_is_reproducible_and_changes_content_not_only_ids():
    a=build_varied_development(seed=53017);b=build_varied_development(seed=53017);c=build_varied_development(seed=53018)
    assert a==b
    assert {semantic_episode_id_v3(e) for e in a}!={semantic_episode_id_v3(e) for e in c}
    assert Counter(len(v[0]["initial_memory"]) for v in by_root(c).values())==Counter(MEMORY_COUNTS)


def test_prepare_snapshots_seven_sources_and_refuses_overwriting(tmp_path):
    history=tmp_path/"history";history.mkdir();old=deepcopy(build_varied_development()[0])
    old["root_id"]="old_fixture_root";old["user_messages"]=[{"role":"user","turn_id":"u1","text":"Un caso anterior distinto."}]
    for name in ("dev_smoke","train","validation","test_id","test_ood","test_external"):
        (history/f"{name}.jsonl").write_text(json.dumps(old)+"\n")
    previous=tmp_path/"previous.jsonl";previous.write_text(json.dumps(old)+"\n")
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in history.iterdir()}
    data,report=tmp_path/"new_data",tmp_path/"report"
    result=prepare_varied_development(data,report,historical_directory=history,previous_development=previous)
    assert result["roots"]==50 and result["episodes"]==60
    assert result["historical_hashes_before"]==result["historical_hashes_after"]
    assert len(result["historical_hashes_before"])==7
    assert hashlib.sha256((data/"validation.jsonl").read_bytes()).hexdigest()==result["bank_sha256"]
    assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in history.iterdir()}==before
    with pytest.raises(FileExistsError):prepare_varied_development(data,report,historical_directory=history,previous_development=previous)
