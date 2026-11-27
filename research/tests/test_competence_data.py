from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from research.competence_data import (
    DEFAULT_SIZES, build_candidates, partition_candidates, teacher_for_episode,
    teacher_examples, prepare, load_and_check, discover_previous_banks, PUBLIC_KEYS,
)
from research.src.data.semantic_partition_v3 import (
    assert_semantic_split_isolation_v3, observable_episode_id_v3, SemanticSplitLeakage,
)


SMALL={"train":5,"validation":5,"evaluation":5}


@pytest.fixture(scope="module")
def candidates():
    return build_candidates(seed=64031,sizes=SMALL)


def test_fresh_recipes_full_values_roles_histories_and_independent_literal_controls(candidates):
    assert len(candidates)==18 and len({e["root_id"] for e in candidates})==15
    for e in candidates:
        assert 1<=len(e["initial_memory"])<=6 and 2<=len(e["user_messages"])<=8
        assert e["current_query"]["date"]>"2030-01-01"
        assert all(" " in record["value"] for record in e["initial_memory"])
        assert not any(m["role"]=="assistant" and "prefiero" in m["text"] for m in e["user_messages"])
        sources={m["turn_id"]:(s["date"],m) for s in e["prior_sessions"] for m in s["messages"]}
        for memory in e["initial_memory"]:
            for quote in memory["evidence_quotes"]:
                at,message=sources[quote["turn_id"]]
                assert message["role"]=="user" and message["text"]==quote["text"]
                assert at==memory["recorded_at"]
        assert observable_episode_id_v3(e)==e["observable_state_id"]


def test_default_pool_and_partition_quotas_are_reproducible_and_keep_worlds_together():
    candidates=build_candidates(seed=64031)
    splits=partition_candidates(candidates,DEFAULT_SIZES,seed=64032)
    assert sum(len(rows) for rows in splits.values())==336
    assert {name:len({e["root_id"] for e in rows}) for name,rows in splits.items()}==DEFAULT_SIZES
    assert splits==partition_candidates(list(reversed(candidates)),DEFAULT_SIZES,seed=64032)
    assert {name:len(rows) for name,rows in splits.items()}=={"train":240,"validation":36,"evaluation":60}
    for name,rows in splits.items():
        roots={e["root_id"]:e for e in rows}
        assert Counter(e["authoring"]["stratum"] for e in roots.values())==Counter({s:DEFAULT_SIZES[name]//5 for s in
            ("ongoing_correction","separate_interval","contextual_scope","ambiguous_duration","unrelated_attribute_control")})
        assert any(e["authoring"].get("observability_control",{}).get("kind")=="single_memory_negative_control" for e in roots.values())
        assert any(e["authoring"].get("observability_control",{}).get("kind")=="expired_latest_same_attribute_record" for e in roots.values())
    assert_semantic_split_isolation_v3(splits)


def test_equivalent_aliases_are_not_split_to_satisfy_a_quota(candidates):
    rows=deepcopy(candidates)
    roots=defaultdict(list)
    for e in rows:roots[e["root_id"]].append(e)
    stratum="ongoing_correction"
    pair=[rid for rid,group in roots.items() if group[0]["authoring"]["stratum"]==stratum][:2]
    for index,e in enumerate(rows):
        if e["root_id"]==pair[1]:
            alias=deepcopy(roots[pair[0]][0]);alias["root_id"]=pair[1];rows[index]=alias
    with pytest.raises(ValueError,match="equivalence component"):
        partition_candidates(rows,SMALL,seed=64032)


def test_teacher_is_real_observable_only_and_hidden_changes_do_not_change_targets(candidates,tmp_path):
    e=deepcopy(next(e for e in candidates if e["authoring"]["stratum"]=="ambiguous_duration"))
    e["split"]="train"
    before=teacher_for_episode(e,work_directory=tmp_path/"original")
    changed=deepcopy(e);changed.update(truth={"secret":"PRIVATE_TRUTH"},future_query={"secret":"PRIVATE_FUTURE_QUERY"},
        opportunity={"secret":"PRIVATE_OPPORTUNITY"},authoring={"secret":"PRIVATE_AUTHOR"})
    after=teacher_for_episode(changed,work_directory=tmp_path/"changed")
    assert [r["phase"] for r in before]==["select","respond","write"]
    assert [(r["observation"],r["action"]) for r in before]==[(r["observation"],r["action"]) for r in after]
    assert all(set(r["observation"])==PUBLIC_KEYS for r in before)
    assert "PRIVATE_" not in json.dumps(before)
    assert all("availability" not in message for row in before for message in row["observation"]["user_messages"])
    assert any(m.get("in_reply_to")=="a_tool_1" for m in before[1]["observation"]["user_messages"])
    assert (tmp_path/"original"/before[0]["teacher_db_file"]).exists()
    # A retained DB can reopen without locks after the helper closes all handles.
    import sqlite3
    with sqlite3.connect(tmp_path/"original"/before[0]["teacher_db_file"]) as db:
        assert db.execute("select count(*) from cm_records").fetchone()[0]==len(e["initial_memory"])


def test_teacher_routes_have_one_natural_select_and_six_complete_phase_targets(candidates,tmp_path):
    episode=deepcopy(candidates[0]);episode["split"]="train"
    examples=list(teacher_examples([episode],work_directory=tmp_path))
    assert len(examples)==7 and Counter(r["phase"] for r in examples)=={"select":1,"respond":3,"write":3}
    assert {r["route"] for r in examples}=={"natural","PROCEED","ASK:TEMPORAL_SCOPE"}
    assert all(json.loads(row["completion"])==row["action"] for row in examples)


def test_previous_bank_discovery_includes_every_canonical_observed_bank():
    paths=discover_previous_banks()
    for name in ("development_v2_20261024_001","development_varied_v3_20261025_001",
                 "development_varied_v3_20261025_002","development_varied_v3_20261025_003"):
        assert f"research/data/{name}/validation.jsonl" in paths
    assert sum(k.startswith("research/data/bank_1_1/") for k in paths)==6
    assert not any("sft_teacher" in k for k in paths)


def test_prepare_audits_train_validation_and_never_executes_evaluation(candidates,tmp_path,monkeypatch):
    import research.competence_data as module
    # Once the real campaign exists, recreating its seed must be rejected as
    # leakage. This independent fixture still checks every discovered bank.
    manifest=prepare(tmp_path/"corpus",seed=64061,partition_seed=64062,sizes=SMALL)
    assert manifest["audit_contracts"]["status"]=="verified"
    audit=json.loads((tmp_path/"corpus/audit_contracts.json").read_text())
    assert set(audit["by_split"])=={"train","validation"}
    assert audit["counts"]["episodes"]==12 and audit["counts"]["counterfactual_phase_prefixes"]==36
    assert audit["PI1_controls"]=={"cases":2,"positive_benefits":0,"negative_controls":2}
    assert len(audit["artifacts"])==3
    assert len((tmp_path/"corpus/audit_counterfactual_prefixes.jsonl").read_text().splitlines())==36
    assert manifest["historical_hashes_before"]==manifest["historical_hashes_after"]
    assert manifest["source_hashes_before"]==manifest["source_hashes_after"]
    assert manifest["teachers"]["train"]["examples"]==42 and manifest["teachers"]["validation"]["examples"]==42
    assert "evaluation" not in manifest["teachers"]
    splits,checked=load_and_check(tmp_path/"corpus")
    assert len(splits["evaluation"])==6 and checked==manifest
    with pytest.raises(FileExistsError):prepare(tmp_path/"corpus",seed=64061,partition_seed=64062,sizes=SMALL)
    path=tmp_path/"corpus/train.jsonl";path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError,match="SHA mismatch"):load_and_check(tmp_path/"corpus")
