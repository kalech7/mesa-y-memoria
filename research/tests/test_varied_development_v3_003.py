from copy import deepcopy
import json
from pathlib import Path

import pytest

from research.varied_development import diversity_plan
from research.varied_development_v3_003 import neutralize_assistant_proposals, prepare_revision, NEUTRAL_QUESTIONS
from research.src.data.semantic_partition_v3 import assert_semantic_split_isolation_v3, SemanticSplitLeakage


PARENT = Path("research/data/development_varied_v3_20261025_002")


def rows():
    return [json.loads(s) for s in (PARENT / "validation.jsonl").open()]


def test_only_nine_assistant_texts_change_without_touching_gold_or_any_other_field():
    old = rows()
    original = deepcopy(old)
    new, changes = neutralize_assistant_proposals(old)
    assert old == original and len(changes) == 9
    assert {c["new_text"] for c in changes} == set(NEUTRAL_QUESTIONS)
    for before, after in zip(old, new):
        restored = deepcopy(after)
        for old_message, new_message in zip(before["user_messages"], restored["user_messages"]):
            if old_message != new_message:
                assert old_message["role"] == new_message["role"] == "assistant"
                assert "prefiero" not in new_message["text"] and "«" not in new_message["text"]
                new_message["text"] = old_message["text"]
        assert restored == before


def test_diversity_is_unchanged_and_parent_versions_cannot_be_independent_splits():
    old = rows()
    new, _ = neutralize_assistant_proposals(old)
    before, after = diversity_plan(old), diversity_plan(new)
    for key in ("memory_counts_by_root", "turn_counts_by_root", "prior_sessions_by_root",
                "dishes_observed", "drinks_observed", "breakfasts_observed"):
        assert before[key] == after[key]
    with pytest.raises(SemanticSplitLeakage):
        assert_semantic_split_isolation_v3({"validation":new}, historical_splits={"parent002":old})


def test_prepare_preserves_51_raw_lines_and_protects_full_lineage(tmp_path):
    data, report = tmp_path / "data", tmp_path / "report"
    manifest = prepare_revision(PARENT, data, report)
    before_lines = (PARENT / "validation.jsonl").read_bytes().splitlines(keepends=True)
    after_lines = (data / "validation.jsonl").read_bytes().splitlines(keepends=True)
    assert len(before_lines) == len(after_lines) == 60
    assert sum(a == b for a,b in zip(before_lines, after_lines)) == 51
    for before, after in zip(map(json.loads,before_lines), map(json.loads,after_lines)):
        for key in ("truth", "initial_memory", "prior_sessions", "replies", "current_query", "future_query"):
            assert before[key] == after[key]
    assert manifest["protected_sources_before"] == manifest["protected_sources_after"]
    assert manifest["lineage"]["ancestor_chain"][0]["parent_bank_sha256"] == "dcb558f4a9cd3bc19c10ea556ac8e435ef58f84ddb2511079e1e05f901bbbb41"
    assert manifest["expected_parent_semantic_overlap_episodes"] == 51
    assert manifest["ancestor001_semantic_overlap_episodes"] == 50
    assert manifest["lineage"]["model_inference_calls_before_revision"] == 0
    with pytest.raises(FileExistsError):
        prepare_revision(PARENT, data, report)
