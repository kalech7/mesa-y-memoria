from copy import deepcopy
import math

import pytest

from research.src.data.development_v2 import build_development_v2
from research.src.data.semantic_partition_v2 import (
    SemanticSplitLeakage, assert_semantic_split_isolation_v2,
    partition_roots_v2, semantic_episode_id_v2,
)


def renamed(episode, suffix):
    result = deepcopy(episode)
    mapping = {}
    for record in result["initial_memory"]:
        mapping[record["memory_id"]] = record["memory_id"] + suffix
        for ref in record["source_turn_ids"]:
            mapping[ref] = ref + suffix
    for message in result["user_messages"]:
        mapping[message["turn_id"]] = message["turn_id"] + suffix
    for reply in result["replies"].values():
        mapping[reply["turn_id"]] = reply["turn_id"] + suffix
    keys = {"memory_id", "turn_id", "source_turn_ids", "supersedes"}

    def replace(value, key=None):
        if isinstance(value, dict):
            return {k: replace(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [replace(v, key) for v in value]
        return mapping.get(value, value) if key in keys and isinstance(value, str) else value
    result = replace(result)
    result.update(root_id=result["root_id"] + suffix, latent_world_id="renamed_world" + suffix,
                  split="different_partition", data_namespace="another_namespace")
    return result


@pytest.fixture
def episode():
    return build_development_v2()[0]


def test_namespaces_and_id_renaming_cannot_evade_split_guard(episode):
    other = renamed(episode, "_new")
    assert semantic_episode_id_v2(other) == semantic_episode_id_v2(episode)
    with pytest.raises(SemanticSplitLeakage, match="SEMANTIC_SPLIT_LEAKAGE") as error:
        assert_semantic_split_isolation_v2({"train": [episode], "validation": [other]})
    assert any(f["kind"] == "behavioral" for f in error.value.findings)


def test_id_renaming_preserves_literal_values_that_happen_to_equal_an_id(episode):
    old_id = episode["initial_memory"][0]["memory_id"]
    episode["initial_memory"][0]["value"] = old_id
    episode["initial_memory"][0]["evidence_quotes"][0]["text"] = old_id
    assert semantic_episode_id_v2(renamed(episode, "_different")) == semantic_episode_id_v2(episode)


@pytest.mark.parametrize("factor", ["food", "utterance", "quote", "future_date", "future_context", "reply", "gold", "opportunity"])
def test_identity_never_normalizes_behavioral_differences(episode, factor):
    changed = deepcopy(episode)
    if factor == "food": changed["initial_memory"][0]["value"] = "otra_comida"
    elif factor == "utterance": changed["user_messages"][0]["text"] += " Otra frase."
    elif factor == "quote": changed["initial_memory"][0]["evidence_quotes"][0]["text"] += " Cita distinta."
    elif factor == "future_date": changed["future_query"]["date"] = "2031-01-01"
    elif factor == "future_context": changed["future_query"]["context"] = "otro_contexto"
    elif factor == "reply": changed["replies"]["GENERIC"]["text"] = "Otra respuesta."
    elif factor == "gold": changed["truth"]["future_value"] = "otra_comida"
    else: changed["opportunity"]["eligible"] = not changed["opportunity"]["eligible"]
    assert semantic_episode_id_v2(changed) != semantic_episode_id_v2(episode)


def test_historical_content_is_rejected_even_with_new_ids(episode):
    with pytest.raises(SemanticSplitLeakage) as error:
        assert_semantic_split_isolation_v2({"validation": [renamed(episode, "_new")]}, historical_splits={"old_train": [episode]})
    assert error.value.findings[0]["kind"] == "historical_behavioral"


def test_root_descendants_are_never_split_even_when_behavior_differs(episode):
    changed = deepcopy(episode)
    changed["truth"]["future_value"] = "different_world"
    with pytest.raises(SemanticSplitLeakage) as error:
        assert_semantic_split_isolation_v2({"train": [episode], "validation": [changed]})
    assert any(f["kind"] == "root" for f in error.value.findings)


def test_assign_partitions_to_equivalence_components_before_labels(episode):
    other = renamed(episode, "_alias")
    distinct = renamed(episode, "_distinct")
    distinct["truth"]["future_value"] = "different_food"
    distinct["user_messages"][0]["text"] += " Una situación distinta."
    result = partition_roots_v2([episode, other, distinct], {"train": .5, "validation": .5}, seed=41)
    owners = {e["root_id"]: split for split, rows in result.items() for e in rows}
    assert owners[episode["root_id"]] == owners[other["root_id"]]
    assert set(result) == {"train", "validation"}
    assert_semantic_split_isolation_v2(result)
    reversed_result = partition_roots_v2([distinct, other, episode], {"train": .5, "validation": .5}, seed=41)
    assert owners == {e["root_id"]: split for split, rows in reversed_result.items() for e in rows}


def test_equivalence_components_are_transitive_across_world_variants(episode):
    a = deepcopy(episode)
    b = deepcopy(episode);b["truth"]["future_value"] = "second"
    b["user_messages"][0]["text"] = "Segunda situación."
    root1 = [a, b]
    root2 = [renamed(b, "_root2")]
    c = renamed(episode, "_root2");c["truth"]["future_value"] = "third"
    c["user_messages"][0]["text"] = "Tercera situación."
    root2.append(c)
    root3 = [renamed(c, "_root3")]
    all_rows = root1 + root2 + root3
    result = partition_roots_v2(all_rows, {"train": .5, "validation": .5})
    assert len(result) == 1  # This component cannot be split to satisfy quotas.
    assert len(next(iter(result.values()))) == len(all_rows)


@pytest.mark.parametrize("fraction", [0, -1, math.nan, math.inf, True])
def test_invalid_partition_targets_fail_closed(episode, fraction):
    with pytest.raises(ValueError):
        partition_roots_v2([episode], {"train": fraction})


def test_compatible_worlds_cannot_cross_splits_under_renamed_roots(episode):
    other_world = renamed(episode, "_other_world")
    other_world["truth"]["future_value"] = "another_possible_value"
    other_world["replies"]["TEMPORAL_SCOPE"]["text"] = "Una respuesta distinta."
    assert semantic_episode_id_v2(other_world) != semantic_episode_id_v2(episode)
    with pytest.raises(SemanticSplitLeakage) as error:
        assert_semantic_split_isolation_v2({"train": [episode], "validation": [other_world]})
    assert any(f["kind"] == "observable" for f in error.value.findings)
