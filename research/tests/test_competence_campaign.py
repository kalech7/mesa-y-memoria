"""Orchestration boundaries which must hold before loading a GPU actor."""
from copy import deepcopy
import json

import pytest

from research.competence_campaign import branch_schedule, checkpoint_subset, signal_episode_ids, verify_plan, write_new
from research.src.artifacts import digest, file_hash


def episodes(split, roots=6):
    result = []
    for family in range(5):
        for root in range(roots):
            for world in range(2 if family == 4 else 1):
                result.append({"root_id": f"f{family}r{root}", "latent_world_id": f"w{world}",
                               "realization_id": "v0", "scenario_family": f"F{family}", "split": split})
    return result


def test_checkpoint_subset_preserves_all_worlds_and_is_order_independent():
    rows = episodes("validation")
    selected = checkpoint_subset(rows)
    assert {r["root_id"] for r in selected} == {f"f{f}r{r}" for f in range(5) for r in range(2)}
    assert len(selected) == 12
    assert {tuple(r.items()) for r in selected} == {tuple(r.items()) for r in checkpoint_subset(list(reversed(rows)))}
    assert all(r in selected for r in rows if r["root_id"] in {e["root_id"] for e in selected})


@pytest.mark.parametrize("split", ["train", "evaluation", "test_id"])
def test_selection_cannot_accept_training_or_heldback(split):
    with pytest.raises(ValueError, match="validation only"):
        checkpoint_subset(episodes(split))


def test_signal_cohort_uses_no_outcome_and_contains_forty_fixed_train_worlds():
    rows = episodes("train", roots=10)
    selected = signal_episode_ids(rows)
    changed = deepcopy(rows)
    for i, row in enumerate(changed):
        row["truth"] = {"future_value": "changed"}
        row["metrics"] = {"J_primary": i % 2}
    assert selected == signal_episode_ids(changed)
    assert len(selected) == len({tuple(k) for k in selected}) == 40
    for family in range(5):
        assert sum(k[0].startswith(f"f{family}") for k in selected) == 8
    with pytest.raises(ValueError, match="train only"):
        signal_episode_ids(episodes("evaluation", roots=10))


def test_preregistration_and_sources_cannot_silently_change(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    plan = {"frozen_files": {str(source): file_hash(source)}, "evidence_sha256": {},
            "capability_thresholds": {"C2": .8}}
    plan["sha256"] = digest(plan)
    write_new(tmp_path / "PREREGISTRATION.json", plan)
    assert verify_plan(tmp_path) == plan
    source.write_text("value = 2\n")
    with pytest.raises(ValueError, match="Frozen source"):
        verify_plan(tmp_path)
    source.write_text("value = 1\n")
    plan["capability_thresholds"]["C2"] = .2
    (tmp_path / "PREREGISTRATION.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Preregistration changed"):
        verify_plan(tmp_path)


def test_existing_evidence_is_never_overwritten(tmp_path):
    target = tmp_path / "evidence.json"
    write_new(target, {"status": "failed"})
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        write_new(target, {"status": "passed"})
    assert target.read_bytes() == original


def test_missing_competence_or_task_signal_blocks_rl_but_not_the_sft_control():
    for capability, signal in [(False, False), (False, True), (True, False)]:
        assert branch_schedule(capability, signal) == ["B3"]
    assert branch_schedule(True, True) == ["B3", "B2-I", "B2-L"]
