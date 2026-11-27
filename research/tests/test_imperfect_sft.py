"""Controlled semantic mistakes: public prefixes, immutable truth, train-only."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from research import imperfect_sft as imperfect
from research.competence_training import prepare_teacher_examples as clean_teacher
from research.src.artifacts import digest, file_hash
from research.src.data.build_bank import build_bank
from research.src.env.episode import EpisodeEnvironment
from research.src.memory.executor import Executor
from research.src.memory.store import Store
from research.src.policy.llm_controller import validate_action
from research.src.scoring.evidence import audit_operation


def episodes(roots=20, split="train", seed=8241):
    rows = deepcopy(build_bank("dev_smoke", roots=roots, seed=seed))
    for row in rows:
        row["split"] = split
    return rows


def test_zero_rate_is_exact_clean_producer_and_annotations_do_not_enter_targets(tmp_path):
    train = episodes(5)
    before = deepcopy(train)
    examples, audit = imperfect.prepare_teacher_examples(train, tmp_path / "imperfect", rate=0)
    assert examples == clean_teacher(train, tmp_path / "clean")
    assert audit["changed_target_count"] == audit["selected_root_count"] == 0
    assert train == before
    changed = deepcopy(train)
    for row in changed:
        row["truth"] = {"secret": "PRIVATE_NOT_FOR_SFT"}
        row["evaluator_annotations"] = {"secret": "PRIVATE_NOT_FOR_SFT"}
    altered, _ = imperfect.prepare_teacher_examples(changed, tmp_path / "annotations", rate=1)
    original, _ = imperfect.prepare_teacher_examples(train, tmp_path / "original", rate=1)
    assert altered == original
    assert "PRIVATE_NOT_FOR_SFT" not in json.dumps(altered)


def test_assignment_is_deterministic_nested_grouped_and_rate_is_roots(tmp_path):
    train = episodes()
    before = digest(train)
    first, a = imperfect.prepare_teacher_examples(train, tmp_path / "a", rate=.1, seed=91)
    second, b = imperfect.prepare_teacher_examples(train, tmp_path / "b", rate=.1, seed=91)
    _, c = imperfect.prepare_teacher_examples(train, tmp_path / "c", rate=.3, seed=91)
    assert first == second and a == b
    assert a["selected_root_count"] == 2 and c["selected_root_count"] == 6
    assert set(a["selected_roots"]) < set(c["selected_roots"])
    assert a["rate_unit"] == "distinct_train_roots"
    assert a["changed_target_fraction"] != a["rate"]
    assert len(first) == 7 * len(train)
    assert digest(train) == before
    assert {m["root_id"] for m in a["mutations"]} == set(a["selected_roots"])
    for index, episode in enumerate(train):
        selected = episode["root_id"] in a["selected_roots"]
        assert (any(m["episode_index"] == index for m in a["mutations"])) == selected


def test_select_mistake_replays_actual_environment_and_forced_routes_remain_clean(tmp_path):
    train = episodes()
    examples, audit = imperfect.prepare_teacher_examples(train, tmp_path / "noisy", rate=1)
    clean = clean_teacher(train, tmp_path / "clean")
    checked = set()
    for mutation in audit["mutations"]:
        episode_index = mutation["episode_index"]
        offset = 7 * episode_index
        assert examples[offset + 3:offset + 7] == clean[offset + 3:offset + 7]
        if mutation["kind"] not in {"skip_clarification", "unnecessary_question"}:
            continue
        checked.add(mutation["kind"])
        env = EpisodeEnvironment(train[episode_index], tmp_path / f"replay_{episode_index}.sqlite")
        try:
            selection = examples[offset]["action"]
            if selection["action"] == "ASK":
                env.ask(selection["slot"])
            assert examples[offset + 1]["observation"] == env.observe("respond", "O2")
            assert examples[offset + 2]["observation"] == env.observe("write", "O2")
            assert all(row["executed_selection"] == selection for row in examples[offset:offset + 3])
        finally:
            env.dispose()
    assert checked == {"skip_clarification", "unnecessary_question"}


def test_wrong_writes_execute_but_scope_errors_fail_unchanged_evidence_audit(tmp_path):
    train = episodes(40)
    examples, audit = imperfect.prepare_teacher_examples(train, tmp_path / "noisy", rate=1)
    kinds = set()
    for mutation in audit["mutations"]:
        row = examples[mutation["example_indices"][-1]]
        validate_action("write", row["action"])
        if mutation["kind"] not in {"temporary_as_permanent", "omit_supported_update"}:
            continue
        kinds.add(mutation["kind"])
        index = mutation["episode_index"]
        store = Store(tmp_path / f"write_{index}.sqlite")
        try:
            store.initialize(row["observation"]["memory"])
            executed = Executor(store).execute(row["action"]["operations"], recorded_at=row["observation"]["query"]["date"])
            assert executed["valid"], executed
        finally:
            store.close()
        if mutation["kind"] == "temporary_as_permanent":
            assert not audit_operation(row["action"]["operations"][0], row["observation"])
        else:
            assert row["action"] == {"operations": [{"op": "NOOP"}]}
            assert mutation["clean"][-1]["action"] != row["action"]
    assert kinds == {"temporary_as_permanent", "omit_supported_update"}


def test_bundle_preserves_clean_originals_and_heldout_bytes(tmp_path):
    train = episodes(5)
    validation = tmp_path / "validation.jsonl"
    from research.competence_data import _shift_dates
    heldout_rows = _shift_dates(episodes(5, "validation", 9897), 1000)
    validation.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in heldout_rows) + "\n")
    before = file_hash(validation)
    bundle = tmp_path / "bundle"
    manifest = imperfect.prepare_bundle(train, bundle, rate=1, heldout_paths={"validation": validation})
    examples, audit = imperfect.load_examples(bundle, train=train, expected_rate=1, expected_seed=7331)
    clean = [json.loads(line) for line in (bundle / "clean_examples.jsonl").read_text().splitlines()]
    assert clean == clean_teacher(train, tmp_path / "independent_clean")
    assert manifest["heldout"]["validation"]["sha256"] == before
    assert file_hash(validation) == file_hash(bundle / "validation.jsonl") == before
    assert audit["train_episode_sha256"] == digest(train)
    assert audit["gold_modified"] is audit["heldout_modified"] is False
    assert len(examples) == len(clean)
    with pytest.raises(FileExistsError):
        imperfect.prepare_bundle(train, bundle, rate=0)
    changed = deepcopy(train)
    changed[0]["truth"]["scope_kind"] = "tampered"
    with pytest.raises(ValueError, match="digest mismatch"):
        imperfect.load_examples(bundle, train=changed)
    with pytest.raises(ValueError, match="rate mismatch"):
        imperfect.load_examples(bundle, expected_rate=.3)
    with pytest.raises(ValueError, match="seed mismatch"):
        imperfect.load_examples(bundle, expected_seed=9)
    with (bundle / "examples.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        imperfect.load_examples(bundle)


@pytest.mark.parametrize("rate", [-.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_configuration_rejected_before_generating_targets(tmp_path, rate):
    with pytest.raises(ValueError, match="rate"):
        imperfect.prepare_teacher_examples(episodes(5), tmp_path, rate=rate)
    assert not any(tmp_path.iterdir())


def test_heldout_cannot_become_training_targets_or_overlap_roots(tmp_path):
    with pytest.raises(ValueError, match="partition"):
        imperfect.prepare_teacher_examples(episodes(5, "validation"), tmp_path / "bad", rate=1)
    train = episodes(5)
    validation = deepcopy(train)
    for row in validation:
        row["split"] = "validation"
    source = tmp_path / "validation.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in validation))
    with pytest.raises(ValueError, match="overlap"):
        imperfect.prepare_bundle(train, tmp_path / "bad_bundle", rate=1, heldout_paths={"validation": source})
    assert not (tmp_path / "bad_bundle").exists()


def test_renamed_equivalent_heldout_is_rejected_before_artifacts(tmp_path):
    train = episodes(5)
    heldout = deepcopy(train)
    for i, row in enumerate(heldout):
        row.update(split="validation", root_id=f"disguised_{i}")
    source = tmp_path / "validation.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in heldout))
    with pytest.raises(ValueError, match="SEMANTIC_SPLIT_LEAKAGE"):
        imperfect.prepare_bundle(train, tmp_path / "bad_bundle", rate=.3, heldout_paths={"validation": source})
    assert not (tmp_path / "bad_bundle").exists()
