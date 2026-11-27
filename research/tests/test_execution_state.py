from copy import deepcopy
import json

import pytest

from research.src.artifacts import digest, file_hash, write_json
from research.src.execution_state import freeze_protocol, lock_test_size, make_selected_checkpoint_manifest
from research.src.protocol import GateError
# Synthetic audit/checkpoint fixture, never real research approval.
from research.tests.test_protocol import gated_root


def read(root, relative):
    return json.loads((root / relative).read_text())


def test_freeze_covers_all_sources_prompts_schemas_and_declared_data(gated_root):
    root = gated_root
    for relative in ("src/new_module.py", "prompts/nested/new.txt", "schemas/new.schema.json", "execute.py", "run.py", "manifests/shared_dependencies.json", "data/train.jsonl"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    manifest = read(root, "manifests/data_manifest.json")
    manifest["data_file_hashes"] = {"data/train.jsonl": file_hash(root / "data/train.jsonl")}
    write_json(root / "manifests/data_manifest.json", manifest)
    result = freeze_protocol(root)
    for relative in ("src/new_module.py", "prompts/nested/new.txt", "schemas/new.schema.json", "execute.py", "run.py", "manifests/shared_dependencies.json", "data/train.jsonl"):
        assert relative in result["files"]
    assert result["human_audit_status"] == "passed"  # synthetic fixture only


def test_freeze_does_not_rehash_changed_semantic_source(gated_root):
    freeze_protocol(gated_root)
    (gated_root / "src/scoring/evidence.py").write_text("changed")
    with pytest.raises(GateError, match="FROZEN_PROTOCOL_CHANGED"):
        freeze_protocol(gated_root)


def test_freeze_rejects_declared_data_hash_mismatch(gated_root):
    path = gated_root / "data.jsonl"
    path.write_text("changed")
    manifest = read(gated_root, "manifests/data_manifest.json")
    manifest["data_file_hashes"] = {"data.jsonl": "a" * 64}
    write_json(gated_root / "manifests/data_manifest.json", manifest)
    with pytest.raises(ValueError, match="DECLARED_DATA_FILE_HASH_MISMATCH"):
        freeze_protocol(gated_root)


def power():
    return {"status": "planning_only", "method": "paired_root_and_seed_resampling_normal_test",
            "recommended_roots": 300, "scenarios": [{"roots": 300, "variance_scale": 1.5, "estimated_power": .85}]}


def prepare_unlocked(root):
    config = read(root, "configs/protocol.yaml")
    config["sample"]["test_id_locked_roots"] = None
    write_json(root / "configs/protocol.yaml", config)
    lock = read(root, "configs/environment.lock.yaml")
    lock["protocol_hash"] = digest(config)
    write_json(root / "configs/environment.lock.yaml", lock)
    # The fixture began with a hash manifest without status=frozen; this creates
    # its initial freeze before the one allowed sample-size change.
    freeze_protocol(root)
    return config, lock


def test_lock_test_size_changes_only_size_and_protocol_hash(gated_root):
    before, old_lock = prepare_unlocked(gated_root)
    out = lock_test_size(300, power(), gated_root)
    changed = read(gated_root, "configs/protocol.yaml")
    changed["sample"]["test_id_locked_roots"] = None
    assert changed == before
    new_lock = read(gated_root, "configs/environment.lock.yaml")
    new_lock["protocol_hash"] = old_lock["protocol_hash"]
    assert new_lock == old_lock
    assert out["status"] == "locked"
    assert out["event"]["evaluation_semantics_changed"] is False
    assert read(gated_root, "manifests/frozen_protocol.json")["files"]["configs/protocol.yaml"] == file_hash(gated_root / "configs/protocol.yaml")
    with pytest.raises(GateError, match="ALREADY_LOCKED"):
        lock_test_size(300, power(), gated_root)


def test_freeze_and_size_lock_refuse_already_opened_test(gated_root):
    prepare_unlocked(gated_root)
    path = gated_root / "reports/test/test_opening.json"
    path.parent.mkdir(parents=True)
    write_json(path, {"status": "opened"})
    with pytest.raises(GateError, match="TEST_ALREADY_OPENED"):
        freeze_protocol(gated_root)
    with pytest.raises(GateError, match="TEST_ALREADY_OPENED"):
        lock_test_size(300, power(), gated_root)


def test_lock_size_requires_validation_power_and_G3(gated_root):
    prepare_unlocked(gated_root)
    bad = power()
    bad["input_split"] = "test_id"
    with pytest.raises(ValueError, match="VALIDATION_ONLY"):
        lock_test_size(300, bad, gated_root)
    gates = read(gated_root, "manifests/gates.json")
    gates["G3"]["status"] = "pending"
    write_json(gated_root / "manifests/gates.json", gates)
    with pytest.raises(GateError, match="GATE_REQUIRED"):
        lock_test_size(300, power(), gated_root)


def campaigns(root):
    sft, continuations = {}, {}
    for seed in (101, 202, 303):
        sft[str(seed)] = None
        continuations[str(seed)] = {}
        parent = root / f"reports/train/{seed}/B1/step_000001"
        parent_hash = None
        for method in ("B1", "B2-L", "B3", "B2-I"):
            directory = root / f"reports/train/{seed}/{method}"
            selected = directory / "step_000001"
            initial = directory / "step_000000"
            selected.mkdir(parents=True)
            initial.mkdir()
            (selected / "adapter_model.safetensors").write_bytes(f"updated {method} {seed}".encode())
            (initial / "adapter_model.safetensors").write_bytes(
                (parent / "adapter_model.safetensors").read_bytes() if method != "B1" else b"initial base")
            result = {"status": "executed", "seed": seed, "best_step": 1,
                      "algorithm": "completion_only_lora_sft" if method in {"B1", "B3"} else "episodic_clipped_grpo",
                      "horizon": "long" if method == "B2-L" else "short" if method == "B2-I" else None,
                      "selected_checkpoint": str(selected), "parent_checkpoint": str(parent) if method != "B1" else None,
                      "kl_reference": str(parent) if method in {"B2-L", "B2-I"} else None,
                      "history": [{"step": 0, "validation": {"utility": 0}}, {"step": 1, "validation": {"utility": 1}}]}
            write_json(directory / "training.json", result)
            if method == "B1":
                sft[str(seed)] = result
            else:
                continuations[str(seed)][method] = result
    # Replace the preexisting synthetic G4 fixture selection with a pending
    # placeholder; a completed manifest may never be silently replaced.
    write_json(root / "manifests/selected_checkpoints.json", {"status": "pending"})
    return sft, continuations


def test_checkpoint_manifest_hashes_actual_step_zero_and_reports(gated_root):
    sft, continuations = campaigns(gated_root)
    out = make_selected_checkpoint_manifest(sft, continuations, gated_root)
    assert len(out["checkpoints"]) == 12
    for entry in out["checkpoints"]:
        assert entry["adapter_hash"] == file_hash(gated_root / entry["adapter_file"])
        assert entry["validation_report_hash"] == file_hash(gated_root / entry["validation_report"])
        if entry["method"] != "B1":
            b1 = next(e for e in out["checkpoints"] if e["method"] == "B1" and e["seed"] == entry["seed"])
            assert entry["initial_adapter_hash"] == entry["parent_checkpoint_hash"] == b1["adapter_hash"]
    assert make_selected_checkpoint_manifest(sft, continuations, gated_root) == out


def test_checkpoint_manifest_rejects_real_initial_parameter_difference(gated_root):
    sft, continuations = campaigns(gated_root)
    directory = gated_root / "reports/train/101/B2-L/step_000000/adapter_model.safetensors"
    directory.write_bytes(b"different initial adapter")
    with pytest.raises(ValueError, match="REAL_STEP_ZERO_ADAPTER_DIFFERS"):
        make_selected_checkpoint_manifest(sft, continuations, gated_root)


def test_checkpoint_manifest_does_not_correct_bad_validation_selection(gated_root):
    sft, continuations = campaigns(gated_root)
    result = sft["101"]
    result["history"][0]["validation"]["utility"] = 2
    write_json(gated_root / "reports/train/101/B1/training.json", result)
    with pytest.raises(ValueError, match="NOT_COMMON_VALIDATION_BEST"):
        make_selected_checkpoint_manifest(sft, continuations, gated_root)


def test_checkpoint_manifest_rejects_mislabeled_RL_horizon(gated_root):
    sft, continuations = campaigns(gated_root)
    result = continuations["101"]["B2-L"]
    result["horizon"] = "short"
    write_json(gated_root / "reports/train/101/B2-L/training.json", result)
    with pytest.raises(ValueError, match="RL_HORIZON_DOES_NOT_MATCH_METHOD"):
        make_selected_checkpoint_manifest(sft, continuations, gated_root)
