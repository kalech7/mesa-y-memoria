"""Audited orchestration artifacts; none of these helpers opens test or trains.

Freezing records content hashes, not approval. Gate evidence is written by the
campaign after successful stages. Checkpoints are inspected without loading GPU
models or rewriting any algorithm's result.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from .artifacts import ROOT, canonical, digest, file_hash, load_config, now, write_json
from .protocol import FROZEN_REQUIRED_FILES, GateError, require_gate


def _root(root):
    return Path(root or ROOT).resolve()


def _read(path):
    value = load_config(path)
    if not isinstance(value, dict):
        raise ValueError("EXPECTED_OBJECT: " + str(path))
    return value


def _inside(root, path):
    path = Path(path)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not path.is_absolute() and not resolved.exists():
        workspace_relative = (root.parent / path).resolve()
        if workspace_relative.is_relative_to(root) and workspace_relative.is_file():
            resolved = workspace_relative
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("ARTIFACT_MISSING_OR_OUTSIDE_RESEARCH: " + str(path))
    return resolved


def _not_opened(root):
    for relative in ("manifests/data_manifest.json", "manifests/run_manifest.json"):
        path = root / relative
        if path.is_file():
            manifest = _read(path)
            if manifest.get("test_opened") is True or manifest.get("test_opened_at"):
                raise GateError("TEST_ALREADY_OPENED: " + relative)
    for name in ("test_opening.json", "test_completion.json"):
        if any(root.rglob(name)):
            raise GateError("TEST_ALREADY_OPENED: " + name)


def _data_hashes(manifest):
    values = manifest.get("data_file_hashes", {})
    if isinstance(values, dict):
        if any(not isinstance(path, str) or not isinstance(value, str) for path, value in values.items()):
            raise ValueError("DATA_FILE_HASHES_REQUIRE_PATH_TO_SHA256")
        return values
    if isinstance(values, list):
        return {entry["path"]: entry["sha256"] for entry in values}
    raise ValueError("INVALID_DATA_FILE_HASHES")


def _freeze_files(root):
    config = _read(root / "configs/protocol.yaml")
    required = set(FROZEN_REQUIRED_FILES)
    if config.get("execution_mode") == "exploratory":
        required.update({"src/data/automatic_audit.py", "manifests/automatic_audit.json",
                         "manifests/exploratory_authorization.json"})
    for relative in ("human_audit.json", "shared_dependencies.json", "selected_checkpoints.json", "test_size_power.json"):
        if (root / "manifests" / relative).is_file():
            required.add("manifests/" + relative)
    for relative in ("run.py", "execute.py"):
        if (root / relative).is_file():
            required.add(relative)
    for directory in ("src", "prompts", "schemas", "configs"):
        for path in (root / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and (directory != "src" or path.suffix == ".py"):
                required.add(str(path.relative_to(root)))
    data = _data_hashes(_read(root / "manifests/data_manifest.json"))
    for relative, expected in data.items():
        path = _inside(root, relative)
        if file_hash(path) != expected:
            raise ValueError("DECLARED_DATA_FILE_HASH_MISMATCH: " + relative)
        required.add(relative)
    files = {relative: file_hash(_inside(root, relative)) for relative in sorted(required)}
    return config, files, data


def _freeze(root, *, allowed_changed=frozenset()):
    _not_opened(root)
    config, files, data = _freeze_files(root)
    previous_path = root / "manifests/frozen_protocol.json"
    if previous_path.is_file():
        previous = _read(previous_path)
        if previous.get("status") == "frozen":
            for relative, old_hash in previous.get("files", {}).items():
                if files.get(relative) != old_hash and relative not in allowed_changed:
                    raise GateError("FROZEN_PROTOCOL_CHANGED: " + relative)
    audit_path = root / "manifests/human_audit.json"
    human = _read(audit_path).get("status", "pending_human_audit") if audit_path.is_file() else "pending_human_audit"
    result = {"status": "frozen", "frozen_at": now(), "protocol_hash": digest(config),
              "research_status": config.get("execution_mode", "confirmatory"),
              "human_audit_status": human, "files": files, "data_file_hashes": data,
              "test_opened": False}
    write_json(previous_path, result)
    return result


def freeze_protocol(root=None):
    """Freeze all implementation, prompts, schemas, locks and declared datasets.

    Existing frozen semantic artifacts cannot silently be replaced. New artifacts
    (e.g. selected checkpoints) may be added before test; changing the final test
    size must use lock_test_size, which proves the config change is size-only.
    """
    return _freeze(_root(root))


def lock_test_size(size, power_report, root=None):
    """Lock sample size once using validation-only planning, before test opens."""
    root = _root(root)
    require_gate("G3", root)
    _not_opened(root)
    if not isinstance(size, int) or isinstance(size, bool) or not 300 <= size <= 1200 or size % 5:
        raise ValueError("TEST_SIZE_MUST_BE_BALANCED_WITHIN_300_1200")
    config = _read(root / "configs/protocol.yaml")
    if config.get("sample", {}).get("test_id_locked_roots") is not None:
        raise GateError("TEST_SAMPLE_SIZE_ALREADY_LOCKED")
    lock = _read(root / "configs/environment.lock.yaml")
    if lock.get("protocol_hash") != digest(config):
        raise GateError("TECHNICAL_LOCK_PROTOCOL_HASH_MISMATCH")
    if isinstance(power_report, dict):
        report = deepcopy(power_report)
        report_path = root / "manifests/test_size_power.json"
    else:
        report_path = _inside(root, power_report)
        report = _read(report_path)
    if (report.get("status") != "planning_only"
            or report.get("method") != "paired_root_and_seed_resampling_normal_test"
            or report.get("input_split", "validation") != "validation"):
        raise ValueError("VALIDATION_ONLY_PAIRED_POWER_REPORT_REQUIRED")
    scenarios = report.get("scenarios", [])
    selected = [row for row in scenarios if row.get("roots") == size and row.get("variance_scale") == 1.5]
    if not selected:
        raise ValueError("SELECTED_SIZE_NOT_EVALUATED_IN_CONSERVATIVE_POWER_SCENARIO")
    # Validate existing artifacts before mutating configuration.
    _, before_files, _ = _freeze_files(root)
    existing_freeze = root / "manifests/frozen_protocol.json"
    if existing_freeze.is_file():
        existing = _read(existing_freeze)
        if existing.get("status") == "frozen":
            for relative, old_hash in existing.get("files", {}).items():
                if before_files.get(relative) != old_hash:
                    raise GateError("FROZEN_PROTOCOL_CHANGED: " + relative)
    before = digest(config)
    changed = deepcopy(config)
    changed["sample"]["test_id_locked_roots"] = size
    proof = deepcopy(changed)
    proof["sample"]["test_id_locked_roots"] = None
    if proof != config:
        raise RuntimeError("TEST_SIZE_UPDATE_CHANGED_PROTOCOL_SEMANTICS")
    if isinstance(power_report, dict):
        write_json(report_path, report)
    write_json(root / "configs/protocol.yaml", changed)
    updated_lock = deepcopy(lock)
    updated_lock["protocol_hash"] = digest(changed)
    write_json(root / "configs/environment.lock.yaml", updated_lock)
    frozen = _freeze(root, allowed_changed={"configs/protocol.yaml", "configs/environment.lock.yaml"})
    event = {"event": "test_sample_size_locked", "created_at": now(), "size": size,
             "reason": "Pre-test final sample choice from validation-only paired-root and seed power analysis.",
             "config_before_hash": before, "config_after_hash": digest(changed),
             "changed_fields": ["sample.test_id_locked_roots"], "evaluation_semantics_changed": False,
             "power_report": str(report_path.relative_to(root)), "power_report_hash": file_hash(report_path),
             "conservative_power": selected[0].get("estimated_power"),
             "research_status": changed.get("execution_mode", "confirmatory"), "test_opened": False}
    path = root / "manifests/preregistration_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(canonical(event) + "\n")
    return {"status": "locked", "size": size, "event": event, "frozen_protocol_hash": digest(frozen)}


def _mapping(value, root):
    return _read(_inside(root, value)) if isinstance(value, (str, Path)) else value


def _result(mapping, seed):
    return mapping[str(seed)] if str(seed) in mapping else mapping[seed]


def _checkpoint(root, result, seed):
    if not isinstance(result, dict) or result.get("status") != "executed" or result.get("seed") != seed:
        raise ValueError("COMPLETED_TRAINING_RESULT_REQUIRED")
    step = result.get("best_step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("INVALID_SELECTED_CHECKPOINT_STEP")
    selected = _inside(root, Path(result["selected_checkpoint"]) / "adapter_model.safetensors")
    directory = selected.parent
    if directory.name != f"step_{step:06d}":
        raise ValueError("SELECTED_CHECKPOINT_PATH_STEP_MISMATCH")
    report = _inside(root, directory.parent / "training.json")
    actual = _read(report)
    for key in ("status", "seed", "best_step", "selected_checkpoint"):
        if actual.get(key) != result.get(key):
            raise ValueError("TRAINING_REPORT_DISAGREES_WITH_SELECTED_RESULT: " + key)
    history = actual.get("history", [])
    if not any(item.get("step") == 0 and isinstance(item.get("validation"), dict) for item in history):
        raise ValueError("STEP_ZERO_VALIDATION_REQUIRED")
    if not any(item.get("step") == result["best_step"] and isinstance(item.get("validation"), dict) for item in history):
        raise ValueError("SELECTED_VALIDATION_RESULT_REQUIRED")
    return directory, selected, report


def make_selected_checkpoint_manifest(sft_results, continuation_results, root=None):
    """Inspect all 12 completed branches and record real file/parent hashes.

    Inputs match run_sft_campaign and run_continuations: seed-keyed dictionaries;
    continuation values are method-keyed dictionaries. Existing reports are never
    corrected. A missing/failed/skipped branch prevents a complete manifest.
    """
    root = _root(root)
    _not_opened(root)
    sft_results, continuation_results = _mapping(sft_results, root), _mapping(continuation_results, root)
    entries = []
    for seed in (101, 202, 303):
        b1 = _result(sft_results, seed)
        b1_dir, b1_adapter, b1_report = _checkpoint(root, b1, seed)
        parent_hash = file_hash(b1_adapter)
        runs = {"B1": b1, **_result(continuation_results, seed)}
        for method in ("B1", "B2-L", "B3", "B2-I"):
            result = runs[method]
            directory, adapter, report = _checkpoint(root, result, seed)
            actual_report = _read(report)
            expected_algorithm = "completion_only_lora_sft" if method in {"B1", "B3"} else "episodic_clipped_grpo"
            if actual_report.get("algorithm") != expected_algorithm:
                raise ValueError("TRAINING_ALGORITHM_DOES_NOT_MATCH_METHOD: " + method)
            if method in {"B2-L", "B2-I"} and actual_report.get("horizon") != ("long" if method == "B2-L" else "short"):
                raise ValueError("RL_HORIZON_DOES_NOT_MATCH_METHOD: " + method)
            entry = {"method": method, "seed": seed, "adapter_file": str(adapter.relative_to(root)),
                     "adapter_hash": file_hash(adapter), "selected_step": result["best_step"],
                     "selection_split": "validation", "step_zero_evaluated": True,
                     "validation_report": str(report.relative_to(root)), "validation_report_hash": file_hash(report),
                     "parent_checkpoint_hash": None, "initial_adapter_hash": None, "kl_reference_hash": None}
            if method != "B1":
                parent = _inside(root, Path(result["parent_checkpoint"]) / "adapter_model.safetensors")
                initial = _inside(root, directory.parent / "step_000000/adapter_model.safetensors")
                if parent.parent != b1_dir or file_hash(parent) != parent_hash:
                    raise ValueError("CONTINUATION_PARENT_NOT_SELECTED_SAME_SEED_B1")
                entry["parent_checkpoint_hash"] = file_hash(parent)
                entry["initial_adapter_hash"] = file_hash(initial)
                if entry["initial_adapter_hash"] != parent_hash:
                    raise ValueError("REAL_STEP_ZERO_ADAPTER_DIFFERS_FROM_B1")
                if method in {"B2-L", "B2-I"}:
                    reference = _inside(root, Path(result["kl_reference"]) / "adapter_model.safetensors")
                    if reference.parent != b1_dir:
                        raise ValueError("KL_REFERENCE_NOT_SELECTED_SAME_SEED_B1")
                    entry["kl_reference_hash"] = file_hash(reference)
            entries.append(entry)
    manifest = {"status": "passed", "selection_split": "validation", "step_zero_eligible": True,
                "created_at": now(), "research_status": _read(root / "configs/protocol.yaml").get("execution_mode", "confirmatory"),
                "checkpoints": entries, "test_opened": False}
    target = root / "manifests/selected_checkpoints.json"
    if target.is_file():
        previous = _read(target)
        if previous.get("status") == "passed":
            if previous.get("checkpoints") != entries:
                raise GateError("SELECTED_CHECKPOINT_MANIFEST_ALREADY_EXISTS_WITH_DIFFERENT_SELECTION")
            return previous
    # Verify the existing common validation selector before publishing status.
    from .training.sft import selection_key
    for entry in entries:
        report = _read(root / entry["validation_report"])
        candidates = [(row["step"], selection_key(row["validation"], row["step"]))
                      for row in report["history"] if isinstance(row.get("validation"), dict)]
        if max(candidates, key=lambda row: row[1])[0] != entry["selected_step"]:
            raise ValueError("SELECTED_CHECKPOINT_IS_NOT_COMMON_VALIDATION_BEST")
    write_json(target, manifest)
    return manifest
