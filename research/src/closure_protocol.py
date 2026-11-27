"""Evidence-backed gates for the separate closure_v4 run, never historical gates."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
from pathlib import Path

from .artifacts import ROOT, digest, file_hash, load_config, now, write_json
from .protocol import GateError

REPO = ROOT.parent
_ACTIVE = ContextVar("closure_v4_execution", default=None)


def source_files():
    paths = []
    for directory in ("research/src", "llm_local_001", "memory", "research/prompts", "research/configs"):
        for path in (REPO / directory).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".txt", ".toml"} and "__pycache__" not in path.parts:
                paths.append(path)
    paths.extend(p for p in (ROOT / "closure_v4.py", REPO / "closure_v4.py", REPO / "pyproject.toml",
                            ROOT / "varied_development.py", ROOT / "varied_evaluation.py") if p.is_file())
    return sorted(set(paths))


def source_hashes():
    return {str(p.relative_to(REPO)): file_hash(p) for p in source_files()}


def read_manifest(directory):
    path = Path(directory) / "MANIFIESTO_V4.json"
    if not path.is_file():
        raise GateError("G0_REQUIRED: run inventory in a new closure directory first")
    manifest = load_config(path)
    if manifest.get("protocol_version") != "closure_v4":
        raise GateError("Wrong protocol version")
    return manifest


def _inside_run(directory):
    directory = Path(directory).resolve()
    if directory == ROOT / "runs/full_exploratory_001" or directory.is_relative_to(ROOT / "runs/full_exploratory_001"):
        raise GateError("HISTORICAL_RUN_IS_READ_ONLY")
    return directory


def record_stage(directory, name, result, evidence=()):
    directory = _inside_run(directory)
    manifest = read_manifest(directory)
    source_snapshot = digest(source_hashes())
    if manifest.get("code_snapshot_sha256") and manifest["code_snapshot_sha256"] != source_snapshot:
        raise GateError("SOURCE_CHANGED_DURING_RUN: preserve partial outputs and create a new inventory")
    path = directory / "stages" / (name + ".json")
    if path.exists():
        raise FileExistsError("Stage already recorded; preserve evidence and choose a new run: " + name)
    files = {str(Path(p).resolve()): file_hash(p) for p in evidence if Path(p).is_file()}
    stage = {"stage": name, "created_at": now(), "result": result, "artifacts_sha256": files,
             "code_snapshot_sha256": source_snapshot}
    write_json(path, stage)
    manifest.setdefault("stages", {})[name] = {"path": str(path.resolve()), "sha256": file_hash(path)}
    write_json(directory / "MANIFIESTO_V4.json", manifest)
    return result


def checked_stage(directory, name, *, require_passed=True):
    manifest = read_manifest(directory)
    record = manifest.get("stages", {}).get(name)
    if not record:
        raise GateError(name + "_REQUIRED: no executed evidence")
    path = Path(record["path"])
    if not path.is_relative_to(Path(directory).resolve()) or not path.is_file() or file_hash(path) != record["sha256"]:
        raise GateError(name + "_EVIDENCE_CHANGED")
    stage = load_config(path)
    if digest(source_hashes()) != stage["code_snapshot_sha256"]:
        raise GateError("SOURCE_CHANGED_SINCE_" + name)
    if not stage.get("artifacts_sha256"):
        raise GateError(name + "_MISSING_EVIDENCE_FILES")
    for filename, expected in stage["artifacts_sha256"].items():
        if not Path(filename).is_file() or file_hash(filename) != expected:
            raise GateError(name + "_EVIDENCE_CHANGED: " + filename)
    result = stage["result"]
    if require_passed and result.get("status") != "passed":
        raise GateError(name + "_NOT_PASSED: " + result.get("status", "unknown"))
    return result


def require_training_authorization(directory):
    """Only an actual inventory, passing fixtures and training evidence release RL."""
    if directory is None:
        raise GateError("CLOSURE_AUTHORIZATION_REQUIRED")
    for stage in ("inventory", "test-environment", "test-training"):
        checked_stage(directory, stage)
    diagnosis = checked_stage(directory, "diagnose", require_passed=False)
    if diagnosis.get("critical_checks_passed") is not True:
        raise GateError("G2_CRITICAL_DIAGNOSTICS_REQUIRED")
    plan_path = Path(directory) / "PILOT_PLAN.json"
    locked = Path(directory) / "PROTOCOLO_BLOQUEADO.json"
    if not plan_path.is_file() and not locked.is_file():
        raise GateError("PREDECLARED_TRAINING_PLAN_REQUIRED")
    if plan_path.is_file():
        manifest = read_manifest(directory)
        if manifest.get("pilot_plan_sha256") != file_hash(plan_path):
            raise GateError("PREDECLARED_PILOT_PLAN_CHANGED")
    if locked.is_file():
        verify_locked_protocol(directory)
    return True


def verify_locked_protocol(directory):
    directory = Path(directory)
    path = directory / "PROTOCOLO_BLOQUEADO.json"
    if not path.is_file():
        raise GateError("LOCKED_PROTOCOL_REQUIRED")
    locked = load_config(path)
    payload = {k: v for k, v in locked.items() if k != "sha256"}
    if locked.get("sha256") != digest(payload):
        raise GateError("LOCKED_PROTOCOL_CHANGED")
    manifest = read_manifest(directory)
    if manifest.get("locked_protocol_sha256") != file_hash(path):
        raise GateError("LOCKED_PROTOCOL_MANIFEST_MISMATCH")
    if locked.get("code_snapshot_sha256") != digest(source_hashes()):
        raise GateError("FROZEN_SOURCE_CHANGED")
    if not locked.get("artifacts_sha256"):
        raise GateError("LOCKED_ARTIFACTS_REQUIRED")
    for filename, expected in locked["artifacts_sha256"].items():
        if not Path(filename).is_file() or file_hash(filename) != expected:
            raise GateError("LOCKED_ARTIFACT_CHANGED: " + filename)
    return locked


def selected_checkpoints(directory):
    locked = verify_locked_protocol(directory)
    path = Path(directory) / "CHECKPOINTS_SELECCIONADOS.json"
    if not path.is_file():
        raise GateError("VALIDATION_SELECTED_CHECKPOINTS_REQUIRED")
    value = load_config(path)
    expected = {(method, seed) for method in ("B1-v4", "B3-v4", "B2-I-v4", "B2-L-v4")
                for seed in locked["plan"]["training_seeds"]}
    rows = value.get("checkpoints", [])
    if len(rows) != len(expected) or {(r.get("method"), r.get("seed")) for r in rows} != expected:
        raise GateError("COMPLETE_COMPARABLE_CHECKPOINT_MATRIX_REQUIRED")
    for row in rows:
        if row.get("selection_split") != "validation" or row.get("selected_before_test") is not True:
            raise GateError("CHECKPOINT_SELECTION_MUST_USE_VALIDATION")
        if not row.get("artifacts_sha256"):
            raise GateError("CHECKPOINT_WEIGHTS_REQUIRED")
        for filename, expected_hash in row["artifacts_sha256"].items():
            if not Path(filename).is_file() or file_hash(filename) != expected_hash:
                raise GateError("SELECTED_CHECKPOINT_CHANGED")
    return rows


@contextmanager
def execution_context(directory, purpose):
    if purpose == "training":
        require_training_authorization(directory)
    elif purpose == "locked-test":
        selected_checkpoints(directory)
    else:
        raise GateError("Unknown closure execution purpose")
    token = _ACTIVE.set((str(Path(directory).resolve()), purpose))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def active_split_authorization(split, training=False):
    active = _ACTIVE.get()
    if active is None:
        return False
    _, purpose = active
    if split.startswith("test"):
        if training or purpose != "locked-test":
            raise GateError("CLOSED_TEST: closure training cannot access test")
    elif training and (purpose != "training" or split != "train"):
        raise GateError("Closure training requires train partition")
    return True
