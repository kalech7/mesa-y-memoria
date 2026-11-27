"""Auditable staged closure of the existing experiment (no network or installs)."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from .src.artifacts import ROOT, canonical, digest, file_hash, load_config, now, write_json
from .src.protocol import GateError
from .src.closure_protocol import (REPO, checked_stage, execution_context, read_manifest,
                                   record_stage, require_training_authorization,
                                   selected_checkpoints, source_hashes, verify_locked_protocol)

DEFAULT_PARENT = ROOT / "runs/full_exploratory_001"


def _new_stage(output, name):
    manifest = read_manifest(output)
    if name in manifest.get("stages", {}) or (Path(output) / "stages" / (name + ".json")).exists():
        raise FileExistsError("Stage already recorded; choose a new run: " + name)
    return manifest


def _read(path, default=None):
    return load_config(path) if Path(path).is_file() else default


def _command(args):
    try:
        value = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {"returncode": value.returncode, "stdout": value.stdout.strip(), "stderr": value.stderr.strip()[-2000:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "not_executed", "reason": str(error)}


def _runtime():
    versions = {}
    for name in ("torch", "transformers", "peft", "trl", "numpy", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "packages": versions, "gpu": _command(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"])}
    if versions["torch"]:
        import torch
        result.update(cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                      bf16_supported=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()))
    return result


def inventory(parent_run, output, *, run_id=None):
    parent, output = Path(parent_run).resolve(), Path(output).resolve()
    if output == parent or output.is_relative_to(parent):
        raise GateError("HISTORICAL_RUN_IS_READ_ONLY")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Inventory requires a new empty directory")
    output.mkdir(parents=True, exist_ok=True)
    checks = []
    context = _read(parent / "context.json", {})
    config = context.get("config", _read(ROOT / "configs/protocol.yaml", {}))
    code = source_hashes()
    for relative in code:
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    git = _command(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    diff = _command(["git", "-C", str(REPO), "diff", "--binary", "HEAD"]) if git.get("returncode") == 0 else None
    model_path = Path(context.get("model_path", REPO / config.get("runtime", {}).get("model_path", "models/Qwen2.5-0.5B-Instruct")))
    if not model_path.is_absolute():
        model_path = REPO / model_path
    pinned = _read(model_path / "download_manifest.json", {})
    model_files, model_errors = {}, []
    for name, item in pinned.get("files", {}).items():
        path = model_path / name
        actual = file_hash(path) if path.is_file() else None
        model_files[str(path)] = actual
        if actual != item.get("sha256"):
            model_errors.append(name)
    model_ok = bool(model_files) and not model_errors
    checks.append({"id": "G0-MODEL", "status": "verified" if model_ok else "failed",
                   "evidence": str(model_path / "download_manifest.json"), "mismatches": model_errors})
    from llm_local_001.model import artifact_hashes
    sft = _read(parent / "sft/sft_campaign.json", {})
    checkpoints = {}
    for seed in config.get("seeds", {}).get("training", [101, 202, 303]):
        item = sft.get(str(seed), {})
        checkpoint = Path(item.get("selected_checkpoint", parent / f"sft/{seed}/B1/missing"))
        # Relocate an old absolute run path only when the corresponding local
        # directory exists; provenance still records original and actual paths.
        if not checkpoint.exists() and "/sft/" in str(checkpoint):
            checkpoint = parent / ("sft/" + str(checkpoint).split("/sft/", 1)[1])
        hashes = artifact_hashes(checkpoint) if checkpoint.is_dir() else {}
        declared = _read(checkpoint / "local_checkpoint.json", {}).get("artifacts_sha256", {})
        good = bool(hashes) and hashes == declared and (checkpoint / "adapter_model.safetensors").is_file()
        checkpoints[str(seed)] = {"path": str(checkpoint), "sha256": digest(hashes) if hashes else None,
                                  "artifacts_sha256": {str(checkpoint / k): v for k, v in hashes.items()},
                                  "adapter_config": _read(checkpoint / "adapter_config.json", {}),
                                  "best_step": item.get("best_step"), "verified": good}
        checks.append({"id": f"G0-B1-{seed}", "status": "verified" if good else "failed",
                       "evidence": str(checkpoint / "local_checkpoint.json")})
    runtime = _runtime()
    paths = {"data_manifest": ROOT / "manifests/data_manifest.json", "historical_context": parent / "context.json",
             "historical_analysis": parent / "reports/ANALISIS.json"}
    artifacts = {str(p): file_hash(p) for p in paths.values() if p.is_file()}
    for name, path in paths.items():
        checks.append({"id": "G0-" + name, "status": "verified" if path.is_file() else "not_identifiable_from_current_artifacts", "evidence": str(path)})
    prompt_hashes = {str(p): file_hash(p) for p in (ROOT / "prompts").glob("*.txt")}
    manifest = {"run_id": run_id or output.name, "protocol_version": "closure_v4", "created_at": now(),
                "spec_path": str(ROOT / "specs/SPEC_CIERRE_TESIS_ENTORNO_SIMULACION_RL_V4.md"),
                "spec_sha256": file_hash(ROOT / "specs/SPEC_CIERRE_TESIS_ENTORNO_SIMULACION_RL_V4.md"),
                "parent_run": str(parent), "code_commit": git.get("stdout") if git.get("returncode") == 0 else "unversioned",
                "code_diff_sha256": hashlib.sha256(diff["stdout"].encode()).hexdigest() if diff and diff.get("returncode") == 0 else None,
                "code_diff_status": "verified" if diff else "not_identifiable_from_current_artifacts",
                "code_snapshot_sha256": digest(code), "source_files_sha256": code,
                "model_id": pinned.get("repo"), "model_revision": pinned.get("revision"),
                "tokenizer_revision": pinned.get("tokenizer_revision", pinned.get("revision")),
                "model_files_sha256": model_files, "model_path": str(model_path),
                "actor_checkpoint_sha256": {s: c["sha256"] for s, c in checkpoints.items()},
                "reference_checkpoint_sha256": {s: c["sha256"] for s, c in checkpoints.items()},
                "reference_identity_status": "weights_only_pending_runtime_logits", "b1_checkpoints": checkpoints,
                "reader_checkpoint_sha256": digest({str(ROOT / "src/readers/symbolic.py"): file_hash(ROOT / "src/readers/symbolic.py")}),
                "reader": "fixed_symbolic", "dataset_manifest_sha256": artifacts.get(str(paths["data_manifest"])),
                "prompt_manifest_sha256": digest(prompt_hashes), "prompt_files_sha256": prompt_hashes,
                "evaluator_version": digest({p: h for p, h in code.items() if "/scoring/" in p}),
                "simulator_version": file_hash(ROOT / "src/env/user_simulator.py"),
                "precision": config.get("model", {}).get("precision"), "quantization": config.get("model", {}).get("quantization"),
                "precision_status": "checkpoint_configuration_pending_runtime", "training_seeds": list(map(int, checkpoints)),
                "runtime": runtime, "historical_artifacts_sha256": artifacts,
                "human_audit": "not_executed", "external_validity": "synthetic_only", "stages": {}}
    write_json(output / "MANIFIESTO_V4.json", manifest)
    write_json(output / "INVENTARIO_V4.json", {"checks": checks, "runtime": runtime, "historical_config": config,
                "model_files_verified": model_ok, "checkpoints": checkpoints,
                "modules": {"generator": "research.src.data.build_bank", "environment": "research.src.env.episode",
                            "simulator": "research.src.env.user_simulator", "writer": "research.src.policy.llm_controller",
                            "memory": "research.src.memory.store", "evaluator": "research.src.scoring.evidence",
                            "trainer": "research.src.training.episodic_grpo"}})
    result = {"status": "passed" if all(c["status"] == "verified" for c in checks) else "blocked_missing_evidence", "checks": checks}
    checkpoint_files = [Path(p) for c in checkpoints.values() for p in c["artifacts_sha256"]]
    record_stage(output, "inventory", result, [output / "INVENTARIO_V4.json", ROOT / "specs/SPEC_CIERRE_TESIS_ENTORNO_SIMULACION_RL_V4.md", *paths.values(), *[Path(p) for p in model_files], *checkpoint_files])
    return result


def test_environment(output):
    _new_stage(output, "test-environment")
    checked_stage(output, "inventory")
    from .src.closure_environment import run_environment_checks
    result = run_environment_checks()
    path = Path(output) / "PRUEBAS_ENTORNO.json"
    write_json(path, result)
    contracts = result.get("contracts", {})
    (Path(output) / "CONTRATOS_ENTORNO.md").write_text(
        "# Contratos auditados del entorno existente\n\n" + json.dumps(contracts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record_stage(output, "test-environment", result, [path, Path(output) / "CONTRATOS_ENTORNO.md"])


def diagnose(output, *, cohort="diagnostic", device="cuda", model_diagnostics=True):
    _new_stage(output, "diagnose")
    checked_stage(output, "test-environment")
    manifest = read_manifest(output)
    from .src.closure_environment import run_diagnostics, run_model_diagnostics
    from .src.closure_export import reexport_results
    from .src.closure_data import prepare_development
    development = prepare_development(Path(manifest["parent_run"]), Path(output), roots=10 if cohort == "smoke" else 50)
    result = run_diagnostics(Path(manifest["parent_run"]), Path(output))
    export = reexport_results(Path(manifest["parent_run"]), Path(output), manifest["run_id"])
    result.update(export=export, development=development, cohort=cohort,
                  new_llm_calls=0, model_diagnostics_status="not_executed",
                  historical_results_scope="exploratory_diagnostic_only")
    # A diagnosis may leave secondary E4/E5/E6 hypotheses open. Critical errors
    # explicitly marked by the environment diagnostic still block training.
    result.setdefault("critical_checks_passed", result.get("status") == "passed")
    if model_diagnostics:
        model_result = run_model_diagnostics(Path(manifest["parent_run"]), Path(output), device=device, max_llm_calls=600)
        result.update(model_diagnostics=model_result, model_diagnostics_status=model_result.get("status"),
                      new_llm_calls=model_result.get("new_llm_calls", model_result.get("llm_calls", 0)),
                      critical_checks_passed=model_result.get("critical_checks_passed", False))
    if export.get("status") != "passed":
        result["critical_checks_passed"] = False
    path = Path(output) / "DIAGNOSTICO_V4.json"
    write_json(path, result)
    evidence = [path, Path(output) / "DIAGNOSTICO_CAUSAL.md", Path(output) / "EXPORTACION_V4.json"]
    return record_stage(output, "diagnose", result, evidence)


def test_training(output, *, device="cuda", check_real_b1=True):
    _new_stage(output, "test-training")
    checked_stage(output, "diagnose", require_passed=False)
    checked_stage(output, "test-environment")
    from .src.closure_training import run_training_checks
    manifest = read_manifest(output)
    result = run_training_checks(Path(manifest["parent_run"]), Path(output), check_real_b1=check_real_b1, device=device)
    path = Path(output) / "PRUEBAS_ENTRENAMIENTO.json"
    write_json(path, result)
    evidence = [path, *(Path(output) / "training_checks").rglob("*.json"), *(Path(output) / "training_checks").glob("*.jsonl")]
    return record_stage(output, "test-training", result, evidence)


def pilot(output, *, device="cuda", updates=5, budget_seconds=300, signal_groups=0):
    manifest = _new_stage(output, "pilot-rl")
    if not 1 <= updates <= 5 or not 0 < budget_seconds <= 3600:
        raise ValueError("Pilot cap: 1..5 updates, 0 < seconds <= 3600")
    output = Path(output)
    plan_path = output / "PILOT_PLAN.json"
    if plan_path.exists():
        raise FileExistsError("Pilot plan already exists; preserve the first attempt")
    diagnosis = checked_stage(output, "diagnose", require_passed=False)
    training = checked_stage(output, "test-training")
    previous_calls = diagnosis.get("new_llm_calls", 0) + training.get("new_llm_calls", 6)
    model_evidence = Path(output) / "DIAGNOSTICO_MODELO.json"
    model_summary = diagnosis.get("model_diagnostics", {}).get("summary", {})
    d4 = model_summary.get("D4", {})
    d1 = model_summary.get("D1", {})
    # An improvement in the supported reference-memory branch demonstrates
    # utility for this fixed symbolic consumer, not competence of the B1 writer.
    useful = (isinstance(d4, dict) and isinstance(d1, dict)
              and d4.get("FJS", d4.get("mean_FJS", 0)) > d1.get("FJS", d1.get("mean_FJS", 0)))
    plan = {"training_seed": 101, "max_updates_per_branch": updates, "budget_seconds_per_branch": budget_seconds,
            "roots_per_batch": 2, "rollouts_per_root": 4, "learning_rate": 5e-6,
            "validation_budget_seconds_per_branch": 180, "validation_roots": 5,
            "checkpoint_save_budget_seconds": 120,
            "selection_validity_guards": {"utility_tolerance": 1e-6,
                "validity_guards": {"invalid_output_rate": .05, "unsupported_writes": .05}},
            "signal_diagnostic_groups": signal_groups,
            "consumer_useful": {"status": "verified" if useful else "not_identifiable_from_current_artifacts",
                "evidence_path": str(model_evidence.resolve()), "evidence_sha256": file_hash(model_evidence) if model_evidence.is_file() else None,
                "comparison": "D4 minus D1 FJS under fixed symbolic consumer", "observed": {"D4": d4, "D1": d1}},
            "train_bank": str(ROOT / "data/bank_1_1/train.jsonl"), "validation_bank": str(ROOT / "data/bank_1_1/validation.jsonl"),
            "max_new_llm_calls": max(0, 600 - previous_calls), "initial_call_budget": 600,
            "previous_new_llm_calls": previous_calls,
            "validation_steps": sorted(set([0, 1, updates])), "created_at": now(),
            "source_sha256": digest(source_hashes())}
    write_json(plan_path, plan)
    manifest["pilot_plan_sha256"] = file_hash(plan_path)
    write_json(output / "MANIFIESTO_V4.json", manifest)
    require_training_authorization(output)
    from .src.closure_training import run_pilot
    with execution_context(output, "training"):
        result = run_pilot(Path(manifest["parent_run"]), output, device=device,
                           updates=updates, roots_per_batch=2, budget_seconds=budget_seconds, authorization_dir=output,
                           signal_diagnostic_groups=signal_groups)
    result.setdefault("rl_evidence", result.get("learning_evidence", "inconclusive"))
    result.setdefault("decision", "close_inconclusive" if not result.get("signal_window_satisfied") else "repair_component")
    write_json(output / "DECISION_RL.json", result)
    _export_pilot_telemetry(output, result)
    return record_stage(output, "pilot-rl", result, [plan_path, output / "DECISION_RL.json"])


def _export_pilot_telemetry(output, result):
    """Keep new on-policy observations separate from the historical reexport."""
    rows = []
    for path in sorted((Path(output) / "pilot").glob("**/group_diagnostics.jsonl")):
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                value = json.loads(line)
                value.update(evidence_origin="closure_v4_pilot", source_path=str(path), source_line=line_number)
                rows.append(value)
    if rows:
        fields = sorted({k for row in rows for k in row})
        with (Path(output) / "CURVAS_RL_PILOTO_V4.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: canonical(v) if isinstance(v, (dict, list, bool)) or v is None else v for k, v in row.items()})
    if result.get("arms"):
        fields = ["method", "evidence_origin", "updates", "episodes", "training_seconds", "validation_seconds", "save_seconds", "input_tokens", "generated_tokens", "trained_tokens"]
        with (Path(output) / "PRESUPUESTOS_PILOTO_V4.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for method, arm in result["arms"].items():
                counters = arm.get("resource_counters", {})
                writer.writerow({"method": method, "evidence_origin": "closure_v4_pilot",
                    **{k: counters.get(k) for k in fields[2:]}})


def lock_protocol(output, plan_path):
    _new_stage(output, "lock-protocol")
    for stage in ("inventory", "test-environment", "test-training"):
        checked_stage(output, stage)
    pilot_result = checked_stage(output, "pilot-rl", require_passed=False)
    if pilot_result.get("decision") not in {"continue", "continuar"}:
        raise GateError("G4_SIGNAL_REQUIRED: a documented continue decision from the pilot is required")
    output = Path(output)
    if (output / "TEST_OPENING_V4.json").exists() or (output / "PROTOCOLO_BLOQUEADO.json").exists():
        raise GateError("PROTOCOL_ALREADY_FROZEN_OR_TEST_OPENED")
    from .src.closure_data import validate_protocol_plan
    plan = load_config(plan_path)
    audit = validate_protocol_plan(plan, historical_root=Path(read_manifest(output)["parent_run"]))
    if audit["status"] != "passed":
        raise GateError("PROTOCOL_NOT_READY: " + "; ".join(audit.get("errors", [])))
    artifacts = {**audit["artifacts_sha256"], str(Path(plan_path).resolve()): file_hash(plan_path)}
    locked = {"status": "locked", "locked_at": now(), "plan": plan, "audit": audit,
              "code_snapshot_sha256": digest(source_hashes()), "artifacts_sha256": artifacts}
    locked["sha256"] = digest(locked)
    write_json(output / "PROTOCOLO_BLOQUEADO.json", locked)
    manifest = read_manifest(output)
    manifest["locked_protocol_sha256"] = file_hash(output / "PROTOCOLO_BLOQUEADO.json")
    write_json(output / "MANIFIESTO_V4.json", manifest)
    return record_stage(output, "lock-protocol", {"status": "passed", "audit": audit}, [output / "PROTOCOLO_BLOQUEADO.json"])


def train_final(output, *, device="cuda"):
    _new_stage(output, "train-final")
    locked = verify_locked_protocol(output)
    if (Path(output) / "TEST_OPENING_V4.json").exists():
        raise GateError("TRAINING_AFTER_TEST_OPEN_FORBIDDEN")
    from .src.closure_training import run_final_training
    manifest = read_manifest(output)
    with execution_context(output, "training"):
        result = run_final_training(Path(manifest["parent_run"]), Path(output),
                    authorization_dir=Path(output), device=device,
                    budget_seconds=locked["plan"]["budgets"]["training_seconds_per_arm_seed"],
                    validation_budget_seconds=locked["plan"]["budgets"]["validation_seconds_per_arm_seed"],
                    seeds=tuple(locked["plan"]["training_seeds"]),
                    train_bank=locked["plan"]["train_bank"], validation_bank=locked["plan"]["validation_bank"])
    write_json(Path(output) / "ENTRENAMIENTO_FINAL_V4.json", result)
    rows = result.get("selected_checkpoints", result.get("checkpoints", []))
    if rows:
        write_json(Path(output) / "CHECKPOINTS_SELECCIONADOS.json", {"checkpoints": rows, "created_at": now()})
    return record_stage(output, "train-final", result, [Path(output) / "ENTRENAMIENTO_FINAL_V4.json", Path(output) / "CHECKPOINTS_SELECCIONADOS.json"])


def evaluate_locked_test(output, *, device="cuda"):
    """Open once, after locks and validation-selected weights have been verified."""
    output = Path(output)
    _new_stage(output, "evaluate-locked-test")
    locked, checkpoints = verify_locked_protocol(output), selected_checkpoints(output)
    checked_stage(output, "train-final")
    marker = output / "TEST_OPENING_V4.json"
    # Exclusive create prevents simultaneous independent openings of this run.
    with marker.open("x", encoding="utf-8") as stream:
        stream.write(canonical({"opened_at": now(), "locked_protocol_sha256": locked["sha256"],
                                 "selected_checkpoints_sha256": file_hash(output / "CHECKPOINTS_SELECCIONADOS.json")}) + "\n")
    from .src.policy.llm_controller import SpecLocalPolicy, LLMController
    from .src.evaluation.runner import run_episode
    from .src.training.campaign import _release_memory
    from .src.readers.symbolic import SymbolicReader
    trace_path = output / "TRAZAS_TEST_V4.jsonl"
    count = 0
    try:
        with execution_context(output, "locked-test"), trace_path.open("x", encoding="utf-8") as stream:
            for selected in checkpoints:
                backend = SpecLocalPolicy.from_checkpoint(selected["path"], trainable=False, device=device)
                if any(p.requires_grad for p in backend.model.parameters()):
                    raise GateError("TEST_TIME_WEIGHT_UPDATES_FORBIDDEN")
                controller = LLMController(backend)
                for split, key in (("test_id", "test_id_bank"), ("test_composition", "test_composition_bank")):
                    with Path(locked["plan"][key]).open() as bank:
                        for line in bank:
                            if not line.strip():
                                continue
                            episode = json.loads(line)
                            # PI1 and PI2 diagnostic arms share the primary B1.
                            arms = [("O2", None)]
                            if selected["method"] == "B1-v4" and split == "test_id":
                                arms.append(("O1", None))
                                if episode.get("opportunity", {}).get("eligible"):
                                    arms.extend([("O2", "I1"), ("O2", "I2")])
                            for condition, intervention in arms:
                                row = run_episode(episode, controller, method=selected["method"], condition=condition,
                                    intervention=intervention,
                                    forced_slot=episode.get("opportunity", {}).get("slot") if intervention else None,
                                    rollout_seed=selected["seed"])
                                row.update(run_id=read_manifest(output)["run_id"], checkpoint_id=selected["checkpoint_id"],
                                           training_seed=selected["seed"], split=split, future_probe_id="primary",
                                           parent_protocol_version=row.get("protocol_version"), protocol_version="closure_v4")
                                row["metrics"]["J_primary"] = row["metrics"]["C2"] * row["metrics"]["FJS"]
                                # The -1 current utility already identifies incorrect,
                                # unsupported or invalid answers; faithful deferral is 0.
                                row["metrics"]["current_error"] = int(row["rewards_by_component"]["components"]["current"] == -1)
                                probes = episode.get("future_probes")
                                if probes is not None:
                                    from copy import deepcopy
                                    from .src.scoring.evidence import Evaluator
                                    scored = []
                                    for probe in probes:
                                        if probe.get("applicable") is False:
                                            scored.append({"probe_id": probe["probe_id"], "status": "not_applicable"})
                                            continue
                                        probe_trace = deepcopy(row["trace"])
                                        probe_trace["reuse_observation"]["query"] = deepcopy(probe["query"])
                                        probe_trace["episode"]["future_query"] = deepcopy(probe["query"])
                                        probe_trace["episode"]["truth"]["future_value"] = probe["factual_value"]
                                        probe_trace["future_response"] = SymbolicReader().read(probe_trace["final_memory"], probe["query"])
                                        score = Evaluator().score_future(probe_trace)
                                        scored.append({"probe_id": probe["probe_id"], "query": probe["query"],
                                                       "response": probe_trace["future_response"], "score": score})
                                    # Fixed denominator per episode, independent of method.
                                    applicable = [p for p in scored if "score" in p]
                                    row["scope_probes"] = scored
                                    row["metrics"]["scope_error"] = int(any(p["score"]["utility"] == -1 for p in applicable)) if applicable else None
                                row["condition_id"] = digest({"observation": condition, "intervention": intervention,
                                    "reader": "fixed_symbolic", "nonresponse_rate": None, "protocol": locked["sha256"]})
                                stream.write(canonical(row) + "\n")
                                stream.flush()
                                count += 1
                del backend, controller
                _release_memory()
        result = {"status": "passed", "rows": count, "traces_sha256": file_hash(trace_path), "human_audit": "not_executed"}
    except BaseException as error:
        write_json(output / "EVALUACION_TEST_V4.json", {"status": "failed", "rows_completed": count,
                   "error": str(error), "partial_outputs_preserved": True})
        raise
    write_json(output / "EVALUACION_TEST_V4.json", result)
    from .src.closure_export import flatten_episode, _cell
    flat = []
    with trace_path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = flatten_episode(json.loads(line), read_manifest(output)["run_id"], source_path=trace_path, source_line=line_number)
            row["evidence_origin"] = "closure_v4_locked_test"
            flat.append(row)
    fields = sorted({k for row in flat for k in row})
    with (output / "RESULTADOS_TEST_V4.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: _cell(v) for k, v in row.items()} for row in flat)
    return record_stage(output, "evaluate-locked-test", result, [trace_path, output / "EVALUACION_TEST_V4.json"])


def _verification_matrix(output, manifest):
    fields = ["check_id", "variable", "expected", "observed", "status", "evidence_path", "evidence_sha256", "code_location", "affected_claim", "severity", "action"]
    with (output / "MATRIZ_VERIFICACION.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, entry in manifest.get("stages", {}).items():
            stage = load_config(entry["path"])
            for item in stage["result"].get("checks", []):
                evidence = item.get("evidence", item.get("evidence_path", entry["path"]))
                evidence = evidence if isinstance(evidence, str) else canonical(evidence)
                candidate_path = Path(evidence)
                try:
                    is_file = len(evidence) < 2000 and candidate_path.is_file()
                except OSError:
                    is_file = False
                if not is_file:
                    evidence = entry["path"]
                    candidate_path = Path(evidence)
                actual_hash = file_hash(candidate_path)
                status = {"passed": "verified", "skipped": "not_executed", "partial": "not_identifiable_from_current_artifacts"}.get(item.get("status"), item.get("status", "not_executed"))
                writer.writerow({"check_id": item.get("id", item.get("check_id")), "variable": item.get("variable", name),
                    "expected": item.get("expected", "verified under stated fixture"), "observed": canonical(item.get("details", item)),
                    "status": status, "evidence_path": evidence, "evidence_sha256": actual_hash,
                    "code_location": item.get("code_location"), "affected_claim": item.get("affected_claim", "PI1/PI2/PI3"),
                    "severity": item.get("severity", "critical"), "action": item.get("action")})


def report(output):
    output = Path(output)
    manifest = read_manifest(output)
    stages, integrity_errors = {}, {}
    for name in manifest.get("stages", {}):
        try:
            stages[name] = checked_stage(output, name, require_passed=False)
        except GateError as error:
            stages[name] = {"status": "failed", "reason": str(error)}
            integrity_errors[name] = str(error)
    env = stages.get("test-environment", {})
    training = stages.get("test-training", {})
    decision = _read(output / "DECISION_RL.json", {"status": "not_executed", "decision": "inconclusive", "rl_evidence": "not_executed",
                     "reason": "Pilot not executed; inspect environment/training gates before continuing"})
    if not (output / "DECISION_RL.json").exists():
        write_json(output / "DECISION_RL.json", decision)
    infrastructure = env.get("status", "blocked_missing_evidence")
    if integrity_errors:
        infrastructure = "failed"
    if infrastructure not in {"passed", "failed", "blocked_missing_evidence"}:
        infrastructure = "blocked_missing_evidence"
    analyses = {"status": "not_executed", "reason": "New locked test not executed", "planned_claims": 11,
                "individual_confidence": 1 - .05 / 11, "human_audit": "not_executed", "historical_results": "preserved_exploratory"}
    if stages.get("evaluate-locked-test", {}).get("status") == "passed":
        analyses = analyze_locked_results(output)
    write_json(output / "ANALISIS_V4.json", analyses)
    _verification_matrix(output, manifest)
    limitations = ["Evidencia sintética; human_audit: not_executed. No se realizó un estudio con usuarios reales.",
                   "Los tests históricos observados se usan como diagnóstico exploratorio, sin confirmación independiente de v4.",
                   "La ausencia de .git impide reconstruir un diff histórico exacto; se identifican fuentes actuales por SHA256."]
    if integrity_errors:
        limitations.append("Integridad fallida: " + canonical(integrity_errors))
    for name in ("diagnose", "test-training", "pilot-rl", "lock-protocol", "train-final", "evaluate-locked-test"):
        result = stages.get(name)
        if result is None or result.get("status") != "passed":
            limitations.append(f"{name}: {result.get('status') if result else 'not_executed'}. Consultar evidencia y checks de la etapa.")
    (output / "LIMITACIONES_V4.md").write_text("# Limitaciones de esta ejecución\n\n" + "\n\n".join(limitations) + "\n", encoding="utf-8")
    model = stages.get("diagnose", {}).get("model_diagnostics", {})
    competence = model.get("summary", {})
    model_details = ""
    if competence:
        model_details = "\n## Diagnóstico B1 sin entrenamiento\n\n" + "```json\n" + json.dumps(competence, ensure_ascii=False, indent=2) + "\n```\n\nEstas métricas son de desarrollo y no estiman eficacia general.\n"
    pilot_details = ""
    if decision.get("arms"):
        pilot_details = "\n## Piloto RL ejecutado\n\n"
        for method, arm in decision["arms"].items():
            pilot_details += (f"{method}: {arm.get('resource_counters', {}).get('updates', 0)} actualizaciones reales, "
                              f"paso seleccionado {arm.get('best_step')}, parada {arm.get('stop_reason')}.\n\n")
        pilot_details += f"Grupos observados: {decision.get('groups_observed')}; fracción sin señal de tarea: {decision.get('zero_task_signal_fraction')}.\n"
    response = ("# Respuesta a Vicent — cierre v4\n\n"
        f"Infraestructura y contratos: **{infrastructure}** en los fixtures ejecutados. "
        f"Auditoría de entrenamiento: **{training.get('status', 'not_executed')}**. "
        f"Evidencia nueva de aprendizaje RL: **{decision.get('rl_evidence', 'inconclusive' if 'pilot-rl' in stages else 'not_executed')}**. "
        "Validez externa: alcance sintético, sin revisión humana independiente.\n\n"
        "## PI1 — Evidencia disponible\n\n"
        "La reexportación histórica conserva O1/O2 y sus denominadores. El efecto suelo histórico de B0 limita su interpretación. "
        "La comparación nueva B1 O2−O1 solo puede afirmarse con el test v4 bloqueado; "
        f"su análisis está {analyses.get('status')}. No se sustituye por el rendimiento del controlador de reglas.\n\n"
        "## PI2 — Aclaración y persistencia\n\n"
        "I1 e I2 deben compartir evidencia previa y respuesta actual; difieren en la disponibilidad de aclaración para WRITE. "
        "Los contratos y la separación SQLite de sesiones se documentan en PRUEBAS_ENTORNO.json y CONTRATOS_ENTORNO.md. "
        "El beneficio histórico de persistencia y el daño observado en casos claros se conservan como antecedentes, "
        "sin atribuirlos a selección autónoma óptima de preguntas. Los resultados nuevos se identifican en ANALISIS_V4.json.\n\n"
        "## PI3 — Qué añade RL\n\n"
        f"Piloto: {decision.get('status', 'not_executed')}; decisión: {decision.get('decision', 'inconclusive')}. "
        "No se demuestra aprendizaje por tener un adaptador ni por seleccionar paso cero. "
        "Sin contrastes nuevos y guardas con precisión suficiente, no se declara superioridad sobre más SFT o RL inmediato. "
        "La selección longitudinal de RL inmediato también utiliza información futura de validación.\n\n"
        "Los estados y las dependencias pendientes se conservan en MATRIZ_VERIFICACION.csv y LIMITACIONES_V4.md. "
        "La campaña full_exploratory_001 se conserva intacta.\n" + model_details + pilot_details)
    (output / "RESPUESTA_VICENT.md").write_text(response, encoding="utf-8")
    result = {"status": "passed", "infrastructure": infrastructure, "model_competence": "see episode results; no generic correctness seal",
              "rl_evidence": decision.get("rl_evidence", "not_executed" if "pilot-rl" not in stages else "inconclusive"),
              "external_validity": "synthetic_only", "human_audit": "not_executed", "report": str(output / "RESPUESTA_VICENT.md")}
    result["stage_integrity_errors"] = integrity_errors
    write_json(output / "DICTAMEN_V4.json", result)
    return result


def analyze_locked_results(output):
    from .src.closure_statistics import paired_contrast
    locked = verify_locked_protocol(output)
    with (Path(output) / "TRAZAS_TEST_V4.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    claims = []
    specifications = [
        ("PI1_v4", "C2", {"method": "B1-v4", "split": "test_id", "observation_condition": "O2", "intervention": None},
         {"method": "B1-v4", "split": "test_id", "observation_condition": "O1", "intervention": None}),
        ("PI2_v4", "FJS", {"method": "B1-v4", "split": "test_id", "intervention": "I2"},
         {"method": "B1-v4", "split": "test_id", "intervention": "I1"})]
    for name, comparator in (("PI3a_v4", "B3-v4"), ("PI3b_v4", "B2-I-v4"), ("PI3c_v4", "B1-v4")):
        specifications.append((name, "J_primary", {"method": "B2-L-v4", "split": "test_composition", "intervention": None},
                               {"method": comparator, "split": "test_composition", "intervention": None}))
    for name, metric, candidate, control in specifications:
        result = paired_contrast(rows, candidate, control, metric)
        result["claim_id"] = name
        claims.append(result)
        if name in {"PI3a_v4", "PI3b_v4"}:
            for guard in ("current_error", "unsupported_write_episode", "scope_error"):
                try:
                    value = paired_contrast(rows, candidate, control, guard, harm_margin=locked["plan"]["harm_margin"])
                except ValueError as error:
                    value = {"status": "blocked_missing_evidence", "reason": str(error), "metric": guard,
                             "guard": {"passed": False}}
                value["claim_id"] = name.replace("_v4", "") + "_" + guard
                claims.append(value)
    return {"status": "executed", "claims": claims, "planned_claims": 11, "individual_confidence": 1 - .05 / 11,
            "human_audit": "not_executed", "analysis_seed": 7331,
            "rl_advantage_claims": {name: all(c.get("guard", {}).get("passed", False) for c in claims
                                            if c["claim_id"] in {name.replace("_v4", "") + "_" + g for g in ("current_error", "unsupported_write_episode", "scope_error")})
                 and next(c for c in claims if c["claim_id"] == name)["interval"][0] > 0 for name in ("PI3a_v4", "PI3b_v4")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None, help="Closure run directory; inventory requires a new directory")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "test-environment", "diagnose", "test-training", "pilot-rl", "lock-protocol", "train-final", "evaluate-locked-test", "report"):
        command = sub.add_parser(name)
        command.add_argument("--output", dest="command_output")
        if name == "inventory":
            command.add_argument("--parent-run", default=str(DEFAULT_PARENT))
            command.add_argument("--run-id")
        if name == "diagnose":
            command.add_argument("--cohort", choices=("smoke", "diagnostic"), default="diagnostic")
            command.add_argument("--no-model-diagnostics", action="store_true", help="Offline diagnosis only; cannot release RL")
        if name in {"diagnose", "test-training", "pilot-rl", "train-final", "evaluate-locked-test"}:
            command.add_argument("--device", default="cuda")
        if name == "test-training":
            command.add_argument("--no-real-b1", action="store_true", help="Record real-checkpoint checks as not executed; cannot release training")
        if name == "pilot-rl":
            command.add_argument("--updates", type=int, default=5)
            command.add_argument("--budget-seconds", type=float, default=300)
            command.add_argument("--signal-groups", type=int, default=0, help="Additional frozen-policy groups, included in the preflight call budget")
        if name == "lock-protocol":
            command.add_argument("--plan", required=True)
    args = parser.parse_args(argv)
    output = Path(args.command_output or args.output or ROOT / "runs" / ("closure_v4_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
    try:
        if args.command == "inventory":
            result = inventory(args.parent_run, output, run_id=args.run_id)
        elif args.command == "test-environment":
            result = test_environment(output)
        elif args.command == "diagnose":
            result = diagnose(output, cohort=args.cohort, device=args.device, model_diagnostics=not args.no_model_diagnostics)
        elif args.command == "test-training":
            result = test_training(output, device=args.device, check_real_b1=not args.no_real_b1)
        elif args.command == "pilot-rl":
            result = pilot(output, device=args.device, updates=args.updates, budget_seconds=args.budget_seconds, signal_groups=args.signal_groups)
        elif args.command == "lock-protocol":
            result = lock_protocol(output, args.plan)
        elif args.command == "train-final":
            result = train_final(output, device=args.device)
        elif args.command == "evaluate-locked-test":
            result = evaluate_locked_test(output, device=args.device)
        else:
            result = report(output)
        # Full evidence stays in files; terminal output remains reviewable.
        print(json.dumps({"command": args.command, "output": str(output), "status": result.get("status"),
                          "report": result.get("report")}, ensure_ascii=False))
        return 0 if result.get("status") in {"passed", "executed"} else 2
    except (GateError, FileExistsError, ValueError, FileNotFoundError) as error:
        print(json.dumps({"command": args.command, "output": str(output), "status": "blocked_missing_evidence", "reason": str(error)}, ensure_ascii=False))
        return 2
    except Exception as error:
        if (output / "MANIFIESTO_V4.json").is_file():
            failure = {"status": "failed", "error_type": type(error).__name__, "reason": str(error),
                       "partial_outputs_preserved": True, "failed_at": now()}
            path = output / ("FALLO_" + args.command + ".json")
            write_json(path, failure)
            try:
                record_stage(output, args.command, failure, [path])
            except FileExistsError:
                pass
        print(json.dumps({"command": args.command, "output": str(output), "status": "failed", "reason": str(error)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
