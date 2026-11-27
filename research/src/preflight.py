"""Read-only G0 inventory. Missing approved candidates are a failed gate."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from .artifacts import ROOT, digest, load_config, now, write_json


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()[-2000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def inventory(root=None, model_paths=()):
    root = Path(root or ROOT)
    repo = root.parent
    config = load_config(root / "configs/protocol.yaml")
    versions = {}
    for package in ("torch", "transformers", "peft", "trl", "PyYAML", "pytest"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    paths = list(model_paths) + list((repo / "models").glob("*"))
    for name in ("Qwen2.5-0.5B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-1.5B-Instruct"):
        cache = Path.home() / ".cache/huggingface/hub" / ("models--Qwen--" + name) / "snapshots"
        paths.extend(cache.glob("*"))
    models = []
    for path in sorted(set(Path(p).resolve() for p in paths)):
        if not (path / "config.json").is_file():
            continue
        metadata = json.loads((path / "config.json").read_text())
        models.append({"path": str(path), "model_type": metadata.get("model_type"),
                       "hidden_size": metadata.get("hidden_size"), "layers": metadata.get("num_hidden_layers"),
                       "declared_origin": metadata.get("_name_or_path"),
                       "weight_bytes": sum(p.stat().st_size for p in path.glob("*.safetensors")),
                       "candidate_name_matches": path.name in ("Qwen2.5-0.5B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-1.5B-Instruct")})
    return {"created_at": now(), "python": sys.version, "interpreter": sys.executable,
            "hardware": {"platform": platform.platform(), "cpu": command(["lscpu"]),
                         "ram_bytes": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
                         "logical_cpus": os.cpu_count(), "nvidia_smi": command(["nvidia-smi"]),
                         "gpu": command(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"])},
            "versions": versions, "models": models,
            "git": command(["git", "-C", str(repo), "rev-parse", "HEAD"]),
            "reuse": {"sqlite": "memory.store.Store: owner protection and base tables",
                      "temporal": "rl_longitudinal.temporal_memory: historical semantics retained; new attribute/scope schema is versioned",
                      "llm": "llm_local_001.model.LocalPolicy: offline HF, LoRA, checkpoints and token likelihoods",
                      "historical_results": "reports/ untouched"},
            "protocol_hash": digest(config)}


def run_g0(root=None, model_path=None, device=None):
    root = Path(root or ROOT)
    config = load_config(root / "configs/protocol.yaml")
    runtime = config.get("runtime", {})
    device = device or runtime.get("device", "cpu")
    if model_path is None and runtime.get("model_path"):
        model_path = Path(runtime["model_path"])
        if not model_path.is_absolute():
            model_path = root.parent / model_path
    report = inventory(root, [model_path] if model_path else [])
    write_json(root / "reports/inventory.json", report)
    selected = config["model"]["selected_id"]
    approved = {config["model"]["preferred"], config["model"]["technical_fallback"]}
    smoke = {"status": "not_executed", "reason": "No approved locally pinned model selected"}
    if selected in approved and model_path:
        from .training.episodic_grpo import technical_smoke
        output = root / "reports/g0_smoke"
        index = 2
        while output.exists() and any(output.iterdir()):
            output = root / f"reports/g0_smoke_{index:03d}"
            index += 1
        smoke = technical_smoke(model_path, output, device=device)
        provenance = smoke.get("candidate_provenance", {})
        if any(provenance.get(key) != config["model"][field] for key, field in
               (("model_id", "selected_id"), ("revision", "revision"), ("tokenizer_revision", "tokenizer_revision"))):
            smoke.update(status="failed", g0_eligible=False, error="Pinned model/tokenizer identity differs from selected protocol")
        if smoke.get("precision") != config["model"]["precision"]:
            smoke.update(status="failed", g0_eligible=False, error="Actual model precision differs from protocol")
    lock = {"status": "blocked", "created_at": now(), "protocol_hash": digest(config),
            "hardware": report["hardware"], "versions": {"python": sys.version, **report["versions"],
                "trainer": "local_episodic_grpo:" + digest({"source": (root / "src/training/episodic_grpo.py").read_text()})},
            "model": config["model"], "adaptation": config["adaptation"],
            "training_plan": {"batch": config["rl"]["roots_per_effective_batch"], "rollouts": config["rl"]["rollouts_per_root"],
                              "device": device, "reference_device": "cpu", "gradient_checkpointing": True,
                              "microbatch": 1, "accumulation": config["rl"]["roots_per_effective_batch"] * config["rl"]["rollouts_per_root"], "token_limits": config["limits"],
                              "accelerator_seconds_cap": config["budget"]["post_sft_accelerator_seconds_cap"]},
            "technical_smoke": smoke,
            "reason": "TECHNICAL_LOCK_REQUIRED: candidate, revision, tokenizer, measured resource budget and successful paired updates required"}
    # A successful update alone does not close G0: all pinned fields and measured
    # resources must also be reviewed and frozen. Never infer them from a filename.
    # Try closure only with the pinned fields, measured resource allocation and
    # matching real update evidence. Missing requirements restore blocked state.
    lock["status"] = "passed"
    write_json(root / "configs/environment.lock.yaml", lock)
    from .protocol import gate_status
    failures = gate_status(root)["G0"]["reasons"]
    if failures:
        lock["status"], lock["missing_requirements"] = "blocked", failures
    else:
        lock["reason"] = "Technical candidate and resource checks passed"
    write_json(root / "configs/environment.lock.yaml", lock)
    return lock
