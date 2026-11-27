"""Gated local campaigns with durable progress and validation-only selection.

Completed branches can be reused after artifact verification. A partial branch
is never silently restarted with a new optimizer or overwritten.
"""
from __future__ import annotations

from collections import defaultdict
import gc
import json
from pathlib import Path
from statistics import mean
import time

from llm_local_001.model import artifact_hashes
from ..artifacts import ROOT, digest, load_config
from ..protocol import require_gate
from ..policy.llm_controller import SpecLocalPolicy, LLMController
from ..data.build_bank import teacher_trajectories
from ..data.semantic_partition_v2 import assert_semantic_split_isolation_v2
from ..evaluation.runner import run_episode
from .sft import train_sft, _write_json
from .episodic_grpo import train_grpo, load_b1_reference
from .rollouts import make_rollout_collector


def load_bank(path, split):
    episodes = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not episodes or any(e["split"] != split for e in episodes):
        raise ValueError("Campaign bank must contain only " + split)
    return episodes


def aggregate_validation(rows):
    roots = defaultdict(list)
    for row in rows:
        roots[(row["scenario_family"], row["root_id"])].append(row)
    if not roots:
        raise ValueError("Validation produced no scheduled rows")
    families = defaultdict(list)
    for (family, _), group in roots.items():
        worlds = defaultdict(lambda: defaultdict(list))
        for index, row in enumerate(group):
            worlds[row.get("latent_world_id", "unrecorded")][row.get("realization_id", index)].append(row)
        def hierarchical_mean(value):
            return mean(mean(mean(value(row) for row in realization)
                             for realization in world.values()) for world in worlds.values())
        families[family].append({
            "utility": hierarchical_mean(lambda r: r["rewards_by_component"]["R_long"]),
            "unsupported_writes": hierarchical_mean(lambda r: r["metrics"]["unsupported_write_episode"]),
            "questions": hierarchical_mean(lambda r: r["metrics"]["questions"]),
            "future_success": hierarchical_mean(lambda r: r["metrics"]["FJS"]),
            "current_success": hierarchical_mean(lambda r: r["metrics"]["C2"]),
            "invalid_output_rate": hierarchical_mean(lambda r: r["metrics"]["invalid_outputs"] / 3),
        })
    return {metric: mean(mean(r[metric] for r in group) for group in families.values())
            for metric in next(iter(families.values()))[0]}


def validation_callback(episodes, *, root=None, log_directory=None, progress_every=10, batch_size=8):
    if not episodes or any(e["split"] != "validation" for e in episodes):
        raise ValueError("Checkpoint selection requires a validation-only bank")
    if batch_size < 1 or progress_every < 1:
        raise ValueError("Invalid validation progress/batch configuration")
    log_directory = Path(log_directory) if log_directory is not None else None
    if log_directory is not None:
        log_directory.mkdir(parents=True, exist_ok=True)
    calls = [0]
    bank_hash = digest(episodes)

    def validate(backend):
        calls[0] += 1
        started = time.monotonic()
        if batch_size > 1:
            from ..evaluation.batched import evaluate_batched
            evaluated = evaluate_batched(episodes, backend, method="validation", batch_size=batch_size,
                                         root=root, seed=backend.config.seed)
        else:
            evaluated = (run_episode(episode, LLMController(backend), method="validation", root=root,
                                     rollout_seed=backend.config.seed) for episode in episodes)
        rows = []
        stream = None
        try:
            if log_directory is not None:
                trace_path = log_directory / f"validation_{calls[0]:04d}.jsonl"
                if trace_path.exists():
                    raise FileExistsError("Validation attempts must preserve their original traces")
                stream = trace_path.open("w", encoding="utf-8")
            for index, row in enumerate(evaluated, start=1):
                rows.append(row)
                if stream is not None:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                if index % progress_every == 0 or index == len(episodes):
                    elapsed = time.monotonic() - started
                    print(json.dumps({"validation_progress": {"call": calls[0], "episodes": index,
                        "scheduled_episodes": len(episodes), "batch_size": batch_size,
                        "elapsed_seconds": elapsed,
                        "estimated_remaining_seconds": elapsed / index * (len(episodes) - index)}}), flush=True)
        finally:
            if stream is not None:
                stream.close()
        if len(rows) != len(episodes):
            raise RuntimeError("Validation omitted scheduled episodes")
        metrics = aggregate_validation(rows)
        if log_directory is not None:
            _write_json(log_directory / f"validation_{calls[0]:04d}.json",
                        {"status": "executed", "metrics": metrics, "episodes": len(rows),
                         "bank_hash": bank_hash, "batch_size": batch_size,
                         "elapsed_seconds": time.monotonic() - started})
        return metrics
    return validate


def _completed_run(target):
    """Verify and reuse an already completed branch; partials require recovery."""
    target = Path(target)
    if not target.exists() or not any(target.iterdir()):
        return None
    summary = target / "training.json"
    if not summary.is_file():
        raise RuntimeError("PARTIAL_RUN_REQUIRES_EXPLICIT_RECOVERY: " + str(target))
    result = json.loads(summary.read_text())
    if result.get("status") != "executed":
        raise RuntimeError("PARTIAL_RUN_REQUIRES_EXPLICIT_RECOVERY: " + str(target))
    for field in ("selected_checkpoint", "final_checkpoint"):
        checkpoint = Path(result[field])
        manifest = json.loads((checkpoint / "local_checkpoint.json").read_text())
        if artifact_hashes(checkpoint) != manifest["artifacts_sha256"]:
            raise RuntimeError("Completed checkpoint changed: " + str(checkpoint))
    return result


def _record_plan(destination, plan):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "campaign_plan.json"
    if path.exists() and load_config(path) != plan:
        raise RuntimeError("CAMPAIGN_PLAN_CHANGED: preserve prior results and choose a new destination")
    _write_json(path, plan)


def _release_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _selected_metrics(result):
    step = result["best_step"]
    found = [row["validation"] for row in result["history"] if row["step"] == step and "validation" in row]
    if len(found) != 1:
        raise ValueError("Selected B1 validation result is missing or ambiguous")
    return dict(found[0])


def ceiling_diagnostic(metrics, *, future_threshold=.95, write_threshold=.01):
    detected = metrics.get("future_success", -1) >= future_threshold and metrics["unsupported_writes"] <= write_threshold
    return {"status": "ceiling_detected" if detected else "margin_available",
            "future_threshold": future_threshold, "unsupported_write_threshold": write_threshold,
            "selected_validation": metrics,
            "interpretation": "No weaken SFT or modify the bank; report incremental comparisons in their executed scope"}


def run_sft_campaign(model_path, train_bank, validation_bank, destination, *, device="cuda", root=None):
    root = Path(root or ROOT)
    require_gate("G2", root)
    config = load_config(root / "configs/protocol.yaml")
    episodes, validation = load_bank(train_bank, "train"), load_bank(validation_bank, "validation")
    assert_semantic_split_isolation_v2({"train": episodes, "validation": validation})
    examples = list(teacher_trajectories(episodes))
    epoch_interval = config["sft"].get("checkpoint_every_epochs", 1)
    if not isinstance(epoch_interval, int) or epoch_interval < 1:
        raise ValueError("SFT checkpoint_every_epochs must be a positive integer")
    checkpoint_every = len(examples) * epoch_interval
    progress_every = config.get("runtime", {}).get("training_progress_every_updates", 10)
    batch_size = config.get("runtime", {}).get("evaluation_batch_size", 8)
    plan = {"method": "B1", "seeds": config["seeds"]["training"], "epochs": config["sft"]["epochs"],
            "examples": len(examples), "scheduled_updates_per_seed": len(examples) * config["sft"]["epochs"],
            "checkpoint_every_examples": checkpoint_every, "validation_batch_size": batch_size,
            "train_bank_hash": digest(episodes), "validation_bank_hash": digest(validation),
            "protocol_hash": digest(config), "model_path": str(Path(model_path).resolve())}
    destination = Path(destination)
    _record_plan(destination, plan)
    result = {}
    for seed in config["seeds"]["training"]:
        target = destination / str(seed) / "B1"
        existing = _completed_run(target)
        if existing is not None:
            result[str(seed)] = existing
            _write_json(destination / "sft_campaign.json", result)
            continue
        policy = SpecLocalPolicy(model_path, device=device, precision=config["model"]["precision"],
                                 seed=seed, gradient_checkpointing=config.get("runtime", {}).get("gradient_checkpointing", True))
        try:
            out = train_sft(policy, examples, target,
                validation_callback(validation, root=root, log_directory=destination / str(seed) / "B1_validation", batch_size=batch_size),
                epochs=config["sft"]["epochs"], checkpoint_every=checkpoint_every, progress_every=progress_every,
                learning_rate=config["sft"]["learning_rate"], seed=seed, protocol_root=root)
            out["ceiling_diagnostic"] = ceiling_diagnostic(_selected_metrics(out))
            result[str(seed)] = out
            _write_json(destination / "sft_campaign.json", result)
        finally:
            del policy
            _release_memory()
    return result


def run_continuations(sft_campaign, train_bank, validation_bank, destination, *, device="cuda", root=None):
    root = Path(root or ROOT)
    require_gate("G3", root)
    config = load_config(root / "configs/protocol.yaml")
    parents = load_config(sft_campaign)
    episodes, validation = load_bank(train_bank, "train"), load_bank(validation_bank, "validation")
    assert_semantic_split_isolation_v2({"train": episodes, "validation": validation})
    parent_plan_path = Path(sft_campaign).parent / "campaign_plan.json"
    if not parent_plan_path.is_file() or load_config(parent_plan_path).get("validation_bank_hash") != digest(validation):
        raise ValueError("Selected B1 validation cache does not match this frozen validation bank")
    examples = list(teacher_trajectories(episodes))
    batch_size = config.get("runtime", {}).get("evaluation_batch_size", 8)
    saturation_action = config.get("saturation", {}).get("action", "run_all")
    if saturation_action not in {"run_all", "stop_continuations"}:
        raise ValueError("Saturation action must be fixed before running continuations")
    cap = config["budget"]["post_sft_accelerator_seconds_cap"]
    plan = {"methods": ["B2-L", "B3", "B2-I"], "seeds": config["seeds"]["training"],
            "budget_per_branch_seconds": cap, "sft_checkpoint_every_examples": len(examples),
            "rl_checkpoint_every_updates": config["rl"]["checkpoint_every_updates"],
            "validation_batch_size": batch_size, "initial_validation_reuse": "selected_B1_same_artifacts",
            "reference_execution": config.get("runtime", {}).get("reference_execution", "stored_device"),
            "saturation_action": saturation_action, "protocol_hash": digest(config),
            "train_bank_hash": digest(episodes), "validation_bank_hash": digest(validation),
            "parents": {str(seed): parents[str(seed)]["selected_checkpoint"] for seed in config["seeds"]["training"]}}
    destination = Path(destination)
    _record_plan(destination, plan)
    result = {}
    for seed in config["seeds"]["training"]:
        parent = parents[str(seed)]["selected_checkpoint"]
        initial_metrics = _selected_metrics(parents[str(seed)])
        ceiling = ceiling_diagnostic(initial_metrics)
        result[str(seed)] = {}
        for method in ("B2-L", "B3", "B2-I"):
            target = destination / str(seed) / method
            existing = _completed_run(target)
            if existing is not None:
                result[str(seed)][method] = existing
                _write_json(destination / "continuations_campaign.json", result)
                continue
            if saturation_action == "stop_continuations" and ceiling["status"] == "ceiling_detected":
                result[str(seed)][method] = {"status": "not_executed", "reason": "ceiling_detected", "diagnostic": ceiling}
                _write_json(destination / "continuations_campaign.json", result)
                continue
            policy = SpecLocalPolicy.from_checkpoint(parent, trainable=True, device=device)
            reference = None
            try:
                callback = validation_callback(validation, root=root,
                    log_directory=destination / str(seed) / (method + "_validation"), batch_size=batch_size)
                if method == "B3":
                    out = train_sft(policy, examples, target, callback, continuation=True, seed=seed,
                        budget_seconds=cap, learning_rate=config["sft"]["learning_rate"], protocol_root=root,
                        checkpoint_every=len(examples), initial_validation_metrics=initial_metrics,
                        progress_every=config.get("runtime", {}).get("training_progress_every_updates", 10))
                else:
                    reference = load_b1_reference(parent, device=config.get("runtime", {}).get("reference_device", "cpu"))
                    collect = make_rollout_collector(destination / str(seed) / (method + "_rollouts"), method=method, root=root)
                    out = train_grpo(policy, reference, episodes, collect, target, callback,
                        horizon="long" if method == "B2-L" else "short", seed=seed,
                        updates=config["rl"]["max_updates"], roots_per_batch=config["rl"]["roots_per_effective_batch"],
                        rollouts_per_root=config["rl"]["rollouts_per_root"], budget_seconds=cap,
                        learning_rate=config["rl"]["learning_rate"], kl_coefficient=config["rl"]["kl_coefficient"],
                        clip_epsilon=config["rl"]["clip_epsilon"], checkpoint_every=config["rl"]["checkpoint_every_updates"],
                        stop_after_zero_signal_updates=config["rl"]["stop_after_zero_signal_updates"], protocol_root=root,
                        initial_validation_metrics=initial_metrics,
                        reference_execution=config.get("runtime", {}).get("reference_execution", "stored_device"))
                out["initial_validation_source_checkpoint"] = parent
                out["initial_validation_bank_hash"] = digest(validation)
                result[str(seed)][method] = out
                _write_json(destination / "continuations_campaign.json", result)
            finally:
                del policy
                if reference is not None:
                    del reference
                _release_memory()
    return result
