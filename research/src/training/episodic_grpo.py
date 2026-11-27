"""Explicit episodic GRPO over three actual actor prefixes and deferred returns.

The collector executes the environment. This module receives rewards only after
execution, never hidden scenario labels as policy inputs. A rollout's memory and
sampling seed must be independent of all other members of its group.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time

from llm_local_001.model import Generation, artifact_hashes
from research.src.policy.llm_controller import SpecLocalPolicy
from research.src.training.sft import (
    _gate, _optimizer_step, _prepare, _validate, _write_json, parameter_hash,
    _initial_metrics, _save_training_state, RunJournal, TrainingBudget, CheckpointSaveBudget,
    selection_is_better, selection_guards_passed,
)

# Official config.json files, checked 2026-10-23; see research/TRAINING.md.
QWEN_ARCHITECTURES = {
    "Qwen/Qwen2.5-0.5B-Instruct": {"hidden_size": 896, "intermediate_size": 4864,
        "num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2},
    "Qwen/Qwen2.5-3B-Instruct": {"hidden_size": 2048, "intermediate_size": 11008,
        "num_hidden_layers": 36, "num_attention_heads": 16, "num_key_value_heads": 2},
    "Qwen/Qwen2.5-1.5B-Instruct": {"hidden_size": 1536, "intermediate_size": 8960,
        "num_hidden_layers": 28, "num_attention_heads": 12, "num_key_value_heads": 2},
}


def inspect_candidate(model_path):
    """Check independent local provenance and architecture, never a directory name.

    A download manifest records repo, pinned revision and SHA256 for every model
    and tokenizer artifact. G0 still reviews/fixes the full runtime and budget.
    """
    model_path = Path(model_path).resolve()
    config = json.loads((model_path / "config.json").read_text())
    matches = []
    common = {"model_type": "qwen2", "vocab_size": 151936,
              "tie_word_embeddings": True, "max_position_embeddings": 32768}
    for model_id, architecture in QWEN_ARCHITECTURES.items():
        if all(config.get(key) == value for key, value in {**common, **architecture}.items()):
            matches.append(model_id)
    result = {"architecture_matches": matches, "verified": False, "reasons": []}
    if not matches:
        result["reasons"].append("architecture does not match an allowed backbone")
        return result
    manifest_path = model_path / "download_manifest.json"
    if not manifest_path.is_file():
        result["reasons"].append("pinned download_manifest.json is absent")
        return result
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("repo") not in matches:
        result["reasons"].append("manifest model identity differs from architecture")
    if not re.fullmatch(r"[0-9a-f]{40,64}", str(manifest.get("revision", ""))):
        result["reasons"].append("manifest must pin an immutable model/tokenizer revision")
    required = {"config.json", "tokenizer_config.json", "tokenizer.json"}
    required.update(path.name for path in model_path.glob("*.safetensors"))
    required.update(path.name for path in model_path.glob("pytorch_model*.bin"))
    required.update(path.name for path in model_path.glob("*.index.json"))
    if not any(name.endswith((".safetensors", ".bin")) for name in required):
        result["reasons"].append("no local model weights")
    files = manifest.get("files", {})
    for name in sorted(required):
        expected = files.get(name, {})
        path = model_path / name
        if not path.is_file() or not isinstance(expected, dict) or not expected.get("sha256"):
            result["reasons"].append("missing pinned artifact: " + name)
            continue
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
        if sha.hexdigest() != expected["sha256"]:
            result["reasons"].append("artifact hash differs: " + name)
    result.update(model_id=manifest.get("repo"), revision=manifest.get("revision"),
                  tokenizer_revision=manifest.get("tokenizer_revision", manifest.get("revision")))
    result["verified"] = not result["reasons"]
    return result


@dataclass
class PhaseSample:
    phase: str
    generation: Generation
    old_logprobs: list[float] = field(default_factory=list)

    @property
    def actor_mask(self):
        return [0] * len(self.generation.prompt_ids) + [1] * len(self.generation.completion_ids)


@dataclass
class EpisodeRollout:
    root_id: str
    latent_world_id: str
    rollout_seed: int
    phases: list[PhaseSample]
    r2: float
    r3: float
    questions: int = 0
    unsupported_operations: int = 0
    invalid_outputs: int = 0
    memory_path: str = ""
    split: str = "train"

    def reward(self, horizon="long"):
        if horizon not in {"long", "short"}:
            raise ValueError("horizon must be long or short")
        value = self.r2 + (self.r3 if horizon == "long" else 0.)
        return value - .10 * self.questions - .50 * self.unsupported_operations - .20 * self.invalid_outputs


def group_advantages(returns, epsilon=1e-6):
    if not returns or not all(math.isfinite(float(value)) for value in returns):
        raise ValueError("A finite nonempty reward group is required")
    mean = statistics.fmean(returns)
    std = statistics.pstdev(returns)
    if std == 0:
        return [0.] * len(returns)
    return [(value - mean) / (std + epsilon) for value in returns]


def group_telemetry(rollouts, horizon="long"):
    """Keep task variation distinct from penalty-only variation."""
    task = [r.r2 + (r.r3 if horizon == "long" else 0.) for r in rollouts]
    returns = [r.reward(horizon) for r in rollouts]
    actions, states = [], []
    for rollout in rollouts:
        phase = rollout.phases[0] if rollout.phases else None
        try:
            action = json.loads(phase.generation.text)
        except (ValueError, AttributeError):
            action = {"action": "INVALID", "slot": None}
        if not isinstance(action, dict):
            action = {"action": "INVALID", "slot": None}
        actions.append({"action": action.get("action", "INVALID"), "slot": action.get("slot")})
        states.append(hashlib.sha256(json.dumps(phase.generation.prompt_ids if phase else [], separators=(",", ":")).encode()).hexdigest())
    counts = {}
    for action in actions:
        key = json.dumps(action, sort_keys=True)
        counts[key] = counts.get(key, 0) + 1
    frequencies = [value / len(actions) for value in counts.values()]
    return {"root_id": rollouts[0].root_id, "latent_world_id": rollouts[0].latent_world_id,
            "observable_prompt_token_sha256": states,
            "actions": actions, "semantic_action_diversity": len(counts),
            "empirical_action_entropy_nats": -sum(p * math.log(p) for p in frequencies),
            "entropy_scope": "empirical sampled SELECT action frequencies; not token or full legal-action entropy",
            "task_returns": task, "returns": returns,
            "returns_by_component": [{"current": r.r2, "future": r.r3,
                "question_penalty": -.1 * r.questions, "unsupported_write_penalty": -.5 * r.unsupported_operations,
                "invalid_output_penalty": -.2 * r.invalid_outputs} for r in rollouts],
            "std_task": statistics.pstdev(task), "std_total": statistics.pstdev(returns),
            "std_convention": "population_ddof_0", "advantage_epsilon": 1e-6,
            "advantages": group_advantages(returns)}


def clipped_token_loss(new_logprobs, old_logprobs, reference_logprobs, advantage,
                       *, clip_epsilon=.20, kl_coefficient=.02):
    """Sampled-token GRPO objective, normalized by actor tokens per trajectory.

    The k3 KL estimator is nonnegative and zero at the frozen selected B1.
    All tensors refer to completion tokens at their original observed prefixes.
    """
    ratio = (new_logprobs - old_logprobs).exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1. - clip_epsilon, 1. + clip_epsilon) * advantage
    log_ratio_reference = reference_logprobs - new_logprobs
    kl = log_ratio_reference.exp() - log_ratio_reference - 1.
    objective = unclipped.minimum(clipped)
    return (-objective + kl_coefficient * kl).mean(), kl.mean()


def validate_group(rollouts, *, expected_group_size=4, smoke=False):
    if len(rollouts) != expected_group_size:
        raise ValueError("Every root/world must have exactly the declared rollout group size")
    if len({(r.root_id, r.latent_world_id) for r in rollouts}) != 1:
        raise ValueError("A group cannot combine different roots or latent worlds")
    if len({r.rollout_seed for r in rollouts}) != len(rollouts):
        raise ValueError("Rollout random generators must have independent seeds")
    select_prefixes = [tuple(r.phases[0].generation.prompt_ids) for r in rollouts if r.phases]
    if len(set(select_prefixes)) > 1:
        raise ValueError("A GRPO group must share the same pre-ASK observable SELECT prefix")
    paths = [str(Path(r.memory_path).resolve()) for r in rollouts]
    if any(not r.memory_path or r.memory_path == ":memory:" for r in rollouts) or len(set(paths)) != len(paths):
        raise ValueError("Each rollout requires its own persisted SQLite path")
    for rollout in rollouts:
        if rollout.split != ("dev_smoke" if smoke else "train"):
            raise ValueError("Unauthorized rollout partition")
        if [phase.phase.lower() for phase in rollout.phases] != ["select", "respond", "write"]:
            raise ValueError("SELECT, RESPOND and WRITE must retain separate actual prefixes")
        if not all(math.isfinite(float(x)) for x in (rollout.r2, rollout.r3)):
            raise ValueError("Nonfinite environment return")
        for phase in rollout.phases:
            if not phase.generation.prompt_ids or not phase.generation.completion_ids:
                raise ValueError("Actor token prefixes and outputs are required even for invalid JSON")
            if phase.old_logprobs and len(phase.old_logprobs) != len(phase.generation.completion_ids):
                raise ValueError("Misaligned old actor log probabilities")


def root_batch(episodes, update_index, count):
    """Distinct independent roots per effective batch; rotate their worlds.

    Both RL horizons call the same deterministic schedule. A root's worlds and
    paraphrases remain grouped rather than becoming extra independent roots.
    """
    groups = {}
    for episode in episodes:
        key = episode.get("root_id") if isinstance(episode, dict) else episode
        groups.setdefault(key, []).append(episode)
    keys = list(groups)
    if count > len(keys):
        raise ValueError("Effective batch requires distinct independent roots")
    result = []
    for offset in range(count):
        position = update_index * count + offset
        variants = groups[keys[position % len(keys)]]
        result.append(variants[(position // len(keys)) % len(variants)])
    return result


def load_b1_reference(checkpoint, *, device="cpu"):
    reference = SpecLocalPolicy.from_checkpoint(checkpoint, trainable=False, device=device)
    for parameter in reference.model.parameters():
        parameter.requires_grad_(False)
    return reference


def reference_logprobs_on_actor_device(reference, policy, generation):
    """Compute exact frozen-B1 token likelihoods, then offload before backward.

    Storage remains CPU between calls. This option needs a resource probe at the
    declared context/output limits before enabling it in a frozen campaign.
    ``finally`` restores CPU storage even when reference inference fails.
    """
    original_device = reference.device
    compute_device = policy.device
    try:
        if str(original_device) != str(compute_device):
            reference.model.to(compute_device)
        with reference.torch.no_grad(), reference.likelihood_mode():
            values = reference.sequence_logprobs(generation, sampling_distribution=True).detach()
        return values
    finally:
        if str(reference.device) != str(original_device):
            reference.model.to(original_device)


def verify_reference(policy, reference, *, smoke=False):
    if reference is policy or reference.model is policy.model:
        raise ValueError("KL reference must be a separate frozen selected B1 model")
    if any(parameter.requires_grad for parameter in reference.model.parameters()):
        raise ValueError("KL reference has trainable parameters")
    if not hasattr(reference.model, "peft_config"):
        raise ValueError("KL reference must include the selected SFT adapter")
    if not smoke:
        if not policy.origin_checkpoint or not reference.origin_checkpoint:
            raise ValueError("Both continuation and reference must originate at the selected B1")
        if Path(policy.origin_checkpoint).resolve() != Path(reference.origin_checkpoint).resolve():
            raise ValueError("KL reference checkpoint differs from B1 continuation parent")
        manifest = json.loads((Path(policy.origin_checkpoint) / "local_checkpoint.json").read_text())
        if artifact_hashes(policy.origin_checkpoint) != manifest["artifacts_sha256"]:
            raise ValueError("Selected B1 artifacts changed")
        if policy.config.seed != reference.config.seed:
            raise ValueError("Reference and actor must share the selected B1 seed")
    actor_names = {name for name, parameter in policy.model.named_parameters() if parameter.requires_grad}
    reference_names = dict(reference.model.named_parameters())
    if not actor_names <= reference_names.keys():
        raise ValueError("Reference does not contain identical adapter parameter names")
    if policy.config.temperature != reference.config.temperature:
        raise ValueError("KL reference and actor sampling temperatures differ")
    if parameter_hash(policy) != parameter_hash(reference):
        raise ValueError("Actor and frozen B1 reference effective weights differ before the first update")


def train_grpo(policy, reference, roots, collect_rollout, destination, validation,
               *, horizon="long", updates=120, seed=101, roots_per_batch=8,
               rollouts_per_root=4, learning_rate=5e-6, kl_coefficient=.02,
               clip_epsilon=.20, checkpoint_every=10, budget_seconds=None,
               stop_after_zero_signal_updates=10, smoke=False, protocol_root=None,
               progress_every=1, initial_validation_metrics=None, reference_execution="stored_device",
               closure_authorization=None, checkpoint_steps=None, checkpoint_time_fractions=None,
               closure_selection=None, checkpoint_save_budget_seconds=None):
    """``collect_rollout(policy, root, seed)`` must execute one isolated episode.

    The collector is called before any update in its effective batch. Old log
    probabilities are evaluated on those exact phase token arrays and retained
    alongside masks, returns and advantages in ``rollouts.jsonl``.
    """
    _gate(smoke, continuation=True, root=protocol_root, closure_authorization=closure_authorization)
    if horizon not in {"long", "short"}:
        raise ValueError("Unknown training horizon")
    if reference_execution not in {"stored_device", "cpu_storage_gpu_compute"}:
        raise ValueError("Unknown frozen-reference execution strategy")
    if reference_execution == "cpu_storage_gpu_compute" and str(reference.device) != "cpu":
        raise ValueError("Temporarily moving reference computation requires CPU storage")
    if updates < 1 or roots_per_batch < 1 or rollouts_per_root != 4 or checkpoint_every < 1 or progress_every < 1:
        raise ValueError("Invalid episodic group/batch configuration")
    if learning_rate <= 0 or kl_coefficient < 0 or not 0 < clip_epsilon < 1:
        raise ValueError("Invalid GRPO optimizer configuration")
    if checkpoint_steps is not None and checkpoint_time_fractions is not None:
        raise ValueError("Use step or time checkpoint selection, not both")
    clock = TrainingBudget(budget_seconds, checkpoint_time_fractions, separate=closure_authorization is not None)
    save_budget = CheckpointSaveBudget(checkpoint_save_budget_seconds)
    if smoke and (updates != 1 or roots_per_batch != 1):
        raise ValueError("Technical GRPO smoke requires one group and one update")
    if not smoke and (budget_seconds is None or budget_seconds <= 0):
        raise ValueError("Freeze a post-SFT accelerator budget before continuation")
    roots = list(roots)
    if not roots:
        raise ValueError("A fixed sequence of training roots is required")
    destination, parameters = _prepare(policy, destination, seed)
    plan = {"seed": seed, "horizon": horizon, "max_updates": updates,
            "roots_per_batch": roots_per_batch, "rollouts_per_root": rollouts_per_root,
            "checkpoint_every": checkpoint_every, "budget_seconds": budget_seconds,
            "initial_validation_reused": initial_validation_metrics is not None,
            "reference_device": str(reference.device), "learning_rate": learning_rate,
            "reference_execution": reference_execution,
            "kl_coefficient": kl_coefficient, "clip_epsilon": clip_epsilon,
            "checkpoint_steps": checkpoint_steps, "checkpoint_time_fractions": checkpoint_time_fractions,
            "closure_selection": closure_selection,
            "checkpoint_save_budget_seconds": checkpoint_save_budget_seconds,
            "optimizer": {"name": "AdamW", "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.},
            "scheduler": "constant", "max_grad_norm": 1.,
            "gradient_accumulation_trajectories": roots_per_batch * rollouts_per_root,
            "rollout_optimization_passes": 1, "std_convention": "population_ddof_0",
            "advantage_epsilon": 1e-6, "telemetry_version": "closure_v4"}
    with RunJournal(destination, "episodic_clipped_grpo", plan) as journal:
        verify_reference(policy, reference, smoke=smoke)
        frozen_before = parameter_hash(policy, trainable=False)
        reference_before = parameter_hash(reference)
        adapter_before = parameter_hash(policy, trainable=True)
        optimizer = policy.torch.optim.AdamW([p for _, p in parameters], lr=learning_rate, weight_decay=0.)
        started = time.monotonic()
        initial_evaluation_started = time.monotonic()
        journal.event("validation_started", step=0, reused=initial_validation_metrics is not None)
        metrics, best_key = _initial_metrics(policy, validation, initial_validation_metrics)
        best_metrics = metrics
        journal.event("validation_completed", step=0, metrics=metrics)
        best_step = 0
        history = [{"step": 0, "validation": metrics,
                    "selection_guard_passed": selection_guards_passed(metrics, closure_selection)}]
        counters = {"updates": 0, "episodes": 0, "questions": 0, "input_tokens": 0,
                    "generated_tokens": 0, "trained_tokens": 0, "validation_evaluations": int(initial_validation_metrics is None),
                    "validation_reused": int(initial_validation_metrics is not None),
                    "allocated_seconds": budget_seconds}
        def save_state(checkpoint, step):
            def save():
                policy.save_checkpoint(checkpoint)
                _save_training_state(policy, optimizer, checkpoint, step, counters)
            save_budget.save(save)
        save_state(destination / "step_000000", 0)
        clock.pause(time.monotonic() - initial_evaluation_started)
        clock.start()
        zero_signal_streak = 0
        stop_reason = "max_updates"
        with (destination / "rollouts.jsonl").open("w", encoding="utf-8") as trace, \
             (destination / "collected_rollouts.jsonl").open("w", encoding="utf-8") as collected, \
             (destination / "group_diagnostics.jsonl").open("w", encoding="utf-8") as group_log:
            for update_index in range(updates):
                save_budget.ensure_available()
                if clock.exhausted(time.monotonic() - started):
                    stop_reason = "time_budget"
                    break
                groups = []
                for root_index, root in enumerate(root_batch(roots, update_index, roots_per_batch)):
                    rollouts = []
                    for sample_index in range(rollouts_per_root):
                        rollout_seed = seed * 10000000 + update_index * roots_per_batch * rollouts_per_root + root_index * rollouts_per_root + sample_index
                        policy.seed(rollout_seed)
                        rollout = collect_rollout(policy, root, rollout_seed)
                        for phase in rollout.phases:
                            with policy.torch.no_grad(), policy.likelihood_mode():
                                actual_old = policy.sequence_logprobs(phase.generation, sampling_distribution=True)
                            if phase.old_logprobs:
                                supplied = policy.torch.tensor(phase.old_logprobs, device=actual_old.device)
                                if not bool(policy.torch.allclose(actual_old, supplied, atol=1e-5, rtol=1e-5)):
                                    raise ValueError("Old log probabilities do not match the generated policy")
                            phase.old_logprobs = actual_old.detach().cpu().tolist()
                        rollouts.append(rollout)
                        collected.write(json.dumps({"update": update_index + 1,
                            "group_id": f"{update_index}:{root_index}", "return": rollout.reward(horizon),
                            "rollout": asdict(rollout)}, ensure_ascii=False) + "\n")
                        collected.flush()
                        journal.event("collecting", step=update_index, pending_update=update_index + 1,
                                      group_index=root_index, rollout_index=sample_index,
                                      rollout_return=rollout.reward(horizon),
                                      invalid_outputs=rollout.invalid_outputs)
                    validate_group(rollouts, expected_group_size=rollouts_per_root, smoke=smoke)
                    advantages = group_advantages([r.reward(horizon) for r in rollouts])
                    groups.append((rollouts, advantages))
                    group_log.write(json.dumps({"optimizer_update": update_index + 1,
                        "group_id": f"{update_index}:{root_index}", **group_telemetry(rollouts, horizon)}, ensure_ascii=False) + "\n")
                    group_log.flush()
                signal_groups = sum(any(a != 0 for a in advantages) for _, advantages in groups)
                task_signal_groups = sum(statistics.pstdev([r.r2 + (r.r3 if horizon == "long" else 0.)
                                                          for r in rollouts]) > 0 for rollouts, _ in groups)
                zero_signal_streak = 0 if signal_groups else zero_signal_streak + 1
                optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.
                batch_kl = 0.
                ratios, clip_fractions, sampled_surprisals = [], [], []
                gradient_probe = {}
                for group_index, (rollouts, advantages) in enumerate(groups):
                    for rollout, advantage in zip(rollouts, advantages):
                        total_actor_tokens = sum(len(phase.generation.completion_ids) for phase in rollout.phases)
                        for phase in rollout.phases:
                            generated = phase.generation
                            if reference_execution == "cpu_storage_gpu_compute":
                                ref_logps = reference_logprobs_on_actor_device(reference, policy, generated)
                            else:
                                with reference.torch.no_grad(), reference.likelihood_mode():
                                    ref_logps = reference.sequence_logprobs(generated, sampling_distribution=True)
                            with policy.likelihood_mode():
                                new_logps = policy.sequence_logprobs(generated, sampling_distribution=True)
                                old_logps = policy.torch.tensor(phase.old_logprobs, device=new_logps.device)
                                ref_logps = ref_logps.to(new_logps.device)
                                loss, kl = clipped_token_loss(new_logps, old_logps, ref_logps, advantage,
                                                             clip_epsilon=clip_epsilon, kl_coefficient=kl_coefficient)
                                weight = len(generated.completion_ids) / total_actor_tokens / (roots_per_batch * rollouts_per_root)
                                weighted = loss * weight
                                if not bool(policy.torch.isfinite(weighted)):
                                    raise RuntimeError("NONFINITE_GRPO_LOSS: technical stop")
                                ratio = (new_logps.detach() - old_logps).exp()
                                ratios.extend(ratio.cpu().tolist())
                                clip_fractions.append(float(((ratio < 1. - clip_epsilon) | (ratio > 1. + clip_epsilon)).float().mean().cpu()))
                                sampled_surprisals.extend((-new_logps.detach()).cpu().tolist())
                                if not gradient_probe:
                                    trainable = [p for _, p in parameters]
                                    task_grads = policy.torch.autograd.grad((loss - kl_coefficient * kl) * weight,
                                        trainable, retain_graph=True, allow_unused=True)
                                    kl_grads = policy.torch.autograd.grad(kl_coefficient * kl * weight,
                                        trainable, retain_graph=True, allow_unused=True)
                                    def norm(grads):
                                        return math.sqrt(sum(float(g.detach().float().square().sum().cpu()) for g in grads if g is not None))
                                    gradient_probe = {"task_gradient_norm_sample": norm(task_grads),
                                        "regularization_gradient_norm_sample": norm(kl_grads),
                                        "gradient_probe_scope": "first trajectory SELECT phase per optimizer update"}
                                weighted.backward()
                            batch_loss += float(weighted.detach().cpu())
                            batch_kl += float(kl.detach().cpu()) * weight
                            counters["input_tokens"] += len(generated.prompt_ids)
                            counters["generated_tokens"] += len(generated.completion_ids)
                            counters["trained_tokens"] += len(generated.completion_ids)
                        counters["episodes"] += 1
                        counters["questions"] += rollout.questions
                        journal.event("optimizing", step=update_index, pending_update=update_index + 1,
                                      group_index=group_index, rollout_seed=rollout.rollout_seed)
                        trace.write(json.dumps({"update": update_index + 1, "group_id": f"{update_index}:{group_index}",
                                                "return": rollout.reward(horizon), "advantage": advantage,
                                                "rollout": asdict(rollout),
                                                "actor_masks": [phase.actor_mask for phase in rollout.phases]}, ensure_ascii=False) + "\n")
                    trace.flush()
                step_telemetry = {}
                grad_norm = _optimizer_step(policy, optimizer, parameters, step_telemetry)
                step = update_index + 1
                counters["updates"] = step
                entry = {"step": step, "loss": batch_loss, "kl": batch_kl, "grad_norm": grad_norm,
                         "policy_loss": batch_loss - kl_coefficient * batch_kl,
                         "kl_loss": kl_coefficient * batch_kl,
                         "ratio_min": min(ratios), "ratio_max": max(ratios),
                         "ratio_mean": statistics.fmean(ratios),
                         "clip_fraction_phase_mean": statistics.fmean(clip_fractions),
                         "sampled_token_surprisal_mean": statistics.fmean(sampled_surprisals),
                         "zero_task_signal_fraction": 1. - task_signal_groups / roots_per_batch,
                         "zero_signal_fraction": 1. - signal_groups / roots_per_batch,
                         "zero_signal_streak": zero_signal_streak, **step_telemetry, **gradient_probe}
                if str(policy.device).startswith("cuda"):
                    entry.update(cuda_peak_allocated_bytes=policy.torch.cuda.max_memory_allocated(policy.device),
                                 cuda_peak_reserved_bytes=policy.torch.cuda.max_memory_reserved(policy.device))
                if step % progress_every == 0 or step == updates:
                    journal.progress(step, updates, loss=batch_loss, zero_signal_streak=zero_signal_streak,
                                     resource_counters=dict(counters))
                checkpoint_due = clock.checkpoint_due(time.monotonic() - started)
                eligible_step = step in checkpoint_steps if checkpoint_steps is not None else (step % checkpoint_every == 0 or step == updates)
                if checkpoint_due or (checkpoint_time_fractions is None and eligible_step):
                    pause_started = time.monotonic()
                    journal.event("validation_started", step=step)
                    metrics, key = _validate(policy, validation, step)
                    counters["validation_evaluations"] += 1
                    entry["validation"] = metrics
                    entry["selection_guard_passed"] = selection_guards_passed(metrics, closure_selection)
                    checkpoint = destination / f"step_{step:06d}"
                    save_state(checkpoint, step)
                    journal.event("validation_completed", step=step, metrics=metrics)
                    clock.pause(time.monotonic() - pause_started)
                    entry["checkpoint_eligible"] = True
                    entry["checkpoint_training_seconds"] = clock.elapsed(time.monotonic() - started)
                    if selection_is_better(metrics, step, best_metrics, best_step, closure_selection):
                        best_step, best_key, best_metrics = step, key, metrics
                history.append(entry)
                if zero_signal_streak >= stop_after_zero_signal_updates:
                    stop_reason = "zero_total_return_signal"
                    break
        clock.finish()
        final_evaluation_started = time.monotonic()
        final_step = counters["updates"]
        if final_step and "validation" not in history[-1]:
            journal.event("validation_started", step=final_step, final=True)
            metrics, key = _validate(policy, validation, final_step)
            counters["validation_evaluations"] += 1
            history[-1]["validation"] = metrics
            checkpoint = destination / f"step_{final_step:06d}"
            save_state(checkpoint, final_step)
            journal.event("validation_completed", step=final_step, metrics=metrics)
            if checkpoint_time_fractions is None and checkpoint_steps is None and selection_is_better(metrics, final_step, best_metrics, best_step, closure_selection):
                best_step, best_key, best_metrics = final_step, key, metrics
        save_state(destination / "last", final_step)
        clock.pause(time.monotonic() - final_evaluation_started)
        frozen_after = parameter_hash(policy, trainable=False)
        reference_after = parameter_hash(reference)
        if frozen_after != frozen_before or reference_after != reference_before:
            raise RuntimeError("FROZEN_PARAMETERS_CHANGED: technical stop")
        result = {"algorithm": "episodic_clipped_grpo", "horizon": horizon, "seed": seed,
                  "status": "executed", "smoke": smoke, "best_step": best_step, "plan": plan,
                  "selected_checkpoint": str(destination / f"step_{best_step:06d}"),
                  "final_checkpoint": str(destination / "last"), "stop_reason": stop_reason,
                  "parent_checkpoint": policy.origin_checkpoint, "kl_reference": reference.origin_checkpoint,
                  "trainable_before": adapter_before, "trainable_after": parameter_hash(policy, trainable=True),
                  "frozen_before": frozen_before, "frozen_after": frozen_after,
                  "reference_before": reference_before, "reference_after": reference_after,
                  "history": history, "resource_counters": {**counters, "elapsed_seconds": time.monotonic() - started,
                  **save_budget.counters(),
                  "training_seconds": clock.elapsed(time.monotonic() - started),
                  "validation_and_checkpoint_seconds": clock.paused_seconds,
                  "budget_overrun_seconds": max(0., clock.elapsed(time.monotonic() - started) - budget_seconds) if budget_seconds else 0.}}
        _write_json(destination / "training.json", result)
        journal.event("completed", step=final_step, selected_step=best_step, stop_reason=stop_reason,
                      resource_counters=result["resource_counters"])
        return result


def technical_smoke(model_path, output_dir, device="cpu", *, allow_nonprotocol_model=False):
    """One real LoRA SFT step and one four-rollout GRPO gradient probe.

    Synthetic, assigned rewards deliberately provide a known gradient signal.
    They are recorded as a technical probe, never as task performance. A small
    architecture fixture can exercise this path with allow_nonprotocol_model;
    such a result explicitly cannot close G0 for a different backbone.
    """
    from research.src.policy.llm_controller import LLMController
    from research.src.training.sft import train_sft
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "scope": "technical_gradient_probe_only",
              "model_path": str(Path(model_path).resolve()), "device": device,
              "protocol_candidate": False, "load": "not_executed", "generation": "not_executed",
              "sft": "not_executed", "grpo": "not_executed"}
    started = time.monotonic()
    try:
        provenance = inspect_candidate(model_path)
        report["candidate_provenance"] = provenance
        candidate = provenance["verified"]
        report["protocol_candidate"] = candidate
        if not candidate and not allow_nonprotocol_model:
            raise ValueError("TECHNICAL_LOCK_REQUIRED: local model is not an allowed protocol candidate")
        policy = SpecLocalPolicy(model_path, device=device, precision="float32" if device == "cpu" else "auto",
                                 max_new_tokens=8, seed=101)
        report["loaded_parameter_count"] = sum(parameter.numel() for parameter in policy.model.parameters())
        if candidate:
            lower, upper = {"Qwen/Qwen2.5-0.5B-Instruct": (.45e9, .55e9),
                            "Qwen/Qwen2.5-1.5B-Instruct": (1.3e9, 1.8e9),
                            "Qwen/Qwen2.5-3B-Instruct": (2.8e9, 3.4e9)}[provenance["model_id"]]
            if not lower <= report["loaded_parameter_count"] <= upper:
                raise ValueError("Loaded parameter count differs from the verified candidate architecture")
        report["load"] = "executed"
        report["precision"] = policy.config.precision
        report["gradient_checkpointing"] = policy.config.gradient_checkpointing
        if device.startswith("cuda"):
            policy.torch.cuda.reset_peak_memory_stats(policy.device)
        generated = policy.generate("Devuelve una preferencia alimentaria breve.", sample=False)
        report["generation"] = {"status": "executed", "input_tokens": len(generated.prompt_ids),
                                "output_tokens": len(generated.completion_ids), "raw_output": generated.text}
        observation = {"memory": [], "user_messages": [{"turn_id": "u2", "text": "Solo hoy prefiero pasta."}],
                       "query": {"attribute": "dinner_preference", "date": "2026-01-12", "session_id": "s2", "context": None},
                       "reuse_regime": "later_session_without_questions"}
        examples = [{"phase": "select", "observation": observation,
                     "completion": {"action": "PROCEED", "slot": None}, "split": "dev_smoke"}]
        sft_result = train_sft(policy, examples, output_dir / "sft", lambda _: {"utility": 0.},
                               updates=1, seed=101, smoke=True)
        report["sft"] = sft_result
        # The final updated SFT checkpoint defines both continuation and KL reference.
        checkpoint = output_dir / "sft" / "last"
        del policy
        actor = SpecLocalPolicy.from_checkpoint(checkpoint, trainable=True, device=device)
        reference = load_b1_reference(checkpoint, device="cpu")
        controller = LLMController(actor)
        assigned_returns = [1., 0., -1., .5]
        calls = [0]

        def collector(model, root, rollout_seed):
            index = calls[0]
            calls[0] += 1
            phases = []
            for phase in ("select", "respond", "write"):
                # Exact phase prefix, sampled once; no future tokens are appended.
                prompt = controller.prompt(phase, observation)
                generation = model.generate(prompt, sample=True)
                phases.append(PhaseSample(phase, generation))
            return EpisodeRollout("gradient_probe", "fixed_world", rollout_seed, phases,
                                  assigned_returns[index], 0., memory_path=str(output_dir / f"probe_{index}.sqlite"),
                                  split="dev_smoke")

        grpo_result = train_grpo(actor, reference, ["probe"], collector, output_dir / "grpo",
                                 lambda _: {"utility": 0.}, updates=1, roots_per_batch=1,
                                 seed=101, smoke=True)
        report["grpo"] = grpo_result
        report["reference_device"] = str(reference.device)
        if device.startswith("cuda"):
            from .resources import probe_context_capacity
            report["context_capacity"] = probe_context_capacity(actor, input_tokens=2048, output_tokens=512)
            report["cuda_peak_allocated_bytes"] = actor.torch.cuda.max_memory_allocated(actor.device)
            report["cuda_peak_reserved_bytes"] = actor.torch.cuda.max_memory_reserved(actor.device)
        changed_sft = sft_result["trainable_before"] != sft_result["trainable_after"]
        changed_rl = grpo_result["trainable_before"] != grpo_result["trainable_after"]
        if not changed_sft or not changed_rl:
            raise RuntimeError("Technical smoke requires real nonzero LoRA updates in both algorithms")
        report["status"] = "passed" if candidate else "passed_nonprotocol_fixture"
        report["g0_eligible"] = candidate
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["g0_eligible"] = False
    report["elapsed_seconds"] = time.monotonic() - started
    _write_json(output_dir / "technical_smoke.json", report)
    return report
