"""Completion-only LoRA SFT, with validation-only selection including step zero."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from research.src.policy.llm_controller import LLMController, PHASE_LIMITS, MAX_INPUT_TOKENS, validate_action


def completion_label_mask(prompt_ids, completion_ids):
    return [-100] * len(prompt_ids) + list(completion_ids)


class TrainingBudget:
    """Optional v4 clock: rollout/backward time excludes validation and saves.

    Historical callers retain the original elapsed-time budget. With fractions,
    complete an atomic update crossing a threshold and never start another update
    after exhausting the allocation. Several thresholds crossed by one operation
    refer to the same checkpoint, not additional selection opportunities.
    """

    def __init__(self, seconds, fractions=None, *, separate=False):
        self.seconds, self.fractions = seconds, fractions
        self.separate = separate or fractions is not None
        if fractions is not None:
            if not seconds or seconds <= 0 or list(fractions) != sorted(set(fractions)) or not fractions or fractions[-1] != 1.:
                raise ValueError("Time checkpoints require a positive budget and increasing fractions ending at 1")
            if any(not 0 < fraction <= 1 for fraction in fractions):
                raise ValueError("Checkpoint fractions must lie in (0,1]")
        self.pending = list(fractions or [])
        self.paused_seconds = 0.
        self.paused_after_start = 0.
        self.started = None
        self.finished_elapsed = None

    def start(self):
        self.started = time.monotonic()

    def elapsed(self, legacy_elapsed):
        if not self.separate:
            return legacy_elapsed
        if self.finished_elapsed is not None:
            return self.finished_elapsed
        return max(0., time.monotonic() - self.started - self.paused_after_start)

    def exhausted(self, legacy_elapsed):
        return self.seconds is not None and self.elapsed(legacy_elapsed) >= self.seconds

    def checkpoint_due(self, legacy_elapsed):
        due = False
        elapsed = self.elapsed(legacy_elapsed)
        while self.pending and elapsed >= self.pending[0] * self.seconds:
            self.pending.pop(0)
            due = True
        return due

    def pause(self, seconds):
        self.paused_seconds += seconds
        if self.started is not None:
            self.paused_after_start += seconds

    def finish(self):
        if self.separate:
            self.finished_elapsed = self.elapsed(0.)


class CheckpointSaveBudget:
    """Measure complete adapter+optimizer saves as one indivisible operation."""
    def __init__(self, seconds=None):
        if seconds is not None and seconds <= 0:
            raise ValueError("Checkpoint save budget must be positive")
        self.seconds = seconds
        self.elapsed_seconds = 0.
        self.operations = 0

    def ensure_available(self):
        if self.seconds is not None and self.elapsed_seconds >= self.seconds:
            raise RuntimeError("CHECKPOINT_SAVE_BUDGET_EXHAUSTED: preserve partial run; no new save may start")

    def save(self, callback):
        self.ensure_available()
        started = time.monotonic()
        try:
            return callback()
        finally:
            self.elapsed_seconds += time.monotonic() - started
            self.operations += 1

    def counters(self):
        return {"checkpoint_save_seconds": self.elapsed_seconds, "checkpoint_save_operations": self.operations,
                "checkpoint_save_allocated_seconds": self.seconds,
                "checkpoint_save_overrun_seconds": max(0., self.elapsed_seconds - self.seconds) if self.seconds else 0.}


def parameter_hash(policy, *, trainable=None):
    digest = hashlib.sha256()
    for name, parameter in policy.model.named_parameters():
        if trainable is not None and parameter.requires_grad != trainable:
            continue
        digest.update(name.encode())
        tensor = parameter.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(policy.torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def selection_key(metrics, step):
    if isinstance(metrics, (int, float)):
        metrics = {"utility": metrics}
    utility = float(metrics["utility"])
    if not math.isfinite(utility):
        raise ValueError("Nonfinite validation utility")
    return (utility, -float(metrics.get("unsupported_writes", 0)),
            -float(metrics.get("questions", 0)), -step)


def selection_guards_passed(metrics, closure_selection):
    if closure_selection is None:
        return True
    guards = closure_selection.get("validity_guards", {})
    if not guards:
        raise ValueError("Closure checkpoint selection requires predeclared validity guards")
    for name, maximum in guards.items():
        if name not in metrics or not math.isfinite(float(metrics[name])):
            return False
        if float(metrics[name]) > float(maximum):
            return False
    return True


def selection_is_better(metrics, step, best_metrics, best_step, closure_selection=None):
    """V4 keeps earliest numerical ties; historical callers retain their rule."""
    if closure_selection is None:
        return selection_key(metrics, step) > selection_key(best_metrics, best_step)
    tolerance = float(closure_selection.get("utility_tolerance", 1e-6))
    if tolerance < 0:
        raise ValueError("Selection tolerance must be nonnegative")
    if not selection_guards_passed(metrics, closure_selection):
        return False
    if not selection_guards_passed(best_metrics, closure_selection):
        return True
    difference = float(metrics["utility"]) - float(best_metrics["utility"])
    return difference > tolerance or (abs(difference) <= tolerance and step < best_step)


def _gate(smoke, continuation=False, root=None, closure_authorization=None):
    if closure_authorization is not None:
        if smoke:
            raise ValueError("A closure campaign cannot bypass authorization with smoke=True")
        from research.src.closure_protocol import require_training_authorization
        require_training_authorization(Path(closure_authorization))
        return
    if not smoke:
        from research.src.protocol import require_gate
        require_gate("G3" if continuation else "G2", root=root)


def _prepare(policy, destination, seed):
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Existing run preserved: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    policy.seed(seed)
    policy.config.seed = seed
    policy.enable_training()
    trainable = [(n, p) for n, p in policy.model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise ValueError("Only integrated LoRA parameters may be trained")
    return destination, trainable


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class RunJournal:
    """Durable progress and failure state; partial runs are never auto-resumed."""

    def __init__(self, destination, algorithm, plan):
        self.destination = Path(destination)
        self.algorithm = algorithm
        self.plan = plan
        self.started = time.monotonic()
        self.last = {}
        self.last_completed_step = 0
        self.stream = (self.destination / "progress.jsonl").open("a", encoding="utf-8")
        self.event("started", plan=plan)

    def event(self, stage, **details):
        elapsed = time.monotonic() - self.started
        row = {"algorithm": self.algorithm, "stage": stage, "elapsed_seconds": elapsed, **details}
        self.last = row
        if stage in {"training", "validation_completed", "completed"}:
            self.last_completed_step = max(self.last_completed_step, details.get("step", 0))
        self.stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        status = "executed" if stage == "completed" else "failed" if stage == "failed" else "running"
        _write_json(self.destination / "run_state.json", {"status": status, "plan": self.plan, **row,
                    "partial_recovery": "explicit_optimizer_and_rng_recovery_required"})
        print(json.dumps({"training_progress": row}, ensure_ascii=False), flush=True)

    def progress(self, step, total_updates, **details):
        elapsed = time.monotonic() - self.started
        eta = elapsed / step * (total_updates - step) if step and total_updates < 2**31 - 1 else None
        self.event("training", step=step, total_updates=total_updates,
                   estimated_remaining_seconds=eta, **details)

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            self.event("failed", error_type=kind.__name__, error=str(error),
                       last_completed_step=self.last_completed_step,
                       recovery="Preserve this directory; explicitly recover optimizer/RNG or use a new run destination")
        self.stream.close()


def _save_training_state(policy, optimizer, destination, step, counters, sampler_state=None):
    """A recovery artifact; loading it is an explicit operation, never implicit."""
    state = {"step": step, "optimizer": optimizer.state_dict(), "counters": dict(counters),
             "python_rng": random.getstate(), "torch_rng": policy.torch.get_rng_state(),
             "cuda_rng": policy.torch.cuda.get_rng_state_all() if str(policy.device).startswith("cuda") else [],
             "parent_checkpoint": policy.origin_checkpoint, "sampler_state": sampler_state}
    destination = Path(destination)
    # Keep recovery state beside the model directory: adding it inside would
    # invalidate the immutable adapter artifact manifest written by the backend.
    target = destination.parent / (destination.name + ".optimizer_state.pt")
    policy.torch.save(state, target.with_name(target.name + ".tmp"))
    target.with_name(target.name + ".tmp").replace(target)


def _initial_metrics(policy, callback, cached=None):
    if cached is None:
        return _validate(policy, callback, 0)
    if not isinstance(cached, dict):
        raise ValueError("Cached initial validation metrics must be a selected B1 result")
    return dict(cached), selection_key(cached, 0)


def _validate(policy, callback, step):
    policy.model.eval()
    with policy.torch.no_grad():
        metrics = callback(policy)
    key = selection_key(metrics, step)
    if isinstance(metrics, (int, float)):
        metrics = {"utility": float(metrics)}
    return dict(metrics), key


def _optimizer_step(policy, optimizer, parameters, telemetry=None):
    grads = [p.grad for _, p in parameters if p.grad is not None]
    if not grads or not all(bool(policy.torch.isfinite(g).all()) for g in grads):
        raise RuntimeError("NONFINITE_OR_ABSENT_GRADIENT: technical stop")
    norm = policy.torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1.0)
    if not bool(policy.torch.isfinite(norm)):
        raise RuntimeError("NONFINITE_GRADIENT_NORM: technical stop")
    if telemetry is not None:
        post = policy.torch.linalg.vector_norm(policy.torch.stack([
            policy.torch.linalg.vector_norm(g.detach().float()) for g in grads]))
        telemetry.update(gradient_norm_pre_clip=float(norm.detach().cpu()),
                         gradient_norm_post_clip=float(post.detach().cpu()),
                         parameters_with_gradient=[name for name, p in parameters if p.grad is not None],
                         learning_rates=[float(group["lr"]) for group in optimizer.param_groups],
                         optimizer_step_executed=True, mixed_precision_skipped_steps=0)
    optimizer.step()
    return float(norm.detach().cpu())


def train_sft(policy, examples, destination, validation, *, updates=None, epochs=2,
              seed=101, smoke=False, continuation=False, budget_seconds=None,
              learning_rate=2e-4, checkpoint_every=10, prompt_root=None, protocol_root=None,
              progress_every=10, initial_validation_metrics=None,
              closure_authorization=None, checkpoint_time_fractions=None, closure_selection=None,
              checkpoint_save_budget_seconds=None):
    """Completion-only training; durable prefixes, validation and optimizer snapshots.

    Full SFT consumes every example once per epoch unless an explicit update cap
    is supplied. Campaigns select checkpoints at epoch boundaries; B3 repeats the
    same corpus until its frozen wall-time budget. Partial runs fail closed.
    """
    _gate(smoke, continuation, root=protocol_root, closure_authorization=closure_authorization)
    examples = list(examples)
    if not examples:
        raise ValueError("SFT corpus is empty")
    allowed = {"dev_smoke"} if smoke else {"train"}
    if any(example.get("split") not in allowed for example in examples):
        raise ValueError("SFT corpus split not authorized")
    if smoke and updates not in {None, 1}:
        raise ValueError("Technical smoke is limited to one update")
    if not smoke and continuation and (budget_seconds is None or budget_seconds <= 0):
        raise ValueError("B3 requires the frozen post-SFT time budget")
    if continuation and not getattr(policy, "origin_checkpoint", None):
        raise ValueError("B3 must load its selected B1 checkpoint before resetting the optimizer")
    if initial_validation_metrics is not None and not continuation:
        raise ValueError("Only a continuation may reuse selected B1 validation")
    if epochs < 1 or checkpoint_every < 1 or learning_rate <= 0 or progress_every < 1:
        raise ValueError("Invalid SFT configuration")
    clock = TrainingBudget(budget_seconds, checkpoint_time_fractions, separate=closure_authorization is not None)
    save_budget = CheckpointSaveBudget(checkpoint_save_budget_seconds)
    destination, parameters = _prepare(policy, destination, seed)
    max_updates = 1 if smoke else (updates if updates is not None else
                  (2**31 - 1 if continuation else len(examples) * epochs))
    if max_updates < 1:
        raise ValueError("At least one requested SFT update is required")
    plan = {"seed": seed, "epochs": epochs, "examples_per_epoch": len(examples),
            "max_updates": max_updates, "checkpoint_every": checkpoint_every,
            "progress_every": progress_every, "budget_seconds": budget_seconds,
            "initial_validation_reused": initial_validation_metrics is not None,
            "continuation": continuation, "learning_rate": learning_rate,
            "checkpoint_time_fractions": checkpoint_time_fractions,
            "closure_selection": closure_selection,
            "checkpoint_save_budget_seconds": checkpoint_save_budget_seconds,
            "optimizer": {"name": "AdamW", "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.},
            "scheduler": "constant", "max_grad_norm": 1., "gradient_accumulation_trajectories": 1}
    with RunJournal(destination, "completion_only_lora_sft", plan) as journal:
        before = parameter_hash(policy, trainable=True)
        frozen_before = parameter_hash(policy, trainable=False)
        optimizer = policy.torch.optim.AdamW([p for _, p in parameters], lr=learning_rate, weight_decay=0.)
        controller = LLMController(policy, prompt_root)
        started = time.monotonic()
        initial_evaluation_started = time.monotonic()
        journal.event("validation_started", step=0, reused=initial_validation_metrics is not None)
        metrics, best_key = _initial_metrics(policy, validation, initial_validation_metrics)
        best_metrics = metrics
        journal.event("validation_completed", step=0, metrics=metrics)
        best_step, history = 0, [{"step": 0, "validation": metrics,
                                "validation_reused": initial_validation_metrics is not None,
                                "selection_guard_passed": selection_guards_passed(metrics, closure_selection)}]
        examples_rng = random.Random(seed)
        schedule = []
        for _ in range(epochs):
            shuffled = list(range(len(examples)))
            examples_rng.shuffle(shuffled)
            schedule.extend(shuffled)
        counters = {"input_tokens": 0, "trained_tokens": 0, "updates": 0,
                    "validation_evaluations": int(initial_validation_metrics is None),
                    "validation_reused": int(initial_validation_metrics is not None),
                    "allocated_seconds": budget_seconds}
        def save_state(checkpoint, step):
            def save():
                policy.save_checkpoint(checkpoint)
                _save_training_state(policy, optimizer, checkpoint, step, counters,
                    sampler_state={"schedule": list(schedule), "examples_rng": examples_rng.getstate(),
                                   "next_update_index": step})
            save_budget.save(save)
        save_state(destination / "step_000000", 0)
        clock.pause(time.monotonic() - initial_evaluation_started)
        clock.start()
        stop_reason = "completed_epochs" if not continuation and updates is None else "max_updates"
        with (destination / "prefixes.jsonl").open("w", encoding="utf-8") as prefixes, \
             (destination / "updates.jsonl").open("w", encoding="utf-8") as update_log:
            for index in range(max_updates):
                save_budget.ensure_available()
                if clock.exhausted(time.monotonic() - started):
                    stop_reason = "time_budget"
                    break
                # Shuffle before picking the first item of a repeated schedule;
                # every complete cycle still contains every example exactly.
                if index and index % len(schedule) == 0:
                    examples_rng.shuffle(schedule)
                example_index = schedule[index % len(schedule)]
                example = examples[example_index]
                phase = example["phase"].lower()
                completion = example.get("completion", example.get("action"))
                if completion is None:
                    raise ValueError("Supervised example lacks its observable teacher action")
                if not isinstance(completion, str):
                    completion = json.dumps(completion, ensure_ascii=False, separators=(",", ":"))
                validate_action(phase, json.loads(completion))
                prompt = controller.prompt(phase, example["observation"])
                generation = policy.supervised_generation(prompt, completion)
                if len(generation.prompt_ids) > MAX_INPUT_TOKENS:
                    raise ValueError("INPUT_TOKEN_LIMIT: no silent SFT truncation")
                if len(generation.completion_ids) > PHASE_LIMITS[phase]:
                    raise ValueError("OUTPUT_TOKEN_LIMIT: teacher exceeds the declared phase budget")
                step = index + 1
                prefixes.write(json.dumps({"step": step, "example_index": example_index,
                    "phase": phase, "generation": asdict(generation),
                    "actor_mask": [0] * len(generation.prompt_ids) + [1] * len(generation.completion_ids)},
                    ensure_ascii=False) + "\n")
                prefixes.flush()
                optimizer.zero_grad(set_to_none=True)
                with policy.likelihood_mode():
                    logprobs = policy.sequence_logprobs(generation)
                    loss = -logprobs.mean()
                    if not bool(policy.torch.isfinite(loss)):
                        raise RuntimeError("NONFINITE_SFT_LOSS: technical stop")
                    loss.backward()
                step_telemetry = {}
                grad_norm = _optimizer_step(policy, optimizer, parameters, step_telemetry)
                counters["updates"] = step
                counters["input_tokens"] += len(generation.prompt_ids)
                counters["trained_tokens"] += len(generation.completion_ids)
                entry = {"step": step, "epoch": index // len(examples) + 1,
                         "loss": float(loss.detach().cpu()), "grad_norm": grad_norm,
                         "phase": phase, "example_index": example_index,
                         "prompt_tokens": len(generation.prompt_ids),
                         "trained_tokens": len(generation.completion_ids), **step_telemetry}
                if step % progress_every == 0 or step == max_updates:
                    journal.progress(step, max_updates, loss=entry["loss"],
                                     epoch=entry["epoch"], resource_counters=dict(counters))
                checkpoint_due = clock.checkpoint_due(time.monotonic() - started)
                if checkpoint_due or (checkpoint_time_fractions is None and (step % checkpoint_every == 0 or step == max_updates)):
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
                update_log.write(json.dumps(entry, ensure_ascii=False) + "\n")
                update_log.flush()
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
            if checkpoint_time_fractions is None and selection_is_better(metrics, final_step, best_metrics, best_step, closure_selection):
                best_step, best_key, best_metrics = final_step, key, metrics
        save_state(destination / "last", final_step)
        clock.pause(time.monotonic() - final_evaluation_started)
        frozen_after = parameter_hash(policy, trainable=False)
        if frozen_after != frozen_before:
            raise RuntimeError("FROZEN_BACKBONE_CHANGED: technical stop")
        result = {"algorithm": "completion_only_lora_sft", "status": "executed",
                  "seed": seed, "smoke": smoke, "best_step": best_step,
                  "selected_checkpoint": str(destination / f"step_{best_step:06d}"),
                  "final_checkpoint": str(destination / "last"), "stop_reason": stop_reason,
                  "parent_checkpoint": policy.origin_checkpoint, "plan": plan,
                  "trainable_before": before, "trainable_after": parameter_hash(policy, trainable=True),
                  "frozen_before": frozen_before, "frozen_after": frozen_after,
                  "history": history, "resource_counters": {**counters,
                    **save_budget.counters(),
                    "elapsed_seconds": time.monotonic() - started,
                    "training_seconds": clock.elapsed(time.monotonic() - started),
                    "validation_and_checkpoint_seconds": clock.paused_seconds,
                    "budget_overrun_seconds": max(0., clock.elapsed(time.monotonic() - started) - budget_seconds) if budget_seconds else 0.}}
        _write_json(destination / "training.json", result)
        journal.event("completed", step=final_step, selected_step=best_step, stop_reason=stop_reason,
                      resource_counters=result["resource_counters"])
        return result
