"""Closure-v4 training audit and explicitly gated, bounded continuations.

The historical campaign is read-only. Algebra fixtures do not certify the
reference Qwen; loading each real B1 is a separate check with separate evidence.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from research.src.artifacts import ROOT, digest, file_hash
from research.src.training.sft import _write_json, completion_label_mask, parameter_hash
from research.src.training.episodic_grpo import (
    clipped_token_loss, group_advantages, group_telemetry, load_b1_reference,
    reference_logprobs_on_actor_device, verify_reference,
)


def _check(identifier, status, evidence, *, critical=True, **details):
    return {"check_id": identifier, "id": identifier, "status": status,
            "critical": critical, "evidence": evidence, **details}


def _fresh(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Preserve previous training evidence: " + str(path))
    path.mkdir(parents=True)
    return path


def _selected_b1(parent_run, seed):
    path = Path(parent_run) / "sft" / str(seed) / "B1" / "training.json"
    value = json.loads(path.read_text())
    checkpoint = Path(value["selected_checkpoint"])
    if not checkpoint.is_dir():
        # Copies may move the workspace; map only within the verified B1 branch.
        checkpoint = path.parent / checkpoint.name
    return checkpoint, value


def algebra_fixtures():
    """Independent, non-LLM tensor probe with a known probability gradient."""
    import torch
    from research.src.training.sft import _optimizer_step
    from types import SimpleNamespace
    checks = []
    zero = group_advantages([1., 1., 1., 1.])
    varied = group_advantages([1., 0., -1., 0.])
    checks.append(_check("TRAIN-GRPO-01", "verified" if zero == [0.] * 4 and varied[0] > 0 > varied[2] and varied[1] == 0 else "failed",
                         "tensor_fixtures", zero_advantages=zero, varied_advantages=varied,
                         std_convention="population_ddof_0", epsilon=1e-6))
    logp = torch.tensor([-.7, -.8], requires_grad=True)
    old = logp.detach().clone()
    loss, kl = clipped_token_loss(logp, old, old, 1.)
    loss.backward()
    ok = float(kl.detach()) == 0. and bool(torch.allclose((logp.detach() - old).exp(), torch.ones(2))) and bool((logp.grad < 0).all())
    losses = []
    returns = [1., 0., -1., 0.]
    for index in range(4):
        losses.append(float(clipped_token_loss(old, old, old, varied[index])[0]))
    permutation = [2, 0, 3, 1]
    permuted = group_advantages([returns[i] for i in permutation])
    ok &= all(abs(permuted[j] - varied[i]) < 1e-12 for j, i in enumerate(permutation))
    ok &= abs(sum(losses) - sum(losses[i] for i in permutation)) < 1e-12
    checks.append(_check("TRAIN-GRPO-02", "verified" if ok else "failed", "tensor_fixtures",
                         identical_policy_kl=float(kl.detach()), old_policy_ratio=[1., 1.], permutation=permutation))
    # This tiny residual linear adapter tests the sign of optimizer credit. It is
    # deliberately not described as Qwen training or task-level RL learning.
    torch.manual_seed(101)
    base = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    lora_a = torch.nn.Parameter(torch.tensor([[1.]]))
    lora_b = torch.nn.Parameter(torch.zeros(2, 1))
    parameters = [("lora_A", lora_a), ("lora_B", lora_b)]
    optimizer = torch.optim.AdamW([p for _, p in parameters], lr=.05, weight_decay=0.)
    policy = SimpleNamespace(torch=torch)
    logits = base + (lora_b @ lora_a).flatten()
    probability_before = float(logits.softmax(-1)[0].detach())
    before = [p.detach().clone() for _, p in parameters]
    objective = -logits.log_softmax(-1)[0]
    objective.backward()
    telemetry = {}
    _optimizer_step(policy, optimizer, parameters, telemetry)
    probability_after = float((base + (lora_b @ lora_a).flatten()).softmax(-1)[0].detach())
    changed = any(not torch.equal(p, old_p) for (_, p), old_p in zip(parameters, before))
    checks.append(_check("TRAIN-GRAD-01", "verified" if changed and probability_after > probability_before and not base.requires_grad and base.grad is None else "failed",
                         "tensor_fixtures", scope="tiny residual LoRA algebra only; not Qwen or task efficacy",
                         selected_probability_before=probability_before, selected_probability_after=probability_after, **telemetry))
    labels = completion_label_mask([10, 11, 12], [20, 21])
    checks.append(_check("TRAIN-MASK-01", "verified" if labels == [-100, -100, -100, 20, 21] else "failed",
                         "tensor_fixtures", labels=labels, padding_policy="unbatched likelihood, no padding targets"))
    return checks


def audit_b3_window(parent_run, output, start=3400, end=3600):
    """Inspect original examples and masks; never claim an unexecuted replay."""
    output = Path(output)
    branch = Path(parent_run) / "continuations/303/B3"
    trace = branch / "prefixes.jsonl"
    updates_path = branch / "updates.jsonl"
    if not trace.is_file() or not updates_path.is_file():
        return {"status": "blocked_missing_evidence", "exact_replay": "not_executed",
                "missing": [str(p) for p in (trace, updates_path) if not p.is_file()]}
    updates = {}
    with updates_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if start <= row["step"] <= end:
                updates[row["step"]] = row
    selected, masks_ok = [], True
    window_path = output / "B3_303_WINDOW.jsonl"
    with trace.open() as stream, window_path.open("x") as target:
        for line in stream:
            row = json.loads(line)
            if row["step"] > end:
                break
            if start <= row["step"] <= end:
                generation = row["generation"]
                expected = [0] * len(generation["prompt_ids"]) + [1] * len(generation["completion_ids"])
                masks_ok &= row["actor_mask"] == expected
                row["historical_update"] = updates.get(row["step"])
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
                selected.append({"step": row["step"], "example_index": row["example_index"], "phase": row["phase"],
                                 "prompt_tokens": len(generation["prompt_ids"]), "completion_tokens": len(generation["completion_ids"]),
                                 "update": updates.get(row["step"])})
    training = json.loads((branch / "training.json").read_text()) if (branch / "training.json").is_file() else {}
    checkpoints = []
    for checkpoint in sorted(branch.glob("step_*")):
        if checkpoint.is_dir() and checkpoint.name.split("_")[-1].isdigit():
            step = int(checkpoint.name.split("_")[-1])
            optimizer = checkpoint.with_name(checkpoint.name + ".optimizer_state.pt")
            checkpoints.append({"step": step, "path": str(checkpoint), "optimizer_path": str(optimizer),
                                "optimizer_present": optimizer.is_file(),
                                "optimizer_sha256": file_hash(optimizer) if optimizer.is_file() else None})
    previous = [entry for entry in checkpoints if entry["step"] < start and entry["optimizer_present"]]
    result = {"status": "inspected" if len(selected) == end - start + 1 and masks_ok else "failed",
              "window": [start, end], "examples": len(selected), "mask_correct": masks_ok,
              "window_path": str(window_path), "window_sha256": file_hash(window_path),
              "observed_max_loss": max((row.get("loss", -math.inf) for row in updates.values()), default=None),
              "step_3502": updates.get(3502), "training_plan": training.get("plan"),
              "checkpoints": checkpoints, "nearest_pre_window_recovery": max(previous, key=lambda x: x["step"]) if previous else None,
              "exact_replay": "not_executed",
              "cause_classification": "hypothesis",
              "limitation": "Masks and logged loss are inspected, not causal proof of overfitting. Exact replay needs recovery from the nearest prior model+optimizer+RNG, replaying all intervening updates on the historical stack. No such replay was run."}
    _write_json(output / "B3_303_DIAGNOSTIC.json", result)
    return result


def _b1_probe(checkpoint, output, device, *, profile_capacity=False, call_counter=None):
    """Load actual B1 twice, check causal alignment and adapter round trip."""
    import torch
    from llm_local_001.model import Generation, artifact_hashes
    from research.src.policy.llm_controller import SpecLocalPolicy
    checkpoint, output = Path(checkpoint), Path(output)
    call_counter = call_counter if call_counter is not None else [0]
    initial_calls = call_counter[0]
    output.mkdir(parents=True)
    actor = reference = reloaded = None
    try:
        actor = SpecLocalPolicy.from_checkpoint(checkpoint, trainable=True, device=device)
        reference = load_b1_reference(checkpoint, device="cpu")
        verify_reference(actor, reference)
        # A fixed public task without gold/latent metadata. Labels have an exact,
        # independently known shift and the prompt contains a simulator-like turn.
        generation = actor.supervised_generation("Usuario: prefiero pasta. Simulador: solo hoy. Selecciona una acción JSON.",
                                                 '{"action":"PROCEED","slot":null}')
        with torch.no_grad(), actor.likelihood_mode():
            actual = actor.sequence_logprobs(generation, sampling_distribution=True)
            ids = generation.prompt_ids + generation.completion_ids
            full = torch.tensor([ids], device=actor.device)
            logits = actor.model(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False).logits
            shifted = logits[0, len(generation.prompt_ids) - 1:-1, :].float() / actor.config.temperature
            targets = torch.tensor(generation.completion_ids, device=actor.device)
            expected = shifted.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
            mask_error = float((expected - actual).abs().max().cpu())
        reference_values = reference_logprobs_on_actor_device(reference, actor, generation)
        # BF16 checkpointing and full projection may have different kernel shapes.
        tolerance = .02 if actor.config.precision in {"bfloat16", "float16"} else 1e-5
        reference_error = float((actual - reference_values.to(actual.device)).abs().max().cpu())
        kl_difference = reference_values.to(actual.device) - actual
        initial_kl = float((kl_difference.exp() - kl_difference - 1.).mean().cpu())
        frozen_reference = not any(p.requires_grad for p in reference.model.parameters())
        adapter_names = [name for name, parameter in actor.model.named_parameters() if parameter.requires_grad]
        effective_before = parameter_hash(actor)
        call_counter[0] += 1
        greedy_before = actor.generate("Responde con una palabra: pasta.", sample=False)
        roundtrip = output / "roundtrip_adapter"
        actor.save_checkpoint(roundtrip)
        del reference
        reference = None
        del actor
        actor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reloaded = SpecLocalPolicy.from_checkpoint(roundtrip, trainable=True, device=device)
        with torch.no_grad(), reloaded.likelihood_mode():
            restored = reloaded.sequence_logprobs(generation, sampling_distribution=True)
        roundtrip_error = float((actual.cpu() - restored.cpu()).abs().max())
        call_counter[0] += 1
        greedy_after = reloaded.generate("Responde con una palabra: pasta.", sample=False)
        effective_after = parameter_hash(reloaded)
        # One known supervised signal on an isolated copy checks the actual
        # Qwen/LoRA optimizer path. This is a technical probe, not task training.
        from research.src.training.sft import _optimizer_step, _save_training_state
        parameters = [(name, p) for name, p in reloaded.model.named_parameters() if p.requires_grad]
        frozen_before = parameter_hash(reloaded, trainable=False)
        adapter_before = parameter_hash(reloaded, trainable=True)
        optimizer = torch.optim.AdamW([p for _, p in parameters], lr=5e-6, weight_decay=0.)
        optimizer.zero_grad(set_to_none=True)
        with reloaded.likelihood_mode():
            probe_logprobs = reloaded.sequence_logprobs(generation, sampling_distribution=True)
            probe_loss = -probe_logprobs.mean()
            probe_loss.backward()
        gradient = {}
        _optimizer_step(reloaded, optimizer, parameters, gradient)
        with torch.no_grad(), reloaded.likelihood_mode():
            updated_logprobs = reloaded.sequence_logprobs(generation, sampling_distribution=True)
        adapter_after = parameter_hash(reloaded, trainable=True)
        frozen_after = parameter_hash(reloaded, trainable=False)
        probability_direction = float(updated_logprobs.mean().cpu()) - float(probe_logprobs.detach().mean().cpu())
        signal_checkpoint = output / "known_signal_adapter"
        reloaded.save_checkpoint(signal_checkpoint)
        _save_training_state(reloaded, optimizer, signal_checkpoint, 1, {"updates": 1, "scope": "technical_known_signal_only"})
        actual_gradient_ok = (adapter_before != adapter_after and frozen_before == frozen_after
                              and probability_direction > 0 and gradient["gradient_norm_post_clip"] <= 1.0001)
        capacity = {"status": "not_executed", "reason": "Maximum-length profile is shared across identical model/adapter shapes; executed on seed 101"}
        if profile_capacity:
            from research.src.training.resources import probe_context_capacity, probe_generation_capacity
            reloaded.model.zero_grad(set_to_none=True)
            if str(reloaded.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(reloaded.device)
            maximum_backward = probe_context_capacity(reloaded, input_tokens=2048, output_tokens=512)
            call_counter[0] += 1
            maximum_generation = probe_generation_capacity(reloaded, input_tokens=2048, output_tokens=512)
            capacity = {"status": "passed", "scope": "synthetic typical and maximum hardware capacity, not semantic evidence coverage",
                "typical_input_tokens": len(generation.prompt_ids), "typical_actor_output_tokens": len(generation.completion_ids),
                "maximum_forward_backward": maximum_backward, "maximum_generation": maximum_generation}
            if str(reloaded.device).startswith("cuda"):
                free_bytes, total_bytes = torch.cuda.mem_get_info(reloaded.device)
                capacity.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(reloaded.device),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(reloaded.device),
                    cuda_free_bytes=free_bytes, cuda_total_bytes=total_bytes,
                    bf16_supported=bool(torch.cuda.is_bf16_supported()))
        result = {"status": "passed" if mask_error <= tolerance and reference_error <= tolerance and roundtrip_error <= tolerance and frozen_reference and effective_before == effective_after and greedy_before.completion_ids == greedy_after.completion_ids else "failed",
                  "checkpoint": str(checkpoint), "checkpoint_artifacts_sha256": artifact_hashes(checkpoint),
                  "device": device, "precision": reloaded.config.precision, "atol_logprob": tolerance,
                  "causal_shift_max_logprob_error": mask_error, "reference_max_logprob_error": reference_error,
                  "initial_kl": initial_kl, "roundtrip_max_logprob_error": roundtrip_error,
                  "effective_weight_hash_before": effective_before, "effective_weight_hash_after": effective_after,
                  "reference_frozen": frozen_reference, "trainable_names": adapter_names,
                  "all_trainable_are_lora": bool(adapter_names) and all("lora_" in name for name in adapter_names),
                  "greedy_completion_before": greedy_before.completion_ids,
                  "greedy_completion_after": greedy_after.completion_ids,
                  "known_signal_probe": {"status": "passed" if actual_gradient_ok else "failed",
                    "optimizer_updates": 1, "mean_target_logprob_change": probability_direction,
                    "trainable_before": adapter_before, "trainable_after": adapter_after,
                    "frozen_before": frozen_before, "frozen_after": frozen_after,
                    "scope": "single technical known-signal update on isolated B1 copy; no RL task-efficacy evidence", **gradient},
                  "capacity_profile": capacity,
                  "new_llm_calls": call_counter[0] - initial_calls,
                  "generation": {"prompt_ids": generation.prompt_ids, "completion_ids": generation.completion_ids},
                  "scope": "fixed public likelihood and deterministic decoding; not task competence or learning efficacy"}
        if not result["all_trainable_are_lora"] or not actual_gradient_ok:
            result["status"] = "failed"
        _write_json(output / "B1_IDENTITY.json", result)
        return result
    finally:
        del actor, reference, reloaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_training_checks(parent_run, output, *, run_torch_fixtures=True, check_real_b1=True,
                        device="cuda", training_seeds=(101, 202, 303), model_path=None):
    """Return matrix-compatible checks and actual evidence in a fresh subfolder."""
    output = _fresh(Path(output) / "training_checks")
    checks = []
    call_counter = [0]
    started = time.monotonic()
    try:
        if run_torch_fixtures:
            checks.extend(algebra_fixtures())
        else:
            checks.append(_check("TRAIN-TENSORS", "not_executed", "run_torch_fixtures=False"))
    except Exception as exc:
        checks.append(_check("TRAIN-TENSORS", "failed", "tensor_fixtures", error=f"{type(exc).__name__}: {exc}"))
    for seed in training_seeds:
        try:
            checkpoint, training = _selected_b1(parent_run, seed)
            from llm_local_001.model import artifact_hashes
            manifest = json.loads((checkpoint / "local_checkpoint.json").read_text())
            artifact_ok = artifact_hashes(checkpoint) == manifest["artifacts_sha256"]
            checks.append(_check(f"TRAIN-B1-ARTIFACTS-{seed}", "verified" if artifact_ok else "failed", str(checkpoint),
                                 seed=seed, artifact_hashes=artifact_hashes(checkpoint)))
            if check_real_b1:
                evidence = _b1_probe(checkpoint, output / str(seed), device, profile_capacity=seed == training_seeds[0], call_counter=call_counter)
                checks.append(_check(f"TRAIN-B1-IDENTITY-{seed}", "verified" if evidence["status"] == "passed" else "failed",
                                     str(output / str(seed) / "B1_IDENTITY.json"), **{key: value for key, value in evidence.items() if key != "status"}))
            else:
                checks.append(_check(f"TRAIN-B1-IDENTITY-{seed}", "not_executed", str(checkpoint),
                                     note="Artifact hashes do not replace effective-weight, logits and reload checks"))
        except (FileNotFoundError, ImportError) as exc:
            checks.append(_check(f"TRAIN-B1-IDENTITY-{seed}", "not_identifiable_from_current_artifacts", str(parent_run), error=str(exc)))
        except Exception as exc:
            checks.append(_check(f"TRAIN-B1-IDENTITY-{seed}", "failed", str(parent_run), error=f"{type(exc).__name__}: {exc}"))
    b3 = audit_b3_window(parent_run, output)
    checks.append(_check("TRAIN-B3-303-WINDOW", "verified" if b3["status"] == "inspected" else "not_identifiable_from_current_artifacts" if b3["status"] == "blocked_missing_evidence" else "failed",
                         str(output / "B3_303_DIAGNOSTIC.json"), critical=False, exact_replay="not_executed"))
    checks.append(_check("TRAIN-B3-303-EXACT-REPLAY", "not_executed", str(output / "B3_303_DIAGNOSTIC.json"), critical=False,
                         reason="Preserved optimizer/RNG and prefixes permit a replay from step zero; 3400+ intervening updates were not run"))
    critical = [check for check in checks if check["critical"]]
    failed = any(check["status"] == "failed" for check in critical)
    passed = bool(critical) and all(check["status"] == "verified" for check in critical)
    result = {"status": "passed" if passed else "failed" if failed else "blocked_missing_evidence",
              "critical_checks_passed": passed, "checks": checks, "b3_303": b3,
              "evidence_path": str(output / "PRUEBAS_ENTRENAMIENTO.json"),
              "elapsed_seconds": time.monotonic() - started,
              "new_llm_calls": call_counter[0],
              "learning_evidence": "not_executed", "scope": "technical training audit, no task-level RL efficacy conclusion"}
    _write_json(output / "PRUEBAS_ENTRENAMIENTO.json", result)
    return result


def _read_training_banks(train_bank, validation_bank, validation_roots=None):
    from research.src.training.campaign import load_bank
    train = load_bank(train_bank, "train")
    validation = load_bank(validation_bank, "validation")
    # Validate the full selection bank before any pilot subset is chosen. A
    # small selection cohort must not hide leakage in its omitted roots.
    from research.src.data.semantic_partition_v2 import assert_semantic_split_isolation_v2
    assert_semantic_split_isolation_v2({"train": train, "validation": validation})
    if validation_roots is not None:
        chosen = []
        for episode in validation:
            if episode["root_id"] not in chosen:
                chosen.append(episode["root_id"])
        selected = set(chosen[:validation_roots])
        validation = [episode for episode in validation if episode["root_id"] in selected]
    if {e["root_id"] for e in train} & {e["root_id"] for e in validation}:
        raise ValueError("Training and selection roots overlap")
    return train, validation


class BudgetedValidation:
    """Predeclared accounting for complete, frozen selection cohorts."""
    def __init__(self, callback, budget_seconds):
        if budget_seconds <= 0:
            raise ValueError("Validation budget must be positive")
        self.callback = callback
        self.budget_seconds = budget_seconds
        self.elapsed_seconds = 0.
        self.calls = 0

    def __call__(self, policy):
        if self.elapsed_seconds >= self.budget_seconds:
            raise RuntimeError("VALIDATION_BUDGET_EXHAUSTED: no new selection evaluation may start")
        started = time.monotonic()
        try:
            return self.callback(policy)
        finally:
            self.elapsed_seconds += time.monotonic() - started
            self.calls += 1

    def counters(self):
        return {"elapsed_seconds": self.elapsed_seconds, "allocated_seconds": self.budget_seconds,
                "evaluations": self.calls, "overrun_seconds": max(0., self.elapsed_seconds - self.budget_seconds)}


def _training_plan(authorization_dir, *, final):
    from research.src.closure_protocol import require_training_authorization, verify_locked_protocol
    require_training_authorization(authorization_dir)
    if final:
        locked = verify_locked_protocol(authorization_dir)
        return locked["plan"]
    return json.loads((Path(authorization_dir) / "PILOT_PLAN.json").read_text())


def _assert_plan_value(plan, key, actual):
    if key in plan and plan[key] != actual:
        raise ValueError(f"PREDECLARED_PLAN_MISMATCH: {key}: {plan[key]!r} != {actual!r}")


def _selection_config(plan, *, required=False):
    value = plan.get("selection_validity_guards")
    if value is None:
        if required:
            raise ValueError("Locked v4 selection requires selection_validity_guards")
        return None
    if not isinstance(value, dict):
        raise ValueError("Selection validity guards must be a dictionary")
    if "validity_guards" in value:
        return value
    return {"utility_tolerance": float(plan.get("selection_utility_tolerance", 1e-6)), "validity_guards": value}


def run_pilot(parent_run, output, *, device="cuda", updates=5, roots_per_batch=2,
              budget_seconds=300, authorization_dir=None, train_bank=None,
              validation_bank=None, validation_roots=5, validation_budget_seconds=180,
              signal_diagnostic_groups=0):
    """Actual Qwen branches, seed 101; no synthetic reward substitution.

    Authorize through the closure evidence chain before loading any model. The
    resulting 5-step pilot characterizes mechanics and signal, never efficacy.
    """
    from research.src.closure_protocol import execution_context
    from research.src.training.campaign import validation_callback, _release_memory
    from research.src.training.episodic_grpo import train_grpo
    from research.src.training.rollouts import make_rollout_collector
    from research.src.policy.llm_controller import SpecLocalPolicy
    if not 1 <= updates <= 5 or not 1 <= roots_per_batch <= 2 or not 0 < budget_seconds <= 3600 or not 0 <= signal_diagnostic_groups <= 40:
        raise ValueError("The v4 pilot permits at most 5 updates, 2 groups/update and a fixed positive budget <=3600 seconds")
    authorization_dir = Path(authorization_dir or output)
    plan = _training_plan(authorization_dir, final=False)
    selection = _selection_config(plan)
    save_budget_seconds = plan.get("checkpoint_save_budget_seconds", plan.get("budgets", {}).get("save_seconds_per_arm_seed"))
    if save_budget_seconds is None:
        raise ValueError("Pilot requires an explicit separate checkpoint save budget")
    for key, actual in {"training_seed": 101, "max_updates_per_branch": updates,
                        "roots_per_batch": roots_per_batch, "budget_seconds_per_branch": budget_seconds,
                        "validation_budget_seconds_per_branch": validation_budget_seconds,
                        "validation_roots": validation_roots,
                        "signal_diagnostic_groups": signal_diagnostic_groups}.items():
        _assert_plan_value(plan, key, actual)
    train_bank = Path(train_bank or plan.get("train_bank", ROOT / "data/bank_1_1/train.jsonl"))
    validation_bank = Path(validation_bank or plan.get("validation_bank", ROOT / "data/bank_1_1/validation.jsonl"))
    if "train_bank" in plan and Path(plan["train_bank"]).resolve() != train_bank.resolve():
        raise ValueError("Predeclared training bank changed")
    if "validation_bank" in plan and Path(plan["validation_bank"]).resolve() != validation_bank.resolve():
        raise ValueError("Predeclared validation bank changed")
    train, validation = _read_training_banks(train_bank, validation_bank, validation_roots)
    destination = _fresh(Path(output) / "pilot")
    checkpoint, _ = _selected_b1(parent_run, 101)
    eligible_steps = sorted(set((1, updates)))
    rollout_calls_bound = 2 * updates * roots_per_batch * 4 * 3
    validation_calls_bound = (1 + 2 * len(eligible_steps)) * len(validation) * 3
    calls_bound = rollout_calls_bound + validation_calls_bound + signal_diagnostic_groups * 4 * 3
    call_budget = plan.get("max_new_llm_calls")
    if call_budget is not None and calls_bound > call_budget:
        result = {"status": "not_executed_budget", "learning_evidence": "not_executed",
                  "decision": "close_inconclusive", "planned_max_llm_calls": calls_bound,
                  "available_llm_calls": call_budget,
                  "reason": "The complete predeclared pilot exceeds the remaining call budget; no model was loaded or branch partially executed"}
        _write_json(destination / "PILOTO_RL.json", result)
        return result
    result = {"status": "running", "scope": "technical and task-signal pilot only; no efficacy inference",
              "training_seed": 101, "parent_checkpoint": str(checkpoint), "arms": {},
              "train_bank_sha256": file_hash(train_bank), "validation_bank_sha256": file_hash(validation_bank),
              "selection_cohort_sha256": digest(validation), "new_training_updates": 0}
    result.update(planned_max_llm_calls=calls_bound, available_llm_calls=call_budget,
                  shared_step_zero_validation=True)
    started = time.monotonic()
    initial_metrics = None
    with execution_context(authorization_dir, "training"):
        for horizon, method in (("long", "B2-L-v4"), ("short", "B2-I-v4")):
            actor = reference = None
            try:
                actor = SpecLocalPolicy.from_checkpoint(checkpoint, trainable=True, device=device)
                actor.config.temperature = 1.
                reference = load_b1_reference(checkpoint, device="cpu")
                reference.config.temperature = 1.
                validate = BudgetedValidation(validation_callback(validation,
                    log_directory=destination / method / "validation", batch_size=1), validation_budget_seconds)
                arm = train_grpo(actor, reference, train,
                    make_rollout_collector(destination / method / "episodes", method=method),
                    destination / method / "training", validate, horizon=horizon, updates=updates,
                    seed=101, roots_per_batch=roots_per_batch, budget_seconds=budget_seconds,
                    learning_rate=5e-6, checkpoint_steps=eligible_steps,
                    initial_validation_metrics=initial_metrics,
                    closure_selection=selection,
                    checkpoint_save_budget_seconds=save_budget_seconds,
                    closure_authorization=authorization_dir, reference_execution="cpu_storage_gpu_compute")
                if initial_metrics is None:
                    initial_metrics = dict(arm["history"][0]["validation"])
                arm["validation_budget"] = validate.counters()
                arm["resource_counters"].update(validation_seconds=validate.elapsed_seconds,
                    save_seconds=arm["resource_counters"].get("checkpoint_save_seconds", 0.))
                result["arms"][method] = arm
                result["new_training_updates"] += arm["resource_counters"]["updates"]
            except Exception as exc:
                result["status"] = "failed"
                result["error"] = f"{type(exc).__name__}: {exc}"
                _write_json(destination / "PILOTO_RL.json", result)
                raise
            finally:
                del actor, reference
                _release_memory()
        # Additional signal observations use the original selected B1 and a
        # fixed schedule. They are not updates and are never recycled as GRPO
        # on-policy samples of either updated actor.
        if signal_diagnostic_groups:
            from research.src.training.episodic_grpo import root_batch, validate_group
            frozen = SpecLocalPolicy.from_checkpoint(checkpoint, trainable=False, device=device)
            frozen.config.temperature = 1.
            collector = make_rollout_collector(destination / "signal_diagnostic/episodes", method="B1-frozen-signal-diagnostic")
            with (destination / "signal_diagnostic/group_diagnostics.jsonl").open("x") as stream:
                for group_index in range(signal_diagnostic_groups):
                    episode = root_batch(train, group_index, 1)[0]
                    rollouts = []
                    for sample_index in range(4):
                        seed = 101000000000 + group_index * 4 + sample_index
                        frozen.seed(seed)
                        rollouts.append(collector(frozen, episode, seed))
                    validate_group(rollouts)
                    horizon = "long" if group_index % 2 == 0 else "short"
                    stream.write(json.dumps({"diagnostic_group": group_index,
                        "source": "frozen_selected_B1_no_optimizer_update", "horizon": horizon,
                        "checkpoint": str(checkpoint), **group_telemetry(rollouts, horizon)}, ensure_ascii=False) + "\n")
                    stream.flush()
            del frozen
            _release_memory()
    group_rows = []
    diagnostic_paths = list(destination.glob("*/training/group_diagnostics.jsonl"))
    diagnostic_paths.extend(destination.glob("signal_diagnostic/group_diagnostics.jsonl"))
    for path in diagnostic_paths:
        with path.open() as stream:
            group_rows.extend(json.loads(line) for line in stream)
    task_zero = sum(row["std_task"] == 0. for row in group_rows)
    consumer = plan.get("consumer_useful", {})
    consumer_verified = (isinstance(consumer, dict) and consumer.get("status") == "verified"
                         and consumer.get("evidence_path") and Path(consumer["evidence_path"]).is_file()
                         and file_hash(consumer["evidence_path"]) == consumer.get("evidence_sha256"))
    weight_changes = all(arm["trainable_before"] != arm["trainable_after"] for arm in result["arms"].values())
    continue_allowed = (len(group_rows) >= 40 and task_zero / len(group_rows) <= .9 and consumer_verified and weight_changes
                        and result["new_training_updates"] > 0)
    result.update(status="passed", learning_evidence="inconclusive", groups_observed=len(group_rows),
                  zero_task_signal_fraction=task_zero / len(group_rows) if group_rows else None,
                  signal_window_satisfied=len(group_rows) >= 40,
                  decision="continue" if continue_allowed else "close_inconclusive",
                  consumer_usefulness_verified=bool(consumer_verified),
                  all_branches_changed_weights=weight_changes,
                  frozen_diagnostic_groups=signal_diagnostic_groups,
                  signal_decision="diagnostic_required" if len(group_rows) >= 40 and task_zero / len(group_rows) > .9 else "insufficient_window" if len(group_rows) < 40 else "task_signal_observed",
                  elapsed_seconds=time.monotonic() - started)
    _write_json(destination / "PILOTO_RL.json", result)
    return result


def run_final_training(parent_run, output, *, authorization_dir=None, device="cuda",
                       train_bank=None, validation_bank=None, seeds=(101, 202, 303),
                       budget_seconds=3600, validation_budget_seconds=1800, roots_per_batch=2):
    """Execute the three continuations only under a sealed v4 protocol.

    Time-based selection candidates are step zero and the atomic updates crossing
    one third, two thirds and the complete training allocation. Final diagnostic
    snapshots are preserved and cannot become extra eligible candidates.
    """
    from research.src.closure_protocol import execution_context
    from research.src.training.campaign import validation_callback, _release_memory
    from research.src.training.episodic_grpo import train_grpo
    from research.src.training.sft import train_sft
    from research.src.training.rollouts import make_rollout_collector
    from research.src.data.build_bank import teacher_trajectories
    from research.src.policy.llm_controller import SpecLocalPolicy
    from llm_local_001.model import artifact_hashes
    authorization_dir = Path(authorization_dir or output)
    plan = _training_plan(authorization_dir, final=True)
    selection = _selection_config(plan, required=True)
    save_budget_seconds = plan.get("checkpoint_save_budget_seconds", plan.get("budgets", {}).get("save_seconds_per_arm_seed"))
    if save_budget_seconds is None:
        raise ValueError("Final continuations require an explicit separate checkpoint save budget")
    _assert_plan_value(plan, "training_seeds", list(seeds))
    _assert_plan_value(plan, "training_budget_seconds_per_branch", budget_seconds)
    _assert_plan_value(plan, "validation_budget_seconds_per_branch", validation_budget_seconds)
    _assert_plan_value(plan, "roots_per_batch", roots_per_batch)
    if train_bank is None:
        train_bank = plan.get("train_bank")
    if validation_bank is None:
        validation_bank = plan.get("validation_bank")
    if train_bank is None or validation_bank is None:
        raise ValueError("Locked final training requires explicit train_bank and validation_bank")
    for name, path in (("train_bank", train_bank), ("validation_bank", validation_bank)):
        if name in plan and Path(path).resolve() != Path(plan[name]).resolve():
            raise ValueError("Locked final training bank changed: " + name)
    train, validation = _read_training_banks(train_bank, validation_bank)
    destination = _fresh(Path(output) / "final_training")
    examples = list(teacher_trajectories(train))
    selected = []
    results = []
    with execution_context(authorization_dir, "training"):
        for seed in seeds:
            checkpoint, _ = _selected_b1(parent_run, seed)
            selected.append({"method": "B1-v4", "seed": seed, "checkpoint": str(checkpoint), "path": str(checkpoint), "checkpoint_id": checkpoint.name,
                             "selected_step": 0, "selection_split": "validation", "selected_before_test": True,
                             "artifacts_sha256": {str(checkpoint / filename): value for filename, value in artifact_hashes(checkpoint).items()}})
            for method, horizon in (("B3-v4", None), ("B2-I-v4", "short"), ("B2-L-v4", "long")):
                actor = reference = None
                branch = destination / str(seed) / method
                try:
                    actor = SpecLocalPolicy.from_checkpoint(checkpoint, trainable=True, device=device)
                    actor.config.temperature = 1.
                    validate = BudgetedValidation(validation_callback(validation,
                        log_directory=branch / "validation", batch_size=1), validation_budget_seconds)
                    common = dict(seed=seed, budget_seconds=budget_seconds,
                                  closure_authorization=authorization_dir,
                                  closure_selection=selection,
                                  checkpoint_save_budget_seconds=save_budget_seconds,
                                  checkpoint_time_fractions=[1/3, 2/3, 1.])
                    if horizon is None:
                        arm = train_sft(actor, examples, branch / "training", validate,
                                        continuation=True, learning_rate=1e-5, **common)
                    else:
                        reference = load_b1_reference(checkpoint, device="cpu")
                        reference.config.temperature = 1.
                        arm = train_grpo(actor, reference, train,
                            make_rollout_collector(branch / "episodes", method=method),
                            branch / "training", validate, horizon=horizon, updates=2**31 - 1,
                            roots_per_batch=roots_per_batch, learning_rate=5e-6,
                            reference_execution="cpu_storage_gpu_compute", **common)
                    arm["validation_budget"] = validate.counters()
                    arm["resource_counters"].update(validation_seconds=validate.elapsed_seconds,
                        save_seconds=arm["resource_counters"].get("checkpoint_save_seconds", 0.))
                    results.append(arm)
                    chosen = Path(arm["selected_checkpoint"])
                    selected.append({"method": method, "seed": seed, "checkpoint": str(chosen), "path": str(chosen), "checkpoint_id": chosen.name,
                        "selected_step": arm["best_step"], "selection_split": "validation", "selected_before_test": True,
                        "selection_training_path": str(branch / "training/training.json"),
                        "artifacts_sha256": {str(chosen / filename): value for filename, value in artifact_hashes(chosen).items()}})
                finally:
                    del actor, reference
                    _release_memory()
    result = {"status": "passed", "training_seeds": list(seeds), "arms": results,
              "checkpoints": selected, "selected_checkpoints": selected, "selection_split": "validation"}
    _write_json(destination / "ENTRENAMIENTO_FINAL.json", result)
    _write_json(Path(output) / "CHECKPOINTS_SELECCIONADOS.json", {"checkpoints": selected})
    return result
