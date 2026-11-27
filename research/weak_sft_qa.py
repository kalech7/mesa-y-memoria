"""Audit the weak SFT pilot from saved artifacts, without loading model weights."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import random
from math import isclose, isfinite
from pathlib import Path
from statistics import fmean, pstdev

from .competence_evaluation import aggregate_rows
from .competence_training import checkpoint_evidence
from .imperfect_sft import load_examples
from .src.artifacts import digest, file_hash, write_json
from .src.closure_protocol import read_manifest, require_training_authorization
from .src.data.semantic_partition_v3 import assert_semantic_split_isolation_v3
from .src.scoring.evidence import Evaluator

METHODS = {"B1-weak", "B3", "B2-L", "B3-terminal", "B2-L-terminal"}
ADAPTER = "adapter_model.safetensors"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path):
    return json.loads(Path(path).read_text())


def _rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _identity(row):
    return tuple(row[key] for key in ("root_id", "latent_world_id", "realization_id"))


def _close(left, right):
    return isclose(left, right, rel_tol=0, abs_tol=1e-9)


def _archived_paths(run, plan, manifest):
    """Map original absolute evidence paths to the current copy and its repository."""
    repo = Path(__file__).resolve().parents[1]
    old_repo = Path(plan["runner_path"]).parents[1]
    old_run = Path(manifest["stages"]["inventory"]["path"]).parents[1]
    parent = Path(plan["parent"]["path"])
    # The parent may come from a different historical clone of this project.
    parent_runs = next((base for base in parent.parents if base.parts[-2:] == ("research", "runs")), None)
    prefixes = [(old_run, run), (old_repo, repo)]
    if parent_runs is not None and not parent.is_relative_to(old_repo):
        prefixes.append((parent, repo / "research/runs" / parent.relative_to(parent_runs)))

    def resolve(recorded):
        path = Path(recorded)
        for original, current in prefixes:
            if path.is_relative_to(original):
                return (current / path.relative_to(original)).resolve()
        return path.resolve()

    return resolve


def _check_archived_stages(run, manifest, resolve):
    """Check the recorded authorization evidence without authorizing new training."""
    _require(manifest["pilot_plan_sha256"] == file_hash(run / "PILOT_PLAN.json"),
             "Archived pilot plan changed")
    for name in ("inventory", "test-environment", "test-training", "diagnose"):
        record = manifest["stages"][name]
        path = run / "stages" / f"{name}.json"
        _require(resolve(record["path"]) == path and record["sha256"] == file_hash(path), f"Archived stage changed: {name}")
        stage = _json(path)
        _require(stage["stage"] == name and stage["code_snapshot_sha256"] == manifest["code_snapshot_sha256"]
                 and stage["artifacts_sha256"], f"Unbound archived stage: {name}")
        for filename, expected in stage["artifacts_sha256"].items():
            _require(file_hash(resolve(filename)) == expected, f"Archived stage evidence changed: {filename}")
        result = stage["result"]
        passed = result.get("critical_checks_passed") is True if name == "diagnose" else result["status"] == "passed"
        _require(passed, f"Archived stage did not pass: {name}")


def _checkpoint(path):
    evidence = checkpoint_evidence(path)
    manifest = _json(Path(path) / "local_checkpoint.json")
    return {
        "checkpoint_path": evidence["path"],
        "checkpoint_manifest_sha256": evidence["manifest_sha256"],
        "adapter_sha256": evidence["artifacts_sha256"][ADAPTER],
        "policy_fingerprint": digest({
            "base": manifest["base_artifacts_sha256"],
            "precision": manifest["config"]["precision"],
            "artifacts": {name: value for name, value in evidence["artifacts_sha256"].items()
                          if name != "README.md"},
        }),
        "base": manifest["base_artifacts_sha256"],
    }


def _rebased_checkpoint(record, resolve):
    return {**record, "path": str(resolve(record["path"]).resolve())}


def _panel(summary_path, bank, role, seed, *, selection_path=None, method=None, resolve=Path):
    summary_path = Path(summary_path)
    summary = _json(summary_path)
    _require(summary["status"] == "completed" and summary["role"] == role, f"Incomplete or wrong-role panel: {summary_path}")
    _require(summary["training_seed"] == seed, f"Wrong seed in {summary_path}")
    if method is not None:
        _require(summary["method"] == method, f"Wrong method in {summary_path}")
    _require(summary["weights_unchanged"] and summary["actor_before"] == summary["actor_after"]
             and summary["actor_before"]["sha256"], f"Panel lacks frozen actor evidence: {summary_path}")
    _require(summary["bank_content_sha256"] == digest(bank), f"Wrong bank digest: {summary_path}")
    predeclaration_path = resolve(summary["paths"]["predeclaration"])
    trajectories_path = resolve(summary["paths"]["trajectories"])
    declaration = _json(predeclaration_path)
    completion = _json(summary_path.with_name("completion.json"))
    _require(completion["status"] == "completed" and completion["training"] is False, f"Wrong completion: {summary_path}")
    _require(declaration["decoding"] == "greedy" and declaration["test_only"] is False
             and declaration["actor_before"] == summary["actor_before"], f"Invalid actor declaration: {summary_path}")
    for path in (summary_path, predeclaration_path, trajectories_path):
        name = str(path.resolve().relative_to(summary_path.parent.resolve()))
        _require(completion["artifacts_sha256"][name] == file_hash(path), f"Panel artifact changed: {path}")
    if selection_path is not None:
        authorization = declaration["protocol"]["heldback_authorization"]
        _require(resolve(authorization["selection_manifest_path"]).resolve() == selection_path.resolve()
                 and authorization["selection_manifest_sha256"] == file_hash(selection_path)
                 and authorization["locked_bank_sha256"] == digest(bank), f"Selection lock does not match {summary_path}")

    episodes = {_identity(row): row for row in bank}
    rows = _rows(trajectories_path)
    _require(len(rows) == len(episodes) == summary["completed_episodes"]
             and {_identity(row) for row in rows} == episodes.keys(), f"Wrong or duplicate panel cohort: {summary_path}")
    scorer = Evaluator()
    for row in rows:
        episode = episodes[_identity(row)]
        _require(row["condition"] == "O2" and row["route"] == "natural"
                 and row["method"] == summary["method"] and row["training_seed"] == seed
                 and row["evaluation_role"] == row["split"] == role
                 and row["scenario_family"] == episode["scenario_family"]
                 and row["technical_status"] == "completed"
                 and row["trace"]["episode"] == episode, f"Unmatched panel trajectory: {_identity(row)}")
        trace = row["trace"]
        current = scorer.score_current(trace)
        future = scorer.score_future(trace)
        memory = scorer.score_memory(trace)
        asked = int(trace.get("asked_slot") is not None)
        metrics = row["metrics"]
        _require(metrics["C2"] == current["success"] and metrics["FJS"] == future["success"]
                 and metrics["J_primary"] == current["success"] * future["success"]
                 and metrics["questions"] == asked
                 and metrics["write_operation_count"] == memory["operations"]
                 and metrics["unsupported_write_count"] == memory["unsupported_operations"]
                 and metrics["unsupported_write_episode"] == int(memory["unsupported_operations"] > 0)
                 and metrics["any_invalid_output"] == int(any(phase["error"] is not None for phase in row["phases"].values())),
                 f"Scoring mismatch for {_identity(row)}")
        persistence = row["persistence_verification"]
        _require(persistence["status"] == "verified"
                 and persistence["sqlite_sha256"] == file_hash(resolve(persistence["sqlite_path"])),
                 f"Unverified persistent state for {_identity(row)}")

    recomputed = aggregate_rows(rows)
    for key in ("metrics", "families", "by_root", "by_family"):
        _require(recomputed[key] == summary[key], f"Aggregation mismatch in {summary_path}: {key}")
    return recomputed, datetime.fromisoformat(declaration["created_at"])


def _check_steps(result):
    _require(result["status"] == "executed" and result["smoke"] is False, "Non-production optimizer result")
    _require(result["frozen_before"] == result["frozen_after"], "Frozen backbone changed")
    steps = result["history"][1:]
    count = result["resource_counters"]["updates"]
    _require(count > 0 and [row["step"] for row in steps] == list(range(1, count + 1)), "Missing optimizer steps")
    _require(all(row["optimizer_step_executed"] and not row["mixed_precision_skipped_steps"]
                 and isfinite(row["grad_norm"]) and isfinite(row["loss"]) for row in steps), "Invalid optimizer step evidence")
    return steps


def _sft(result, path, steps, bank, candidate, resolve=Path):
    bundle = resolve(candidate["bundle_manifest"]).parent
    examples, _ = load_examples(bundle, train=bank, expected_rate=candidate["demonstration_rate"],
                                expected_seed=7331)
    natural = [row for row in examples if row["route"] == "natural"]
    prefixes = _rows(Path(path).with_name("prefixes.jsonl"))
    _require(len(prefixes) == len(steps) and result["plan"]["examples_per_epoch"] == len(natural),
             "Missing supervised prefixes or unexpected demonstrations")
    schedule = list(range(len(natural)))
    rng = random.Random(result["seed"])
    rng.shuffle(schedule)
    trained = 0
    for step, prefix in zip(steps, prefixes, strict=True):
        offset = (step["step"] - 1) % len(schedule)
        if not offset and step["step"] > 1:
            rng.shuffle(schedule)
        expected_index = schedule[offset]
        example = natural[expected_index]
        generation = prefix["generation"]
        completion = example.get("completion", example.get("action"))
        expected_completion = json.loads(completion) if isinstance(completion, str) else completion
        _require(prefix["example_index"] == expected_index and prefix["phase"] == example["phase"].lower()
                 and json.loads(generation["text"]) == expected_completion,
                 f"SFT example differs from the frozen natural mixture at step {step['step']}")
        prompt_size, answer_size = len(generation["prompt_ids"]), len(generation["completion_ids"])
        _require(prefix["step"] == step["step"] and prefix["example_index"] == step["example_index"]
                 and prefix["phase"] == step["phase"] and answer_size > 0
                 and prefix["actor_mask"] == [0] * prompt_size + [1] * answer_size
                 and step["trained_tokens"] == answer_size, f"Invalid SFT token mask at step {step['step']}")
        trained += answer_size
    _require(trained == result["resource_counters"]["trained_tokens"], "SFT trained-token counter mismatch")


def _calibration(path, plan, splits, resolve=Path):
    calibration = _json(path)
    candidates, seen, checked = [], set(), {}
    for row in calibration["candidates"]:
        checkpoint = _checkpoint(resolve(row["checkpoint_path"]))
        _require(checkpoint["checkpoint_path"] not in seen and row["step"] > 0, "Duplicate or untrained calibration candidate")
        seen.add(checkpoint["checkpoint_path"])
        validation = resolve(row["validation_summary"])
        summary, created = _panel(validation, splits["validation"], "validation", plan["seed"], resolve=resolve)
        path = resolve(row["training_summary"])
        _require(row["training_summary_sha256"] == file_hash(path)
                 and row["validation_summary_sha256"] == file_hash(validation)
                 and _rebased_checkpoint(row["checkpoint"], resolve) == checkpoint_evidence(resolve(row["checkpoint_path"])),
                 "Calibration artifact changed after selection")
        result = _json(path)
        _require(resolve(result["parent_checkpoint"]).resolve() == resolve(plan["parent"]["path"]).resolve(),
                 "Calibration did not start from the declared parent")
        step = row["step"]
        _require(result["algorithm"] == "completion_only_lora_sft" and result["seed"] == plan["seed"],
                 "Wrong optimizer or seed during SFT calibration")
        if path not in checked:
            _sft(result, path, _check_steps(result), splits["train"], row, resolve)
            checked[path] = result
        _require(Path(checkpoint["checkpoint_path"]).parent == path.resolve().parent
                 and Path(checkpoint["checkpoint_path"]).name == f"step_{step:06d}", "Candidate not backed by SFT checkpoint")
        metrics = summary["metrics"]
        history = result["history"][step]["validation"]
        _require(_close(history["J_primary"], metrics["J_primary"])
                 and _close(history["any_invalid_output"], metrics["any_invalid_output"]),
                 "Calibration metrics disagree with recorded optimizer validation")
        j, invalid = metrics["J_primary"], metrics["any_invalid_output"]
        _require(_close(row["j"], j) and _close(row["invalid"], invalid), "Published calibration metrics mismatch")
        candidates.append({**row, "checkpoint_path": checkpoint["checkpoint_path"], "j": j,
                           "invalid": invalid, "created": created})
    target = plan["calibration_target"]
    tolerance = plan["calibration_tolerance"]
    format_limit = plan.get("calibration_max_invalid", .20)
    eligible = [row for row in candidates if abs(row["j"] - target) <= tolerance + 1e-12
                and row["invalid"] <= format_limit + 1e-12]
    primary = [row for row in candidates if row["attempt"] == 0]
    _require(bool(primary), "Primary calibration sweep is missing")
    expected = {0: plan["calibration"]["primary"]}
    if not any(row in eligible for row in primary):
        fallback = "fallback_if_no_eligible_and_any_at_least_half" if any(row["j"] >= target for row in primary) else "fallback_if_no_eligible_and_all_below_half"
        expected[1] = plan["calibration"][fallback]
    _require({row["attempt"] for row in candidates} == expected.keys(), "Calibration fallback differs from the preregistered rule")
    for attempt, protocol in expected.items():
        attempt_rows = [row for row in candidates if row["attempt"] == attempt]
        paths = {resolve(row["training_summary"]) for row in attempt_rows}
        _require(len(paths) == 1, "Multiple optimizers were merged into one calibration sweep")
        result = checked[paths.pop()]
        _require(result["plan"]["checkpoint_every"] == protocol["every"] and result["plan"]["max_updates"] == protocol["steps"]
                 and all(row["demonstration_rate"] == protocol["rate"] for row in attempt_rows)
                 and [row["step"] for row in attempt_rows] == [
                     entry["step"] for entry in result["history"] if entry["step"] and "validation" in entry
                 ], "Calibration configuration or candidates differ from the preregistered sweep")
    selected = min(eligible or candidates, key=lambda row: (
        round(abs(row["j"] - target), 12), row["invalid"], row["step"], row["attempt"]
    ))
    _require(selected["checkpoint_path"] == str(resolve(calibration["selected_checkpoint_path"]).resolve()),
             "Published weak checkpoint violates calibration selection rule")
    target_met = abs(selected["j"] - target) <= tolerance + 1e-12
    format_guard_met = selected["invalid"] <= format_limit + 1e-12
    _require(calibration["target_met"] is target_met and calibration["format_guard_met"] is format_guard_met,
             "Published calibration attainment flags are wrong")
    latest = max(row["created"] for row in candidates)
    return {key: value for key, value in selected.items() if key != "created"} | {
        "candidate_count": len(candidates), "target_met": target_met, "format_guard_met": format_guard_met,
    }, latest


def _rl(result, path, training_bank, steps, plan, resolve=Path):
    path = Path(path)
    counters, config = result["resource_counters"], result["plan"]
    groups = _rows(path.with_name("group_diagnostics.jsonl"))
    traces = _rows(path.with_name("rollouts.jsonl"))
    group_size = config["rollouts_per_root"]
    batch = config["roots_per_batch"]
    _require(result["horizon"] == "long" and result["reference_before"] == result["reference_after"],
             "RL horizon or frozen reference mismatch")
    _require(len(groups) == len(steps) * batch and len(traces) == counters["episodes"] == len(groups) * group_size,
             "Incomplete RL rollouts")
    train = {_identity(row): row for row in training_bank}
    root_order = plan["root_order"]
    temperature = plan["temperature"]
    _require(len(groups) <= len(root_order) and [group["root_id"] for group in groups] == root_order[:len(groups)],
             "RL optimizer followed the wrong root order")
    memory_paths, seeds = set(), set()
    scorer, trained = Evaluator(), 0
    example = None
    for index, group in enumerate(groups):
        returns, tasks, advantages = [], [], []
        update = index // batch + 1
        _require(group["optimizer_update"] == update, "RL group is misaligned with optimizer step")
        for record in traces[index * group_size:(index + 1) * group_size]:
            rollout = record["rollout"]
            memory_path = resolve(rollout["memory_path"])
            seed = rollout["rollout_seed"]
            _require(record["update"] == update and record["group_id"] == group["group_id"]
                     and seed not in seeds and memory_path not in memory_paths
                     and memory_path.is_file(), "Duplicated or missing RL rollout state")
            seeds.add(seed)
            memory_paths.add(memory_path)
            raw = _json(memory_path.with_suffix(".json"))
            trace = raw["trace"]
            episode = train.get(_identity(raw))
            _require(episode is not None and episode == trace["episode"] and raw["seed"] == seed
                     and rollout["split"] == "train" and raw["root_id"] == rollout["root_id"] == group["root_id"]
                     and raw["latent_world_id"] == rollout["latent_world_id"] == group["latent_world_id"],
                     "RL rollout is outside the frozen training bank or in the wrong group")
            current = scorer.score_current(trace)["utility"]
            future = scorer.score_future(trace)["utility"]
            unsupported = scorer.score_memory(trace)["unsupported_operations"]
            _require((rollout["r2"], rollout["r3"], rollout["unsupported_operations"],
                      rollout["questions"], rollout["invalid_outputs"])
                     == (current, future, unsupported, raw["metrics"]["questions"], raw["metrics"]["invalid_outputs"]),
                     "RL rollout utility or penalties disagree with environment trace")
            task = current + future
            total = task - .1 * rollout["questions"] - .5 * unsupported - .2 * rollout["invalid_outputs"]
            _require(_close(record["return"], total), "Wrong RL return")
            tasks.append(task)
            returns.append(total)
            advantages.append(record["advantage"])
            _require([phase["phase"].lower() for phase in rollout["phases"]] == ["select", "respond", "write"],
                     "Missing RL actor phase")
            for phase, mask, raw_phase in zip(rollout["phases"], record["actor_masks"], trace["phase_records"], strict=True):
                generation = phase["generation"]
                answer_size = len(generation["completion_ids"])
                _require(generation == raw_phase["generation"] and answer_size > 0
                         and generation["metadata"]["sampled"] is True and generation["metadata"]["temperature"] == temperature
                         and len(phase["old_logprobs"]) == answer_size and all(map(isfinite, phase["old_logprobs"]))
                         and mask == [0] * len(generation["prompt_ids"]) + [1] * answer_size, "Invalid RL actor token mask")
                trained += answer_size
        task_std, reward_std = pstdev(tasks), pstdev(returns)
        mean = fmean(returns)
        expected_adv = [(value - mean) / (reward_std + 1e-6) if reward_std else 0. for value in returns]
        _require(group["task_returns"] == tasks and _close(group["std_task"], task_std)
                 and _close(group["std_total"], reward_std)
                 and all(_close(actual, expected) for actual, expected in zip(group["returns"], returns, strict=True))
                 and all(_close(actual, expected) for actual, expected in zip(advantages, expected_adv, strict=True)),
                 "Wrong RL group reward or normalized advantages")
        if example is None and task_std > 0:
            example = {"root_id": group["root_id"], "group_id": group["group_id"], "returns": returns,
                       "task_returns": tasks, "advantages": expected_adv}
    _require(trained == counters["trained_tokens"] == counters["generated_tokens"], "RL trained-token counter mismatch")
    return {"trajectories": len(traces), "groups": len(groups),
            "groups_with_task_signal": sum(g["std_task"] > 0 for g in groups),
            "groups_with_any_signal": sum(g["std_total"] > 0 for g in groups),
            "groups_with_cost_only_signal": sum(g["std_task"] == 0 and g["std_total"] > 0 for g in groups),
            "groups_without_return_signal": sum(g["std_total"] == 0 for g in groups),
            "example_task_signal": example}


def _continuation(method, path, plan, chosen, selected, terminal, splits, selected_at, resolve=Path):
    result = _json(path)
    parent = Path(chosen["checkpoint_path"]).resolve()
    _require(result["seed"] == plan["seed"] and resolve(result["parent_checkpoint"]).resolve() == parent,
             f"{method} did not start from the calibrated weak checkpoint")
    steps = _check_steps(result)
    history = result["history"]
    initial = history[0]["validation"]
    final = history[-1]["validation"]
    _require(result["plan"]["initial_validation_reused"] and result["resource_counters"]["validation_reused"] == 1
             and _close(initial["J_primary"], chosen["j"]) and _close(initial["any_invalid_output"], chosen["invalid"]),
             f"{method} did not reuse the calibrated validation panel")
    _require(_close(initial["utility"], initial["J_primary"]) and _close(final["utility"], final["J_primary"]),
             f"{method} did not select by J")
    panel_path = Path(path).parent.parent / "validation/evaluation_00/summary.json"
    panel, panel_created = _panel(panel_path, splits["validation"], "validation", plan["seed"], method=method, resolve=resolve)
    _require(panel_created <= selected_at and _close(panel["metrics"]["J_primary"], final["J_primary"])
             and _close(panel["metrics"]["any_invalid_output"], final["any_invalid_output"]),
             f"{method} terminal validation is not bound to the locked selection")
    tolerance = result["plan"]["closure_selection"]["utility_tolerance"]
    expected_step = len(steps) if final["J_primary"] > initial["J_primary"] + tolerance else 0
    _require(result["best_step"] == selected["step"] == expected_step and terminal["step"] == len(steps)
             and resolve(selected["checkpoint_path"]).resolve() == resolve(result["selected_checkpoint"]).resolve()
             and resolve(terminal["checkpoint_path"]).resolve() == resolve(result["final_checkpoint"]).resolve(),
             f"{method} selection and terminal records do not match the optimizer")
    start_evidence = _checkpoint(Path(path).parent / "step_000000")
    parent_evidence = _checkpoint(parent)
    _require(start_evidence["base"] == parent_evidence["base"]
             and start_evidence["adapter_sha256"] == parent_evidence["adapter_sha256"],
             f"{method} step zero does not contain the parent weights")
    terminal_changed = terminal["adapter_sha256"] != parent_evidence["adapter_sha256"]
    _require(terminal_changed == (result["trainable_after"] != result["trainable_before"]),
             f"{method} checkpoint and optimizer disagree on weight changes")

    extra = {}
    if method == "B2-L":
        _require(result["algorithm"] == "episodic_clipped_grpo"
                 and resolve(result["kl_reference"]).resolve() == parent, "RL KL anchor is not the weak checkpoint")
        for key in ("max_updates", "roots_per_batch", "rollouts_per_root", "learning_rate",
                    "kl_coefficient", "clip_epsilon", "budget_seconds", "horizon"):
            _require(result["plan"][key] == plan["rl"][key], f"RL setting changed: {key}")
        extra = _rl(result, path, splits["train"], steps, plan["rl"], resolve)
    else:
        _require(result["algorithm"] == "completion_only_lora_sft" and result["plan"]["continuation"]
                 and result["plan"]["learning_rate"] == plan["more_sft"]["learning_rate"], "Wrong SFT control optimizer")
        _sft(result, path, steps, splits["train"], chosen, resolve)
    counters = result["resource_counters"]
    return {"updates": len(steps), "selected_step": expected_step, "initial_validation_j": initial["J_primary"],
            "terminal_validation_j": final["J_primary"], "terminal_adapter_changed": terminal_changed,
            "nonzero_gradient_steps": sum(row["grad_norm"] > 0 for row in steps),
            "budget_seconds": counters["allocated_seconds"], "training_seconds": counters["training_seconds"],
            "trained_tokens": counters["trained_tokens"], "stop_reason": result["stop_reason"],
            "training_summary_sha256": file_hash(Path(path)), **extra}


def _paired(left, right):
    _require(left["families"] == right["families"], "Paired comparison has different families")
    strata, signs = defaultdict(list), Counter()
    for root, family in left["families"].items():
        delta = left["by_root"][root]["J_primary"] - right["by_root"][root]["J_primary"]
        strata[family].append(delta)
        signs["improved" if delta > 0 else "worse" if delta < 0 else "tied"] += 1
    by_family = {family: fmean(diffs) for family, diffs in strata.items()}
    delta = fmean(by_family.values())
    _require(_close(delta, left["metrics"]["J_primary"] - right["metrics"]["J_primary"]),
             "Paired root differences do not reproduce the declared contrast")
    return {"delta_J_pp": 100 * delta, "delta_J_pp_by_family": {family: 100 * value for family, value in by_family.items()},
            **{key: signs[key] for key in ("improved", "tied", "worse")}}


def verify(run, *, archived=False):
    run = Path(run).resolve()
    comparison = _json(run / "COMPARISON.json")
    _require(comparison["status"] == "completed", "Comparison is incomplete")
    plan = _json(run / "PILOT_PLAN.json")
    resolve = Path
    if archived:
        manifest = read_manifest(run)
        resolve = _archived_paths(run, plan, manifest)
        _check_archived_stages(run, manifest, resolve)
    else:
        require_training_authorization(run)
    _require(plan["script_sha256"] == file_hash(resolve(plan["runner_path"])), "Runner changed after registration")
    bank_dir = resolve(plan["bank_dir"])
    _require(plan["bank_manifest_sha256"] == file_hash(bank_dir / "manifest.json"), "Bank manifest changed")
    splits = {role: _rows(bank_dir / f"{role}.jsonl") for role in ("train", "validation", "evaluation")}
    for role, rows in splits.items():
        _require({row["split"] for row in rows} == {role} and len({_identity(row) for row in rows}) == len(rows)
                 and plan["input_sha256"][role] == digest(rows), f"Wrong, changed, or duplicate {role} episodes")
    isolation = assert_semantic_split_isolation_v3(splits)
    original_parent = checkpoint_evidence(resolve(plan["parent"]["path"]))
    _require(original_parent == _rebased_checkpoint(plan["parent"], resolve), "Registered calibration parent changed")
    chosen, latest_calibration = _calibration(run / "CALIBRATION.json", plan, splits, resolve)

    selection_path = run / "SELECTION.json"
    selection = _json(selection_path)
    selected_at = datetime.fromisoformat(selection["created_at"])
    _require(selection["version"] == "competence_selection_v1" and selection["selection_complete"] is True
             and selection["selection_split"] == "validation" and selection["selected_before_evaluation"] is True
             and selected_at >= latest_calibration
             and plan["heldback_sha256"] == selection["heldback_episodes_sha256"] == digest(splits["evaluation"]),
             "Selection lock does not precede evaluation or binds the wrong bank")
    records = {row["method"]: row for row in selection["rows"]}
    _require(len(records) == len(selection["rows"]) and records.keys() == comparison["summaries"].keys() == METHODS,
             "Missing, duplicated or extra evaluation arm")
    _require(records["B1-weak"]["step"] == chosen["step"]
             and resolve(records["B1-weak"]["checkpoint_path"]).resolve() == Path(chosen["checkpoint_path"]),
             "Weak SFT evaluation did not use the calibrated checkpoint")
    parent_base = _checkpoint(chosen["checkpoint_path"])["base"]
    for method, row in records.items():
        evidence = _checkpoint(resolve(row["checkpoint_path"]))
        _require(row["training_seed"] == plan["seed"] and evidence["base"] == parent_base,
                 f"Wrong seed or base weights for {method}")
        for key in ("checkpoint_manifest_sha256", "adapter_sha256", "policy_fingerprint"):
            _require(row[key] == evidence[key], f"{method} checkpoint evidence changed: {key}")

    training = {
        method: _continuation(method, resolve(comparison["training"][method]), plan, chosen, records[method],
                              records[method + "-terminal"], splits, selected_at, resolve)
        for method in ("B3", "B2-L")
    }
    _require(_close(training["B3"]["budget_seconds"], training["B2-L"]["training_seconds"]),
             "SFT control was not allocated the observed GRPO optimization time")
    evaluated, panels = {}, {}
    for method, reference in comparison["summaries"].items():
        actual = reference.get("same_adapter_as", method)
        _require(records[actual]["policy_fingerprint"] == records[method]["policy_fingerprint"],
                 f"Improper evaluation reuse for {method}")
        if actual not in evaluated:
            panel, created_at = _panel(resolve(reference["summary"]), splits["evaluation"], "evaluation", plan["seed"],
                                       selection_path=selection_path, method=actual, resolve=resolve)
            _require(created_at >= selected_at, f"{method} evaluation predates locked selection")
            evaluated[actual] = panel
        panel = evaluated[actual]
        for key in ("metrics", "by_root", "families"):
            _require(panel[key] == reference[key], f"Incorrect comparison aggregate: {method} {key}")
        panels[method] = panel
    roots = len({row["root_id"] for row in splits["evaluation"]})
    episodes = len(splits["evaluation"])
    _require(comparison["evaluation_roots"] == roots and comparison["evaluation_episodes"] == episodes,
             "Wrong evaluation sample size")
    contrasts = [("B3", "B1-weak"), ("B2-L", "B1-weak"), ("B2-L", "B3"),
                 ("B3-terminal", "B1-weak"), ("B2-L-terminal", "B1-weak"), ("B2-L-terminal", "B3-terminal")]
    return {"status": "verified", "calibration": chosen, "split_isolation": isolation, "training": training,
            "scope": "archive_read_only" if archived else "current_workspace",
            "selected_steps": {method: row["step"] for method, row in records.items()},
            "evaluation_roots": roots, "evaluation_episodes": episodes, "distinct_panels": len(evaluated),
            "evaluation_metrics": {method: panel["metrics"] for method, panel in panels.items()},
            "evaluation_by_family": {method: panel["by_family"] for method, panel in panels.items()},
            "paired_contrasts": [{"method": method, "control": control, **_paired(panels[method], panels[control])}
                                 for method, control in contrasts],
            "comparison_sha256": file_hash(run / "COMPARISON.json"), "selection_sha256": file_hash(selection_path)}


def _print_summary(result):
    print(f"Reevaluación offline verificada · {result['evaluation_roots']} historias / "
          f"{result['evaluation_episodes']} episodios · {result['distinct_panels']} políticas con trazas propias")
    print(f"{'Método':<25} {'Paso':>5} {'C2':>7} {'FJS':>7} {'J':>7} {'Preg.':>6} {'Sin respaldo':>13}")
    names = {"B1-weak": "SFT calibrado", "B3": "Más SFT", "B2-L": "GRPO seleccionado",
             "B3-terminal": "SFT terminal", "B2-L-terminal": "GRPO terminal"}
    for method, name in names.items():
        panel = result["evaluation_metrics"][method]
        step = result["selected_steps"][method]
        print(f"{name:<25} {step:>5} {panel['C2']:>7.1%} {panel['FJS']:>7.1%} {panel['J_primary']:>7.1%} "
              f"{panel['questions']:>6.2f} {panel['unsupported_write_episode']:>13.1%}")
    print("C2 mide el acierto actual, FJS el posterior y J exige ambos en el mismo episodio.")
    print("Preg. es el promedio por episodio. Sin respaldo mide episodios con alguna escritura sin evidencia.")
    print("\nDiferencia en J y conteo de historias emparejadas")
    for pair in result["paired_contrasts"]:
        print(f"{names[pair['method']]} frente a {names[pair['control']]}: {pair['delta_J_pp']:+.1f} pp · "
              f"{pair['improved']} mejoran, {pair['tied']} empatan, {pair['worse']} empeoran")
    grpo = result["training"]["B2-L"]
    print(f"\nGRPO · {grpo['updates']} pasos, {grpo['trajectories']} trayectorias, "
          f"{grpo['nonzero_gradient_steps']} pasos con gradiente no nulo")
    print(f"{grpo['groups_with_task_signal']}/{grpo['groups']} grupos con variación de la utilidad de tarea, "
          f"{grpo['groups_with_cost_only_signal']} con señal solo de costes y "
          f"{grpo['groups_without_return_signal']} con retorno constante.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path, nargs="?", help="Optional JSON output at a fresh path")
    parser.add_argument("--archived", action="store_true",
                        help="Recheck saved stage evidence without applying the live authorization gate")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Use a fresh QA output path")
    result = verify(args.run, archived=args.archived)
    if args.output is not None:
        write_json(args.output, result)
    _print_summary(result)


if __name__ == "__main__":
    main()
