"""Train a third matched GRPO arm with feedback from a local language model."""

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from math import isclose
from pathlib import Path
from statistics import mean, pstdev
import time

from research import ai_reward_judge
from research.competence_data import load_and_check
from research.competence_training import PROMPTS, _release, checkpoint_evidence, verify_initialization
from research.joint_reward_experiment import evaluation_summary
from research.reward_review_experiment import components, task_score
from research.src.artifacts import canonical, digest, file_hash, now, write_json
from research.src.closure_protocol import execution_context
from research.src.policy.llm_controller import SpecLocalPolicy
from research.src.training.episodic_grpo import group_advantages, train_grpo
from research.src.training.rollouts import make_rollout_collector
from research.weak_sft_experiment import SEED, SELECTION, authorize, evaluate, read, selection_record, training_cohort, validation_callback
from research.weak_sft_qa import _rows

METHOD = "RL-AI"
TERMINAL = METHOD + "-terminal"
JUDGE = "Codex subagent /root/objective_review"
DESIGN = Path("reports/reward_review_20261104_001/AI_DESIGN.json")


def request_verdicts(run, name, samples, *, timeout=300):
    """Publish evidence only and wait for the designated AI reviewer's batch."""
    queue = run / "judge_queue"
    queue.mkdir(exist_ok=True)
    request = queue / f"{name}.request.json"
    response = queue / f"{name}.response.json"
    if request.exists() or response.exists():
        raise FileExistsError(name)
    payload = {"rubric": ai_reward_judge.RUBRIC, "samples": samples}
    temporary = request.with_suffix(".tmp")
    write_json(temporary, payload)
    temporary.rename(request)
    print("awaiting AI feedback", name, flush=True)
    deadline = time.monotonic() + timeout
    while not response.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"AI feedback not received for {name}")
        time.sleep(1)
    reply = read(response)
    if reply["judge"] != JUDGE or reply["request_sha256"] != file_hash(request):
        raise ValueError("AI response does not match its request")
    verdicts = reply["verdicts"]
    if set(verdicts) != {sample["id"] for sample in samples}:
        raise ValueError("Incomplete AI feedback batch")
    return {key: ai_reward_judge._parse_verdict(canonical(value)) for key, value in verdicts.items()}


def ai_task(parts, verdict):
    feedback = (
        .50 * bool(verdict["response_supported"] and parts["current"])
        + .25 * bool(verdict["memory_appropriate"] and parts["supported_write"])
        + .25 * bool(verdict["future_supported"] and parts["faithful_future"])
    )
    task = 2 * parts["joint"] + .50 * parts["benefit"] + feedback
    relief = .10 if parts["informative_question"] and verdict["clarification_appropriate"] else 0.
    return task, relief, feedback


def prepare(review, fresh, output):
    review_plan = read(review / "PILOT_PLAN.json")
    if (review / "SELECTION.json").exists():
        raise ValueError("Freeze the AI arm before opening the shared evaluation bank")
    _, data = load_and_check(fresh)
    sources = {str(p.resolve()): file_hash(p) for p in (
        Path(__file__), Path(ai_reward_judge.__file__),
        Path(__file__).with_name("reward_review_experiment.py"),
        Path(__file__).with_name("reward_memory_credit.py"), DESIGN,
    )}
    plan = {"created_at": now(), "seed": SEED, "rules_run": str(review), "fresh_bank": str(fresh),
            "previous": review_plan["prior"], "rules_plan_sha256": file_hash(review / "PILOT_PLAN.json"),
            "fresh_manifest_sha256": file_hash(fresh / "manifest.json"),
            "heldback_sha256": review_plan["fresh_evaluation_sha256"],
            "source_sha256": sources, "judge": JUDGE,
            "judge_version": "codex_evidence_rubric_v1", "rl": review_plan["rl"],
            "ai_time_cap_seconds": 2400, "design": read(DESIGN), "fresh_data_seed": data["seed"]}
    if review_plan["rl"]["roots_per_batch"] != 1:
        raise ValueError("AI feedback collector expects one root per group")
    output.mkdir(parents=True)
    authorize(output, fresh, plan)
    print("AI arm frozen before rules evaluation", output, flush=True)


def train(run, parent, cohort, validation, initial, old, seconds):
    actor = SpecLocalPolicy.from_checkpoint(parent, trainable=True, device="cuda")
    reference = None
    try:
        reference = SpecLocalPolicy.from_checkpoint(parent, trainable=False, device="cpu")
        actor.config.temperature = reference.config.temperature = old["temperature"]
        write_json(run / "INITIALIZATION.json", verify_initialization(actor, parent, reference))
        collect = make_rollout_collector(run / "environment_rollouts", method=METHOD, prompt_root=PROMPTS)
        with (run / "AI_FEEDBACK.jsonl").open("x") as stream, execution_context(run, "training"):
            pending = {}
            def rollout(policy, episode, seed):
                if not pending:
                    samples = []
                    for draw in range(old["rollouts_per_root"]):
                        sampled_seed = seed + draw
                        policy.seed(sampled_seed)
                        result = collect(policy, episode, sampled_seed)
                        row = read(Path(result.memory_path).with_suffix(".json"))
                        pending[sampled_seed] = (result, row)
                        samples.append({"id": str(sampled_seed), "evidence": ai_reward_judge.judge_input(row)})
                    started = time.monotonic()
                    verdicts = request_verdicts(run, f"group_{seed}", samples)
                    elapsed = time.monotonic() - started
                    for sample in samples:
                        sampled_seed = int(sample["id"])
                        result, row = pending[sampled_seed]
                        raw = verdicts[sample["id"]]
                        verdict = {**raw, "model": JUDGE, "elapsed_seconds": elapsed / len(samples),
                                   "input_sha256": digest(sample["evidence"]), "raw_output": canonical(raw)}
                        parts = components(row)
                        task, relief, feedback = ai_task(parts, verdict)
                        stream.write(canonical({"seed": sampled_seed, "root_id": episode["root_id"], "verdict": verdict,
                            "components": parts, "feedback_points": feedback, "task": task, "ask_cost_waiver": relief}) + "\n")
                        stream.flush()
                        pending[sampled_seed] = replace(result, r2=task, r3=relief)
                return pending.pop(seed)
            return train_grpo(actor, reference, cohort, rollout,
                run / "optimizer", validation_callback(validation, run / "validation", METHOD),
                horizon="long", updates=old["max_updates"], seed=SEED, roots_per_batch=old["roots_per_batch"],
                rollouts_per_root=old["rollouts_per_root"], learning_rate=old["learning_rate"],
                kl_coefficient=old["kl_coefficient"], clip_epsilon=old["clip_epsilon"],
                checkpoint_every=old["max_updates"], budget_seconds=seconds, initial_validation_metrics=initial,
                closure_authorization=run, checkpoint_save_budget_seconds=120, closure_selection=SELECTION,
                reference_execution=old["reference_execution"])
    finally:
        _release(reference)
        _release(actor)


def audit(run, review, result):
    feedback = _rows(run / "AI_FEEDBACK.jsonl")
    traces = _rows(run / "optimizer/rollouts.jsonl")
    control = _rows(review / "continuations/RL-revised/optimizer/rollouts.jsonl")
    settings = result["plan"]
    count = settings["max_updates"] * settings["roots_per_batch"] * settings["rollouts_per_root"]
    if len(traces) != count or result["resource_counters"]["updates"] != settings["max_updates"]:
        raise ValueError("Incomplete matched GRPO update budget")
    if result["frozen_before"] != result["frozen_after"] or result["reference_before"] != result["reference_after"]:
        raise ValueError("AI feedback changed frozen weights")
    groups, counts = defaultdict(list), Counter()
    judge_seconds = 0.
    for label, row, old in zip(feedback, traces, control, strict=True):
        rollout, old_rollout = row["rollout"], old["rollout"]
        keys = ("root_id", "latent_world_id", "rollout_seed", "split")
        if any(rollout[key] != old_rollout[key] for key in keys) or label["seed"] != rollout["rollout_seed"]:
            raise ValueError("Mismatched judge feedback or training schedule")
        if row["update"] == 1 and [p["generation"]["completion_ids"] for p in rollout["phases"]] != [
            p["generation"]["completion_ids"] for p in old_rollout["phases"]
        ]:
            raise ValueError("Initial trajectories differed before the reward update")
        evidence = read(Path(rollout["memory_path"]).with_suffix(".json"))
        verdict = label["verdict"]
        if digest(ai_reward_judge.judge_input(evidence)) != verdict["input_sha256"]:
            raise ValueError("Judge evaluated different evidence")
        if any(verdict[key] != value for key, value in json.loads(verdict["raw_output"]).items()):
            raise ValueError("Saved verdict differs from model output")
        parts = components(evidence)
        task, relief, points = ai_task(parts, verdict)
        metrics = evidence["metrics"]
        costs = .10 * metrics["questions"] + .50 * metrics["unsupported_operations"] + .20 * metrics["invalid_outputs"]
        total = task + relief - costs
        rule_total = task_score(parts, True) + .10 * parts["informative_question"] - costs
        if parts != label["components"] or not all(isclose(left, right, abs_tol=1e-9) for left, right in (
            (task, label["task"]), (relief, label["ask_cost_waiver"]), (points, label["feedback_points"]),
            (total, row["return"]),
        )):
            raise ValueError("Judge feedback did not reach optimizer as registered")
        groups[row["group_id"]].append((total, rule_total, row["advantage"]))
        counts["joint_successes"] += parts["joint"]
        counts["helpful_writes"] += parts["benefit"] > 0
        counts["harmful_writes"] += parts["benefit"] < 0
        counts["changed_memory"] += evidence["trace"]["initial_memory_hash"] != evidence["trace"]["final_memory_hash"]
        counts["unsupported_writes"] += metrics["unsupported_operations"] > 0
        counts["informative_questions"] += parts["informative_question"]
        counts["scored_feedback"] += points > 0
        counts["returns_changed_by_AI"] += not isclose(total, rule_total, abs_tol=1e-9)
        judge_seconds += verdict["elapsed_seconds"]
    for group in groups.values():
        new, old, stored = zip(*group)
        advantages = group_advantages(new)
        if any(not isclose(a, b, abs_tol=1e-9) for a, b in zip(advantages, stored, strict=True)):
            raise ValueError("Wrong optimizer advantages")
        counts["groups_with_total_signal"] += pstdev(new) > 0
        counts["groups_with_changed_advantages"] += any(
            not isclose(a, b, abs_tol=1e-6) for a, b in zip(advantages, group_advantages(old), strict=True))
    return {**counts, "updates": result["resource_counters"]["updates"], "judge_calls": len(feedback),
            "judge_elapsed_seconds": judge_seconds, "adapter_changed": result["trainable_before"] != result["trainable_after"],
            "validation_J": {str(e["step"]): e["validation"]["J_primary"] for e in result["history"] if "validation" in e},
            "selected_step": result["best_step"], "matched_initial_completions": True}


def execute(run, calibration):
    plan = read(run / "PILOT_PLAN.json")
    review = Path(plan["rules_run"])
    if any(file_hash(path) != sha for path, sha in plan["source_sha256"].items()):
        raise ValueError("AI experiment source changed since freezing")
    if file_hash(review / "PILOT_PLAN.json") != plan["rules_plan_sha256"]:
        raise ValueError("Rules plan changed since freezing")
    check = read(calibration)
    if check["status"] != "passed" or check["model"] != plan["judge"]:
        raise ValueError("Judge calibration is required")
    write_json(run / "JUDGE_CALIBRATION.json", {"path": str(calibration.resolve()), "sha256": file_hash(calibration), **check})
    controls = read(review / "COMPARISON.json")
    start = read(review / "STARTING_POLICY.json")
    prior = Path(plan["previous"])
    prior_plan = read(prior / "PILOT_PLAN.json")
    old_data, _ = load_and_check(Path(prior_plan["bank_dir"]))
    cohort = training_cohort(old_data["train"])
    if digest(cohort) != read(review / "PILOT_PLAN.json")["train_cohort_sha256"]:
        raise ValueError("AI arm cohort differs from rule arms")
    old = plan["rl"]
    prior_training = read(review / "continuations/RL-revised/optimizer/training.json")
    parent = start["checkpoint"]
    if prior_training["parent_checkpoint"] != parent:
        raise ValueError("AI and rule arms must start from the same checkpoint")
    write_json(run / "SHARED_PARENT.json", checkpoint_evidence(parent))
    initial = prior_training["history"][0]["validation"]
    result = train(run, parent, cohort, old_data["validation"], initial, old, plan["ai_time_cap_seconds"])
    verified = audit(run, review, result)
    write_json(run / "QA_TRAINING.json", verified)
    print("AI training verified", verified, flush=True)
    data, _ = load_and_check(Path(plan["fresh_bank"]))
    bank = data["evaluation"]
    if digest(bank) != plan["heldback_sha256"]:
        raise ValueError("Shared evaluation bank changed")
    selection = run / "SELECTION.json"
    final_step = result["resource_counters"]["updates"]
    write_json(selection, {"version": "competence_selection_v1", "created_at": now(),
        "selection_complete": True, "selection_split": "validation", "selected_before_evaluation": True,
        "heldback_episodes_sha256": digest(bank), "rows": [
            selection_record(TERMINAL, result["final_checkpoint"], final_step)]})
    report = evaluate(run, bank, selection)
    terminal = evaluation_summary(Path(report[TERMINAL]["summary"]), bank, selection, TERMINAL, SEED)
    panels = {**controls["panels"], TERMINAL: terminal}
    picked = TERMINAL if result["best_step"] == final_step else "RL-start"
    panels[METHOD] = {**panels[picked], "same_policy_as": picked}
    paired = {}
    for method in ("RL-start", "RL-joint-terminal", "RL-revised-terminal", "B3", "B1-weak"):
        diffs = [score - panels[method]["by_root"][root] for root, score in terminal["by_root"].items()]
        paired[method] = {"delta_J_pp": 100 * mean(diffs), "improved": sum(d > 0 for d in diffs),
                          "worse": sum(d < 0 for d in diffs), "tied": sum(d == 0 for d in diffs)}
    write_json(run / "COMPARISON.json", {"status": "verified", "judge": plan["judge"], "training": verified,
        "panels": panels, "paired_AI_terminal_vs": paired, "evaluation_roots": len(terminal["by_root"]),
        "evaluation_episodes": len(bank)})
    print("AI comparison verified", {name: row["metrics"]["J_primary"] for name, row in panels.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("prepare")
    for name in ("review", "fresh", "output"):
        pre.add_argument("--" + name, type=Path, required=True)
    exe = sub.add_parser("execute")
    exe.add_argument("--output", type=Path, required=True)
    exe.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.review.resolve(), args.fresh.resolve(), args.output.resolve())
    else:
        execute(args.output.resolve(), args.calibration.resolve())


if __name__ == "__main__":
    main()
