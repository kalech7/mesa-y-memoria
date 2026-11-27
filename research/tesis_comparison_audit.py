"""Verify a completed selector comparison from saved trajectories and artifacts."""
from __future__ import annotations

import argparse
import json
from math import isclose
from pathlib import Path

from research.competence_evaluation import _verify_sqlite, aggregate_rows
from research.grpo_select_pilot import choose, policy_return, scores
from research.legal_selection_policy import ACTION_CATALOG
from research.src.artifacts import canonical, digest, file_hash, write_json
from research.src.readers.symbolic import read
from research.src.scoring.evidence import Evaluator
from research.src.scoring.reward import rewards
from research.src.training.episodic_grpo import group_advantages
from research.tesis_comparison import METHODS, checkpoint_record, episode_id, summarize


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_checkpoint(record):
    actual = checkpoint_record(record["method"], record["checkpoint_path"],
                               record["step"], record["training_seed"])
    require(actual == record, "Checkpoint differs from its recorded artifacts")


def verify_trajectory(row):
    trace = row["trace"]
    phases = row["phases"]
    require(set(phases) == {"select", "respond", "write"}, "Unexpected trajectory phases")
    require(all(digest(trace["observations"][name]) == phase["observation_sha256"]
                for name, phase in phases.items()), "Phase observation digest changed")
    require(trace["parsed_actions"] == {name: phase["action"] for name, phase in phases.items()}
            and trace["response"] == phases["respond"]["action"], "Trajectory action does not match phases")
    errors = [phases[name]["error"] for name in ("select", "respond", "write") if phases[name]["error"]]
    require(trace["errors"] == errors, "Trajectory errors do not match phases")
    selected = phases["select"]
    action = selected["action"]
    asked = action["slot"] if selected["error"] is None and action["action"] == "ASK" else None
    require(trace.get("asked_slot") == asked, "Delivered question does not match SELECT")
    persistence = row["persistence_verification"]
    require(_verify_sqlite(persistence["sqlite_path"], trace) == persistence,
            "Persisted memory differs from its recorded artifacts")
    reuse = trace["reuse_observation"]
    require(trace["future_response"] == read(reuse["memory"], reuse["query"]),
            "Symbolic future response does not reproduce")
    evaluator = Evaluator()
    expected = {"current": evaluator.score_current(trace), "memory": evaluator.score_memory(trace),
                "future": evaluator.score_future(trace)}
    require(row["scoring"] == expected, "Raw scoring does not reproduce")
    current, memory, future = expected["current"], expected["memory"], expected["future"]
    questions, invalid, unsupported = int(asked is not None), len(errors), memory["unsupported_operations"]
    reward = rewards(current["utility"], future["utility"], questions, unsupported, invalid)
    require(row["rewards_by_component"] == reward, "Archived reward components do not reproduce")
    derived = {"C2": current["success"], "FJS": future["success"], "questions": questions,
               "unsupported_write_count": unsupported, "unsupported_write_episode": int(unsupported > 0),
               "invalid_output_count": invalid, "any_invalid_output": int(invalid > 0),
               "R_long": reward["R_long"], "R_short": reward["R_short"]}
    for key, value in derived.items():
        require(row["metrics"][key] == value, f"Archived metric {key} differs from the original evidence")


def verify_panel(trajectories, expected_episode_hash, seed, role, method):
    directory = Path(trajectories).parent
    completion = load(directory / "completion.json")
    predeclaration = load(directory / "predeclaration.json")
    hashes = completion["artifacts_sha256"]
    require(completion["status"] == "completed" and completion["role"] == role
            and predeclaration["method"] == method and predeclaration["role"] == role
            and predeclaration["training_seed"] == seed
            and predeclaration["bank_content_sha256"] == expected_episode_hash,
            "Panel completion does not match the predeclared model, seed or bank")
    for artifact in ("predeclaration.json", "trajectories.jsonl", "summary.json"):
        require(file_hash(directory / artifact) == hashes[artifact], f"Completed panel artifact changed: {artifact}")
    rows = read_rows(trajectories)
    require(bool(rows), "Empty trajectory panel")
    episodes = []
    for row in rows:
        require(row["method"] == method and row["training_seed"] == seed and row["evaluation_role"] == role
                and row["split"] == role and row["technical_status"] == "completed"
                and (row["route"], row["condition"], row["reader_id"]) == ("natural", "O2", "fixed_symbolic"),
                "Unexpected method, seed, split, reader or condition in trajectory")
        episode = row["trace"]["episode"]
        require(episode_id(row) == episode_id(episode), "Trajectory episode identity changed")
        verify_trajectory(row)
        episodes.append(episode)
        row["metrics"] = scores(row)
    require(digest(episodes) == expected_episode_hash, "Trajectory bank content changed")
    return aggregate_rows(rows)


def verify_group(row, expected, training_episode, seed, method, passes, draws):
    require(row["episode"] == expected["episode"] and row["prompt_sha256"] == expected["prompt_sha256"],
            "Training episode or prompt differs from the frozen common order")
    require(len(row["optimization"]) == passes, "Optimizer step count mismatch")
    if method == METHODS[1]:
        require(row["target_action"] == json.loads(ACTION_CATALOG[expected["teacher_action"]]),
                "Supervised action differs from the frozen observable teacher")
        return False
    require(len(row["draws"]) == draws, "GRPO draw count mismatch")
    rewards, tasks = [], []
    for draw in row["draws"]:
        branch = load(draw["trajectory"])
        verify_trajectory(branch)
        require(branch["trace"]["episode"] == training_episode
                and list(episode_id(branch)) == expected["episode"] and branch["training_seed"] == seed
                and branch["method"] == "GRPO-SELECT-TRAIN" and branch["split"] == "train"
                and (branch["route"], branch["condition"], branch["reader_id"]) == ("natural", "O2", "fixed_symbolic"),
                "GRPO branch differs from the training episode or condition")
        select = branch["phases"]["select"]
        require(select["observation_sha256"] == expected["observation_sha256"]
                and select["action"] == draw["action"], "GRPO branch action or prefix changed")
        metrics = scores(branch)
        task, reward = policy_return(metrics)
        require(draw["metrics"] == metrics and isclose(draw["task_return"], task)
                and isclose(draw["return"], reward), "GRPO branch reward does not reproduce")
        rewards.append(reward)
        tasks.append(task)
    advantages = group_advantages(rewards)
    require(len(row["advantages"]) == len(advantages)
            and all(isclose(old, new, rel_tol=1e-6, abs_tol=1e-6)
                    for old, new in zip(row["advantages"], advantages)),
            "GRPO advantages do not reproduce")
    signal = len(set(tasks)) > 1
    require(row["semantic_signal"] == signal, "Task signal does not reproduce")
    return signal


def verify_training(run, plan, selection, results, train):
    report = {}
    allocated = plan["continuation"]["allocation_seconds_per_arm_per_seed"]
    passes = plan["continuation"]["passes_per_group"]
    draws = plan["continuation"]["draws_per_grpo_group"]
    for seed in plan["seeds"]:
        key = str(seed)
        parent = plan["clean_parents"][key]
        verify_checkpoint(parent)
        validation_hash = plan["split_hashes"]["validation"]
        baseline_path = run / "validation" / key / "SFT/trajectories.jsonl"
        baseline = verify_panel(baseline_path, validation_hash, seed, "validation", METHODS[0])
        arm_reports = {}
        initial_hashes = set()
        for method in METHODS[1:]:
            arm = run / "training" / key / method
            summary = load(arm / "TRAINING.json")
            require(summary == results["training"][key][method], "Training summary differs from results")
            initial_hashes.add(summary["initial_policy_sha256"])
            require(summary["allocated_seconds"] == allocated and summary["training_seconds"] >= allocated
                    and isclose(summary["overrun_seconds"], summary["training_seconds"] - allocated)
                    and summary["initial_policy_sha256"] == summary["frozen_policy_sha256"],
                    "Budget or common initialization invariant failed")
            groups = read_rows(arm / "training.jsonl")
            require(len(groups) == summary["groups"] and len(groups) * passes == summary["optimizer_steps"],
                    "Training update count differs from trace")
            expected = plan["train_prefixes"][key]
            task_signal = 0
            for index, row in enumerate(groups, 1):
                require(row["group"] == index, "Training log group order changed")
                case = expected[(index - 1) % len(expected)]
                task_signal += verify_group(row, case, train[tuple(case["episode"])], seed, method, passes, draws)
            require(summary["groups_with_task_signal"] == task_signal, "GRPO task signal count changed")
            validations = {0: baseline}
            for step in sorted(int(value) for value in summary["validation"] if int(value)):
                path = arm / "validation" / f"step_{step:06d}" / "trajectories.jsonl"
                validations[step] = verify_panel(path, validation_hash, seed, "validation", method)
            require({str(step): panel["metrics"] for step, panel in validations.items()} == summary["validation"],
                    "Archived validation metrics do not reproduce")
            row = selection[(seed, method)]
            require(choose(baseline, validations) == row["step"] and summary["selection"] == row,
                    "Validation-only selection does not reproduce")
            arm_reports[method] = {"groups": len(groups), "optimizer_steps": summary["optimizer_steps"],
                                   "task_signal_groups": task_signal, "selected_step": row["step"],
                                   "allocated_seconds": allocated, "training_seconds": summary["training_seconds"],
                                   "overrun_seconds": summary["overrun_seconds"],
                                   "validation_and_save_seconds": summary["validation_and_save_seconds"],
                                   "adapter_changed": summary["adapter_changed"]}
        require(len(initial_hashes) == 1, "Continuation initialization differs within a seed")
        report[key] = arm_reports
    return report


def verify_bank(plan):
    bank = Path(plan["bank"])
    manifest_path = bank / "manifest.json"
    require(file_hash(manifest_path) == plan["bank_manifest_sha256"], "Frozen bank manifest changed")
    manifest = load(manifest_path)
    partitions = {}
    for role, entry in manifest["partitions"].items():
        path = bank / entry["path"]
        require(file_hash(path) == entry["sha256"], f"Frozen {role} partition changed")
        rows = read_rows(path)
        require(digest(rows) == plan["split_hashes"][role]
                and len({row["root_id"] for row in rows}) == plan["split_roots"][role],
                f"Frozen {role} partition does not match the comparison plan")
        partitions[role] = rows
    return {episode_id(row): row for row in partitions["train"]}


def audit(run):
    plan_path = run / "PLAN.json"
    selection_path = run / "SELECTION.json"
    plan = load(plan_path)
    manifest = load(selection_path)
    results = load(run / "RESULTS.json")
    require(results["status"] == "completed_exploratory", "No completed exploratory comparison")
    require(manifest["plan_sha256"] == file_hash(plan_path)
            and results["selection_sha256"] == file_hash(selection_path),
            "Selection or plan digest mismatch")
    require(file_hash(run / "TOKEN_PREFLIGHT.json") == plan["token_preflight_sha256"],
            "Token preflight changed")
    for relative, expected in plan["source_sha256"].items():
        require(file_hash(run / "source_snapshot" / relative) == expected, f"Source snapshot changed: {relative}")
    evaluation_hash = plan["split_hashes"]["evaluation"]
    require(manifest["version"] == "competence_selection_v1" and manifest["selection_complete"] is True
            and manifest["selection_split"] == "validation" and manifest["selected_before_evaluation"] is True
            and manifest["heldback_episodes_sha256"] == evaluation_hash,
            "Missing validation-only selection evidence")
    selection = {}
    for row in manifest["rows"]:
        verify_checkpoint(row)
        key = row["training_seed"], row["method"]
        require(key not in selection, "Duplicate selected policy")
        selection[key] = row
    require(set(selection) == {(seed, method) for seed in plan["seeds"] for method in METHODS},
            "Incomplete comparison selection")
    require(all(selection[(seed, METHODS[0])] == plan["clean_parents"][str(seed)] for seed in plan["seeds"]),
            "Selected SFT reference differs from its frozen clean parent")
    training = verify_training(run, plan, selection, results, verify_bank(plan))
    panels = {}
    for seed in plan["seeds"]:
        key = str(seed)
        panels[seed] = {}
        methods = results["panels"][key]
        require(set(methods) == set(METHODS), "Incomplete reserved comparison")
        for method, report in methods.items():
            traces = Path(report["trajectories"])
            require(file_hash(traces) == report["trajectories_sha256"], "Reserved trajectories changed")
            alias = report.get("same_policy_as")
            if alias is not None:
                require(selection[(seed, alias)]["policy_fingerprint"]
                        == selection[(seed, method)]["policy_fingerprint"]
                        and traces == Path(methods[alias]["trajectories"]), "Invalid step-zero alias")
            predeclared = load(traces.parent / "predeclaration.json")
            auth = predeclared["protocol"]["heldback_authorization"]
            require(auth["selection_manifest_sha256"] == results["selection_sha256"]
                    and auth["locked_bank_sha256"] == evaluation_hash
                    and predeclared["bank_content_sha256"] == evaluation_hash
                    and predeclared["training_seed"] == seed
                    and predeclared["method"] == (alias or method), "Evaluation was not authorized by frozen selection")
            panel = verify_panel(traces, evaluation_hash, seed, "evaluation", alias or method)
            require(all(report[key] == value for key, value in panel.items()),
                    "Archived reserved metrics do not reproduce")
            panels[seed][method] = panel
    summary = summarize(panels, resamples=plan["statistics"]["resamples"])
    require(all(results[key] == value for key, value in summary.items()),
            "Archived paired contrasts do not reproduce")
    return {"status": "verified", "seeds": plan["seeds"], "selection_sha256": results["selection_sha256"],
            "reserved_roots": plan["split_roots"]["evaluation"], "training": training,
            "means": summary["means"], "comparison_status": "exploratory"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run.resolve())
    if args.output:
        if args.output.exists():
            parser.error("Use a fresh output path")
        write_json(args.output, result)
    print(canonical(result))


if __name__ == "__main__":
    main()
