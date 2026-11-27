"""Independent CPU checks of TRAIN likelihood traces; never opens held-out data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def _read(path):
    return json.loads(Path(path).read_text())


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(values):
    """Recompute population-normalized advantages without importing trainer."""
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite or empty reward group")
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return [0.0 if sd == 0 else (value - mean) / (sd + 1e-6) for value in values]


def review_train(optimizer):
    optimizer = Path(optimizer).resolve()
    training = _read(optimizer / "training.json")
    rows = _rows(optimizer / "rollouts.jsonl")
    groups, token_counts, families = defaultdict(list), Counter(), Counter()
    for row in rows:
        rollout = row["rollout"]
        if "return_value" in row:
            row["return"] = row["return_value"]
        expected_return = (rollout["r2"] + rollout["r3"] - .1 * rollout["questions"]
                           - .5 * rollout["unsupported_operations"] - .2 * rollout["invalid_outputs"])
        if not math.isclose(row["return"], expected_return, abs_tol=1e-9):
            raise ValueError("Rollout return transport mismatch")
        if rollout.get("phase_rewards") and not math.isclose(
                row["return"], statistics.fmean(rollout["phase_rewards"].values()), abs_tol=1e-9):
            raise ValueError("Phase reward mean and episode reward differ")
        if rollout["split"] != "train":
            raise ValueError("This audit accepts TRAIN traces only")
        evidence_path = Path(rollout["memory_path"]).with_suffix(".json")
        evidence = _read(evidence_path)
        if evidence["split"] != "train" or evidence["root_id"] != rollout["root_id"]:
            raise ValueError("Trace/evidence identity mismatch")
        if [phase["phase"].lower() for phase in rollout["phases"]] != ["select", "respond", "write"]:
            raise ValueError("Unexpected actor phases")
        for index, phase in enumerate(rollout["phases"]):
            generation = phase["generation"]
            prompt, completion = generation["prompt_ids"], generation["completion_ids"]
            if not completion or len(phase["old_logprobs"]) != len(completion):
                raise ValueError("Completion likelihood length mismatch")
            if not all(math.isfinite(value) for value in phase["old_logprobs"]):
                raise ValueError("Non-finite old logprobs")
            if "actor_masks" in row and row["actor_masks"][index] != [0] * len(prompt) + [1] * len(completion):
                raise ValueError("Prompt/completion mask mismatch")
            token_counts[phase["phase"]] += len(completion)
        m, write = evidence["metrics"], evidence["trace"].get("write_result")
        write_valid = isinstance(write, dict) and write["valid"]
        strict = bool(m["C2"] == 1 and m["FJS"] == 1 and write_valid
                      and m["unsupported_operations"] == 0)
        groups[row["group_id"]].append({"row": row, "family": evidence["scenario_family"],
            "current": m["C2"], "future": m["FJS"], "strict": strict,
            "unsafe": bool(not write_valid or m["unsupported_operations"]),
            "evidence_sha256": _hash(evidence_path)})
    result, counts = [], Counter()
    for group_id, members in groups.items():
        if len(members) != 4 or len({(m["row"]["rollout"]["root_id"], m["row"]["rollout"]["latent_world_id"]) for m in members}) != 1:
            raise ValueError("Expected four isolated samples of one root/world")
        families[members[0]["family"]] += 1
        returns = [m["row"]["return"] for m in members]
        expected = normalized(returns)
        # New phase-credit objectives can retain an episodic diagnostic advantage.
        for member, advantage in zip(members, expected, strict=True):
            if "advantage" in member["row"] and not math.isclose(member["row"]["advantage"], advantage, abs_tol=1e-9):
                raise ValueError("Episodic advantage mismatch")
        phase_signal = {}
        if "phase_rewards" in members[0]["row"]["rollout"]:
            for phase in ("select", "respond", "write"):
                phase_values = [member["row"]["rollout"]["phase_rewards"][phase] for member in members]
                phase_expected = normalized(phase_values)
                phase_signal[phase] = statistics.pstdev(phase_values) > 0
                for member, value in zip(members, phase_expected, strict=True):
                    if not math.isclose(member["row"]["phase_advantages"][phase], value, abs_tol=1e-9):
                        raise ValueError("Phase advantage mismatch")
        flags = {"all_current_bad": not any(m["current"] for m in members),
                 "all_joint_bad": not any(m["strict"] for m in members),
                 "mixed_joint": 0 < sum(m["strict"] for m in members) < 4,
                 "all_unsafe": all(m["unsafe"] for m in members),
                 "zero_return_signal": statistics.pstdev(returns) == 0}
        counts.update(key for key, flag in flags.items() if flag)
        result.append({"group_id": group_id, "family": members[0]["family"], **flags,
            "returns": returns, "advantages": expected,
            "current": [m["current"] for m in members], "future": [m["future"] for m in members],
            "strict": [m["strict"] for m in members], "unsafe": [m["unsafe"] for m in members]})
        result[-1]["phase_signal"] = phase_signal
    history = [entry for entry in training["history"] if "grad_norm" in entry]
    pass_path = optimizer / "optimizer_passes.jsonl"
    pass_summary = None
    if pass_path.is_file():
        passes = _rows(pass_path)
        n_passes = training["plan"]["rollout_optimization_passes"]
        n_updates = training["resource_counters"]["updates"]
        if len(passes) != n_updates * n_passes:
            raise ValueError("Missing optimization passes")
        expected_tokens = sum(token_counts.values()) * n_passes
        if training["resource_counters"]["trained_tokens"] != expected_tokens:
            raise ValueError("Optimized token counter does not include every pass")
        if training["resource_counters"]["generated_tokens"] != sum(token_counts.values()):
            raise ValueError("Generated token counter should count unique samples")
        if training["resource_counters"]["optimizer_steps"] != len(passes):
            raise ValueError("Adam step counter mismatch")
        by_pass = defaultdict(list)
        for index, entry in enumerate(passes, start=1):
            if entry["optimizer_step"] != index or entry["optimization_pass"] != (index - 1) % n_passes + 1:
                raise ValueError("Optimization pass order mismatch")
            if not entry["optimizer_step_executed"] or not all(math.isfinite(entry[key]) for key in
                    ("loss", "kl", "grad_norm", "ratio_min", "ratio_max")):
                raise ValueError("Optimization pass was skipped or nonfinite")
            by_pass[entry["optimization_pass"]].append(entry)
        pass_summary = {str(index): {"count": len(entries),
            "nonzero_gradient_steps": sum(entry["grad_norm"] > 0 for entry in entries),
            "ratio_min": min(entry["ratio_min"] for entry in entries),
            "ratio_max": max(entry["ratio_max"] for entry in entries),
            "max_sampled_k3_kl": max(entry["kl"] for entry in entries),
            "mean_clip_fraction": statistics.fmean(entry["clip_fraction_phase_mean"] for entry in entries)}
            for index, entries in by_pass.items()}
    return {"scope": "TRAIN only; no validation/evaluation opened", "optimizer": str(optimizer),
        "source_sha256": {name: _hash(optimizer / name) for name in ("training.json", "rollouts.jsonl")},
        "samples": len(rows), "groups": len(groups), "group_counts": dict(counts),
        "family_group_counts": dict(families), "completion_tokens_one_collection": dict(token_counts),
        "resource_counters": training["resource_counters"],
        "frozen_unchanged": training["frozen_before"] == training["frozen_after"],
        "reference_unchanged": training["reference_before"] == training["reference_after"],
        "adapter_changed": training["trainable_before"] != training["trainable_after"],
        "nonzero_outer_gradients": sum(entry["grad_norm"] > 0 for entry in history),
        "outer_gradients_count": len(history),
        "optimization_passes": pass_summary,
        "ratio_min": min((entry["ratio_min"] for entry in history), default=None),
        "ratio_max": max((entry["ratio_max"] for entry in history), default=None),
        "groups_detail": result,
        "interpretation": ["Nonzero gradient proves optimization, not improvement.",
            "At initial ratio=1 a centered scalar loss can be zero with a nonzero derivative.",
            "One-pass optimization never exercises PPO clipping; this is not an absence of policy gradient.",
            "Within-group normalization makes auxiliary-only ranking differences influential.",
            "No correct sampled alternative means a strict-outcome reward alone has no positive example."]}


def review_environment(train_path):
    """Execute only public-rule CPU controls on TRAIN, with real persisted stores."""
    from copy import deepcopy
    import tempfile
    from research.src.env.episode import EpisodeEnvironment
    from research.src.policy.dialogue import declarations
    from research.src.policy.rule_controller import RuleController
    from research.src.readers.symbolic import read
    from research.src.scoring.evidence import Evaluator

    class Control(RuleController):
        def __init__(self, *, no_ask=False, no_write=False):
            self.no_ask, self.no_write = no_ask, no_write

        def select(self, observation):
            return {"action": "PROCEED", "slot": None} if self.no_ask else super().select(observation)

        def write(self, observation):
            return {"operations": [{"op": "NOOP"}]} if self.no_write else super().write(observation)

    def execute_control(episode, policy):
        # Same public controller, executor and scorer as run_episode; omit only
        # its unrelated exhaustive calendar/context diagnostic probes.
        with tempfile.TemporaryDirectory(prefix="targeted_cpu_qa_") as work:
            env = EpisodeEnvironment(episode, Path(work) / "memory.sqlite")
            try:
                decision = policy.select(env.observe("select"))
                if decision["action"] == "ASK":
                    env.ask(decision["slot"])
                response = policy.respond(env.observe("respond"))
                operation = policy.write(env.observe("write"))
                result = env.apply_memory_operations(operation["operations"])
                interaction = deepcopy(env.private_trace["observations"]["respond"])
                questions = env.question_count
                env.close_session()
                persisted = env.reopen_for_reuse()
                trace = deepcopy(env.private_trace)
                trace.update(episode=deepcopy(episode), interaction_observation=interaction,
                             response=response, future_response=read(persisted["memory"], persisted["query"]))
                evaluator = Evaluator()
                memory = evaluator.score_memory(trace)
                return {"metrics": {"C2": evaluator.score_current(trace)["success"],
                    "FJS": evaluator.score_future(trace)["success"], "questions": questions,
                    "invalid_outputs": int(not result["valid"]),
                    "unsupported_operations": memory["unsupported_operations"]},
                    "executed_operations": trace["executed_operations"]}
            finally:
                env.dispose()

    train_path = Path(train_path).resolve()
    before = _hash(train_path)
    episodes = _rows(train_path)
    if any(episode["split"] != "train" for episode in episodes):
        raise ValueError("Environment audit accepts TRAIN only")
    totals, by_family, details = Counter(), defaultdict(Counter), []
    policies = {"public": {}, "no_write": {"no_write": True}, "no_ask": {"no_ask": True},
                "no_ask_no_write": {"no_ask": True, "no_write": True}}
    for index, episode in enumerate(episodes):
        panels = {name: execute_control(episode, Control(**options))
                  for name, options in policies.items()}
        family = episode["scenario_family"]
        facts = declarations({"user_messages": episode["user_messages"], "query": episode["current_query"]})
        public, noop, never = (panels[name]["metrics"] for name in ("public", "no_write", "no_ask"))
        flags = {"initial_memory_sufficient_no_ask": panels["no_ask_no_write"]["metrics"]["FJS"] == 1,
            "initial_memory_sufficient_after_public_ask": noop["FJS"] == 1,
            "positive_write_effect": public["FJS"] > noop["FJS"],
            "positive_ask_effect_current": public["C2"] > never["C2"],
            "positive_ask_effect_future": public["FJS"] > never["FJS"],
            "public_asks": public["questions"] > 0,
            "public_writes_queried_attribute": any(op.get("record", {}).get("attribute") == episode["future_query"]["attribute"]
                                                  for op in panels["public"]["executed_operations"]),
            "query_differs_from_last_declaration": bool(facts and facts[-1][1] != episode["current_query"]["attribute"])}
        totals.update(key for key, flag in flags.items() if flag)
        by_family[family].update(key for key, flag in flags.items() if flag)
        details.append({"root_id": episode["root_id"], "latent_world_id": episode["latent_world_id"],
            "family": family, **flags, "panels": {name: {key: row["metrics"][key] for key in
                ("C2", "FJS", "questions", "invalid_outputs", "unsupported_operations")} for name, row in panels.items()}})
    if _hash(train_path) != before:
        raise RuntimeError("TRAIN source changed during CPU review")
    by_root = defaultdict(list)
    for item in details:
        by_root[item["root_id"]].append(item)
    root_panels = {name: {metric: statistics.fmean(
        statistics.fmean(item["panels"][name][metric] for item in variants)
        for variants in by_root.values()) for metric in ("C2", "FJS", "questions")}
        for name in policies}
    root_effects = {flag: statistics.fmean(statistics.fmean(float(item[flag]) for item in variants)
        for variants in by_root.values()) for flag in (
            "positive_write_effect", "positive_ask_effect_current", "positive_ask_effect_future",
            "initial_memory_sufficient_no_ask", "initial_memory_sufficient_after_public_ask")}
    return {"scope": "TRAIN only; four public-rule controls, no Qwen/gradients/held-out data",
        "train_path": str(train_path), "train_sha256": before, "episodes": len(episodes),
        "roots": len({episode["root_id"] for episode in episodes}), "counts": dict(totals),
        "family_counts": {family: dict(counts) for family, counts in by_family.items()}, "details": details,
        "root_weighted_panels": root_panels, "root_weighted_effects": root_effects,
        "identification": "Paired policy interventions use the same authored episode and real SQLite executor; conclusions concern this finite scripted environment."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--optimizer")
    source.add_argument("--train")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = review_train(args.optimizer) if args.optimizer else review_environment(args.train)
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with Path(args.output).open("x") as stream:
            stream.write(content)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
