"""Reproduce the exploration and normalization diagnosis for archived GRPO runs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean, median, pstdev

from research.src.artifacts import file_hash, write_json
from research.tesis_interactive_env import InteractiveEpisode, reference_action
from research.tesis_interactive_reward import metrics, reward


RUNS = ("tesis_interactive_v1", "tesis_interactive_low_supervision_v2")
METHOD = "GRPO-longitudinal"
SFT_CONTROL = "SFT-adicional"


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _first_visit_resolution(story, group, counts):
    visit = story["visits"][0]
    if not visit["slot"] or visit["style"] == "refused":
        return
    prefix = [visit["slot"]] * (2 if visit["style"] == "partial" else 1)
    successes = sum(actions[:len(prefix)] == prefix for actions in group["actions"])
    counts["resolvable_first_visit"] += 1
    counts["sampled_resolving_prefix"] += successes > 0
    counts["all_draws_sampled_resolving_prefix"] += successes == len(group["actions"])


def _retrospective(run):
    results = json.loads((run / "RESULTS.json").read_bytes())
    train = {(story["root_id"], story["world_id"]): story for story in rows(run / "bank/train.jsonl")}
    sources = {"RESULTS.json": file_hash(run / "RESULTS.json"),
               "bank/train.jsonl": file_hash(run / "bank/train.jsonl")}
    family_counts, normalization, seed_outcomes = defaultdict(Counter), defaultdict(list), {}
    for seed in results["seeds"]:
        group_path = run / str(seed) / f"{METHOD}-grupos.jsonl"
        sources[str(group_path.relative_to(run))] = file_hash(group_path)
        for group in rows(group_path):
            story = train[group["root_id"], group["world_id"]]
            counters = family_counts[story["scenario_family"]]
            counters["groups"] += 1
            _first_visit_resolution(story, group, counters)
            category = (
                "task_varies" if group["optimization_task_varies"]
                else "only_penalties_vary" if group["only_penalties_vary"] else "identical"
            )
            counters[category] += 1
            totals = [entry["total"] for entry in group["returns"]["longitudinal"]]
            advantages = group["advantages"]
            normalization[category].append((max(totals) - min(totals), pstdev(advantages),
                                             mean(abs(advantage) for advantage in advantages)))
        panels = results["panels"][str(seed)]
        series = {}
        for method in (SFT_CONTROL, f"{METHOD}-terminal"):
            panel = panels[method]
            trace_path = run / panel["traces"]
            if file_hash(trace_path) != panel["trace_sha256"]:
                raise ValueError(f"Changed trace at {trace_path}")
            sources[panel["traces"]] = panel["trace_sha256"]
            series[method] = {(row["root_id"], row["world_id"]): row for row in rows(trace_path)}
        control, rl = series.values()
        if control.keys() != rl.keys():
            raise ValueError("Unpaired historical trajectories")
        control_misses, visits_with_refusal = 0, 0
        lost = []
        for key, trace in control.items():
            raw_future = [query["supported"] for query in trace["future"]]
            misses = len(raw_future) - sum(raw_future)
            control_misses += misses
            if any(event["delivery"] == "unavailable" for visit in trace["visits"] for event in visit["events"]):
                visits_with_refusal += misses
            changed = rl[key]
            rl_future = [query["supported"] for query in changed["future"]]
            if any(before > after for before, after in zip(raw_future, rl_future, strict=True)):
                lost.append({"root_id": key[0], "world_id": key[1], "family": trace["scenario_family"],
                             "sft_first_visit_questions": [e["action"] for e in trace["visits"][0]["events"]],
                             "grpo_first_visit_questions": [e["action"] for e in changed["visits"][0]["events"]]})
        seed_outcomes[seed] = {"sft_unsupported_probes": control_misses,
                               "sft_unsupported_probes_on_episodes_with_refusal": visits_with_refusal,
                               "grpo_terminal_questions": panels[f"{METHOD}-terminal"]["macro"]["questions"],
                               "episodes_with_lost_future_probe": lost}
    return {"source_sha256": sources, "exploration_by_family": family_counts, "reserved_outcomes": seed_outcomes,
            "normalization": {name: {"groups": len(samples),
                                     "mean_raw_return_span": mean(span for span, _, _ in samples),
                                     "median_raw_return_span": median(span for span, _, _ in samples),
                                     "median_normalized_advantage_std": median(scale for _, scale, _ in samples),
                                     "mean_absolute_normalized_advantage": mean(value for _, _, value in samples)}
                              for name, samples in normalization.items()}}


def _execute(story, prefix, *, follow_rule=False):
    env = InteractiveEpisode(story)
    try:
        for action in prefix:
            env.step(action)
        while follow_rule and not env.done:
            action = reference_action(env.observe())
            env.step(action if action in env.legal_actions else "PROCEED")
        if env.done:
            trace = env.summary()
            return metrics(trace), reward(trace)["total"], ()
        return None, None, env.legal_actions
    finally:
        env.close()


def _validation_bound(run):
    """Allow a privileged choice between continuing and the actual resolving question."""
    source = run / "bank/validation.jsonl"
    counts = Counter()
    for story in rows(source):
        teacher, teacher_return, _ = _execute(story, (), follow_rule=True)
        leaves = [(teacher, teacher_return)]
        pending = [()]
        while pending:
            prefix = pending.pop()
            measured, score, legal = _execute(story, prefix)
            if measured is not None:
                leaves.append((measured, score))
                continue
            visit = story["visits"][prefix.count("PROCEED")]
            pending.extend(prefix + (action,) for action in legal if action in ("PROCEED", visit["slot"]))
        future = max(item["future_supported"] for item, _ in leaves)
        current = max(item["current_supported"] for item, _ in leaves if item["future_supported"] == future)
        min_questions = min(item["questions"] for item, _ in leaves
                            if item["future_supported"] == future and item["current_supported"] == current)
        counts["episodes"] += 1
        counts["higher_future_possible"] += future > teacher["future_supported"]
        counts["higher_current_without_future_loss_possible"] += current > teacher["current_supported"]
        counts["same_accuracy_with_fewer_questions_possible"] += min_questions < teacher["questions"]
        counts["higher_return_possible"] += max(score for _, score in leaves) > teacher_return + 1e-10
    return {"scope": "Historical validation only; restricted to PROCEED and each visit's privately authored slot. "
                     "The privileged branch choice is a diagnostic, not a deployable policy.",
            "source_sha256": file_hash(source), "counts": dict(counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("research/runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"scope": "Retrospective on previously completed runs; the new experiment is excluded.",
              "runs": {name: _retrospective(args.runs / name) for name in RUNS},
              "target_or_proceed_bound": _validation_bound(args.runs / RUNS[0])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)


if __name__ == "__main__":
    main()
