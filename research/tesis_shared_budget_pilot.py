"""A CPU development pilot. The reserved reviewer packets are never opened here."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from research.src.analysis.paired_statistics import paired_bootstrap
from research.src.artifacts import digest, file_hash, write_json
from research.tesis_interactive_reward import DEFAULT_CONFIG
from research.tesis_shared_budget_bank import audit_development, build_development
from research.tesis_shared_budget_env import ACTIONS, rule_action
from research.tesis_shared_budget_planner import (
    enumerate_world, evaluate, fit_public_policy, passive, public_state, training_groups, traverse,
)


RULE, MYOPIC, PLAN, PASSIVE = "regla-publica", "valor-un-paso", "planificador-publico", "continuar"
VERIFIER_OFF = "planificador-sin-veto"
SEEDS = (101, 202, 303)
GATE = {"min_future_gain": .05, "max_current_loss": .02, "max_extra_questions": .10}
PRIMARY = "future_supported"


def stratified_mean(rows, name):
    strata = defaultdict(list)
    for row in rows:
        strata[row["scenario_family"]].append(row["metrics"][name])
    return mean(mean(items) for items in strata.values())


def summarize(rows):
    methods = sorted({row["method"] for row in rows})
    summary = {}
    for method in methods:
        own = [row for row in rows if row["method"] == method]
        families = sorted({row["scenario_family"] for row in own})
        summary[method] = {
            "macro": {name: stratified_mean(own, name) for name in own[0]["metrics"]},
            "families": {
                family: {
                    **{name: mean(row["metrics"][name] for row in own if row["scenario_family"] == family)
                       for name in (PRIMARY, "current_supported", "questions_first", "questions_second")},
                    "first_actions": dict(Counter(max(row["first_actions"], key=row["first_actions"].get)
                                                  for row in own if row["scenario_family"] == family)),
                } for family in families
            },
        }
    return summary


def paired(rows, left, right, metric):
    return paired_bootstrap(rows, left, right, metric, confidence=.95, resamples=2000)


def gate(rows):
    contrasts = {}
    for other in (RULE, MYOPIC):
        future = paired(rows, PLAN, other, PRIMARY)
        current = paired(rows, PLAN, other, "current_supported")
        questions = paired(rows, PLAN, other, "questions")
        unsupported = paired(rows, PLAN, other, "future_unsupported_reuse")
        positive = (
            future["interval"][0] > 0
            and current["estimate"] >= -GATE["max_current_loss"]
            and questions["estimate"] <= GATE["max_extra_questions"]
            and unsupported["estimate"] <= 1e-12
        )
        passed = positive and future["estimate"] >= GATE["min_future_gain"]
        contrasts[other] = {"future": future, "current": current, "questions": questions,
                            "unsupported_reuse": unsupported, "passed": passed, "positive": positive}
    return {"passed": all(item["passed"] for item in contrasts.values()),
            "exploratory_training_trigger": all(item["positive"] for item in contrasts.values()),
            "comparisons": contrasts}


def source_hashes():
    project = Path(__file__).resolve().parent.parent
    paths = [project / f"research/tesis_shared_budget_{name}.py"
             for name in ("pilot", "planner", "train", "env", "bank")]
    paths.extend(project / f"research/{name}.py" for name in
                 ("tesis_interactive_env", "tesis_interactive_reward", "tesis_bank", "competence_data"))
    paths.extend((project / "research/src").rglob("*.py"))
    return {str(path.relative_to(project)): file_hash(path) for path in sorted(paths)}


def coverage(trees, policy):
    total = unseen = 0
    for tree in trees:
        path, _ = traverse(tree, policy)
        for depth in range(len(path)):
            node = tree.decisions[path[:depth]]
            total += 1
            unseen += public_state(node.observation, node.legal) not in policy.actions
    return {"decision_visits": total, "unseen_decision_visits": unseen}


def train_matched(train_trees, validation, planner, output, reference):
    import torch
    from research.tesis_shared_budget_train import ARMS, LONGITUDINAL, fit

    groups = training_groups(train_trees, planner)
    rows, accounting = [], []
    def rank(policy):
        candidate = evaluate(validation, policy, method="candidate")
        future = stratified_mean(candidate, PRIMARY)
        current = stratified_mean(candidate, "current_supported")
        questions = stratified_mean(candidate, "questions")
        unsupported = stratified_mean(candidate, "future_unsupported_reuse")
        eligible = (
            current >= reference["current_supported"] - GATE["max_current_loss"]
            and questions <= reference["questions"] + GATE["max_extra_questions"]
            and unsupported <= reference["future_unsupported_reuse"] + 1e-12
        )
        return (eligible, round(future, 12), round(-questions, 12))

    before = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for seed in SEEDS:
            policies, ledger = fit(groups, seed, actions=ACTIONS, state=public_state, validation=rank)
            write_json(output / f"selectors_{seed}.json", {name: policy.artifact() for name, policy in policies.items()})
            accounting.append(ledger)
            for method, policy in policies.items():
                rows.extend(evaluate(validation, policy.action, method=method, seed=seed))
            print(f"Matched selectors, seed {seed}: done", flush=True)
    finally:
        torch.set_num_threads(before)
    return {
        "status": "development_only",
        "group_digest": digest(groups),
        "summary": summarize(rows),
        "accounting": accounting,
        "paired_future": {arm: paired(rows, LONGITUDINAL, arm, PRIMARY)
                          for arm in ARMS if arm != LONGITUDINAL},
        "rows": rows,
    }


def enumerate_split(stories, workers):
    if workers == 1:
        return [enumerate_world(story) for story in stories]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(enumerate_world, stories, chunksize=2))


def run(output, *, cpu_only=False, workers=1):
    if workers < 1:
        raise ValueError("At least one CPU worker is required")
    output = Path(output)
    splits = build_development()
    if set(splits) != {"train", "validation"}:
        raise ValueError("The development runner accepts TRAIN and validation only")
    audit = audit_development(splits)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "PLAN.json", {
        "status": "development_only",
        "source_hashes": source_hashes(),
        "split_digests": {split: digest(stories) for split, stories in splits.items()},
        "budget": 1, "primary": PRIMARY, "gate": GATE,
        "reward": asdict(DEFAULT_CONFIG),
        "bootstrap": {"unit": "root", "strata": "scenario_family", "resamples": 2000, "confidence": .95},
        "training_seeds": SEEDS,
        "training": "common TRAIN sibling branches; same continuation for all horizons; validation checkpoints only",
        "exploratory_training_trigger": "positive paired future CI vs both rule and myopic with guards; the 5 pp practical criterion is separate",
        "public_planning": "compressed visible states + action history; TRAIN transitions only; utility ties favor fewer writes",
        "unnecessary_questions": "remove each question and replay the other actions; current/future support and harm must be no worse",
        "checkpoint_ranking": "current, interaction and unsupported reuse guards vs public rule; future; fewer questions; earliest",
        "reserved_access": False,
    })
    write_json(output / "BANK_AUDIT.json", audit)
    print("Enumerating TRAIN", flush=True)
    train = enumerate_split(splits["train"], workers)
    planner = fit_public_policy(train, plan=True)
    myopic = fit_public_policy(train, plan=False)
    print("Evaluating on development validation", flush=True)
    validation = enumerate_split(splits["validation"], workers)
    policies = {RULE: rule_action, MYOPIC: myopic, PLAN: planner, PASSIVE: passive}
    rows = [row for method, policy in policies.items() for row in evaluate(validation, policy, method=method)]
    rows.extend(evaluate(validation, planner, method=VERIFIER_OFF, verifier="off"))
    comparison = gate(rows)
    summary = summarize(rows)
    write_json(output / "CPU.json", {
        "gate": comparison, "summary": summary, "rows": rows,
        "state_coverage": {PLAN: coverage(validation, planner), MYOPIC: coverage(validation, myopic)},
        "interactions": {
            "train_replay_steps": sum(tree.environment_steps for tree in train),
            "validation_replay_steps": sum(tree.environment_steps for tree in validation),
            "train_terminal_paths": sum(len(tree.outcomes) for tree in train),
            "validation_terminal_paths": sum(len(tree.outcomes) for tree in validation),
        },
        "verifier_veto_minus_off": {
            name: paired(rows, PLAN, VERIFIER_OFF, name)
            for name in (PRIMARY, "unsupported_writes", "expired_reuse", "future_unsupported_reuse")
        },
    })
    print(f"CPU gate passed: {comparison['passed']}", flush=True)
    if comparison["exploratory_training_trigger"] and not cpu_only:
        write_json(output / "TRAINING.json", train_matched(train, validation, planner, output, summary[RULE]["macro"]))
    else:
        write_json(output / "TRAINING.json", {"status": "cpu_only" if cpu_only else "cpu_gate_not_met"})
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpu-only", action="store_true", help="Stop after the CPU gate")
    parser.add_argument("--workers", default=1, type=int, help="Number of CPU processes for branch replay")
    args = parser.parse_args()
    run(args.output, cpu_only=args.cpu_only, workers=args.workers)


if __name__ == "__main__":
    main()
