"""Enumerate SQLite trajectories and fit decisions on public TRAIN observations."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import fsum, isclose
from statistics import mean

from research.tesis_interactive_reward import metrics, reward
from research.tesis_shared_budget_env import QUESTION_ACTIONS, SharedBudgetEpisode, decision_state, rule_action


@dataclass(frozen=True)
class Decision:
    observation: dict
    legal: tuple[str, ...]


def public_state(observation, legal):
    evidence = decision_state(observation)
    return (
        evidence["strategy_cue"], evidence["visit"], evidence["questions_remaining"],
        evidence["missing_slot"], evidence["clarification_state"],
        evidence["current_resolved"], evidence["scope_ambiguous"],
        tuple(observation["action_history"]), tuple(legal),
    )


def outcome(trace):
    measured = metrics(trace)
    future = trace["future"]
    return {
        "future_supported": measured["future_supported"],
        "current_supported": measured["current_supported"],
        "strict_joint": measured["strict_joint"],
        "questions": measured["questions"],
        "questions_first": trace["visits"][0]["questions"],
        "questions_second": trace["visits"][1]["questions"],
        "uninformative_questions": measured["questions"] - measured["informative_questions"],
        "redundant_questions": measured["redundant_questions"],
        "applied_writes": sum(operation["op"] != "NOOP" for visit in trace["visits"]
                              for operation in visit["executed_operations"]),
        "unsupported_writes": measured["unsupported_writes"],
        "rejected_proposals": sum(visit["rejected_proposals"] for visit in trace["visits"]),
        "future_unsupported_reuse": mean(item["unsupported_commitment"] for item in future),
        "expired_reuse": mean(item["expired_reuse"] for item in future),
        "provisional_responses": sum(visit["session_only_recommendation"] for visit in trace["visits"]),
        "longitudinal": reward(trace, horizon="longitudinal")["total"],
        "immediate": reward(trace, horizon="immediate")["total"],
    }


@dataclass
class WorldTree:
    story: dict
    decisions: dict
    outcomes: dict
    environment_steps: int

    @property
    def mass(self):
        return self.story["world_probability"]


def simulate(story, prefix, *, verifier="veto"):
    """Replay a branch with its own database. Only the evaluator receives the world."""
    env = SharedBudgetEpisode(story, question_budget=story["question_budget"], verifier=verifier)
    try:
        for action in prefix:
            env.step(action)
        if env.done:
            return outcome(env.summary())
        observation = {**env.observe(), "action_history": tuple(prefix)}
        return Decision(observation, tuple(env.legal_actions))
    finally:
        env.close()


def enumerate_world(story):
    pending = [()]
    decisions, outcomes = {}, {}
    steps = 0
    while pending:
        prefix = pending.pop()
        result = simulate(story, prefix)
        steps += len(prefix)
        if isinstance(result, Decision):
            decisions[prefix] = result
            pending.extend(prefix + (action,) for action in result.legal)
        else:
            outcomes[prefix] = result
    return WorldTree(story, decisions, outcomes, steps)


def group_trees(trees):
    roots = defaultdict(list)
    for tree in trees:
        roots[tree.story["root_id"]].append(tree)
    for root, worlds in roots.items():
        weights = [world.mass for world in worlds]
        if not weights or min(weights) <= 0 or not isclose(fsum(weights), 1.0):
            raise ValueError(f"World probabilities do not sum to one for {root}")
        if len({world.story["world_id"] for world in worlds}) != len(worlds):
            raise ValueError(f"Duplicate world for {root}")
        if any(world.decisions[()] != worlds[0].decisions[()] for world in worlds):
            raise ValueError(f"Different initial public observations for {root}")
    return roots


def passive(observation, legal):
    return "PROCEED"


def traverse(tree, policy, prefix=()):
    while prefix in tree.decisions:
        node = tree.decisions[prefix]
        action = policy(node.observation, node.legal)
        if action not in node.legal:
            raise ValueError(f"Policy chose an illegal action: {action}")
        prefix += (action,)
    return prefix, tree.outcomes[prefix]


def weighted(values):
    values = list(values)
    mass = fsum(probability for probability, _ in values)
    return fsum(probability * value for probability, value in values) / mass


class FittedPolicy:
    def __init__(self):
        self.actions = {}

    def __call__(self, observation, legal):
        return self.actions.get(public_state(observation, legal)) or rule_action(observation, legal)


def fit_public_policy(trees, *, plan):
    """Fit a finite decision table on TRAIN only, sharing decisions by public state.

    Both policies optimize the same utility. The one-step policy estimates the
    effect of the current action with no further questions. The planner uses
    backward induction with later decision rules fitted on the same TRAIN tree.
    """
    if not trees or any(tree.story["split"] != "train" for tree in trees):
        raise ValueError("A public policy may be fitted only on TRAIN trees")
    group_trees(trees)
    buckets = defaultdict(list)
    for tree in trees:
        for prefix, node in tree.decisions.items():
            buckets[public_state(node.observation, node.legal)].append((tree, prefix, node))
    policy = FittedPolicy()
    # The public key includes the action history, so each bucket has one depth.
    for key, bucket in sorted(buckets.items(), key=lambda item: -len(item[1][0][1])):
        choices = bucket[0][2].legal
        continuation = policy if plan else passive
        values = {}
        for action in choices:
            branches = [(tree.mass, traverse(tree, continuation, prefix + (action,))[1])
                        for tree, prefix, _ in bucket]
            utility = weighted((mass, result["longitudinal"]) for mass, result in branches)
            writes = weighted((mass, result["applied_writes"]) for mass, result in branches)
            values[action] = (round(utility, 12), round(-writes, 12), action == "PROCEED")
        policy.actions[key] = max(choices, key=values.__getitem__)
    return policy


def training_groups(trees, continuation):
    """Pair every action from a common public prefix, averaging compatible worlds."""
    if any(tree.story["split"] != "train" for tree in trees):
        raise ValueError("Only TRAIN trees can produce optimization groups")
    groups = []
    for root_id, worlds in group_trees(trees).items():
        buckets = defaultdict(list)
        for tree in worlds:
            for prefix, node in tree.decisions.items():
                if len(node.legal) > 1:
                    buckets[public_state(node.observation, node.legal)].append((tree, prefix, node))
        for state_key, bucket in buckets.items():
            node = bucket[0][2]
            branches = {
                action: [(tree.mass, *traverse(tree, continuation, prefix + (action,)))
                         for tree, prefix, _ in bucket]
                for action in node.legal
            }
            groups.append({
                "root_id": root_id, "state_key": str(state_key), "split": "train",
                "observation": node.observation, "legal": node.legal,
                "reference": rule_action(node.observation, node.legal),
                "longitudinal": tuple(weighted((mass, record["longitudinal"]) for mass, _, record in branches[action])
                                      for action in node.legal),
                "immediate": tuple(weighted((mass, record["immediate"]) for mass, _, record in branches[action])
                                   for action in node.legal),
                "interactions": sum(len(leaf) - len(bucket[index][1])
                                    for records in branches.values()
                                    for index, (_, leaf, _) in enumerate(records)),
            })
    return groups


def evaluate(trees, policy, *, method, seed=0, verifier="veto"):
    """Return one weighted row per root; worlds are never independent samples."""
    rows = []
    for root_id, worlds in group_trees(trees).items():
        evaluated = []
        first_actions = defaultdict(float)
        for tree in worlds:
            path, measured = traverse(tree, policy)
            def measure(other_path):
                return (tree.outcomes[other_path] if verifier == "veto"
                        else simulate(tree.story, other_path, verifier=verifier))
            if verifier != "veto":
                measured = measure(path)
            if isinstance(measured, Decision):
                raise ValueError("Verifier comparison must replay the same complete action path")
            unnecessary = 0
            for index, action in enumerate(path):
                if action in QUESTION_ACTIONS:
                    without = measure(path[:index] + path[index + 1:])
                    unnecessary += int(
                        without["future_supported"] >= measured["future_supported"]
                        and without["current_supported"] >= measured["current_supported"]
                        and without["future_unsupported_reuse"] <= measured["future_unsupported_reuse"]
                        and without["expired_reuse"] <= measured["expired_reuse"]
                        and without["unsupported_writes"] <= measured["unsupported_writes"]
                    )
            first_actions[path[0]] += tree.mass
            evaluated.append((tree.mass, {**measured, "counterfactually_unnecessary_questions": unnecessary}))
        rows.append({
            "method": method, "seed": seed, "root_id": root_id,
            "scenario_family": worlds[0].story["scenario_family"],
            "first_actions": dict(first_actions),
            "metrics": {name: weighted((mass, record[name]) for mass, record in evaluated)
                        for name in evaluated[0][1]},
        })
    return rows
