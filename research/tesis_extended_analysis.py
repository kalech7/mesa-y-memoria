"""Shared views, branch targets and paired root metrics for the extended pilot."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from math import fsum

from research.src.artifacts import canonical, digest
from research.tesis_extended_env import evaluate as evaluate_roots
from research.tesis_shared_budget_env import QUESTION_ACTIONS
from research.tesis_shared_budget_planner import Decision, group_trees, traverse, weighted
from research.tesis_shared_budget_qwen import encode, model_inputs, prompt_messages
from research.tesis_shared_budget_qwen_context import with_public_context


BRANCHES_PER_ROLE = 4
WEIGHT_BY_ROLE = {"ask_or_wait": .60, "partial_follow_up": .25, "close": .15}


def actor_trees(trees):
    """Annotate only model-visible observations, keeping the simulator fixed."""
    cache = {}
    annotated = []
    for tree in trees:
        decisions = {}
        for prefix, node in tree.decisions.items():
            key = canonical(node.observation)
            if key not in cache:
                cache[key] = with_public_context(node.observation)
            decisions[prefix] = Decision(cache[key], node.legal)
        annotated.append(replace(tree, decisions=decisions))
    return annotated


def validation_inputs(trees, tokenizer, max_tokens):
    decisions = [replace(tree, decisions={prefix: node for prefix, node in tree.decisions.items()
                                           if len(node.legal) > 1}) for tree in trees]
    return model_inputs(decisions, tokenizer, max_tokens)


def evaluate(trees, policy, *, method, seed):
    """Average simulated worlds within each public root before comparing methods."""
    def choose(observation, legal):
        return legal[0] if len(legal) == 1 else policy(observation, legal)

    return evaluate_roots(trees, choose, method=method, seed=seed)


def role_for(node):
    obs = node.observation
    if not set(node.legal).intersection(QUESTION_ACTIONS):
        return "close"
    return "partial_follow_up" if obs["asked_slots"] else "ask_or_wait"


def branch_data(trees, model_trees, planner, tokenizer, max_tokens):
    """Select public groups deterministically, then score every legal branch."""
    if not trees or any(tree.story["split"] != "train" for tree in trees):
        raise ValueError("Only TRAIN decisions can enter the optimizer")
    private_groups = group_trees(trees)
    visible_groups = group_trees(model_trees)
    records, interactions = [], 0
    for root, worlds in private_groups.items():
        prompts = defaultdict(dict)
        visible = {world.story["world_id"]: world for world in visible_groups[root]}
        for world in worlds:
            model_world = visible[world.story["world_id"]]
            for prefix, node in world.decisions.items():
                if len(node.legal) < 2:
                    continue
                observed = model_world.decisions[prefix]
                role = (node.observation["visit"], role_for(node))
                prompt = tuple(row["content"] for row in prompt_messages(observed.observation, observed.legal))
                prompts[role].setdefault(prompt, []).append((world, prefix, observed))
        for (visit, role), positions in prompts.items():
            # Each selected group keeps all compatible worlds. Prefer the shortest
            # histories so early decisions and immediate follow-ups are represented.
            ordered = sorted(positions.values(), key=lambda states: (len(states[0][1]),
                               digest(prompt_messages(states[0][2].observation, states[0][2].legal))))
            for states in ordered[:BRANCHES_PER_ROLE]:
                node = states[0][2]
                returns, current_returns = [], []
                for action in node.legal:
                    outcomes = [(world.mass, *traverse(world, planner, prefix + (action,)))
                                for world, prefix, _ in states]
                    returns.append(weighted((mass, measured["longitudinal"]) for mass, _, measured in outcomes))
                    current_returns.append(weighted((mass, measured["current_supported"] - .01 * measured["questions"])
                                                    for mass, _, measured in outcomes))
                    interactions += sum(len(end) - len(start) for (_, end, _), (_, start, _) in zip(outcomes, states))
                records.append({
                    "record": encode(tokenizer, node.observation, node.legal, max_tokens),
                    "returns": tuple(returns), "current_returns": tuple(current_returns),
                    "root_id": root, "role": role, "visit": visit,
                    "split": "train",
                })
    if not records or not any(row["role"] == "partial_follow_up" for row in records):
        raise ValueError("The curriculum needs public follow-ups after a partial reply")
    occurrences = Counter((row["role"], row["root_id"]) for row in records)
    root_count = Counter(role for role, _ in occurrences)
    for row in records:
        role, root = row["role"], row["root_id"]
        row["weight"] = WEIGHT_BY_ROLE[role] / root_count[role] / occurrences[role, root]
    ledger = {
        "groups": len(records), "sha256": digest(records),
        "phase_counts": {f"{visit}:{role}": count for (visit, role), count in
                         Counter((row["visit"], row["role"]) for row in records).items()},
        "role_weights": {role: fsum(row["weight"] for row in records if row["role"] == role)
                         for role in WEIGHT_BY_ROLE},
        "logged_transition_references": interactions,
    }
    return records, ledger
