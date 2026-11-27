"""Public Qwen views and root-level analysis for the diverse-user study."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from research.src.artifacts import canonical
from research.tesis_profiles_env import evaluate as evaluate_roots, public_features
from research.tesis_shared_budget_planner import Decision
from research.tesis_shared_budget_qwen_context import with_public_context


CONTEXT_TURN = "assistant_profile_evidence"


def with_profile_context(observation, legal):
    """Keep estimated timing from visible messages apart from simulator facts."""
    original = {**observation, "user_messages": [
        turn for turn in observation["user_messages"] if turn.get("turn_id") != CONTEXT_TURN
    ]}
    annotated = with_public_context(original)
    features = public_features(original, legal)
    message = (
        "Indicadores estimados exclusivamente del diálogo visible sobre disponibilidad, "
        "retorno y planes: " + canonical(features)
    )
    return {**annotated, "user_messages": [
        *annotated["user_messages"], {"role": "assistant", "turn_id": CONTEXT_TURN, "text": message},
    ]}


def actor_trees(trees):
    cache = {}
    result = []
    for tree in trees:
        decisions = {}
        for prefix, node in tree.decisions.items():
            key = canonical([node.observation, node.legal])
            if key not in cache:
                cache[key] = Decision(with_profile_context(node.observation, node.legal), node.legal)
            decisions[prefix] = cache[key]
        result.append(replace(tree, decisions=decisions))
    return result


def evaluate(trees, policy, *, method, seed):
    def choose(observation, legal):
        return legal[0] if len(legal) == 1 else policy(observation, legal)

    return evaluate_roots(trees, choose, method=method, seed=seed)


def population(trees):
    families = defaultdict(set)
    for tree in trees:
        families[tree.story["scenario_family"]].add(tree.story["root_id"])
    return {"roots": sum(map(len, families.values())), "worlds": len(trees),
            "roots_by_family": {family: len(roots) for family, roots in sorted(families.items())}}
