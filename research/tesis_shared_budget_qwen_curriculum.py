"""Weight the public decisions that determine when to spend a shared question."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from math import fsum

from research.src.artifacts import digest
from research.tesis_shared_budget_env import decision_state
from research.tesis_shared_budget_planner import group_trees, traverse, weighted
from research.tesis_shared_budget_qwen import encode, prompt_messages


MASS = {
    "first_ask": .24, "first_reserve": .24, "second_ambiguous": .24,
    "second_settled": .14, "close_after_ask": .14,
}
WRITE_COST = .001
THREE_REWRITES = {
    "ready": (
        "Tengo un ratito libre para seguir con esto.",
        "Me quedo pendiente del chat mientras organizo la comida.",
        "Antes de irme todavía puedo responder si hace falta precisar algo.",
    ),
    "busy": (
        "Me están llamando y voy a soltar el teléfono.",
        "Estoy entrando a una reunión y en cualquier momento cierro el chat.",
        "Voy con prisa y apenas podré mirar el teléfono.",
    ),
    "regular": (
        "Lo de mañana debería salir como siempre.",
        "Por ahora tengo encaminada la comida de mañana.",
        "Mañana cuento con hacer mi comida de costumbre.",
    ),
    "contingent": (
        "Todavía no sé cómo se me va a organizar la comida de mañana.",
        "Lo de mañana puede cambiar en cuanto me confirmen los planes.",
        "Me falta saber cómo queda el horario para decidir lo de mañana.",
    ),
}
CLOSE_PREFIXES = {
    ("TEMPORAL_SCOPE",),
    ("TEMPORAL_SCOPE", "PROCEED"),
    ("TEMPORAL_SCOPE", "PROVISIONAL"),
    ("PROCEED", "TEMPORAL_SCOPE"),
    ("PROVISIONAL", "TEMPORAL_SCOPE"),
}


def train_bucket(prefix, node, planner):
    if not prefix:
        return "first_ask" if planner(node.observation, node.legal) == "TEMPORAL_SCOPE" else "first_reserve"
    if prefix in (("PROCEED",), ("PROVISIONAL",)):
        state = decision_state(node.observation)
        return "second_ambiguous" if state["missing_slot"] == "TEMPORAL_SCOPE" else "second_settled"
    if prefix in CLOSE_PREFIXES:
        return "close_after_ask"
    return None


def first_variants(observation):
    """Rephrase availability and tomorrow's plans without modifying world outcomes."""
    yield observation
    availability, later = observation["strategy_cue"].split("_")
    for current, next_meal in zip(THREE_REWRITES[availability], THREE_REWRITES[later]):
        variant = deepcopy(observation)
        variant["user_messages"][0]["text"] = f"Quiero organizar la comida. {current} {next_meal}"
        yield variant


def build_records(trees, planner, tokenizer, max_tokens):
    """Freeze returns by actual visible prompt and normalize strata by root."""
    if not trees or any(tree.story["split"] != "train" for tree in trees):
        raise ValueError("The branch curriculum only accepts TRAIN trees")
    records = []
    interactions = 0
    for root, worlds in group_trees(trees).items():
        prompts = defaultdict(list)
        for tree in worlds:
            for prefix, node in tree.decisions.items():
                bucket = train_bucket(prefix, node, planner)
                if bucket is not None:
                    key = (bucket, tuple(item["content"] for item in prompt_messages(node.observation, node.legal)))
                    prompts[key].append((tree, prefix, node))
        for (bucket, _), states in prompts.items():
            _, prefix, node = states[0]
            returns = []
            for action in node.legal:
                outcomes = [(world.mass, *traverse(world, planner, specific_prefix + (action,)))
                            for world, specific_prefix, _ in states]
                returns.append(weighted((mass, measured["longitudinal"] - WRITE_COST * measured["applied_writes"])
                                        for mass, _, measured in outcomes))
                interactions += sum(len(path) - len(item[1]) for (_, path, _), item in zip(outcomes, states))
            variants = first_variants(node.observation) if not prefix else (node.observation,)
            for observation in variants:
                encoded = encode(tokenizer, observation, node.legal, max_tokens)
                records.append({"record": encoded, "returns": tuple(returns), "bucket": bucket,
                                "root_id": root})

    roots_per_bucket = defaultdict(set)
    counts = Counter((row["bucket"], row["root_id"]) for row in records)
    for bucket, root in counts:
        roots_per_bucket[bucket].add(root)
    if set(roots_per_bucket) != set(MASS):
        raise ValueError("The curriculum needs both first actions and all second-stage strata")
    for row in records:
        bucket = row["bucket"]
        row["weight"] = (MASS[bucket] / len(roots_per_bucket[bucket]) / counts[bucket, row["root_id"]])

    return records, {
        "sha256": digest(records), "examples": dict(Counter(row["bucket"] for row in records)),
        "independent_roots": {bucket: len(roots) for bucket, roots in roots_per_bucket.items()},
        "weight_by_bucket": {bucket: fsum(row["weight"] for row in records if row["bucket"] == bucket)
                             for bucket in MASS},
        "logged_branch_transition_references": interactions, "write_cost_in_training": WRITE_COST,
    }
