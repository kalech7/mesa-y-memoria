"""Matched offline controllers from a single fixed batch of TRAIN transitions."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile

from research.src.artifacts import canonical
from research.tesis_shared_budget_planner import group_trees


VERSION = "tesis_profiles_offline_v1"
SFT = "SFT-ramas-fijas"
RL = "RL-Bellman"
IMMEDIATE = "RL-inmediato"
METHODS = (SFT, RL, IMMEDIATE)


def _key(observation, legal, public_features):
    return canonical((tuple(observation["action_history"]), tuple(legal),
                      public_features(observation, legal)))


@dataclass
class _Action:
    mass: float = 0.
    fixed_return: float = 0.
    terminal_longitudinal: float = 0.
    terminal_immediate: float = 0.
    terminal_mass: float = 0.
    successors: dict = field(default_factory=lambda: defaultdict(float))

    def update(self, weight, fixed_return, successor, terminal):
        self.mass += weight
        self.fixed_return += weight * fixed_return
        if successor is not None:
            self.successors[successor] += weight
        else:
            self.terminal_longitudinal += weight * terminal[0]
            self.terminal_immediate += weight * terminal[1]
            self.terminal_mass += weight

    def fitted_value(self, continuation, *, immediate):
        direct = self.terminal_immediate if immediate else self.terminal_longitudinal
        return (direct + sum(mass * continuation[key] for key, mass in self.successors.items())) / self.mass


@dataclass
class _State:
    depth: int
    legal: tuple
    actions: dict


class Policy:
    """A frozen public policy table with measurable fallback coverage."""

    def __init__(self, table, public_features, fallback):
        self.table = table
        self.public_features = public_features
        self.fallback = fallback
        self._calls = 0
        self._fallbacks = Counter()

    def __call__(self, observation, legal):
        key = _key(observation, legal, self.public_features)
        self._calls += 1
        if key in self.table:
            return self.table[key]["action"]
        self._fallbacks[key] += 1
        return self.fallback(observation, legal)

    def action_values(self, observation, legal):
        """Return per-action values for observed TRAIN keys, or None."""
        row = self.table.get(_key(observation, legal, self.public_features))
        return None if row is None else dict(row["values"])

    def coverage(self, *, reset=False):
        report = {
            "decision_visits": self._calls,
            "fallback_decision_visits": self._fallbacks.total(),
            "unique_unseen_keys": len(self._fallbacks),
        }
        if reset:
            self._calls = 0
            self._fallbacks.clear()
        return report


@dataclass
class Study:
    policies: dict
    telemetry: dict

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "version": VERSION,
            "telemetry": self.telemetry,
            "policies": {method: policy.table for method, policy in self.policies.items()},
        }
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            try:
                file.write(canonical(artifact) + "\n")
                file.flush()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)


def load_policies(path, *, public_features, fallback):
    """Read the fitted public tables without the TRAIN trajectories."""
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("version") != VERSION or set(artifact["policies"]) != set(METHODS):
        raise ValueError("Incompatible controller artifact")
    return {method: Policy(table, public_features, fallback)
            for method, table in artifact["policies"].items()}


def _logged_batch(trees, public_features, fixed_policy, reward_metric, immediate_metric):
    states = {}
    counts = Counter()
    trace_digest = sha256()
    for tree in trees:
        key_for = {}
        fixed_values = {}
        for prefix, node in tree.decisions.items():
            if tuple(node.observation["action_history"]) != prefix:
                raise ValueError("The recorded action path must be recoverable from public history")
            if not node.legal or len(set(node.legal)) != len(node.legal):
                raise ValueError("Each logged decision needs distinct legal actions")
            key_for[prefix] = _key(node.observation, node.legal, public_features)
        for prefix, node in sorted(tree.decisions.items(), key=lambda item: -len(item[0])):
            key, legal = key_for[prefix], tuple(node.legal)
            if key not in states:
                states[key] = _State(len(prefix), legal, dict((action, _Action()) for action in legal))
            state = states[key]
            reference = fixed_policy(node.observation, legal)
            if reference not in legal:
                raise ValueError("The fixed public continuation selected an illegal action")
            for action in legal:
                following = (*prefix, action)
                successor = key_for.get(following)
                terminal = None
                if successor is None:
                    outcome = tree.outcomes[following]
                    terminal = (float(outcome[reward_metric]), float(outcome[immediate_metric]))
                    if not all(map(isfinite, terminal)):
                        raise ValueError("Terminal rewards must be finite")
                    fixed_return = terminal[0]
                    counts["terminal_transition_rows"] += 1
                else:
                    fixed_return = fixed_values[following]
                    counts["cross_visit_transition_rows"] += int(
                        tree.decisions[following].observation["visit"] != node.observation["visit"])
                if action == reference:
                    fixed_values[prefix] = fixed_return
                state.actions[action].update(tree.mass, fixed_return, successor, terminal)
                trace_digest.update(canonical((key, action, successor, terminal, fixed_return, tree.mass)).encode())
                trace_digest.update(b"\n")
                counts["transition_rows"] += 1
            counts["decision_rows"] += 1
    return states, counts, trace_digest.hexdigest()


def fit_from_trees(trees, *, public_features, fixed_policy, reward_metric="longitudinal",
                   immediate_metric="immediate"):
    """Fit three tables with identical logged TRAIN transition exposure.

    SFT labels each public state with the best sibling action's mean terminal
    return under the fixed public continuation. Bellman uses empirical
    transitions and greedily reoptimizes successor choices. The immediate
    ablation changes only the recorded reward, retaining the same horizon,
    transition table and Bellman objective. Both RL tables give intermediate
    transitions zero reward and use undiscounted episode scores at termination.
    """
    trees = tuple(trees)
    if not trees or any(tree.story["split"] != "train" for tree in trees):
        raise ValueError("The controllers may only fit a nonempty TRAIN batch")
    roots = group_trees(trees)
    states, counts, log_digest = _logged_batch(trees, public_features, fixed_policy,
                                              reward_metric, immediate_metric)
    tables = {name: {} for name in METHODS}
    continuation = {name: {} for name in (RL, IMMEDIATE)}
    for key in sorted(states, key=lambda key: -states[key].depth):
        entry = states[key]
        estimates = {
            SFT: {action: row.fixed_return / row.mass for action, row in entry.actions.items()},
            RL: {action: row.fitted_value(continuation[RL], immediate=False)
                 for action, row in entry.actions.items()},
            IMMEDIATE: {action: row.fitted_value(continuation[IMMEDIATE], immediate=True)
                        for action, row in entry.actions.items()},
        }
        for method, values in estimates.items():
            chosen = max(entry.legal, key=lambda action: (round(values[action], 12), action == "PROCEED"))
            tables[method][key] = {"action": chosen, "values": values}
            if method in continuation:
                continuation[method][key] = values[chosen]
    state_action_pairs = sum(len(entry.actions) for entry in states.values())
    telemetry = {
        "train_roots": len(roots), "train_worlds": len(trees),
        "public_states": len(states), "public_state_action_pairs": state_action_pairs,
        "shared_transition_sha256": log_digest,
        "shared_environment_replay_steps": sum(tree.environment_steps for tree in trees),
        **dict(counts),
        "stochastic_state_action_pairs": sum(
            len(row.successors) + bool(row.terminal_mass) > 1
            for entry in states.values() for row in entry.actions.values()),
        "visits_in_train": sorted({node.observation["visit"] for tree in trees for node in tree.decisions.values()}),
        "reward_metrics": {SFT: reward_metric, RL: reward_metric, IMMEDIATE: immediate_metric},
        "arms": {method: {"logged_transition_rows": counts["transition_rows"],
                          "state_action_updates": state_action_pairs,
                          "new_environment_steps": 0, "policy_tokens": 0}
                 for method in METHODS},
    }
    return Study({method: Policy(table, public_features, fixed_policy) for method, table in tables.items()},
                 telemetry)
