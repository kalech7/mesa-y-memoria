"""Matched offline policy updates on public decision groups from a TRAIN tree."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite

import torch


PARENT = "SFT-inicial"
SUPERVISED = "SFT-ramas"
IMMEDIATE = "GRPO-sin-terminal"
LONGITUDINAL = "GRPO-longitudinal"
ARMS = (SUPERVISED, IMMEDIATE, LONGITUDINAL)
TRAINING = {
    "epochs": 64, "warmup_epochs": 24, "passes_per_epoch": 2,
    "learning_rate": .025, "clip": .20, "kl_weight": .02,
    "hidden_features": 16,
}


def _features(public_state):
    # Positions distinguish meanings without providing identifiers to the policy.
    return tuple((index, value) for index, value in enumerate(public_state))


class Policy(torch.nn.Module):
    """A small masked selector with no tokenizer and only public state inputs."""

    def __init__(self, actions, vocabulary, state):
        super().__init__()
        self.actions = tuple(actions)
        self.vocabulary = {feature: index for index, feature in enumerate(vocabulary)}
        self.state = state
        hidden = TRAINING["hidden_features"]
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(len(vocabulary), hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, len(actions)),
        )

    def encode(self, observation, legal):
        values = torch.zeros(len(self.vocabulary))
        for feature in _features(self.state(observation, legal)):
            index = self.vocabulary.get(feature)
            if index is not None:
                values[index] = 1.
        mask = torch.tensor([action in legal for action in self.actions])
        if not bool(mask.any()):
            raise ValueError("Policy requires a known legal action")
        return values, mask

    def logprobs(self, values, legal):
        return self.layers(values).masked_fill(~legal, -1e9).log_softmax(dim=-1)

    def action(self, observation, legal):
        values, mask = self.encode(observation, legal)
        with torch.no_grad():
            index = int(self.logprobs(values, mask).argmax())
        return self.actions[index]

    def artifact(self):
        return {
            "actions": self.actions,
            "vocabulary": list(self.vocabulary),
            "parameters": {key: value.tolist() for key, value in self.state_dict().items()},
        }


@dataclass(frozen=True)
class TrainingBatch:
    values: torch.Tensor
    legal: torch.Tensor
    teacher: torch.Tensor
    longitudinal: torch.Tensor
    immediate: torch.Tensor
    weights: torch.Tensor


def _validate(groups, actions):
    keys = set()
    if not groups or len(actions) < 2 or len(actions) != len(set(actions)):
        raise ValueError("Training requires nonempty groups and a unique action vocabulary")
    for group in groups:
        if group.get("split") != "train":
            raise ValueError("Only TRAIN groups may reach the optimizer")
        key = (group["root_id"], group["state_key"])
        if key in keys:
            raise ValueError("Merge indistinguishable worlds before submitting a TRAIN state")
        keys.add(key)
        legal = group["legal"]
        if (len(legal) < 2 or len(legal) != len(set(legal)) or
                not set(legal).issubset(actions) or group["reference"] not in legal):
            raise ValueError("Each group needs unique legal alternatives and a legal reference action")
        for horizon in ("longitudinal", "immediate"):
            returns = group[horizon]
            if len(returns) != len(legal) or not all(isfinite(value) for value in returns):
                raise ValueError("Every legal action needs finite returns from both horizons")
        if type(group["interactions"]) is not int or group["interactions"] < len(legal):
            raise ValueError("Account for the transition references used by every legal branch")


def _batch(policy, groups):
    encoded = [policy.encode(group["observation"], group["legal"]) for group in groups]
    values, legal = map(torch.stack, zip(*encoded))
    action_ids = {action: index for index, action in enumerate(policy.actions)}
    teacher = torch.tensor([action_ids[group["reference"]] for group in groups])
    returns = {}
    for horizon in ("longitudinal", "immediate"):
        rewards = torch.zeros_like(legal, dtype=torch.float32)
        for index, group in enumerate(groups):
            alternatives = [action_ids[action] for action in group["legal"]]
            rewards[index, alternatives] = torch.tensor(group[horizon], dtype=rewards.dtype)
        returns[horizon] = rewards
    counts = Counter(group["root_id"] for group in groups)
    weights = torch.tensor([1. / counts[group["root_id"]] for group in groups])
    return TrainingBatch(values, legal, teacher, returns["longitudinal"], returns["immediate"],
                         weights / weights.sum())


def _optimize(policy, optimizer, loss):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
    optimizer.step()


def _warmup(policy, batch, epochs):
    optimizer = torch.optim.Adam(policy.parameters(), lr=TRAINING["learning_rate"])
    for _ in range(epochs):
        logp = policy.logprobs(batch.values, batch.legal)
        chosen = logp.gather(1, batch.teacher[:, None]).squeeze(1)
        _optimize(policy, optimizer, -batch.weights @ chosen)


def _branch_step(policy, optimizer, batch, returns, anchor, *, supervised):
    if supervised:
        masked_returns = returns.masked_fill(~batch.legal, -torch.inf)
        best = torch.isclose(masked_returns, masked_returns.max(dim=1, keepdim=True).values,
                             rtol=0., atol=1e-6) & batch.legal
    else:
        with torch.no_grad():
            old = policy.logprobs(batch.values, batch.legal)
            mass = old.exp()
            # As in the existing all-action branched GRPO, center by behavior mass
            # and retain reward magnitude across groups with unequal practical stakes.
            advantages = returns - (mass * returns).sum(dim=1, keepdim=True)
    for _ in range(TRAINING["passes_per_epoch"]):
        now = policy.logprobs(batch.values, batch.legal)
        if supervised:
            loss_per_group = -now.masked_fill(~best, 0.).sum(dim=1) / best.sum(dim=1)
        else:
            ratio = (now - old).exp()
            clip = TRAINING["clip"]
            objective = torch.minimum(ratio * advantages, ratio.clamp(1. - clip, 1. + clip) * advantages)
            kl = (now.exp() * (now - anchor)).sum(dim=1)
            loss_per_group = -(mass * objective).sum(dim=1) + TRAINING["kl_weight"] * kl
        _optimize(policy, optimizer, batch.weights @ loss_per_group)


def fit(groups, seed, *, actions, state, validation=None, epochs=TRAINING["epochs"],
        warmup_epochs=TRAINING["warmup_epochs"]):
    """Fit matched selectors. Validation, if supplied, ranks independent checkpoints.

    Each return vector comes from sibling actions on the same public decision.
    The immediate ablation removes only terminal reward from the logged
    continuations. Training does not request any more simulator trajectories.
    The caller supplies a validation function returning a comparable tuple,
    with larger values preferred. Ties select the earliest checkpoint.
    """
    if type(epochs) is not int or epochs < 1 or type(warmup_epochs) is not int or warmup_epochs < 0:
        raise ValueError("Training requires positive continuation epochs and nonnegative warmup epochs")
    groups = tuple(groups)
    actions = tuple(actions)
    _validate(groups, actions)
    vocabulary = dict.fromkeys(
        feature for group in groups for feature in _features(state(group["observation"], group["legal"]))
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        parent = Policy(actions, vocabulary, state)
    batch = _batch(parent, groups)
    _warmup(parent, batch, warmup_epochs)
    with torch.no_grad():
        anchor = parent.logprobs(batch.values, batch.legal)
    root_count = len({group["root_id"] for group in groups})
    accounting = {
        "seed": seed,
        "train_roots": root_count,
        "public_decision_groups": len(groups),
        "logged_action_returns": sum(len(group["legal"]) for group in groups),
        "shared_transition_references": sum(group["interactions"] for group in groups),
        "shared_parent_updates": warmup_epochs,
        "shared_parent_observations_scored": len(groups) * warmup_epochs,
        "checkpoint_candidates": sorted({0, epochs // 2, epochs}),
        "settings": dict(TRAINING, epochs=epochs, warmup_epochs=warmup_epochs),
        "arms": {},
    }
    parent_score = None if validation is None else tuple(validation(parent.action))
    policies = {PARENT: parent}
    for arm in ARMS:
        policy = deepcopy(parent)
        optimizer = torch.optim.Adam(policy.parameters(), lr=TRAINING["learning_rate"])
        immediate = arm == IMMEDIATE
        returns = batch.immediate if immediate else batch.longitudinal
        best_score = parent_score
        selected = parent
        selected_epoch = 0
        scores = {} if validation is None else {0: parent_score}
        for epoch in range(1, epochs + 1):
            _branch_step(policy, optimizer, batch, returns, anchor, supervised=arm == SUPERVISED)
            if validation is not None and epoch in accounting["checkpoint_candidates"]:
                score = tuple(validation(policy.action))
                scores[epoch] = score
                if score[0] and score > best_score:
                    selected = deepcopy(policy)
                    best_score, selected_epoch = score, epoch
        if validation is None:
            selected, selected_epoch = policy, epochs
        updates = epochs * TRAINING["passes_per_epoch"]
        policies[arm] = selected
        accounting["arms"][arm] = {
            "selected_epoch": selected_epoch, "validation_ranks": scores,
            "additional_optimizer_updates": updates,
            "update_observations_scored": updates * len(groups),
            "old_policy_observations_scored": 0 if arm == SUPERVISED else epochs * len(groups),
            "online_environment_interactions": 0,
            "policy_tokens": 0,
        }
    return policies, accounting
