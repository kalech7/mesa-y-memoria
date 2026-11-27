"""Matched supervised and GRPO selectors for an interactive memory benchmark."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np
import torch

from research.src.analysis.paired_statistics import paired_bootstrap
from research.src.artifacts import canonical, digest, file_hash, write_json
from research.src.training.episodic_grpo import group_advantages
from research.tesis_interactive_env import ACTIONS, InteractiveEpisode, build_bank, public_evidence, reference_action
from research.tesis_interactive_reward import DEFAULT_CONFIG, aggregate, paired_group, reward


SEEDS = (101, 202, 303)
SFT, MORE, IMMEDIATE, LONG, RULE, NEVER = (
    "SFT", "SFT-adicional", "GRPO-inmediato", "GRPO-longitudinal", "Reglas-observables", "Continuar-siempre")
TRAINED = (MORE, IMMEDIATE, LONG)
METHODS = (SFT, *TRAINED, RULE, NEVER)
TERMINAL = (f"{IMMEDIATE}-terminal", f"{LONG}-terminal")
CLIP = .20
KL_WEIGHT = .02
ROLLOUT_TEMPERATURE = 2.
PRACTICAL_FUTURE_GAIN = .05
MAX_CURRENT_LOSS = .02
FEATURES = (
    "question_budget_remaining", "second_visit", "has_context", "distinct_initial_memory_values",
    "user_turn_count", "current_resolved", "scope_ambiguous", "initial", "partial", "unavailable", "resolved",
    *(f"missing_{action}" for action in ACTIONS), *(f"previous_{action}" for action in ACTIONS[1:]),
)


def features(observation):
    """A small public representation shared by both optimization methods."""
    interpretation = public_evidence(observation)
    asked = observation["asked_slots"]
    query = observation["query"]
    values = {record["value"] for record in observation["memory"] if record["attribute"] == query["attribute"]}
    return torch.tensor((observation["questions_remaining"] / 2, observation["visit"] == 2,
        query.get("context") is not None, min(len(values), 3) / 3,
        min(sum(turn.get("role", "user") == "user" for turn in observation["user_messages"]), 8) / 8,
        bool(interpretation["current_resolved"]), bool(interpretation["scope_ambiguous"]),
        *(interpretation["clarification_state"] == state for state in ("initial", "partial", "unavailable", "resolved")),
        *((interpretation["missing_slot"] or "PROCEED") == action for action in ACTIONS),
        *(action in asked for action in ACTIONS[1:])), dtype=torch.float32)


@dataclass(frozen=True)
class State:
    features: torch.Tensor
    mask: torch.Tensor
    target: int


def frozen_observation(observation):
    """Serialize the actor-facing state while excluding private authoring metadata."""
    public_fields = ("memory", "user_messages", "query", "reuse_regime", "questions_remaining", "asked_slots", "visit")
    return {key: observation[key] for key in public_fields if key in observation}


def state_of(env, observation):
    observation = frozen_observation(observation)
    return _state(canonical(observation), tuple(env.legal_actions))


@lru_cache(maxsize=16384)
def _state(observation_text, actions):
    observation = json.loads(observation_text)
    legal = set(actions)
    mask = torch.tensor([action in legal for action in ACTIONS], dtype=torch.bool)
    if not mask.any():
        raise ValueError("A nonterminal state needs a legal action")
    teacher = reference_action(observation)
    return State(features(observation), mask, ACTIONS.index(teacher if teacher in legal else "PROCEED"))


class Selector(torch.nn.Module):
    """A learned controller on top of public evidence from the existing parser."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(len(FEATURES), 16), torch.nn.Tanh(),
                                          torch.nn.Linear(16, len(ACTIONS)))

    def logprobs(self, states, *, temperature=1.):
        inputs = torch.stack([state.features for state in states])
        masks = torch.stack([state.mask for state in states])
        logits = (self.layers(inputs) / temperature).masked_fill(~masks, -1e9)
        return torch.log_softmax(logits, dim=-1)


def choose_action(policy, state, rng=None):
    if policy == RULE:
        return state.target
    if policy == NEVER:
        return ACTIONS.index("PROCEED")
    with torch.no_grad():
        logp = policy.logprobs([state], temperature=1. if rng is None else ROLLOUT_TEMPERATURE)[0]
        return int(logp.argmax() if rng is None else torch.multinomial(logp.exp(), 1, generator=rng))


def rollout(story, policy, *, database=":memory:", rng=None):
    env = InteractiveEpisode(story, database)
    states, actions = [], []
    try:
        observation = env.observe()
        while observation is not None:
            state = state_of(env, observation)
            action = choose_action(policy, state, rng)
            states.append(state)
            actions.append(action)
            observation, done = env.step(ACTIONS[action])
            if done:
                return states, actions, env.summary()
        raise ValueError("Environment ended without a terminal trace")
    finally:
        env.close()


class TrainingRollouts:
    """Cache deterministic transitions within TRAIN. Evaluation replays SQLite."""

    def __init__(self):
        self.cache = defaultdict(dict)

    def collect(self, story, policy, rng=None):
        if story["split"] != "train":
            raise ValueError("Only TRAIN trajectories may be used by the optimizer")
        key = (story["root_id"], story["world_id"])
        transitions = self.cache[key]
        prefix, states = (), []
        env = None
        try:
            while True:
                if prefix not in transitions:
                    if env is None:
                        env = InteractiveEpisode(story, ":memory:")
                        observation = env.observe()
                        for previous in prefix:
                            observation, done = env.step(ACTIONS[previous])
                        if prefix and done:
                            transitions[prefix] = (None, env.summary())
                            continue
                    transitions[prefix] = (state_of(env, observation), None)
                state, terminal = transitions[prefix]
                if terminal is not None:
                    return states, list(prefix), terminal
                action = choose_action(policy, state, rng)
                prefix += (action,)
                states.append(state)
                if env is not None:
                    observation, done = env.step(ACTIONS[action])
                    if done:
                        transitions.setdefault(prefix, (None, env.summary()))
        finally:
            if env is not None:
                env.close()


def supervised_update(policy, optimizer, states):
    logp = policy.logprobs(states)
    targets = torch.tensor([state.target for state in states], dtype=torch.long)
    loss = -logp.gather(1, targets[:, None]).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def grouped_update(policy, optimizer, parent, rollouts, *, horizon, passes):
    """Apply the clipped per-decision surrogate with one normalized return per trajectory."""
    states, actions, advantages, weights = [], [], [], []
    traces = [trace for _, _, trace in rollouts]
    measured = paired_group(traces)
    returns = [entry["total"] for entry in measured["returns"][horizon]]
    normalized = group_advantages(returns)
    for (trajectory_states, trajectory_actions, _), advantage in zip(rollouts, normalized):
        states.extend(trajectory_states)
        actions.extend(trajectory_actions)
        advantages.extend([advantage] * len(trajectory_states))
        weights.extend([1 / (len(rollouts) * len(trajectory_states))] * len(trajectory_states))
    chosen = torch.tensor(actions, dtype=torch.long)[:, None]
    advantages, weights = torch.tensor(advantages), torch.tensor(weights)
    with torch.no_grad():
        old = policy.logprobs(states, temperature=ROLLOUT_TEMPERATURE).gather(1, chosen).flatten()
        reference = parent.logprobs(states, temperature=ROLLOUT_TEMPERATURE)
    steps = []
    for _ in range(passes):
        current = policy.logprobs(states, temperature=ROLLOUT_TEMPERATURE)
        ratios = (current.gather(1, chosen).flatten() - old).exp()
        surrogate = torch.minimum(ratios * advantages, ratios.clamp(1 - CLIP, 1 + CLIP) * advantages)
        kl = (current.exp() * (current - reference)).sum(dim=1)
        loss = ((-surrogate + KL_WEIGHT * kl) * weights).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
        optimizer.step()
        steps.append({"loss": float(loss.detach()), "gradient_norm": float(gradient)})
    return {"actions": [[ACTIONS[index] for index in actions] for _, actions, _ in rollouts],
            "advantages": normalized, "steps": steps,
            "optimization_task_varies": len({item["task"] for item in measured["returns"][horizon]}) > 1,
            **measured}


def clone(policy):
    result = Selector()
    result.load_state_dict(policy.state_dict())
    return result


def model_artifact(policy):
    return {name: value.tolist() for name, value in policy.state_dict().items()}


def write_rows(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")


def exact_root(stories, policy):
    """Enumerate all legal trajectories for a TRAIN root with a common policy.

    The upper bound assumes knowledge of each latent world. Expected return
    instead uses the selector's public features and a fresh SQLite per leaf.
    """
    if not stories or any(story["split"] != "train" for story in stories):
        raise ValueError("Exact audit accepts TRAIN stories only")
    records = []
    for story in stories:
        if story["root_id"] != stories[0]["root_id"]:
            raise ValueError("Exact audit requires one root")
        pending = [((), 1.)]
        leaves = []
        while pending:
            prefix, probability = pending.pop()
            env = InteractiveEpisode(story, ":memory:")
            try:
                observation = env.observe()
                done = False
                for previous in prefix:
                    observation, done = env.step(ACTIONS[previous])
                if done:
                    trace = env.summary()
                    leaves.append((probability, {horizon: reward(trace, horizon=horizon)["total"]
                                                  for horizon in ("longitudinal", "immediate")}))
                    continue
                state = state_of(env, observation)
                with torch.no_grad():
                    probabilities = policy.logprobs([state], temperature=ROLLOUT_TEMPERATURE)[0].exp().tolist()
                pending.extend((prefix + (index,), probability * probabilities[index])
                               for index, legal in enumerate(state.mask.tolist()) if legal)
            finally:
                env.close()
        if abs(sum(p for p, _ in leaves) - 1.) > 1e-5:
            raise ValueError("Exact action tree lost probability mass")
        records.append({"world_id": story["world_id"], "terminal_trajectories": len(leaves),
                        "expected": {horizon: sum(p * returns[horizon] for p, returns in leaves)
                                     for horizon in ("longitudinal", "immediate")},
                        "world_informed_upper_bound": {horizon: max(returns[horizon] for _, returns in leaves)
                                                       for horizon in ("longitudinal", "immediate")}})
    return {"root_id": stories[0]["root_id"], "temperature": ROLLOUT_TEMPERATURE, "worlds": records,
            "warning": "World-informed upper bound is privileged and not an achievable shared decision rule."}


def evaluate(policy, stories, role, *, disk=False):
    if any(story["split"] != role for story in stories):
        raise ValueError(f"Evaluation expected only {role} stories")
    with TemporaryDirectory(prefix="tesis-selector-eval-") as directory:
        rows = [rollout(story, policy, database=Path(directory) / f"{i}.sqlite" if disk else ":memory:")[2]
                for i, story in enumerate(stories)]
    return aggregate(rows), rows


def rank(summary, step):
    macro = summary["macro"]
    return tuple(round(macro[key], 10) for key in ("longitudinal_return", "strict_joint", "future_supported")) + (-step,)


def passes_guards(candidate, baseline):
    after, before = candidate["macro"], baseline["macro"]
    return (after["current_supported"] >= before["current_supported"] - MAX_CURRENT_LOSS - 1e-10
            and after["unsupported_writes"] <= before["unsupported_writes"] + 1e-10)


def order_train(stories, count, rng):
    """Balance families, then sample roots and worlds uniformly within each."""
    families = defaultdict(lambda: defaultdict(list))
    for story in stories:
        families[story["scenario_family"]][story["root_id"]].append(story)
    names = sorted(families)
    selected = []
    for _ in range(0, count, len(names)):
        for family in rng.permutation(names):
            roots = families[family]
            group = roots[rng.choice(sorted(roots))]
            selected.append(group[int(rng.integers(len(group)))])
    return selected[:count]


def matched_rows(panels, family_by_root):
    return [{"method": method, "seed": seed, "root_id": root, "scenario_family": family_by_root[root],
             "metrics": values} for seed, methods in panels.items() for method, result in methods.items()
            for root, values in result["by_root"].items()]


def contrasts(panels, families):
    rows = matched_rows(panels, families)
    return {f"{LONG} menos {other}": {key: paired_bootstrap(rows, LONG, other, key, confidence=.95, resamples=2000)
            for key in ("future_supported", "longitudinal_return", "current_supported", "strict_joint",
                        "questions", "unsupported_writes")} for other in (IMMEDIATE, SFT, MORE, RULE)}


def train_seed(seed, splits, plan, output, cache):
    output.mkdir()
    started = monotonic()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    settings = plan["settings"]
    train, validation = splits["train"], splits["validation"]
    steps, passes, draws = (settings[key] for key in ("groups", "passes", "draws"))
    warm = Selector()
    demonstrations = [state for episode in train for state in cache.collect(episode, RULE)[0]]
    optimizer = torch.optim.AdamW(warm.parameters(), lr=settings["warmup_learning_rate"], weight_decay=0.)
    for _ in range(settings["warmup_steps"]):
        supervised_update(warm, optimizer, [demonstrations[i] for i in rng.integers(len(demonstrations), size=16)])
    validation_dir = output / "validation"
    validation_dir.mkdir()
    base, base_rows = evaluate(warm, validation, "validation")
    parent_artifact = model_artifact(warm)
    write_json(output / "SFT.json", parent_artifact)
    base_trace = validation_dir / "SFT.jsonl"
    write_rows(base_trace, base_rows)
    schedule = order_train(train, steps, rng)
    write_rows(output / "TRAIN_ORDER.jsonl", ({"root_id": story["root_id"], "world_id": story["world_id"]}
                                                  for story in schedule))
    selected = {SFT: warm, RULE: RULE, NEVER: NEVER}
    terminal = {}
    summaries = {SFT: {"group": 0, "validation": base["macro"], "weights_sha256": digest(parent_artifact),
                       "validation_trace_sha256": file_hash(base_trace)}}
    for method in TRAINED:
        arm_started = monotonic()
        actor, incumbent, best_score = clone(warm), warm, rank(base, 0)
        optimizer = torch.optim.AdamW(actor.parameters(), lr=settings["continuation_learning_rate"], weight_decay=0.)
        rollout_rng = torch.Generator().manual_seed(seed)
        checkpoints = {0: base["macro"]}
        evidence = {0: {"weights_sha256": digest(parent_artifact), "trace_sha256": file_hash(base_trace)}}
        groups = []
        visits = 0
        best_step = 0
        for i, story in enumerate(schedule, 1):
            if method == MORE:
                states, _, trace = cache.collect(story, RULE)
                losses = [supervised_update(actor, optimizer, states) for _ in range(passes)]
                visits += len(trace["visits"])
                group = {"losses": losses}
            else:
                trajectories = [cache.collect(story, actor, rollout_rng) for _ in range(draws)]
                horizon = "immediate" if method == IMMEDIATE else "longitudinal"
                visits += sum(len(trace["visits"]) for _, _, trace in trajectories)
                group = grouped_update(actor, optimizer, warm, trajectories, horizon=horizon, passes=passes)
            groups.append({"group": i, "root_id": story["root_id"], "world_id": story["world_id"], **group})
            if i in plan["checkpoints"]:
                summary, rows = evaluate(actor, validation, "validation")
                checkpoints[i] = summary["macro"]
                path = validation_dir / f"{method}-{i:04d}.jsonl"
                write_rows(path, rows)
                write_json(validation_dir / f"{method}-{i:04d}.json", model_artifact(actor))
                evidence[i] = {"weights_sha256": digest(model_artifact(actor)), "trace_sha256": file_hash(path),
                               "passed_guards": passes_guards(summary, base)}
                current_rank = rank(summary, i)
                if passes_guards(summary, base) and current_rank > best_score:
                    incumbent, best_score, best_step = clone(actor), current_rank, i
        selected[method] = incumbent
        if method != MORE:
            terminal[f"{method}-terminal"] = actor
        write_rows(output / f"{method}-grupos.jsonl", groups)
        artifact = model_artifact(incumbent)
        write_json(output / f"{method}.json", artifact)
        summaries[method] = {"group": best_step, "validation": checkpoints[best_step], "checkpoints": checkpoints,
                             "validation_evidence": evidence,
                             "weights_sha256": digest(artifact), "initial_weights_sha256": digest(parent_artifact),
                             "terminal_weights_sha256": digest(model_artifact(actor)), "groups": steps,
                             "optimizer_steps": passes * steps, "training_visits": visits,
                             "seconds_with_validation": monotonic() - arm_started,
                             "signal": {key: sum(bool(group.get(key)) for group in groups) for key in
                                        ("task_varies", "optimization_task_varies", "only_penalties_vary", "future_varies_with_current_fixed",
                                         "horizon_changes_preference")} if method != MORE else None}
        print(f"{seed} · {method} · selección validación grupo {best_step}", flush=True)
    write_json(output / "TRAINING.json", {"seed": seed, "warmup_demonstration_states": len(demonstrations),
                                          "seconds": monotonic() - started, "methods": summaries})
    return {**selected, **terminal}, summaries


def run(output, *, groups=100, warmup_steps=60, draws=6, passes=2, bank_seed=None):
    if groups < 2 or warmup_steps < 1 or draws < 2 or passes < 1:
        raise ValueError("Training needs positive budgets and at least two grouped draws")
    torch.set_num_threads(1)
    output = Path(output).resolve()
    splits = build_bank(**({} if bank_seed is None else {"seed": bank_seed}))
    families = {story["root_id"]: story["scenario_family"] for story in splits["evaluation"]}
    output.mkdir(parents=True)
    bank = output / "bank"
    bank.mkdir()
    for name, rows in splits.items():
        write_rows(bank / f"{name}.jsonl", rows)
    research = Path(__file__).resolve().parent
    sources = (Path(__file__), research / "tesis_interactive_env.py", research / "tesis_interactive_reward.py",
               research / "tesis_bank.py", research / "tesis_memory_guard.py", research / "src/env/episode.py",
               research / "src/memory/executor.py", research / "src/memory/store.py", research / "src/readers/symbolic.py",
               research / "src/policy/rule_controller.py", research / "src/policy/dialogue.py",
               research / "src/scoring/evidence.py", research / "src/training/episodic_grpo.py")
    plan = {"scope": "exploratory_compact_selector_multi_decision_not_Qwen_LoRA", "bank_seed": bank_seed,
            "runtime": {"device": "cpu", "torch": torch.__version__, "numpy": np.__version__, "threads": 1},
            "seeds": SEEDS, "methods": METHODS, "terminal_diagnostic": TERMINAL,
            "features": FEATURES, "actions": ACTIONS,
            "architecture": "16-unit tanh network on public features, including missing-slot and ambiguity from the shared rule parser",
            "training_comparison": "Matched group starts and optimizer steps. GRPO uses more interaction feedback and may consume more wall time.",
            "reward": asdict(DEFAULT_CONFIG), "clip": CLIP, "kl_to_warmup": KL_WEIGHT,
            "rollout_temperature": ROLLOUT_TEMPERATURE, "evaluation_action": "legal argmax",
            "settings": {"groups": groups, "warmup_steps": warmup_steps, "draws": draws, "passes": passes,
                         "warmup_learning_rate": .012, "continuation_learning_rate": .008},
            "checkpoints": sorted({groups // 4, groups // 2, groups} - {0}),
            "exact_training_diagnostic": "Enumerate all legal paths for lexicographically first F4 TRAIN root and seed 101.",
            "selection": ["longitudinal_return", "strict_joint", "future_supported", "earliest_group"],
            "primary_contrast": "GRPO-longitudinal minus GRPO-inmediato in held-out supported future accuracy",
            "practical_future_gain": PRACTICAL_FUTURE_GAIN,
            "guard_metrics": {"max_current_loss": MAX_CURRENT_LOSS, "max_increase_unsupported_writes": 0.},
            "split_sha256": {key: digest(value) for key, value in splits.items()},
            "source_sha256": {str(path): file_hash(path) for path in sources}}
    plan_path = output / "PLAN.json"
    write_json(plan_path, plan)
    selections, policies = {}, {}
    cache = TrainingRollouts()
    for seed in SEEDS:
        policies[seed], selections[seed] = train_seed(seed, splits, plan, output / str(seed), cache)
    if any(file_hash(path) != checksum for path, checksum in plan["source_sha256"].items()):
        raise ValueError("Sources changed while training")
    selection_path = output / "SELECTION.json"
    write_json(selection_path, {"split": "validation", "selected_before_evaluation": True,
                                "evaluation_sha256": plan["split_sha256"]["evaluation"],
                                "plan_sha256": file_hash(plan_path), "selections": selections})
    selected_diagnostic_root = min(story["root_id"] for story in splits["train"] if story["scenario_family"] == "F4")
    diagnostic_worlds = [story for story in splits["train"] if story["root_id"] == selected_diagnostic_root]
    exact = exact_root(diagnostic_worlds, policies[SEEDS[0]][LONG])
    write_json(output / "TRAIN_EXACT.json", exact)
    panels = defaultdict(dict)
    for seed, methods in policies.items():
        for method, policy in methods.items():
            summary, rows = evaluate(policy, splits["evaluation"], "evaluation", disk=True)
            path = output / str(seed) / f"{method}-evaluacion.jsonl"
            write_rows(path, rows)
            panels[seed][method] = {**summary, "traces": str(path.relative_to(output)), "trace_sha256": file_hash(path)}
    comparisons = contrasts(panels, families)
    primary = comparisons[f"{LONG} menos {IMMEDIATE}"]
    terminal_rows = matched_rows(panels, families)
    terminal_effects = {key: paired_bootstrap(terminal_rows, TERMINAL[1], TERMINAL[0], key,
                                              confidence=.95, resamples=2000)
                        for key in ("future_supported", "longitudinal_return", "current_supported", "strict_joint")}
    decision = {"practical_future_gain": primary["future_supported"]["estimate"] >= PRACTICAL_FUTURE_GAIN,
                "paired_interval_above_zero": primary["future_supported"]["interval"][0] > 0,
                "current_accuracy_guard": primary["current_supported"]["estimate"] >= -MAX_CURRENT_LOSS - 1e-10,
                "unsupported_write_guard": primary["unsupported_writes"]["estimate"] <= 1e-10}
    result = {"status": "completed_exploratory", "scope": plan["scope"], "seeds": SEEDS,
              "evaluation_roots": len(families), "evaluation_episodes_per_seed": len(splits["evaluation"]),
              "selection_sha256": file_hash(selection_path), "panels": panels,
              "means": {method: {key: mean(panel[method]["macro"][key] for panel in panels.values())
                                 for key in panels[SEEDS[0]][method]["macro"]} for method in METHODS},
              "contrasts": comparisons, "terminal_contrasts_longitudinal_minus_immediate": terminal_effects,
              "primary_criteria": {**decision, "all_met": all(decision.values())}}
    write_json(output / "RESULTS.json", result)
    return result


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def audit_run(output):
    """Re-score archived traces and verify matched budgets and validation-only selection."""
    output = Path(output).resolve()
    plan_path, selection_path = output / "PLAN.json", output / "SELECTION.json"
    plan = json.loads(plan_path.read_bytes())
    selection = json.loads(selection_path.read_bytes())
    result = json.loads((output / "RESULTS.json").read_bytes())
    if (selection["plan_sha256"] != file_hash(plan_path) or result["selection_sha256"] != file_hash(selection_path)
            or selection["split"] != "validation" or not selection["selected_before_evaluation"]
            or selection["evaluation_sha256"] != plan["split_sha256"]["evaluation"]):
        raise ValueError("Validation lock or reserved hash mismatch")
    if any(file_hash(path) != checksum for path, checksum in plan["source_sha256"].items()):
        raise ValueError("Implementation changed since the experiment was frozen")
    splits = {name: read_jsonl(output / "bank" / f"{name}.jsonl") for name in plan["split_sha256"]}
    if any(digest(rows) != plan["split_sha256"][name] for name, rows in splits.items()):
        raise ValueError("Archived bank changed")
    panel_count = groups_checked = trace_count = 0
    for seed in plan["seeds"]:
        root = output / str(seed)
        training = selection["selections"][str(seed)]
        parent = json.loads((root / "SFT.json").read_bytes())
        baseline_path = root / "validation/SFT.jsonl"
        if file_hash(baseline_path) != training[SFT]["validation_trace_sha256"]:
            raise ValueError("Parent validation trace changed")
        base = aggregate(read_jsonl(baseline_path))
        if base["macro"] != training[SFT]["validation"] or digest(parent) != training[SFT]["weights_sha256"]:
            raise ValueError("Parent validation outcome or weights changed")
        schedule = read_jsonl(root / "TRAIN_ORDER.jsonl")
        for method in TRAINED:
            arm = training[method]
            if arm["initial_weights_sha256"] != digest(parent):
                raise ValueError("Continuation did not start from the common checkpoint")
            ledger = read_jsonl(root / f"{method}-grupos.jsonl")
            steps_key = "losses" if method == MORE else "steps"
            if (len(ledger) != plan["settings"]["groups"] or
                    sum(len(group[steps_key]) for group in ledger) != arm["optimizer_steps"] or
                    len(schedule) != len(ledger)):
                raise ValueError("Matched training schedule or optimizer budget changed")
            for expected, group in zip(schedule, ledger):
                if any(group[key] != value for key, value in expected.items()):
                    raise ValueError("A continuation differs from the predeclared TRAIN schedule")
                if method != MORE:
                    horizon = "immediate" if method == IMMEDIATE else "longitudinal"
                    if (len(group["actions"]) != plan["settings"]["draws"] or group["advantages"] !=
                            group_advantages([row["total"] for row in group["returns"][horizon]])):
                        raise ValueError("GRPO group size or group-relative advantages changed")
                    groups_checked += 1
            candidates = {0: base}
            for step in plan["checkpoints"]:
                trace = root / "validation" / f"{method}-{step:04d}.jsonl"
                weights = root / "validation" / f"{method}-{step:04d}.json"
                evidence = arm["validation_evidence"][str(step)]
                if (file_hash(trace) != evidence["trace_sha256"] or
                        digest(json.loads(weights.read_bytes())) != evidence["weights_sha256"]):
                    raise ValueError("Validation checkpoint evidence changed")
                summary = aggregate(read_jsonl(trace))
                if summary["macro"] != arm["checkpoints"][str(step)]:
                    raise ValueError("Archived validation outcome changed")
                if passes_guards(summary, base):
                    candidates[step] = summary
            winner = max(candidates, key=lambda step: rank(candidates[step], step))
            chosen = json.loads((root / f"{method}.json").read_bytes())
            if (winner != arm["group"] or digest(chosen) != arm["weights_sha256"] or
                    digest(chosen) != arm["validation_evidence"][str(winner)]["weights_sha256"]):
                raise ValueError("Selected checkpoint was not the validation winner")
        expected = [(row["root_id"], row["world_id"]) for row in splits["evaluation"]]
        for method in (*METHODS, *TERMINAL):
            panel = result["panels"][str(seed)][method]
            path = output / panel["traces"]
            rows = read_jsonl(path)
            if (file_hash(path) != panel["trace_sha256"] or any(row["split"] != "evaluation" for row in rows)
                    or [(row["root_id"], row["world_id"]) for row in rows] != expected
                    or aggregate(rows) != {key: panel[key] for key in ("roots", "episodes", "by_root", "by_family", "macro")}):
                raise ValueError("Reserved evaluation trace or aggregate changed")
            panel_count += 1
            trace_count += len(rows)
    return {"status": "verified", "evaluation_panels": panel_count, "trajectories_scored": trace_count,
            "grpo_groups_verified": groups_checked, "seeds": plan["seeds"],
            "selection_sha256": file_hash(selection_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bank-seed", type=int)
    parser.add_argument("--groups", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument("--draws", type=int, default=6)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--audit", action="store_true", help="Verify a completed run under --output")
    args = parser.parse_args()
    if args.audit:
        print(canonical(audit_run(args.output)))
        return
    result = run(args.output, groups=args.groups, warmup_steps=args.warmup_steps, draws=args.draws,
                 passes=args.passes, bank_seed=args.bank_seed)
    print(canonical({"status": result["status"], "means": result["means"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
