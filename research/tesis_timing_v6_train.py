"""Train Qwen on matched action branches continued by its evolving selector."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import fcntl
import gzip
from io import TextIOWrapper
import json
import math
from pathlib import Path
import random

from research.src.artifacts import canonical, digest, file_hash
from research.tesis_extended_experiment import MODEL, PROJECT, open_model
from research.tesis_profiles_optimizer_v5 import TEMPERATURE_V5
from research.tesis_shared_budget_qwen import atomic_json, batches, checkpoint, padded_scoring
from research.tesis_shared_budget_qwen_matched import GRPO, SFT, branch_loss, load_adapter

PARENT = PROJECT / "reports/tesis_profiles_qwen_a100_v5/seed-101/common-parent/checkpoint-000/adapter"
RL, IMMEDIATE = "GRPO-continuacion", "GRPO-inmediato"
FIXED = "GRPO-continuacion-fija"
ARMS = (SFT, RL, IMMEDIATE, FIXED)
OPTIONS = ("WAIT", "ASK")
# Columns A and B retain the meanings used by the inherited Qwen adapter.
MODEL_ACTIONS = ("PROCEED", "TEMPORAL_SCOPE")
T = TEMPERATURE_V5
SAMPLES = 8
LR = 1e-4
PASSES = 2
MAX_TOKENS = 2048
SYSTEM = (
    "Eres responsable de las aclaraciones de un asistente en español. Lee la conversación, "
    "la agenda pública, la memoria confirmada y el presupuesto compartido entre visitas. "
    "A. Responder con la evidencia disponible y reservar la pregunta. "
    "B. Hacer la pregunta disponible ahora para precisar la preferencia. "
    "Responde con una sola letra."
)


def messages(text):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


class MarginScorer:
    """Serve the two actual action logits with identical formatting in TRAIN and evaluation."""

    def __init__(self, batch_size=8, *, trainable=False, base=MODEL, parent=PARENT):
        self.model, self.tokenizer, self.labels = open_model(base, parent, trainable=trainable)
        self.batch_size = batch_size
        self.active = Path(parent).resolve()
        self.encodings, self.cache = {}, {}
        self.count = Counter()

    def load(self, adapter_path):
        path = Path(adapter_path).resolve()
        if path != self.active:
            load_adapter(self.model, path)
            self.active = path
            self.cache.clear()

    def record(self, text):
        if text not in self.encodings:
            ids = self.tokenizer.apply_chat_template(messages(text), tokenize=True, add_generation_prompt=True)
            if not 0 < len(ids) <= MAX_TOKENS:
                raise ValueError(f"V6 public prompt length {len(ids)} exceeds the limit of {MAX_TOKENS}")
            self.encodings[text] = (ids, MODEL_ACTIONS)
        return self.encodings[text]

    def score_texts(self, texts):
        import torch

        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        self.model.eval()
        with torch.inference_mode():
            for group in batches(missing, self.batch_size):
                logits = padded_scoring(self.model, [self.record(text) for text in group], self.labels,
                                        self.tokenizer.pad_token_id, "cuda", self.count)
                margins = (logits[:, 1] - logits[:, 0]).tolist()
                if not all(math.isfinite(value) for value in margins):
                    raise ValueError("Nonfinite action margin")
                self.cache.update(zip(group, margins, strict=True))
        return [self.cache[text] for text in texts]

    def margins(self, observations):
        from research.tesis_timing_v6_env import prompt_text

        return self.score_texts([prompt_text(observation) for observation in observations])


@dataclass
class Start:
    tree: object
    prefix: tuple


@dataclass
class Group:
    root: str
    budget: int
    visit: int
    public: str
    starts: list
    weight: float


def train_groups(trees_by_budget, render):
    grouped = defaultdict(list)
    for budget, trees in trees_by_budget.items():
        for tree in trees:
            if tree.story["split"] != "train":
                raise ValueError("The trajectory collector may only access TRAIN worlds")
            for prefix, node in tree.decisions.items():
                if len(node.legal) == 1:
                    continue
                if set(node.legal) != set(OPTIONS):
                    raise ValueError("V6 requires two actions at each unconstrained decision")
                key = (tree.story["root_id"], budget, len(prefix) + 1, render(node.observation))
                grouped[key].append(Start(tree, prefix))
    # Give each root, budget and visit equal total loss weight, then split that
    # weight by the prior world mass of the compatible public histories.
    totals = Counter()
    for (root, budget, visit, _), starts in grouped.items():
        totals[root, budget, visit] += sum(item.tree.mass for item in starts)
    return [Group(root, budget, visit, text, starts, sum(s.tree.mass for s in starts) / totals[root, budget, visit])
            for (root, budget, visit, text), starts in sorted(grouped.items())]


def ask_probability(margin):
    scaled = margin / T
    if scaled >= 0:
        return 1 / (1 + math.exp(-scaled))
    power = math.exp(scaled)
    return power / (1 + power)


def uniforms(seed, world_id, budget, depth, size):
    # The same stratified uniform draws couple both counterfactual branches
    # and all rounds. Changes across rounds hence come from the new Qwen logits.
    key = digest([seed, world_id, budget, depth])
    offset = random.Random(int(key[:16], 16)).random()
    return [(sample + offset) / size for sample in range(size)]


def collect(groups, scores, render, *, seed, samples=SAMPLES, write_trace=None):
    """Force each action once, then follow this Qwen checkpoint to terminal in every world."""
    if samples < 1:
        raise ValueError("Need at least one continuation sample")
    rows, counts = [], Counter()
    sampled_actions = Counter()
    for group in groups:
        rewards, immediate = [], []
        for action in OPTIONS:
            total = sum(start.tree.mass for start in group.starts)
            longitudinal = current = 0.
            for start in group.starts:
                tree, prefix = start.tree, start.prefix
                draws = uniforms(seed, tree.story["world_id"], group.budget, len(prefix), samples)
                paths = Counter()
                for draw in draws:
                    path = prefix + (action,)
                    while path in tree.decisions:
                        node = tree.decisions[path]
                        if len(node.legal) == 1:
                            next_action = node.legal[0]
                            counts["forced_continuation_decisions"] += 1
                        else:
                            # V3 is the terminal memory test. Each branch has
                            # at most one subsequent choice in this simulator.
                            if len(path) != len(prefix) + 1:
                                raise ValueError("A longer horizon needs fresh draws at each future visit")
                            next_action = "ASK" if draw < ask_probability(scores[render(node.observation)]) else "WAIT"
                            sampled_actions[next_action] += 1
                            counts["qwen_continuation_decisions"] += 1
                        path += (next_action,)
                    leaf = tree.outcomes[path]
                    mass = tree.mass / (total * samples)
                    longitudinal += mass * leaf["longitudinal"]
                    metric = "current_first_supported" if group.visit == 1 else "current_second_supported"
                    current += mass * (leaf[metric] - .08 * (action == "ASK"))
                    counts["terminal_rollouts"] += 1
                    counts["replayed_decision_transitions"] += len(path) - len(prefix)
                    paths[path] += 1
                if write_trace:
                    write_trace({"root": group.root, "world": tree.story["world_id"], "mass": tree.mass,
                                 "budget": group.budget, "visit": group.visit, "prefix": prefix,
                                 "public_sha256": digest(group.public), "forced_action": action,
                                 "paths": [{"actions": path, "samples": n, "terminal": tree.outcomes[path]}
                                           for path, n in sorted(paths.items())]})
            rewards.append(longitudinal)
            immediate.append(current)
        rows.append({"root": group.root, "budget": group.budget, "visit": group.visit,
                     "public": group.public, "weight": group.weight,
                     "returns": rewards, "immediate_returns": immediate})
    return rows, {**counts, "sampled_continuation_actions": dict(sampled_actions)}


def score_map(scorer, texts, path):
    scorer.load(path)
    counts_before = scorer.count.copy()
    vals = scorer.score_texts(texts)
    return dict(zip(texts, vals, strict=True)), dict(scorer.count - counts_before)


def log_probs(margins, device):
    import torch

    margin = torch.tensor(margins, dtype=torch.float32, device=device)
    return torch.stack((torch.zeros_like(margin), margin / T), dim=-1).log_softmax(-1)


def fit_arm(scorer, optimizer, rows, order, arm, anchor, old, batch_size):
    import torch

    total_weight = sum(row["weight"] for row in rows) / len(rows)
    counts = Counter()
    loss_sum = 0.
    scorer.model.train()
    scorer.active = None
    scorer.cache.clear()
    for indices in batches(order, batch_size):
        subset = [rows[i] for i in indices]
        weights = torch.tensor([r["weight"] / total_weight for r in subset], dtype=torch.float32, device="cuda")
        key = "immediate_returns" if arm == IMMEDIATE else "returns"
        returns = torch.tensor([r[key] for r in subset], dtype=torch.float32, device="cuda")
        legal = torch.ones_like(returns, dtype=torch.bool)
        records = [scorer.record(r["public"]) for r in subset]
        old_logp = log_probs([old[r["public"]] for r in subset], "cuda")
        ref_logp = log_probs([anchor[r["public"]] for r in subset], "cuda")
        for _ in range(PASSES):
            logits = padded_scoring(scorer.model, records, scorer.labels, scorer.tokenizer.pad_token_id,
                                    "cuda", counts)[:, :2]
            losses = (branch_loss(SFT, logits, legal, returns) if arm == SFT else
                      branch_loss(GRPO, logits / T, legal, returns, old=old_logp, anchor=ref_logp))
            loss = (weights * losses).mean()
            if not bool(loss.isfinite()):
                raise ValueError(f"Nonfinite loss for {arm}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            counts["optimizer_updates"] += 1
            loss_sum += float(loss.detach())
    return {**counts, "mean_loss": loss_sum / counts["optimizer_updates"]}


def parameters(run, seed, arm, stage):
    return (run / f"seed-{seed}" / arm / f"checkpoint-{stage:03d}" / "adapter") if stage else PARENT


def read(path):
    return json.loads(path.read_text())


def collect_and_save(directory, groups, scores, render, seed):
    branch_path = directory / "BRANCHES.json"
    trace_path = directory / "TRACES.jsonl.gz"
    trace_temp = trace_path.with_suffix(".tmp")
    with trace_temp.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=3) as compressed:
            with TextIOWrapper(compressed, encoding="utf-8") as trace:
                rows, counts = collect(groups, scores, render, seed=seed,
                                       write_trace=lambda item: trace.write(canonical(item) + "\n"))
    trace_temp.replace(trace_path)
    atomic_json(branch_path, rows)
    return rows, {"trajectory_counts": counts, "groups": len(rows),
                  "shared_rows_sha256": file_hash(branch_path), "traces_sha256": file_hash(trace_path),
                  "shared_public_sha256": digest([r["public"] for r in rows]),
                  "longitudinal_return_sha256": digest([r["returns"] for r in rows]),
                  "immediate_return_sha256": digest([r["immediate_returns"] for r in rows])}


def fit_round(scorer, groups, texts, anchor, render, run, seed, stage, batch_size, resume):
    import torch

    directory = run / f"seed-{seed}" / f"round-{stage:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    collector_path = parameters(run, seed, RL, stage - 1)
    collector_hash = file_hash(collector_path / "adapter_model.safetensors")
    complete = directory / "RESULT.json"
    if complete.exists():
        if not resume:
            raise FileExistsError(complete)
        result = read(complete)
        if result["collector"]["sha256"] != collector_hash:
            raise ValueError("Previous GRPO checkpoint changed")
        for name, checksum in (("BRANCHES.json", "shared_rows_sha256"), ("TRACES.jsonl.gz", "traces_sha256")):
            if file_hash(directory / name) != result["shared"][checksum]:
                raise ValueError("Published round data changed")
        for arm, checkpoint_data in result["arms"].items():
            path = parameters(run, seed, arm, stage)
            if file_hash(path / "adapter_model.safetensors") != checkpoint_data["weights"]:
                raise ValueError("Published adapter changed")
        return result
    full, collection_tokens = score_map(scorer, texts, collector_path)
    rows, shared = collect_and_save(directory, groups, full, render, seed)
    initial_data = run / f"seed-{seed}" / "round-001" / "BRANCHES.json"
    initial_rows = rows if stage == 1 else read(initial_data)
    if [(r["public"], r["weight"]) for r in initial_rows] != [(r["public"], r["weight"]) for r in rows]:
        raise ValueError("Fixed and refreshed continuation data cover different decisions")
    fixed_hash = file_hash(initial_data)
    parent_hash = file_hash(PARENT / "adapter_model.safetensors")
    order = list(range(len(rows)))
    random.Random(seed + stage).shuffle(order)
    result = {"seed": seed, "round": stage, "collector": {
        "method": RL, "checkpoint": str(collector_path), "sha256": collector_hash,
        "temperature": T, "scored_public_nodes": len(texts), "inference": collection_tokens},
        "shared": shared, "schedule_sha256": digest(order), "arms": {}}
    for arm in ARMS:
        previous = parameters(run, seed, arm, stage - 1)
        current = parameters(run, seed, arm, stage)
        ledger = current.parent / "LEDGER.json"
        observations = initial_rows if arm == FIXED else rows
        row_digest = fixed_hash if arm == FIXED else shared["shared_rows_sha256"]
        continuation_hash = parent_hash if arm == FIXED else collector_hash
        print(f"seed={seed} round={stage} arm={arm} groups={len(rows)}", flush=True)
        if ledger.exists():
            if not resume:
                raise FileExistsError(ledger)
            update = read(ledger)[0]
            if (update["shared_rows_sha256"] != row_digest or update["collector_sha256"] != continuation_hash
                    or update["schedule_sha256"] != result["schedule_sha256"]):
                raise ValueError("Resumed checkpoint belongs to another trajectory set")
        else:
            scorer.load(previous)
            if arm in (IMMEDIATE, FIXED):
                old, behavior_tokens = score_map(scorer, texts, previous)
            else:
                old, behavior_tokens = full, {}
            optimizer = torch.optim.AdamW([p for p in scorer.model.parameters() if p.requires_grad], lr=LR)
            if stage > 1:
                optimizer.load_state_dict(torch.load(previous.parent / "OPTIMIZER.pt", map_location="cpu", weights_only=True))
            torch.manual_seed(seed + stage)
            update = {**fit_arm(scorer, optimizer, observations, order, arm, anchor, old, batch_size),
                      "shared_rows_sha256": row_digest, "collector_sha256": continuation_hash,
                      "schedule_sha256": result["schedule_sha256"], "behavior_inference": behavior_tokens}
            arm_directory = current.parent.parent
            arm_directory.mkdir(parents=True, exist_ok=True)
            checkpoint(arm_directory, stage, scorer.model, optimizer, [update])
            del optimizer
        result["arms"][arm] = {**update, "adapter": str(current.relative_to(run)),
                               "weights": file_hash(current / "adapter_model.safetensors"),
                               "config": file_hash(current / "adapter_config.json")}
    counters = ("optimizer_updates", "input_tokens", "padded_input_tokens", "decisions", "forward_batches")
    if len({tuple(row[key] for key in counters) for row in result["arms"].values()}) != 1:
        raise ValueError("The arms received unequal optimization budgets")
    atomic_json(complete, result)
    return result


def train(args):
    from importlib.metadata import version
    from research.tesis_timing_v6_bank import build_worlds
    from research.tesis_timing_v6_env import enumerate_world, prompt_text
    import torch

    run = args.run.resolve()
    worlds = build_worlds("train")
    plan = {"seeds": args.seeds, "rounds": args.rounds, "batch_size": args.batch_size, "passes": PASSES,
            "learning_rate": LR, "temperature": T, "samples_per_forced_action_world": SAMPLES,
            "arms": ARMS, "budgets": [1, 2], "train_sha256": digest(worlds), "parent": str(PARENT),
            "parent_sha256": file_hash(PARENT / "adapter_model.safetensors"),
            "parent_config_sha256": file_hash(PARENT / "adapter_config.json"),
            "model": str(MODEL), "model_files": {p.name: file_hash(p) for p in sorted(MODEL.iterdir())
                         if p.suffix in (".json", ".safetensors", ".txt")},
            "continuation": "both actions forced at TRAIN nodes, subsequent visits sampled from prior-round GRPO",
            "shared_data": "identical refreshed trajectories for SFT, longitudinal GRPO and immediate GRPO; fixed control replays round 1",
            "fixed_continuation_control": "static common-parent future decisions; identical prompts, optimization tokens and update counts",
            "immediate_ablation": "current visit supported response minus 0.08 for this action only",
            "source_sha256": {s: file_hash(PROJECT / "research" / s) for s in (
                "tesis_timing_v6_train.py", "tesis_timing_v6_env.py", "tesis_timing_v6_bank.py",
                "tesis_shared_budget_qwen.py", "tesis_shared_budget_qwen_matched.py",
                "tesis_extended_experiment.py", "tesis_profiles_optimizer_v5.py")},
            "versions": {name: version(name) for name in ("torch", "transformers", "peft")}}
    plan_path = run / "TRAIN_PLAN.json"
    if plan_path.exists():
        if not args.resume or canonical(read(plan_path)) != canonical(plan):
            raise ValueError("Training run exists with incompatible options or without --resume")
    else:
        atomic_json(plan_path, plan)
    print(f"Enumerating {len(worlds)} TRAIN worlds at two budgets", flush=True)
    trees = {b: [enumerate_world(w, budget=b) for w in worlds] for b in (1, 2)}
    groups = train_groups(trees, prompt_text)
    texts = sorted({g.public for g in groups})
    if not groups:
        raise ValueError("Empty TRAIN groups")
    scorer = MarginScorer(args.batch_size, trainable=True)
    anchor, ref_tokens = score_map(scorer, texts, PARENT)
    totals = []
    for seed in args.seeds:
        for stage in range(1, args.rounds + 1):
            results = fit_round(scorer, groups, texts, anchor, prompt_text, run, seed, stage, args.batch_size, args.resume)
            totals.append(results)
    summary = {"device": torch.cuda.get_device_name(0), "seeds": args.seeds, "rounds": args.rounds,
               "training_plan_sha256": file_hash(plan_path), "common_parent_reference_inference": ref_tokens,
               "groups": len(groups), "unique_public_prompts": len(texts),
               "rounds_completed": [{"seed": r["seed"], "round": r["round"], "collector": r["collector"]["sha256"],
                                      "shared_rows_sha256": r["shared"]["shared_rows_sha256"],
                                      "trajectory_counts": r["shared"]["trajectory_counts"], "arms": r["arms"]}
                                     for r in totals]}
    atomic_json(run / "TRAIN_RESULT.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.rounds < 2 or args.batch_size < 1 or len(args.seeds) != len(set(args.seeds)):
        parser.error("Need distinct seeds, positive batches and at least two policy rounds")
    args.run.mkdir(parents=True, exist_ok=True)
    with (args.run / "TRAIN.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(args)


if __name__ == "__main__":
    main()
