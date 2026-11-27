"""Train matched Qwen LoRA SFT and sampled GRPO memory actors."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import json
import random

from research.grpo_select_checkpoint import load_selector
from research.src.artifacts import canonical, file_hash, write_json
from research.tesis_meal_exploration import load_weights
from research.tesis_meal_train import load, logits, reset, state, write_rows
from research.tesis_temporal_memory_policy import option_tokens, prompt, recall_actions, write_actions
from research.tesis_temporal_memory_reward import REWARD, score_action
from web_demo.cases import BASE, MODELS, ROOT


RUN = ROOT / "research/runs/tesis_temporal_memory_grpo_v1"
QA = ROOT / "research/data/meal_memory_grpo_heldout.json"
PARENT = MODELS["sft"].path
SEEDS = (101, 202, 303)
METHODS = ("sft", "grpo")
BATCH = 4
EPOCHS = 2
WARM = 1
PASSES = 2
DRAWS = 12
TEMPERATURE = 1.5
EXPLORATION = .25
KL = .06
CLIP = .20
LR = 1.2e-4
WARM_LR = 1.5e-4


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def batches(items):
    for start in range(0, len(items), BATCH):
        yield items[start:start + BATCH]


def preparation(rows):
    """Score supervised labels and runtime distractors using public training fixtures."""
    data = []
    for row in rows:
        action_source = write_actions if row["phase"] == "write" else recall_actions
        goal = row["actions"][row["correct_index"]]
        best = score_action(row, goal)
        unique = {canonical(action): action for action in (*row["actions"], *action_source(row["observation"]))}
        options = list(unique.values())
        rewards = [score_action(row, action) for action in options]
        if not options or max(rewards) > best + 1e-6:
            raise ValueError(f"Gold is inconsistent with reward: {row['id']}, {rewards}, {best}")
        # Keep runtime candidates and difficult authored negatives. The
        # gold action is included only in these train/dev fixtures.
        data.append((row, options, rewards, best))
    return data


def pack(actor, data, rng):
    packed = []
    for row, options, rewards, best in data:
        order = list(range(len(options)))
        rng.shuffle(order)
        order = order[:22]
        gold = row["actions"][row["correct_index"]]
        gold_id = next(i for i, action in enumerate(options) if action == gold)
        if gold_id not in order:
            order[-1] = gold_id
            rng.shuffle(order)
        actions = [options[i] for i in order]
        values = [rewards[i] for i in order]
        ids = actor.encode_prompt(prompt(row["phase"], row["observation"], actions))
        packed.append((row, ids, values, [abs(v - best) < 1e-6 for v in values]))
    return packed


def tensor_batch(actor, tokens, group, *, temperature=1.):
    import torch

    width = max(len(x[2]) for x in group)
    inputs = [x[1] for x in group]
    legal = torch.arange(width, device=actor.device)[None, :] < torch.tensor(
        [len(row[2]) for row in group], device=actor.device)[:, None]
    values = logits(actor, inputs)[:, tokens[:width]] / temperature
    return values.masked_fill(~legal, -1e4).log_softmax(-1), legal


def develop(actor, tokens, packed):
    import torch

    totals = Counter()
    actor.model.eval()
    with torch.inference_mode():
        for group in batches(packed):
            logp, _ = tensor_batch(actor, tokens, group)
            for (row, _, rewards, good), option in zip(group, logp.argmax(-1).tolist(), strict=True):
                phase = row["phase"]
                totals[phase] += 1
                totals[f"{phase}_hits"] += good[option]
                totals["reward"] += rewards[option]
    return dict(totals)


def rank(metrics, epoch):
    # One poorly learned phase cannot be concealed by an easier or bigger one.
    rates = [metrics[f"{phase}_hits"] / metrics[phase] for phase in ("write", "recall")]
    return (sum(rates) / 2, min(rates), metrics["reward"], -epoch)


def exploratory(logp, legal):
    return ((1 - EXPLORATION) * logp.exp() + EXPLORATION * legal / legal.sum(-1, keepdim=True)).clamp_min(1e-30).log()


def optimize(actor, tokens, schedule, optimizer, method, rng, reference=None, *, passes=PASSES):
    import torch

    actor.model.train()
    params = [p for p in actor.model.parameters() if p.requires_grad]
    log = []
    for index, group in enumerate(batches(schedule)):
        report = {"examples": [row["id"] for row, *_ in group]}
        losses = []
        for update in range(passes):
            optimizer.zero_grad(set_to_none=True)
            logp, legal = tensor_batch(actor, tokens, group, temperature=TEMPERATURE if method == "grpo" else 1.)
            if method == "sft":
                targets = torch.tensor([good + [False] * (logp.shape[1] - len(good)) for _, _, _, good in group],
                                       device=actor.device)
                loss = -logp.masked_fill(~targets, -1e4).logsumexp(-1).mean()
            else:
                # The same rollout group supplies both updates. The ratio uses
                # the actual behavior distribution, including exploration.
                behavior = exploratory(logp, legal)
                if update == 0:
                    old = behavior.detach()
                    drawn = torch.multinomial(old.exp(), DRAWS, replacement=True, generator=rng)
                    rewards = torch.tensor([row[2] + [0.] * (logp.shape[1] - len(row[2])) for row in group],
                                           device=actor.device)
                    returns = rewards.gather(1, drawn)
                    centered = returns - returns.mean(1, keepdim=True)
                    advantage = centered / returns.std(1, keepdim=True, unbiased=False).clamp_min(.4)
                    varied = returns.amax(1) != returns.amin(1)
                    report.update(sampled=drawn.tolist(), rewards=returns.tolist(), nonconstant_groups=int(varied.sum()))
                ratio = (behavior.gather(1, drawn) - old.gather(1, drawn)).exp()
                clipped = ratio.clamp(1 - CLIP, 1 + CLIP)
                penalty = (logp.exp() * (logp - reference[index])).sum(-1).mean()
                loss = -torch.minimum(ratio * advantage, clipped * advantage).mean() + KL * penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        log.append({**report, "losses": losses})
    return log


def optimizer(actor, lr):
    import torch

    return torch.optim.AdamW([p for p in actor.model.parameters() if p.requires_grad], lr=lr, weight_decay=0.)


def _train_plan(output, split, device):
    paths = ("tesis_temporal_memory_train.py", "tesis_temporal_memory_policy.py", "tesis_temporal_memory_reward.py",
             "tesis_temporal_memory_data.py")
    return {"seeds": SEEDS, "device": device, "methods": METHODS, "batch": BATCH, "epochs": EPOCHS,
            "shared_warmup_epochs": WARM, "passes": PASSES, "draws": DRAWS,
            "temperature": TEMPERATURE, "exploration": EXPLORATION,
            "learning_rate": LR, "warm_learning_rate": WARM_LR, "kl": KL, "clip": CLIP, "reward": REWARD,
            "inputs": {name: file_hash(output / f"{name}.jsonl") for name in split},
            "source_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in paths},
            "parent": str(PARENT.relative_to(ROOT)), "parent_manifest_sha256": file_hash(PARENT / "local_checkpoint.json"),
            "parent_adapter_sha256": file_hash(PARENT / "adapter_model.safetensors"),
            "heldout_sha256": file_hash(QA),
            "heldout_selection": False, "checkpoint_rule": "dev balanced write/recall, lower phase, reward, earliest",
            "generation": "greedy over public action options; no verifier"}


def train(output, device):
    import torch
    from research.tesis_temporal_memory_data import build_split

    torch.set_num_threads(4)
    torch.manual_seed(SEEDS[0])
    split = {name: build_split(name) for name in ("train", "dev")}
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in split.items():
        dest = output / f"{name}.jsonl"
        if not dest.exists():
            write_rows(dest, rows)
        elif [json.loads(line) for line in dest.read_text().splitlines()] != rows:
            raise ValueError(f"Frozen {name} data changed.")
    plan = _train_plan(output, split, device)
    plan_path = output / "PLAN.json"
    if plan_path.exists() and canonical(read(plan_path)) != canonical(plan):
        raise ValueError("Frozen temporal training plan changed.")
    write_json(plan_path, plan)
    training, validation = (preparation(split[name]) for name in ("train", "dev"))
    actor = load(PARENT, device)
    actor.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    initial = state(actor)
    letters = option_tokens(actor.tokenizer)
    dev = [x for offset in (918, 317) for x in pack(actor, validation, random.Random(offset))]
    print("Banco", len(training), len(validation), "tokens máximos dev", max(len(row[1]) for row in dev), flush=True)
    for seed in SEEDS:
        folder = output / str(seed)
        folder.mkdir(exist_ok=True)
        if (folder / "SELECTION.json").exists():
            continue
        reset(actor, initial)
        torch.manual_seed(seed)
        rng = torch.Generator(device=actor.device).manual_seed(seed)
        order_rng = random.Random(seed)
        warm = pack(actor, training, order_rng)
        warm_opt = optimizer(actor, WARM_LR)
        for epoch in range(WARM):
            order_rng.shuffle(warm)
            warm_log = optimize(actor, letters, warm, warm_opt, "sft", rng, passes=1)
            write_rows(folder / f"warm-{epoch+1}.jsonl", warm_log)
        del warm_opt
        common = state(actor)
        baseline = develop(actor, letters, dev)
        print(seed, "SFT común", baseline, flush=True)
        schedules = [pack(actor, training, order_rng) for _ in range(EPOCHS)]
        for schedule in schedules:
            order_rng.shuffle(schedule)
        with torch.no_grad():
            references = [[tensor_batch(actor, letters, group, temperature=TEMPERATURE)[0].detach()
                           for group in batches(schedule)] for schedule in schedules]
        results = {}
        for method in METHODS:
            reset(actor, common)
            opt = optimizer(actor, LR)
            validation_runs = {0: baseline}
            best, winner = 0, common
            diversity = 0
            for epoch, (schedule, reference) in enumerate(zip(schedules, references, strict=True), 1):
                ledger = optimize(actor, letters, schedule, opt, method, rng, reference=reference)
                diversity += sum(row.get("nonconstant_groups", 0) for row in ledger)
                write_rows(folder / f"{method}-{epoch}-train.jsonl", ledger)
                metrics = develop(actor, letters, dev)
                validation_runs[epoch] = metrics
                if rank(metrics, epoch) > rank(validation_runs[best], best):
                    best, winner = epoch, state(actor)
                print(seed, method, epoch, metrics, f"{diversity} grupos con señal", flush=True)
            reset(actor, winner)
            checkpoint = folder / method
            actor.save_checkpoint(checkpoint)
            results[method] = {"selected_epoch": best, "validation": validation_runs,
                               "updates": PASSES * sum(len(list(batches(schedule))) for schedule in schedules),
                               "groups_with_signal": diversity, "groups_seen": sum(map(len, schedules)),
                               "adapter_sha256": file_hash(checkpoint / "adapter_model.safetensors"),
                               "manifest_sha256": file_hash(checkpoint / "local_checkpoint.json")}
            del opt
        write_json(folder / "SELECTION.json", results)
    write_json(output / "FREEZE.json", {"plan_sha256": file_hash(plan_path), "selections": {
        str(seed): file_hash(output / str(seed) / "SELECTION.json") for seed in SEEDS}})


def load_frozen(output, device):
    plan = read(output / "PLAN.json")
    receipt = read(output / "FREEZE.json")
    if receipt["plan_sha256"] != file_hash(output / "PLAN.json"):
        raise ValueError("Plan changed after training.")
    for seed in SEEDS:
        file = output / str(seed) / "SELECTION.json"
        if file_hash(file) != receipt["selections"][str(seed)]:
            raise ValueError("Selection changed after training.")
        for method, selection in read(file).items():
            checkpoint = output / str(seed) / method
            if (file_hash(checkpoint / "local_checkpoint.json") != selection["manifest_sha256"]
                    or file_hash(checkpoint / "adapter_model.safetensors") != selection["adapter_sha256"]):
                raise ValueError("Selected weights changed after training.")
    actor = load_selector(output / str(SEEDS[0]) / "sft", device=device, trainable=True, model_path=BASE)
    if device == "cpu":
        actor.model.float()
    actor.model.eval()
    return actor, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=RUN)
    args = parser.parse_args()
    train(args.output, args.device)


if __name__ == "__main__":
    main()
