"""Compare SFT, sampled GRPO and GRPO that evaluates every menu action."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random

from research.src.artifacts import file_hash
from research.tesis_meal_clarification import (
    FIXTURES as CLARIFICATION_FIXTURES, RUN as CLARIFICATION_RUN, SEEDS,
    evaluate, history_rows, menu_fingerprint, safe_json, state_of,
)
from research.tesis_meal_qa import DATA, ROOT, check_disjoint, dialogue_rows, packed, read, replay_rows
from research.tesis_meal_train import scores, write_rows
from research.tesis_meal_varied_bank import LETTERS

RUN = CLARIFICATION_RUN.parent / "tesis_meal_exploration_v1"
FIXTURES = {split: f"meal_exploration_{split}" for split in ("train", "dev", "reserved")}
METHODS = ("sft", "grpo", "grpo_sampled")
EPOCHS = 3
BATCH = 2
PASSES = 2
DRAWS = 8
EPSILON = .25
TEMPERATURE = 2.
CLIP = .2
SCALE = 2.
LR = 4e-5
KL = .12
REPLAY_TOLERANCE = .04
REPLAY_COUNT = 32
PRIOR_COUNT = 42
REWARD = {"dish": 1., "ask_exhausted": 1.5, "ask_unknown": 1.,
          "conflict": -2.5, "unnecessary_ask": -.65, "missed_unknown": -1.75}
ORDERS_DEV = (0, 2, 4)


def checkpoint(seed, method):
    return CLARIFICATION_RUN / str(seed) / method


def score_action(row, action):
    evaluation = scores(row, action)
    if evaluation["success"]:
        state = state_of(row)
        if state in ("exhausted", "unknown"):
            return REWARD[f"ask_{state}"]
        return REWARD["dish"]
    if evaluation["conflict"]:
        return REWARD["conflict"]
    if evaluation["unnecessary_clarification"]:
        return REWARD["unnecessary_ask"]
    if evaluation["missed_clarification"]:
        return REWARD["missed_unknown"]
    raise ValueError("Action has no reward")


def distributions(array, tokens):
    policy = (array[:, tokens] / TEMPERATURE).log_softmax(-1)
    exploratory = ((1 - EPSILON) * policy.exp() + EPSILON / len(tokens)).log()
    return policy, exploratory


def group_advantage(old_logq, reward):
    center = (old_logq.exp() * reward).sum(-1, keepdim=True)
    return (reward - center) / SCALE


def group_objective(current, old, advantage, sampled=None):
    ratio = (current - old).exp()
    clipped = ratio.clamp(1 - CLIP, 1 + CLIP)
    surrogate = (ratio * advantage).minimum(clipped * advantage)
    if sampled is None:
        return -(old.exp() * surrogate).sum(-1).mean()
    return -surrogate.gather(1, sampled).mean()


def training_data():
    target = history_rows(FIXTURES["train"], "train")
    prior = history_rows(CLARIFICATION_FIXTURES["train"], "train")
    dev = history_rows(FIXTURES["dev"], "validation")
    replay = replay_rows("train")
    archived_validation = replay_rows("validation")
    check_disjoint(target, prior + dev + replay)
    check_disjoint(dev, prior + replay + archived_validation)
    return target, prior, dev, replay, random.Random(731).sample(archived_validation, 48)


def schedule(target, prior, replay, seed):
    rng = random.Random(seed)
    subset = rng.sample(prior, PRIOR_COUNT) + rng.sample(replay, REPLAY_COUNT)
    counts = {row["id"]: rng.randrange(6) for row in target + subset}
    orders = []
    for epoch in range(EPOCHS):
        order = [(row["id"], (counts[row["id"]] + epoch) % 6) for row in target + subset]
        rng.shuffle(order)
        orders.append(order)
    return subset, orders


def viable(state):
    correct = sum(state[key]["correct"] for key in ("single", "multiple"))
    count = sum(state[key]["n"] for key in ("single", "multiple"))
    return correct / count


def rank(metrics, baseline, epoch):
    report = metrics["qa"]["all_orders"]
    states = report["states"]
    ask = states["exhausted"]["correct"] / states["exhausted"]["n"]
    valid = viable(states)
    replay = metrics["replay"]["all_orders"]
    base = baseline["replay"]["all_orders"]
    eligible = replay["correct"] / replay["cases"] >= base["correct"] / base["cases"] - REPLAY_TOLERANCE
    return eligible, (ask + valid) / 2, -report["conflicts"], states["unknown"]["correct"], -epoch


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def reference_policy(actor, tokens, bank):
    import torch
    from research.tesis_meal_train import logits

    reference = {}
    with torch.no_grad():
        for group in batches(list(bank), 4):
            logits_array = logits(actor, [bank[key][1] for key in group])
            policy, _ = distributions(logits_array, tokens)
            reference.update(zip(group, policy.unbind(), strict=True))
    return reference


def optimize(actor, tokens, order, bank, reference, optimizer, method, generator, path):
    import torch
    from research.tesis_meal_train import logits

    parameters = [p for p in actor.model.parameters() if p.requires_grad]
    ledger = []
    sampling = defaultdict(Counter)
    for group in batches(order, BATCH):
        rows, inputs = zip(*(bank[key] for key in group))
        if any(row["split"] != "train" for row in rows):
            raise ValueError("Only training states can reach the optimizer")
        ref = torch.stack([reference[key] for key in group])
        entry = {"cases": group, "loss": []}
        sampled = None
        if method != "sft":
            with torch.no_grad():
                _, old = distributions(logits(actor, inputs), tokens)
                reward = torch.tensor([[score_action(row, action) for action in range(len(tokens))] for row in rows],
                                      device=actor.device)
                advantage = group_advantage(old, reward)
                if method == "grpo_sampled":
                    sampled = torch.multinomial(old.exp(), DRAWS, replacement=True, generator=generator)
            full_group = list(range(len(tokens)))
            actions = sampled.tolist() if sampled is not None else [full_group] * len(rows)
            entry.update(rewards=reward.tolist(), old_probabilities=old.exp().tolist(), actions=actions)
            for row, chosen, rewards in zip(rows, actions, entry["rewards"], strict=True):
                stats = sampling[state_of(row)]
                stats["groups"] += 1
                stats["varied"] += len({rewards[index] for index in chosen}) > 1
                stats["with_ask"] += any(not row["options"][index]["ingredients"] for index in chosen)
                stats["all_actions"] += len(set(chosen)) == len(tokens)
        for _ in range(PASSES):
            optimizer.zero_grad(set_to_none=True)
            array = logits(actor, inputs)
            policy, exploratory = distributions(array, tokens)
            kl = (policy.exp() * (policy - ref)).sum(-1).mean()
            if method == "sft":
                whole = array.log_softmax(-1)[:, tokens]
                objective = -torch.stack([line[row["accepted"]].logsumexp(0)
                                          for line, row in zip(whole, rows, strict=True)]).mean()
            else:
                objective = group_objective(exploratory, old, advantage, sampled)
            loss = objective + KL * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            entry["loss"].append(float(loss.detach()))
        ledger.append(entry)
    write_rows(path, ledger)
    return {"presentations": len(order), "updates": len(ledger) * PASSES, "sampling": sampling}


def plan_for(device, target):
    paths = [Path(__file__), ROOT / "research/tesis_meal_clarification.py", ROOT / "research/tesis_meal_qa.py",
             ROOT / "research/tesis_meal_train.py", ROOT / "research/tesis_meal_varied_bank.py",
             ROOT / "web_demo/meal_policy.py", CLARIFICATION_RUN / "FREEZE.json"]
    paths += [DATA / f"{name}.json" for name in (*FIXTURES.values(), CLARIFICATION_FIXTURES["train"])]
    paths += [ROOT / f"research/runs/tesis_meal_lora_varied_v1/{split}.jsonl"
              for split in ("train", "validation", "evaluation")]
    for seed in SEEDS:
        for method in ("sft", "grpo"):
            directory = checkpoint(seed, method)
            paths.extend(directory / name for name in ("adapter_model.safetensors", "local_checkpoint.json"))
    return {"device": device, "seeds": list(SEEDS), "epochs": EPOCHS, "methods": list(METHODS),
            "batch": BATCH, "passes": PASSES, "draws": DRAWS, "epsilon": EPSILON, "temperature": TEMPERATURE,
            "clip": CLIP, "scale": SCALE, "lr": LR, "kl": KL, "replay_tolerance": REPLAY_TOLERANCE,
            "reward": REWARD, "validation_orders": list(ORDERS_DEV), "validation_replay_count": 48,
            "targeted_prefixes": len(target), "prior_prefixes_per_seed": PRIOR_COUNT,
            "archive_prefixes_per_seed": REPLAY_COUNT, "default_seed": SEEDS[0],
            "validation_rule": "replay tolerance, balanced exhausted and viable accuracy, fewer conflicts, unknown, earliest",
            "menu_sha256": menu_fingerprint(),
            "inputs": {str(path.relative_to(ROOT)): file_hash(path) for path in paths}}


def validate_models(plan, path):
    frozen = read(CLARIFICATION_RUN / "FREEZE.json")
    seal = read(path)
    for seed in SEEDS:
        parent_seal = CLARIFICATION_RUN / str(seed) / "SELECTION.json"
        if file_hash(parent_seal) != frozen["selection_sha256"][str(seed)]:
            raise ValueError(f"Source checkpoints changed for seed {seed}")
        for method in ("sft", "grpo"):
            selected = read(parent_seal)["methods"][method]
            folder = checkpoint(seed, method)
            if (file_hash(folder / "local_checkpoint.json") != selected["manifest_sha256"]
                    or file_hash(folder / "adapter_model.safetensors") != selected["adapter_sha256"]):
                raise ValueError(f"Source adapter changed for seed {seed}/{method}")
    if plan != seal:
        raise ValueError("The existing plan differs")


def load_weights(actor, adapter):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    set_peft_model_state_dict(actor.model, load_file(adapter / "adapter_model.safetensors"))


def train(output, device):
    import torch
    from research.tesis_meal_train import load, reset, state
    from web_demo.meal_policy import option_tokens

    torch.set_num_threads(4)
    target, prior, dev, replay, replay_dev = training_data()
    plan = plan_for(device, target)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "PLAN.json"
    if not plan_path.exists():
        safe_json(plan_path, plan)
    validate_models(plan, plan_path)
    actor = load(checkpoint(SEEDS[0], "sft"), device)
    tokens = option_tokens(actor.tokenizer)
    panels = {"qa": packed(actor, dev, ORDERS_DEV), "replay": packed(actor, replay_dev, (0, 3))}
    for seed in SEEDS:
        folder = output / str(seed)
        folder.mkdir(exist_ok=True)
        if (folder / "SELECTION.json").is_file():
            continue
        parent = checkpoint(seed, "sft")
        load_weights(actor, parent)
        actor.origin_checkpoint = str(parent)
        initial = state(actor)
        validation_path = folder / "BASE_VALIDATION.json"
        if validation_path.exists():
            baseline = read(validation_path)
        else:
            baseline = {key: evaluate(actor, tokens, data, folder / f"base-val-{key}.jsonl")
                        for key, data in panels.items()}
            safe_json(validation_path, baseline)
        subset, orders = schedule(target, prior, replay, seed)
        candidates = packed(actor, target + subset, range(6))
        bank = {key: candidates[key] for order in orders for key in order}
        reference = reference_policy(actor, tokens, bank)
        methods = {}
        for method in METHODS:
            receipt = folder / f"{method}-selection.json"
            if receipt.is_file():
                methods[method] = read(receipt)
                continue
            reset(actor, initial)
            torch.manual_seed(seed)
            generator = torch.Generator(device=actor.device).manual_seed(seed)
            optimizer = torch.optim.AdamW([p for p in actor.model.parameters() if p.requires_grad],
                                          lr=LR, weight_decay=0.)
            validation, telemetry = {0: baseline}, {}
            best, weights = 0, initial
            for epoch, order in enumerate(orders, 1):
                prefix = f"{method}-{epoch}"
                telemetry[epoch] = optimize(actor, tokens, order, bank, reference, optimizer, method, generator,
                                            folder / f"{prefix}-train.jsonl")
                result = {key: evaluate(actor, tokens, data, folder / f"{prefix}-val-{key}.jsonl")
                          for key, data in panels.items()}
                validation[epoch] = result
                if rank(result, baseline, epoch) > rank(validation[best], baseline, best):
                    best, weights = epoch, state(actor)
                report = result["qa"]["all_orders"]
                print(seed, method, epoch, report["states"], "replay", result["replay"]["all_orders"]["correct"],
                      flush=True)
            reset(actor, weights)
            path = folder / method
            actor.save_checkpoint(path)
            methods[method] = {"epoch": best, "validation": validation, "training": telemetry,
                               "adapter_sha256": file_hash(path / "adapter_model.safetensors"),
                               "manifest_sha256": file_hash(path / "local_checkpoint.json")}
            safe_json(receipt, methods[method])
            del optimizer
        safe_json(folder / "SELECTION.json", {"seed": seed, "plan_sha256": file_hash(plan_path),
                                               "split": "validation", "methods": methods})
    safe_json(output / "FREEZE.json",
              {"plan_sha256": file_hash(plan_path), "selection_sha256": {
                  str(seed): file_hash(output / str(seed) / "SELECTION.json") for seed in SEEDS}})
    print("FROZEN", SEEDS, flush=True)


def test(output, device):
    import torch
    from research.tesis_meal_train import load
    from web_demo.meal_policy import option_tokens

    torch.set_num_threads(4)
    plan_path, freeze_path = output / "PLAN.json", output / "FREEZE.json"
    plan, freeze = read(plan_path), read(freeze_path)
    if freeze["plan_sha256"] != file_hash(plan_path) or plan["menu_sha256"] != menu_fingerprint():
        raise ValueError("Frozen plan or dinner menu changed")
    for path, checksum in plan["inputs"].items():
        if file_hash(ROOT / path) != checksum:
            raise ValueError(f"Frozen input changed: {path}")
    models = {}
    for seed in SEEDS:
        directory = output / str(seed)
        selection = directory / "SELECTION.json"
        if file_hash(selection) != freeze["selection_sha256"][str(seed)]:
            raise ValueError(f"Frozen selection changed: {seed}")
        for method, manifest in read(selection)["methods"].items():
            path = directory / method
            if (file_hash(path / "adapter_model.safetensors") != manifest["adapter_sha256"]
                    or file_hash(path / "local_checkpoint.json") != manifest["manifest_sha256"]):
                raise ValueError(f"Selected checkpoint changed: {seed}/{method}")
            models[f"{seed}-{method}"] = path
        for method in ("sft", "grpo"):
            models[f"{seed}-{method}_previo"] = checkpoint(seed, method)
    reserved = history_rows(FIXTURES["reserved"], "evaluation")
    target, prior, dev, replay, _ = training_data()
    archived = replay_rows("evaluation")
    previous = []
    for pattern in ("meal_qa_*.json", "meal_clarification_*.json"):
        for path in sorted(DATA.glob(pattern)):
            source = read(path)
            previous.extend(dialogue_rows(source["cases"] if isinstance(source, dict) else source, "previous"))
    check_disjoint(reserved, target + prior + dev + replay + archived + previous)
    actor = load(checkpoint(SEEDS[0], "sft"), device)
    tokens = option_tokens(actor.tokenizer)
    panels = {"reserved": packed(actor, reserved, range(6)),
              "archived": packed(actor, random.Random(928).sample(archived, 54), (0, 3))}
    directory = output / "evaluation"
    directory.mkdir()
    result, hashes = {}, {}
    for model, checkpoint_path in models.items():
        load_weights(actor, checkpoint_path)
        result[model] = {}
        for key, data in panels.items():
            path = directory / f"{model}-{key}.jsonl"
            result[model][key] = evaluate(actor, tokens, data, path)
            hashes[path.name] = file_hash(path)
        metric = result[model]["reserved"]["live_order"]
        print(model, metric["correct"], metric["states"], flush=True)
    safe_json(output / "RESULTS.json", {"device": device, "freeze_sha256": file_hash(freeze_path),
                                        "profiles": result, "traces": hashes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "test"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    (train if args.phase == "train" else test)(args.output, args.device)


if __name__ == "__main__":
    main()
