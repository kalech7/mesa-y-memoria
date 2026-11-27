"""Compare Qwen LoRA training for exhausted menus, with a matched reward control."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random
from statistics import mean

from research.src.artifacts import digest, file_hash, write_json
from research.tesis_meal_qa import (
    DATA, ROOT, check_disjoint, dialogue_rows, load_cases, read, replay_rows, summarize,
)
from research.tesis_meal_train import rewards, scores, write_rows
from research.tesis_meal_varied_bank import LETTERS
from web_demo.meal_policy import recommendation_menu

RUN = ROOT / "research/runs/tesis_meal_clarification_v1"
PARENT = RUN.parent / "tesis_meal_qa_v1/sft"
FIXTURES = {split: f"meal_clarification_{split}" for split in ("train", "dev", "reserved")}
BASELINES = {"sft_previo": PARENT, "grpo_previo": PARENT.parent / "grpo"}
METHODS = ("sft", "grpo", "grpo_control")
SEEDS = (101, 202, 303)
EPOCHS = 2
BATCH = 2
EVAL_BATCH = 4
PASSES = 2
DRAWS = 12
TEMPERATURE = 1.6
CLIP = .2
LR = 5e-5
KL = .15
REPLAY_TOLERANCE = .04
ADVANTAGE_SCALE = 1.5
REVISED = {"valid": 1., "exhausted_dish": -2., "missed_unknown": -1.5,
           "unnecessary_ask": -1.25, "incompatible_dish": -1.}
STATES = ("exhausted", "unknown", "single", "multiple")


def kind(menu, forbidden, clarification_required):
    feasible = sum(bool(option["ingredients"]) and not set(option["ingredients"]).intersection(forbidden)
                   for option in menu)
    if not feasible:
        return "exhausted"
    if clarification_required:
        return "unknown"
    return "single" if feasible == 1 else "multiple"


def state_of(row):
    ask = not row["options"][row["accepted"][0]]["ingredients"]
    return kind(row["options"], row["forbidden"], ask)


def revised_reward(row, action):
    if action in row["accepted"]:
        return REVISED["valid"]
    if not row["options"][action]["ingredients"]:
        return REVISED["unnecessary_ask"]
    state = state_of(row)
    if state == "exhausted":
        return REVISED["exhausted_dish"]
    if state == "unknown":
        return REVISED["missed_unknown"]
    return REVISED["incompatible_dish"]


def cohort_stats(rows):
    by_state = {state: {"n": 0, "correct": 0, "asked": 0, "conflicts": 0} for state in STATES}
    histories = defaultdict(dict)
    for row in rows:
        group = by_state[row["state"]]
        group["n"] += 1
        group["correct"] += row["success"]
        group["asked"] += row["asked"]
        group["conflicts"] += row["conflict"]
        histories[row["conversation"], row["order"]][row["turn"]] = row
    transitions = {"exhausted_to_viable": {"n": 0, "correct": 0},
                   "viable_to_exhausted": {"n": 0, "correct": 0}}
    for turns in histories.values():
        ordered = [turns[i] for i in sorted(turns)]
        for before, after in zip(ordered, ordered[1:]):
            if before["state"] == "exhausted" and after["state"] in {"single", "multiple"}:
                transition = transitions["exhausted_to_viable"]
            elif before["state"] in {"single", "multiple"} and after["state"] == "exhausted":
                transition = transitions["viable_to_exhausted"]
            else:
                continue
            transition["n"] += 1
            transition["correct"] += before["success"] and after["success"]
    return {"states": by_state, "transitions": transitions}


def metrics(rows):
    output = summarize(rows)
    for order, subset in (("live_order", [row for row in rows if row["order"] == 0]), ("all_orders", rows)):
        output[order].update(cohort_stats(subset))
    return output


def rank(validation, baseline, epoch):
    qa = validation["qa"]["all_orders"]["states"]
    replay = validation["replay"]["all_orders"]
    replay_base = baseline["replay"]["all_orders"]
    exhausted = qa["exhausted"]["correct"] / qa["exhausted"]["n"]
    viable = sum(qa[key]["correct"] for key in ("single", "multiple")) / sum(
        qa[key]["n"] for key in ("single", "multiple"))
    passed = replay["correct"] / replay["cases"] >= replay_base["correct"] / replay_base["cases"] - REPLAY_TOLERANCE
    return passed, (exhausted + viable) / 2, min(exhausted, viable), -qa["unknown"]["n"] + qa["unknown"]["correct"], -epoch


def history_rows(name, split):
    return list(dialogue_rows(load_cases(name), split))


def data_before_freeze():
    target = history_rows(FIXTURES["train"], "train")
    validation = history_rows(FIXTURES["dev"], "validation")
    replay = replay_rows("train")
    archive_validation = replay_rows("validation")
    check_disjoint(target, validation + replay + archive_validation)
    check_disjoint(validation, replay + archive_validation)
    common = random.Random(1042)
    replay_validation = common.sample(archive_validation, 36)
    return target, validation, replay, replay_validation


def menu_fingerprint():
    return digest(recommendation_menu([], "dinner_preference", shuffle=False))


def schedule(target, replay, seed):
    rng = random.Random(seed)
    replay_subset = rng.sample(replay, 48)
    starts = {row["id"]: rng.randrange(6) for row in target + replay_subset}
    orders = []
    for epoch in range(EPOCHS):
        entries = [(row["id"], (starts[row["id"]] + epoch) % 6) for row in target + replay_subset]
        rng.shuffle(entries)
        orders.append(entries)
    return replay_subset, orders


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def logprobs(array, tokens):
    return (array[:, tokens] / TEMPERATURE).log_softmax(-1)


def cache_references(actor, tokens, packed, cache):
    import torch
    from research.tesis_meal_train import logits

    missing = [key for key in packed if key not in cache]
    with torch.no_grad():
        for group in chunks(missing, EVAL_BATCH):
            values = logprobs(logits(actor, [packed[key][1] for key in group]), tokens)
            cache.update(zip(group, values.unbind(), strict=True))


def evaluate(actor, tokens, data, path):
    import torch
    from research.tesis_meal_train import logits

    traces = []
    with torch.inference_mode():
        for batch in chunks(list(data.values()), EVAL_BATCH):
            output = logits(actor, [ids for _, ids in batch])
            actions = output[:, tokens].argmax(-1).tolist()
            for (row, _), choice, raw in zip(batch, actions, output.argmax(-1).tolist(), strict=True):
                options = row["options"]
                traces.append({
                    "id": row["id"], "conversation": row.get("conversation", row["id"]),
                    "turn": row.get("turn", 1), "family": row["family"], "order": row["rotation"],
                    "state": state_of(row), "asked": not options[choice]["ingredients"],
                    "letter": LETTERS[choice], "choice": options[choice]["name"],
                    "menu": dict(zip(LETTERS, (item["name"] for item in options), strict=True)),
                    "accepted": [options[i]["name"] for i in row["accepted"]], "forbidden": row["forbidden"],
                    "raw_token": actor.tokenizer.decode([raw]),
                    "raw_action_success": int(raw in (tokens[i] for i in row["accepted"])),
                    **scores(row, choice),
                })
    write_rows(path, traces)
    return metrics(traces)


def optimize(actor, tokens, entries, data, reference, optimizer, method, generator, output):
    import torch
    from research.tesis_meal_train import logits

    parameters = [p for p in actor.model.parameters() if p.requires_grad]
    ledger = []
    reward = rewards if method == "grpo_control" else lambda row: [
        revised_reward(row, i) for i in range(len(row["options"]))
    ]
    stats = defaultdict(Counter)
    for group in chunks(entries, BATCH):
        rows, inputs = zip(*(data[key] for key in group))
        if any(row["split"] != "train" for row in rows):
            raise ValueError("Only the training split can reach the optimizer")
        target = torch.stack([reference[key] for key in group])
        entry = {"cases": group, "loss": []}
        if method != "sft":
            with torch.no_grad():
                old = logprobs(logits(actor, inputs), tokens)
                sampled = torch.multinomial(old.exp(), DRAWS, replacement=True, generator=generator)
                values = torch.tensor([reward(row) for row in rows], device=actor.device).gather(1, sampled)
                relative = (values - values.mean(1, keepdim=True)) / ADVANTAGE_SCALE
            sampled_indices = sampled.tolist()
            entry.update(sampled=sampled_indices, rewards=values.tolist())
            for row, indices, returns in zip(rows, sampled_indices, entry["rewards"], strict=True):
                by_state = stats[state_of(row)]
                by_state["groups"] += 1
                by_state["varying"] += len(set(returns)) > 1
                by_state["with_ask"] += any(not row["options"][i]["ingredients"] for i in indices)
        for _ in range(PASSES):
            optimizer.zero_grad(set_to_none=True)
            array = logits(actor, inputs)
            policy = logprobs(array, tokens)
            kl = (policy.exp() * (policy - target)).sum(-1).mean()
            if method == "sft":
                full = array.log_softmax(-1)[:, tokens]
                objective = -torch.stack([line[row["accepted"]].logsumexp(0)
                                          for line, row in zip(full, rows, strict=True)]).mean()
            else:
                ratio = (policy.gather(1, sampled) - old.gather(1, sampled)).exp()
                objective = -torch.minimum(ratio * relative, ratio.clamp(1 - CLIP, 1 + CLIP) * relative).mean()
            loss = objective + KL * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            entry["loss"].append(float(loss.detach()))
        ledger.append(entry)
    write_rows(output, ledger)
    return {"presentations": len(entries), "updates": len(ledger) * PASSES, "sampling": stats}


def safe_json(path, value):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def train(output, device):
    import torch
    from research.tesis_meal_qa import packed
    from research.tesis_meal_train import load, reset, state
    from web_demo.meal_policy import option_tokens

    torch.set_num_threads(4)
    torch.manual_seed(SEEDS[0])
    target, development, replay, replay_validation = data_before_freeze()
    paths = [DATA / f"{name}.json" for name in FIXTURES.values()]
    paths += [Path(__file__), ROOT / "research/tesis_meal_qa.py", ROOT / "web_demo/meal_policy.py",
              ROOT / "research/tesis_meal_train.py",
              ROOT / "research/tesis_meal_varied_bank.py"]
    paths += [ROOT / f"research/runs/tesis_meal_lora_varied_v1/{split}.jsonl"
              for split in ("train", "validation", "evaluation")]
    paths += [path / "adapter_model.safetensors" for path in BASELINES.values()]
    plan = {"device": device, "seeds": SEEDS, "epochs": EPOCHS, "batch": BATCH, "eval_batch": EVAL_BATCH,
            "passes": PASSES, "draws": DRAWS, "learning_rate": LR, "kl": KL,
            "temperature": TEMPERATURE, "clip": CLIP, "advantage_scale": ADVANTAGE_SCALE,
            "reward": REVISED, "control_reward": [1., -1., -.35], "replay_tolerance": REPLAY_TOLERANCE,
            "validation_rule": "replay tolerance then mean exhausted ask and viable correct, worst rate, unknown correct, earliest",
            "default_seed": SEEDS[0], "targeted_prefixes": len(target), "replay_prefixes_per_seed": 48,
            "menu_sha256": menu_fingerprint(),
            "inputs": {str(path.relative_to(ROOT)): file_hash(path) for path in paths}}
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "PLAN.json"
    if plan_path.exists():
        if read(plan_path) != {**plan, "seeds": list(SEEDS)}:
            raise ValueError("Existing experiment has a different registered plan")
    else:
        safe_json(plan_path, plan)
    actor = load(PARENT, device)
    tokens = option_tokens(actor.tokenizer)
    initial = state(actor)
    data = {"qa": packed(actor, development, (0, 2, 4)),
            "replay": packed(actor, replay_validation, (0, 3))}
    baseline_path = output / "BASE_VALIDATION.json"
    if baseline_path.exists():
        baseline = read(baseline_path)
    else:
        baseline = {key: evaluate(actor, tokens, rows, output / f"baseline-val-{key}.jsonl")
                    for key, rows in data.items()}
        safe_json(baseline_path, baseline)
    print("Baseline", baseline["qa"]["all_orders"]["states"], flush=True)
    reference = {}
    for seed in SEEDS:
        folder = output / str(seed)
        folder.mkdir(exist_ok=True)
        if (folder / "SELECTION.json").is_file():
            continue
        rep, orders = schedule(target, replay, seed)
        train_rows = packed(actor, target + rep, range(6))
        scheduled = {key: train_rows[key] for epoch in orders for key in epoch}
        reset(actor, initial)
        cache_references(actor, tokens, scheduled, reference)
        methods = {}
        for method in METHODS:
            recap = folder / f"{method}-selection.json"
            if recap.is_file():
                methods[method] = read(recap)
                continue
            reset(actor, initial)
            optimizer = torch.optim.AdamW([p for p in actor.model.parameters() if p.requires_grad],
                                          lr=LR, weight_decay=0.)
            torch.manual_seed(seed)
            generator = torch.Generator(device=actor.device).manual_seed(seed)
            validations, training = {0: baseline}, {}
            best, weights = 0, initial
            checkpoint = folder / method
            for epoch, order in enumerate(orders, 1):
                prefix = f"{method}-{epoch}"
                training[epoch] = optimize(actor, tokens, order, scheduled, reference, optimizer, method, generator,
                                           folder / f"{prefix}-train.jsonl")
                val = {key: evaluate(actor, tokens, rows, folder / f"{prefix}-val-{key}.jsonl")
                       for key, rows in data.items()}
                validations[epoch] = val
                if rank(val, baseline, epoch) > rank(validations[best], baseline, best):
                    best, weights = epoch, state(actor)
                print(seed, method, epoch, val["qa"]["all_orders"]["states"],
                      "replay", val["replay"]["all_orders"]["correct"], flush=True)
            reset(actor, weights)
            actor.save_checkpoint(checkpoint)
            methods[method] = {"epoch": best, "validation": validations, "training": training,
                               "adapter_sha256": file_hash(checkpoint / "adapter_model.safetensors"),
                               "manifest_sha256": file_hash(checkpoint / "local_checkpoint.json")}
            safe_json(recap, methods[method])
            del optimizer
        safe_json(folder / "SELECTION.json", {"seed": seed, "plan_sha256": file_hash(plan_path),
                                               "split": "validation", "methods": methods})
    safe_json(output / "FREEZE.json", {"plan_sha256": file_hash(plan_path), "selection_sha256": {
        str(seed): file_hash(output / str(seed) / "SELECTION.json") for seed in SEEDS}})
    print("FROZEN", SEEDS, flush=True)


def test(output, device):
    from research.tesis_meal_qa import packed
    from research.tesis_meal_train import load, reset, state
    from web_demo.meal_policy import option_tokens
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    import torch

    torch.set_num_threads(4)
    plan = read(output / "PLAN.json")
    freeze_path = output / "FREEZE.json"
    seal = read(freeze_path)
    if seal["plan_sha256"] != file_hash(output / "PLAN.json"):
        raise ValueError("Frozen plan changed")
    if plan["menu_sha256"] != menu_fingerprint():
        raise ValueError("Frozen dinner menu changed")
    for path, checksum in plan["inputs"].items():
        if file_hash(ROOT / path) != checksum:
            raise ValueError(f"Frozen input changed: {path}")
    checkpoints = dict(BASELINES)
    for seed in SEEDS:
        folder = output / str(seed)
        selection = folder / "SELECTION.json"
        if file_hash(selection) != seal["selection_sha256"][str(seed)]:
            raise ValueError(f"Frozen selection changed: {seed}")
        for method, spec in read(selection)["methods"].items():
            path = folder / method
            if (file_hash(path / "adapter_model.safetensors") != spec["adapter_sha256"]
                    or file_hash(path / "local_checkpoint.json") != spec["manifest_sha256"]):
                raise ValueError(f"Frozen weights changed: {seed}/{method}")
            checkpoints[f"{seed}-{method}"] = path

    rows = history_rows(FIXTURES["reserved"], "evaluation")
    targeted, validation, replay, _ = data_before_freeze()
    old = replay_rows("evaluation")
    previous = []
    for path in sorted(DATA.glob("meal_qa_*.json")):
        item = read(path)
        previous.extend(dialogue_rows(item["cases"] if isinstance(item, dict) else item, "previous"))
    check_disjoint(rows, targeted + validation + replay + old + previous)
    actor = load(PARENT, device)
    initial = state(actor)
    tokens = option_tokens(actor.tokenizer)
    archived = random.Random(991).sample(old, 54)
    data = {"reserved": packed(actor, rows, range(6)), "archived": packed(actor, archived, (0, 3))}
    tables = {}
    trace_hashes = {}
    trace_dir = output / "evaluation"
    trace_dir.mkdir(exist_ok=True)
    for method, path in checkpoints.items():
        if path == PARENT:
            reset(actor, initial)
        else:
            set_peft_model_state_dict(actor.model, load_file(path / "adapter_model.safetensors"))
        tables[method] = {}
        for key, split in data.items():
            out = trace_dir / f"{method}-{key}.jsonl"
            tables[method][key] = evaluate(actor, tokens, split, out)
            trace_hashes[out.name] = file_hash(out)
        print("test", method, tables[method]["reserved"]["live_order"]["states"], flush=True)
    safe_json(output / "RESULTS.json", {"freeze_sha256": file_hash(freeze_path),
                                        "device": device, "profiles": tables, "traces": trace_hashes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "test"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    (train if args.phase == "train" else test)(args.output, args.device)


if __name__ == "__main__":
    main()
