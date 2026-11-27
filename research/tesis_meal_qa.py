"""Retrain Qwen on observed guard-free failures and test on sealed conversations."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import torch

from research.src.artifacts import file_hash, write_json
from research.tesis_meal_train import load, logits, reset, rewards, scores, state, write_rows
from research.tesis_meal_varied import advantages
from research.tesis_meal_varied_bank import LETTERS, normalized, prompt, rotate
from web_demo.meal_policy import option_tokens, recommendation_menu

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research/data"
ARCHIVE = ROOT / "research/runs/tesis_meal_lora_varied_v1"
ARMS = ("sft", "grpo")
BASELINE_IDS = {f"comidas-variado-{arm}" for arm in ARMS}
OBSERVED = ("meal_qa_typos", "meal_qa_corrections")
AUGMENTED = ("meal_qa_augment", "meal_qa_augment_typos")
SEED = 417
EPOCHS = 3
BATCH = 2
EVAL_BATCH = 4
DRAWS = 8
PASSES = 2
LR = 5e-5
TEMPERATURE = 1.4
KL_WEIGHT = .15
CLIP = .2
REPLAY_TOLERANCE = .03


def read(path):
    return json.loads(path.read_bytes())


def load_cases(name):
    return read(DATA / f"{name}.json")


def replay_rows(split):
    return [json.loads(line) for line in (ARCHIVE / f"{split}.jsonl").read_text().splitlines()]


def dialogue_rows(cases, split):
    for case in cases:
        texts = []
        for turn, entry in enumerate(case["turns"], 1):
            texts.append(entry["text"])
            options = recommendation_menu(texts, "dinner_preference", shuffle=True)
            forbidden = set(entry["excluded"])
            accepted = [index for index, option in enumerate(options)
                        if option["ingredients"] and not forbidden.intersection(option["ingredients"])]
            if entry.get("clarify") or not accepted:
                accepted = [next(index for index, option in enumerate(options) if not option["ingredients"])]
            yield {"id": f"{case['id']}:{turn}", "conversation": case["id"], "turn": turn,
                   "turns": texts.copy(), "split": split, "family": case.get("family") or case["category"],
                   "options": options, "accepted": accepted, "forbidden": sorted(forbidden)}


def check_disjoint(left, right):
    # For these independent authored datasets, even individual user messages
    # must be novel. A shared prefix would leak one or more scored turns.
    def entries(rows):
        return {normalized([text]) for row in rows for text in row["turns"]}

    if overlap := entries(left) & entries(right):
        raise ValueError(f"Overlapping user messages across QA splits: {sorted(overlap)}")


def observed_cases():
    observed = []
    failures = set()
    for name in OBSERVED:
        for case in load_cases(name)["cases"]:
            responses = case["results"]
            if set(responses) != BASELINE_IDS or any(
                len(trace) != len(case["turns"]) or not all(turn["direct"] for turn in trace)
                for trace in responses.values()
            ):
                raise ValueError(f"QA requires finished, direct choices from both models: {case['id']}")
            for row, turn_index in zip(dialogue_rows([case], "observed"), range(len(case["turns"])), strict=True):
                menu = [option["name"] for option in row["options"]]
                for trace in responses.values():
                    entry = trace[turn_index]
                    reported_menu = entry["meal_menu"] if "meal_menu" in entry else entry["trace"]["menú"]
                    if (list(reported_menu.values()) != menu
                            or bool(scores(row, menu.index(entry["choice"]))["success"]) != entry["success"]):
                        raise ValueError(f"QA trace does not match the menu or independent labels: {row['id']}")
            if any(not turn["success"] for trace in responses.values() for turn in trace):
                failures.add(case["id"])
            observed.append(case)
    return observed, failures


def packed(actor, rows, offsets):
    items = {}
    for row in rows:
        for offset in offsets:
            item = rotate(row, offset)
            items[row["id"], offset] = (item, actor.encode_prompt(prompt(item["turns"], item["options"])))
    return items


def batches(items, size=BATCH):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def evaluate(actor, tokens, data, output):
    traces = []
    with torch.inference_mode():
        for batch in batches(list(data.values()), EVAL_BATCH):
            predictions = logits(actor, [ids for _, ids in batch])
            choices = predictions[:, tokens].argmax(-1).tolist()
            for (row, _), action, raw_token in zip(batch, choices, predictions.argmax(-1).tolist(), strict=True):
                traces.append({
                    "id": row["id"], "conversation": row.get("conversation", row["id"]),
                    "turn": row.get("turn", 1), "family": row["family"], "order": row["rotation"],
                    "choice": row["options"][action]["name"], "letter": LETTERS[action],
                    "menu": {letter: item["name"] for letter, item in zip(LETTERS, row["options"], strict=True)},
                    "accepted": [row["options"][index]["name"] for index in row["accepted"]],
                    "forbidden": row["forbidden"], "raw_token": actor.tokenizer.decode([raw_token]),
                    "raw_action_success": int(raw_token in (tokens[i] for i in row["accepted"])),
                    **scores(row, action),
                })
    write_rows(output, traces)
    return summarize(traces)


def summarize(traces):
    def measures(rows):
        conversations = defaultdict(list)
        for row in rows:
            conversations[row["conversation"]].append(row)
        final = []
        for items in conversations.values():
            last_turn = max(item["turn"] for item in items)
            final.extend(item["success"] for item in items if item["turn"] == last_turn)
        return {
            "cases": len(rows), "conversations": len(conversations),
            "correct": sum(row["success"] for row in rows),
            "conflicts": sum(row["conflict"] for row in rows),
            "unnecessary_clarifications": sum(row["unnecessary_clarification"] for row in rows),
            "missed_clarifications": sum(row["missed_clarification"] for row in rows),
            "fully_correct": sum(all(row["success"] for row in group) for group in conversations.values()),
            "final_correct": sum(final), "final_presentations": len(final),
            "raw_correct": sum(row["raw_action_success"] for row in rows),
        }

    return {"live_order": measures([row for row in traces if row["order"] == 0]),
            "all_orders": measures(traces)}


def logprobs(array, tokens):
    return (array[:, tokens] / TEMPERATURE).log_softmax(-1)


def references(actor, tokens, train):
    output = {}
    with torch.no_grad():
        for keys in batches(list(train), EVAL_BATCH):
            predictions = logits(actor, [train[key][1] for key in keys])
            output.update(zip(keys, logprobs(predictions, tokens).unbind(), strict=True))
    return output


def optimize(actor, tokens, train, keys, reference, optimizer, method, generator, output):
    history = []
    parameters = [p for p in actor.model.parameters() if p.requires_grad]
    for group in batches(keys):
        rows, inputs = zip(*(train[key] for key in group))
        if any(row["split"] != "train" for row in rows):
            raise ValueError("Only training examples can reach the optimizer")
        target = torch.stack([reference[key] for key in group])
        entry = {"cases": group, "loss": []}
        if method == "grpo":
            with torch.no_grad():
                old = logprobs(logits(actor, inputs), tokens)
                sampled = torch.multinomial(old.exp(), DRAWS, replacement=True, generator=generator)
                values = torch.tensor([rewards(row) for row in rows], device=actor.device).gather(1, sampled)
                relative = advantages(values)
            signal = values.max(1).values != values.min(1).values
            entry.update(sampled=sampled.tolist(), returns=values.tolist(),
                         groups_with_reward_variation=int(signal.sum()))
        for _ in range(PASSES):
            optimizer.zero_grad(set_to_none=True)
            array = logits(actor, inputs)
            policy = logprobs(array, tokens)
            kl = (policy.exp() * (policy - target)).sum(-1).mean()
            if method == "grpo":
                ratio = (policy.gather(1, sampled) - old.gather(1, sampled)).exp()
                objective = -torch.minimum(ratio * relative, ratio.clamp(1 - CLIP, 1 + CLIP) * relative).mean()
            else:
                # Any compatible dish is a valid label. Full-vocabulary
                # normalization also supervises the expected letter format.
                full = array.log_softmax(-1)[:, tokens]
                objective = -torch.stack([line[row["accepted"]].logsumexp(0)
                                          for line, row in zip(full, rows, strict=True)]).mean()
            loss = objective + KL_WEIGHT * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            entry["loss"].append(float(loss.detach()))
        history.append(entry)
        if len(history) % 30 == 0:
            print(output.stem, len(history), "/", (len(keys) + BATCH - 1) // BATCH, flush=True)
    write_rows(output, history)
    return {
        "presentations": len(keys), "updates": len(history) * PASSES,
        "groups_with_reward_variation": sum(item.get("groups_with_reward_variation", 0) for item in history)
        if method == "grpo" else None,
    }


def rank(result, base, epoch):
    qa = result["qa"]["all_orders"]
    replay = result["replay"]["all_orders"]
    allowed = replay["correct"] / replay["cases"] >= base["correct"] / base["cases"] - REPLAY_TOLERANCE
    return allowed, qa["correct"], -qa["conflicts"], -qa["missed_clarifications"], replay["correct"], -epoch


def evaluate_validation(actor, tokens, data, output, name):
    result = {key: evaluate(actor, tokens, packed_rows, output / f"{name}-validation-{key}.jsonl")
              for key, packed_rows in data.items()}
    print(name, "validation", result, flush=True)
    return result


def run(output, device, seed):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    rng = random.Random(seed)
    observed, failures = observed_cases()
    augmented = [case for name in AUGMENTED for case in load_cases(name)]
    targeted = list(dialogue_rows([case for case in observed if case["id"] in failures] + augmented, "train"))
    development = list(dialogue_rows(load_cases("meal_qa_dev"), "validation"))
    old_train = replay_rows("train")
    old_validation = replay_rows("validation")
    replay_train = rng.sample(old_train, len(targeted))
    replay_validation = rng.sample(old_validation, 60)
    observed_rows = list(dialogue_rows(observed, "observed"))
    check_disjoint(development, observed_rows + targeted + old_train + old_validation)
    plan = {
        "seed": seed, "device": device, "epochs": EPOCHS, "batch": BATCH, "eval_batch": EVAL_BATCH, "passes": PASSES,
        "draws": DRAWS, "learning_rate": LR, "temperature": TEMPERATURE, "clip": CLIP,
        "kl_weight": KL_WEIGHT, "replay_tolerance": REPLAY_TOLERANCE,
        "parent_adapter_sha256": file_hash(ARCHIVE / "sft/adapter_model.safetensors"),
        "source_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in (
            Path(__file__), ROOT / "web_demo/meal_policy.py", ROOT / "research/tesis_meal_train.py",
            ROOT / "research/tesis_meal_varied.py", ROOT / "research/tesis_meal_varied_bank.py",)},
        "fixtures": {name: file_hash(DATA / f"{name}.json")
                     for name in (*OBSERVED, *AUGMENTED, "meal_qa_dev", "meal_qa_reserved")},
        "archived_banks": {split: file_hash(ARCHIVE / f"{split}.jsonl")
                           for split in ("train", "validation", "evaluation")},
        "observed_failures": sorted(failures),
        "targeted_conversations": len(failures) + len(augmented),
        "targeted_prefixes": len(targeted), "replay_prefixes": len(replay_train),
        "validation": "QA correctness across orders, conflicts, missed clarifications, archived replay, earliest",
        "reward": {"accepted_choice": 1., "conflict_or_missed_clarification": -1., "unnecessary_clarification": -.35},
        "scope": "single seed, synthetic Spanish dinner conversations, direct categorical Qwen choices",
    }
    output.mkdir(parents=True)
    file_plan = output / "PLAN.json"
    write_json(file_plan, plan)
    actor = load(ARCHIVE / "sft", device)
    tokens = option_tokens(actor.tokenizer)
    train = packed(actor, targeted + replay_train, range(6))
    validation = {"qa": packed(actor, development, range(6)),
                  "replay": packed(actor, replay_validation, (0, 3))}
    initial_weights = state(actor)
    baseline = evaluate_validation(actor, tokens, validation, output, "baseline")
    reference = references(actor, tokens, train)
    starts = {row["id"]: rng.randrange(6) for row in targeted + replay_train}
    schedules = []
    for epoch in range(EPOCHS):
        schedule = [(row["id"], (starts[row["id"]] + 2 * epoch + i) % 6)
                    for row in targeted for i in (0, 1)]
        schedule += [(row["id"], (starts[row["id"]] + epoch) % 6) for row in replay_train]
        rng.shuffle(schedule)
        schedules.append(schedule)

    selected = {"baseline": initial_weights}
    methods = {}
    for method in ARMS:
        reset(actor, initial_weights)
        opt = torch.optim.AdamW([p for p in actor.model.parameters() if p.requires_grad], lr=LR, weight_decay=0.)
        generator = torch.Generator(device=actor.device).manual_seed(seed)
        scores_by_epoch = {0: baseline}
        training = {}
        best, weights = 0, initial_weights
        for epoch, schedule in enumerate(schedules, 1):
            name = f"{method}-{epoch}"
            training[epoch] = optimize(actor, tokens, train, schedule, reference, opt, method, generator,
                                       output / f"{name}-train.jsonl")
            score = evaluate_validation(actor, tokens, validation, output, name)
            scores_by_epoch[epoch] = score
            base = baseline["replay"]["all_orders"]
            if rank(score, base, epoch) > rank(scores_by_epoch[best], base, best):
                best, weights = epoch, state(actor)
        selected[method] = weights
        reset(actor, weights)
        checkpoint = output / method
        actor.save_checkpoint(checkpoint)
        methods[method] = {
            "epoch": best, "validation": scores_by_epoch, "training": training,
            "adapter_sha256": file_hash(checkpoint / "adapter_model.safetensors"),
            "manifest_sha256": file_hash(checkpoint / "local_checkpoint.json"),
        }
        del opt

    seal = output / "SELECTION.json"
    write_json(seal, {"plan_sha256": file_hash(file_plan), "split": "validation", "methods": methods})
    print("Frozen checkpoints", {method: item["epoch"] for method, item in methods.items()}, flush=True)

    # The held-out texts become accessible to the training program here.
    reserved = list(dialogue_rows(load_cases("meal_qa_reserved"), "evaluation"))
    check_disjoint(reserved, observed_rows + targeted + development + old_train + old_validation + replay_rows("evaluation"))
    checks = {
        "reserved": packed(actor, reserved, range(6)),
        "observed": packed(actor, observed_rows, (0,)),
        "archived": packed(actor, replay_rows("evaluation"), range(6)),
    }
    panels = {}
    trace_hashes = {}
    for method, weights in selected.items():
        reset(actor, weights)
        panels[method] = {}
        for key, cases in checks.items():
            path = output / f"{method}-{key}.jsonl"
            panels[method][key] = evaluate(actor, tokens, cases, path)
            trace_hashes[path.name] = file_hash(path)
        print(method, "evaluation", panels[method], flush=True)
    write_json(output / "RESULTS.json", {"selection_sha256": file_hash(seal), "panels": panels,
                                        "trace_sha256": trace_hashes})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run(args.output, args.device, args.seed)
