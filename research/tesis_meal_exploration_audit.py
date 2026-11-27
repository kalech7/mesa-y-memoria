"""Audit six-action exploration, sampled controls, and reserved meal choices."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from math import fsum, isclose, isfinite
from pathlib import Path
from random import Random

from llm_local_001.model import artifact_hashes
from research.src.analysis.paired_statistics import paired_bootstrap
from research.src.artifacts import digest, file_hash
from research.tesis_meal_clarification_audit import metrics, scored_trace, state_of, validation
from research.tesis_meal_qa_audit import ARCHIVE, DATA, ORDERS, ROOT, archived_rows, expand, read, read_rows, rotate
from research.tesis_meal_varied_bank import LETTERS, normalized
from web_demo.meal_policy import recommendation_menu


PRIOR = ROOT / "research/runs/tesis_meal_clarification_v1"
FIXTURES = {split: DATA / f"meal_exploration_{split}.json" for split in ("train", "dev", "reserved")}
PARENT_FIXTURE = DATA / "meal_clarification_train.json"
METHODS = ("sft", "grpo", "grpo_sampled")
REWARDS = {"dish": 1., "ask_exhausted": 1.5, "ask_unknown": 1.,
           "conflict": -2.5, "unnecessary_ask": -.65, "missed_unknown": -1.75}
VIABLE = ("single", "multiple")
COMPARISONS = ("sft", "grpo_sampled", "sft_previo", "grpo_previo")


def validate_plan(run, plan, freeze, results):
    hashes = {"plan_sha256": file_hash(run / "PLAN.json"), "freeze_sha256": file_hash(run / "FREEZE.json"),
              "results_sha256": file_hash(run / "RESULTS.json")}
    if (freeze["plan_sha256"] != hashes["plan_sha256"] or results["freeze_sha256"] != hashes["freeze_sha256"]
            or set(freeze["selection_sha256"]) != {str(seed) for seed in plan["seeds"]}):
        raise ValueError("Changed plan or checkpoint freeze")
    if (tuple(plan["methods"]) != METHODS or plan["reward"] != REWARDS
            or len(plan["seeds"]) != 3 or len(set(plan["seeds"])) != 3
            or plan["default_seed"] != plan["seeds"][0] or not 0 < plan["epsilon"] < 1
            or plan["scale"] <= 0 or plan["temperature"] <= 0 or plan["draws"] < 1
            or plan["menu_sha256"] != digest(recommendation_menu([], "dinner_preference", shuffle=False))):
        raise ValueError("Invalid registered protocol")
    sources = ("research/tesis_meal_exploration.py", "research/tesis_meal_clarification.py",
               "research/tesis_meal_qa.py", "research/tesis_meal_train.py",
               "research/tesis_meal_varied_bank.py", "web_demo/meal_policy.py")
    required = {*FIXTURES.values(), PARENT_FIXTURE, PRIOR / "FREEZE.json", *(ROOT / name for name in sources),
                *(ARCHIVE / f"{split}.jsonl" for split in ("train", "validation", "evaluation"))}
    for seed in plan["seeds"]:
        for method in ("sft", "grpo"):
            folder = PRIOR / str(seed) / method
            required.update(folder / name for name in ("adapter_model.safetensors", "local_checkpoint.json"))
    if not {str(path.relative_to(ROOT)) for path in required} <= set(plan["inputs"]):
        raise ValueError("A frozen input is missing")
    for name, checksum in plan["inputs"].items():
        if file_hash(ROOT / name) != checksum:
            raise ValueError(f"Frozen input changed: {name}")
    prior_audit = read(PRIOR / "AUDIT.json")
    prior_freeze = read(PRIOR / "FREEZE.json")
    if prior_audit["status"] != "verified" or prior_audit["freeze_sha256"] != file_hash(PRIOR / "FREEZE.json"):
        raise ValueError("Previous experiment freeze was not independently audited")
    parents = {}
    for seed in plan["seeds"]:
        file = PRIOR / str(seed) / "SELECTION.json"
        if file_hash(file) != prior_freeze["selection_sha256"][str(seed)]:
            raise ValueError(f"Previous seed selection changed: {seed}")
        choices = read(file)["methods"]
        for method in ("sft", "grpo"):
            folder = PRIOR / str(seed) / method
            checkpoint(folder, choices[method])
            parents[seed, method] = folder
    return hashes, parents


def fixture_messages(document):
    cases = document if isinstance(document, list) else document["cases"]
    return {normalized([turn["text"]]) for case in cases for turn in case["turns"]}


def data_splits(frozen=False):
    fixtures = {name: read(path) for name, path in FIXTURES.items() if name != "reserved" or frozen}
    fixtures["prior"] = read(PARENT_FIXTURE)
    ids = [case["id"] for histories in fixtures.values() for case in histories]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated conversation IDs")
    words = {name: fixture_messages(cases) for name, cases in fixtures.items()}
    for a, b in combinations(words, 2):
        if words[a] & words[b]:
            raise ValueError(f"Repeated text across {a} and {b}")
    archive = {name: archived_rows(name) for name in ("train", "validation", "evaluation")}
    archived_words = {normalized([text]) for rows in archive.values() for row in rows for text in row["turns"]}
    for key in ("train", "dev", "reserved"):
        if words.get(key, set()) & archived_words:
            raise ValueError(f"Shared wording between exploration {key} and archived bank")
    if frozen:
        older = set()
        for pattern in ("meal_qa_*.json", "meal_clarification_*.json", "meal_exploration_exploratory.json"):
            for path in sorted(DATA.glob(pattern)):
                older.update(fixture_messages(read(path)))
        if words["reserved"] & older:
            raise ValueError("Reserved prompt appeared in an earlier fixture")
    split = {key: expand(histories) for key, histories in fixtures.items()}
    for name, roots in split.items():
        explicit = [bool(turn.get("clarify")) for case in fixtures[name] for turn in case["turns"]]
        if any(flag != (state_of(row) == "unknown") for flag, row in zip(explicit, roots, strict=True)):
            raise ValueError(f"Fixture distinguishes unknown from exhaustion incorrectly: {name}")
    return split, archive


def checkpoint(folder, selected, parent=None):
    file = folder / "local_checkpoint.json"
    adapter_hash = file_hash(folder / "adapter_model.safetensors")
    manifest = read(file)
    if (adapter_hash != selected["adapter_sha256"] or file_hash(file) != selected["manifest_sha256"]
            or manifest["artifacts_sha256"] != artifact_hashes(folder)
            or manifest["parameters"]["training_scope"] != "lora_adapters_integrated_in_llm"):
        raise ValueError(f"Invalid adapter receipt: {folder}")
    if parent is not None:
        previous = read(parent / "local_checkpoint.json")
        origin = parent.relative_to(ROOT).parts
        if (Path(manifest["origin_checkpoint"]).parts[-len(origin):] != origin
                or manifest["base_artifacts_sha256"] != previous["base_artifacts_sha256"]
                or (selected["epoch"] == 0 and adapter_hash != previous["artifacts_sha256"]["adapter_model.safetensors"])):
            raise ValueError(f"Wrong initialization or epoch-zero selection: {folder}")


def rank(value, initial, epoch, tolerance):
    qa = value["qa"]["all_orders"]
    states = qa["states"]
    ask = states["exhausted"]["correct"] / states["exhausted"]["n"]
    choose = sum(states[key]["correct"] for key in VIABLE) / sum(states[key]["n"] for key in VIABLE)
    old, new = (value["replay"]["all_orders"] for value in (initial, value))
    eligible = new["correct"] / new["cases"] >= old["correct"] / old["cases"] - tolerance
    return eligible, (ask + choose) / 2, -qa["conflicts"], states["unknown"]["correct"], -epoch


def schedule(plan, seed, split, archive):
    rng = Random(seed)
    prior = rng.sample(split["prior"], plan["prior_prefixes_per_seed"])
    replay = rng.sample(archive["train"], plan["archive_prefixes_per_seed"])
    cases = split["train"] + prior + replay
    offsets = [rng.randrange(len(ORDERS)) for _ in cases]
    orders = []
    for epoch in range(plan["epochs"]):
        order = [[row["id"], (offset + epoch) % len(ORDERS)] for row, offset in zip(cases, offsets, strict=True)]
        rng.shuffle(order)
        orders.append(order)
    return {row["id"]: row for row in cases}, orders


def action_rewards(row):
    state = state_of(row)
    allowed = set(row["accepted"])
    excluded = set(row["forbidden"])
    rewards = []
    for index, option in enumerate(row["options"]):
        ingredients = set(option["ingredients"])
        if index in allowed:
            tag = "dish" if ingredients else f"ask_{state}"
        elif not ingredients:
            tag = "unnecessary_ask"
        elif ingredients & excluded:
            tag = "conflict"
        elif state == "unknown":
            tag = "missed_unknown"
        else:
            raise ValueError(f"Inconsistent feasible label for {row['id']}")
        rewards.append(REWARDS[tag])
    return rewards


def training_stats(path, plan, expected_schedule, cases, method):
    ledger = read_rows(path)
    chunks = [expected_schedule[i:i + plan["batch"]] for i in range(0, len(expected_schedule), plan["batch"])]
    if [entry["cases"] for entry in ledger] != chunks:
        raise ValueError(f"Unequal exposure: {path}")
    sampling = defaultdict(Counter)
    first_group_q = None
    min_probability = 1.
    minimum = plan["epsilon"] / len(ORDERS)
    maximum = 1. - plan["epsilon"] + minimum
    for batch_index, entry in enumerate(ledger):
        if len(entry["loss"]) != plan["passes"] or not all(map(isfinite, entry["loss"])):
            raise ValueError(f"Invalid optimizer steps: {path}")
        if method == "sft":
            continue
        batch = zip(entry["cases"], entry["actions"], entry["rewards"], entry["old_probabilities"], strict=True)
        expected_objectives = []
        for (case, rotation), actions, rewards, q in batch:
            row = rotate(cases[case], rotation)
            if (len(q) != len(ORDERS) or not isclose(fsum(q), 1., abs_tol=2e-6)
                    or any(not minimum - 2e-6 <= value <= maximum + 2e-6 for value in q)):
                raise ValueError(f"Invalid frozen exploration distribution: {path} / {case}")
            min_probability = min(min_probability, *q)
            gold = action_rewards(row)
            if (any(not isclose(a, b, abs_tol=1e-7) for a, b in zip(gold, rewards, strict=True))
                    or any(type(action) is not int or action not in ORDERS for action in actions)
                    or (method == "grpo" and actions != list(ORDERS))
                    or (method == "grpo_sampled" and len(actions) != plan["draws"])):
                raise ValueError(f"Invalid group rewards or action coverage: {path} / {case}")
            center = fsum(probability * reward for probability, reward in zip(q, rewards, strict=True))
            group = sampling[state_of(row)]
            group["groups"] += 1
            group["varied"] += len({gold[action] for action in actions}) > 1
            group["with_ask"] += any(not row["options"][action]["ingredients"] for action in actions)
            group["all_actions"] += len(set(actions)) == len(ORDERS)
            # The ratio is one before each batch's first update.
            if method == "grpo":
                expected_objectives.append(-fsum(p * (r - center) for p, r in zip(q, rewards, strict=True)) /
                                           plan["scale"])
            else:
                expected_objectives.append(-fsum(rewards[i] - center for i in actions) /
                                           (len(actions) * plan["scale"]))
        objective = fsum(expected_objectives) / len(expected_objectives)
        if entry["loss"][0] < objective - 2e-5:
            raise ValueError(f"First-pass surrogate or KL failed reconstruction: {path}")
        if batch_index == 0 and path.name.endswith("-1-train.jsonl"):
            # Reference logits use larger BF16 batches, so the initial
            # forward KL can be slightly above zero even at matching weights.
            if not isclose(entry["loss"][0], objective, abs_tol=1e-3):
                raise ValueError(f"Initial update did not use the frozen parent: {path}")
            first_group_q = entry["old_probabilities"]
    summary = {"presentations": len(expected_schedule), "updates": len(ledger) * plan["passes"],
               "sampling": {state: dict(counts) for state, counts in sampling.items()}}
    return summary, first_group_q, min_probability if sampling else None


def select(run, plan, freeze, parents, split, archive):
    dev = {"qa": (split["dev"], tuple(plan["validation_orders"])),
           "replay": (Random(731).sample(archive["validation"], plan["validation_replay_count"]), (0, 3))}
    selected = {}
    for seed in plan["seeds"]:
        directory = run / str(seed)
        receipt = directory / "SELECTION.json"
        selection = read(receipt)
        if (selection["seed"] != seed or selection["split"] != "validation"
                or selection["plan_sha256"] != file_hash(run / "PLAN.json")
                or freeze["selection_sha256"][str(seed)] != file_hash(receipt)
                or set(selection["methods"]) != set(METHODS)):
            raise ValueError(f"Invalid seed declaration: {seed}")
        initial = validation(directory, "base", dev)
        if initial != read(directory / "BASE_VALIDATION.json"):
            raise ValueError(f"Baseline validation mismatch: {seed}")
        bank, orders = schedule(plan, seed, split, archive)
        selected[str(seed)] = {}
        starting_q = {}
        for method, choice in selection["methods"].items():
            val, telemetry, minima = {0: initial}, {}, []
            for epoch, order in enumerate(orders, 1):
                stem = f"{method}-{epoch}"
                stats, first_q, min_q = training_stats(directory / f"{stem}-train.jsonl", plan, order, bank, method)
                if first_q is not None:
                    starting_q[method] = first_q
                if min_q is not None:
                    minima.append(min_q)
                telemetry[str(epoch)] = stats
                val[epoch] = validation(directory, stem, dev)
            winner = max(val, key=lambda e: rank(val[e], initial, e, plan["replay_tolerance"]))
            if (winner != choice["epoch"] or read(directory / f"{method}-selection.json") != choice
                    or choice["training"] != telemetry or choice["validation"] != {str(e): v for e, v in val.items()}):
                raise ValueError(f"Validation failed to reproduce selected epoch: {seed}/{method}")
            checkpoint(directory / method, choice, parents[seed, "sft"])
            selected[str(seed)][method] = {"epoch": winner, "training": telemetry,
                                           "minimum_exploration_probability": min(minima) if minima else None}
        exact, sampled = starting_q["grpo"], starting_q["grpo_sampled"]
        if any(not isclose(a, b, abs_tol=2e-6)
               for left, right in zip(exact, sampled, strict=True) for a, b in zip(left, right, strict=True)):
            raise ValueError(f"Matched GRPO arms did not share the same first behavior policy: {seed}")
    return selected


def prediction_modes(ledger):
    counts = defaultdict(Counter)
    for row in ledger:
        counts[str(row["order"])][row["letter"]] += 1
    order_zero = [row for row in ledger if row["order"] == 0]
    return {"letters_by_rotation": {order: dict(numbers) for order, numbers in counts.items()},
            "live_dishes": dict(Counter(row["choice"] for row in order_zero)),
            "live_raw_actions": dict(Counter(row["raw_token"] for row in order_zero))}


def paired(ledgers, seeds):
    rows = []
    for seed in seeds:
        for method in (*METHODS, "sft_previo", "grpo_previo"):
            for row in ledgers[f"{seed}-{method}"]:
                rows.append({"method": method, "seed": seed, "root_id": row["conversation"], "family": row["family"],
                             "realization_id": f"{row['turn']}:{row['order']}", "success": row["success"],
                             "order": row["order"]})
    comparisons = {}
    for rival in COMPARISONS:
        comparisons[f"grpo_minus_{rival}"] = {
            name: paired_bootstrap(subset, "grpo", rival, "success", confidence=.95, resamples=2000,
                                   resample_seeds=True)
            for name, subset in (("live_order", [row for row in rows if row["order"] == 0]), ("all_orders", rows))
        }
    return comparisons


def audit(run):
    plan, freeze, official = (read(run / name) for name in ("PLAN.json", "FREEZE.json", "RESULTS.json"))
    hashes, parents = validate_plan(run, plan, freeze, official)
    split, archive = data_splits(frozen=True)
    if len(split["train"]) != plan["targeted_prefixes"]:
        raise ValueError("Incorrect registered training size")
    selected = select(run, plan, freeze, parents, split, archive)
    panels = {"reserved": (split["reserved"], ORDERS),
              "archived": (Random(928).sample(archive["evaluation"], 54), (0, 3))}
    profiles = [f"{seed}-{method}" for seed in plan["seeds"] for method in (*METHODS, "sft_previo", "grpo_previo")]
    if set(profiles) != set(official["profiles"]):
        raise ValueError("Missing matched comparison profile")
    trace_hashes, results, ledgers, modes = {}, {}, {}, {}
    for profile in profiles:
        results[profile] = {}
        for panel, (roots, orders) in panels.items():
            filename = f"{profile}-{panel}.jsonl"
            path = run / "evaluation" / filename
            ledger = scored_trace(path, roots, orders)
            results[profile][panel] = metrics(ledger)
            trace_hashes[filename] = file_hash(path)
            if panel == "reserved":
                ledgers[profile] = ledger
                modes[profile] = prediction_modes(ledger)
    if trace_hashes != official["traces"] or results != official["profiles"]:
        raise ValueError("Official test scores or trace hashes changed")
    return {"status": "verified", **hashes, "audit_source_sha256": file_hash(Path(__file__)),
            "selection_sha256": freeze["selection_sha256"], "selections": selected,
            "profiles": results, "paired": paired(ledgers, plan["seeds"]), "prediction_modes": modes,
            "split_counts": {name: dict(Counter(state_of(row) for row in roots)) for name, roots in split.items()},
            "scope": "first-pass losses, matched replay and exact/sampled weights independently checked from logs; "
                     "paired intervals resample seeds and whole histories with equal family weights"}


def main():
    from research.src.artifacts import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = audit(args.run.resolve())
    temporary = args.run / "AUDIT.tmp"
    write_json(temporary, report)
    temporary.replace(args.run / "AUDIT.json")
    print(f"Verified {args.run}", flush=True)


if __name__ == "__main__":
    main()
