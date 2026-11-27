"""Independently audit the clarification experiment's plans, choices, and returns."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from math import isclose, isfinite
from pathlib import Path
from random import Random

from llm_local_001.model import artifact_hashes
from research.src.analysis.paired_statistics import paired_bootstrap
from research.src.artifacts import digest, file_hash, write_json
from research.tesis_meal_qa_audit import (
    ARCHIVE, DATA, ORDERS, ROOT, archived_rows, expand, option_names, panel, read, read_rows,
    rotate, verified_trace,
)
from research.tesis_meal_varied_bank import normalized
from web_demo.meal_policy import recommendation_menu


PREVIOUS = ROOT / "research/runs/tesis_meal_qa_v1"
FIXTURES = {split: DATA / f"meal_clarification_{split}.json" for split in ("train", "dev", "reserved")}
BASELINES = {"sft_previo": PREVIOUS / "sft", "grpo_previo": PREVIOUS / "grpo"}
METHODS = ("sft", "grpo", "grpo_control")
STATES = ("exhausted", "unknown", "single", "multiple")
VIABLE = {"single", "multiple"}
REVISED_REWARD = {"valid": 1., "exhausted_dish": -2., "missed_unknown": -1.5,
                  "unnecessary_ask": -1.25, "incompatible_dish": -1.}
CONTROL_REWARD = (1., -1., -.35)
TRANSITIONS = ("exhausted_to_viable", "viable_to_exhausted")


def state_of(row):
    accepted = [row["options"][index] for index in row["accepted"]]
    if accepted[0]["ingredients"]:
        return "single" if len(accepted) == 1 else "multiple"
    feasible_before_unknown = any(option["ingredients"] and
                                  set(option["ingredients"]).isdisjoint(row["forbidden"])
                                  for option in row["options"])
    return "unknown" if feasible_before_unknown else "exhausted"


def split_rows():
    fixtures = {split: read(path) for split, path in FIXTURES.items()}
    cases = [case["id"] for fixture in fixtures.values() for case in fixture]
    if len(cases) != len(set(cases)):
        raise ValueError("Reused conversation ID across clarification partitions")
    words = {split: {normalized([turn["text"]]) for case in fixture for turn in case["turns"]}
             for split, fixture in fixtures.items()}
    for a, b in combinations(words, 2):
        if words[a] & words[b]:
            raise ValueError(f"Overlapping texts: {a}, {b}")
    archived = {split: archived_rows(split) for split in ("train", "validation", "evaluation")}
    old_words = {normalized([text]) for rows in archived.values() for row in rows for text in row["turns"]}
    for split in words:
        if words[split] & old_words:
            raise ValueError(f"Clarification prompts overlap archived bank: {split}")
    previous_paths = sorted(DATA.glob("meal_qa_*.json")) + [DATA / "meal_clarification_exploratory.json"]
    for path in previous_paths:
        document = read(path)
        cases = document if isinstance(document, list) else document["cases"]
        old_text = {normalized([turn["text"]]) for case in cases for turn in case["turns"]}
        if words["reserved"] & old_text:
            raise ValueError(f"Reserved prompt was previously used in {path.name}")

    current = {}
    for split, fixture in fixtures.items():
        rows = expand(fixture)
        gold = [entry for case in fixture for entry in case["turns"]]
        if any(bool(entry.get("clarify")) != (state_of(row) == "unknown")
               for entry, row in zip(gold, rows, strict=True)):
            raise ValueError(f"Explicit clarification flags must refer to unknown ingredients: {split}")
        current[split] = rows
    return current, archived


def validate_plan(run, plan, freeze, results):
    plan_hash = file_hash(run / "PLAN.json")
    if (freeze["plan_sha256"] != plan_hash or results["freeze_sha256"] != file_hash(run / "FREEZE.json")
            or set(freeze["selection_sha256"]) != {str(seed) for seed in plan["seeds"]}):
        raise ValueError("Invalid plan/selection freeze chain")
    required = {*FIXTURES.values(), *(path / "adapter_model.safetensors" for path in BASELINES.values()),
                *(ARCHIVE / f"{split}.jsonl" for split in ("train", "validation", "evaluation")),
                *(ROOT / name for name in ("research/tesis_meal_clarification.py", "research/tesis_meal_qa.py",
                                           "research/tesis_meal_train.py", "research/tesis_meal_varied_bank.py",
                                           "web_demo/meal_policy.py"))}
    if not {str(path.relative_to(ROOT)) for path in required} <= set(plan["inputs"]):
        raise ValueError("Required source, fixture or checkpoint missing from the plan")
    for name, checksum in plan["inputs"].items():
        if file_hash(ROOT / name) != checksum:
            raise ValueError(f"Frozen input changed: {name}")
    if (plan["menu_sha256"] != digest(recommendation_menu([], "dinner_preference", shuffle=False))
            or plan["reward"] != REVISED_REWARD or tuple(plan["control_reward"]) != CONTROL_REWARD
            or plan["default_seed"] != plan["seeds"][0] or len(set(plan["seeds"])) != len(plan["seeds"])):
        raise ValueError("Invalid menu, seed or reward configuration")
    previous_selection = read(PREVIOUS / "SELECTION.json")
    previous_audit = read(PREVIOUS / "AUDIT.json")
    if (previous_audit["status"] != "verified"
            or previous_audit["selection_sha256"] != file_hash(PREVIOUS / "SELECTION.json")):
        raise ValueError("Parent checkpoints require their verified selection")
    for method, path in BASELINES.items():
        recorded = previous_selection["methods"][method.removesuffix("_previo")]
        if (file_hash(path / "adapter_model.safetensors") != recorded["adapter_sha256"]
                or file_hash(path / "local_checkpoint.json") != recorded["manifest_sha256"]):
            raise ValueError(f"Prior model mismatch: {method}")


def scored_trace(path, roots, orders):
    rows = verified_trace(path, roots, orders)
    by_id = {root["id"]: root for root in roots}
    for row in rows:
        source = by_id[row["id"]]
        rotated = rotate(source, row["order"])
        dish = rotated["options"][option_names(rotated).index(row["choice"])]
        if row["state"] != state_of(source) or row["asked"] != (not dish["ingredients"]):
            raise ValueError(f"Invalid cohort or clarification action: {path.name} / {row['id']}")
    return rows


def state_metrics(rows):
    tables = {state: {"n": 0, "correct": 0, "asked": 0, "conflicts": 0} for state in STATES}
    grouped = defaultdict(list)
    for row in rows:
        cell = tables[row["state"]]
        cell["n"] += 1
        cell["correct"] += row["success"]
        cell["asked"] += row["asked"]
        cell["conflicts"] += row["conflict"]
        grouped[row["conversation"], row["order"]].append(row)
    transitions = {name: {"n": 0, "correct": 0} for name in TRANSITIONS}
    for history in grouped.values():
        history.sort(key=lambda row: row["turn"])
        for before, after in zip(history, history[1:]):
            tags = before["state"], after["state"]
            if tags[0] == "exhausted" and tags[1] in VIABLE:
                key = "exhausted_to_viable"
            elif tags[0] in VIABLE and tags[1] == "exhausted":
                key = "viable_to_exhausted"
            else:
                continue
            transitions[key]["n"] += 1
            transitions[key]["correct"] += bool(before["success"] and after["success"])
    return {"states": tables, "transitions": transitions}


def metrics(rows):
    result = panel(rows)
    result["live_order"].update(state_metrics([row for row in rows if row["order"] == 0]))
    result["all_orders"].update(state_metrics(rows))
    return result


def validation(run, prefix, data):
    return {key: metrics(scored_trace(run / f"{prefix}-val-{key}.jsonl", rows, orders))
            for key, (rows, orders) in data.items()}


def rank(candidate, initial, epoch, tolerance):
    scores = candidate["qa"]["all_orders"]["states"]
    older = candidate["replay"]["all_orders"]
    base = initial["replay"]["all_orders"]
    exhausted = scores["exhausted"]["correct"] / scores["exhausted"]["n"]
    viable = sum(scores[key]["correct"] for key in VIABLE) / sum(scores[key]["n"] for key in VIABLE)
    return (older["correct"] / older["cases"] >= base["correct"] / base["cases"] - tolerance,
            (exhausted + viable) / 2, min(exhausted, viable),
            scores["unknown"]["correct"] - scores["unknown"]["n"], -epoch)


def seed_schedule(plan, seed, targeted, archived):
    rng = Random(seed)
    replay = rng.sample(archived, plan["replay_prefixes_per_seed"])
    training = targeted + replay
    offsets = [rng.randrange(len(ORDERS)) for _ in training]
    orders = []
    for epoch in range(plan["epochs"]):
        menu_order = [[row["id"], (offset + epoch) % len(ORDERS)]
                      for row, offset in zip(training, offsets, strict=True)]
        rng.shuffle(menu_order)
        orders.append(menu_order)
    return {row["id"]: row for row in training}, orders


def reward(row, action, method):
    accepted = action in row["accepted"]
    is_ask = not row["options"][action]["ingredients"]
    if method == "grpo_control":
        valid, invalid, unnecessary = CONTROL_REWARD
        return valid if accepted else unnecessary if is_ask else invalid
    if accepted:
        return REVISED_REWARD["valid"]
    if is_ask:
        return REVISED_REWARD["unnecessary_ask"]
    penalties = {"exhausted": "exhausted_dish", "unknown": "missed_unknown"}
    return REVISED_REWARD[penalties.get(state_of(row), "incompatible_dish")]


def training_metrics(path, plan, scheduled, cases, method):
    rows = read_rows(path)
    batches = [scheduled[i:i + plan["batch"]] for i in range(0, len(scheduled), plan["batch"])]
    if [entry["cases"] for entry in rows] != batches:
        raise ValueError(f"Training exposure differs: {path.name}")
    sampling = defaultdict(Counter)
    for entry in rows:
        if len(entry["loss"]) != plan["passes"] or not all(map(isfinite, entry["loss"])):
            raise ValueError(f"Invalid optimization steps: {path.name}")
        if method == "sft":
            continue
        groups = zip(entry["cases"], entry["sampled"], entry["rewards"], strict=True)
        for (row_id, order), actions, returns in groups:
            case = rotate(cases[row_id], order)
            if len(actions) != plan["draws"] or any(type(action) is not int or action not in ORDERS
                                                   for action in actions):
                raise ValueError(f"Wrong number of valid policy samples: {path.name}")
            expected = [reward(case, action, method) for action in actions]
            if any(not isclose(a, b, abs_tol=1e-7) for a, b in zip(returns, expected, strict=True)):
                raise ValueError(f"Reward audit failed: {path.name} / {row_id}")
            group = sampling[state_of(case)]
            group["groups"] += 1
            group["varying"] += len(set(expected)) > 1
            group["with_ask"] += any(not case["options"][action]["ingredients"] for action in actions)
    return {"presentations": len(scheduled), "updates": len(rows) * plan["passes"],
            "sampling": {state: dict(counts) for state, counts in sampling.items()}}


def check_checkpoint(path, parent, chosen):
    manifest = path / "local_checkpoint.json"
    content = read(manifest)
    digest_weights = file_hash(path / "adapter_model.safetensors")
    origin = BASELINES["sft_previo"].relative_to(ROOT).parts
    if (digest_weights != chosen["adapter_sha256"] or file_hash(manifest) != chosen["manifest_sha256"]
            or content["artifacts_sha256"] != artifact_hashes(path)
            or content["base_artifacts_sha256"] != parent["base_artifacts_sha256"]
            or content["parameters"]["training_scope"] != "lora_adapters_integrated_in_llm"
            or Path(content["origin_checkpoint"]).parts[-len(origin):] != origin
            or chosen["epoch"] == 0 and digest_weights != parent["artifacts_sha256"]["adapter_model.safetensors"]):
        raise ValueError(f"Checkpoint does not match validation selection: {path}")


def verify_selection(run, plan, freeze, current, archived):
    dev = {"qa": (current["dev"], (0, 2, 4)),
           "replay": (Random(1042).sample(archived["validation"], 36), (0, 3))}
    baseline = validation(run, "baseline", dev)
    if baseline != read(run / "BASE_VALIDATION.json"):
        raise ValueError("Comparison did not use the recorded baseline validation")
    selections = {}
    parent = read(BASELINES["sft_previo"] / "local_checkpoint.json")
    for seed in plan["seeds"]:
        folder = run / str(seed)
        path = folder / "SELECTION.json"
        selected = read(path)
        if (selected["seed"] != seed or selected["plan_sha256"] != file_hash(run / "PLAN.json")
                or freeze["selection_sha256"][str(seed)] != file_hash(path)
                or selected["split"] != "validation" or set(selected["methods"]) != set(METHODS)):
            raise ValueError(f"Invalid frozen seed selection: {seed}")
        training, schedule = seed_schedule(plan, seed, current["train"], archived["train"])
        selections[str(seed)] = {}
        for method, chosen in selected["methods"].items():
            candidates, training_stats = {0: baseline}, {}
            for epoch, order in enumerate(schedule, 1):
                prefix = f"{method}-{epoch}"
                candidates[epoch] = validation(folder, prefix, dev)
                training_stats[str(epoch)] = training_metrics(folder / f"{prefix}-train.jsonl", plan, order,
                                                               training, method)
            winner = max(candidates, key=lambda e: rank(candidates[e], baseline, e, plan["replay_tolerance"]))
            if (chosen["epoch"] != winner or read(folder / f"{method}-selection.json") != chosen
                    or chosen["training"] != training_stats
                    or chosen["validation"] != {str(e): value for e, value in candidates.items()}):
                raise ValueError(f"Validation-only checkpoint replay differs: {seed}/{method}")
            check_checkpoint(folder / method, parent, chosen)
            selections[str(seed)][method] = {"epoch": winner, "training": training_stats}
    return selections


def paired_run_summary(ledgers, seeds):
    rows = []
    for seed in seeds:
        for method in METHODS:
            for row in ledgers[f"{seed}-{method}"]["reserved"]:
                rows.append({"method": method, "seed": seed, "root_id": row["conversation"],
                             "family": row["family"], "realization_id": f"{row['turn']}:{row['order']}",
                             "condition": "live_order" if row["order"] == 0 else "rotated",
                             "success": row["success"]})
    comparisons = {}
    for rival in ("sft", "grpo_control"):
        data = [{**row, "condition": "all_orders"} for row in rows]
        data.extend(row for row in rows if row["condition"] == "live_order")
        comparisons[f"grpo_minus_{rival}"] = {
            order: paired_bootstrap(data, "grpo", rival, "success", condition=order, confidence=.95, resamples=2000,
                                    resample_seeds=True)
            for order in ("live_order", "all_orders")
        }
    return comparisons


def training_contrasts(panels, seeds):
    comparisons = {}
    for rival in ("sft", "grpo_control"):
        comparisons[f"grpo_minus_{rival}"] = {}
        for order in ("live_order", "all_orders"):
            by_seed = {}
            for seed in seeds:
                contender = panels[f"{seed}-grpo"]["reserved"][order]
                alternative = panels[f"{seed}-{rival}"]["reserved"][order]
                differences = {}
                for tag in STATES:
                    a, b = (panel["states"][tag] for panel in (contender, alternative))
                    differences[tag] = (a["correct"] - b["correct"]) / a["n"]
                a, b = (panel["states"] for panel in (contender, alternative))
                viable_delta = (sum(a[tag]["correct"] - b[tag]["correct"] for tag in VIABLE)
                                / sum(a[tag]["n"] for tag in VIABLE))
                differences["viable"] = viable_delta
                differences["balanced_exhausted_viable"] = (differences["exhausted"] + viable_delta) / 2
                by_seed[str(seed)] = differences
            comparisons[f"grpo_minus_{rival}"][order] = by_seed
    return comparisons


def audit(run):
    plan, freeze, official = (read(run / name) for name in ("PLAN.json", "FREEZE.json", "RESULTS.json"))
    validate_plan(run, plan, freeze, official)
    current, archived = split_rows()
    if len(current["train"]) != plan["targeted_prefixes"]:
        raise ValueError("Targeted training prefix count differs")
    selections = verify_selection(run, plan, freeze, current, archived)
    data = {"reserved": (current["reserved"], ORDERS),
            "archived": (Random(991).sample(archived["evaluation"], 54), (0, 3))}
    profiles = [*BASELINES, *(f"{seed}-{method}" for seed in plan["seeds"] for method in METHODS)]
    if set(profiles) != set(official["profiles"]):
        raise ValueError("Missing or unregistered profile")
    observed, hashes, ledgers = {}, {}, {}
    for profile in profiles:
        ledgers[profile] = {}
        for panel_name, (roots, orders) in data.items():
            filename = f"{profile}-{panel_name}.jsonl"
            trace = run / "evaluation" / filename
            ledgers[profile][panel_name] = scored_trace(trace, roots, orders)
            hashes[filename] = file_hash(trace)
        observed[profile] = {name: metrics(rows) for name, rows in ledgers[profile].items()}
    if observed != official["profiles"] or hashes != official["traces"]:
        raise ValueError("Held-out score reconstruction or trace hash failed")
    return {"status": "verified", "plan_sha256": file_hash(run / "PLAN.json"),
            "freeze_sha256": file_hash(run / "FREEZE.json"),
            "results_sha256": file_hash(run / "RESULTS.json"), "audit_source_sha256": file_hash(Path(__file__)),
            "selection_sha256": freeze["selection_sha256"], "selections": selections, "profiles": observed,
            "paired": paired_run_summary(ledgers, plan["seeds"]),
            "per_seed_state_accuracy_differences": training_contrasts(observed, plan["seeds"]),
            "split_counts": {split: dict(Counter(state_of(row) for row in rows))
                             for split, rows in current.items()},
            "scope": "paired seed and conversation bootstrap, family macroaverage on fixed synthetic histories"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    output = args.run.resolve()
    report = audit(output)
    temp = output / "AUDIT.tmp"
    write_json(temp, report)
    temp.replace(output / "AUDIT.json")
    print(f"Verified {output}", flush=True)


if __name__ == "__main__":
    main()
