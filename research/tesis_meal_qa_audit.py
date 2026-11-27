"""Rebuild the QA meal experiment's splits, schedules, selection and scores."""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import json
from math import isclose, isfinite
from pathlib import Path
import random

from llm_local_001.model import artifact_hashes
from research.src.analysis.paired_statistics import paired_bootstrap
from research.src.artifacts import digest, file_hash, write_json
from research.tesis_meal_varied_bank import LETTERS, MENUS, normalized
from web_demo.meal_policy import recommendation_menu


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research/data"
ARCHIVE = ROOT / "research/runs/tesis_meal_lora_varied_v1"
METHODS = ("sft", "grpo")
BASELINE_IDS = {f"comidas-variado-{method}" for method in METHODS}
OBSERVED = ("meal_qa_typos", "meal_qa_corrections")
AUGMENTED = ("meal_qa_augment", "meal_qa_augment_typos")
FIXTURES = (*OBSERVED, *AUGMENTED, "meal_qa_dev", "meal_qa_reserved")
SOURCES = {
    "research/tesis_meal_qa.py", "web_demo/meal_policy.py", "research/tesis_meal_train.py",
    "research/tesis_meal_varied.py", "research/tesis_meal_varied_bank.py",
}
ORDERS = range(len(LETTERS))
REPLAY_INGREDIENTS = {food for menu in MENUS.values() for _, recipe in menu for food in recipe.split()}
REWARD = {"accepted_choice": 1., "conflict_or_missed_clarification": -1.,
          "unnecessary_clarification": -.35}


def read(path):
    return json.loads(path.read_bytes())


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def option_names(row):
    return [item["name"] for item in row["options"]]


def expected(row, action):
    chosen = row["options"][action]
    clarify = not chosen["ingredients"]
    needs_clarification = not row["options"][row["accepted"][0]]["ingredients"]
    return {
        "success": int(action in row["accepted"]),
        "conflict": int(bool(set(chosen["ingredients"]) & set(row["forbidden"]))),
        "unnecessary_clarification": int(clarify and not needs_clarification),
        "missed_clarification": int(needs_clarification and not clarify),
    }


def reward(row, action):
    metrics = expected(row, action)
    if metrics["success"]:
        return REWARD["accepted_choice"]
    if metrics["conflict"] or metrics["missed_clarification"]:
        return REWARD["conflict_or_missed_clarification"]
    return REWARD["unnecessary_clarification"]


def acceptance(menu, excluded, unspecified):
    feasible = [i for i, item in enumerate(menu) if item["ingredients"] and
                set(item["ingredients"]).isdisjoint(excluded)]
    clarify = [i for i, item in enumerate(menu) if not item["ingredients"]]
    if len(clarify) != 1 or len(menu) != len(LETTERS) or len({item['name'] for item in menu}) != len(menu):
        raise ValueError("Invalid menu")
    return clarify if unspecified or not feasible else feasible


def expand(cases):
    prefixes = []
    for case in cases:
        if len(case["turns"]) < 3 or not (case.get("family") or case.get("category")):
            raise ValueError(f"Invalid conversation: {case['id']}")
        visible = []
        for number, entry in enumerate(case["turns"], 1):
            text = entry["text"]
            visible.append(text)
            menu = recommendation_menu(visible, "dinner_preference", shuffle=True)
            # A user can exclude a food absent from the live menu. It has no
            # impact on compatibility, but should still be a known ingredient.
            known = REPLAY_INGREDIENTS | {food for option in menu for food in option["ingredients"]}
            excluded = set(entry["excluded"])
            if not normalized([text]) or not excluded <= known or len(excluded) != len(entry["excluded"]):
                raise ValueError(f"Invalid state: {case['id']} turn {number}")
            accepted = acceptance(menu, excluded, entry.get("clarify", False))
            if "accepted" in entry and set(entry["accepted"]) != {menu[i]["name"] for i in accepted}:
                raise ValueError(f"Independently authored outcomes disagree: {case['id']} turn {number}")
            prefixes.append({"id": f"{case['id']}:{number}", "conversation": case["id"], "turn": number,
                             "family": case.get("family") or case["category"], "turns": visible.copy(),
                             "options": menu, "accepted": accepted, "forbidden": sorted(excluded)})
    return prefixes


def rotate(row, offset):
    indices = list(ORDERS)
    indices = indices[offset:] + indices[:offset]
    return {**row, "options": [row["options"][i] for i in indices],
            "accepted": [indices.index(i) for i in row["accepted"]]}


def validate_online(cases):
    failed = set()
    for case in cases:
        rows = expand([case])
        reported = case["results"]
        if set(reported) != BASELINE_IDS:
            raise ValueError(f"Incomplete baseline arms: {case['id']}")
        for turns in reported.values():
            for row, turn in zip(rows, turns, strict=True):
                if "meal_menu" in turn:
                    menu, letter = turn["meal_menu"], turn["letter"]
                else:
                    menu, letter = turn["trace"]["menú"], turn["trace"]["letra"]
                names = option_names(row)
                order = dict(zip(LETTERS, names, strict=True))
                action = names.index(turn["choice"])
                metrics = expected(row, action)
                if (not turn["direct"] or menu != order or menu.get(letter) != turn["choice"]
                        or turn["success"] != bool(metrics["success"])):
                    raise ValueError(f"Live baseline mismatch: {row['id']}")
                if "conflict" in turn and turn["conflict"] != bool(metrics["conflict"]):
                    raise ValueError(f"Live baseline conflict mismatch: {row['id']}")
                if not metrics["success"]:
                    failed.add(case["id"])
    return failed


def archived_rows(split):
    rows = read_rows(ARCHIVE / f"{split}.jsonl")
    for row in rows:
        expected_id = digest({key: row[key] for key in ("turns", "options")})
        if row["id"] != expected_id or row["split"] != split:
            raise ValueError(f"Invalid archived identity: {row['id']}")
        if acceptance(row["options"], set(row["forbidden"]), row["ambiguous"]) != row["accepted"]:
            raise ValueError(f"Invalid archived labels: {row['id']}")
    return rows


def validate_splits(fixtures, archive):
    authored = [case["id"] for cases in fixtures.values() for case in cases]
    if len(set(authored)) != len(authored):
        raise ValueError("QA conversation ID reused across fixtures")
    normalized_sets = {name: {normalized([turn["text"]]) for case in cases for turn in case["turns"]}
                       for name, cases in fixtures.items()}
    for a, b in combinations(normalized_sets, 2):
        if normalized_sets[a] & normalized_sets[b]:
            raise ValueError(f"Shared text in QA fixtures {a} and {b}")
    archived_texts = {normalized([text]) for rows in archive.values() for row in rows for text in row["turns"]}
    for name in ("meal_qa_dev", "meal_qa_reserved"):
        if normalized_sets[name] & archived_texts:
            raise ValueError(f"Shared text between archived bank and {name}")
    archived_ids = [row["id"] for rows in archive.values() for row in rows]
    if len(archived_ids) != len(set(archived_ids)):
        raise ValueError("Archived case reused across partitions")
    # Older recommendation experiments precede the varied parent. Check all
    # available bank partitions as additional provenance, including old tests.
    for prior in ("tesis_meal_lora_v1", "tesis_meal_lora_v2"):
        for split in archive:
            path = ARCHIVE.with_name(prior) / f"{split}.jsonl"
            if path.is_file():
                prior_texts = {normalized([text]) for row in read_rows(path) for text in row["turns"]}
                if normalized_sets["meal_qa_reserved"] & prior_texts:
                    raise ValueError(f"Reserved text appeared in {prior}/{split}")


def validate_plan(run, plan, selection, results):
    if (selection["split"] != "validation" or selection["plan_sha256"] != file_hash(run / "PLAN.json")
            or results["selection_sha256"] != file_hash(run / "SELECTION.json")):
        raise ValueError("Invalid plan → selection → results hash chain")
    if set(plan["source_sha256"]) != SOURCES or set(plan["fixtures"]) != set(FIXTURES):
        raise ValueError("Incomplete declared inputs")
    if set(plan["archived_banks"]) != {"train", "validation", "evaluation"}:
        raise ValueError("Incomplete archived bank")
    files = {ROOT / name: sha for name, sha in plan["source_sha256"].items()}
    files.update({DATA / f"{name}.json": sha for name, sha in plan["fixtures"].items()})
    files.update({ARCHIVE / f"{name}.jsonl": sha for name, sha in plan["archived_banks"].items()})
    files[ARCHIVE / "sft/adapter_model.safetensors"] = plan["parent_adapter_sha256"]
    for path, checksum in files.items():
        if file_hash(path) != checksum:
            raise ValueError(f"Planned input changed: {path}")
    if plan["reward"] != REWARD or plan["epochs"] < 1 or min(plan["batch"], plan["passes"], plan["draws"]) < 1:
        raise ValueError("Invalid optimization or reward plan")


def verified_trace(path, roots, orders):
    ledger = read_rows(path)
    expected_coverage = [(row["id"], order) for row in roots for order in orders]
    if [(row["id"], row["order"]) for row in ledger] != expected_coverage:
        raise ValueError(f"Mismatched presentation coverage: {path.name}")
    roots_by_id = {row["id"]: row for row in roots}
    for entry in ledger:
        row = rotate(roots_by_id[entry["id"]], entry["order"])
        action = list(LETTERS).index(entry["letter"])
        names = option_names(row)
        menu = dict(zip(LETTERS, names, strict=True))
        accepted = [names[index] for index in row["accepted"]]
        raw_correct = int(entry["raw_token"] in [LETTERS[index] for index in row["accepted"]])
        if (entry["family"] != row["family"] or entry["turn"] != row.get("turn", 1)
                or entry["conversation"] != row.get("conversation", row["id"])
                or entry["menu"] != menu or entry["choice"] != menu[entry["letter"]]
                or entry["accepted"] != accepted or entry["forbidden"] != row["forbidden"]
                or entry["raw_action_success"] != raw_correct
                or any(entry[key] != value for key, value in expected(row, action).items())):
            raise ValueError(f"Incorrect choice scoring: {path.name} / {entry['id']} / {entry['order']}")
    return ledger


def panel(ledger):
    def summarize(rows):
        last = {}
        fully_correct = defaultdict(lambda: True)
        for row in rows:
            conversation = row["conversation"]
            turn, scores = last.get(conversation, (0, []))
            if row["turn"] > turn:
                last[conversation] = (row["turn"], [row["success"]])
            elif row["turn"] == turn:
                scores.append(row["success"])
            fully_correct[conversation] &= bool(row["success"])
        final = [score for _, scores in last.values() for score in scores]
        return {
            "cases": len(rows), "conversations": len(last),
            "correct": sum(row["success"] for row in rows),
            "conflicts": sum(row["conflict"] for row in rows),
            "unnecessary_clarifications": sum(row["unnecessary_clarification"] for row in rows),
            "missed_clarifications": sum(row["missed_clarification"] for row in rows),
            "fully_correct": sum(fully_correct.values()),
            "final_correct": sum(final), "final_presentations": len(final),
            "raw_correct": sum(row["raw_action_success"] for row in rows),
        }
    return {"live_order": summarize([row for row in ledger if row["order"] == 0]), "all_orders": summarize(ledger)}


def validation_rank(scores, base, epoch, tolerance):
    qa, replay = (scores[name]["all_orders"] for name in ("qa", "replay"))
    passes_gate = replay["correct"] / replay["cases"] >= base["correct"] / base["cases"] - tolerance
    return passes_gate, qa["correct"], -qa["conflicts"], -qa["missed_clarifications"], replay["correct"], -epoch


def schedules(rng, targeted, replay, epochs):
    starts = {row["id"]: rng.randrange(len(LETTERS)) for row in targeted + replay}
    for epoch in range(epochs):
        order = [[row["id"], (starts[row["id"]] + 2 * epoch + offset) % len(LETTERS)]
                 for row in targeted for offset in (0, 1)]
        order.extend([row["id"], (starts[row["id"]] + epoch) % len(LETTERS)] for row in replay)
        rng.shuffle(order)
        yield order


def validate_ledger(path, plan, order, training, method):
    ledger = read_rows(path)
    batches = [order[i:i + plan["batch"]] for i in range(0, len(order), plan["batch"])]
    if [row["cases"] for row in ledger] != batches:
        raise ValueError(f"Training order differs: {path.name}")
    varying = 0
    for entry in ledger:
        if len(entry["loss"]) != plan["passes"] or not all(map(isfinite, entry["loss"])):
            raise ValueError(f"Invalid optimizer passes: {path.name}")
        if method == "grpo":
            batch_varying = 0
            for (row_id, offset), draws, returns in zip(entry["cases"], entry["sampled"], entry["returns"], strict=True):
                source = rotate(training[row_id], offset)
                if len(draws) != plan["draws"] or any(type(action) is not int or action not in ORDERS for action in draws):
                    raise ValueError(f"Invalid action group: {path.name}")
                expected_returns = [reward(source, action) for action in draws]
                if any(not isclose(a, b, abs_tol=1e-7) for a, b in zip(returns, expected_returns, strict=True)):
                    raise ValueError(f"Incorrect GRPO return: {path.name}")
                batch_varying += len(set(expected_returns)) > 1
            if batch_varying != entry["groups_with_reward_variation"]:
                raise ValueError(f"Incorrect reward variation: {path.name}")
            varying += batch_varying
    return {"presentations": len(order), "updates": len(batches) * plan["passes"],
            "groups_with_reward_variation": varying if method == "grpo" else None}


def validate_checkpoint(path, chosen, plan):
    manifest_path = path / "local_checkpoint.json"
    manifest = read(manifest_path)
    parent = read(ARCHIVE / "sft/local_checkpoint.json")
    weights_sha = file_hash(path / "adapter_model.safetensors")
    expected_origin = (ARCHIVE.relative_to(ROOT) / "sft").parts
    origin = Path(manifest["origin_checkpoint"]).parts[-len(expected_origin):]
    if (weights_sha != chosen["adapter_sha256"] or file_hash(manifest_path) != chosen["manifest_sha256"]
            or artifact_hashes(path) != manifest["artifacts_sha256"]
            or origin != expected_origin
            or manifest["base_artifacts_sha256"] != parent["base_artifacts_sha256"]
            or manifest["parameters"]["training_scope"] != "lora_adapters_integrated_in_llm"
            or (chosen["epoch"] == 0 and weights_sha != plan["parent_adapter_sha256"])):
        raise ValueError(f"Selected checkpoint differs: {path.name}")


def comparison(ledgers, seed, orders):
    selected = {method: [entry for entry in ledgers[method]["reserved"] if entry["order"] in orders]
                for method in METHODS}
    rows = [{"method": method, "seed": seed, "root_id": entry["conversation"], "family": entry["family"],
             "realization_id": f"{entry['turn']}:{entry['order']}", "success": entry["success"]}
            for method, entries in selected.items() for entry in entries]
    report = paired_bootstrap(rows, "grpo", "sft", "success", confidence=.95, resamples=2000)
    report["paired_presentations"] = {"both_correct": 0, "only_grpo": 0, "only_sft": 0, "both_incorrect": 0}
    outcome = {(1, 1): "both_correct", (0, 1): "only_grpo",
               (1, 0): "only_sft", (0, 0): "both_incorrect"}
    for sft, grpo in zip(selected["sft"], selected["grpo"], strict=True):
        report["paired_presentations"][outcome[sft["success"], grpo["success"]]] += 1
    report["estimand"] = "family macroaverage of paired mean turn accuracy by conversation"
    return report


def audit(run):
    plan, selection, results = (read(run / name) for name in ("PLAN.json", "SELECTION.json", "RESULTS.json"))
    validate_plan(run, plan, selection, results)
    fixtures = {}
    for name in FIXTURES:
        item = read(DATA / f"{name}.json")
        if name in OBSERVED:
            metadata = item.get("scope", item)
            if metadata["verify"] is not False:
                raise ValueError(f"Baseline was verified online: {name}")
            for checkpoint in metadata.get("checkpoints", {}).values():
                if checkpoint["adapter_sha256"] != plan["parent_adapter_sha256"]:
                    raise ValueError(f"Unexpected parent of live baseline: {name}")
            fixtures[name] = item["cases"]
        else:
            fixtures[name] = item
    archive = {split: archived_rows(split) for split in plan["archived_banks"]}
    validate_splits(fixtures, archive)
    observed = [case for name in OBSERVED for case in fixtures[name]]
    failures = validate_online(observed)
    training_cases = [case for case in observed if case["id"] in failures]
    training_cases.extend(case for name in AUGMENTED for case in fixtures[name])
    targeted = expand(training_cases)
    if (sorted(failures) != plan["observed_failures"] or len(targeted) != plan["targeted_prefixes"]
            or len(training_cases) != plan["targeted_conversations"]):
        raise ValueError("Training data do not match the plan")

    rng = random.Random(plan["seed"])
    training_replay = rng.sample(archive["train"], len(targeted))
    validation_replay = rng.sample(archive["validation"], 60)
    if len(training_replay) != plan["replay_prefixes"]:
        raise ValueError("Training replay count differs")
    training = {row["id"]: row for row in targeted + training_replay}
    order_by_epoch = list(schedules(rng, targeted, training_replay, plan["epochs"]))
    development = {"qa": (expand(fixtures["meal_qa_dev"]), ORDERS), "replay": (validation_replay, (0, 3))}

    def validation(prefix):
        return {name: panel(verified_trace(run / f"{prefix}-validation-{name}.jsonl", rows, orders))
                for name, (rows, orders) in development.items()}

    baseline = validation("baseline")
    if set(selection["methods"]) != set(METHODS):
        raise ValueError("Missing training arm")
    telemetry = {}
    for method, chosen in selection["methods"].items():
        epochs = {0: baseline}
        training_counts = {}
        for epoch, order in enumerate(order_by_epoch, 1):
            training_counts[str(epoch)] = validate_ledger(run / f"{method}-{epoch}-train.jsonl", plan, order,
                                                          training, method)
            epochs[epoch] = validation(f"{method}-{epoch}")
        base = baseline["replay"]["all_orders"]
        best = max(epochs, key=lambda epoch: validation_rank(epochs[epoch], base, epoch, plan["replay_tolerance"]))
        if (chosen["epoch"] != best or chosen["training"] != training_counts
                or chosen["validation"] != {str(epoch): scores for epoch, scores in epochs.items()}):
            raise ValueError(f"Reported training or validation selection differs: {method}")
        validate_checkpoint(run / method, chosen, plan)
        telemetry[method] = training_counts

    panels = {"reserved": (expand(fixtures["meal_qa_reserved"]), ORDERS),
              "observed": (expand(observed), (0,)), "archived": (archive["evaluation"], ORDERS)}
    if set(results["panels"]) != {"baseline", *METHODS}:
        raise ValueError("Missing evaluation arm")
    ledgers = defaultdict(dict)
    audited_metrics, trace_hashes = {}, {}
    for method in ("baseline", *METHODS):
        audited_metrics[method] = {}
        for name, (roots, orders) in panels.items():
            filename = f"{method}-{name}.jsonl"
            entries = verified_trace(run / filename, roots, orders)
            ledgers[method][name] = entries
            audited_metrics[method][name] = panel(entries)
            trace_hashes[filename] = file_hash(run / filename)
    if audited_metrics != results["panels"] or trace_hashes != results["trace_sha256"]:
        raise ValueError("Official test summary or test trace hashes differ")

    return {
        "status": "verified", "plan_sha256": file_hash(run / "PLAN.json"),
        "results_sha256": file_hash(run / "RESULTS.json"), "selection_sha256": file_hash(run / "SELECTION.json"),
        "audit_source_sha256": file_hash(Path(__file__)), "panels": audited_metrics,
        "validation_epoch": {method: chosen["epoch"] for method, chosen in selection["methods"].items()},
        "training": telemetry,
        "paired_grpo_minus_sft": {
            "live_order": comparison(ledgers, plan["seed"], (0,)),
            "all_orders": comparison(ledgers, plan["seed"], ORDERS),
        },
        "units": {"cases": "turn/menu-order presentations", "conversations": "authored histories",
                  "fully_correct": "histories correct at every scored turn and order",
                  "final_presentations": "menu-order presentations at the last turn"},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = audit(args.run.resolve())
    write_json(args.run / "AUDIT.json", report)
    print(f"Verified {args.run} • validation epochs {report['validation_epoch']}", flush=True)
