"""Evaluate frozen memory adapters on new conversations with no verifier."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
import random
from tempfile import TemporaryDirectory

from research.src.artifacts import canonical, file_hash, write_json
from research.src.memory.executor import Executor
from research.src.memory.store import Store
from research.src.readers.symbolic import read as symbolic_read
from research.tesis_chat_memory_e2e import memory_check, normalize_text, validate_bank
from research.tesis_meal_exploration import load_weights
from research.tesis_meal_train import logits, write_rows
from research.tesis_temporal_memory_policy import (
    LETTERS, RECALL, mentions_preference, observation_for, option_tokens, prompt,
    question, recall_actions, write_actions,
)
from research.tesis_temporal_memory_train import METHODS, QA, RUN, SEEDS, load_frozen, read
from research.src.policy.dialogue import personal_wording
from web_demo.chat import MEMORY_QUERY, _render_answer


ROOT = Path(__file__).resolve().parents[1]


class LocalActor:
    def __init__(self, actor):
        self.actor = actor
        self.tokens = option_tokens(actor.tokenizer)

    def choose(self, phase, observation, options, key):
        # Different turns vary option order. The paired arms and seeds
        # share the same permutation whenever their visible state agrees.
        rng = random.Random(key)
        order = rng.sample(range(len(options)), len(options))
        menu = [options[index] for index in order]
        text = prompt(phase, observation, menu)
        ids = self.actor.encode_prompt(text)
        with self.actor.torch.inference_mode():
            output = logits(self.actor, [ids])[0]
            legal = output[self.tokens[:len(menu)]]
            choice = int(legal.argmax())
            scores = legal.tolist()
        return menu[choice], {"phase": phase, "letter": LETTERS[choice], "scores": scores,
                               "options": menu, "prompt": text}


def score_reply(expected, answer, slot, memory, query):
    if expected is None:
        return None
    kind = expected["kind"]
    mode = (answer or {}).get("mode")
    if kind == "ask":
        passed = slot in expected["slots"]
        return {"kind": kind, "passed": passed, "slot": slot}
    value = (answer or {}).get("value")
    wanted = {normalize_text(value) for value in expected.get("values", [])}
    refs = set((answer or {}).get("evidence_ids", []))
    # The common reader is used strictly in scoring. An independently
    # grounded answer must cite both reference gold and applicable evidence.
    projection = symbolic_read(memory, query)
    applicable = set(projection["evidence_ids"])
    allowed = set(expected.get("evidence_ids", []))
    value_correct = mode == "ANSWER" and normalize_text(value) in wanted
    grounded = bool(refs) and refs <= applicable and (not allowed or refs <= allowed)
    if kind == "recall":
        passed = value_correct and grounded
    elif kind == "abstain":
        passed = mode == "DEFER" and not refs
    elif kind == "uncertain":
        offered = (answer or {}).get("branches", [])
        branches = {normalize_text(branch["value"]) for branch in offered}
        supported = {normalize_text(record["value"]) for record in memory if record["memory_id"] in refs}
        matching_conditions = sorted(map(canonical, offered)) == sorted(map(canonical, projection["branches"]))
        passed = (mode == "DEFER" and not refs) or (
            mode == "CONDITIONALIZE" and branches == wanted == supported and grounded and matching_conditions)
        passed = passed or slot == "CURRENT_CONTEXT"
    else:
        raise ValueError(f"Unknown memory reply criterion: {kind}")
    return {"kind": kind, "passed": passed, "mode": mode, "value": value, "slot": slot,
            "value_correct": value_correct, "grounded": grounded, "evidence_ids": sorted(refs)}


def _specific_sources(added, expected):
    unmatched = list(added)
    for golden in expected:
        if "source_turn_ids" not in golden:
            continue
        found = next((i for i, record in enumerate(unmatched)
                      if all(record[key] == golden[key] for key in ("attribute", "value", "scope", "status"))
                      and set(record["source_turn_ids"]) == set(golden["source_turn_ids"])), None)
        if found is None:
            return False
        unmatched.pop(found)
    return True


def replay(actor, case, database, *, method, seed):
    session, attr, messages, pending = 1, "dinner_preference", [], None
    opening = None
    with closing(Store(database)) as store:
        store.initialize(case["initial_memory"])
        for index, step in enumerate(case["steps"], 1):
            date = step["date"]
            previous = store.list_records()
            expected = step["expect"]
            choice = answer = slot = trace = execution = None
            query = {"attribute": attr, "date": date, "session_id": f"S{session}", "context": None}
            if step["action"] == "new_session":
                session, messages, pending, opening = session + 1, [], None, None
            else:
                text = step["text"]
                user = {"turn_id": f"u{session}_{len(messages) + 1}", "role": "user", "text": text, "date": date}
                clean = personal_wording(text)
                recall = bool((RECALL.search(clean) or MEMORY_QUERY.search(clean)) and not mentions_preference(clean))
                if pending and not recall:
                    user.update(in_reply_to=pending, about_attribute=attr)
                messages.append(user)
                room = messages if recall else messages[opening:] if opening is not None and pending else [user]
                observation = observation_for(room, previous, date, f"S{session}", attr, recall=recall)
                query = observation["query"]
                attr = query["attribute"]
                write_options = write_actions(observation) if not recall else []
                # A declared preference (including reported speech or a
                # clarification reply) competes with preserve and ask.
                phase, options = ("write", write_options) if len(write_options) > 1 else ("recall", recall_actions(observation))
                choice, trace = actor.choose(phase, observation, options, f"{case['id']}/{index}")
                if "slot" in choice:
                    slot = choice["slot"]
                    if not pending:
                        opening = len(messages) - 1
                    pending = f"a{session}_{len(messages)}"
                    reply = question(slot)
                else:
                    pending, opening = None, None
                    if "operations" in choice:
                        execution = Executor(store).execute(choice["operations"], recorded_at=date)
                        if not execution["valid"]:
                            reply = "La escritura fue rechazada por el formato de SQLite."
                        elif any(op["op"] != "NOOP" for op in execution["executed_operations"]):
                            reply = "Guardé esa preferencia."
                        else:
                            # This is a second learned choice over the actual
                            # state. The app does not insert a correct answer.
                            answer, response_trace = actor.choose(
                                "recall", observation, recall_actions(observation), f"{case['id']}/{index}/response")
                            trace["respond"] = response_trace
                            reply = _render_answer(answer)
                    else:
                        answer = choice
                        reply = _render_answer(answer)
                messages.append({"turn_id": f"a{session}_{len(messages)}", "role": "assistant",
                                 "text": reply, "date": date})
            after = store.list_records()
            memory = memory_check(previous, {"messages": messages, "memory": after}, expected)
            supported = _specific_sources(memory["added"], expected["writes"])
            memory["expected_sources"] = supported
            memory["passed"] = memory["passed"] and supported and (execution is None or execution["valid"])
            grade = score_reply(expected.get("reply"), answer, slot, after, query)
            yield {"case": case["id"], "family": case["family"], "step": index, "session": session,
                   "date": date, "seed": seed, "method": method, "verify": False, "input": step,
                   "reply_score": grade, "memory_score": memory, "query": query,
                   "passed": memory["passed"] and (grade is None or grade["passed"]), "choice": choice,
                   "answer": answer, "model_trace": trace, "execution": execution,
                   "previous": previous, "memory": after}


def summarize(rows):
    by_profile = defaultdict(Counter)
    histories = defaultdict(list)
    for row in rows:
        key = row["seed"], row["method"]
        counter = by_profile[key]
        histories[(*key, row["case"])].append(row["passed"])
        counter["events"] += 1
        counter["events_correct"] += row["passed"]
        memory = row["memory_score"]
        counter["memory_events_correct"] += memory["passed"]
        counter["missing_writes"] += memory["missing_writes"]
        counter["unexpected_writes"] += memory["unexpected_writes"]
        counter["unsupported_citations"] += memory["unsupported_citations"]
        counter["probes"] += len(memory["probes"])
        counter["probes_correct"] += sum(probe["passed"] for probe in memory["probes"])
        if row["input"]["expect"]["writes"]:
            counter["write_expected"] += 1
            counter["write_correct"] += memory["passed"]
        if reply := row["reply_score"]:
            kind = reply["kind"]
            counter[f"{kind}_expected"] += 1
            counter[f"{kind}_correct"] += reply["passed"]
            if kind == "recall":
                counter["recall_value_correct"] += reply["value_correct"]
    for (*key, _), events in histories.items():
        by_profile[tuple(key)]["histories"] += 1
        by_profile[tuple(key)]["histories_correct"] += all(events)
    return by_profile, histories


def paired(histories):
    from statistics import mean

    seeds = sorted({key[0] for key in histories})
    cases = sorted({key[2] for key in histories})
    per_case = [mean(int(all(histories[seed, "grpo", case])) - int(all(histories[seed, "sft", case]))
                     for seed in seeds) for case in cases]
    rng = random.Random(15263)
    sims = sorted(mean(rng.choices(per_case, k=len(per_case))) for _ in range(10000))
    return {"delta": mean(per_case), "interval_95": [sims[250], sims[9749]],
            "unit": "history; paired across methods and seeds"}


def totals_by_method(profiles):
    summaries = {}
    for method in METHODS:
        summary = Counter()
        for seed in SEEDS:
            summary.update(profiles[seed, method])
        summaries[method] = summary
    return summaries


def compare_profiles(output, *, load_profile, metadata):
    plan = read(output / "PLAN.json")
    if file_hash(QA) != plan["heldout_sha256"]:
        raise ValueError("The held-out bank changed after the plan was frozen.")
    cases = validate_bank(read(QA))
    all_rows = []
    with TemporaryDirectory(prefix="tesis-temporal-eval-") as directory:
        for seed in SEEDS:
            for method in METHODS:
                folder = output / str(seed)
                choices = load_profile(seed, method)
                rows = [row for case in cases for row in replay(
                    choices, case, Path(directory) / f"{seed}-{method}-{case['id']}.sqlite", method=method, seed=seed)]
                write_rows(folder / f"{method}-heldout.jsonl", rows)
                all_rows.extend(rows)
                metrics, _ = summarize(rows)
                print(seed, method, dict(metrics[seed, method]), flush=True)
    profiles, histories = summarize(all_rows)
    stats = totals_by_method(profiles)
    families = {}
    for family in sorted({row["family"] for row in all_rows}):
        views, _ = summarize([row for row in all_rows if row["family"] == family])
        families[family] = totals_by_method(views)
    write_json(output / "RESULTS.json", {
        "plan_sha256": file_hash(output / "PLAN.json"), "freeze_sha256": file_hash(output / "FREEZE.json"),
        "bank_sha256": plan["heldout_sha256"], "evaluator_sha256": file_hash(Path(__file__)),
        **metadata, "verifier": False, "cases": len(cases), "seeds": SEEDS,
        "profiles": {str(seed): {method: profiles[seed, method] for method in METHODS} for seed in SEEDS},
        "totals": stats, "by_family": families, "paired_histories_grpo_minus_sft": paired(histories),
        "traces": {f"{seed}/{method}": file_hash(output / str(seed) / f"{method}-heldout.jsonl")
                   for seed in SEEDS for method in METHODS},
    })


def evaluate(output, device):
    actor, plan = load_frozen(output, device)
    for name, sha in plan["source_sha256"].items():
        if file_hash(Path(__file__).with_name(name)) != sha:
            raise ValueError(f"Training source changed: {name}")
    actor.model.requires_grad_(False)
    choices = LocalActor(actor)

    def load_profile(seed, method):
        load_weights(actor, output / str(seed) / method)
        return choices

    compare_profiles(output, load_profile=load_profile, metadata={"device": device, "actor": "Qwen LoRA"})


def evaluate_compact(output):
    from research.tesis_temporal_memory_compact import CompactActor

    import torch

    torch.set_num_threads(2)
    plan_path = output / "PLAN.json"
    plan = read(plan_path)
    lock = read(output / "FREEZE.json")
    if file_hash(plan_path) != lock["plan_sha256"]:
        raise ValueError("Frozen compact plan changed.")
    for name, expected in plan["source_sha256"].items():
        if file_hash(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"Frozen compact source changed: {name}")

    def load_profile(seed, method):
        folder = output / str(seed)
        receipt = folder / "SELECTION.json"
        if file_hash(receipt) != lock["selections"][str(seed)]:
            raise ValueError("Frozen compact selection changed.")
        checkpoint = folder / f"{method}.pt"
        if file_hash(checkpoint) != read(receipt)[method]["adapter_sha256"]:
            raise ValueError("Frozen compact weights changed.")
        return CompactActor.load(checkpoint)

    compare_profiles(output, load_profile=load_profile, metadata={"device": "cpu", "actor": "candidate scorer"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    if args.compact:
        evaluate_compact(args.output)
    else:
        evaluate(args.output, args.device)


if __name__ == "__main__":
    main()
