"""Run held-out longitudinal conversations through the normal local chat handlers."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from time import monotonic
from unicodedata import normalize

from research.src.memory.executor import validate_record
from research.src.readers.symbolic import read
from web_demo.chat import Chats
from web_demo.live import LiveInference
from web_demo.meal_models import meal_models
from web_demo.meal_policy import recommendation_menu


ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "research/data/meal_memory_e2e.json"
SEEDS = (101, 202, 303)
METHODS = ("sft", "grpo_sampled")
FINISHED = "summary"


@dataclass
class SharedDecoding:
    backend: object
    outputs: dict
    counts: Counter

    @property
    def config(self):
        return self.backend.config

    def encode_prompt(self, prompt):
        return self.backend.encode_prompt(prompt)

    def generate(self, prompt, sample=False):
        if sample:
            return self.backend.generate(prompt, sample=True)
        key = (self.backend.local.model.active_adapter, self.config.max_new_tokens, prompt)
        if key not in self.outputs:
            self.outputs[key] = self.backend.generate(prompt)
            self.counts["generated"] += 1
        else:
            self.counts["reused"] += 1
        return self.outputs[key]


class PairedInference(LiveInference):
    """Reuse exact greedy structured calls from the shared memory checkpoint."""

    def __init__(self, device):
        super().__init__(device)
        self.outputs = {}
        self.counts = Counter()

    @contextmanager
    def chat_controller(self, model):
        with super().chat_controller(model) as controller:
            controller.backend = SharedDecoding(controller.backend, self.outputs, self.counts)
            yield controller


@dataclass
class LocalDate:
    value: date | None = None

    def __call__(self):
        return self.value


def normalize_text(value):
    if value is None:
        return ""
    return " ".join(re.findall(r"\w+", normalize("NFKC", value).casefold()))


def effect(record):
    return {**{key: record[key] for key in ("attribute", "value", "scope", "status")},
            "supersedes": sorted(record.get("supersedes", []))}


def valid_citations(records, response):
    user_text = {message["turn_id"]: message["text"] for message in response["messages"]
                 if message["role"] == "user"}
    return all(quote["text"] == user_text.get(quote["turn_id"])
               for record in records for quote in record["evidence_quotes"])


def memory_check(before, response, expected):
    previous = {record["memory_id"]: record for record in before}
    memory = {record["memory_id"]: record for record in response["memory"]}
    preserved = all(previous[key] == memory.get(key) for key in previous)
    new = [record for key, record in memory.items() if key not in previous]
    uncertainty = expected.get("allow_unresolved")
    optional = [] if uncertainty is None else [
        record for record in new if record["scope"]["kind"] == "unknown" and record["status"] == "unresolved"
        and not record.get("supersedes") and all(record[key] == value for key, value in uncertainty.items())
    ]
    registered = [record for record in new if record not in optional]
    expected_effects = expected["writes"]
    took_effect = len(optional) <= 1 and profile_counter(registered) == profile_counter(expected_effects)
    unsupported = [record for record in new if not valid_citations([record], response)]
    unexpected = profile_counter(registered) - profile_counter(expected_effects)
    missing = profile_counter(expected_effects) - profile_counter(registered)
    supported = not unsupported
    probes = []
    for item in expected.get("probes", []):
        observed = read(response["memory"], item["query"])
        wanted = item["expected"]
        values = {branch["value"] for branch in observed["branches"]}
        passed = observed["mode"] == wanted["mode"] and observed["value"] == wanted["value"] and values == set(wanted["branches"])
        probes.append({"query": item["query"], "expected": wanted, "observed": observed, "passed": passed})
    return {"passed": preserved and took_effect and supported and all(p["passed"] for p in probes),
            "preserved_previous_rows": preserved, "expected_writes_applied": took_effect,
            "public_citations": supported, "added": new, "probes": probes,
            "unsupported_citations": len(unsupported), "unexpected_writes": unexpected.total() + max(0, len(optional) - 1),
            "missing_writes": missing.total()}


def profile_counter(records):
    return Counter(json.dumps(effect(record), ensure_ascii=False, sort_keys=True) for record in records)


def first_action(trace, phase):
    for step in trace:
        if step["phase"] == phase:
            return step.get("action") or {}
    return {}


def reply_check(item, response, verify):
    if item is None:
        return None
    message = response["messages"][-1]
    trace = message["trace"]
    visible = message["text"]
    kind = item["kind"]
    result = message.get("memory_read") or first_action(trace, "respond")
    source = result.get("source") or ("model" if result else None)
    outcomes = {"kind": kind, "observed_mode": result.get("mode"), "source": source,
                "answer": result.get("value"),
                "slot": message.get("selected_slot") or first_action(trace, "select").get("slot")}
    if kind == "meal":
        candidates = recommendation_menu((), response["attribute"], shuffle=False)
        menu = {entry["name"]: entry["ingredients"] for entry in candidates}
        recommendation = message.get("recommendation") or {}
        chosen = recommendation.get("chosen") or recommendation.get("model_choice")
        observed = normalize_text(visible)
        if chosen not in menu or normalize_text(chosen) not in observed:
            named = [dish for dish, ingredients in menu.items() if ingredients and normalize_text(dish) in observed]
            chosen = named[0] if len(named) == 1 else None
        incompatible = sorted(set(item["excluded"]).intersection(menu.get(chosen, ())))
        is_question = chosen is None and visible.rstrip().endswith("?")
        is_in_menu = chosen in menu and bool(menu[chosen])
        raw = recommendation.get("raw") is True and recommendation.get("source") == "model"
        requested = item.get("preferred")
        passed = ((is_question if item["clarify"] else is_in_menu and not incompatible)
                  and (verify or raw))
        if requested:
            cited = recommendation.get("memory_used", [])
            stored = {record["memory_id"]: record["value"] for record in response["memory"]}
            passed = passed and normalize_text(chosen) == normalize_text(requested) and bool(cited) and all(
                stored.get(item["memory_id"]) == item["value"]
                and normalize_text(item["value"]) == normalize_text(requested) for item in cited
            )
        outcomes.update(passed=passed, chosen=chosen, raw=raw, clash=incompatible,
                        unnecessary_clarification=is_question and not item["clarify"],
                        missed_clarification=not is_question and item["clarify"],
                        recommendation_source=recommendation.get("source"),
                        memory_used=recommendation.get("memory_used", []),
                        memory_offered=recommendation.get("memory_offered", []))
        return outcomes
    if kind == "ask":
        passed = outcomes["slot"] in item["slots"] and visible.rstrip().endswith("?")
    elif kind == "recall":
        values = {normalize_text(value) for value in item["values"]}
        passed = (result.get("mode") == "ANSWER" and normalize_text(result.get("value")) in values
                  and normalize_text(result.get("value")) in normalize_text(visible))
    elif kind == "abstain":
        passed = result.get("mode") == "DEFER"
    elif kind == "uncertain":
        actual = {normalize_text(branch["value"]) for branch in result.get("branches", [])}
        expected = {normalize_text(value) for value in item["values"]}
        passed = result.get("mode") == "DEFER" or (result.get("mode") == "CONDITIONALIZE" and actual == expected)
        passed = passed or (response["pending"] and visible.rstrip().endswith("?"))
    else:
        raise ValueError(f"Unknown reply expectation {kind}")
    return dict(outcomes, passed=passed and not message.get("fallback"))


def validate_bank(cases):
    identities = set()
    for case in cases:
        if case["id"] in identities:
            raise ValueError(f"Repeated case {case['id']}")
        identities.add(case["id"])
        prior_date = date.min
        for record in case["initial_memory"]:
            validate_record(record, initial=True)
        for step in case["steps"]:
            current = date.fromisoformat(step["date"])
            if current < prior_date:
                raise ValueError(f"Non-monotone history {case['id']}")
            if step["action"] not in {"send", "new_session"}:
                raise ValueError(f"Unrecognized event {step['action']}")
            prior_date = current
    return cases


def dump(handle, data):
    handle.write(json.dumps(data, ensure_ascii=False) + "\n")
    handle.flush()


def replay(chats, clock, profile, case, verify, handle):
    clock.value = date.fromisoformat(case["steps"][0]["date"])
    response = chats.start(profile, initial_memory=case["initial_memory"])
    if profile != response["model"]["id"] or case["initial_memory"] != response["memory"]:
        raise ValueError("Chat did not honor the starting condition")
    for number, step in enumerate(case["steps"], 1):
        clock.value = date.fromisoformat(step["date"])
        before = response["memory"]
        action = step["action"]
        request = {"action": action, "id": response["id"], "revision": response["revision"],
                   "request_id": f"step-{number}"}
        if action == "send":
            request.update(text=step["text"], verify=verify)
        start = monotonic()
        response = chats.handle(request)
        if (response["date"] != step["date"] or response["revision"] != request["revision"] + 1
                or response["id"] != request["id"]):
            raise ValueError(f"Chat response did not honor the dated event {case['id']} step {number}")
        memory = memory_check(before, response, step["expect"])
        reply = reply_check(step["expect"].get("reply"), response, verify)
        assistant = response["messages"][-1] if action == "send" else None
        item = {"type": "step", "model": profile, "verify": verify, "case": case["id"],
                "family": case["family"], "step": number, "input": step, "session": response["session"],
                "memory_score": memory, "reply_score": reply,
                "passed": memory["passed"] and (reply is None or reply["passed"]),
                "memory_before": before, "memory_after": response["memory"], "assistant": assistant,
                "seconds": round(monotonic() - start, 3)}
        dump(handle, item)
        yield item


def summarize(rows):
    totals = defaultdict(Counter)
    by_family = defaultdict(Counter)
    dialogs = defaultdict(list)
    paired = defaultdict(dict)
    for row in rows:
        profile = row["model"], row["verify"]
        method, seed = row["model"].split("-exploracion-")[1].rsplit("-", 1)
        paired[(seed, row["verify"], row["case"], row["step"])][method] = row["passed"]
        area = totals[profile]
        family = by_family[(*profile, row["family"])]
        dialogs[(*profile, row["case"])].append(row["passed"])
        for counter in (area, family):
            counter["events"] += 1
            counter["event_passes"] += row["passed"]
            counter["memory_passes"] += row["memory_score"]["passed"]
            for metric in ("unsupported_citations", "unexpected_writes", "missing_writes"):
                counter[metric] += row["memory_score"][metric]
            reply = row["reply_score"]
            if reply is not None:
                counter["replies"] += 1
                counter["reply_passes"] += reply["passed"]
                counter[f"{reply['kind']}_total"] += 1
                counter[f"{reply['kind']}_passed"] += reply["passed"]
                if reply["kind"] == "meal":
                    counter["meal_conflicts"] += bool(reply["clash"])
                    counter["unnecessary_meal_clarifications"] += reply["unnecessary_clarification"]
                    counter["missed_meal_clarifications"] += reply["missed_clarification"]
            assistant = row["assistant"]
            if assistant:
                counter["generation_errors"] += sum("error" in item for item in assistant["trace"])
                counter["fallbacks"] += bool(assistant.get("fallback"))
                counter["verified_write_repairs"] += (assistant.get("verification") or {}).get("status") == "repaired"
    for (*profile, _), outcomes in dialogs.items():
        totals[tuple(profile)]["dialogues"] += 1
        totals[tuple(profile)]["dialogue_passes"] += all(outcomes)
    outcomes_by_profile = defaultdict(Counter)
    for (seed, verify, _, _), decisions in paired.items():
        if set(decisions) == set(METHODS):
            sft, grpo = (decisions[key] for key in METHODS)
            label = ("neither", "grpo_only", "sft_only", "both")[2 * sft + grpo]
            outcomes_by_profile[(seed, verify)][label] += 1
    return {"type": FINISHED, "paired_event_outcomes": [
        {"seed": int(seed), "verify": verify, **outcomes} for (seed, verify), outcomes in outcomes_by_profile.items()],
        "models": [
        {"model": model, "verify": verify, **details, "families": {
            family: value for (k, v, family), value in by_family.items() if model == k and verify == v}}
        for (model, verify), details in totals.items()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=BANK)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append")
    parser.add_argument("--method", choices=METHODS, action="append")
    parser.add_argument("--verify", choices=("on", "off", "both"), default="both")
    parser.add_argument("--case", action="append", help="Only run an explicitly named scenario")
    parser.add_argument("--validate", action="store_true", help="Check the fixture without loading models")
    args = parser.parse_args()
    raw = args.bank.read_bytes()
    cases = validate_bank(json.loads(raw))
    if args.validate:
        print(json.dumps({"cases": len(cases), "events": sum(len(case["steps"]) for case in cases),
                          "bank_sha256": sha256(raw).hexdigest()}, ensure_ascii=False))
        return
    if args.output is None:
        parser.error("--output is required for live evaluation")
    if args.case:
        available = {case["id"] for case in cases}
        if set(args.case) - available:
            parser.error("Unknown --case")
        cases = [case for case in cases if case["id"] in args.case]
    modes = (False, True) if args.verify == "both" else (args.verify == "on",)
    profiles = [f"comidas-variado-exploracion-{method}-{seed}"
                for seed in args.seed or SEEDS for method in args.method or METHODS]
    available = meal_models()[0]
    if set(profiles) - available.keys():
        parser.error("Requested frozen profiles are unavailable")
    clock = LocalDate()
    inference = PairedInference(args.device)
    chats = Chats(inference, clock=clock)
    sources = ("web_demo/chat.py", "web_demo/chat_history.py", "web_demo/live.py", "web_demo/recommendations.py",
               "web_demo/meal_policy.py", "web_demo/chat_inference.py", "research/src/policy/dialogue.py",
               "research/src/policy/rule_controller.py", "research/src/readers/symbolic.py",
               "research/tesis_memory_guard.py")
    rows = []
    with args.output.open(mode="x", encoding="utf-8") as handle:
        dump(handle, {"type": "metadata", "device": args.device, "bank_sha256": sha256(raw).hexdigest(),
                      "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                      "source_sha256": {path: sha256((ROOT / path).read_bytes()).hexdigest() for path in sources},
                      "start_clock": "America/Guayaquil simulated calendar",
                      "profiles": [available[p].public() for p in profiles], "verify": modes,
                      "cases": [case["id"] for case in cases]})
        for profile in profiles:
            for verify in modes:
                for case in cases:
                    print(f"{profile} verify={verify} {case['id']}", file=sys.stderr, flush=True)
                    rows.extend(replay(chats, clock, profile, case, verify, handle))
        summary = summarize(rows)
        summary["structured_decoding"] = dict(inference.counts)
        dump(handle, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
