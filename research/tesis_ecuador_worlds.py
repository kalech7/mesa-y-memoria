"""External Spanish agendas on the frozen v6 simulator. No training or inference.

Each agenda predicts counts of two categories across six *future* services.
One service falls on each weekday, plus the single service of the other
category. When the text identifies that single day we use it. If the plan
changes, the categories swap. Unspecified dates use the v6 calendar draw.
"""
from collections import Counter, defaultdict
from datetime import date, timedelta
from math import isclose
import re
from statistics import mean

from research.src.artifacts import digest
from research.tesis_timing_v6_bank import PACKETS, WEEKDAYS, _pair, build_worlds as v6_worlds
from research.tesis_timing_v6_env import enumerate_world, prompt_text
from research.tesis_timing_v6_eval import group_worlds
from research.tesis_timing_v7_bank import WORDS, reserve_worlds


SPLIT = "ecuador_external_v1"
PRIVATE = re.compile(r"\b(?:EC\d{2}|EC-packet-\d+|(?:root|world|pair|packet)_id|"
                     r"author|autor [AB]|agenda_first|agenda_second|ask_now|save|"
                     r"ecuador_external_v1|future_queries|expected_value)\b", re.I)


def packet_pairs():
    """Three packets, each alternating five pairs from each author."""
    return {
        f"EC-packet-{packet + 1}": [
            f"EC{number + offset:02d}" for number in range(1 + 5 * packet, 6 + 5 * packet)
            for offset in (0, 15)
        ] for packet in range(3)
    }


def _single_day(cue):
    # Remove the recurring workweek before reading the date of the one-off.
    cue = re.sub(r"de lunes a viernes|lunes, martes, miércoles, jueves y viernes", "", cue.casefold())
    named = {i for i, day in enumerate(WEEKDAYS, 1) if re.search(rf"\b{day}\b", cue)}
    if re.search(r"\b(?:se inaugura|de la salida|la primera reunión)\b", cue):
        named.add(1)
    if re.search(r"\b(?:al cierre|última fecha)\b", cue):
        named.add(5)
    if len(named) > 1:
        raise ValueError(f"Conflicting dates for the single future service: {cue}")
    return next(iter(named), None)


def build_worlds(pairs):
    """Return the 480 v6-compatible worlds in fixed packet and pair order."""
    packets = packet_pairs()
    indexed = {pair["id"]: pair for pair in pairs}
    if len(pairs) != 30 or set(indexed) != {key for members in packets.values() for key in members}:
        raise ValueError("Expected EC01 through EC30 exactly once")
    worlds = []
    for packet, members in packets.items():
        for pair_id in members:
            pair = indexed[pair_id]
            second = pair["second_attribute"]
            if second not in ("breakfast_preference", "drink_preference"):
                raise ValueError(f"Invalid second category in {pair_id}")
            for meal, text in (("first_food", "opening"), ("second_food", "second")):
                food = pair[meal]
                if not food.strip() or food != food.lower() or food not in pair[text]:
                    raise ValueError(f"Missing literal lower-case food in {pair_id} {text}")
            cues = {"ask_now": pair["agenda_first"], "save": pair["agenda_second"]}
            if len(set(cues.values())) != 2 or any(not cue.strip() for cue in cues.values()):
                raise ValueError(f"Expected two different public agendas in {pair_id}")
            day = _single_day(cues["ask_now"])
            if day != _single_day(cues["save"]):
                raise ValueError(f"The two agendas disagree on the single service day in {pair_id}")
            recipe = pair["first_food"], (second, pair["second_food"])
            serial = int(pair_id[2:]) - 1
            for root in _pair("reserved", packet, serial, recipe, WORDS):
                for original in root["worlds"]:
                    first, later, last = original["visits"]
                    cue = cues[root["variant"]]
                    world = dict(original, split=SPLIT, pair_id=pair_id, opening=pair["opening"], cue=cue,
                                 visits=[dict(first, utterance=f"{pair['opening']} {cue}"),
                                         dict(later, utterance=pair["second"]), last])
                    for key in ("world_id", "root_id"):
                        world[key] = "ec-" + digest([SPLIT, key, original[key]])[:18]
                    if day is not None:
                        queries = world["future_queries"]
                        counts = Counter(q["attribute"] for q in queries)
                        single = next(q for q in queries if counts[q["attribute"]] == 1)
                        single["date"] = (date.fromisoformat(last["date"]) + timedelta(days=day)).isoformat()
                        queries.sort(key=lambda q: (q["date"], -counts[q["attribute"]]))
                    worlds.append(world)
    return worlds


def audit(pairs):
    """Check actual prompt strings, historical overlap, and SQLite branch outcomes on CPU."""
    worlds = build_worlds(pairs)
    normalize = lambda cue: " ".join(cue.casefold().split())
    cues = {normalize(w["cue"]) for w in worlds}
    history = {f"v6_{split}": {normalize(w["cue"]) for w in v6_worlds(split)} for split in PACKETS}
    history["v7_train"] = history["v6_train"]
    history["v7_validation"] = history["v6_validation"]
    history["v7_reserved"] = {normalize(w["cue"]) for w in reserve_worlds()}
    overlap = {name: sorted(cues & old) for name, old in history.items()}
    if len(cues) != 60 or any(overlap.values()):
        raise ValueError(f"Repeated external or historical agenda: {overlap}")
    # group_worlds also checks that all eight worlds expose identical live
    # observations before the first question and after waiting in visit one.
    trees = [enumerate_world(w, budget=1) for w in worlds]
    public_prompts = set()
    for tree in trees:
        for node in tree.decisions.values():
            prompt = prompt_text(node.observation)
            if PRIVATE.search(prompt) or any(tree.story[k] in prompt for k in ("world_id", "root_id")):
                raise ValueError("Private metadata leaked into a public prompt")
            public_prompts.add(prompt)
    roots = group_worlds(trees)
    routes = defaultdict(list)
    for root in roots:
        for path, metrics in root.outcomes.items():
            if not isclose(metrics["current_supported"], 1) or not isclose(metrics["future_unsupported_reuse"], 0):
                raise ValueError(f"Unsupported meal or memory in {root.pair}")
            routes[(root.variant, path)].append(metrics["future_supported"])
    expected = {("ask_now", ("ASK", "WAIT")): .85, ("save", ("WAIT", "ASK")): .85,
                ("ask_now", ("WAIT", "ASK")): .65, ("save", ("ASK", "WAIT")): .65,
                ("ask_now", ("WAIT", "WAIT")): .5, ("save", ("WAIT", "WAIT")): .5}
    for route, score in expected.items():
        if len(routes[route]) != 30 or not all(isclose(x, score) for x in routes[route]):
            raise ValueError(f"Unexpected SQLite branch score for {route}")
    return {"pairs": len(pairs), "roots": len(roots), "worlds": len(worlds), "packets": packet_pairs(),
            "worlds_sha256": digest(worlds), "distinct_external_agendas": len(cues),
            "historical_agenda_overlap": overlap, "distinct_public_prompts_checked": len(public_prompts),
            "private_metadata_in_prompts": False,
            "future_accuracy": {variant: {"-".join(path): mean(scores)
                                for (v, path), scores in routes.items() if v == variant}
                                for variant in ("ask_now", "save")}}
