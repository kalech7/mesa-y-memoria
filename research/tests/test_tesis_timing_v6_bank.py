from collections import Counter, defaultdict
from datetime import date
from itertools import combinations
from math import fsum
import re

import pytest

from research import tesis_timing_v6_bank as bank


@pytest.mark.parametrize("split", ["train", "validation", "reserved"])
def test_sealed_population_pairs_packet_and_disclosure_timing(split):
    meta = bank.split_metadata(split)
    worlds = bank.build_worlds(split)
    bank.verify_worlds(split, worlds)
    assert len({w["world_id"] for w in worlds}) == len(worlds) == meta["worlds"]
    roots = bank.build_bank(split)
    pairs, packets = defaultdict(list), defaultdict(list)
    for root in roots:
        packets[root["packet_id"]].append(root)
        pairs[root["pair_id"]].append(root)
        alternatives = root["worlds"]
        assert len(alternatives) == 8
        assert fsum(w["mass"] for w in alternatives) == 1
        for visit in (0, 1):
            public = {(v["utterance"], v["date"], v["question"], v["attribute"], v["value"])
                      for w in alternatives for v in [w["visits"][visit]]}
            assert len(public) == 1
            assert {w["visits"][visit]["scope"] for w in alternatives} == {"ongoing", "session"}
        assert len({w["visits"][2]["utterance"] for w in alternatives}) == 2
        assert len({w["visits"][0]["utterance"] for w in alternatives}) == 1
        assert alternatives[0]["visits"][0]["utterance"] == root["opening"] + " " + root["cue"]
    assert len(roots) == meta["roots"]
    assert len(pairs) == meta["pairs"]
    assert len(packets) == meta["packets"]
    assert all(len(packet) == 20 for packet in packets.values())
    for first, second in pairs.values():
        assert {first["variant"], second["variant"]} == {"ask_now", "save"}
        assert first["packet_id"] == second["packet_id"]
        assert first["opening"] == second["opening"]
        assert first["cue"] != second["cue"]
        for one, two in zip(first["worlds"], second["worlds"], strict=True):
            assert one["mass"] == two["mass"]
            assert one["visits"][1] == two["visits"][1]
            assert {k: v for k, v in one["visits"][0].items() if k != "utterance"} == {
                k: v for k, v in two["visits"][0].items() if k != "utterance"}


def test_public_hint_changes_expected_value_without_revealing_latent_scopes():
    for root in bank.build_bank("train"):
        scope_mass = Counter()
        predicted_mass, first_known, second_known = [], [], []
        for w in root["worlds"]:
            first, second, final = w["visits"]
            answers = {v["attribute"]: v["value"] if v["scope"] == "ongoing" else None for v in (first, second)}
            for visit in (first, second):
                marker = "habitual" if visit["scope"] == "ongoing" else "solo por hoy"
                assert marker in visit["reply"].lower()
            scope_mass[first["scope"], second["scope"]] += w["mass"]
            qs = w["future_queries"]
            assert len(qs) == 6
            assert len({(q["attribute"], q["date"]) for q in qs}) == 6
            assert all(q["date"] > final["date"] and q["expected_value"] == answers[q["attribute"]] for q in qs)
            counts = Counter(q["attribute"] for q in qs)
            high, low = counts.most_common()
            assert high[1] == 5 and low[1] == 1
            matches_hint = high[0] == (first if root["variant"] == "ask_now" else second)["attribute"]
            if matches_hint:
                predicted_mass.append(w["mass"])
            # Gain in positively supported future services if only one visit is clarified.
            first_known.append(w["mass"] * counts[first["attribute"]] * (first["scope"] == "ongoing") / len(qs))
            second_known.append(w["mass"] * counts[second["attribute"]] * (second["scope"] == "ongoing") / len(qs))
        assert fsum(predicted_mass) == pytest.approx(.8)
        assert set(scope_mass.values()) == {.25}
        difference = fsum(first_known) - fsum(second_known)
        assert difference == pytest.approx(.2 if root["variant"] == "ask_now" else -.2)


def test_train_loading_is_disjoint_and_does_not_generate_reserved(monkeypatch):
    partitions = {s: bank.build_worlds(s) for s in bank.PACKETS}
    for a, b in combinations(partitions.values(), 2):
        for field in ("packet_id", "pair_id", "root_id", "world_id", "opening", "cue"):
            assert {w[field] for w in a}.isdisjoint(w[field] for w in b)
        values = lambda rows: {v["value"] for w in rows for v in w["visits"][:2]}
        assert values(a).isdisjoint(values(b))

    def forbidden():
        pytest.fail("TRAIN loaded a different partition")

    monkeypatch.setitem(bank._WORDS, "validation", forbidden)
    monkeypatch.setitem(bank._WORDS, "reserved", forbidden)
    bank.verify_worlds("train", bank.build_worlds("train"))
    with pytest.raises(ValueError, match="sealed"):
        bank.verify_worlds("train", partitions["train"][:-1])
    with pytest.raises(ValueError, match="Unknown"):
        bank.build_bank("all")

@pytest.mark.parametrize("split", ["train", "validation", "reserved"])
def test_calendar_matches_actual_workweek_and_any_named_single_day(split):
    saw_named_day = False
    for root in bank.build_bank(split):
        for w in root["worlds"]:
            v1, v2, v3 = [date.fromisoformat(v["date"]) for v in w["visits"]]
            assert [d.isoweekday() for d in (v1, v2, v3)] == [4, 5, 7]
            counts = Counter(q["attribute"] for q in w["future_queries"])
            many, few = (a for a, count in counts.most_common())
            active_days = sorted(date.fromisoformat(q["date"]) for q in w["future_queries"] if q["attribute"] == many)
            assert [d.isoweekday() for d in active_days] == [1, 2, 3, 4, 5]
            assert (active_days[0] - v3).days == 1
            # A plan-changing world may expand or contract the originally
            # expected attribute. Named dates are checked for fulfilled plans.
            expected_many = w["visits"][0 if root["variant"] == "ask_now" else 1]["attribute"]
            if many != expected_many:
                continue
            named_day = re.search(r"(?:reservado el|solo el) (lunes|martes|miércoles|jueves|viernes)", root["cue"])
            if named_day:
                saw_named_day = True
                single_date = next(date.fromisoformat(q["date"]) for q in w["future_queries"] if q["attribute"] == few)
                assert bank.WEEKDAYS[single_date.weekday()] == named_day[1]
    if split != "train":
        assert saw_named_day
