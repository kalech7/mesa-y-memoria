from collections import Counter, defaultdict
from datetime import date
import json
from math import fsum
from pathlib import Path

import pytest

from research import tesis_ecuador_worlds as bank
from research.tesis_timing_v6_bank import build_worlds as v6_worlds


PAIRS = Path(__file__).resolve().parents[2] / "reports/tesis_ecuador_30_v1/PARES.json"


def pairs():
    return json.loads(PAIRS.read_text(encoding="utf-8"))


def test_population_packets_probabilities_and_dates():
    authored = pairs()
    worlds = bank.build_worlds(authored)
    assert bank.build_worlds(authored[::-1]) == worlds
    assert len({w["world_id"] for w in worlds}) == len(worlds) == 480
    assert len({w["root_id"] for w in worlds}) == 60
    expected_day = {"EC01": 4, "EC02": 1, "EC03": 2, "EC04": 1, "EC05": 3,
                    "EC06": 5, "EC07": 1, "EC08": 5, "EC09": 5, "EC10": 4,
                    "EC11": 2, "EC13": 1, "EC14": 5, "EC15": 1}
    by_pair = {p["id"]: p for p in authored}
    packet_members = defaultdict(set)
    root_worlds = defaultdict(list)
    for w in worlds:
        packet_members[w["packet_id"]].add(w["pair_id"])
        root_worlds[w["root_id"]].append(w)
        pair = by_pair[w["pair_id"]]
        first, second, third = w["visits"]
        assert first["utterance"] == pair["opening"] + " " + pair[
            "agenda_first" if w["variant"] == "ask_now" else "agenda_second"]
        assert second["utterance"] == pair["second"]
        queries = w["future_queries"]
        assert len({(q["attribute"], q["date"]) for q in queries}) == len(queries) == 6
        count = Counter(q["attribute"] for q in queries)
        assert sorted(count.values()) == [1, 5]
        for q in queries:
            visit = first if q["attribute"] == first["attribute"] else second
            assert q["expected_value"] == (visit["value"] if visit["scope"] == "ongoing" else None)
            assert q["date"] > third["date"]
            if count[q["attribute"]] == 1 and w["pair_id"] in expected_day:
                assert date.fromisoformat(q["date"]).isoweekday() == expected_day[w["pair_id"]]
    assert len(packet_members) == 3
    for packet, members in bank.packet_pairs().items():
        assert packet_members[packet] == set(members) and len(members) == 10
        assert all(int(x[2:]) <= 15 for x in members[::2])
        assert all(int(x[2:]) == int(y[2:]) + 15 for y, x in zip(members[::2], members[1::2], strict=True))
    for alternatives in root_worlds.values():
        assert len(alternatives) == 8 and fsum(w["mass"] for w in alternatives) == 1
        scope_mass = Counter()
        matches_agenda = []
        for w in alternatives:
            first, second = w["visits"][:2]
            scope_mass[first["scope"], second["scope"]] += w["mass"]
            high = first if w["variant"] == "ask_now" else second
            if sum(q["attribute"] == high["attribute"] for q in w["future_queries"]) == 5:
                matches_agenda.append(w["mass"])
        assert len(scope_mass) == 4 and set(scope_mass.values()) == {.25}
        assert fsum(matches_agenda) == pytest.approx(.8)


def test_cpu_audit_checks_public_prompts_and_sqlite_routes():
    result = bank.audit(pairs())
    assert (result["pairs"], result["roots"], result["worlds"]) == (30, 60, 480)
    assert result["distinct_external_agendas"] == 60
    assert result["distinct_public_prompts_checked"] >= 120
    assert not any(result["historical_agenda_overlap"].values())
    assert result["private_metadata_in_prompts"] is False
    assert result["future_accuracy"]["ask_now"]["ASK-WAIT"] == pytest.approx(.85)
    assert result["future_accuracy"]["save"]["ASK-WAIT"] == pytest.approx(.65)
    assert result["future_accuracy"]["save"]["WAIT-ASK"] == pytest.approx(.85)
    assert result["future_accuracy"]["save"]["WAIT-WAIT"] == pytest.approx(.5)


def test_rejects_duplicate_ids_missing_food_and_historical_cue():
    rows = pairs()
    with pytest.raises(ValueError, match="exactly once"):
        bank.build_worlds([rows[1], *rows[1:]])
    with pytest.raises(ValueError, match="literal"):
        bank.build_worlds([{**rows[0], "first_food": "plato ausente"}, *rows[1:]])
    historical = next(w["cue"] for w in v6_worlds("reserved") if not bank._single_day(w["cue"]))
    rows[0].update(agenda_first=historical, agenda_second="Para el desayuno tengo cinco ocasiones y para la cena una.")
    with pytest.raises(ValueError, match="Repeated"):
        bank.audit(rows)
