from collections import Counter
import json

import pytest

from research.tesis_shared_budget_bank import audit_development, build_development, group_roots
from research.tesis_shared_budget_env import SharedBudgetEpisode
from research.tesis_shared_budget_reserve import freeze, verify


def test_development_cues_and_probability_mass_are_root_local():
    splits = build_development(roots_per_regime=1)
    audit = audit_development(splits)
    expected = dict.fromkeys(("ready_regular", "ready_contingent", "busy_regular", "busy_contingent"), 1)
    for split, rows in splits.items():
        assert audit[split] == {"roots": 4, "worlds": 48, "roots_per_cue": expected}
        for worlds in group_roots(rows).values():
            assert Counter(world["visits"][0]["fact"]["scope"]["kind"] for world in worlds) == {
                "ongoing": 6, "session": 6,
            }
            assert len({world["visits"][0]["expected"] for world in worlds}) == 1
            assert len({world["visits"][1]["slot"] for world in worlds}) == 2


def test_audit_rejects_a_public_cue_that_reveals_the_world():
    splits = build_development(roots_per_regime=1)
    split = splits["validation"]
    split[1]["visits"][0]["messages"][0]["text"] += " Esta vez sí podré contestar."
    with pytest.raises(ValueError, match="Prefijo"):
        audit_development(splits)


def test_provisional_can_avoid_a_redundant_session_write_in_both_partitions():
    for rows in build_development(roots_per_regime=2).values():
        story = next(row for row in rows if row["second_visit_kind"] == "temporary")
        traces = {}
        for action in ("PROCEED", "PROVISIONAL"):
            episode = SharedBudgetEpisode(story)
            try:
                episode.step("PROCEED")
                episode.step(action)
                traces[action] = episode.summary()
            finally:
                episode.close()
        proceed, provisional = (traces[action]["visits"][1] for action in ("PROCEED", "PROVISIONAL"))
        assert proceed["outcome"] == provisional["outcome"]
        assert proceed["current_supported"] == provisional["current_supported"] == 1
        assert proceed["executed_operations"][0]["record"]["scope"]["kind"] == "session"
        assert provisional["memory_after"] == provisional["memory_before"]
        assert traces["PROCEED"]["future"] == traces["PROVISIONAL"]["future"]


def test_reserved_review_packets_are_blind_pending_and_frozen(tmp_path):
    folder = tmp_path / "reserve"
    manifest = freeze(folder)
    assert verify(folder) == manifest
    assert not manifest["evaluation_ready"] and not manifest["human_authored"]
    assert manifest["human_reviews_received"] == 0
    orders = []
    for reviewer in ("a", "b"):
        file = folder / f"revisor_{reviewer}/01_casos.json"
        cases = json.loads(file.read_text())["cases"]
        assert all(set(case) == {"audit_id", "memory", "visits"} for case in cases)
        orders.append([case["audit_id"] for case in cases])
    assert set(orders[0]) == set(orders[1]) and orders[0] != orders[1]
    file.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="congelado"):
        verify(folder)
