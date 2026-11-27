"""Regression checks for the controlled action menu and calendar probes."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import unicodedata

import pytest

from research.src.memory.store import canonical
from research.tesis_temporal_memory_data import build_split
from research.tesis_temporal_memory_reward import score_action


ROOT = Path(__file__).resolve().parents[1]


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def turn_texts(rows):
    return {
        normalized(turn["text"])
        for row in rows
        for turn in row["observation"]["user_messages"]
        if turn["role"] == "user"
    }


@pytest.mark.parametrize(("split", "total"), [("train", 114), ("dev", 38)])
def test_every_temporal_group_has_variation_and_a_unique_supported_maximum(split, total):
    rows = build_split(split)
    assert len(rows) == len({row["id"] for row in rows}) == total
    for row in rows:
        assert len({canonical(action) for action in row["actions"]}) == len(row["actions"]) == 6
        maximum = score_action(row, row["actions"][row["correct_index"]])
        assert maximum > max(score_action(row, action) for i, action in enumerate(row["actions"])
                             if i != row["correct_index"])


def test_train_dev_and_previous_regression_banks_use_distinct_user_turns():
    train, dev = (turn_texts(build_split(name)) for name in ("train", "dev"))
    previous = set()
    for bank in ("meal_memory_e2e.json", "meal_memory_confirmatory.json"):
        cases = json.loads((ROOT / "data" / bank).read_text(encoding="utf-8"))
        previous.update(normalized(step["text"]) for case in cases for step in case["steps"] if "text" in step)
    assert train.isdisjoint(dev)
    assert previous.isdisjoint(train | dev)


def test_daily_exception_crosses_session_but_expires_preceding_the_next_calendar_day():
    rows = build_split("train")
    variants = {row["family"]: row for row in rows if row["id"].endswith("-0")}
    calendar, session = variants["day"], variants["session"]
    for row, persists in ((calendar, True), (session, False)):
        obs = row["observation"]
        turn = obs["query"]
        habitual = obs["memory"][0]["value"]
        temporary = row["actions"][0]["operations"][0]["record"]["value"]
        next_session = [p for p in row["probes"] if p["query"]["session_id"] != turn["session_id"]
                        and p["query"]["attribute"] == turn["attribute"]]
        assert [p["expected"]["value"] for p in next_session] == [
            temporary if persists else habitual, habitual, habitual, habitual,
        ]
    presence = Counter(row["phase"] for row in rows)
    assert presence == {"write": 60, "recall": 54}
