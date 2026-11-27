"""Exercise the public HTTP receipts for durable, expired and absent memories."""

from collections import Counter
import json
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen

import pytest


@pytest.fixture(scope="module")
def timing_url():
    from web_demo import timing_service

    run = Path(__file__).resolve().parents[1] / "reports" / "tesis_timing_qwen_a100_v6"
    with pytest.MonkeyPatch.context() as patch:
        # Forced routes use the frozen training scenario and actual SQLite
        # writes/readbacks, and never require Qwen's inference dependencies.
        patch.setattr(
            timing_service,
            "check_sources",
            lambda path: timing_service.read(path / "PLAN.json"),
        )
        lab = timing_service.TimingLab(run=run)
        with timing_service.make_server(lab, port=0) as server:
            thread = Thread(target=server.serve_forever, args=(0.05,), daemon=True)
            thread.start()
            try:
                yield f"http://127.0.0.1:{server.server_port}"
            finally:
                server.shutdown()
                thread.join()


def exercise_receipts(url, agenda, route, scope):
    options = {
        "agenda": agenda,
        "route": route,
        "scope": scope,
        "arm": "SFT-ramas",
        "seed": 101,
    }
    req = Request(
        url + "/api/timing/run",
        json.dumps(options).encode("utf-8"),
        {"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=15) as response:
        assert response.status == 200
        result = json.load(response)

    asked = {"ask_today": 1, "save_for_tomorrow": 2, "never_ask": None}[route]
    assert result["metrics"]["questions"] == int(asked is not None)
    assert [visit["visit"] for visit in result["visits"]] == [1, 2, 3]
    original = {
        message["turn_id"]: message["text"]
        for visit in result["visits"]
        for message in visit["messages"]
        if message["role"] == "user"
    }
    chosen_attribute = "dinner_preference" if asked == 1 else "breakfast_preference"
    chosen_value = "sopa" if asked == 1 else "pan con queso"
    record_id = None
    citations = []
    for visit in result["visits"]:
        number = visit["visit"]
        user_ids = [
            message["turn_id"]
            for message in visit["messages"]
            if message["role"] == "user"
        ]
        expected_ids = (
            [f"u{number}", f"u{number}-r"] if number == asked else [f"u{number}"]
        )
        assert user_ids == expected_ids
        assert [operation["op"] for operation in visit["operations"]] == (
            ["ADD"] if number == asked else []
        )
        assert visit["action"] == (
            "READ" if number == 3 else "ASK" if number == asked else "WAIT"
        )
        assert visit["remaining_before"] == int(asked is None or number <= asked)
        assert visit["remaining_after"] == int(asked is None or number < asked)
        memory = visit["memory"]
        if asked is None or number < asked:
            assert memory == []
            continue
        assert len(memory) == 1
        record = memory[0]
        assert (record["attribute"], record["value"]) == (
            chosen_attribute,
            chosen_value,
        )
        assert record["recorded_at"] == result["visits"][asked - 1]["date"]
        if number == asked:
            assert visit["operations"][0]["record"] == {
                k: v for k, v in record.items() if k != "valid"
            }
        assert record["scope"]["kind"] == scope
        assert record["valid"] is (scope == "ongoing" or number == asked)
        references = record["evidence_quotes"]
        assert {q["turn_id"] for q in references} == {f"u{asked}", f"u{asked}-r"}
        assert all(original[q["turn_id"]] == q["text"] for q in references)
        if record_id is not None:
            assert (record["memory_id"], references) == (record_id, citations)
        record_id, citations = record["memory_id"], references

    assert result["memory"] == result["visits"][-1]["memory"]
    assert len(result["future"]) == 6
    frequencies = Counter(q["attribute"] for q in result["future"])
    assert frequencies["dinner_preference"] == (5 if agenda == "ask_now" else 1)
    assert frequencies["breakfast_preference"] == (1 if agenda == "ask_now" else 5)
    for query in result["future"]:
        expect_answer = (
            asked is not None
            and scope == "ongoing"
            and query["attribute"] == chosen_attribute
        )
        assert bool(query["correct"]) == (expect_answer or scope == "session")
        if expect_answer:
            assert (query["mode"], query["value"]) == ("ANSWER", chosen_value)
            assert query["evidence_ids"] == [record_id]
            assert {q["turn_id"]: q["text"] for q in query["quotes"]} == {
                q["turn_id"]: q["text"] for q in citations
            }
        else:
            assert query["mode"] == "DEFER" and query["value"] is None
            assert query["quotes"] == [] and query["evidence_ids"] == []
    return result


@pytest.mark.parametrize("scope", ["session", "ongoing"])
@pytest.mark.parametrize("route", ["ask_today", "save_for_tomorrow", "never_ask"])
@pytest.mark.parametrize("agenda", ["ask_now", "save"])
def test_visit_memory_and_future_citations(timing_url, agenda, route, scope):
    exercise_receipts(timing_url, agenda, route, scope)
