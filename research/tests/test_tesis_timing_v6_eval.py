from math import isclose

import pytest

from research.tesis_timing_v6_eval import (
    allocate_packet,
    group_worlds,
    evaluate_pool,
    independent_timing,
    paired_difference,
)
from research.tesis_timing_v6_env import enumerate_world
from research.tesis_timing_v6_bank import build_worlds


def test_tomorrow_only_scores_people_who_waited_and_exactly_ten_questions():
    ids = list(map(str, range(20)))
    first = {i: (1 if int(i) < 4 else -1) for i in ids}
    calls = []

    def score_later(remaining):
        calls.append(set(remaining))
        return {i: int(i) for i in remaining}

    paths = allocate_packet(ids, first, score_later, 0.0)
    assert calls == [set(ids[4:])]
    assert [i for i, p in paths.items() if p[0] == "ASK"] == ids[:4]
    assert [i for i, p in paths.items() if p[1] == "ASK"] == ids[14:]
    assert sum(path.count("ASK") for path in paths.values()) == 10
    allocate_packet(ids, {i: 2 for i in ids}, lambda _: pytest.fail("unauthorized future lookup"), 0.0)
    assert allocate_packet(list(reversed(ids)), first, score_later, 0.0) == paths


@pytest.fixture(scope="module")
def packet():
    worlds = build_worlds("train")
    first = worlds[0]["packet_id"]
    return group_worlds([enumerate_world(w, 1) for w in worlds if w["packet_id"] == first])


def test_native_cue_switch_is_separate_from_forced_spending(packet):
    ideal = {r.identity: (1 if r.variant == "ask_now" else -1, 1.0) for r in packet}
    refuses_tomorrow = {i: (first, -1.0) for i, (first, _) in ideal.items()}
    native = independent_timing(packet, refuses_tomorrow)
    assert native["both_timings_correct_native"] == 0
    assert native["both_timings_correct_at_one_question"] == 1
    assert independent_timing(packet, ideal)["both_timings_correct_native"] == 1
    pool = evaluate_pool(packet, ideal, 0.0)
    assert isclose(pool["macro"]["future_supported"], 0.675) and pool["macro"]["questions"] == 0.5
    assert pool["macro"]["asked_first"] == 0.5 and pool["macro"]["asked_second"] == 0
    assert pool["macro"]["future_correct_abstentions"] == pytest.approx(3)


def test_paired_bootstrap_resamples_the_shared_unit(packet):
    left = evaluate_pool(packet, {r.identity: (2.0, 1.0) for r in packet}, 0)
    result = paired_difference({"101": left, "202": left}, {"101": left, "202": left}, draws=250)
    assert result["interval"] == [0, 0] and result["packets"] == 1 and result["per_seed"]["101"] == 0
