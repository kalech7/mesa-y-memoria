from copy import deepcopy

import pytest

from research.tesis_timing_v6_bank import build_worlds
from research.tesis_timing_v6_demo import EXAMPLE, build_demo, checked_adapter, demo_pair
from research.tesis_timing_v6_experiment import ARMS, SEEDS, WEIGHTS, signature


class FakeScorer:
    def __init__(self):
        self.paths = []
        self.observations = []

    def load(self, path):
        self.paths.append(path)

    def margins(self, observations):
        for obs in observations:
            assert not {"world_id", "variant", "future_queries", "root_id", "scope"} & set(obs)
            assert obs["visit"] in (1, 2) and obs["questions_remaining"] == 1
        self.observations.extend(observations)
        # Select different native routes for identical food preferences.
        return [1. if "Para la cena estoy armando" in o["cue"] or o["visit"] == 2 else -1. for o in observations]


def make_lock(run):
    lock = {"selected": {}}
    for seed in SEEDS:
        lock["selected"][str(seed)] = {}
        for arm in ARMS:
            folder = run / f"seed-{seed}" / arm / "checkpoint-001/adapter"
            folder.mkdir(parents=True)
            for name in WEIGHTS:
                (folder / name).write_text(str(seed) + "/" + arm)
            lock["selected"][str(seed)][arm] = {"adapter": str(folder.relative_to(run)), "round": 1,
                                                 "sha256": signature(folder, WEIGHTS)}
    return lock


def test_native_sqlite_demo_on_locked_train_pair(tmp_path):
    worlds = build_worlds("train")
    lock = make_lock(tmp_path)
    scorer = FakeScorer()
    result = build_demo(worlds, tmp_path, lock, scorer)
    assert result["opening"] == EXAMPLE and result["split"] == "train"
    assert set(result["models"]) == set(map(str, SEEDS))
    assert len(scorer.paths) == len(ARMS) * len(SEEDS)
    for root_id, member in result["members"].items():
        modes = member["expected_over_all_worlds"]
        keys = ("ask_today", "save_for_tomorrow") if member["agenda"] == "ask_now" else ("save_for_tomorrow", "ask_today")
        assert modes[keys[0]]["future_supported"] == pytest.approx(.85)
        assert modes[keys[1]]["future_supported"] == pytest.approx(.65)
        assert modes["never_ask"]["future_supported"] == pytest.approx(.50)
        traces = member["fulfilled_world_with_two_habits"]["traces"]
        assert traces[keys[0]]["metrics"]["future_positive_supported"] == 5
        assert traces[keys[1]]["metrics"]["future_positive_supported"] == 1
        for trace in traces.values():
            assert trace["memory"][0]["evidence_quotes"]
        row = result["models"]["101"]["SFT-ramas"]["roots"][root_id]
        assert row["actions"] == (["ASK", "WAIT"] if member["agenda"] == "ask_now" else ["WAIT", "ASK"])
        assert (row["v2_source"] == "one_legal_action") == (row["actions"][0] == "ASK")
    chosen = lock["selected"]["101"][ARMS[0]]
    (tmp_path / chosen["adapter"] / WEIGHTS[0]).write_text("modified")
    with pytest.raises(ValueError, match="changed"):
        checked_adapter(tmp_path, 101, ARMS[0], chosen)


def test_deterministic_fallback_and_training_only():
    worlds = [w for w in build_worlds("train") if w["opening"] != EXAMPLE]
    chosen = demo_pair(worlds)
    assert chosen and {w["pair_id"] for w in chosen} == {worlds[0]["pair_id"]}
    invalid = deepcopy(worlds[:1])
    invalid[0]["split"] = "reserved"
    with pytest.raises(ValueError, match="TRAIN"):
        demo_pair(invalid)
