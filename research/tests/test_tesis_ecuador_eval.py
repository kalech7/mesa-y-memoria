"""The native pair endpoint and the equal-budget endpoint have different units."""
import json

import pytest

from research import tesis_ecuador_eval as evaluate
from research.src.artifacts import file_hash
from research.tesis_timing_v6_eval import ASK, WAIT, Root


def synthetic_roots():
    roots = []
    for i in range(30):
        for variant in ("ask_now", "save"):
            early, late = (.85, .65) if variant == "ask_now" else (.65, .85)
            outcomes = {
                (ASK, WAIT): {"future_supported": early, "questions": 1},
                (WAIT, ASK): {"future_supported": late, "questions": 1},
                (WAIT, WAIT): {"future_supported": .5, "questions": 0},
            }
            roots.append(Root(f"{i:02d}-{variant}", f"{i:02d}", f"group-{i // 10}", variant, {}, {}, outcomes))
    return roots


def test_native_pair_requires_later_positive_margin_after_initial_wait():
    first, later = synthetic_roots()[:2]
    raw = {first.identity: (1, -1), later.identity: (0, 1)}
    assert evaluate.pair_scores([first, later], raw) == {"00": 1}
    raw[later.identity] = (0, 0)
    assert evaluate.pair_scores([first, later], raw) == {"00": 0}
    raw[later.identity] = (0, float("nan"))
    with pytest.raises(ValueError, match="Invalid margins"):
        evaluate.pair_scores([first, later], raw)


def test_pair_bootstrap_keeps_six_policies_together_and_quota_uses_frozen_cutoff(tmp_path, monkeypatch):
    roots = synthetic_roots()
    monkeypatch.setattr(evaluate, "RUN", tmp_path)
    monkeypatch.setattr(evaluate, "V7", tmp_path)
    seal = tmp_path / "SELLO.json"
    seal.write_text("{}")
    methods = {}
    for arm in evaluate.ARMS:
        methods[arm] = {}
        for seed in evaluate.SEEDS:
            margins = {}
            for root in roots:
                # In the first 15 pairs only the paired arm wins twice, and in
                # the last 15 only SFT wins twice. Seed 303 always ties.
                correct = seed == 303 or ((int(root.pair) < 15) == (arm == "sft-pares"))
                early = int(correct and root.variant == "ask_now")
                margins[root.identity] = [early, 1]
            methods[arm][str(seed)] = margins
    cuts = {str(seed): {arm: {"cutoff": "never"} for arm in evaluate.ARMS} for seed in evaluate.SEEDS}
    (tmp_path / "LOCK.json").write_text(json.dumps({"selected": cuts, "planner_cutoff": "never"}))
    raw = {"seal_sha256": file_hash(seal), "methods": methods, "planner": methods["sft"]["303"]}
    (tmp_path / "MARGENES.json").write_text(json.dumps(raw))
    report = evaluate.summarise(roots, raw)
    assert report["primary_difference"] == 0
    assert report["net_wins_per_pair"] == {"2": 15, "-2": 15}
    assert report["primary_pair_bootstrap_95"] == pytest.approx([-2 / 9, 2 / 9])
    for arm in evaluate.ARMS:
        assert [report["methods"][arm][str(s)]["correct_pairs"] for s in evaluate.SEEDS] == [15, 15, 30]
        for result in report["methods"][arm].values():
            for packet in result["packets"].values():
                assert packet["metrics"]["asked_first"] == 0
                assert packet["metrics"]["asked_second"] == .5
    assert report["matched_future_difference"] == 0
