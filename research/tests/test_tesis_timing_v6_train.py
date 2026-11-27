"""The longitudinal target must depend on the model's future actions, at TRAIN only."""
from collections import defaultdict
from types import SimpleNamespace

import pytest

from research.tesis_timing_v6_bank import build_worlds
from research.tesis_timing_v6_env import enumerate_world, prompt_text
from research.tesis_timing_v6_train import collect, log_probs, train_groups


@pytest.fixture(scope="module")
def groups():
    # The eight worlds of the same root differ only in latent outcomes.
    worlds = build_worlds("train")[:8]
    roots = {story["root_id"] for story in worlds}
    assert len(roots) == 1
    return train_groups({b: [enumerate_world(story, b) for story in worlds] for b in (1, 2)}, prompt_text)


def test_refreshing_the_future_policy_changes_first_visit_target(groups):
    def run(margin):
        return collect(groups, {g.public: margin for g in groups}, prompt_text, seed=202)

    always_wait, wait_log = run(-9000.)
    always_ask, ask_log = run(9000.)
    assert wait_log["sampled_continuation_actions"] == {"WAIT": wait_log["qwen_continuation_decisions"]}
    assert ask_log["sampled_continuation_actions"] == {"ASK": ask_log["qwen_continuation_decisions"]}
    assert ask_log["qwen_continuation_decisions"] > 0
    for a, b in zip(always_wait, always_ask, strict=True):
        assert a["public"] == b["public"]
        assert a["immediate_returns"] == pytest.approx([1., .92])
        assert a["immediate_returns"] == b["immediate_returns"]
        if a["visit"] == 1 and a["budget"] == 1:
            # The WAIT branch can change. The ASK branch spends the only
            # question now and fixes the next action in both experiments.
            assert a["returns"][0] != pytest.approx(b["returns"][0])
            assert a["returns"][1] == pytest.approx(b["returns"][1])
        if a["visit"] == 2:
            assert a["returns"] == b["returns"]


def test_shared_groups_use_conditional_world_mass_and_balance_visits(groups):
    weights = defaultdict(float)
    for group in groups:
        weights[group.root, group.budget, group.visit] += group.weight
        assert "ask_now" not in group.public and "variant" not in group.public
    assert len(weights) == 4 and all(v == pytest.approx(1.) for v in weights.values())
    scores = {g.public: 0. for g in groups}
    traces = []
    rows, summary = collect(groups, scores, prompt_text, seed=17, samples=8, write_trace=traces.append)
    repeated, summary2 = collect(groups, scores, prompt_text, seed=17, samples=8)
    assert rows == repeated and summary == summary2
    assert summary["replayed_decision_transitions"] > summary["terminal_rollouts"]
    assert sum(sum(p["samples"] for p in trace["paths"]) for trace in traces) == summary["terminal_rollouts"]
    for trace in traces:
        for path in trace["paths"]:
            assert path["actions"][trace["visit"]-1] == trace["forced_action"]
            assert path["actions"].count("ASK") <= trace["budget"]
    # At zero margin, stratification at eight samples represents the 50/50
    # second-visit behavior exactly, even for two counterfactual histories.
    assert summary["sampled_continuation_actions"]["ASK"] == summary["sampled_continuation_actions"]["WAIT"]


def test_reject_heldout_without_collecting():
    with pytest.raises(ValueError, match="only access TRAIN"):
        train_groups({1: [SimpleNamespace(story={"split": "validation"})]}, str)


def test_tempered_behavior_matches_rollout_probabilities():
    pytest.importorskip("torch")
    from research.tesis_timing_v6_train import ask_probability

    margins = [-16., 0., 12.]
    actual = log_probs(margins, "cpu").exp()[:, 1].tolist()
    assert actual == pytest.approx([ask_probability(m) for m in margins])


def test_longitudinal_control_retains_initial_returns_across_rounds(tmp_path, monkeypatch, groups):
    torch = pytest.importorskip("torch")
    from research import tesis_timing_v6_train as study

    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "adapter_model.safetensors").write_bytes(b"common")
    monkeypatch.setattr(study, "PARENT", parent)
    scorer = SimpleNamespace(model=torch.nn.Linear(1, 1), load=lambda path: None)

    def mock_scores(model, texts, path):
        return {text: -1000. if path == parent else 1000. for text in texts}, {}

    def mock_train(model, optimizer, rows, order, arm, anchor, old, batch_size):
        return {"input_tokens": 42, "padded_input_tokens": 44, "optimizer_updates": 2, "decisions": len(rows),
                "forward_batches": 2, "trained_rewards": study.digest([
                    r["immediate_returns" if arm == study.IMMEDIATE else "returns"] for r in rows])}

    def mock_checkpoint(directory, stage, model, optimizer, ledger):
        location = directory / f"checkpoint-{stage:03d}"
        (location / "adapter").mkdir(parents=True)
        (location / "adapter" / "adapter_model.safetensors").write_text(str(stage))
        (location / "adapter" / "adapter_config.json").write_text("{}")
        torch.save(optimizer.state_dict(), location / "OPTIMIZER.pt")
        study.atomic_json(location / "LEDGER.json", ledger)

    monkeypatch.setattr(study, "score_map", mock_scores)
    monkeypatch.setattr(study, "fit_arm", mock_train)
    monkeypatch.setattr(study, "checkpoint", mock_checkpoint)
    texts = sorted({g.public for g in groups})
    args = (scorer, groups, texts, {}, prompt_text, tmp_path, 101)
    first = study.fit_round(*args, 1, 8, False)
    second = study.fit_round(*args, 2, 8, False)
    assert study.fit_round(*args, 2, 8, True) == second
    first_target = first["arms"][study.RL]["trained_rewards"]
    assert first_target == first["arms"][study.FIXED]["trained_rewards"]
    assert second["arms"][study.RL]["trained_rewards"] != first_target
    assert second["arms"][study.FIXED]["trained_rewards"] == first_target
    assert second["arms"][study.SFT]["trained_rewards"] == second["arms"][study.RL]["trained_rewards"]
    assert second["arms"][study.IMMEDIATE]["trained_rewards"] == first["arms"][study.IMMEDIATE]["trained_rewards"]
    assert second["arms"][study.FIXED]["collector_sha256"] == first["collector"]["sha256"]
    assert second["arms"][study.RL]["collector_sha256"] == second["collector"]["sha256"]
