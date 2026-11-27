from collections import Counter
import numpy as np
import pytest
import torch

from research import tesis_interactive_train as trainer


def test_cache_matches_live_rollout_and_restricts_training():
    bank = trainer.build_bank(sizes={"train": 5, "validation": 5, "evaluation": 5})
    episode = next(row for row in bank["train"] if row["scenario_family"] == "F4")
    cache = trainer.TrainingRollouts()
    for policy in (trainer.RULE, trainer.NEVER, trainer.RULE):
        states, actions, trace = cache.collect(episode, policy)
        reference, expected_actions, expected_trace = trainer.rollout(episode, policy)
        assert actions == expected_actions
        assert trace == expected_trace
        assert all(torch.equal(left.features, right.features) for left, right in zip(states, reference))
    with pytest.raises(ValueError, match="TRAIN"):
        cache.collect(bank["evaluation"][0], trainer.RULE)


def test_balanced_schedule_and_checkpoint_guards():
    stories = [{"scenario_family": f"F{i}", "root_id": str(i), "world_id": j} for i in range(5) for j in range(2)]
    order = trainer.order_train(stories, 10, np.random.default_rng(15))
    assert Counter(story["scenario_family"] for story in order) == {f"F{i}": 2 for i in range(5)}
    baseline = {"macro": {"current_supported": .70, "unsupported_writes": 0., "longitudinal_return": 1.,
                          "strict_joint": .2, "future_supported": .6}}
    assert trainer.passes_guards(baseline, baseline)
    assert not trainer.passes_guards({"macro": {**baseline["macro"], "current_supported": .67}}, baseline)
    assert not trainer.passes_guards({"macro": {**baseline["macro"], "unsupported_writes": .01}}, baseline)
    assert trainer.rank(baseline, 0) > trainer.rank(baseline, 25)


def test_grouped_optimization_updates_trajectory_actions_toward_reward():
    torch.manual_seed(101)
    actor = trainer.Selector()
    reference = trainer.clone(actor)
    features = torch.ones(len(trainer.FEATURES))
    mask = torch.tensor([index < 2 for index in range(len(trainer.ACTIONS))])
    state = trainer.State(features, mask, 0)
    action = 1
    before = float(actor.logprobs([state])[0][action].detach())
    optimizer = torch.optim.AdamW(actor.parameters(), lr=.02, weight_decay=0.)

    def trace(future, questions):
        return {"root_id": "episode", "world_id": "world", "split": "train",
                "visits": [{"current_supported": 1, "questions": questions}],
                "future": [{"supported": future, "correct": future}]}

    group = trainer.grouped_update(actor, optimizer, reference,
                                   [([state], [0], trace(0, 0)), ([state], [action], trace(1, 1))],
                                   horizon="longitudinal", passes=2)
    assert group["horizon_changes_preference"]
    assert group["future_varies_with_current_fixed"]
    assert group["advantages"][0] < 0 < group["advantages"][1]
    assert float(actor.logprobs([state])[0][action].detach()) > before
    assert float(reference.logprobs([state])[0][action].detach()) == before


def test_exact_audit_excludes_heldout():
    with pytest.raises(ValueError, match="TRAIN"):
        trainer.exact_root([{"split": "evaluation"}], trainer.Selector())


def test_paired_worlds_share_features_actions_and_probabilities_before_reply():
    bank = trainer.build_bank(sizes={"train": 5, "validation": 5, "evaluation": 5})
    worlds = [row for row in bank["train"] if row["scenario_family"] == "F4"]
    assert len(worlds) == 2 and len({row["root_id"] for row in worlds}) == 1
    states = []
    for world in worlds:
        env = trainer.InteractiveEpisode(world, ":memory:")
        try:
            states.append(trainer.state_of(env, env.observe()))
        finally:
            env.close()
    assert torch.equal(states[0].features, states[1].features)
    assert torch.equal(states[0].mask, states[1].mask)
    logprobs = trainer.Selector().logprobs(states)
    assert torch.equal(logprobs[0], logprobs[1])
