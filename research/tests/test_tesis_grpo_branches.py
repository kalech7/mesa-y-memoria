import pytest
import torch

from research import tesis_grpo_branches as comparison
from research import tesis_interactive_train as base


def test_sibling_actions_include_proceed_and_topic_specific_followup():
    story = next(row for row in base.build_bank()["train"]
                 if row["scenario_family"] == "F4" and row["visits"][0]["style"] == "partial")
    cache = comparison.BranchRollouts()
    _, reference, _ = cache.collect(story, base.RULE)
    scope = base.ACTIONS.index("TEMPORAL_SCOPE")
    assert reference[:2] == [scope, scope]
    state, prefix, actions, group = cache.slate(story, 1)
    assert prefix == (scope,)
    assert actions == [index for index, legal in enumerate(state.mask.tolist()) if legal]
    assert group["returns"]["longitudinal"][actions.index(scope)]["total"] > group["returns"]["longitudinal"][0]["total"]
    with pytest.raises(ValueError, match="TRAIN"):
        cache.complete({**story, "split": "evaluation"}, ())


def test_local_update_credits_the_high_return_action():
    torch.manual_seed(3)
    policy = base.Selector()
    parent = base.clone(policy)
    state = base.State(torch.ones(len(base.FEATURES)), torch.tensor([True, True, False, False, False]), 0)
    group = {"returns": {"longitudinal": [{"total": 1.}, {"total": 1.67}]}}
    opt = torch.optim.AdamW(policy.parameters(), lr=.02)
    before = float(parent.logprobs([state])[0, 1].detach())
    update = comparison.branch_update(policy, opt, parent, (state, (), [0, 1], group), filtered=False)
    assert update["advantages"][0] < 0 < update["advantages"][1]
    assert update["advantages"][1] - update["advantages"][0] == pytest.approx(.67)
    assert float(policy.logprobs([state])[0, 1].detach()) > before
