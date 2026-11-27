from collections import Counter

import pytest

from research.grpo_select_pilot import categorical_loss, choose, policy_return, schedule
from research.src.training.episodic_grpo import group_advantages


def test_schedule_keeps_both_ambiguous_worlds_and_clear_controls():
    def row(root, stratum, world="ongoing", split="train"):
        return {"root_id": root, "split": split, "authoring": {"stratum": stratum},
                "truth": {"scope_kind": world}}

    ambiguous = "ambiguous_duration"
    rows = [row(root, ambiguous, world)
            for root in ("a", "b") for world in ("ongoing", "session")]
    rows.extend(row(str(i), stratum) for i, stratum in
                enumerate(("contextual", "stable", "interval", "permanent")))
    scheduled = schedule(rows, rounds=2)
    assert Counter(episode["root_id"] for episode in scheduled) == {"a": 2, "b": 2, **{str(i): 1 for i in range(4)}}
    assert scheduled == schedule(list(reversed(rows)), rounds=2)
    with pytest.raises(ValueError, match="TRAIN"):
        schedule([*rows, row("held", "stable", split="validation")], rounds=1)


def test_task_and_question_cost_are_reported_separately():
    metrics = dict(strict_joint=1., C2=1., FJS=1., causal_write_delta=1.,
                   questions=1, unsupported_write=0, invalid_output=0)
    assert policy_return(metrics) == (3.5, 3.4)
    metrics["questions"] = 0
    assert policy_return(metrics) == (3.5, 3.5)


def test_selection_respects_safety_then_longitudinal_then_friction():
    def panel(joint, future=.5, uninformative=.5, invalid=.1, unsupported=.1):
        return {"metrics": dict(strict_joint=joint, FJS=future, uninformative_questions=uninformative,
                                invalid_output=invalid, unsupported_write=unsupported)}

    initial = panel(.5)
    assert choose(initial, {0: initial, 4: panel(.75, unsupported=.11), 8: panel(.75, invalid=.2)}) == 0
    assert choose(initial, {0: initial, 4: panel(.5, future=.7), 8: panel(.5, future=.7, uninformative=.3)}) == 8
    assert choose(initial, {0: initial, 4: panel(.5)}) == 0


def test_categorical_grpo_gradient_uses_fixed_old_and_reference():
    torch = pytest.importorskip("torch")
    logits = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    old = logits.detach().log_softmax(0)
    actions = torch.tensor([0, 0, 1, 2])
    advantages = old.new_tensor(group_advantages([1., 1., -1., -1.]))
    loss, kl, ratios = categorical_loss(torch, logits.log_softmax(0), old, old, actions, advantages)
    loss.backward()
    assert kl.item() == pytest.approx(0.)
    assert ratios.tolist() == pytest.approx([1.] * 4)
    assert logits.grad[0] < 0 < logits.grad[1]
    assert logits.grad[3].item() == pytest.approx(0.)
    candidate = torch.tensor([2., -2., -2., 0., 0.], dtype=old.dtype).log_softmax(0)
    _, kl, ratios = categorical_loss(torch, candidate, old, old, actions, advantages)
    assert kl.item() > 0.
    assert ratios.max().item() > 1.2 and ratios.min().item() < .8
