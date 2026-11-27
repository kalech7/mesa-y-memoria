import pytest
import torch

from research.budget_memory_training import categorical_loss, leave_one_out


def test_leave_one_out_is_not_population_normalization():
    assert leave_one_out([1., 0., 0., 0.]) == [1., -1/3, -1/3, -1/3]
    assert leave_one_out([-2., -2., -2., -2.]) == [0., 0., 0., 0.]
    assert leave_one_out([-.1] * 4) == [0.] * 4
    assert leave_one_out([.38] * 4) == [0.] * 4
    with pytest.raises(ValueError):
        leave_one_out([1.])
    with pytest.raises(ValueError):
        leave_one_out([1., float('nan')])


def test_selected_good_action_gradient_and_exact_kl():
    logits = torch.zeros(3, requires_grad=True)
    lp = logits.log_softmax(-1)
    ref = torch.tensor([.2, .3, .5]).log()
    loss, policy, kl = categorical_loss(lp, ref, 1, 1., .03)
    expected = sum((1/3) * (torch.log(torch.tensor(1/3)) - x) for x in ref)
    assert kl.item() == pytest.approx(expected.item())
    assert loss.item() == pytest.approx((policy + .03 * kl).item())
    loss.backward()
    assert logits.grad[1] < 0
    assert logits.grad[0] > 0 and logits.grad[2] > 0


def test_no_task_gradient_in_flat_group_with_identical_reference():
    logits = torch.tensor([.1, -.2, .4], requires_grad=True)
    lp = logits.log_softmax(-1)
    loss, _, kl = categorical_loss(lp, lp.detach(), 0, 0., .03)
    loss.backward()
    assert kl.item() == pytest.approx(0.)
    assert torch.allclose(logits.grad, torch.zeros(3), atol=1e-8)
