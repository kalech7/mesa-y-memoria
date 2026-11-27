import pytest
import torch

from research.decision_memory_training import enumerated_loss


def test_expected_action_gradient_matches_direct_return_gradient():
    logits = torch.tensor([.2, -.3, .5], requires_grad=True)
    q = torch.tensor([-.1, .9, -.2])
    logprobs = logits.log_softmax(-1)
    loss, _, kl, _, _ = enumerated_loss(logprobs, logprobs.detach(), q,
                                       selection_weight=.5, beta=.02)
    derivative = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    direct = torch.autograd.grad(-.5 * (logits.softmax(-1) * q).sum(), logits)[0]
    assert torch.allclose(derivative, direct, atol=1e-7)
    assert kl.item() == 0.


def test_constant_observed_returns_have_exact_zero_task_gradient():
    logits = torch.tensor([.2, -.3, .5], requires_grad=True)
    lp = logits.log_softmax(-1)
    loss, _, _, _, advantages = enumerated_loss(lp, lp.detach(), torch.tensor([-.14]*3))
    loss.backward()
    assert torch.equal(advantages, torch.zeros(3))
    assert torch.equal(logits.grad, torch.zeros(3))


def test_nonfinite_feedback_and_invalid_weight_rejected():
    lp = torch.tensor([.5, .25, .25]).log()
    with pytest.raises(ValueError):
        enumerated_loss(lp, lp, torch.tensor([1., float('nan'), 0.]))
    with pytest.raises(ValueError):
        enumerated_loss(lp, lp, torch.tensor([1., 0., 0.]), selection_weight=2.)
