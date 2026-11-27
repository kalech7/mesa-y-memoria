"""Regression checks for the finite-action learning objective."""

import pytest

torch = pytest.importorskip("torch")

from research.tesis_meal_exploration import EPSILON, distributions, group_advantage, group_objective, score_action
from research.tesis_meal_qa import dialogue_rows
from research.tesis_meal_varied_bank import rotate


def test_exact_surrogate_is_the_expected_sampled_surrogate_even_when_clipped():
    tokens = list(range(6))
    _, old = distributions(torch.tensor([[5., -1., -1., 1., 0., -3.]]), tokens)
    _, current = distributions(torch.tensor([[-3., 0., 1., -1., -1., 5.]]), tokens)
    reward = torch.tensor([[-2.5, -2.5, -2.5, -2.5, -2.5, 1.5]])
    advantages = group_advantage(old, reward)
    total = group_objective(current, old, advantages)
    expectation = sum(old[0, index].exp() * group_objective(
        current, old, advantages, torch.tensor([[index]])) for index in tokens)
    torch.testing.assert_close(total, expectation)


def test_exact_group_reinforces_a_rare_clarification_on_exhausted_menu():
    raw = torch.tensor([[2., -1., -1., -2., -3., -5.]], requires_grad=True)
    _, policy = distributions(raw, list(range(6)))
    old = policy.detach()
    reward = torch.tensor([[-2.5, -2.5, -2.5, -2.5, -2.5, 1.5]])
    advantages = group_advantage(old, reward)
    assert torch.all(old.exp() >= EPSILON / 6)
    ground = (old.exp() * advantages).sum(-1)
    torch.testing.assert_close(ground, torch.zeros_like(ground), atol=1e-6, rtol=0.)
    group_objective(policy, old, advantages).backward()
    assert raw.grad[0, -1] < 0.


def test_reward_prefers_asking_to_a_conflict_but_prefers_a_valid_dish_to_asking():
    case = {"id": "ejemplo", "family": "inventario", "turns": [
        {"text": "Hoy sin papas ni aceite. ¿Qué ceno?", "excluded": ["papa", "aceite"]},
        {"text": "Ya traje papas. Sigue sin aceite. ¿Qué ceno?", "excluded": ["aceite"]},
    ]}
    first, second = dialogue_rows([case], "train")
    for offset in range(6):
        exhausted, feasible = rotate(first, offset), rotate(second, offset)
        for row in (exhausted, feasible):
            assert all((score_action(row, action) > 0) == (action in row["accepted"]) for action in range(6))
        ask = next(i for i, option in enumerate(feasible["options"]) if not option["ingredients"])
        fish = next(i for i, option in enumerate(feasible["options"]) if "pescado" in option["ingredients"])
        assert score_action(feasible, fish) < score_action(feasible, ask) < score_action(feasible, feasible["accepted"][0])
