from math import fsum, isfinite, log

import pytest

from research import tesis_rl_v8_rewards as rewards
from research.tesis_timing_v6_env import replay


@pytest.fixture(scope="module")
def bank():
    return rewards.TrainRewards()


def test_all_train_pairs_have_both_visits_and_expected_bellman_sign(bank):
    assert len(bank.records) == 120
    assert len({r.root for r in bank.records}) == 60
    assert len({r.pair for r in bank.records}) == 30
    for i in range(0, len(bank.records), 4):
        members = bank.records[i:i + 4]
        assert len({r.pair for r in members}) == 1
        assert [(r.variant, r.visit) for r in members] == [
            ("ask_now", 1), ("ask_now", 2), ("save", 1), ("save", 2),
        ]
    audit = bank.audit(bank.exact_returns(continuation="optimal"))
    assert audit["pairs_aligned_v1"] == 30
    assert audit["pairs_with_a_tie_v1"] == 0


def test_policy_and_public_optimum_match_weighted_sqlite_paths(bank):
    first, second = bank.records[:2]
    worlds = [w for w in rewards.build_worlds("train") if w["root_id"] == first.root]
    assert len(worlds) == 8
    def outcomes(a, b):
        return [replay(w, (a, b), budget=1)["metrics"]["longitudinal"] for w in worlds]
    def expected(values):
        return fsum(w["mass"] * v for w, v in zip(worlds, values, strict=True))
    ask_now = expected(outcomes("ASK", "WAIT"))
    wait_twice, ask_later = outcomes("WAIT", "WAIT"), outcomes("WAIT", "ASK")
    wait, ask = expected(wait_twice), expected(ask_later)
    scores = dict.fromkeys(bank.prompts, 0.)
    scores[second.public] = 4 * log(3)  # 75% ASK at the second public decision
    values = bank.exact_returns(scores)
    assert values[0] == pytest.approx((.25 * wait + .75 * ask, ask_now))
    assert values[1] == pytest.approx((wait, ask))
    optimal_wait = bank.exact_returns(continuation="optimal")[0][0]
    assert optimal_wait == pytest.approx(max(wait, ask))
    assert optimal_wait < expected(map(max, wait_twice, ask_later))


def test_policy_requires_only_finite_public_continuation_scores(bank):
    later = {r.public: 0. for r in bank.records if r.visit == 2}
    assert len(bank.exact_returns(later)) == len(bank.records)
    public = next(iter(later))
    del later[public]
    with pytest.raises(ValueError, match="Missing"):
        bank.exact_returns(later)
    later[public] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        bank.exact_returns(later)


def test_relative_loss_gradient_and_regularizers_on_cpu():
    torch = pytest.importorskip("torch")
    from research.tesis_rl_v8 import ENTROPY, relative_loss
    def check(pair):
        margin = torch.zeros(1, requires_grad=True)
        loss, stats = relative_loss(margin, torch.tensor([pair]), margin.detach(), margin.detach())
        loss.backward()
        assert stats["kl"] == 0.
        assert stats["clipped_fraction"] == 0.
        assert stats["entropy"] == pytest.approx(log(2))
        assert isfinite(loss.item()) and isfinite(margin.grad.item())
        return loss.item(), margin.grad.item()
    _, favor_wait = check((1., 0.))
    _, favor_ask = check((0., 1.))
    assert favor_wait > 0 > favor_ask  # descent lowers or raises the ASK margin
    assert check((5., 15.))[1] == pytest.approx(favor_ask)
    tied_loss, tied_gradient = check((2., 2.))
    assert tied_loss == pytest.approx(-ENTROPY * log(2))
    assert tied_gradient == 0.
