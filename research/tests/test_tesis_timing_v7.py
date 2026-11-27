from copy import deepcopy

import pytest

from research.src.artifacts import digest
from research.tesis_timing_v6_bank import _WORDS, build_worlds
from research.tesis_timing_v6_eval import group_worlds
from research.tesis_timing_v6_experiment import forest
from research.tesis_timing_v7 import RESERVE_SHA, loss_for, training_pairs
from research.tesis_timing_v7_bank import WORDS, reserve_worlds


def test_independent_reserve_seal_partitions_and_budget_units():
    fresh = reserve_worlds()
    assert digest(fresh) == RESERVE_SHA
    assert len({row["world_id"] for row in fresh}) == len(fresh) == 1280
    packets = {}
    for row in fresh:
        packets.setdefault(row["packet_id"], set()).add(row["root_id"])
    assert len(packets) == 8
    assert all(len(group) == 20 for group in packets.values())
    for split, words in _WORDS.items():
        for field in WORDS:
            assert set(WORDS[field]).isdisjoint(words()[field])
        assert {row["root_id"] for row in build_worlds(split)}.isdisjoint(
            row["root_id"] for row in fresh)


def test_targets_follow_observed_consequences_even_when_private_flag_reversed():
    first_pair = reserve_worlds()[:16]
    roots = group_worlds(forest(first_pair, 1))
    expected = training_pairs(roots)
    assert len(expected) == 1 and len(set(expected[0])) == 4
    scrambled = deepcopy(roots)
    for root in scrambled:
        root.variant = "unused"
    assert training_pairs(scrambled) == expected
    scrambled[0].outcomes = scrambled[1].outcomes
    with pytest.raises(ValueError, match="opposite"):
        training_pairs(scrambled)


def test_extra_pairwise_gradient_only_pushes_opposite_first_visits_apart():
    torch = pytest.importorskip("torch")
    margins = torch.tensor([0.0] * 4, requires_grad=True)
    point = torch.autograd.grad(loss_for(margins, "sft"), margins)[0]
    ranked = torch.autograd.grad(loss_for(margins, "sft-pares"), margins)[0]
    assert ranked[0] < point[0] < 0
    assert ranked[1] > point[1] > 0
    assert torch.equal(point[2:], ranked[2:])
