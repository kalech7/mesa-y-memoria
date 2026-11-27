from collections import defaultdict
from itertools import combinations

import pytest

from research.tesis_meal_cases import CAPTURE
from research.tesis_meal_varied_bank import SIZE, build_varied, normalized, rotate


@pytest.fixture(scope="module")
def bank():
    return build_varied()


def test_dialogues_and_recipes_split_before_augmentation(bank):
    assert {split: len(rows) for split, rows in bank.items()} == SIZE
    assert any(CAPTURE in row["turns"] for row in bank["train"])
    for first, second in combinations(bank.values(), 2):
        assert not ({normalized(row["turns"]) for row in first} & {normalized(row["turns"]) for row in second})
        assert not ({item["name"] for row in first for item in row["options"] if item["ingredients"]} &
                    {item["name"] for row in second for item in row["options"] if item["ingredients"]})


def test_rotations_preserve_valid_options_and_balance_each_position(bank):
    for case in bank["evaluation"]:
        valid = {case["options"][i]["name"] for i in case["accepted"]}
        for offset in range(6):
            row = rotate(case, offset)
            assert {row["options"][i]["name"] for i in row["accepted"]} == valid
        for index in range(6):
            assert {rotate(case, offset)["order"][index] for offset in range(6)} == set(range(6))


def test_held_out_counterfactuals_change_only_one_exclusion_and_require_a_flip(bank):
    pairs = defaultdict(list)
    for row in bank["evaluation"]:
        if row["pair"]:
            pairs[row["pair"]].append(row)
    assert len(pairs) == 15
    for left, right in pairs.values():
        assert left["options"] == right["options"]
        assert len(set(left["forbidden"]) ^ set(right["forbidden"])) == 2
        assert len(left["accepted"]) == len(right["accepted"]) == 1
        assert left["accepted"] != right["accepted"]


def test_noncyclic_orders_preserve_the_same_allowed_dishes(bank):
    from research.tesis_meal_shuffle_check import variants

    roots = bank["evaluation"][:10]
    by_id = {case["id"]: case for case in roots}
    for case, _, order in variants(roots):
        original = by_id[case["id"]]
        assert order not in [rotate(original, offset)["order"] for offset in range(6)]
        assert {case["options"][i]["name"] for i in case["accepted"]} == {
            original["options"][i]["name"] for i in original["accepted"]}
