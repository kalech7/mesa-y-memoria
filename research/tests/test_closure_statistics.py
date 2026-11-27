import pytest

from research.src.closure_statistics import CONFIDENCE, aggregate_condition, paired_contrast


def row(method, value, *, world="w1", realization="r1", rid="root1", family="F0", seed=101):
    return {"method": method, "root_id": rid, "latent_world_id": world,
            "realization_id": realization, "scenario_family": family, "training_seed": seed,
            "condition_id": "common", "J_primary": value}


def test_worlds_have_equal_weight_despite_unequal_realizations():
    rows = [row("a", 1, realization=str(i)) for i in range(4)] + [row("a", 0, world="w2")]
    assert aggregate_condition(rows, {"method": "a"}, "J_primary")[(101, "root1")][1] == .5


def test_no_deduplication_or_averaging_conditions():
    first = row("a", 1)
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_condition([first, first], {"method": "a"}, "J_primary")
    second = {**row("a", 0, realization="r2"), "condition_id": "other"}
    with pytest.raises(ValueError, match="Ambiguous condition"):
        aggregate_condition([first, second], {"method": "a"}, "J_primary")


def test_inference_pairs_seeds_without_counting_them_as_users():
    rows = [row(m, v, rid=str(r), family="F0" if r < 2 else "F1", seed=s)
            for r in range(4) for s in (101, 202, 303) for m, v in (("a", 1), ("b", 0))]
    result = paired_contrast(rows, {"method": "a"}, {"method": "b"}, "J_primary", resamples=40)
    assert result["estimate"] == 1
    assert result["independent_roots"] == 4
    assert len(result["training_seeds"]) == 3
    assert result["confidence"] == CONFIDENCE


def test_unmatched_seeds_rejected():
    rows = [row("a", 1), row("b", 0, seed=202)]
    with pytest.raises(ValueError, match="paired"):
        paired_contrast(rows, {"method": "a"}, {"method": "b"}, "J_primary", resamples=10)


def test_same_roots_with_different_worlds_are_not_paired():
    with pytest.raises(ValueError, match="worlds"):
        paired_contrast([row("a", 1), row("b", 0, world="other")], {"method": "a"}, {"method": "b"}, "J_primary", resamples=10)


def test_reader_cannot_vary_between_roots_in_a_contrast_arm():
    rows = [{**row("a", 1, rid="r1"), "reader_id": "symbolic"},
            {**row("a", 0, rid="r2"), "reader_id": "llm"}]
    with pytest.raises(ValueError, match="across roots"):
        aggregate_condition(rows, {"method": "a"}, "J_primary")


def test_zero_discordance_small_sample_cannot_prove_guard():
    rows = [row(m, 0, rid=str(r)) for m in ("a", "b") for r in range(5)]
    result = paired_contrast(rows, {"method": "a"}, {"method": "b"}, "J_primary",
                             resamples=20, harm_margin=.02)
    assert result["interval"] == [0, 0]
    assert result["zero_discordance_sensitivity"]["max_absolute_mean_difference_if_metric_in_0_1"] > .02
    assert result["guard"]["passed"] is False


def test_family_weight_is_equal_not_episode_weight():
    rows = [row(m, v, rid=str(r), family=f) for m in ("a", "b")
            for r, f, av in ((0, "F0", 1), (1, "F0", 1), (2, "F1", 0))
            for v in [av if m == "a" else 0]]
    assert paired_contrast(rows, {"method": "a"}, {"method": "b"}, "J_primary", resamples=20)["estimate"] == .5
