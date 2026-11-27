import pytest

from research.src.analysis.power_from_validation import plan_from_rows


def validation_rows():
    return [{"method": method, "seed": seed, "root_id": str(root), "latent_world_id": "w", "realization_id": "v",
             "scenario_family": "F" + str(root % 5), "split": "validation", "metrics": {"FJS": 1}}
            for method in ("B1", "B2-L", "B3", "B2-I") for seed in (101, 202, 303) for root in range(20)]


def test_zero_variance_pilot_has_uncertainty_floor_and_no_power_one_claim(tmp_path):
    result = plan_from_rows(validation_rows(), tmp_path, simulations=40, sizes=[300, 1200])
    assert result["validation_only"]
    assert len(result["contrasts"]) == 5
    for contrast in result["contrasts"][:3]:
        assert contrast["homogeneous_pilot"]
        assert contrast["additional_root_variance_floor"] > 0
        assert all(0 < s["estimated_power"] < 1 for s in contrast["scenarios"])
        assert contrast["training_seeds"] == [101, 202, 303]


def test_power_refuses_test_and_missing_pairs_are_explicit(tmp_path):
    rows = validation_rows()
    rows[0]["split"] = "test_id"
    with pytest.raises(ValueError, match="VALIDATION_ONLY"):
        plan_from_rows(rows, tmp_path, simulations=20, sizes=[300])
    rows = validation_rows()[:-1]
    result = plan_from_rows(rows, tmp_path, simulations=20, sizes=[300])
    assert result["precision_limited"]
    assert result["contrasts"][2]["status"] == "inconclusive"
