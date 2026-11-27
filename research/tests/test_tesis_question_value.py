from pathlib import Path

import pytest

from research.tesis_bank import build_splits
from research.tesis_question_value import prefix_summary, run


def test_best_question_is_the_same_for_identical_public_prefixes():
    def world(identity, values):
        return {"latent_world_id": identity, "natural_action": 0,
                "branches": [{"return_before_question_cost": value, "questions": int(index != 0),
                              "C2": 1, "FJS": int(index != 0), "strict_joint": int(index != 0)}
                             for index, value in enumerate(values)]}

    summary = prefix_summary([world("A", (0, 3, 0, 0, 0)), world("B", (0, 0, 1, 0, 0))])
    at_zero = summary["frontier"][0]
    assert at_zero["action"] == "TEMPORAL_SCOPE"
    assert at_zero["best_shared_return"] == 1.5
    assert at_zero["per_world_upper_bound"] == 2
    assert at_zero["perfect_information_gap"] == .5
    assert summary["question_break_even_cost"] == 1.5
    assert summary["frontier"][-1]["action"] == "PROCEED"


def test_temporal_pair_holds_current_response_fixed_when_hiding_reply(tmp_path):
    splits = build_splits({"train": 5, "validation": 5, "evaluation": 5})
    rows = [row for row in splits["validation"] if row["scenario_family"] == "F4"]
    output = tmp_path / "result"
    result = run(rows, output, provenance={})
    assert result["episodes"] == 2
    assert result["roots"] == 1
    assert result["legal_branches"] == 10
    assert result["prefixes"][0]["worlds"] == 2
    assert result["writer_intervention"]["denominator"] == 2
    assert result["writer_intervention"]["improved"] == 2
    assert (output / "COSTE_POR_FAMILIA.csv").is_file()
    with pytest.raises(FileExistsError):
        run(rows, output, provenance={})


def test_branch_audit_refuses_heldback_data(tmp_path):
    output = Path(tmp_path) / "audit"
    with pytest.raises(ValueError, match="validation"):
        run([{"split": "evaluation"}], output, provenance={})
    assert not output.exists()
