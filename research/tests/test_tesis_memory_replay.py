from copy import deepcopy

import pytest

from research.competence_evaluation import _run_trajectory
from research.src.artifacts import canonical, digest, file_hash, write_json
from research.src.policy.rule_controller import RuleController
from research.tesis_bank import build_splits
from research.tesis_comparison import METHODS
from research.tesis_memory_replay import run, run_comparison


@pytest.fixture
def ongoing_change():
    splits = build_splits({"train": 5, "validation": 5, "evaluation": 5})
    return next(episode for episode in splits["evaluation"] if episode["scenario_family"] == "F1")


def archive(tmp_path, episode, controller):
    row = _run_trajectory(controller, episode, tmp_path / "source.sqlite")
    row["evaluation_role"] = "evaluation"
    path = tmp_path / "trajectories.jsonl"
    path.write_text(canonical(row) + "\n", encoding="utf-8")
    return path


def counts(report, mode, metric, family="all"):
    entry = report["write_contrasts"][mode][family][metric]
    return entry["numerator"], entry["denominator"]


def test_reserved_replay_preserves_a_supported_write_for_every_method(tmp_path, ongoing_change):
    source = archive(tmp_path, ongoing_change, RuleController())
    selection = tmp_path / "SELECTION.json"
    write_json(selection, {"selection_complete": True, "selection_split": "validation",
                           "selected_before_evaluation": True,
                           "heldback_episodes_sha256": digest([ongoing_change])})
    results = tmp_path / "RESULTS.json"
    write_json(results, {"status": "completed_exploratory", "seeds": [101],
                         "selection_sha256": file_hash(selection),
                         "panels": {"101": {method: {"trajectories": source.name,
                                                     "trajectories_sha256": file_hash(source)} for method in METHODS}}})
    output = tmp_path / "replay"
    combined = run_comparison(results, output)
    assert combined["scope"]["independent_roots"] == 1
    assert combined["scope"]["complete_paired_cells"] == 1
    assert set(combined["methods"]) == set(METHODS)
    for method, index in combined["methods"].items():
        assert file_hash(output / index["result"]) == index["result_sha256"]
        for mode in ("veto", "repair"):
            assert counts(index, mode, "supported_proposals_blocked", "F1") == (0, 1)
            assert counts(index, mode, "supported_effects_lost", "F1") == (0, 1)
    expected = source.read_text(encoding="utf-8")
    source.write_text(expected + "\n", encoding="utf-8")
    invalid_output = tmp_path / "invalid-replay"
    with pytest.raises(ValueError, match="Trajectory hash mismatch"):
        run_comparison(results, invalid_output)
    assert not invalid_output.exists()


def test_bad_batch_keeps_proposal_counts_separate_from_executions(tmp_path, ongoing_change):
    class InvalidBatch(RuleController):
        def write(self, observation):
            good = super().write(observation)["operations"][0]
            return {"operations": [good, {"op": "ADD"}]}

    original = archive(tmp_path, ongoing_change, InvalidBatch())
    result = run(original, tmp_path / "replay")
    assert result["methods"]["original"]["invalid_output"]["all"]["numerator"] == 1
    assert result["methods"]["veto"]["invalid_output"]["all"]["numerator"] == 0
    for mode in ("veto", "repair"):
        assert counts(result, mode, "invalid_proposals_rejected") == (1, 1)
        assert counts({"write_contrasts": result["write_contrasts_by_seed"][101]}, mode,
                      "invalid_proposals_rejected", "F1") == (1, 1)
        assert counts(result, mode, "supported_proposals_blocked") == (0, 1)
        assert counts(result, mode, "supported_effects_lost") == (0, 0)


def test_replacing_a_wrong_citation_avoids_unsupported_effect(tmp_path, ongoing_change):
    class BadCitation(RuleController):
        def write(self, observation):
            operation = deepcopy(super().write(observation)["operations"][0])
            operation["record"]["evidence_quotes"][0]["text"] = "Una cita que el usuario nunca dijo."
            return {"operations": [operation]}

    original = archive(tmp_path, ongoing_change, BadCitation())
    result = run(original, tmp_path / "replay")
    for mode in ("veto", "repair"):
        assert counts(result, mode, "unsupported_proposals_rejected") == (1, 1)
        assert counts(result, mode, "unsupported_effects_avoided") == (1, 1)
    assert result["methods"]["repair"]["future_supported"]["all"]["numerator"] == 1
