"""Independent tiny contracts; no archived bank, model or GPU is opened."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_local_001.model import Generation
from research.competence_evaluation import (
    DEFAULT_THRESHOLDS, aggregate_rows, capability_gate, descriptive_contrasts, evaluate_selected,
)
from research.src.artifacts import digest, file_hash
from research.src.policy.rule_controller import RuleController


def _record(mid, value, start, *, kind="ongoing", end=None):
    text = "En general prefiero " + value + " para cenar."
    return {"memory_id": mid, "attribute": "dinner_preference", "value": value,
            "scope": {"kind": kind, "valid_from": start, "valid_until_exclusive": end,
                      "session_id": None, "context": None}, "status": "confirmed",
            "source_turn_ids": ["s1_" + mid], "evidence_quotes": [{"turn_id": "s1_" + mid, "text": text}],
            "recorded_at": start, "supersedes": []}


def _episode(*, ambiguous=False, world="ongoing", split="validation"):
    initial = [_record("m0", "pasta", "2026-01-01"),
               _record("m1", "arroz", "2026-01-19", kind="interval", end="2026-01-20")]
    return {"root_id": "r_ambiguous" if ambiguous else "r_clear", "latent_world_id": "w_" + world,
            "realization_id": "v0", "split": split, "scenario_family": "F4" if ambiguous else "F0",
            "initial_memory": initial,
            "user_messages": [{"turn_id": "u2", "text": "Ahora prefiero sopa. ¿Qué ceno hoy?" if ambiguous else "¿Qué ceno hoy?"}],
            "current_query": {"attribute": "dinner_preference", "date": "2026-01-21", "session_id": "S2", "context": None},
            "future_query": {"attribute": "dinner_preference", "date": "2026-01-28", "session_id": "S3", "context": None},
            "replies": {"TEMPORAL_SCOPE": {"turn_id": "u3", "text": "Como preferencia habitual desde ahora." if world == "ongoing" else "Solo en esta conversación."}},
            "truth": {"current_value": "sopa" if ambiguous else "pasta",
                      "future_value": "sopa" if ambiguous and world == "ongoing" else "pasta"},
            "opportunity": {"eligible": ambiguous, "negative_control": not ambiguous, "slot": "TEMPORAL_SCOPE" if ambiguous else None}}


def _protocol(**overrides):
    return {"thresholds": {**DEFAULT_THRESHOLDS, "minimum_roots": 1}, **overrides}


def _rows(output):
    return [json.loads(line) for line in (output / "trajectories.jsonl").read_text().splitlines()]


def test_real_sqlite_and_strict_phase_metrics_with_positive_teacher(tmp_path):
    output = tmp_path / "validation"
    result = evaluate_selected(RuleController(), [_episode()], output, protocol=_protocol(), include_interventions=False)
    assert result["status"] == "completed"
    assert result["metrics"]["C2"] == result["metrics"]["FJS"] == 1
    assert result["metrics"]["select_format"] == result["metrics"]["write_faithful_episode"] == 1
    assert result["capability_gate"]["passed"] is True
    row = _rows(output)[0]
    assert row["persistence_verification"]["status"] == "verified"
    assert Path(row["persistence_verification"]["sqlite_path"]).exists()
    assert set(row["trace"]["reuse_observation"]) == {"memory", "query"}
    assert row["select_gold_independent"] is False
    assert result["teacher_independent_gold"] is False
    assert (output / "source_snapshot/research/competence_evaluation.py").is_file()


def test_pi1_really_forces_proceed_and_pi2_reuses_identical_response(tmp_path):
    output = tmp_path / "panels"
    episodes = [_episode(), _episode(ambiguous=True), _episode(ambiguous=True, world="session")]
    result = evaluate_selected(RuleController(), episodes, output, protocol=_protocol())
    assert result["status"] == "completed", result.get("failure")
    rows = _rows(output)
    pi1 = [r for r in rows if r["route"] == "PROCEED"]
    assert len(pi1) == 6
    assert all(r["phases"]["select"]["source"] == "predeclared_intervention" for r in pi1)
    assert all(r["metrics"]["questions"] == 0 for r in pi1)
    clear = [r for r in pi1 if r["root_id"] == "r_clear"]
    assert {r["condition"]: r["metrics"]["C2"] for r in clear} == {"O1": 0, "O2": 1}
    pairs = {}
    for row in rows:
        if row["route"] in {"I1", "I2"}:
            pairs.setdefault((row["root_id"], row["latent_world_id"]), {})[row["route"]] = row
    for pair in pairs.values():
        a, b = pair["I1"], pair["I2"]
        assert a["trace"]["response"] == b["trace"]["response"]
        assert a["metrics"]["C2"] == b["metrics"]["C2"]
        assert a["phases"]["respond"]["observation_sha256"] == b["phases"]["respond"]["observation_sha256"]
        assert a["paired_response_sha256"] == b["paired_response_sha256"]
        assert b["phases"]["respond"]["source"] == "exact_same_paired_response"
        assert len(a["trace"]["observations"]["write"]["user_messages"]) + 1 == len(b["trace"]["observations"]["write"]["user_messages"])


def test_correct_value_with_unrelated_citation_gets_no_current_task_credit(tmp_path):
    class WrongCitation(RuleController):
        def respond(self, observation):
            return {"mode": "ANSWER", "value": "pasta", "branches": [], "evidence_ids": ["invented"]}
    result = evaluate_selected(WrongCitation(), [_episode()], tmp_path / "bad", protocol=_protocol(),
                               include_interventions=False, include_teacher=False)
    assert result["metrics"]["respond_value_correct"] == 1
    assert result["metrics"]["respond_citations_supported"] == 0
    assert result["metrics"]["C2"] == 0
    assert result["metrics"]["J_primary"] == 0
    assert result["capability_gate"]["passed"] is False


def test_unfaithful_write_is_not_hidden_by_wellformed_json(tmp_path):
    class Unfaithful(RuleController):
        def write(self, observation):
            record = _record("bad", "pizza", "2026-01-21")
            record.pop("memory_id")
            record.pop("recorded_at")
            record["source_turn_ids"] = ["u2"]
            record["evidence_quotes"] = [{"turn_id": "u2", "text": observation["user_messages"][0]["text"]}]
            return {"operations": [{"op": "ADD", "record": record}]}
    result = evaluate_selected(Unfaithful(), [_episode()], tmp_path / "badwrite", protocol=_protocol(),
                               include_interventions=False, include_teacher=False)
    assert result["metrics"]["write_format"] == 1
    assert result["metrics"]["write_faithful_episode"] == 0
    assert result["metrics"]["unsupported_write_count"] == 1
    assert result["capability_gate"]["passed"] is False


def test_raw_invalid_json_and_truncation_are_preserved_without_repair(tmp_path):
    class Backend:
        config = SimpleNamespace(max_new_tokens=64)
        def encode_prompt(self, prompt):
            return [1, 2]
        def generate(self, prompt, sample=False):
            assert sample is False
            return Generation(text='{"action":', prompt_ids=[1, 2], completion_ids=[3], truncated=True)
    result = evaluate_selected(Backend(), [_episode()], tmp_path / "raw", protocol=_protocol(), include_interventions=False, include_teacher=False)
    assert result["status"] == "completed"
    assert result["metrics"]["invalid_output_count"] == 3
    assert result["metrics"]["C2"] == 0
    row = _rows(tmp_path / "raw")[0]
    for phase in row["phases"].values():
        record = phase["records"][0]
        assert record["raw_output"] == '{"action":'
        assert record["truncated"] is True
        assert record["fallback_used"] is False


def test_technical_failure_preserves_partial_evidence_and_blocks_gate(tmp_path):
    class Crash(RuleController):
        def write(self, observation):
            raise RuntimeError("synthetic device failure")
    result = evaluate_selected(Crash(), [_episode()], tmp_path / "crash", protocol=_protocol(), include_interventions=False, include_teacher=False)
    assert result["status"] == "failed"
    assert result["completed_episodes"] == 0
    assert result["capability_gate"]["status"] == "blocked_missing_evidence"
    assert "synthetic device failure" in result["failure"]["error"]
    assert list((tmp_path / "crash/states").glob("*.sqlite"))
    assert (tmp_path / "crash/predeclaration.json").is_file()


def test_validation_selection_never_accepts_final_or_training_episodes(tmp_path):
    for split in ("train", "evaluation", "test_id"):
        with pytest.raises(ValueError, match="validation-only"):
            evaluate_selected(RuleController(), [_episode(split=split)], tmp_path / split, protocol=_protocol())
    with pytest.raises(ValueError, match="Explicit capability thresholds"):
        evaluate_selected(RuleController(), [_episode()], tmp_path / "missing")


def test_heldback_requires_hashes_and_can_never_be_used_for_capability_selection(tmp_path):
    episode = _episode(split="evaluation")
    with pytest.raises(ValueError, match="validation-selection"):
        evaluate_selected(RuleController(), [episode], tmp_path / "unauthorized", protocol=_protocol(), role="evaluation")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"version": "competence_selection_v1", "selection_complete": True,
        "selection_split": "validation", "selected_before_evaluation": True, "heldback_episodes_sha256": digest([episode]),
        "rows": [{"method": "B1c", "training_seed": 101, "fake_actor_fingerprint": "contract_fixture"}]}))
    authorization = {"selection_manifest_path": str(selection), "selection_manifest_sha256": file_hash(selection),
                     "locked_bank_sha256": digest([episode])}
    result = evaluate_selected(RuleController(), [episode], tmp_path / "heldback", role="evaluation",
                               protocol=_protocol(heldback_authorization=authorization, test_only=True, fake_actor_fingerprint="contract_fixture"), include_interventions=False)
    assert result["selection_eligible"] is False
    assert result["capability_gate"]["status"] == "not_applicable"
    assert result["capability_gate"]["passed"] is False


def test_thresholds_cannot_be_changed_after_seeing_scores(tmp_path):
    result = evaluate_selected(RuleController(), [_episode()], tmp_path / "eval", protocol=_protocol(), include_interventions=False)
    with pytest.raises(ValueError, match="frozen before generation"):
        capability_gate(result, {**result["thresholds"], "C2": 0})


def test_unequal_realizations_do_not_overweight_a_world():
    rows = []
    for world, realizations, value in (("w1", 3, 0), ("w2", 1, 1)):
        for index in range(realizations):
            rows.append({"root_id": "r", "latent_world_id": world, "realization_id": str(index),
                         "scenario_family": "F0", "route": "natural", "condition": "O2", "metrics": {"C2": value}})
    result = aggregate_rows(rows)
    assert result["metrics"]["C2"] == .5


def test_descriptive_contrasts_preserve_negative_results_and_missing_arms():
    rows = []
    for seed in (101, 202, 303):
        for method, value in (("B1c", 1), ("B3", 1), ("B2-I", 1), ("B2-L", 0)):
            rows.append({"method": method, "training_seed": seed, "root_id": "r", "latent_world_id": "w", "realization_id": "v0",
                         "scenario_family": "F0", "route": "natural", "condition": "O2", "metrics": {"J_primary": value}, "evaluation_role": "evaluation"})
    result = descriptive_contrasts(rows)
    assert result["status"] == "incomplete"
    assert all(r["outcome"] == "negative_descriptive" for r in result["contrasts"] if r["status"] == "completed")
    assert result["contrasts"][0]["status"] == "incomplete"
    incomplete = descriptive_contrasts([r for r in rows if r["training_seed"] != 303])
    assert all(r["status"] == "incomplete" for r in incomplete["contrasts"])


def test_nonidentical_worlds_cannot_form_a_descriptive_pair():
    rows = []
    for seed in (101, 202, 303):
        for method in ("B2-L", "B1c"):
            rows.append({"method": method, "training_seed": seed, "root_id": "r", "latent_world_id": method, "realization_id": "v0",
                         "scenario_family": "F0", "route": "natural", "condition": "O2", "metrics": {"J_primary": 1}, "evaluation_role": "evaluation"})
    result = descriptive_contrasts(rows)
    assert next(r for r in result["contrasts"] if r["contrast"] == "PI3_RLL_minus_B1c")["status"] == "incomplete"


def test_unaffected_control_is_declared_by_authorship_not_meal_domain(tmp_path):
    for attribute in ("drink_preference", "breakfast_preference"):
        episode = _episode()
        for memory in episode["initial_memory"]:
            memory["attribute"] = attribute
        episode["current_query"]["attribute"] = attribute
        episode["future_query"]["attribute"] = attribute
        result = evaluate_selected(RuleController(), [episode], tmp_path / attribute, protocol=_protocol(),
                                   include_interventions=False, include_teacher=False)
        assert result["status"] == "completed"
        assert _rows(tmp_path / attribute)[0]["pi2_cohort"] == "clear_control"
    episode = _episode()
    episode["authoring"] = {"stratum": "unrelated_attribute_control"}
    evaluate_selected(RuleController(), [episode], tmp_path / "control", protocol=_protocol(), include_interventions=False, include_teacher=False)
    assert _rows(tmp_path / "control")[0]["pi2_cohort"] == "unaffected_attribute"


def test_programming_or_context_errors_are_technical_and_not_model_format(tmp_path):
    class ProgrammingBug(RuleController):
        def respond(self, observation):
            raise KeyError("programming bug")
    result = evaluate_selected(ProgrammingBug(), [_episode()], tmp_path / "bug", protocol=_protocol(), include_interventions=False)
    assert result["status"] == "failed"
    assert result["capability_gate"]["passed"] is False
    partial = json.loads(next((tmp_path / "bug/states").glob("*.partial.json")).read_text())
    assert partial["scores"] is None
    assert "KeyError" in partial["error"]


def test_pi2_primary_and_controls_are_reported_separately_without_outcome_selection():
    rows = []
    for seed in (101, 202, 303):
        for cohort, first, second in (("eligible_ambiguous", 0, 1), ("clear_control", 1, 0)):
            for route, value in (("I1", first), ("I2", second)):
                rows.append({"method": "B1c", "training_seed": seed, "root_id": cohort, "latent_world_id": "w", "realization_id": "v0",
                             "scenario_family": "F4", "route": route, "condition": "O2", "metrics": {"FJS": value},
                             "evaluation_role": "evaluation", "pi2_cohort": cohort})
    result = descriptive_contrasts(rows)["contrasts"][1]
    assert result["status"] == "completed"
    assert result["estimate"] == 1
    assert result["by_cohort"]["clear_control"]["estimate"] == -1
    assert result["pooled_cohorts_diagnostic_only"]["estimate"] == 0


def test_new_data_teacher_factory_runs_as_separate_control_on_exact_public_inputs(tmp_path):
    from research.competence_data import teacher_for_episode
    result = evaluate_selected(RuleController(), [_episode(ambiguous=True)], tmp_path / "teacher", protocol=_protocol(),
                               include_interventions=False, teacher_factory=teacher_for_episode)
    assert result["status"] == "completed", result["failure"]
    assert result["teacher_control"]["metrics"]["C2"] == 1
    assert result["teacher_control"]["metrics"]["FJS"] == 1
    assert result["teacher_independent_gold"] is False
