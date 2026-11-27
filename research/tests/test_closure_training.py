"""Critical closure training checks, causal evidence and resource accounting."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_local_001.model import Generation
from research.src.closure_training import (
    BudgetedValidation, algebra_fixtures, audit_b3_window, run_pilot, run_training_checks,
)
from research.src.training.episodic_grpo import EpisodeRollout, PhaseSample, group_telemetry, verify_reference
from research.src.training.sft import TrainingBudget


def rollout(seed, *, r2=1, r3=1, questions=0):
    phases = [PhaseSample("select", Generation([10, 11], [12], '{"action":"ASK","slot":"TEMPORAL_SCOPE"}'))]
    return EpisodeRollout("root", "world", seed, phases, r2, r3, questions=questions)


def test_group_telemetry_does_not_mistake_penalty_variation_for_task_signal():
    result = group_telemetry([rollout(1), rollout(2, questions=1), rollout(3), rollout(4)])
    assert result["std_task"] == 0
    assert result["std_total"] > 0
    assert result["semantic_action_diversity"] == 1
    assert result["empirical_action_entropy_nats"] == 0
    assert len(set(result["observable_prompt_token_sha256"])) == 1
    assert result["advantages"][1] < 0
    assert result["returns_by_component"][1]["question_penalty"] == -.1


def test_short_group_task_returns_ignore_future():
    result = group_telemetry([rollout(i, r3=i - 2) for i in range(4)], "short")
    assert result["task_returns"] == [1] * 4
    assert result["std_total"] == 0
    assert group_telemetry([rollout(i, r3=i - 2) for i in range(4)], "long")["std_task"] > 0


def test_identical_lora_names_do_not_certify_reference_weights():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    model.peft_config = {}
    reference_model = torch.nn.Linear(2, 2)
    reference_model.peft_config = {}
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    actor = SimpleNamespace(model=model, torch=torch, config=SimpleNamespace(temperature=1.))
    reference = SimpleNamespace(model=reference_model, torch=torch, config=SimpleNamespace(temperature=1.))
    with pytest.raises(ValueError, match="effective weights differ"):
        verify_reference(actor, reference, smoke=True)
    reference_model.load_state_dict(model.state_dict())
    verify_reference(actor, reference, smoke=True)


def test_tiny_tensor_signal_changes_probability_in_expected_direction():
    pytest.importorskip("torch")
    checks = algebra_fixtures()
    assert all(check["status"] == "verified" for check in checks)
    gradient = next(check for check in checks if check["check_id"] == "TRAIN-GRAD-01")
    assert gradient["selected_probability_after"] > gradient["selected_probability_before"]
    assert gradient["gradient_norm_post_clip"] <= 1.00001
    assert "not Qwen" in gradient["scope"]


def test_time_budget_excludes_validation_without_adding_candidates(monkeypatch):
    current = [0.]
    monkeypatch.setattr("research.src.training.sft.time.monotonic", lambda: current[0])
    clock = TrainingBudget(30., [1/3, 2/3, 1.])
    clock.pause(5.)  # initial evaluation is accounted, before the clock starts
    clock.start()
    current[0] = 10.
    assert clock.checkpoint_due(999.)
    assert not clock.checkpoint_due(999.)
    current[0] = 110.
    clock.pause(100.)
    assert clock.elapsed(999.) == 10.
    assert not clock.exhausted(999.)
    current[0] = 131.
    assert clock.checkpoint_due(999.)  # crosses remaining thresholds atomically
    assert clock.exhausted(999.)
    clock.finish()
    current[0] = 500.
    assert clock.elapsed(999.) == 31.
    assert clock.paused_seconds == 105.


def test_legacy_budget_remains_wall_time():
    budget = TrainingBudget(10.)
    budget.start()
    budget.pause(100.)
    assert budget.elapsed(8.) == 8.
    assert not budget.exhausted(8.)
    assert budget.exhausted(11.)


def test_validation_budget_refuses_new_complete_evaluation(monkeypatch):
    current = [0.]
    monkeypatch.setattr("research.src.closure_training.time.monotonic", lambda: current[0])
    def evaluate(_):
        current[0] += 11
        return {"utility": 1.}
    validator = BudgetedValidation(evaluate, 10.)
    assert validator(None) == {"utility": 1.}
    assert validator.counters()["overrun_seconds"] == 1.
    with pytest.raises(RuntimeError, match="VALIDATION_BUDGET_EXHAUSTED"):
        validator(None)
    assert validator.calls == 1


def test_b3_window_preserves_original_examples_and_limits_causal_claim(tmp_path):
    branch = tmp_path / "parent/continuations/303/B3"
    branch.mkdir(parents=True)
    rows = [{"step": step, "example_index": step, "phase": "write",
             "generation": {"prompt_ids": [1, 2], "completion_ids": [3], "text": "{}"},
             "actor_mask": [0, 0, 1]} for step in (3400, 3401)]
    (branch / "prefixes.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    (branch / "updates.jsonl").write_text("\n".join(json.dumps({"step": row["step"], "loss": 10.}) for row in rows))
    (branch / "training.json").write_text(json.dumps({"plan": {"learning_rate": .0002}}))
    output = tmp_path / "output"
    output.mkdir()
    result = audit_b3_window(tmp_path / "parent", output, 3400, 3401)
    assert result["status"] == "inspected"
    assert result["exact_replay"] == "not_executed"
    assert result["cause_classification"] == "hypothesis"
    assert result["mask_correct"]
    assert result["training_plan"]["learning_rate"] == .0002
    assert len((output / "B3_303_WINDOW.jsonl").read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        audit_b3_window(tmp_path / "parent", output, 3400, 3401)


def test_unexecuted_checks_cannot_pass_training_gate(tmp_path):
    result = run_training_checks(tmp_path / "missing", tmp_path, run_torch_fixtures=False,
                                 check_real_b1=False, training_seeds=(101,))
    assert result["status"] == "blocked_missing_evidence"
    assert not result["critical_checks_passed"]
    assert result["learning_evidence"] == "not_executed"
    with pytest.raises(FileExistsError):
        run_training_checks(tmp_path / "missing", tmp_path)


def test_pilot_refuses_missing_authorization_before_any_model_load(tmp_path):
    from research.src.protocol import GateError
    with pytest.raises(GateError, match="G0_REQUIRED"):
        run_pilot(tmp_path / "parent", tmp_path / "output")
    assert not (tmp_path / "output/pilot").exists()


def test_v4_selection_obeys_validity_guards_and_keeps_earliest_numeric_tie():
    from research.src.training.sft import selection_is_better
    config = {"utility_tolerance": 1e-6, "validity_guards": {"invalid_output_rate": .02, "unsupported_writes": .02}}
    initial = {"utility": 1., "invalid_output_rate": .01, "unsupported_writes": .01, "questions": 1.}
    # Lower question cost or fewer writes do not displace an equal-utility start.
    assert not selection_is_better({**initial, "questions": 0., "unsupported_writes": 0.}, 10, initial, 0, config)
    assert not selection_is_better({**initial, "utility": 1.0000001}, 10, initial, 0, config)
    assert selection_is_better({**initial, "utility": 1.1}, 10, initial, 0, config)
    assert not selection_is_better({**initial, "utility": 2., "invalid_output_rate": .03}, 10, initial, 0, config)
    assert selection_is_better(initial, 10, {**initial, "invalid_output_rate": .03}, 0, config)


def test_validation_weights_realizations_then_worlds_not_raw_rows():
    from research.src.training.campaign import aggregate_validation
    def row(world, realization, success):
        return {"scenario_family": "F4", "root_id": "r", "latent_world_id": world, "realization_id": realization,
                "rewards_by_component": {"R_long": 2 * success},
                "metrics": {"FJS": success, "C2": success, "unsupported_write_episode": 0, "questions": 0, "invalid_outputs": 0}}
    rows = [row("a", "one", 1), row("a", "two", 1), row("a", "three", 1), row("b", "one", 0)]
    assert aggregate_validation(rows)["future_success"] == .5
    assert aggregate_validation(rows + [rows[0]])["future_success"] == .5


def test_checkpoint_save_budget_is_separate_and_stops_after_atomic_overrun(monkeypatch):
    from research.src.training.sft import CheckpointSaveBudget
    current = [0.]
    monkeypatch.setattr("research.src.training.sft.time.monotonic", lambda: current[0])
    budget = CheckpointSaveBudget(10.)
    def save():
        current[0] = 11.
    budget.save(save)
    assert budget.counters()["checkpoint_save_seconds"] == 11.
    assert budget.counters()["checkpoint_save_overrun_seconds"] == 1.
    assert budget.counters()["checkpoint_save_operations"] == 1
    with pytest.raises(RuntimeError, match="CHECKPOINT_SAVE_BUDGET_EXHAUSTED"):
        budget.save(lambda: None)
    with pytest.raises(RuntimeError, match="CHECKPOINT_SAVE_BUDGET_EXHAUSTED"):
        budget.ensure_available()
    assert budget.operations == 1
