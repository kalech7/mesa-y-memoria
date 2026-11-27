"""Campaign scheduling, grouped validation and safe recovery contracts."""
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.src.training.campaign import (
    aggregate_validation, _completed_run, _record_plan, _selected_metrics,
    ceiling_diagnostic, validation_callback,
)
from research.src.training.sft import RunJournal, _initial_metrics
from research.src.training.episodic_grpo import reference_logprobs_on_actor_device


def row(family, root, future, utility):
    return {"scenario_family": family, "root_id": root,
            "rewards_by_component": {"R_long": utility},
            "metrics": {"FJS": future, "C2": 1, "unsupported_write_episode": 0,
                        "questions": 1, "invalid_outputs": 0}}


def test_validation_averages_worlds_roots_and_families_in_that_order():
    rows = [row("F0", "a", 1, 2), row("F0", "a", 0, 0),
            row("F0", "b", 1, 2), row("F1", "c", 0, 0)]
    metrics = aggregate_validation(rows)
    assert metrics["future_success"] == .375
    assert metrics["utility"] == .75
    assert aggregate_validation(rows[:2] + rows[:2] + rows[2:]) == metrics


def test_validation_cache_avoids_reexecuting_identical_b1():
    def prohibited(_):
        raise AssertionError("Selected B1 validation must not be run again")
    metrics = {"utility": 1.5, "unsupported_writes": 0, "questions": .2}
    copy, key = _initial_metrics(None, prohibited, metrics)
    assert copy == metrics
    assert copy is not metrics
    assert key == (1.5, 0, -.2, 0)


def test_partial_campaign_never_resets_optimizer_silently(tmp_path):
    (tmp_path / "run_state.json").write_text('{"status":"running"}')
    with pytest.raises(RuntimeError, match="EXPLICIT_RECOVERY"):
        _completed_run(tmp_path)


def test_campaign_configuration_changes_need_a_new_run_directory(tmp_path):
    _record_plan(tmp_path, {"epochs": 2, "checkpoint_every_examples": 7140})
    _record_plan(tmp_path, {"epochs": 2, "checkpoint_every_examples": 7140})
    with pytest.raises(RuntimeError, match="PLAN_CHANGED"):
        _record_plan(tmp_path, {"epochs": 1, "checkpoint_every_examples": 10})


def test_journal_preserves_last_completed_step_after_failure(tmp_path):
    with pytest.raises(RuntimeError, match="probe_failure"):
        with RunJournal(tmp_path, "sft", {"epochs": 2}) as journal:
            journal.progress(10, 100, loss=.5)
            journal.event("validation_started", step=10)
            raise RuntimeError("probe_failure")
    state = json.loads((tmp_path / "run_state.json").read_text())
    assert state["status"] == "failed"
    assert state["last_completed_step"] == 10
    assert "optimizer" in state["recovery"]
    assert len((tmp_path / "progress.jsonl").read_text().splitlines()) == 4


def test_ceiling_reports_negative_margin_without_modifying_the_plan():
    metrics = {"future_success": .95, "unsupported_writes": .01, "utility": 1.8}
    assert ceiling_diagnostic(metrics)["status"] == "ceiling_detected"
    assert ceiling_diagnostic({**metrics, "unsupported_writes": .02})["status"] == "margin_available"
    assert _selected_metrics({"best_step": 0, "history": [{"step": 0, "validation": metrics}]}) == metrics


def test_validation_rejects_a_training_bank():
    with pytest.raises(ValueError, match="validation-only"):
        validation_callback([{"split": "train"}])


def test_reference_is_offloaded_before_actor_backward_even_after_failure():
    moves = []
    reference = SimpleNamespace(device="cpu", torch=SimpleNamespace(no_grad=nullcontext),
                                likelihood_mode=nullcontext)

    class Model:
        def to(self, device):
            moves.append(device)
            reference.device = device
            return self

    value = SimpleNamespace()
    value.detach = lambda: value
    reference.model = Model()
    reference.sequence_logprobs = lambda generation, sampling_distribution: value
    actor = SimpleNamespace(device="cuda:0")
    assert reference_logprobs_on_actor_device(reference, actor, object()) is value
    assert moves == ["cuda:0", "cpu"]
    assert reference.device == "cpu"

    def fail(*args, **kwargs):
        raise RuntimeError("probe_failure")
    reference.sequence_logprobs = fail
    with pytest.raises(RuntimeError, match="probe_failure"):
        reference_logprobs_on_actor_device(reference, actor, object())
    assert moves[-2:] == ["cuda:0", "cpu"]
    assert reference.device == "cpu"


def test_full_sft_campaign_keeps_both_epochs_and_selects_at_epoch_boundaries(tmp_path, monkeypatch):
    from research.src.training import campaign
    from research.src.closure_environment import _episode
    config = {"seeds": {"training": [101]}, "sft": {"epochs": 2, "learning_rate": .0002,
              "checkpoint_every_epochs": 1}, "model": {"precision": "bfloat16"},
              "runtime": {"evaluation_batch_size": 8, "training_progress_every_updates": 10}}
    monkeypatch.setattr(campaign, "require_gate", lambda *args: None)
    monkeypatch.setattr(campaign, "load_config", lambda path: config)
    def bank(path, split):
        episode = _episode()
        episode.update(split=split, root_id=split + "_root")
        if split == "validation":
            episode["user_messages"][0]["text"] = "Ahora prefiero arroz. ¿Qué ceno hoy?"
            episode["truth"].update(current_value="arroz", future_value="arroz")
        return [episode]
    monkeypatch.setattr(campaign, "load_bank", bank)
    monkeypatch.setattr(campaign, "teacher_trajectories", lambda episodes: list(range(7)))
    monkeypatch.setattr(campaign, "SpecLocalPolicy", lambda *args, **kwargs: object())
    monkeypatch.setattr(campaign, "_release_memory", lambda: None)
    calls = []

    def fake_train(policy, examples, destination, callback, **kwargs):
        calls.append(kwargs)
        return {"status": "executed", "best_step": 0,
                "history": [{"step": 0, "validation": {"utility": 1., "future_success": .5,
                             "unsupported_writes": 0, "questions": 0}}]}

    monkeypatch.setattr(campaign, "train_sft", fake_train)
    campaign.run_sft_campaign("local_model", "train", "validation", tmp_path / "campaign", device="cpu")
    assert calls[0]["epochs"] == 2
    assert calls[0]["checkpoint_every"] == 7
    assert "updates" not in calls[0]
    plan = json.loads((tmp_path / "campaign/campaign_plan.json").read_text())
    assert plan["scheduled_updates_per_seed"] == 14
    assert plan["validation_batch_size"] == 8
