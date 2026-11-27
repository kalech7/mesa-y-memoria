"""New exploratory adaptation: no GPU, no weakening of historical gates."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_local_001.model import Generation, LocalPolicy
from research import competence_training as training
from research.src.artifacts import file_hash, write_json


def test_training_rejects_other_partitions_before_authorization_or_model_load(tmp_path):
    loaded = []
    with pytest.raises(ValueError, match="partition"):
        training.run_training("B1c", "parent", [{"split": "evaluation"}], [{"split": "validation"}], tmp_path / "run",
            seed=101, authorization_dir=tmp_path, plan_path=tmp_path / "plan.json",
            policy_factory=lambda *a, **kw: loaded.append(True))
    assert not loaded
    assert not (tmp_path / "run").exists()


def test_closure_authorization_is_required_before_model_load(tmp_path, monkeypatch):
    def denied(path):
        raise RuntimeError("real authorization missing")
    monkeypatch.setattr("research.src.closure_protocol.require_training_authorization", denied)
    loaded = []
    with pytest.raises(RuntimeError, match="authorization missing"):
        training.run_training("B1c", "parent", [{"split": "train"}], [{"split": "validation"}], tmp_path / "run",
            seed=101, authorization_dir=tmp_path, plan_path=tmp_path / "plan.json",
            policy_factory=lambda *a, **kw: loaded.append(True))
    assert loaded == []


def test_real_causal_completion_likelihood_masks_prompt_targets():
    torch = pytest.importorskip("torch")
    class PositionalLogits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(1, 5, 7))
        def forward(self, input_ids, attention_mask, use_cache):
            return SimpleNamespace(logits=self.logits)
    policy = LocalPolicy.__new__(LocalPolicy)
    policy.torch, policy.model = torch, PositionalLogits()
    policy._check_length = lambda n: None
    generation = Generation([1, 2, 3], [4, 5], "target")
    loss = -policy.sequence_logprobs(generation).mean()
    loss.backward()
    gradient = policy.model.logits.grad[0]
    assert torch.count_nonzero(gradient[:2]) == 0  # no predicting prompt tokens
    assert torch.count_nonzero(gradient[2:4]) > 0  # prompt's last position predicts completion[0]
    assert torch.count_nonzero(gradient[4:]) == 0  # final token has no extra target
    assert gradient[2, 4] < 0 and gradient[3, 5] < 0


def test_initialization_checks_reference_tensor_equality_and_frozen_state(tmp_path):
    torch = pytest.importorskip("torch")
    actor_model = torch.nn.Linear(2, 2)
    reference_model = deepcopy(actor_model)
    actor_model.peft_config = reference_model.peft_config = {}
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    # Real artifact validation remains active, using an actual tiny manifest.
    write_json(tmp_path / "local_checkpoint.json", {"artifacts_sha256": {}})
    config = SimpleNamespace(seed=101, temperature=1.)
    actor = SimpleNamespace(model=actor_model, torch=torch, config=config, origin_checkpoint=str(tmp_path))
    reference = SimpleNamespace(model=reference_model, torch=torch, config=config, origin_checkpoint=str(tmp_path))
    evidence = training.verify_initialization(actor, tmp_path, reference)
    assert evidence["actor_effective_parameter_sha256"] == evidence["reference_effective_parameter_sha256"]
    assert evidence["reference_all_frozen"]
    with torch.no_grad():
        reference_model.weight[0, 0] += .01
    with pytest.raises(ValueError, match="effective weights differ"):
        training.verify_initialization(actor, tmp_path, reference)
    with pytest.raises(ValueError, match="origin"):
        training.verify_initialization(actor, tmp_path / "different")


def test_competence_score_does_not_modify_recorded_reward():
    metrics = {"C2": .2, "FJS": .4, "write_faithful_episode": .6,
               "select_format": 1., "respond_format": .5, "write_format": .0, "utility": -1.2}
    result = training.competence_metrics(metrics)
    assert result["utility"] == pytest.approx(.425)
    assert result["reward_utility"] == -1.2
    assert metrics["utility"] == -1.2
    with pytest.raises(ValueError, match="Missing"):
        training.competence_metrics({"C2": 1.})


def test_terminal_snapshot_is_eligible_and_numerical_ties_choose_earliest(tmp_path):
    for step in (0, 10, 20):
        (tmp_path / f"step_{step:06d}").mkdir()
    result = {"selected_checkpoint": str(tmp_path / "step_000000"), "history": [
        {"step": 0, "validation": {"utility": 0.}},
        {"step": 10, "validation": {"utility": .5}, "checkpoint_eligible": True},
        {"step": 20, "validation": {"utility": .5 + 1e-7}, "checkpoint_eligible": False}]}
    assert training.select_competence_checkpoint(result)["best_step"] == 10
    result["history"][-1]["validation"]["utility"] = .7
    assert training.select_competence_checkpoint(result)["best_step"] == 20


def test_rl_signal_requires_task_variability_and_integrated_success(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "checkpoint_evidence", lambda _: {"checkpoint": "same"})
    path = tmp_path / "signal.json"
    groups = [{"task_returns": [1, 1, 1, 1]} for _ in range(40)]
    row = {"checkpoint_evidence": {"checkpoint": "same"}, "groups": groups, "updates": 0, "split": "train",
           "integrated_improvements": 1, "status": "passed", "parameter_sha256_before": "a" * 64,
           "parameter_sha256_after": "a" * 64}
    write_json(path, row)
    with pytest.raises(ValueError, match="Insufficient"):
        training.verify_signal(path, file_hash(path), "same")
    for group in groups[:4]:
        group["task_returns"] = [0, 1, 0, 1]
    write_json(path, row)
    training.verify_signal(path, file_hash(path), "same")
    row["integrated_improvements"] = 0
    write_json(path, row)
    with pytest.raises(ValueError, match="Insufficient"):
        training.verify_signal(path, file_hash(path), "same")
    with pytest.raises(ValueError, match="hash"):
        training.verify_signal(path, "f" * 64, "same")


def test_teacher_token_preflight_never_truncates():
    examples = [{"phase": phase, "observation": {"memory": [], "user_messages": [], "query": {}, "reuse_regime": "later"},
                 "completion": "{}"} for phase in ("select", "respond", "write")]
    backend = SimpleNamespace(supervised_generation=lambda p, c: Generation([1] * 2049, [2], c))
    with pytest.raises(ValueError, match="no silent truncation"):
        training.profile_teacher(backend, examples)


def test_historical_202_is_selected_7140_not_last_14280():
    assert training.historical_b1(202).name == "step_007140"


def test_teacher_uses_actual_ask_prefixes_and_ignores_private_gold(tmp_path):
    from research.src.data.build_bank import build_bank
    episode = deepcopy(build_bank("dev_smoke", roots=5, seed=9123)[0])
    episode["split"] = "train"
    first = training.prepare_teacher_examples([episode], tmp_path / "first")
    assert [row["phase"] for row in first] == ["select", "respond", "write", "respond", "write", "respond", "write"]
    assert first[3]["route"] == "PROCEED"
    assert first[5]["route"] == "ASK:TEMPORAL_SCOPE"
    assert len(first[5]["observation"]["user_messages"]) > len(first[3]["observation"]["user_messages"])
    changed = deepcopy(episode)
    changed["truth"] = {"invented_private_answer": "not visible to the teacher"}
    changed["evaluator_annotations"] = {"answer": "DO NOT LEAK"}
    second = training.prepare_teacher_examples([changed], tmp_path / "second")
    assert first == second
    assert "DO NOT LEAK" not in json.dumps(first)


def test_selection_callback_real_evaluator_receives_frozen_thresholds(tmp_path):
    from research.src.data.build_bank import build_bank
    from research.src.policy.rule_controller import RuleController
    from research.competence_evaluation import DEFAULT_THRESHOLDS
    episodes = deepcopy(build_bank("dev_smoke", roots=5, seed=9146))
    for episode in episodes:
        episode["split"] = "validation"
    plan = {"budgets": {"validation_seconds": 100, "max_validation_evaluations": 4},
            "capability_thresholds": dict(DEFAULT_THRESHOLDS)}
    callback = training.SelectionValidation(episodes, tmp_path, plan, "B1c", 101)
    metrics = callback(RuleController())
    assert 0 <= metrics["utility"] <= 1
    assert "reward_utility" in metrics
    predeclaration = json.loads((tmp_path / "evaluation_00/predeclaration.json").read_text())
    assert predeclaration["thresholds"] == plan["capability_thresholds"]
    assert predeclaration["include_interventions"] is False
    assert predeclaration["include_teacher"] is False
    assert callback.calls == 1 and callback.elapsed_seconds > 0


def test_incomplete_validation_never_selects_from_partial_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr("research.competence_evaluation.evaluate_selected", lambda *a, **kw:
        {"status": "failed", "completed_episodes": 1, "scheduled_episodes": 2,
         "metrics": {"C2": 1., "FJS": 1.}})
    callback = training.SelectionValidation([], tmp_path,
        {"budgets": {"validation_seconds": 10, "max_validation_evaluations": 4}, "capability_thresholds": {}}, "B1c", 101)
    with pytest.raises(RuntimeError, match="INCOMPLETE_SELECTION_VALIDATION"):
        callback(None)
    assert callback.calls == 1


def test_plan_factory_and_preflight_require_sealed_data_sources_and_parent(tmp_path, monkeypatch):
    from llm_local_001.model import artifact_hashes
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "adapter_model.safetensors").write_bytes(b"fixture, not a loadable model")
    write_json(parent / "local_checkpoint.json", {"artifacts_sha256": artifact_hashes(parent)})
    preregister = tmp_path / "PREREGISTER.json"
    write_json(preregister, {"research_status": "exploratory", "frozen_before_training": True})
    source = tmp_path / "extra_source.txt"
    source.write_text("frozen")
    train = [{"root_id": "train_1", "split": "train"}]
    validation = [{"root_id": "validation_1", "split": "validation"}]
    auth = tmp_path / "authorization"
    auth.mkdir()
    plan = training.make_training_plan("B1c", parent, train, validation, seed=101,
        authorization_dir=auth, frozen_files={str(source): file_hash(source)},
        config={"preregistration": {"path": str(preregister), "sha256": file_hash(preregister)}})
    path = tmp_path / "PLAN.json"
    write_json(path, plan)
    write_json(auth / "PILOT_PLAN.json", {"competence_plans": {str(path): file_hash(path)}})
    checked = []
    monkeypatch.setattr("research.src.closure_protocol.require_training_authorization", lambda p: checked.append(p))
    monkeypatch.setattr(training, "historical_b1", lambda seed: parent)
    kwargs = dict(method="B1c", seed=101, parent_checkpoint=parent, train=train, validation=validation, authorization_dir=auth)
    assert training.inspect_plan(path, **kwargs)["status"] == "passed"
    assert checked == [auth]
    with pytest.raises(ValueError, match="data mismatch"):
        training.inspect_plan(path, **{**kwargs, "train": [{"root_id": "changed", "split": "train"}]})
    source.write_text("modified after freeze")
    with pytest.raises(ValueError, match="source changed"):
        training.inspect_plan(path, **kwargs)
    source.write_text("frozen")
    plan["budgets"]["training_seconds"] = 1801
    write_json(path, plan)
    with pytest.raises(ValueError, match="not linked"):
        training.inspect_plan(path, **kwargs)
    write_json(auth / "PILOT_PLAN.json", {"competence_plans": {str(path): file_hash(path)}})
    with pytest.raises(ValueError, match="cap exceeded"):
        training.inspect_plan(path, **kwargs)


def test_integrated_signal_uses_natural_greedy_j_and_valid_samples():
    def row(c2, fjs, invalid=0, unsupported=0):
        return {"metrics": {"C2": c2, "FJS": fjs, "invalid_outputs": invalid, "unsupported_operations": unsupported}}
    baseline = row(0, 1)
    samples = [row(1, 1, invalid=1), row(1, 1, unsupported=1), row(1, 0), row(1, 1)]
    result = training.integrated_group_comparison(baseline, samples)
    assert result["integrated_improvement"] is True
    assert result["improving_samples"] == 1
    assert result["sample_valid_improvements"] == [False, False, False, True]
    assert training.integrated_group_comparison(row(1, 1), samples)["integrated_improvement"] is False
    with pytest.raises(ValueError, match="four"):
        training.integrated_group_comparison(baseline, samples[:3])


def test_signal_diagnosis_selection_and_dataset_guards_precede_gpu(tmp_path, monkeypatch):
    rows = [{"split": "train", "root_id": f"r{i}", "latent_world_id": "w0", "realization_id": "v0", "scenario_family": f"F{i // 8}"}
            for i in range(40)]
    prereg = {"research_status": "exploratory", "data_sha256": {"train": training.digest(rows)},
              "signal_max_new_generations": 600, "signal_episode_ids": [[r["root_id"], "w0", "v0"] for r in rows]}
    path = tmp_path / "PREREGISTER.json"
    write_json(path, prereg)
    selection = tmp_path / "COMPETENCE_TRAINING.json"
    write_json(selection, {"method": "B3", "seed": 101, "selected_checkpoint": str(tmp_path / "checkpoint"), "status": "executed"})
    monkeypatch.setattr("research.src.closure_protocol.require_training_authorization", lambda _: True)
    loaded = []
    monkeypatch.setattr(training.SpecLocalPolicy, "from_checkpoint", lambda *a, **kw: loaded.append(True))
    kwargs = dict(authorization_dir=tmp_path, preregistration_path=path, selection_evidence_path=selection, seed=101)
    with pytest.raises(ValueError, match="selected B1c"):
        training.collect_signal_diagnostic(tmp_path / "checkpoint", rows, tmp_path / "signal", **kwargs)
    prereg["data_sha256"]["train"] = "f" * 64
    write_json(path, prereg)
    with pytest.raises(ValueError, match="preregistration/data/budget"):
        training.collect_signal_diagnostic(tmp_path / "checkpoint", rows, tmp_path / "signal", **kwargs)
    assert loaded == [] and not (tmp_path / "signal").exists()
