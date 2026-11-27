"""Independent orchestration QA using real bundles and injected training jobs."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from research import imperfection_experiment as experiment
from research.src.artifacts import file_hash, write_json


def test_prepare_creates_three_auditable_rates_with_unchanged_heldout(tmp_path, monkeypatch):
    from research import competence_data
    from research.imperfect_sft import load_examples

    sizes = {"train": 5, "validation": 5, "evaluation": 5}
    candidates = competence_data.build_candidates(seed=97621, sizes=sizes)
    splits = competence_data.partition_candidates(candidates, sizes, seed=97622)
    data = tmp_path / "data"
    data.mkdir()
    for split, rows in splits.items():
        (data / f"{split}.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    hashes = {split: file_hash(data / f"{split}.jsonl") for split in splits}
    before = deepcopy(splits)
    monkeypatch.setattr(competence_data, "load_and_check", lambda _: (deepcopy(splits), {}))
    result = experiment.prepare(data, tmp_path / "experiment", seed=97623)
    assert [arm["rate"] for arm in result["arms"]] == [0., .1, .3]
    assert result["training_steps"] == result["model_calls"] == 0
    assert result["faults_during_training"] is None
    for arm, selected in zip(result["arms"], [0, 1, 2]):
        bundle = Path(arm["bundle_path"])
        _, audit = load_examples(bundle, train=splits["train"], expected_rate=arm["rate"], expected_seed=97623)
        assert audit["selected_root_count"] == selected
        for split in ("validation", "evaluation"):
            assert file_hash(bundle / f"{split}.jsonl") == hashes[split]
    assert splits == before
    assert hashes == {split: file_hash(data / f"{split}.jsonl") for split in splits}
    assert experiment._load_experiment(tmp_path / "experiment")[1] == result


def _fake_campaigns(tmp_path, monkeypatch):
    from research import competence_campaign as campaign
    from research import competence_training as training

    arms = [{"rate": rate, "campaign_output": str(tmp_path / f"arm{index}")}
            for index, rate in enumerate((0., .3))]
    plan = {"sha256": "frozen_experiment", "arms": arms}
    for arm in arms:
        Path(arm["campaign_output"]).mkdir()
    write_json(tmp_path / "CAMPAIGNS.json", {
        "experiment_sha256": plan["sha256"], "campaigns": [
            {"rate": a["rate"], "run": a["campaign_output"], "sha256": "frozen_campaign"} for a in arms]})
    monkeypatch.setattr(experiment, "_load_experiment", lambda _: (tmp_path, plan))
    monkeypatch.setattr(campaign, "verify_plan", lambda _: {"sha256": "frozen_campaign"})
    events = []

    def checkpoint(path):
        path = Path(path)
        path.mkdir(exist_ok=True)
        (path / "adapter").write_bytes(b"fixed weights")
        return str(path)

    def adapt(run):
        run = Path(run)
        events.append(("adapt", run.name))
        value = {"results": {"101": {"selected_checkpoint": checkpoint(run / "b1c")}}}
        write_json(run / "ADAPTATION_COMPLETION.json", value)
        return value

    def signal(run):
        events.append(("signal", Path(run).name))

    def continuations(run):
        run = Path(run)
        events.append(("continuations", run.name))
        value = {"results": {"101": {"B3": {"selected_checkpoint": checkpoint(run / "b3")}}}}
        write_json(run / "CONTINUATIONS_COMPLETION.json", value)
        return value

    def evidence(path):
        path = Path(path)
        return {"path": str(path), "weights_sha256": file_hash(path / "adapter")}

    def evaluate(run):
        assert sum(name == "continuations" for name, _ in events) == len(arms)
        assert (tmp_path / "SELECTED_ALL_ARMS.json").is_file()
        events.append(("evaluate", Path(run).name))

    monkeypatch.setattr(campaign, "adapt", adapt)
    monkeypatch.setattr(campaign, "signal", signal)
    monkeypatch.setattr(campaign, "continuations", continuations)
    monkeypatch.setattr(campaign, "evaluate", evaluate)
    monkeypatch.setattr(campaign, "report", lambda run: {"status": "completed"})
    monkeypatch.setattr(training, "checkpoint_evidence", evidence)
    return campaign, arms, events


def test_prepare_campaigns_binds_every_arm_and_does_not_start_training(tmp_path, monkeypatch):
    from research import competence_campaign as campaign

    arms = [{"rate": rate, "bundle_path": str(tmp_path / f"bundle{index}"),
             "campaign_output": str(tmp_path / f"campaign{index}")}
            for index, rate in enumerate((0., .1, .3))]
    plan = {"sha256": "frozen_experiment", "data": str(tmp_path / "data"), "arms": arms}
    monkeypatch.setattr(experiment, "_load_experiment", lambda _: (tmp_path, plan))
    calls = []

    def prepare(data, output, **options):
        calls.append((data, output, options))
        Path(output).mkdir()
        return {"sha256": "frozen_" + Path(output).name}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Preparing campaigns must not start training/evaluation")

    monkeypatch.setattr(campaign, "prepare", prepare)
    for method in ("adapt", "signal", "continuations", "evaluate"):
        monkeypatch.setattr(campaign, method, forbidden)
    result = experiment.prepare_campaigns(tmp_path, tests_report="passing.xml", technical_report="gpu.json")
    assert result["status"] == "prepared_not_trained"
    assert result["experiment_sha256"] == plan["sha256"]
    assert calls == [(plan["data"], arm["campaign_output"], {
        "tests_report": "passing.xml", "technical_report": "gpu.json",
        "imperfect_sft_bundle": arm["bundle_path"]}) for arm in arms]
    with pytest.raises(FileExistsError):
        experiment.prepare_campaigns(tmp_path, tests_report="passing.xml", technical_report="gpu.json")


def test_execute_finishes_all_selection_before_any_heldout_call(tmp_path, monkeypatch):
    _, arms, events = _fake_campaigns(tmp_path, monkeypatch)
    result = experiment.execute(tmp_path)
    assert result["status"] == "completed"
    assert events == [(phase, f"arm{index}") for index in range(len(arms))
                      for phase in ("adapt", "signal", "continuations")] + [
                          ("evaluate", f"arm{index}") for index in range(len(arms))]
    assert not (tmp_path / "execution.lock").exists()
    with pytest.raises(FileExistsError):
        experiment.execute(tmp_path)


@pytest.mark.parametrize("preopened", ["SELECTED_CHECKPOINTS.json", "EVALUATION_COMPLETION.json", "evaluation"])
def test_preopened_arm_is_rejected_before_any_training(tmp_path, monkeypatch, preopened):
    _, arms, events = _fake_campaigns(tmp_path, monkeypatch)
    marker = Path(arms[0]["campaign_output"]) / preopened
    if preopened == "evaluation":
        marker.mkdir()
    else:
        write_json(marker, {})
    with pytest.raises(ValueError, match="evaluat|opened"):
        experiment.execute(tmp_path)
    assert events == []
    assert not (tmp_path / "execution.lock").exists()


def test_selection_record_tampering_after_barrier_cannot_change_evaluated_checkpoint(tmp_path, monkeypatch):
    campaign, arms, events = _fake_campaigns(tmp_path, monkeypatch)
    original = campaign.evaluate

    def evaluate_and_tamper(run):
        original(run)
        if Path(run).name == "arm0":
            next_run = Path(arms[1]["campaign_output"])
            path = next_run / "ADAPTATION_COMPLETION.json"
            value = json.loads(path.read_text())
            value["results"]["101"]["selected_checkpoint"] = str(next_run / "b3")
            write_json(path, value)

    monkeypatch.setattr(campaign, "evaluate", evaluate_and_tamper)
    with pytest.raises(ValueError, match="[Ss]elect|[Cc]ompletion|barrier"):
        experiment.execute(tmp_path)
    assert ("evaluate", "arm1") not in events
    assert not (tmp_path / "execution.lock").exists()
    assert not (tmp_path / "COMPARISON.json").exists()


def test_robustness_compared_methods_receive_same_mask_and_original_store(tmp_path):
    from research.src.data.build_bank import build_bank
    from research.src.policy.rule_controller import RuleController

    episodes = build_bank("dev_smoke", roots=5, seed=97624)
    profiles = {"clean": {}, "missing": {"memory_omission_rate": .5}}
    result = experiment.evaluate_profiles(episodes, tmp_path / "robustness", profiles=profiles,
        policy_factories={"policy_a": RuleController, "policy_b": RuleController}, seed=97625)
    rows = [json.loads(line) for line in (tmp_path / "robustness/trajectories.jsonl").read_text().splitlines()]
    by_profile = {}
    for row in rows:
        key = (row["profile"], row["root_id"], row["latent_world_id"], row["realization_id"])
        before = by_profile.setdefault(key, row)
        assert row["trace"]["observations"]["select"] == before["trace"]["observations"]["select"]
        assert row["trace"]["initial_memory"] == before["trace"]["initial_memory"]
        assert row["trace"]["reuse_observation"]["memory"] == row["trace"]["final_memory"]
    assert result["trajectory_count"] == len(episodes) * 4
    assert result["episodes_unchanged"]
    for row in rows:
        if row["profile"] == "clean":
            assert "imperfections" not in row["trace"]
