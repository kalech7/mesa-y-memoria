"""Independent integration checks for opt-in training/environment imperfections."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.src.data.build_bank import build_bank
from research.src.env.episode import EpisodeEnvironment
from research.src.policy.llm_controller import LLMController, validate_action


def _train_episodes():
    rows = deepcopy(build_bank("dev_smoke", roots=10, seed=82184))
    for episode in rows:
        episode["split"] = "train"
    return rows


def _public_targets(rows):
    return [{key: row[key] for key in ("phase", "action", "completion", "observation", "route")} for row in rows]


def test_qa_imperfect_targets_depend_on_observation_and_not_hidden_labels(tmp_path):
    from research.imperfect_sft import prepare_teacher_examples

    episodes = _train_episodes()
    changed = deepcopy(episodes)
    for episode in changed:
        episode["truth"] = {"secret": "QA_HIDDEN_TRUTH_MUST_NOT_ENTER_TARGET"}
        episode["opportunity"] = {"secret": "QA_PRIVATE_ANNOTATION"}
        episode["future_query"] = {"secret": "QA_UNSEEN_FUTURE_QUERY"}
    first, _ = prepare_teacher_examples(episodes, tmp_path / "first", rate=.3, seed=912)
    second, _ = prepare_teacher_examples(changed, tmp_path / "second", rate=.3, seed=912)
    assert _public_targets(first) == _public_targets(second)
    assert "QA_HIDDEN" not in json.dumps(first)
    assert "QA_PRIVATE" not in json.dumps(first)
    assert "QA_UNSEEN" not in json.dumps(first)
    for row in first:
        assert validate_action(row["phase"], row["action"]) == row["action"]
        assert json.loads(row["completion"]) == row["action"]
        LLMController(None).prompt(row["phase"], row["observation"])


def test_qa_clean_imperfect_preparation_has_existing_teacher_targets(tmp_path):
    from research.imperfect_sft import prepare_teacher_examples
    from research.competence_training import prepare_teacher_examples as clean_teacher

    episodes = _train_episodes()
    clean = clean_teacher(episodes, tmp_path / "legacy")
    actual, _ = prepare_teacher_examples(episodes, tmp_path / "opt_in", rate=0, seed=912)
    assert _public_targets(actual) == _public_targets(clean)


@pytest.mark.parametrize("split", ["validation", "evaluation", "test_id", "test_ood"])
def test_qa_no_imperfect_targets_can_be_built_from_heldout_partitions(tmp_path, split):
    from research.imperfect_sft import prepare_teacher_examples

    episodes = _train_episodes()
    episodes[0]["split"] = split
    with pytest.raises(ValueError):
        prepare_teacher_examples(episodes, tmp_path / split, rate=.3, seed=912)


def test_qa_retrieval_loss_is_visible_only_and_does_not_break_interventions(tmp_path):
    from research.src.env.imperfections import ImperfectionConfig

    episode = deepcopy(build_bank("dev_smoke", roots=5, seed=82185)[0])
    original = deepcopy(episode)
    env = EpisodeEnvironment(episode, tmp_path / "memory.sqlite",
                             imperfections=ImperfectionConfig(memory_omission_rate=1., seed=912))
    try:
        before_hash = env.store.hash()
        assert env.observe("select")["memory"] == []
        env.ask("TEMPORAL_SCOPE")
        assert env.observe("respond", intervention="I1") == env.observe("respond", intervention="I2")
        assert env.observe("write", intervention="I0") == env.observe("write", intervention="I1")
        for phase in ("select", "respond", "write"):
            observation = env.observe(phase)
            assert observation["memory"] == []
            assert set(observation) == {"memory", "query", "user_messages", "reuse_regime"}
            LLMController(None).prompt(phase, observation)
        assert env.store.hash() == before_hash
        assert env.store.list_records() == original["initial_memory"]
        env.close_session()
        reuse = env.reopen_for_reuse()
        assert reuse["memory"] == original["initial_memory"]
        assert set(reuse) == {"memory", "query"}
        assert episode == original
    finally:
        env.dispose()


def test_qa_no_faults_retain_exact_existing_public_observations(tmp_path):
    from research.src.env.imperfections import ImperfectionConfig

    episode = build_bank("dev_smoke", roots=5, seed=82186)[0]
    baseline = EpisodeEnvironment(episode, tmp_path / "baseline.sqlite")
    configured = EpisodeEnvironment(episode, tmp_path / "configured.sqlite", imperfections=ImperfectionConfig())
    try:
        for phase in ("select", "respond", "write"):
            assert baseline.observe(phase) == configured.observe(phase)
        assert baseline.ask("TEMPORAL_SCOPE") == configured.ask("TEMPORAL_SCOPE")
        for phase in ("respond", "write"):
            assert baseline.observe(phase) == configured.observe(phase)
    finally:
        baseline.dispose()
        configured.dispose()


def test_qa_batched_replay_uses_same_faults_as_direct_execution(monkeypatch):
    from research.src.evaluation import batched
    from research.src.evaluation.runner import run_episode
    from research.src.policy.rule_controller import RuleController

    episodes = build_bank("dev_smoke", roots=5, seed=82188)
    config = {"memory_omission_rate": .4, "partial_reply_rate": .2,
              "unavailable_reply_rate": .3, "seed": 913}

    def generate(_backend, phase, observations, **_options):
        controller = RuleController()
        return [{"phase": phase, "observation": deepcopy(observation),
                 "prompt": LLMController(None).prompt(phase, observation),
                 "parsed_action": getattr(controller, phase)(deepcopy(observation)),
                 "technical_status": "completed"} for observation in observations]

    monkeypatch.setattr(batched, "generate_batch", generate)
    backend = SimpleNamespace(config=SimpleNamespace())
    rows = list(batched.evaluate_batched(episodes, backend, method="B1", batch_size=3,
                                         imperfections=config))
    for episode, row in zip(episodes, rows):
        direct = run_episode(episode, RuleController(), method="B1", imperfections=config)
        assert row["metrics"] == direct["metrics"]
        assert row["parsed_actions"] == direct["parsed_actions"]
        assert row["trace"]["observations"] == direct["trace"]["observations"]
        assert row["trace"]["imperfections"] == direct["trace"]["imperfections"]


def test_qa_robustness_report_accounts_for_clean_memory_and_critical_sources(tmp_path):
    from research.imperfection_experiment import evaluate_profiles
    from research.src.policy.rule_controller import RuleController

    episodes = build_bank("dev_smoke", roots=5, seed=82189)
    report = evaluate_profiles(episodes, tmp_path / "report", profiles={"clean": {}},
                               policy_factories={"B4": RuleController})
    clean = report["results"]["clean"]["B4"]
    assert clean["select_memory_records"]["offered"] == sum(len(e["initial_memory"]) for e in episodes)
    assert clean["select_memory_records"]["omitted"] == 0
    paths = set(report["source_sha256"])
    from research import imperfection_experiment
    root = Path(imperfection_experiment.__file__).parent
    assert str((root / "src/policy/rule_controller.py").resolve()) in paths
    assert str((root / "src/memory/executor.py").resolve()) in paths
    assert str((root / "src/readers/symbolic.py").resolve()) in paths


def test_qa_tampered_training_bundle_rejected_before_model_factory(tmp_path, monkeypatch):
    from research import competence_training
    from research.imperfect_sft import prepare_bundle
    from research.src.artifacts import file_hash

    train = _train_episodes()
    bundle = tmp_path / "bundle"
    manifest = prepare_bundle(train, bundle, rate=.3, seed=914)
    specification = {"bundle_path": str(bundle), "manifest_sha256": file_hash(bundle / "manifest.json"),
                     "rate": manifest["rate"], "seed": manifest["seed"]}
    monkeypatch.setattr(competence_training, "inspect_plan", lambda *a, **k:
                        {"plan": {"imperfect_sft": specification}})
    with (bundle / "examples.jsonl").open("a") as stream:
        stream.write("{}\n")
    loaded = []
    with pytest.raises(ValueError, match="hash mismatch"):
        competence_training.run_training("B1c", tmp_path / "parent", train, [{"split": "validation"}],
            tmp_path / "training", seed=101, authorization_dir=tmp_path / "authorization",
            plan_path=tmp_path / "plan.json", policy_factory=lambda *a, **k: loaded.append(True))
    assert loaded == []
