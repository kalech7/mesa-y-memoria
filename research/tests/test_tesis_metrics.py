from copy import deepcopy
import json

import pytest

from research.competence_evaluation import _run_trajectory
from research.src.artifacts import file_hash
from research.src.policy.rule_controller import RuleController
from research.tesis_metrics import Rate, compare_trajectories, episode_rates, main
from research.tests.test_competence_evaluation import _episode


class NeverAsk(RuleController):
    def select(self, observation):
        return {"action": "PROCEED", "slot": None}

    def write(self, observation):
        return {"operations": [{"op": "NOOP"}]}


def _scoped(world):
    episode = _episode(ambiguous=True, world=world)
    episode["truth"].update(decisive_slot="TEMPORAL_SCOPE", scope_attribute="dinner_preference",
                            new_value="sopa", initial_value="pasta",
                            scope={"kind": world, "valid_from": "2026-01-21", "valid_until_exclusive": None,
                                   "session_id": "S2" if world == "session" else None, "context": None})
    return episode


def test_ask_can_help_future_alone_and_missing_scope_is_not_a_writer_failure(tmp_path):
    episode = _scoped("ongoing")
    teacher = episode_rates(_run_trajectory(RuleController(), episode, tmp_path / "asks.db"))
    bypass = episode_rates(_run_trajectory(NeverAsk(), episode, tmp_path / "skips.db"))
    assert teacher["informative_now"] == Rate(0, 1)
    assert teacher["informative_later"] == Rate(1, 1)
    assert teacher["strict_joint"] == Rate(1, 1)
    assert teacher["missed_authorized_permanent_write"] == Rate(0, 1)
    assert bypass["missed_ask"] == Rate(1, 1)
    assert bypass["informative_ask"].value is None
    assert bypass["permanent_intent_unmet"] == Rate(1, 1)
    assert bypass["missed_authorized_permanent_write"] == Rate(0, 0)


def test_temporary_as_permanent_damages_outside_its_session(tmp_path):
    class PromotesTemporary(RuleController):
        def write(self, observation):
            operation = super().write(observation)["operations"][0]
            operation["op"] = "SUPERSEDE"
            operation["target_ids"] = ["m0"]
            operation["record"]["scope"].update(kind="ongoing", session_id=None)
            return {"operations": [operation]}

    episode = _scoped("session")
    bad = episode_rates(_run_trajectory(PromotesTemporary(), episode, tmp_path / "bad.db"))
    good = episode_rates(_run_trajectory(RuleController(), episode, tmp_path / "good.db"))
    assert bad["temporary_stored_without_expiry"] == Rate(1, 1)
    assert bad["unaffected_memory_damage"] == Rate(1, 1)
    assert good["temporary_stored_without_expiry"] == Rate(0, 1)
    assert good["unaffected_memory_damage"] == Rate(0, 1)
    assert good["unaffected_probe_damage"].denominator == bad["unaffected_probe_damage"].denominator
    assert bad["unsupported_write"] == Rate(1, 1)


def test_matched_contrasts_weight_roots_and_do_not_count_seeds_as_users(tmp_path):
    episodes = [_episode(), _scoped("ongoing"), _scoped("session")]
    episodes[0]["truth"]["decisive_slot"] = None
    panels = {"reference": [], "candidate": []}
    for label, controller in (("reference", NeverAsk()), ("candidate", RuleController())):
        for i, episode in enumerate(episodes):
            row = _run_trajectory(controller, episode, tmp_path / f"{label}_{i}.db")
            panels[label].extend({**row, "training_seed": seed} for seed in (101, 202, 303))
    result = compare_trajectories(panels, reference="reference")
    assert result["scope"]["independent_roots"] == 2
    assert result["scope"]["scheduled_paired_cells"] == 9
    assert result["scope"]["training_seeds"] == [101, 202, 303]
    assert result["methods"]["candidate"]["missed_ask"]["F4"] == {
        "macro": 0, "eligible_roots": 1, "eligible_episodes": 6, "numerator": 0, "denominator": 6}
    contrast = result["paired_vs_reference"]["candidate"]
    assert contrast["missed_ask"]["macro_difference"] == -1
    assert contrast["missed_ask"]["paired_eligible_roots"] == 1
    assert contrast["informative_ask"]["macro_difference"] is None
    assert contrast["informative_ask"]["only_method_eligible_episodes"] == 6
    missing = {**panels, "candidate": panels["candidate"][1:]}
    with pytest.raises(ValueError, match="Unpaired episodes"):
        compare_trajectories(missing, reference="reference")
    changed = deepcopy(panels)
    changed["candidate"][0]["trace"]["episode"]["future_query"]["date"] = "2026-01-31"
    with pytest.raises(ValueError, match="Mismatched episode"):
        compare_trajectories(changed, reference="reference")


def test_explicit_technical_failure_excludes_pair_not_the_other_method_only(tmp_path):
    episode = _episode()
    row = _run_trajectory(RuleController(), episode, tmp_path / "finished.db")
    failed = {"status": "technical_failure", "episode": episode, "condition": "O2", "route": "natural",
              "reader_id": "fixed_symbolic", "training_seed": 101}
    result = compare_trajectories({"ok": [row], "failed": [failed]}, reference="ok")
    assert result["scope"]["complete_paired_cells"] == 0
    assert result["technical_exclusions"] == {"by_method": {"ok": 0, "failed": 1}, "paired_cells_excluded": 1}
    assert result["methods"]["ok"]["current_supported"]["all"]["macro"] is None


def test_comparison_artifact_checks_hashes_and_allows_step_zero_alias(tmp_path):
    row = _run_trajectory(RuleController(), _episode(), tmp_path / "run.db")
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(row) + "\n")
    result = tmp_path / "RESULTS.json"
    panel = {"trajectories": path.name, "trajectories_sha256": file_hash(path)}
    result.write_text(json.dumps({"status": "completed_exploratory", "seeds": [101],
                                  "panels": {"101": {"SFT": panel, "GRPO": {**panel, "same_policy_as": "SFT"}}}}))
    out = tmp_path / "metrics.json"
    args = ["--comparison-results", str(result), "--reference", "SFT", "--output"]
    main([*args, str(out)])
    report = json.loads(out.read_text())
    assert report["methods"]["SFT"] == report["methods"]["GRPO"]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Trajectory hash mismatch"):
        main([*args, str(tmp_path / "bad.json")])
