from copy import deepcopy

from research.src.data.build_bank import build_bank
from research.src.policy.rule_controller import RuleController
from research.src.evaluation.runner import run_episode
from research.src.scoring.diagnostics import episode_diagnostics


def test_forced_ask_uses_visible_slot_or_generic():
    policy = RuleController()
    assert policy.select_forced_ask({"memory": [], "user_messages": [], "query": {}}) == {"action": "ASK", "slot": "GENERIC"}
    episode = next(e for e in build_bank() if e["opportunity"]["eligible"])
    observation = {"memory": episode["initial_memory"], "user_messages": episode["user_messages"], "query": episode["current_query"]}
    assert policy.select_forced_ask(observation)["slot"] == "TEMPORAL_SCOPE"


def test_future_only_acquisition_and_coverage_have_actual_denominators():
    episode = next(e for e in build_bank() if e["opportunity"]["eligible"])
    row = run_episode(episode, RuleController())
    diagnostic = episode_diagnostics(row["trace"], row["metrics"])
    assert diagnostic["acquisition_questions_denominator"] == 1
    assert diagnostic["useful_acquisition_current_numerator"] == 0
    assert diagnostic["useful_acquisition_future_numerator"] == 1
    assert diagnostic["memory_coverage_at_reuse"] == 1
    assert diagnostic["memory_coverage"] is None
    assert diagnostic["scope_probe_wrong_commitments"] == 0
    assert diagnostic["scope_error_rate"] is None


def test_removed_memory_loses_coverage_despite_private_log():
    episode = next(e for e in build_bank() if e["opportunity"]["eligible"] and e["truth"]["scope_kind"] == "ongoing")
    row = run_episode(episode, RuleController())
    trace = deepcopy(row["trace"])
    trace["final_memory"] = [r for r in trace["final_memory"] if r["attribute"] != "dinner_preference"]
    diagnostic = episode_diagnostics(trace, row["metrics"])
    assert diagnostic["memory_coverage_at_reuse_numerator"] == 0
    assert diagnostic["memory_coverage_at_reuse_denominator"] == 1


def test_incomplete_trace_never_fabricates_metrics():
    assert episode_diagnostics({}, {})["diagnostics_status"] == "not_identifiable"
