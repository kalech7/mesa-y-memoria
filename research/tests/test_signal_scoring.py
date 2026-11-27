from copy import deepcopy

from research.signal_scoring import DiagnosticEvaluator
from research.src.closure_environment import _episode, _record
from research.src.evaluation.runner import run_episode
from research.src.policy.rule_controller import RuleController
from research.src.scoring.evidence import Evaluator, LegacyEvaluator
from research.src.scoring.reward import rewards


def trace():
    return run_episode(_episode(), RuleController(), method="C-NEVER")["trace"]


def test_new_persisted_id_cites_faithful_conditional_neutral_not_wrong():
    example = trace()
    assert example["future_response"]["mode"] == "CONDITIONALIZE"
    assert LegacyEvaluator().score_future(example)["utility"] == -1
    assert Evaluator().score_future(example)["utility"] == 0
    corrected = DiagnosticEvaluator().score_future(example)
    assert corrected["utility"] == 0
    assert corrected["success"] == 0


def test_conditional_neutrality_independent_of_hidden_compatible_world():
    example = trace()
    one = DiagnosticEvaluator().score_future(example)
    example["episode"]["truth"]["future_value"] = "sopa"
    assert DiagnosticEvaluator().score_future(example)["utility"] == one["utility"] == 0


def test_forged_memory_citation_stays_penalized():
    example = trace()
    example["future_response"]["evidence_ids"].append("fabricated")
    assert DiagnosticEvaluator().score_future(example)["utility"] == -1


def test_forged_source_quote_stays_penalized():
    example = trace()
    example["final_memory"][-1]["evidence_quotes"][0]["text"] = "invented quotation"
    assert DiagnosticEvaluator().score_future(example)["utility"] == -1


def test_invented_conditional_branch_stays_penalized():
    example = trace()
    example["future_response"]["branches"].append({"condition": "invento", "value": "invented_value"})
    assert DiagnosticEvaluator().score_future(example)["utility"] == -1


def test_irrelevant_stored_citation_does_not_support_conditional():
    example = trace()
    example["final_memory"].append(_record("drink", "agua", attribute="drink_preference"))
    example["future_response"]["evidence_ids"] = ["drink"]
    assert DiagnosticEvaluator().score_future(example)["utility"] == -1


def test_answer_and_defer_contracts_unchanged():
    example = run_episode(_episode(), RuleController())["trace"]
    assert DiagnosticEvaluator().score_future(example) == Evaluator().score_future(example)
    example["future_response"] = {"mode": "DEFER", "value": None, "branches": [], "evidence_ids": []}
    assert DiagnosticEvaluator().score_future(example) == Evaluator().score_future(example)


def test_future_scoring_correction_does_not_change_immediate_reward():
    example = trace()
    current = Evaluator().score_current(example)["utility"]
    old = rewards(current, LegacyEvaluator().score_future(example)["utility"])
    new = rewards(current, DiagnosticEvaluator().score_future(example)["utility"])
    assert new["R_long"] == old["R_long"] + 1
    assert new["R_short"] == old["R_short"]
