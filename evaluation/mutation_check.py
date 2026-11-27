"""Ocho fallos deliberados, sin modificar fuentes; cada uno debe romper una prueba."""
import argparse
import contextlib
import io
import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from agents import policies
from memory.store import Store
from simulator.user import UserSimulator, Reply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original_respond = UserSimulator.respond
    original_messages = policies.build_messages
    original_add = Store.add

    def irrelevant_leak(self, question, context):
        return Reply(self._state["preference"], "relevant", 1.0, True, "preference")

    def leaked_messages(*values, **kwargs):
        result = original_messages(*values, **kwargs)
        result.append({"role": "user", "content": "GOLD_SENTINEL_DO_NOT_SEND"})
        return result

    def omit_clarification(self, message, *values, **kwargs):
        if message["source"] == "clarification":
            return message["message_id"]
        return original_add(self, message, *values, **kwargs)

    def ignore_duplicate(self, question, context):
        return replace(original_respond(self, question, context), duplicate=False, resolved=True, information_gain=1.0)

    mutants = [
        ("irrelevant_question_unlocks_truth", "simulator.user.UserSimulator.respond", irrelevant_leak, "tests.test_simulator.SimulatorTests.test_irrelevant_question_does_not_reveal_preference"),
        ("hidden_answer_in_prompt", "agents.policies.build_messages", leaked_messages, "tests.test_api_policy.APIAndPolicyTests.test_hidden_answer_canary_never_enters_prompt"),
        ("scope_difference_treated_as_conflict", "simulator.user.interpret_question", lambda *a: "preference", "tests.test_simulator.SimulatorTests.test_scope_mismatch_does_not_unlock_hidden_state"),
        ("clarification_not_saved", "memory.store.Store.add", omit_clarification, "tests.test_memory.MemoryTests.test_saved_clarification_survives_reopening"),
        ("negation_erased_without_alarm", "memory.extraction.negation_loss", lambda *a: False, "tests.test_memory.MemoryTests.test_extraction_that_erases_negation_is_detected"),
        ("one_retrieved_side_counted_as_pair", "memory.retrieval.pair_recall", lambda *a: True, "tests.test_memory.MemoryTests.test_bge_missing_one_side_is_detected"),
        ("duplicate_question_rewarded", "simulator.user.UserSimulator.respond", ignore_duplicate, "tests.test_simulator.SimulatorTests.test_duplicate_ask_is_penalized"),
        ("contradictory_final_answer_passes", "evaluation.scoring.conservative_fidelity", lambda *a: "pass", "tests.test_evaluation.EvaluationTests.test_response_violating_clarification_is_not_pass"),
    ]
    report = []
    for name, target, replacement, test in mutants:
        with patch(target, replacement), contextlib.redirect_stdout(io.StringIO()):
            stream = io.StringIO()
            result = unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromName(test))
        report.append({"mutation": name, "test": test, "killed": bool(result.failures) and not result.errors,
                       "failures": len(result.failures), "errors": len(result.errors)})
    with args.output.open("x", encoding="utf8") as stream:
        json.dump({"mutations": report, "all_killed": all(r["killed"] for r in report)}, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"mutations": len(report), "killed": sum(r["killed"] for r in report)}, ensure_ascii=False))
    return 0 if all(r["killed"] for r in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
