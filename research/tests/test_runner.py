from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from research.src.data.build_bank import build_bank
from research.src.evaluation.runner import run_episode, run_attempts
from research.src.evaluation.observability import evaluate_snapshots
from research.src.evaluation.persistence_interventions import run_interventions
from research.src.policy.rule_controller import RuleController
from research.src.protocol import GateError
from research.src.artifacts import canonical


def f4():
    return next(e for e in build_bank(roots=5) if e["scenario_family"] == "F4" and e.get("opportunity", {}).get("eligible", True))


class RunnerTests(unittest.TestCase):
    def test_phases_persistence_and_json_trace(self):
        row = run_episode(f4(), RuleController())
        self.assertEqual(set(row["trace"]["observations"]), {"select", "respond", "write"})
        self.assertEqual(row["metrics"]["FJS"], 1)
        self.assertIn('"future_response"', canonical(row))
        self.assertEqual(set(row["trace"]["reuse_observation"]), {"memory", "query"})

    def test_intervention_bytes_not_only_fields(self):
        result = run_interventions(f4(), RuleController, forced_slot="TEMPORAL_SCOPE", method="mechanism_reference")
        self.assertTrue(all(result["invariants"].values()))
        self.assertEqual(result["effects"]["Delta_actual"], 0)
        self.assertEqual(result["effects"]["Delta_persistencia"], 1)

    def test_invalid_actor_is_not_retried(self):
        class Invalid(RuleController):
            def select(self, obs): return {}
            def respond(self, obs): return {}
            def write(self, obs): return {}
        row = run_attempts(f4(), Invalid)
        self.assertEqual(row["retry_count"], 0)
        self.assertEqual(row["metrics"]["invalid_outputs"], 3)
        self.assertEqual(row["metrics"]["C2"], 0)

    def test_infrastructure_attempts_preserved_and_denominated(self):
        class Broken(RuleController):
            def select(self, obs): raise OSError("fixture infrastructure failure")
        row = run_attempts(f4(), Broken)
        self.assertEqual(len(row["attempts"]), 2)
        self.assertEqual(row["technical_status"], "failed")
        self.assertEqual(row["metrics"]["FJS"], 0)

    def test_invalid_frozen_reader_is_failure_without_retry(self):
        class Reader:
            def read(self, memory, query): raise ValueError("model JSON invalid")
        row = run_attempts(f4(), RuleController, reader=Reader())
        self.assertEqual(row["retry_count"], 0)
        self.assertEqual(row["metrics"]["future_invalid"], 1)
        self.assertEqual(row["metrics"]["FJS"], 0)

    def test_e1_never_selects_or_writes(self):
        class CurrentOnly(RuleController):
            def select(self, obs): raise AssertionError("E1 must not select")
            def write(self, obs): raise AssertionError("E1 must not write")
        rows = evaluate_snapshots(build_bank(roots=5), CurrentOnly)
        self.assertEqual({r["observation_condition"] for r in rows}, {"O1", "O2"})

    def test_closed_test_cannot_be_read_via_smoke_api(self):
        episode = f4()
        episode["split"] = "test_id"
        with tempfile.TemporaryDirectory() as empty_root:
            with self.assertRaisesRegex(GateError, "TECHNICAL_LOCK_REQUIRED"):
                run_episode(episode, RuleController(), root=empty_root)

    def test_never_ask_ablation_does_not_reuse_reply(self):
        row = run_episode(f4(), RuleController(), method="C-NEVER")
        self.assertEqual(row["metrics"]["questions"], 0)
        self.assertIsNone(row["trace"]["user_reply"])
        self.assertNotIn("u3", canonical(row["trace"]["observations"]["write"]))
