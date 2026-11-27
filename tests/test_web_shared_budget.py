import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch

from web_demo.shared_budget import BudgetError, BudgetLab


class SharedBudgetWebTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = Path(self.directory.name) / "run"
        self.run.mkdir()
        self.lab = BudgetLab("cpu", self.run)
        self.case = "busy_contingent"

    def play(self, **fields):
        return self.lab.play({"case_id": self.case, **fields})

    def test_manual_reservation_preserves_the_question_across_visits(self):
        first = self.play()
        second = self.play(actions=["PROCEED"])
        self.assertEqual(first["questions_remaining"], 1)
        self.assertEqual(second["questions_remaining"], 1)
        self.assertEqual(second["observation"]["visit"], 2)
        self.assertNotIn("future", second)
        self.assertNotIn("strategy_cue", second["observation"])
        self.assertNotIn("expected", second["visits"][0])
        late = self.play(actions=["PROCEED", "TEMPORAL_SCOPE", "PROCEED"])
        early = self.play(actions=["TEMPORAL_SCOPE", "PROCEED", "PROCEED"])
        self.assertTrue(late["done"])
        self.assertEqual(late["questions_remaining"], 0)
        self.assertGreater(late["metrics"]["future_supported"], early["metrics"]["future_supported"])
        with self.assertRaises(BudgetError):
            self.play(actions=["TEMPORAL_SCOPE", "PROCEED", "TEMPORAL_SCOPE"])

    def test_pending_run_keeps_rule_usable_and_rejects_model_inference(self):
        catalog = self.lab.catalog()
        self.assertTrue(next(item["available"] for item in catalog["policies"] if item["id"] == "rule"))
        self.assertFalse(next(item["available"] for item in catalog["policies"] if item["id"] == "grpo"))
        with self.assertRaises(BudgetError) as unavailable:
            self.play(step_with="grpo")
        self.assertEqual(unavailable.exception.status, 503)
        self.assertEqual(len(self.play(step_with="rule")["actions"]), 1)
        with self.assertRaises(BudgetError):
            self.play(compare=["rule", "rule"])

    def test_published_adapters_use_real_registry_paths_and_selection(self):
        parent = self.run.parent
        base = parent / "base"
        base.mkdir()
        for name in ("config.json", "tokenizer.json", "model.safetensors"):
            (base / name).touch()
        adapter = self.run / "selected"
        adapter.mkdir()
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            (adapter / name).touch()
        manifest = {
            "status": "completed", "base_model": "base",
            "models": {"sft": {"adapter": "selected", "adapter_sha256": "a" * 64}},
            "selection": {"sft": {"epoch": 0}},
        }
        path = self.run / "LAB_MODELS.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with patch("web_demo.shared_budget.ROOT", parent), patch("web_demo.shared_budget.find_spec", return_value=True):
            registry = {item["id"]: item for item in self.lab.catalog()["policies"]}
            self.assertTrue(registry["sft"]["available"])
            self.assertEqual(registry["sft"]["epoch"], 0)
            self.assertFalse(registry["grpo"]["available"])
            with patch.object(self.lab, "_infer", return_value="PROCEED") as inference:
                results = self.play(compare=["rule", "sft"])["comparisons"]
                self.assertEqual([item["id"] for item in results], ["rule", "sft"])
                self.assertEqual(results[1]["epoch"], 0)
                self.assertEqual(inference.call_args.args[-1]["sft"][0], adapter)
                self.assertNotIn("enhanced", inference.call_args.args[1])
                module = ModuleType("research.tesis_shared_budget_qwen_context")
                module.with_public_context = lambda observation: {**observation, "enhanced": True}
                manifest["prompt_format"] = "public-context-v2"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with patch.dict(sys.modules, {module.__name__: module}):
                    self.play(step_with="sft")
                self.assertTrue(inference.call_args.args[1]["enhanced"])
            manifest["prompt_format"] = "unknown"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(BudgetError, "Formato de entrada desconocido"):
                self.play(step_with="sft")


if __name__ == "__main__":
    unittest.main()
