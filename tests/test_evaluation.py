import copy
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from agents.client import API, TechnicalError
from agents.policies import LMPolicy, decision
from evaluation import scoring
from evaluation.metrics import summarize, grouped_bootstrap
from evaluation.runner import run_case, main, export_reports, load_jsonl
from tests.helpers import OfflineTest, completion


class EvaluationTests(OfflineTest):
    def test_response_violating_clarification_is_not_pass(self):
        _, gold = self.case()
        state = {**gold["hidden_state"], "preference": "sin tofu"}
        self.assertEqual(scoring.conservative_fidelity("Te recomiendo con tofu", state), "fail")
        self.assertEqual(scoring.conservative_fidelity("Te recomiendo sin tofu", state), "pass")
        self.assertEqual(scoring.conservative_fidelity("Si prefieres sin tofu o con tofu, decide", state), "uncertain")

    def test_fake_api_full_pipeline_and_future_uses_saved_memory(self):
        case, gold = copy.deepcopy(self.case())
        gold["hidden_state"]["preference"] = "sin tofu"
        responses = [completion(decision("ASK", question="¿Prefieres tofu actualmente?", conflict=True, unresolved=True)),
                     completion(decision("ANSWER", answer="Te recomiendo sin tofu")),
                     completion(decision("ANSWER", answer="Te recomiendo sin tofu"))]
        with TemporaryDirectory() as temporary, patch("agents.client.http_json", side_effect=responses) as transport:
            row = run_case(case, gold, LMPolicy(API("private-placeholder", "fake")), "fake", Path(temporary) / "case")
            self.assertFalse(row["technical_errors"])
            self.assertTrue(row["clarification_resolved"])
            self.assertTrue(row["confirmation_memory_saved"])
            self.assertTrue(row["followup_confirmation_retrieved"])
            self.assertEqual(row["followup_fidelity"], "pass")
            self.assertFalse(row["end_to_end_success"])  # Éxito de tarea sin juzgar -> no inventarlo.
            requests = [call.args[1]["messages"] for call in transport.call_args_list]
            self.assertNotIn("de ahora en adelante prefiero sin tofu", json.dumps(requests[0], ensure_ascii=False).lower())
            self.assertEqual(requests[-1][-1]["content"], case["followup"])
            self.assertNotIn("Te recomiendo sin tofu", json.dumps(requests[-1], ensure_ascii=False))
            self.assertTrue(any("fuente clarification" in m["content"] for m in requests[-1]))

    def test_no_storage_ablation_does_not_magic_memory_into_future(self):
        case, gold = self.case()
        class Scripted:
            def __init__(self):
                self.calls = []
            def decide(self, query, records, reference_time, recent=None):
                self.calls.append((query, records, recent))
                if len(self.calls) == 1:
                    return decision("ASK", question="¿Prefieres tofu actualmente?", conflict=True, unresolved=True), []
                return decision("ANSWER", answer="No puedo demostrar la preferencia sin evidencia"), []
        policy = Scripted()
        with TemporaryDirectory() as temporary:
            row = run_case(case, gold, policy, "ablation", Path(temporary) / "case", persist_clarification=False)
        self.assertTrue(row["clarification_resolved"])
        self.assertFalse(row["confirmation_memory_saved"])
        self.assertFalse(row["followup_confirmation_retrieved"])
        self.assertIsNone(policy.calls[-1][2])
        self.assertFalse(any(m["source"] == "clarification" for m in policy.calls[-1][1]))

    def test_occasion_only_clarification_cannot_earn_future_resolution_credit(self):
        case, gold = self.case()
        class Scripted:
            def __init__(self):
                self.calls = 0
            def decide(self, *args):
                self.calls += 1
                if self.calls == 1:
                    return decision("ASK", question="¿Quieres incluir tofu esta vez?"), []
                return decision("ANSWER", answer="Te recomiendo sin tofu"), []
        with TemporaryDirectory() as temporary:
            row = run_case(case, gold, Scripted(), "occasion", Path(temporary) / "case")
        self.assertTrue(row["confirmation_memory_saved"])
        self.assertFalse(row["general_preference_confirmed"])
        self.assertNotIn("followup_fidelity", row)
        self.assertFalse(row["end_to_end_success"])

    def test_second_question_allowed_without_rewarding_irrelevant_first(self):
        case, gold = self.case()
        class Scripted:
            def __init__(self):
                self.count = 0
            def decide(self, *args):
                self.count += 1
                if self.count == 1:
                    return decision("ASK", question="¿Qué presupuesto tienes para tofu?"), []
                if self.count == 2:
                    return decision("ASK", question="¿Prefieres tofu actualmente?"), []
                return decision("ANSWER", answer="Ofrezco una propuesta"), []
        with TemporaryDirectory() as temporary:
            row = run_case(case, gold, Scripted(), "fake", Path(temporary) / "case")
        self.assertEqual(len(row["questions"]), 2)
        self.assertFalse(row["questions"][0]["resolved"])
        self.assertTrue(row["questions"][1]["resolved"])

    def test_technical_failure_does_not_get_wrong_action_label(self):
        case, gold = self.case()
        class Broken:
            def decide(self, *args):
                raise TechnicalError("empty_content")
        with TemporaryDirectory() as temporary:
            row = run_case(case, gold, Broken(), "fake", Path(temporary) / "case")
        self.assertNotIn("action_correct", row)
        self.assertEqual(row["technical_errors"][0]["kind"], "empty_content")
        metrics = summarize([row])["conditions"]["fake"]
        self.assertIsNone(metrics["action_accuracy"]["value"])
        self.assertEqual(metrics["initial_not_evaluable"], 1)

    def test_denominators_keep_missing_future_and_failed_calls(self):
        case, gold = self.case()
        class Ask:
            def decide(self, *args):
                return decision("ASK", question="¿Prefieres tofu actualmente?"), []
        with TemporaryDirectory() as temporary:
            row = run_case(case, gold, Ask(), "fake", Path(temporary) / "case")
        metrics = summarize([row])["conditions"]["fake"]
        self.assertEqual(metrics["followup_memory_consistency"]["denominator"], 1)
        self.assertEqual(metrics["followup_uncertain_or_missing"], 1)
        self.assertEqual(metrics["unnecessary_ask_after_clarification"]["value"], 1)

    def test_bootstrap_clusters_people_not_rows(self):
        rows = [{"person_id": "one", "correct": True}] * 100
        self.assertIsNone(grouped_bootstrap(rows, "correct")["low"])
        result = grouped_bootstrap(rows + [{"person_id": "two", "correct": False}], "correct", repeats=100)
        self.assertEqual(result["clusters"], 2)

    def test_cli_dry_run_is_offline_and_refuses_overwrite(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "run"
            argv = ["--dry-run", "--output", str(path), "--dev-people", "2", "--test-people", "3"]
            self.assertEqual(main(argv), 0)
            with self.assertRaises(SystemExit):
                main(argv)
            self.assertEqual(main(argv + ["--resume"]), 0)
            with self.assertRaises(SystemExit):
                main(argv + ["--resume", "--seed", "99"])

    def test_cli_heuristic_resume_is_idempotent(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "run"
            argv = ["--output", str(path), "--dev-people", "2", "--test-people", "3", "--limit", "2", "--policies", "simple_conflict", "--two-turn"]
            self.assertEqual(main(argv), 0)
            before = (path / "per_case.jsonl").read_bytes()
            self.assertEqual(main(argv + ["--resume"]), 0)
            self.assertEqual(before, (path / "per_case.jsonl").read_bytes())
            self.assertEqual(len(load_jsonl(path / "per_case.jsonl")), 2)

    def test_human_blind_packet_contains_evidence_without_policy_or_auto_scores(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "run"
            main(["--output", str(path), "--dev-people", "2", "--test-people", "3", "--limit", "1", "--policies", "simple_conflict", "--two-turn"])
            with (path / "human_review_blind.csv").open(newline="", encoding="utf8") as stream:
                review = next(csv.DictReader(stream))
            self.assertNotIn("condition", review)
            self.assertNotIn("variant", review)
            self.assertNotIn("resolved", review["dialogue"])
            self.assertNotIn("information_gain", review["dialogue"])
            self.assertTrue(review["history"])
            review["notes"] = "Anotación que no se debe perder"
            with (path / "human_review_blind.csv").open("w", newline="", encoding="utf8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(review))
                writer.writeheader()
                writer.writerow(review)
            rows = load_jsonl(path / "per_case.jsonl")
            public = load_jsonl(path / "dataset" / "casos_agent.jsonl")
            export_reports(path, rows, {c["case_id"]: c for c in public}, 42)
            self.assertIn(review["notes"], (path / "human_review_blind.csv").read_text())

    def test_live_gate_stops_before_inference_on_unreviewed_data(self):
        with TemporaryDirectory() as temporary, patch("evaluation.runner.subprocess.run") as process:
            process.return_value.returncode = 0
            process.return_value.stderr = "Ran 50 tests"
            self.assertEqual(main(["--output", str(Path(temporary) / "run"), "--dev-people", "2", "--test-people", "3", "--policies", "base", "--url", "private-placeholder"]), 2)
            self.assertFalse((Path(temporary) / "run").exists())

    def test_technical_pilot_is_restricted_to_dev_and_at_most_ten_cases(self):
        with TemporaryDirectory() as temporary:
            for extra in (["--limit", "0"], ["--limit", "11"], ["--split", "test"]):
                with self.assertRaises(SystemExit):
                    main(["--technical-pilot", "--dry-run", "--output", str(Path(temporary) / "run")] + extra)

    def test_technical_flag_cannot_bypass_human_validation(self):
        with TemporaryDirectory() as temporary, patch("evaluation.runner.subprocess.run") as process:
            process.return_value.returncode = 0
            process.return_value.stderr = "Ran 50 tests"
            destination = Path(temporary) / "run"
            self.assertEqual(main(["--technical-pilot", "--limit", "5", "--output", str(destination),
                "--dev-people", "2", "--test-people", "3", "--policies", "guided", "--url", "private-placeholder"]), 2)
            self.assertFalse(destination.exists())
