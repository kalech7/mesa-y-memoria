"""SPEC-001 integration, privacy, independent fixtures and metric contracts."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from data_generation.food_scenarios import make_food_dataset
from evaluation.food_preferences import evaluate_case, score_output, rate, summarize, run_evaluation
from memory.store import Store
from simulator.food_user import SyntheticFoodUser
from simulator.preferences import Preference, Restriction
from simulator.resolution import FoodPreferenceEngine


class FoodPreferenceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.public, self.gold, self.private = make_food_dataset(42, 240)

    def test_twenty_canonical_scenarios_independent_expectations(self):
        for case, expected in zip(self.public[:20], self.gold[:20]):
            with self.subTest(category=case["category"]):
                result, preferences, restrictions = evaluate_case(case)
                self.assertEqual(score_output(case, expected, result, preferences, restrictions)["issues"], [])

    def test_clarification_reveals_only_requested_context_then_decides(self):
        for category, slot in (("S14", "day_scope"), ("S15", "meal_scope")):
            case, expected, private = next((c, g, p) for c, g, p in zip(self.public, self.gold, self.private) if c["category"] == category)
            with self.subTest(category=category), TemporaryDirectory() as directory:
                path = Path(directory) / "memory.sqlite"
                store = Store(path, case["user_id"])
                engine = FoodPreferenceEngine(store)
                engine.ingest(case["events"])
                result = engine.decide(case["scenario_id"], case["user_id"], case["query"], case["reference_time"])
                self.assertEqual(result["selected_action"], "CLARIFY")
                user = SyntheticFoodUser(private)
                reply = user.respond(result["clarification_question"])
                self.assertEqual(reply["communicated_slots"], [slot])
                self.assertNotIn("true_state", reply["text"])
                event = {**case["events"][0], "message_id": "clarification", "at": case["reference_time"],
                         "source": "clarification", "text": reply["text"]}
                engine.ingest([event])
                # User reply is now visible. Gold is not used to construct query.
                visible_query = case["query"] + " " + reply["text"]
                after = engine.decide(case["scenario_id"], case["user_id"], visible_query, case["reference_time"])
                followup = expected["clarification_followup"]
                self.assertIn(after["selected_action"], followup["acceptable_actions"])
                self.assertEqual(after["selected_food_items"], followup["expected_food_items"])
                self.assertTrue(set(followup["expected_evidence_ids"]) <= set(after["evidence_ids"]))
                snapshot = store.food_snapshot()
                store.close()
                reopened = Store(path, case["user_id"])
                self.assertEqual(snapshot, reopened.food_snapshot())
                self.assertIn("clarification", {e["event_id"] for e in reopened.all()})
                future = FoodPreferenceEngine(reopened).decide("future", case["user_id"], visible_query, "2026-10-28T12:00:00+00:00")
                self.assertEqual(future["selected_food_items"], after["selected_food_items"])
                self.assertFalse(user.respond(result["clarification_question"])["new_information"])
                reopened.close()

    def test_private_state_canary_and_gold_do_not_enter_engine(self):
        case = self.public[0]
        original = evaluate_case(case)[0]
        private = {"true_state": "SECRET_HIDDEN_CANARY", "clarification_responses": {}}
        self.assertNotIn("SECRET_HIDDEN_CANARY", json.dumps(original))
        self.assertNotIn("SECRET_HIDDEN_CANARY", SyntheticFoodUser(private).respond("¿Qué prefieres?")["text"])
        # Engine decisions cannot consult category/family/expected actions.
        changed = {**case, "category": "SECRET_HIDDEN_CANARY", "family": "FAKE", "split": "test",
                   "hidden_state": private, "expected_food_items": ["inventado"]}
        other = evaluate_case(changed)[0]
        for key in original.keys() - {"execution_time_ms"}:
            self.assertEqual(original[key], other[key])
        store = Store(":memory:", case["user_id"])
        self.addCleanup(store.close)
        with self.assertRaises(ValueError):
            FoodPreferenceEngine(store).ingest([{**case["events"][0], "context": {"hidden_state": "SECRET_HIDDEN_CANARY"}}])
        self.assertFalse(store.all())

    def test_persisted_extraction_is_immutable_and_owner_checked(self):
        case = self.public[0]
        store = Store(":memory:", case["user_id"])
        self.addCleanup(store.close)
        engine = FoodPreferenceEngine(store)
        engine.ingest(case["events"])
        pref = store.preferences()[0]
        engine.ingest(case["events"])
        self.assertEqual(len(store.preferences()), 1)
        with self.assertRaises(ValueError):
            store.add_preference(replace(pref, polarity="DISLIKE"))
        with self.assertRaises(ValueError):
            store.add_preference(replace(pref, preference_id="other", user_id="other"))
        with self.assertRaises(ValueError):
            store.add_preference(replace(pref, preference_id="new", evidence_id="absent"))

    def test_metric_uses_expectations_not_action_text(self):
        case, gold = self.public[0], self.gold[0]
        result, prefs, restrictions = evaluate_case(case)
        self.assertTrue(score_output(case, gold, result, prefs, restrictions)["decision_correct"])
        fabricated = {**result, "selected_food_items": ["invento"], "explanation": "La etiqueta es correcta"}
        scores = score_output(case, gold, fabricated, prefs, restrictions)
        self.assertFalse(scores["decision_correct"])
        self.assertFalse(scores["fidelity"])
        fabricated = {**result, "evidence_ids": ["inexistente"]}
        self.assertFalse(score_output(case, gold, fabricated, prefs, restrictions)["evidence_correct"])
        fabricated = {**result, "detected_conflict_type": "BOC"}
        self.assertFalse(score_output(case, gold, fabricated, prefs, restrictions)["decision_correct"])

    def test_restriction_metric_uses_typed_expectation_not_only_category_name(self):
        case, gold = self.public[0], self.gold[0]
        result, prefs, restrictions = evaluate_case(case)
        food = result["selected_food_items"][0]
        expected = {**gold, "restriction_food_items": [food]}
        self.assertTrue(score_output(case, expected, result, prefs, restrictions)["restriction_violation"])

    def test_no_denominator_is_none_not_zero_or_success(self):
        self.assertIsNone(rate(0, 0)["value"])
        self.assertIsNone(rate(0, 0)["wilson_95"])
        metrics = summarize([])
        self.assertIsNone(metrics["necessary_clarification_rate"]["value"])
        self.assertIsNone(metrics["restriction_violation_rate"]["value"])

    def test_evaluation_accepts_list_fixture_and_does_not_claim_gate_a_without_tests(self):
        with TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture.json"
            fixture.write_text(json.dumps([{"case": self.public[0], "expected": self.gold[0]}]), encoding="utf8")
            report = run_evaluation(Path(directory) / "report", seeds=(42,), instances=20,
                                    run_tests=False, challenge_path=fixture)
            self.assertEqual(report["challenge"]["evaluable"], 1)
            self.assertIsNone(report["gates"]["A"])
            self.assertNotEqual(report["verdict"], "FUNCIONAL Y VALIDADO EN PRUEBAS SINTÉTICAS")
            self.assertTrue((Path(directory) / "report" / "summary.md").exists())

    def test_four_combinations_are_not_enough_all_eight_are_required(self):
        cases = [case for case in self.public if case["category"] == "S08"]
        signatures = set()
        for case in cases:
            result, *_ = evaluate_case(case)
            self.assertEqual(result["selected_action"], "ANSWER")
            signatures.add((result["resolved_context"]["day_scope"], result["resolved_context"]["meal_scope"]))
        self.assertEqual(len(signatures), 8)

    def test_all_output_fields_are_present(self):
        result, *_ = evaluate_case(self.public[0])
        self.assertTrue({"scenario_id", "user_id", "query", "resolved_context", "retrieved_preference_ids",
                         "applicable_preference_ids", "detected_conflict_type", "selected_action", "selected_food_items",
                         "evidence_ids", "clarification_question", "explanation", "execution_time_ms"} <= result.keys())
        self.assertGreaterEqual(result["execution_time_ms"], 0)

    def test_event_metadata_collision_cannot_reinterpret_original(self):
        store = Store(":memory:", "u")
        self.addCleanup(store.close)
        engine = FoodPreferenceEngine(store)
        message = {"message_id": "e", "at": "2026-10-01", "session_id": "s", "role": "user",
                   "channel": "chat", "source": "USER_STATEMENT", "text": "Me gusta cereal.",
                   "food_aliases": {"avena": ["cereal"]}, "context": {"place": "home"}}
        engine.ingest([message])
        before = store.food_snapshot()
        with self.assertRaises(ValueError):
            engine.ingest([{**message, "food_aliases": {"arroz": ["cereal"]}}])
        self.assertEqual(before, store.food_snapshot())


class EightIndependentContextTests(unittest.TestCase):
    """Each combination has its own unittest result and competes with seven others."""
    def check_combination(self, query, food):
        declarations = [
            "Entre semana para el desayuno prefiero avena.",
            "Entre semana para el almuerzo prefiero arroz.",
            "Entre semana para la cena prefiero sopa.",
            "Entre semana para la merienda prefiero fruta.",
            "Los fines de semana para el desayuno prefiero tostadas.",
            "Los fines de semana para el almuerzo prefiero lentejas.",
            "Los fines de semana para la cena prefiero pizza.",
            "Los fines de semana para la merienda prefiero yogur.",
        ]
        store = Store(":memory:", "u")
        self.addCleanup(store.close)
        engine = FoodPreferenceEngine(store)
        engine.ingest([{"message_id": f"e{i}", "session_id": f"s{i}", "at": "2026-10-01T12:00:00+00:00",
                        "role": "user", "channel": "chat", "source": "USER_STATEMENT", "text": text}
                       for i, text in enumerate(declarations)])
        result = engine.decide("combination", "u", query, "2026-10-19T12:00:00+00:00")
        self.assertEqual(result["selected_action"], "ANSWER")
        self.assertEqual(result["selected_food_items"], [food])
        self.assertEqual(len(result["applicable_preference_ids"]), 1)
        self.assertEqual(len(result["retrieved_preference_ids"]), 8)
        self.assertIsNone(result["detected_conflict_type"])
        self.assertEqual(len(store.preferences()), 8)

    def test_weekday_breakfast(self):
        self.check_combination("¿Qué desayuno prefiero el lunes?", "avena")

    def test_weekday_lunch(self):
        self.check_combination("¿Qué almuerzo prefiero el martes?", "arroz")

    def test_weekday_dinner(self):
        self.check_combination("¿Qué cena prefiero el miércoles?", "sopa")

    def test_weekday_snack(self):
        self.check_combination("¿Qué merienda prefiero el jueves?", "fruta")

    def test_weekend_breakfast(self):
        self.check_combination("¿Qué desayuno prefiero el sábado?", "tostadas")

    def test_weekend_lunch(self):
        self.check_combination("¿Qué almuerzo prefiero el domingo?", "lentejas")

    def test_weekend_dinner(self):
        self.check_combination("¿Qué cena prefiero el sábado?", "pizza")

    def test_weekend_snack(self):
        self.check_combination("¿Qué merienda prefiero el domingo?", "yogur")
