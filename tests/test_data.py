import copy
import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from data_generation.build import make_dataset, write_dataset, VARIANTS
from data_generation.schema import validate_dataset, validate_public, assert_no_private_keys, digest
from evaluation.runner import balanced_select, check_gates
from evaluation.annotate import apply_reviews
from tests.helpers import OfflineTest


class DatasetTests(OfflineTest):
    def test_reproducible_and_split_by_person(self):
        public, gold = make_dataset(42, 2, 3)
        self.assertEqual(public, self.public)
        self.assertEqual(gold, self.gold)
        self.assertEqual(validate_dataset(public, gold), {"dev": 2, "test": 3})
        self.assertEqual(len(public), 5 * len(VARIANTS))

    def test_public_projection_contains_no_hidden_fields_or_variant_ids(self):
        for case in self.public:
            assert_no_private_keys(case)
            self.assertNotIn("__", case["case_id"])
            self.assertNotIn("MEMORIA A", json.dumps(case))

    def test_hidden_field_injection_is_rejected(self):
        case = copy.deepcopy(self.public[0])
        case["truth"] = "SECRET"
        with self.assertRaises(ValueError):
            validate_public(case)

    def test_nested_private_field_is_rejected(self):
        with self.assertRaises(ValueError):
            assert_no_private_keys({"history": [{"rationale": "SECRET"}]})

    def test_chronology_is_frozen_and_longitudinal(self):
        for case in self.public:
            start = datetime.fromisoformat(case["history"][0]["at"])
            reference = datetime.fromisoformat(case["reference_time"])
            self.assertTrue(90 <= (reference - start).days <= 550)
            self.assertTrue(8 <= len({m["session_id"] for m in case["history"]}) <= 25)
            self.assertTrue(8 <= len(case["history"]) <= 25)
            self.assertEqual({m["role"] for m in case["history"]}, {"user", "assistant"})

    def test_evidence_tampering_is_rejected(self):
        gold = copy.deepcopy(self.gold)
        gold[0]["evidence"][0]["quote"] = "Texto inventado"
        with self.assertRaises(ValueError):
            validate_dataset(self.public, gold)

    def test_scope_control_is_not_a_direct_conflict(self):
        _, gold = self.case("scope")
        self.assertFalse(gold["historical_conflict"])
        self.assertFalse(gold["unresolved_conflict"])
        self.assertEqual(gold["kind"], "scope_mismatch")

    def test_current_and_correction_resolve_conflict(self):
        for variant in ("current", "correction"):
            _, gold = self.case(variant)
            self.assertTrue(gold["historical_conflict"])
            self.assertFalse(gold["unresolved_conflict"])
            self.assertEqual(gold["acceptable_actions"], ["ANSWER"])

    def test_open_allows_conditional_when_task_can_be_bifurcated(self):
        self.assertEqual(set(self.case("open")[1]["acceptable_actions"]), {"ASK", "CONDITIONAL"})
        self.assertEqual(self.case("open_single")[1]["acceptable_actions"], ["ASK"])

    def test_smoke_does_not_select_only_variants_of_one_person(self):
        public, gold = make_dataset(42, 20, 1)
        selected = balanced_select(public, gold, "dev", 10, 42)
        self.assertEqual(len({c["person_id"] for c, _ in selected}), 10)
        self.assertEqual(len({g["variant"] for _, g in selected}), 10)

    def test_private_dataset_permissions_and_no_overwrite(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "data"
            manifest = write_dataset(path, self.public, self.gold)
            self.assertEqual(manifest["human_validated"], 0)
            self.assertEqual((path / "private" / "casos_gold.jsonl").stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                write_dataset(path, self.public, self.gold)

    def test_human_gate_is_not_satisfied_by_synthetic_ids(self):
        self.assertTrue(check_gates([(self.public[0], self.gold[0])], self.gold))

    def test_review_import_keeps_original_and_requires_evidence_of_review(self):
        case, _ = self.case("open_single")
        row = {"case_id": case["case_id"], "validated": "true", "reviewer": "R1", "notes": "Dos declaraciones aplicables; la consulta exige decidir.",
            "historical_conflict": "true", "unresolved_conflict": "true", "kind": "unresolved_conflict", "acceptable_actions": "ASK"}
        result = apply_reviews(self.public, self.gold, [row])
        changed = next(g for g in result if g["case_id"] == case["case_id"])
        self.assertTrue(changed["annotation"]["validated"])
        self.assertIn("initial_proposal", changed)
        row["reviewer"] = ""
        with self.assertRaises(ValueError):
            apply_reviews(self.public, self.gold, [row])
