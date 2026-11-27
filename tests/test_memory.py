import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from agents.client import TechnicalError
from memory.store import Store
from memory import retrieval, extraction
from tests.helpers import OfflineTest


class MemoryTests(OfflineTest):
    def test_roundtrip_preserves_provenance_and_negation(self):
        case, _ = self.case()
        message = case["history"][0]
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "memory.sqlite"
            store = Store(path, case["person_id"])
            store.add(message, text="No quiero tofu", source="extracted")
            store.close()
            reopened = Store(path, case["person_id"])
            row = reopened.all()[0]
            self.assertEqual(row["text"], "No quiero tofu")
            self.assertEqual(row["raw_text"], message["text"])
            self.assertEqual(row["origin_id"], message["message_id"])
            reopened.close()

    def test_scenario_databases_are_isolated(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "one.sqlite"
            store = Store(path, "one")
            store.close()
            with self.assertRaises(ValueError):
                Store(path, "two")

    def test_add_is_idempotent_but_collision_rejected(self):
        case, _ = self.case()
        store = Store(":memory:", case["person_id"])
        self.addCleanup(store.close)
        message = case["history"][0]
        store.add(message)
        store.add(message)
        self.assertEqual(len(store.all()), 1)
        with self.assertRaises(ValueError):
            store.add({**message, "text": "Otro contenido"})

    def test_saved_clarification_survives_reopening(self):
        case, _ = self.case()
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "memory.sqlite"
            store = Store(path, case["person_id"])
            message = {**case["history"][0], "source": "clarification", "text": "De ahora en adelante sin tofu"}
            store.add(message)
            store.close()
            reopened = Store(path, case["person_id"])
            self.assertTrue(any(r["source"] == "clarification" and r["text"] == message["text"] for r in reopened.all()))
            reopened.close()

    def test_explicit_replacement_keeps_original_history_and_links(self):
        case, _ = self.case()
        store = Store(":memory:", case["person_id"])
        self.addCleanup(store.close)
        original = case["history"][0]
        old = store.add(original)
        new = {**original, "message_id": "new", "source": "clarification", "text": "De ahora en adelante sin tofu"}
        store.clarify(new, [old], resolved=True)
        records = store.all()
        self.assertEqual(len(records), 2)
        self.assertEqual(next(r for r in records if r["event_id"] == old)["status"], "superseded")
        self.assertEqual(next(r for r in records if r["event_id"] == old)["text"], original["text"])
        self.assertIn(old, next(r for r in records if r["event_id"] == "new")["linked_ids"])

    def test_uncommunicated_correction_cannot_be_saved(self):
        case, _ = self.case()
        store = Store(":memory:", case["person_id"])
        self.addCleanup(store.close)
        with self.assertRaises(ValueError):
            store.clarify(case["history"][0], [], resolved=False)

    def test_snapshot_is_stable_and_cannot_be_overwritten(self):
        case, _ = self.case()
        store = Store(":memory:", case["person_id"])
        self.addCleanup(store.close)
        before = store.snapshot()
        store.add(case["history"][0])
        self.assertNotEqual(before, store.snapshot())
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.sqlite"
            store.backup(path)
            with self.assertRaises(FileExistsError):
                store.backup(path)

    def test_bge_missing_one_side_is_detected(self):
        case, gold = self.case()
        store = Store(":memory:", case["person_id"])
        self.addCleanup(store.close)
        for msg in case["history"]:
            store.add(msg)
        records = store.all()
        target = gold["evidence"][0]["message_id"]
        class FakeEmbeddings:
            def vectors(self, texts):
                return [[1, 0] if r["origin_id"] == target else [0, 1] for r in records] + [[1, 0]]
        one_side = retrieval.retrieve(records, case["query"], "bge", 1, FakeEmbeddings())
        self.assertFalse(retrieval.pair_recall(one_side, gold))
        self.assertTrue(retrieval.pair_recall(records, gold))

    def test_embedding_batch_count_and_dimensions_are_checked(self):
        with patch("memory.retrieval.http_json", return_value={"embeddings": [[1, 0]]}):
            with self.assertRaises(TechnicalError):
                retrieval.BGE("private-placeholder").vectors(["one", "two"])
        with self.assertRaises(TechnicalError):
            retrieval.cosine([1], [1, 2])
        with self.assertRaises(TechnicalError):
            retrieval.cosine([float("nan")], [1])

    def test_embedding_authentication_is_passed_without_logging_secret(self):
        logs = []
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "test-secret"}), patch(
            "memory.retrieval.http_json", return_value={"embeddings": [[1, 0]]}
        ) as transport:
            retrieval.BGE("private-placeholder", recorder=logs.append).vectors(["one"])
        self.assertEqual(transport.call_args.args[3], "test-secret")
        self.assertNotIn("test-secret", str(logs))

    def test_extraction_that_erases_negation_is_detected(self):
        self.assertTrue(extraction.negation_loss("No quiero tofu", ["Quiero tofu"]))
        self.assertFalse(extraction.negation_loss("No quiero tofu", ["Prefiero sin tofu"]))

    def test_extractor_sees_only_past_message_and_context(self):
        case, _ = self.case()
        requests = []
        class FakeClient:
            def complete(self, messages):
                requests.extend(messages)
                return {"memories": ["No quiero tofu"]}, []
        extractor = extraction.Extractor(FakeClient())
        extractor.extract(case["history"][0], "Contexto previo público")
        self.assertNotIn(case["followup"], str(requests))
        self.assertNotIn("hidden_state", str(requests))
