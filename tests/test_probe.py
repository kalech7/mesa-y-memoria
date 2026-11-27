from evaluation.retrieval_probe import probe
from tests.helpers import OfflineTest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
from data_generation.build import write_dataset
from evaluation.retrieval_probe import main


class RetrievalProbeTests(OfflineTest):
    def test_probe_does_not_send_gold_to_embedding_service(self):
        case, gold = self.case()
        class FakeEmbeddings:
            def vectors(self, texts):
                self.texts = texts
                return [[1, 0] for _ in texts]
        embedder = FakeEmbeddings()
        rows = probe(case, gold, embedder)
        self.assertEqual([r["topk"] for r in rows], [4, 8, 16])
        self.assertEqual(embedder.texts[-1], case["query"])
        self.assertNotIn("hidden_state", str(embedder.texts))
        self.assertEqual(len(embedder.texts), len(case["history"]) + 1)

    def test_remote_probe_requires_human_validation_before_network(self):
        with TemporaryDirectory() as temporary, patch("evaluation.retrieval_probe.BGE") as embedder:
            root = Path(temporary)
            write_dataset(root / "dataset", self.public, self.gold)
            self.assertEqual(main(["--dataset", str(root / "dataset"), "--output", str(root / "probe"),
                "--url", "private-placeholder", "--limit", "5"]), 2)
            embedder.assert_not_called()
            self.assertFalse((root / "probe").exists())
