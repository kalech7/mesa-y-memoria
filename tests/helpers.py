import socket
import unittest
from unittest.mock import patch
from data_generation.build import make_dataset


class OfflineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public, cls.gold = make_dataset(seed=42, dev_people=2, test_people=3)

    def setUp(self):
        self.network = patch("urllib.request.urlopen", side_effect=AssertionError("La prueba intentó acceder a red"))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.sockets = patch.object(socket, "create_connection", side_effect=AssertionError("Socket prohibido en pruebas"))
        self.sockets.start()
        self.addCleanup(self.sockets.stop)

    def case(self, variant="open_single", topic="tofu"):
        for case, gold in zip(self.public, self.gold):
            if gold["variant"] == variant and gold["topic_id"] == topic:
                return case, gold
        raise AssertionError("Fixture no encontrada")


def completion(obj, finish="stop", content=None, reasoning=3):
    import json
    return {"model": "fake-local", "choices": [{"finish_reason": finish, "message": {"content": json.dumps(obj) if content is None else content}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 9, "completion_tokens_details": {"reasoning_tokens": reasoning}}}
