from evaluation.behavior import describe
from tests.helpers import OfflineTest


class BehaviorTests(OfflineTest):
    def test_observation_does_not_depend_on_hidden_labels(self):
        rows = [{"case_id": "one", "condition": "neutral", "initial_action": "ANSWER",
                 "initial_answer": "Compra salmón", "acceptable_actions": ["ASK"],
                 "hidden_state": "CANARY_PRIVATE", "action_correct": False}]
        observations, summary = describe(rows)
        self.assertNotIn("CANARY_PRIVATE", str(observations))
        self.assertNotIn("acceptable_actions", str(observations))
        self.assertEqual(summary["conditions"]["neutral"]["actions_declared"]["ANSWER"], 1)
        rows[0]["acceptable_actions"] = ["ANSWER"]
        rows[0]["action_correct"] = True
        self.assertEqual(describe(rows), (observations, summary))

    def test_timeout_is_not_counted_as_answer_or_failure_of_policy(self):
        rows = [{"case_id": "one", "condition": "guided", "initial_action": "ASK"},
                {"case_id": "two", "condition": "guided", "technical_errors": [{"kind": "connection_or_timeout"}]}]
        _, summary = describe(rows)
        result = summary["conditions"]["guided"]
        self.assertEqual(result["initial_evaluable"], 1)
        self.assertEqual(result["initial_missing"], 1)
        self.assertEqual(result["ASK_rate"], 1)
        self.assertEqual(result["actions_declared"]["ANSWER"], 0)
