from evaluation.natural_baseline import select_reviewed, natural_messages
from tests.helpers import OfflineTest


class NaturalBaselineTests(OfflineTest):
    def test_action_review_does_not_enter_agent_messages(self):
        case, _ = self.case()
        review = {"case_id": case["case_id"], "reviewed_fields": ["acceptable_actions"],
                  "acceptable_actions": ["ASK"], "reviewer": "human", "source_reply": "PRIVATE_REVIEW_CANARY"}
        selected = select_reviewed(self.public, [review], [case["case_id"]])
        messages = natural_messages(selected[0])
        self.assertNotIn("PRIVATE_REVIEW_CANARY", str(messages))
        self.assertNotIn("acceptable_actions", str(messages))
        self.assertNotIn("historical_conflict", str(messages))
        self.assertEqual(len(messages), len(case["history"]) + 2)
        self.assertEqual(messages[-1]["content"], case["query"])
        self.assertNotIn(case["followup"], str(messages))

    def test_question_usefulness_review_cannot_unlock_unreviewed_case(self):
        case, _ = self.case()
        review = {"case_id": case["case_id"], "reviewed_fields": ["question_useful_human"],
                  "question_useful_human": True, "reviewer": "human", "source_reply": "Useful"}
        with self.assertRaises(ValueError):
            select_reviewed(self.public, [review], [case["case_id"]])

    def test_holdout_is_not_used_for_diagnostic(self):
        case = next(c for c in self.public if c["split"] == "test")
        review = {"case_id": case["case_id"], "reviewed_fields": ["acceptable_actions"],
                  "acceptable_actions": ["ASK"], "reviewer": "human", "source_reply": "Ask"}
        with self.assertRaises(ValueError):
            select_reviewed(self.public, [review], [case["case_id"]])
