from evaluation.human import aggregate
from tests.helpers import OfflineTest


class HumanReviewTests(OfflineTest):
    def test_blank_packets_are_pending_not_human_success(self):
        result = aggregate([{"review_id": "a", "reviewer": ""}], [{"review_id": "a", "condition": "guided"}])
        self.assertTrue(result["pending"])
        self.assertEqual(result["human_rating_rows"], 0)

    def test_disagreement_is_preserved(self):
        rows = [{"review_id": "a", "reviewer": "R1", "question_useful_human": "true"},
                {"review_id": "a", "reviewer": "R2", "question_useful_human": "false"}]
        result = aggregate(rows, [{"review_id": "a", "condition": "guided"}])
        self.assertEqual(result["agreement"]["question_useful_human"]["disagreements"], 1)
        self.assertEqual(result["conditions"]["guided"]["question_useful_human"], {"true": 1, "false": 1})

    def test_duplicate_reviewer_is_rejected(self):
        row = {"review_id": "a", "reviewer": "R1"}
        with self.assertRaises(ValueError):
            aggregate([row, row], [{"review_id": "a", "condition": "guided"}])
