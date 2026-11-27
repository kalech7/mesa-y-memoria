import json
from unittest.mock import patch
from agents.client import API, TechnicalError
from agents import policies
from tests.helpers import OfflineTest, completion


class APIAndPolicyTests(OfflineTest):
    def test_reasoning_and_retry_attempts_are_preserved(self):
        empty = {"choices": [{"finish_reason": "length", "message": {"content": None}}], "usage": {"completion_tokens": 32, "completion_tokens_details": {"reasoning_tokens": 32}}}
        good = completion({"ok": True})
        log = []
        with patch("agents.client.http_json", side_effect=[empty, good]) as transport:
            obj, attempts = API("private-placeholder", "fake", max_tokens=32, recorder=log.append).complete([])
        self.assertTrue(obj["ok"])
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0]["reasoning_tokens"], 32)
        self.assertEqual(attempts[0]["content"], None)
        self.assertEqual(transport.call_args_list[1].args[1]["max_tokens"], 64)
        self.assertEqual(transport.call_args_list[0].args[1]["response_format"]["type"], "json_schema")
        self.assertEqual(len(log), 2)

    def test_empty_content_is_technical_not_wrong_action(self):
        empty = {"choices": [{"finish_reason": "length", "message": {"content": None}}], "usage": {}}
        with patch("agents.client.http_json", return_value=empty):
            with self.assertRaises(TechnicalError) as caught:
                API("private-placeholder", "fake").complete([])
        self.assertEqual(caught.exception.kind, "empty_content")
        self.assertEqual(len(caught.exception.attempts), 2)

    def test_http_errors_are_not_retried_or_exposed(self):
        with patch("agents.client.http_json", side_effect=TechnicalError("http_error", status=401)) as transport:
            with self.assertRaises(TechnicalError) as caught:
                API("private-placeholder", "fake").complete([])
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(caught.exception.status, 401)
        self.assertNotIn("private-placeholder", str(caught.exception))

    def test_parse_error_is_technical(self):
        with patch("agents.client.http_json", return_value=completion({}, content="not JSON")):
            with self.assertRaises(TechnicalError) as caught:
                API("private-placeholder", "fake").complete([])
        self.assertEqual(caught.exception.kind, "parse_error")

    def test_natural_response_does_not_request_json_or_parse_free_text(self):
        with patch("agents.client.http_json", return_value=completion({}, content="Sí, compra salmón.")) as transport:
            text, attempts = API("private-placeholder", "fake").complete_text([])
        self.assertEqual(text, "Sí, compra salmón.")
        self.assertNotIn("response_format", transport.call_args.args[1])
        self.assertNotIn("response_format", attempts[0])
        self.assertEqual(attempts[0]["content"], text)

    def test_minimal_prompt_does_not_instruct_conflict_detection_or_actions(self):
        case, _ = self.case()
        messages = policies.build_messages(case["query"], [], case["reference_time"], system_prompt=policies.MINIMAL)
        for forbidden in ("conflicto", "ASK", "ANSWER", "CONDITIONAL", "JSON", "pregunta", "recencia"):
            self.assertNotIn(forbidden, messages[0]["content"])
        self.assertEqual(messages[-1]["content"], case["query"])

    def test_strict_decision_schema(self):
        good = policies.decision("ASK", question="¿Qué prefieres?")
        self.assertEqual(policies.standardize(good), good)
        for change in ({"answer": "No debe estar"}, {"historical_conflict": "true"}, {"question": ""}):
            with self.assertRaises(TechnicalError):
                policies.standardize({**good, **change})

    def test_hidden_answer_canary_never_enters_prompt(self):
        case, _ = self.case()
        records = [{"at": case["reference_time"], "source": "conversation", "session_id": "public", "role": "user", "status": "active", "text": "Mensaje público", "raw_text": "GOLD_SENTINEL_DO_NOT_SEND", "truth": "GOLD_SENTINEL_DO_NOT_SEND"}]
        messages = policies.build_messages(case["query"], records, case["reference_time"])
        self.assertNotIn("GOLD_SENTINEL_DO_NOT_SEND", json.dumps(messages))
        self.assertNotIn('"truth"', json.dumps(messages))

    def test_real_roles_are_preserved(self):
        records = [{"at": "2026-01-01", "source": "conversation", "session_id": "x", "role": role, "status": "active", "text": role} for role in ("user", "assistant")]
        messages = policies.build_messages("Consulta", records, "2026-10-01")
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])

    def test_baseline_receives_clarification_as_public_turn(self):
        result, _ = policies.Heuristic("recency_only").decide("¿Compro tofu?", [], "2026-10-10", [{"role": "user", "content": "De ahora en adelante prefiero sin tofu"}])
        self.assertIn("sin tofu", result["answer"])

    def test_medical_restrictions_are_not_presented_as_interchangeable_tastes(self):
        self.assertIn("no son gustos intercambiables", policies.NEUTRAL)
