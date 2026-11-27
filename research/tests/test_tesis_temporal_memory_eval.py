from research.tesis_temporal_memory_eval import replay, score_reply
from research.tesis_temporal_memory_policy import response
from research.tesis_temporal_memory_train import QA, read


def test_out_of_scope_prediction_is_executed_without_repair(tmp_path):
    case = next(item for item in read(QA) if item["id"] == "qa-hilo-cruza-medianoche")

    class AlwaysHabitual:
        def choose(self, phase, observation, choices, key):
            if phase == "write":
                selected = next((candidate for candidate in choices
                                 if candidate.get("operations", [{}])[0].get("record", {}).get("scope", {}).get("kind")
                                 == "ongoing"), choices[0])
            else:
                selected = next((candidate for candidate in choices if candidate.get("mode") == "ANSWER"), choices[0])
            return selected, {"phase": phase}

    rows = list(replay(AlwaysHabitual(), case, tmp_path / "actual.sqlite", method="grpo", seed=101))
    written = next(row for row in rows if row["input"]["expect"]["writes"])
    assert written["execution"]["valid"]
    assert written["memory_score"]["added"][0]["scope"]["kind"] == "ongoing"
    assert not written["memory_score"]["passed"]
    assert not written["verify"]
    assert any(not row["passed"] for row in rows if row["input"]["action"] == "new_session")


def test_a_correct_value_with_an_unrelated_citation_is_a_failed_recall():
    case = next(item for item in read(QA) if item["id"] == "qa-fecha-de-consulta-distinta")
    query = {"attribute": case["initial_memory"][0]["attribute"], "session_id": "S1",
             "date": case["steps"][0]["date"], "context": None}
    recalled = case["initial_memory"][0]["value"]
    gold = {"kind": "recall", "values": [recalled], "evidence_ids": [case["initial_memory"][0]["memory_id"]]}
    grade = score_reply(gold, response("ANSWER", recalled, evidence=["inventado"]), None, case["initial_memory"], query)
    assert grade["value_correct"] and not grade["passed"] and not grade["grounded"]
