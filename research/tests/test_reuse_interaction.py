from research.src.data.build_bank import build_bank
from research.src.memory.store import Store
from research.src.env.user_simulator import UserSimulator
from research.src.evaluation.reuse_interaction import evaluate_interactive_reuse


def test_secondary_reuse_is_fresh_and_has_one_question(tmp_path):
    episode = build_bank(roots=5)[0]
    path = tmp_path / "s3.sqlite"
    store = Store(path)
    store.initialize(episode["initial_memory"])
    store.close()
    seen = []
    class FixedController:
        def select(self, obs):
            seen.append(obs)
            return {"action": "ASK", "slot": "CURRENT_CONTEXT"}
        def respond(self, obs):
            seen.append(obs)
            return {"mode": "DEFER", "value": None, "branches": [], "evidence_ids": []}
    simulator = UserSimulator({"CURRENT_CONTEXT": {"turn_id": "u4", "text": "Ahora estoy en casa."}})
    result = evaluate_interactive_reuse(path, episode["future_query"], FixedController(), simulator,
        score=lambda response, obs: {"success": 0}, previously_clarified_slots=["CURRENT_CONTEXT"],
        same_scope_still_applies=False)
    assert seen[0]["user_messages"] == []
    assert len(seen[1]["user_messages"]) == 1
    assert "u2" not in str(seen)
    assert result["questions_session_3"] == 1
    assert result["unnecessary_repeated_scope_questions"] == 0
    assert result["regime"] == "secondary_interactive_S3"
