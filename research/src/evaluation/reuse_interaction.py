"""E6 secondary regime: one new question after loading persisted S3 memory.

The same frozen controller and reply table must be used across all methods.
Metrics from this regime are never merged into autonomous S3 accuracy.
"""
from copy import deepcopy

from ..memory.store import Store
from .runner import select_valid
from ..scoring.evidence import response_valid


def evaluate_interactive_reuse(memory_path, query, controller, simulator, *, score,
                               previously_clarified_slots=(), same_scope_still_applies=False):
    backend = getattr(controller, "backend", None)
    if backend is not None and any(p.requires_grad for p in backend.model.parameters()):
        raise ValueError("E6_REQUIRES_COMMON_FROZEN_CONTROLLER")
    if hasattr(controller, "phase_records"):
        controller.phase_records.clear()
    store = Store(memory_path)
    try:
        memory = store.list_records()
    finally:
        store.close()
    observation = {"memory": memory, "query": deepcopy(query), "user_messages": [],
                   "reuse_regime": "Esta sesión de reutilización permite como máximo una pregunta."}
    errors = []
    try:
        selection = controller.select(deepcopy(observation))
    except (ValueError, TypeError, KeyError) as exc:
        selection = {}
        errors.append(repr(exc))
    asked, reply, repeated = 0, None, 0
    if not select_valid(selection):
        errors.append("INVALID_SELECT")
    elif selection["action"] == "ASK":
        reply = simulator.reply(selection["slot"])
        observation["user_messages"].append(deepcopy(reply))
        asked = 1
        repeated = int(same_scope_still_applies and selection["slot"] in previously_clarified_slots)
    try:
        response = controller.respond(deepcopy(observation))
    except (ValueError, TypeError, KeyError) as exc:
        response = {}
        errors.append(repr(exc))
    result = score(response, deepcopy(observation)) if response_valid(response) else {"success": 0, "valid": False}
    records = controller.serialized_records() if hasattr(controller, "serialized_records") else []
    if hasattr(controller, "phase_records"):
        controller.phase_records.clear()
    return {"regime": "secondary_interactive_S3", "questions_session_3": asked,
            "unnecessary_repeated_scope_questions": repeated,
            "response": response, "user_reply": reply, "observation": observation,
            "selection": selection, "score": result, "errors": errors, "phase_records": records}
