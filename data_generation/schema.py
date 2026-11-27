import hashlib
import json
from datetime import datetime

PUBLIC_KEYS = {"case_id", "person_id", "split", "synthetic", "reference_time", "history", "query", "followup", "future_distractors"}
MESSAGE_KEYS = {"message_id", "session_id", "at", "role", "channel", "source", "text", "context"}
PRIVATE_KEYS = {"truth", "gold_action", "user_reply", "rationale", "option_A", "option_B", "hidden_state", "acceptable_actions", "annotation", "variant"}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_public(case):
    if set(case) != PUBLIC_KEYS:
        raise ValueError("Esquema público inesperado; posible filtración")
    if case["synthetic"] is not True or case["split"] not in {"dev", "test"}:
        raise ValueError("Origen o partición inválidos")
    reference = datetime.fromisoformat(case["reference_time"])
    seen, previous = set(), None
    for msg in case["history"]:
        if set(msg) != MESSAGE_KEYS or msg["role"] not in {"user", "assistant"}:
            raise ValueError("Mensaje público inválido")
        at = datetime.fromisoformat(msg["at"])
        if at >= reference or (previous and at < previous) or msg["message_id"] in seen:
            raise ValueError("Cronología/identificadores inválidos")
        seen.add(msg["message_id"])
        previous = at
    for query in [case["query"], case["followup"]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Consulta vacía")
    return case


def assert_no_private_keys(value):
    if isinstance(value, dict):
        if set(value) & PRIVATE_KEYS:
            raise ValueError("Campo privado en entrada del agente")
        for item in value.values():
            assert_no_private_keys(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_private_keys(item)


def validate_dataset(public, gold):
    if len(public) != len(gold):
        raise ValueError("Gold sin pareja pública")
    ids, people = set(), {"dev": set(), "test": set()}
    for case, annotation in zip(public, gold):
        validate_public(case)
        assert_no_private_keys(case)
        if case["case_id"] in ids or annotation["case_id"] != case["case_id"]:
            raise ValueError("IDs duplicados/desalineados")
        ids.add(case["case_id"])
        people[case["split"]].add(case["person_id"])
        messages = {m["message_id"]: m for m in case["history"]}
        for evidence in annotation["evidence"]:
            if evidence["message_id"] not in messages or evidence["quote"] != messages[evidence["message_id"]]["text"]:
                raise ValueError("Evidencia no trazable")
        if annotation["historical_conflict"] and len(annotation["evidence"]) < 2:
            raise ValueError("Contradicción sin evidencia bilateral")
        if not set(annotation["acceptable_actions"]) <= {"ASK", "ANSWER", "CONDITIONAL"}:
            raise ValueError("Acciones inválidas")
    if people["dev"] & people["test"]:
        raise ValueError("Persona presente en desarrollo y evaluación")
    return {k: len(v) for k, v in people.items()}
