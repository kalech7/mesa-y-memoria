from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Barrier

import pytest

from research.tesis_profiles_parser_v5 import public_evidence
from web_demo.profiles_chat import ProfilesChats
from web_demo.profiles_runtime import ProfilesError


class Runtime:
    def __init__(self, *, barrier=None):
        self.calls = []
        self.barrier = barrier
        self.fail = False

    def resolve(self, identity=None):
        return {"id": identity or "sft-101", "method": "SFT-ramas", "eligible": True,
                "available": True, "epoch": 2, "seed": 101, "panel": "validation_selected"}

    def catalog(self):
        return {"models": [self.resolve()], "default_model": "sft-101"}

    def decide(self, model, observation, legal):
        self.calls.append(deepcopy(observation))
        if self.fail:
            raise ProfilesError("fixture unavailable", 503)
        if self.barrier:
            self.barrier.wait(timeout=5)
        slot = public_evidence(observation)["missing_slot"]
        action = slot if slot in legal else "PROCEED"
        return action, {"source": "fixture", "tokens": 42, "model": model}


def start(chats, **options):
    return chats.handle({"action": "start", "date": "2026-12-03", **options})


def payload(chat, action, **values):
    return {"action": action, "id": chat["id"], "owner_token": chat["owner_token"],
            "revision": chat["revision"], "request_id": str(chat["revision"]), **values}


def send(chats, chat, text):
    return {**chats.handle(payload(chat, "send", text=text)), "owner_token": chat["owner_token"]}


def new_visit(chats, chat, **values):
    return {**chats.handle(payload(chat, "new_visit", **values)), "owner_token": chat["owner_token"]}


def test_real_reply_quotes_and_memory_survive_process_and_visit_reload(tmp_path):
    runtime = Runtime()
    first = ProfilesChats(storage=tmp_path, runtime=runtime)
    chat = start(first)
    chat = send(first, chat, "Para la cena, prefiero pasta.")
    assert chat["pending"]["slot"] == "TEMPORAL_SCOPE"
    assert chat["memory"] == [] and chat["questions_remaining"] == 1
    response = payload(chat, "send", text="Déjala como mi cena habitual para las próximas veces.")
    committed = first.handle(response)
    assert len(committed["memory"]) == 1
    record = committed["memory"][0]
    assert record["value"] == "pasta" and record["scope"]["kind"] == "ongoing"
    assert {quote["text"] for quote in record["evidence_quotes"]} <= {
        message["text"] for message in committed["messages"] if message["role"] == "user"}
    assert committed["messages"][-1]["operations"][0]["op"] == "ADD"
    second = ProfilesChats(storage=tmp_path, runtime=Runtime())
    assert second.handle(response) == committed
    chat = second.handle({"action": "load", "id": chat["id"], "owner_token": chat["owner_token"]})
    chat["owner_token"] = response["owner_token"]
    chat = new_visit(second, chat)
    assert chat["messages"] == [] and chat["visit"] == 2 and chat["questions_remaining"] == 1
    chat = send(second, chat, "¿Qué cena tengo guardada?")
    assert chat["messages"][-1]["response"]["value"] == "pasta"
    assert chat["messages"][-1]["citations"][0]["source"] == "sqlite"
    assert chat["memory"][0]["memory_id"] == record["memory_id"]


def test_interval_expiry_and_session_scope_are_applied_after_reopening_sqlite(tmp_path):
    chats = ProfilesChats(storage=tmp_path, runtime=Runtime())
    chat = start(chats)
    chat = send(chats, chat, "Para la cena, prefiero pasta de forma habitual.")
    chat = send(chats, chat, "Para la cena, prefiero sopa. Desde hoy hasta el 2026-12-04 inclusive.")
    assert chat["memory"][-1]["scope"]["valid_until_exclusive"] == "2026-12-05"
    chat = new_visit(chats, chat, date="2026-12-04")
    chat = send(chats, chat, "¿Qué cena tengo guardada?")
    assert chat["messages"][-1]["response"]["value"] == "sopa"
    chat = new_visit(chats, chat, date="2026-12-05")
    chat = send(chats, chat, "¿Qué cena tengo guardada?")
    assert chat["messages"][-1]["response"]["value"] == "pasta"
    assert chat["memory"][-1]["applicable"] is False
    chat = send(chats, chat, "Para la cena, prefiero pizza solo para esta sesión.")
    assert chat["memory"][-1]["scope"]["kind"] == "session"
    chat = new_visit(chats, chat)
    chat = send(chats, chat, "¿Qué cena tengo guardada?")
    assert chat["messages"][-1]["response"]["value"] == "pasta"
    assert chat["memory"][-1]["scope"]["session_id"] != f"S{chat['visit']}"


def test_owner_token_and_revision_protect_persistent_conversation(tmp_path):
    chats = ProfilesChats(storage=tmp_path, runtime=Runtime())
    chat = start(chats)
    with pytest.raises(ProfilesError, match="identidad"):
        chats.handle({"action": "load", "id": chat["id"], "owner_token": "another-owner"})
    with pytest.raises(ProfilesError):
        chats.handle({"action": "load", "id": "../../escape", "owner_token": chat["owner_token"]})
    original = payload(chat, "send", text="Para la bebida, prefiero agua de forma habitual.")
    after = chats.handle(original)
    assert chats.handle(original) == after
    with pytest.raises(ProfilesError, match="otro mensaje"):
        chats.handle({**original, "text": "Prefiero café."})
    with pytest.raises(ProfilesError, match="versión"):
        chats.handle({**original, "request_id": "another"})
    assert "owner_token" not in after
    assert oct((tmp_path / f"{chat['id']}.sqlite").stat().st_mode & 0o777) == "0o600"


def test_failed_inference_leaves_history_memory_and_revision_unchanged(tmp_path):
    runtime = Runtime()
    chats = ProfilesChats(storage=tmp_path, runtime=runtime)
    chat = start(chats)
    runtime.fail = True
    with pytest.raises(ProfilesError):
        chats.handle(payload(chat, "send", text="Para la cena, prefiero pasta."))
    loaded = chats.handle({"action": "load", "id": chat["id"], "owner_token": chat["owner_token"]})
    assert loaded["revision"] == 0 and loaded["messages"] == [] and loaded["memory"] == []


def test_identical_concurrent_requests_commit_once_across_service_instances(tmp_path):
    barrier = Barrier(2)
    one = ProfilesChats(storage=tmp_path, runtime=Runtime(barrier=barrier))
    two = ProfilesChats(storage=tmp_path, runtime=Runtime(barrier=barrier))
    chat = start(one)
    request = payload(chat, "send", text="Para la bebida, prefiero café.")
    with ThreadPoolExecutor(max_workers=2) as workers:
        replies = list(workers.map(lambda service: service.handle(request), (one, two)))
    assert replies[0] == replies[1]
    assert replies[0]["revision"] == 1 and len(replies[0]["messages"]) == 2


def test_today_wording_routine_and_named_others_keep_original_evidence(tmp_path):
    chats = ProfilesChats(storage=tmp_path, runtime=Runtime())
    chat = start(chats)
    chat = send(chats, chat, "Suelo cenar locro de papa. Solo por hoy prefiero sopa de zapallo.")
    assert [record["value"] for record in chat["memory"]] == ["locro de papa", "sopa de zapallo"]
    assert all(record["evidence_quotes"][0]["text"] == chat["messages"][0]["text"] for record in chat["memory"])
    assert chat["messages"][-1]["trace"]["live_wording_adaptations"]
    chat = new_visit(chats, chat, date="2026-12-04")
    chat = send(chats, chat, "¿Qué tengo de cena habitual?")
    assert chat["messages"][-1]["response"]["value"] == "locro de papa"
    previous = deepcopy(chat["memory"])
    chat = send(chats, chat, "Para Ana, prefiero arroz de forma habitual. Mi cena sigue igual.")
    assert chat["memory"] == previous
    assert chat["messages"][-1]["response"]["value"] == "locro de papa"


def test_question_time_and_place_do_not_change_the_current_visit(tmp_path):
    chats = ProfilesChats(storage=tmp_path, runtime=Runtime())
    chat = start(chats)
    chat = send(chats, chat, "Mi cena habitual es pasta.")
    chat = send(chats, chat, "Para la cena, prefiero sopa hasta el 2026-12-03 inclusive.")
    chat = send(chats, chat, "¿Qué ceno mañana?")
    assert chat["date"] == "2026-12-03"
    assert chat["messages"][-1]["trace"]["query"]["date"] == "2026-12-04"
    assert chat["messages"][-1]["response"]["value"] == "pasta"
    chat = send(chats, chat, "¿Qué ceno el 2026-12-04?")
    assert chat["messages"][-1]["response"]["value"] == "pasta"
    chat = new_visit(chats, chat, date="2026-12-05", context="oficina")
    chat = send(chats, chat, "En la oficina, prefiero arroz de forma habitual.")
    chat = send(chats, chat, "¿Qué ceno en casa?")
    assert chat["context"] == "oficina"
    assert chat["messages"][-1]["trace"]["query"]["context"] == "casa"
    assert chat["messages"][-1]["response"]["value"] == "pasta"


def test_different_attribute_can_use_remaining_clarification_budget(tmp_path):
    chats = ProfilesChats(storage=tmp_path, runtime=Runtime())
    chat = start(chats)
    chat = send(chats, chat, "Para la cena, prefiero pasta.")
    chat = send(chats, chat, "Que sea de forma habitual.")
    assert chat["questions_remaining"] == 1
    chat = send(chats, chat, "Para la bebida, prefiero agua mineral.")
    assert chat["pending"]["slot"] == "TEMPORAL_SCOPE"
    assert chat["pending"]["attribute"] == "drink_preference"
    assert chat["questions_remaining"] == 0


def test_mixed_statement_answers_the_final_query_attribute(tmp_path):
    chats = ProfilesChats(storage=tmp_path, runtime=Runtime())
    chat = start(chats)
    chat = send(chats, chat, "Mi cena habitual es pasta.")
    chat = send(chats, chat, "Mi bebida habitual es agua mineral. ¿Qué cena tengo?")
    assert chat["messages"][-1]["response"]["value"] == "pasta"
    assert {record["attribute"] for record in chat["memory"]} == {"dinner_preference", "drink_preference"}


def test_current_visit_can_recall_a_provisional_choice_without_persisting_it(tmp_path):
    class Proceed(Runtime):
        def decide(self, model, observation, legal):
            return "PROCEED", {"source": "fixture", "tokens": 1, "model": model}

    chats = ProfilesChats(storage=tmp_path, runtime=Proceed())
    chat = start(chats)
    chat = send(chats, chat, "Para la cena, prefiero pasta.")
    assert chat["memory"] == []
    chat = send(chats, chat, "¿Qué ceno hoy?")
    assert chat["messages"][-1]["response"]["value"] == "pasta"
    assert chat["memory"] == []
