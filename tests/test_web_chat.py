from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from web_demo.chat import ChatError, Chats
from web_demo.recommendations import meal_request
from research.src.policy.llm_controller import InvalidOutput


class Actor:
    def __init__(self):
        self.phase_records = []
        self.backend = self
        self.observations = []

    def prompt(self, phase, observation):
        assert all("trace" not in message for message in observation["user_messages"])
        return phase

    def encode_prompt(self, prompt):
        return [1]

    def record(self, observation, action):
        self.observations.append(observation)
        self.phase_records.append({"raw_output": str(action)})
        return action

    def select(self, observation):
        ask = "Prefiero pasta." == observation["user_messages"][-1]["text"]
        return self.record(observation, {"action": "ASK" if ask else "PROCEED",
                                         "slot": "TEMPORAL_SCOPE" if ask else None})

    def respond(self, observation):
        self.record(observation, "malformed")
        raise InvalidOutput("Malformed response")

    def recommendation(self, observation, choices):
        # A syntactically valid but semantically incompatible model answer.
        return self.record(observation, {"mode": "ANSWER", "value": "polloo"})

    def write(self, observation):
        return self.record(observation, {"operations": [{"op": "NOOP"}]})


class Local:
    def __init__(self):
        self.actor = Actor()

    def status(self):
        return True, None

    @contextmanager
    def chat_controller(self, model):
        yield self.actor


class MealPolicy:
    def __init__(self, dish="Pescado al horno con papas", ingredients=("pescado", "papa")):
        self.dish = dish
        self.ingredients = list(ingredients)
        self.turns = []

    def choose(self, turns, attribute):
        self.turns.append((list(turns), attribute))
        return {"name": self.dish, "ingredients": self.ingredients}, {"phase": "recommendation", "raw": self.dish}


def test_chat_uses_the_real_reply_and_persisted_memory():
    local = Local()
    chats = Chats(local)
    try:
        chat = chats.handle({"action": "start", "model": "grpo"})

        def turn(chat, action, text=""):
            return {"action": action, "id": chat["id"], "revision": chat["revision"],
                    "request_id": str(chat["revision"]), "text": text}

        chat = chats.handle(turn(chat, "send", "Prefiero pasta."))
        assert chat["pending"]
        assert chat["memory"] == []
        payload = turn(chat, "send", "Es una preferencia habitual.")
        chat = chats.handle(payload)
        assert chats.handle(payload) == chat
        assert len(chat["memory"]) == 1
        assert chat["memory"][0]["value"] == "pasta"
        assert chat["messages"][2]["in_reply_to"] == chat["messages"][1]["turn_id"]
        assert chat["messages"][-1]["confirmed_write"]
        with pytest.raises(ChatError, match="versión más reciente"):
            chats.handle({**payload, "request_id": "stale"})
        chat = chats.handle(turn(chat, "new_session"))
        assert chat["messages"] == []
        assert chat["session"] == 2
        chat = chats.handle(turn(chat, "send", "¿Qué prefiero para cenar?"))
        assert chat["messages"][-1]["text"] == "pasta"
        assert chat["messages"][-1]["memory_read"]["source"] == "sqlite"
        assert [step["phase"] for step in chat["messages"][-1]["trace"]] == ["select", "respond"]
        assert len(chat["memory"]) == 1
        assert chats.handle({"action": "start", "model": "sft"})["memory"] == []
    finally:
        chats.directory.cleanup()


def test_simulated_history_is_isolated_and_uses_calendar_scope():
    clock = ["2026-12-03"]
    local = Local()
    policy = local.actor.meal_policy = MealPolicy()
    chats = Chats(local, clock=lambda: clock[0])
    try:
        regular = chats.handle({"action": "start", "model": "sft"})
        demo = chats.handle({"action": "start", "model": "sft", "simulated": True})
        assert demo["id"] != regular["id"] and regular["memory"] == []
        assert demo["simulated"] and demo["session"] == 3 and demo["messages"] == []
        assert all(session["simulated"] for session in demo["past_sessions"])
        assert {ref for record in demo["memory"] for ref in record["source_turn_ids"]} <= {
            message["turn_id"] for session in demo["past_sessions"] for message in session["messages"]
            if message["role"] == "user"
        }

        demo = send_chat(chats, demo, "¿Qué puedo cenar hoy?")
        reply = demo["messages"][-1]["recommendation"]
        assert reply["reason"] == "remembered_preference"
        assert reply["chosen"] == "Arroz con lentejas y zanahoria"
        assert reply["memory_used"][0]["memory_id"] == "demo-m3"
        assert policy.turns

        demo = send_chat(chats, demo, "¿Qué puedo cenar hoy sin arroz, lentejas ni tomate?")
        reply = demo["messages"][-1]["recommendation"]
        assert reply["excluded"] == ["arroz", "lenteja", "tomate"]
        assert "memory_used" not in reply

        demo = chats.handle({"action": "new_session", "id": demo["id"], "revision": demo["revision"],
                             "request_id": "archive"})
        assert demo["messages"] == [] and not demo["past_sessions"][-1]["simulated"]
        demo = send_chat(chats, demo, "¿Qué prefiero para cenar hoy?")
        assert demo["messages"][-1]["memory_read"]["evidence_ids"] == ["demo-m3"]
        clock[0] = "2026-12-04"
        demo = send_chat(chats, demo, "¿Qué prefiero para cenar hoy?")
        assert demo["messages"][-1]["memory_read"]["evidence_ids"] == ["demo-m1"]
    finally:
        chats.directory.cleanup()


CASES = json.loads(Path(__file__).with_name("fixtures").joinpath("chat_recommendations.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_recommendation_cases(case):
    history = [{"role": "user", "text": text} for text in case["turns"]]
    if case.get("answering"):
        history[-1]["answers_food_exclusion"] = True
    request = meal_request(history, case["attribute"], follow_up=case.get("follow_up", False))
    if case.get("memory"):
        assert request is None
        return
    assert request.excluded == set(case["excluded"])
    assert ("¿" in request.replacement()) == case.get("clarify", False)
    for dish, ingredients in request.suggestions:
        assert request.accepts(dish)
        assert not request.excluded.intersection(ingredients)


def test_live_chat_blocks_conflicting_model_answer_and_leaves_memory_untouched():
    chats = Chats(Local())
    try:
        chat = chats.handle({"action": "start", "model": "grpo"})
        result = chats.handle({"action": "send", "id": chat["id"], "revision": 0,
                               "request_id": "captura", "verify": False, "text": CASES[0]["turns"][0]})
        reply = result["messages"][-1]
        assert "pollo" not in reply["text"]
        assert reply["recommendation"] == {"source": "catalog", "excluded": ["pollo"]}
        assert reply["trace"][0]["action"]["value"] == "polloo"
        assert result["memory"] == []
        result = chats.handle({"action": "send", "id": chat["id"], "revision": 1,
                               "request_id": "follow-up", "text": "Tampoco quiero huevo"})
        assert result["messages"][-1]["recommendation"]["excluded"] == ["huevo", "pollo"]
        new = chats.handle({"action": "new_session", "id": chat["id"], "revision": 2, "request_id": "new"})
        result = chats.handle({"action": "send", "id": chat["id"], "revision": new["revision"],
                               "request_id": "next-query", "text": "¿Qué puedo comer hoy?"})
        assert result["messages"][-1]["recommendation"]["excluded"] == []
        assert result["memory"] == []
    finally:
        chats.directory.cleanup()


SCREENSHOT_REQUEST = "ayer comi pollo no quiero comer poll quiero agl orico para comer hoy"
SCREENSHOT_FOLLOW_UP = "para esta ocacion"
PLURAL_REQUEST = "ayer comi pollo no quiero pollo comer hoy recomiendame cosas para comer"
PLURAL_FOLLOW_UP = "solo por hoy"


def send_chat(chats, chat, text, *, verify=True):
    return chats.handle({"action": "send", "id": chat["id"], "revision": chat["revision"],
                         "request_id": str(chat["revision"]), "verify": verify, "text": text})


@pytest.mark.parametrize(("text", "excluded"), [
    ("No quiero comer poll ni huev quiero algo rico para cenar", {"pollo", "huevo"}),
    ("esta noche no quiero pollo, qiero agl para comer", {"pollo"}),
    ("quiero un desayuno sin huevo", {"huevo"}),
    ("quiero algo que no lleve queso", {"queso"}),
    ("Hay algo del menú que prefiero no comer. ¿Qué cena me propones?", set()),
    ("ayer comí pollo quiero algo distinto para comer hoy", {"pollo"}),
    ("no quiero huevo quiero otra cosa para comer hoy", {"huevo"}),
    (PLURAL_REQUEST, {"pollo"}),
    ("no quiero huevo recomiendame cosas para cenar", {"huevo"}),
    ("sugiéreme algunas ideas para desayunar, sin huevo", {"huevo"}),
    ("Para cenar juntas, prefiero que no tenga huevo. ¿Cuál de esos platos podemos compartir?", {"huevo"}),
    ("no   quiero comer algo rico para cenar", None),
    ("Ayer comí pollo. Hoy no quiero pollo. ¿Recuerdas qué prefiero?", None),
])
def test_colloquial_meal_routing_distinguishes_food_exclusions_from_intent(text, excluded):
    request = meal_request([{"role": "user", "text": text}], "dinner_preference")
    if excluded is None:
        assert request is None
    else:
        assert request.excluded == excluded


@pytest.mark.parametrize("verify", [True, False])
@pytest.mark.parametrize("messages", [
    (SCREENSHOT_REQUEST, SCREENSHOT_FOLLOW_UP),
    (PLURAL_REQUEST, PLURAL_FOLLOW_UP),
])
def test_colloquial_request_with_typos_reaches_the_meal_model(verify, messages):
    local = Local()
    policy = local.actor.meal_policy = MealPolicy()
    chats = Chats(local)
    try:
        chat = chats.handle({"action": "start", "model": "grpo"})
        for position, text in enumerate(messages):
            chat = send_chat(chats, chat, text, verify=verify)
            reply = chat["messages"][-1]
            assert [step["phase"] for step in reply["trace"]] == ["recommendation"]
            assert reply["text"] == "Podrías preparar pescado al horno con papas."
            assert reply["recommendation"]["excluded"] == ["pollo"]
            assert reply["recommendation"]["source"] == "model"
            assert reply["recommendation"]["raw"] is not verify
            assert policy.turns[-1] == (list(messages[:position + 1]), "dinner_preference")
            assert not chat["pending"] and chat["memory"] == []
    finally:
        chats.directory.cleanup()


@pytest.mark.parametrize(("verify", "source"), [(True, "catalog"), (False, "model")])
def test_verification_mode_only_changes_whether_the_conflict_is_replaced(verify, source):
    local = Local()
    local.actor.meal_policy = MealPolicy("Pollo con papas", ("pollo", "papa"))
    chats = Chats(local)
    try:
        chat = chats.handle({"action": "start", "model": "grpo"})
        chat = send_chat(chats, chat, SCREENSHOT_REQUEST, verify=verify)
        reply = chat["messages"][-1]
        assert reply["recommendation"]["source"] == source
        assert reply["recommendation"]["model_choice"] == "Pollo con papas"
        assert ("pollo con papas" in reply["text"].lower()) is not verify
        assert chat["memory"] == []
    finally:
        chats.directory.cleanup()


@pytest.mark.parametrize(("text", "verify", "source"), [
    (PLURAL_REQUEST, True, "catalog"),
    (PLURAL_REQUEST, False, "model"),
    ("¿Qué puedo cenar sin papa ni aceite?", True, "model"),
    ("¿Qué puedo cenar sin papa ni aceite?", False, "model"),
])
def test_model_clarification_is_preserved_without_verifier_or_when_no_dish_fits(text, verify, source):
    local = Local()
    local.actor.meal_policy = MealPolicy("Pedir una aclaración", ())
    chats = Chats(local)
    try:
        chat = chats.handle({"action": "start", "model": "grpo"})
        chat = send_chat(chats, chat, text, verify=verify)
        reply = chat["messages"][-1]
        assert reply["recommendation"]["source"] == source
        assert reply["recommendation"]["model_choice"] == "Pedir una aclaración"
        assert reply["recommendation"]["raw"] is not verify
        if source == "catalog":
            assert reply["recommendation"]["reason"] == "unnecessary_clarification"
            assert "pollo" not in reply["text"]
        else:
            assert reply["text"].startswith("¿")
        assert chat["memory"] == []
    finally:
        chats.directory.cleanup()


def test_verifier_uses_the_dinner_menu_the_model_received():
    local = Local()
    local.actor.meal_policy = MealPolicy()
    chats = Chats(local)
    try:
        chat = chats.handle({"action": "start", "model": "grpo"})
        text = "¿Qué ceno sin papa, sin quinua y sin arroz?"
        reply = send_chat(chats, chat, text)["messages"][-1]
        assert reply["recommendation"]["source"] == "catalog"
        assert reply["recommendation"]["model_choice"] == "Pescado al horno con papas"
        assert reply["text"].startswith("¿")
    finally:
        chats.directory.cleanup()


@pytest.mark.parametrize("follow_up", [
    "Ya conseguí papas, el aceite sigue fuera. ¿Qué ceno?",
    "Ya conseguí papas y el aceite sigue fuera. ¿Qué ceno?",
])
def test_verifier_recommends_the_only_dish_after_a_user_finds_an_ingredient(follow_up):
    local = Local()
    local.actor.meal_policy = MealPolicy("Pedir una aclaración", ())
    chats = Chats(local)
    try:
        chat = chats.handle({"action": "start", "model": "grpo"})
        chat = send_chat(chats, chat, "¿Qué ceno hoy sin aceite ni papas?")
        assert chat["messages"][-1]["text"].startswith("¿")
        chat = send_chat(chats, chat, follow_up)
        assert chat["messages"][-1]["text"] == "Podrías preparar pollo con papas."
        assert chat["messages"][-1]["recommendation"]["excluded"] == ["aceite"]
        assert chat["memory"] == []
    finally:
        chats.directory.cleanup()


@pytest.mark.parametrize("messages", [
    (SCREENSHOT_REQUEST, SCREENSHOT_FOLLOW_UP),
    (PLURAL_REQUEST, PLURAL_FOLLOW_UP),
])
def test_scope_reply_recovers_a_meal_request_from_existing_pending_state(messages):
    local = Local()
    policy = local.actor.meal_policy = MealPolicy()
    chats = Chats(local)
    try:
        chat = chats.handle({"action": "start", "model": "grpo"})
        before = chats.sessions[chat["id"]]
        before.pending = "a_tool_1_1"
        before.history = [
            {"turn_id": "u1_1", "role": "user", "text": messages[0],
             "date": datetime.now(ZoneInfo("America/Guayaquil")).date().isoformat()},
            {"turn_id": before.pending, "role": "assistant",
             "text": "¿Lo dices solo para esta ocasión o como preferencia habitual?"},
        ]
        chat = send_chat(chats, chat, messages[1])
        assert policy.turns == [(list(messages), "dinner_preference")]
        assert chat["messages"][-1]["text"] == "Podrías preparar pescado al horno con papas."
        assert chat["messages"][-1]["recommendation"]["excluded"] == ["pollo"]
        assert not chat["pending"] and chat["memory"] == []
    finally:
        chats.directory.cleanup()


@pytest.mark.parametrize("verify", [True, False])
def test_legacy_grpo_keeps_a_plural_meal_request_and_its_temporal_reply_in_recommendations(verify):
    chats = Chats(Local())
    try:
        chat = chats.handle({"action": "start", "model": "grpo"})
        for text in (PLURAL_REQUEST, PLURAL_FOLLOW_UP):
            chat = send_chat(chats, chat, text, verify=verify)
            reply = chat["messages"][-1]
            assert [step["phase"] for step in reply["trace"]] == ["recommendation"]
            assert reply["recommendation"]["excluded"] == ["pollo"]
            assert not chat["pending"] and chat["memory"] == []
    finally:
        chats.directory.cleanup()


def test_plain_exclusion_and_actual_preference_keep_their_original_routes():
    assert meal_request([{"role": "user", "text": "Hoy no quiero comer pollo."}], "dinner_preference") is None
    local = Local()
    policy = local.actor.meal_policy = MealPolicy()
    chats = Chats(local)
    try:
        chat = chats.handle({"action": "start", "model": "grpo"})
        chat = send_chat(chats, chat, "Prefiero pasta.")
        assert chat["pending"]
        chat = send_chat(chats, chat, "Es una preferencia habitual.")
        assert not chat["pending"] and [row["value"] for row in chat["memory"]] == ["pasta"]
        assert policy.turns == []
    finally:
        chats.directory.cleanup()
