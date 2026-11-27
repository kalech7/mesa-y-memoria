"""Independent free-text checks for the live chat, without loading Qwen weights."""
from copy import deepcopy
from uuid import uuid4

import pytest

from research.tesis_profiles_parser_v5 import public_evidence
from web_demo.profiles_chat import ProfilesChats
from web_demo.profiles_runtime import ProfilesError


class ControlledRuntime:
    def __init__(self, *, passive=False):
        self.passive = passive
        self.calls = []

    def resolve(self, identity=None):
        return {"id": "qa-controlled", "method": "QA", "available": True,
                "eligible": True, "seed": 0, "epoch": 0, "panel": "QA"}

    def decide(self, model, observation, legal):
        self.calls.append(deepcopy(observation))
        slot = public_evidence(observation)["missing_slot"]
        action = slot if not self.passive and slot in legal else "PROCEED"
        return action, {"source": "controlled_qa", "tokens": 0, "model": model}


class Conversation:
    def __init__(self, folder, *, passive=False, context=None):
        self.runtime = ControlledRuntime(passive=passive)
        self.service = ProfilesChats(storage=folder, runtime=self.runtime)
        self.state = self.service.handle({"action": "start", "date": "2026-12-03", "context": context})
        self.token = self.state["owner_token"]

    def request(self, action, **values):
        self.state = self.service.handle({
            "action": action, "id": self.state["id"], "owner_token": self.token,
            "revision": self.state["revision"], "request_id": uuid4().hex, **values,
        })
        return self.state

    def send(self, text):
        return self.request("send", text=text)

    @property
    def response(self):
        return self.state["messages"][-1]["response"]


@pytest.mark.parametrize("target,expected_date", [("mañana", "2026-12-04"),
                                                 ("el 2026-12-04", "2026-12-04"),
                                                 ("pasado mañana", "2026-12-05")])
def test_query_date_can_differ_from_visit_date(tmp_path, target, expected_date):
    chat = Conversation(tmp_path)
    chat.send("Mi cena habitual es pasta.")
    chat.send("Para la cena prefiero sopa hasta el 2026-12-03 inclusive.")
    chat.send(f"¿Qué ceno {target}?")
    assert chat.response["value"] == "pasta"
    assert chat.state["date"] == "2026-12-03"
    assert chat.state["messages"][-1]["trace"]["query"]["date"] == expected_date


def test_invalid_calendar_query_is_rejected_without_mutating_chat(tmp_path):
    chat = Conversation(tmp_path)
    chat.send("Mi cena habitual es pasta.")
    previous = deepcopy(chat.state)
    with pytest.raises(ProfilesError) as error:
        chat.send("¿Qué ceno el 2026-02-30?")
    assert error.value.status == 400
    loaded = chat.service.handle({"action": "load", "id": chat.state["id"], "owner_token": chat.token})
    assert loaded == previous


def test_query_context_can_differ_from_current_location(tmp_path):
    chat = Conversation(tmp_path, context="oficina")
    chat.send("Mi cena habitual es pasta.")
    chat.send("Para la cena en la oficina prefiero arroz de forma habitual.")
    chat.send("¿Qué ceno en casa?")
    assert chat.response["value"] == "pasta"
    assert chat.state["context"] == "oficina"


def test_same_slot_for_new_attribute_uses_available_question(tmp_path):
    chat = Conversation(tmp_path)
    chat.send("Para la cena prefiero pasta.")
    chat.send("Guárdala como mi cena habitual.")
    assert chat.state["questions_remaining"] == 1
    chat.send("Para la bebida prefiero agua mineral.")
    assert chat.state["pending"]["slot"] == "TEMPORAL_SCOPE"
    assert chat.state["questions_remaining"] == 0


def test_named_recipient_does_not_replace_own_preference(tmp_path):
    chat = Conversation(tmp_path)
    chat.send("Mi cena habitual es pasta.")
    chat.send("Para Ana, prefiero arroz de forma habitual. Mi cena sigue igual.")
    assert not any(record["value"] == "arroz" for record in chat.state["memory"])
    chat.request("new_visit", date="2026-12-04")
    chat.send("¿Qué cena habitual tengo guardada?")
    assert chat.response["value"] == "pasta"


def test_current_choice_remains_known_in_same_visit_without_durable_write(tmp_path):
    chat = Conversation(tmp_path, passive=True)
    chat.send("Para la cena prefiero pasta.")
    assert chat.response["value"] == "pasta"
    assert chat.state["memory"] == []
    chat.send("¿Qué ceno hoy?")
    assert chat.response["value"] == "pasta"
    assert chat.state["memory"] == []


def test_unknown_duration_keeps_future_alternatives(tmp_path):
    chat = Conversation(tmp_path, passive=True)
    chat.send("Mi cena habitual es locro de papa.")
    chat.send("Para la cena prefiero pasta.")
    chat.send("¿Qué ceno mañana?")
    assert chat.response["mode"] == "CONDITIONALIZE"
    assert {branch["value"] for branch in chat.response["branches"]} == {"locro de papa", "pasta"}
    assert not any(record["value"] == "pasta" for record in chat.state["memory"])


def test_final_question_selects_attribute_independently_of_new_write(tmp_path):
    chat = Conversation(tmp_path)
    chat.send("Mi cena habitual es pasta.")
    chat.send("Mi bebida habitual es agua mineral. ¿Qué cena tengo?")
    assert chat.response["value"] == "pasta"
    assert any(record["attribute"] == "drink_preference" and record["value"] == "agua mineral"
               for record in chat.state["memory"])


def test_today_prefix_is_saved_as_a_bounded_exception(tmp_path):
    chat = Conversation(tmp_path)
    chat.send("Mi cena habitual es pasta.")
    chat.send("Solo por hoy prefiero sopa para la cena.")
    assert chat.state["pending"] is None
    soup = next(record for record in chat.state["memory"] if record["value"] == "sopa")
    assert soup["scope"]["valid_until_exclusive"] == "2026-12-04"
    chat.request("new_visit", date="2026-12-04")
    chat.send("¿Qué cena habitual tengo?")
    assert chat.response["value"] == "pasta"


def test_ui_example_recovers_habit_after_today_exception(tmp_path):
    chat = Conversation(tmp_path)
    chat.send("Suelo cenar locro de papa. Solo por hoy prefiero sopa de zapallo.")
    chat.request("new_visit", date="2026-12-04")
    chat.send("¿Qué tengo de cena habitual?")
    assert chat.response["value"] == "locro de papa"


def test_partial_followup_spends_shared_budget_once_per_question(tmp_path):
    chat = Conversation(tmp_path)
    chat.send("Para la cena prefiero pasta.")
    assert chat.state["questions_remaining"] == 1
    chat.send("Por ahora sí. Dame un momento para mirar el alcance.")
    assert chat.state["questions_remaining"] == 0
    chat.send("Guarda pasta como mi cena habitual.")
    chat.request("new_visit", date="2026-12-04")
    chat.send("Para la bebida prefiero agua mineral.")
    assert chat.state["questions_remaining"] == 0
    assert chat.state["pending"] is None


@pytest.mark.parametrize("initial,text,attribute,value,question", [
    ("Mi cena habitual es mote con queso.",
     "Anota una rectificación para mí: mi cena habitual ahora es menestra de fréjol. "
     "El mote con queso lo preparo para mi sobrino desde hoy.",
     "dinner_preference", "menestra de fréjol", "¿Qué cena habitual tengo guardada?"),
    ("Mi bebida habitual es café pasado.",
     "Registra esta actualización en mi ficha: mi bebida habitual ahora es agua de cedrón. "
     "El jugo de tomate lo llevo para mi sobrina desde hoy.",
     "drink_preference", "agua de cedrón", "¿Qué bebida habitual tengo guardada?"),
])
def test_instruction_prefix_uses_explicit_preference_and_keeps_original_quote(
        tmp_path, initial, text, attribute, value, question):
    chat = Conversation(tmp_path)
    chat.send(initial)
    chat.send(text)
    assert chat.response["value"] == value
    reply = chat.state["messages"][-1]
    writes = [operation["record"] for operation in reply["operations"] if "record" in operation]
    assert len(writes) == 1
    assert writes[0]["attribute"] == attribute and writes[0]["value"] == value
    assert writes[0]["scope"]["kind"] == "ongoing"
    assert writes[0]["evidence_quotes"] == [{"turn_id": chat.state["messages"][-2]["turn_id"], "text": text}]
    adaptations = reply["trace"]["live_wording_adaptations"]
    assert any(item["kind"] == "instruction_prefix_before_explicit_preference"
               and item["prefix"].endswith(":") for item in adaptations)
    chat.request("new_visit", date="2026-12-04")
    chat.send(question)
    assert chat.response["value"] == value
    assert any(citation["source"] == "sqlite" for citation in chat.state["messages"][-1]["citations"])


def test_direct_instruction_with_a_food_retains_its_meaning(tmp_path):
    chat = Conversation(tmp_path)
    chat.send("Anota arroz como mi cena habitual.")
    assert chat.response["value"] == "arroz"
    assert chat.state["memory"][-1]["value"] == "arroz"
    assert not any(item["kind"] == "instruction_prefix_before_explicit_preference"
                   for item in chat.state["messages"][-1]["trace"]["live_wording_adaptations"])
