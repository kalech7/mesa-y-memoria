from copy import deepcopy
from uuid import uuid4

import pytest

from web_demo.food_chat import FinalFoodChats
from web_demo.profiles_runtime import ProfilesError


class Trial:
    def __init__(self, folder, decisions=()):
        self.decisions = iter(decisions)
        self.public = []
        self.chat = FinalFoodChats(self, folder)
        self.state = self.chat.handle({"action": "start", "model": "SFT-ramas", "date": "2026-11-18"})
        self.owner = self.state["owner_token"]

    def describe_model(self, arm):
        return {"arm": arm, "seed": 101, "round": 2, "label": arm}

    def score_public(self, arm, observation):
        self.public.append(deepcopy(observation))
        selected = next(self.decisions)
        return {"action": selected, "source": "qwen", "margin": 1.0 if selected == "ASK" else -1.0,
                "model": self.describe_model(arm), "device": "test"}

    def act(self, action, **extra):
        self.state = self.chat.handle({"action": action, "id": self.state["id"], "owner_token": self.owner,
                                      "revision": self.state["revision"], "request_id": uuid4().hex, **extra})
        return self.state

    def say(self, text):
        return self.act("send", text=text)["messages"][-1]


@pytest.mark.parametrize(("confirmation", "scope", "later"), [
    ("Sí, guárdala como habitual.", "ongoing", "sopa"),
    ("Es mi preferencia habitual.", "ongoing", "sopa"),
    ("Solo por hoy.", "interval", None),
    ("Solo para esta visita.", "session", None),
    ("Todavía no sé.", None, None),
    ("No quiero que la guardes.", None, None),
])
def test_live_clarification_cites_only_the_actual_reply(tmp_path, confirmation, scope, later):
    trial = Trial(tmp_path, ["ASK"])
    opening = "Hoy quiero sopa para cenar. Estoy armando el menú de toda la semana."
    assert trial.say(opening)["trace"]["action"] == "ASK"
    assert trial.state["questions_remaining"] == 0 and trial.state["memory"] == []
    assert trial.public[0]["messages"] == [{"role": "user", "text": opening}]
    trial.say(confirmation)
    if scope:
        record, = trial.state["memory"]
        assert (record["attribute"], record["value"], record["scope"]["kind"]) == ("dinner_preference", "sopa", scope)
        assert [item["text"] for item in record["evidence_quotes"]] == [opening, confirmation]
    else:
        assert trial.state["memory"] == []
    trial.act("new_visit", date="2026-11-19")
    answer = trial.say("¿Qué cena tengo guardada?")
    assert answer["response"]["value"] == later
    assert answer["trace"]["model_used"] is False
    if later:
        assert answer["citations"][0]["source"] == "sqlite"
    if scope:
        assert trial.state["memory"][0]["applicable"] is bool(later)
    assert len(trial.public) == 1


def test_wait_reserves_question_then_uses_actual_second_visit_and_first_agenda(tmp_path):
    trial = Trial(tmp_path, ["WAIT", "ASK"])
    opening = "Hoy quiero sopa para cenar. La cena de hoy ya está resuelta. Mañana elegimos los desayunos de toda la semana."
    saved = trial.say(opening)
    assert saved["trace"]["action"] == "WAIT" and saved["response"]["value"] == "sopa"
    assert trial.state["attribute"] == "dinner_preference"
    assert trial.state["memory"] == [] and trial.state["questions_remaining"] == 1
    trial.act("new_visit", date="2026-11-19")
    today = "Para el desayuno prefiero pan con queso."
    assert trial.say(today)["trace"]["action"] == "ASK"
    assert [m["text"] for m in trial.public[1]["messages"] if m["role"] == "user"] == [opening, today]
    trial.say("De ahora en adelante.")
    assert [(r["attribute"], r["value"]) for r in trial.state["memory"]] == [("breakfast_preference", "pan con queso")]
    assert trial.state["questions_remaining"] == 0


def test_catalog_respects_constraints_recalls_evidence_and_keeps_recipes_out_of_memory(tmp_path):
    trial = Trial(tmp_path)
    preference = "Mi cena habitual es pescado al horno con papas."
    trial.say(preference)
    assert trial.state["questions_remaining"] == 1
    recommended = trial.say("¿Qué puedo cenar?")
    assert recommended["recommendation"]["chosen"] == "Pescado al horno con papas"
    assert recommended["citations"][0]["quotes"][0]["text"] == preference
    recipe = trial.say("¿Cómo lo preparo?")
    assert recipe["trace"]["source"] == "recipe" and "horno" in recipe["text"].lower()
    alternative = trial.say("Dame otra opción sin huevo.")
    assert alternative["recommendation"]["chosen"] != recommended["recommendation"]["chosen"]
    assert "huevo" not in alternative["recommendation"]["ingredients"]
    assert alternative["citations"] == []
    breakfast = trial.say("¿Qué puedo desayunar sin leche?")
    assert breakfast["recommendation"]["attribute"] == "breakfast_preference"
    assert "leche" not in breakfast["recommendation"]["ingredients"]
    drink = trial.say("¿Qué bebida me recomiendas?")
    assert drink["recommendation"]["attribute"] == "drink_preference"
    assert len(trial.state["memory"]) == 1 and not trial.public


def test_habit_and_calendar_exception_can_be_reopened_without_demo_data(tmp_path):
    trial = Trial(tmp_path)
    trial.say("Suelo cenar locro de papa. Solo por hoy prefiero sopa de zapallo.")
    assert len(trial.state["memory"]) == 2
    trial.act("new_visit", date="2026-11-19")
    trial.chat = FinalFoodChats(trial, tmp_path)
    assert trial.say("¿Qué cena habitual tengo guardada?")["response"]["value"] == "locro de papa"
    assert not next(r for r in trial.state["memory"] if r["value"] == "sopa de zapallo")["applicable"]
    with pytest.raises(ProfilesError):
        trial.chat.handle({"action": "load", "id": trial.state["id"], "owner_token": "otro"})


def test_third_party_and_exclusions_never_become_personal_preferences(tmp_path):
    trial = Trial(tmp_path)
    trial.say("Para mi hermano, quiero sopa de forma habitual.")
    trial.say("Para Ana, prefiero arroz de forma habitual.")
    trial.say("Evita la cebolla.")
    dinner = trial.say("Dame una cena.")
    assert "cebolla" in dinner["recommendation"]["excluded"]
    assert "cebolla" not in dinner["recommendation"]["ingredients"]
    assert not trial.state["memory"] and not trial.public


@pytest.mark.parametrize("question", [
    "¿Qué cena tengo guardada mañana?", "Qué cena tengo guardada mañana",
    "que cena tengo guardada mañana", "Recuerdas qué prefiero para cenar mañana",
    "Y para mañana, qué cena tengo guardada",
])
def test_recall_of_tomorrow_excludes_today_only_exception(tmp_path, question):
    trial = Trial(tmp_path)
    trial.say("Mi cena habitual es locro de papa.")
    trial.say("Solo por hoy prefiero sopa de zapallo para la cena.")
    memory = deepcopy(trial.state["memory"])
    reply = trial.say(question)
    assert reply["response"]["value"] == "locro de papa"
    assert reply["trace"]["query"]["date"] == "2026-11-19"
    assert trial.state["memory"] == memory and trial.state["date"] == "2026-11-18"


def test_unpunctuated_recall_interrupts_a_clarification_without_confirming_it(tmp_path):
    trial = Trial(tmp_path, ["ASK"])
    trial.say("Mi bebida habitual es agua mineral.")
    assert trial.say("Hoy quiero sopa para cenar.")["trace"]["action"] == "ASK"
    reply = trial.say("Qué bebida tengo guardada")
    assert reply["response"]["value"] == "agua mineral"
    assert trial.state["pending"] is None and trial.state["questions_remaining"] == 0
    assert [(r["attribute"], r["value"]) for r in trial.state["memory"]] == [
        ("drink_preference", "agua mineral"),
    ]


@pytest.mark.parametrize("alternative", ["Dame otra opción.", "Dame otra opción para cenar."])
def test_alternative_after_recipe_keeps_target_day_and_restrictions(tmp_path, alternative):
    trial = Trial(tmp_path)
    first = trial.say("¿Qué puedo cenar mañana sin pescado?")["recommendation"]
    assert first["date"] == "2026-11-19"
    trial.say("¿Cómo lo preparo?")
    second = trial.say(alternative)["recommendation"]
    assert (second["date"], second["attribute"]) == (first["date"], first["attribute"])
    assert second["chosen"] != first["chosen"]
    assert second["excluded"] == ["pescado"] and "pescado" not in second["ingredients"]


def test_write_confirmation_is_distinct_from_applicable_read_and_ingredient_match(tmp_path):
    trial = Trial(tmp_path)
    trial.say("Solo por hoy prefiero sopa para la cena.")
    habit = "Mi cena habitual es locro de papa."
    confirmed = trial.say(habit)
    assert confirmed["response"] is None
    assert "Guardé locro de papa" in confirmed["text"] and "sopa" not in confirmed["text"]
    assert [q["text"] for c in confirmed["citations"] for q in c["quotes"]] == [habit]
    assert trial.say("¿Qué cena tengo guardada hoy?")["response"]["value"] == "sopa"
    trial.act("new_visit", date="2026-11-19")
    exact = trial.say("¿Qué puedo cenar hoy?")["recommendation"]
    assert (exact["chosen"], exact["reason"]) == ("locro de papa", "remembered_dish")
    mixed = trial.say("¿Qué puedo cenar sin pescado?")
    assert mixed["recommendation"]["reason"] == "shared_ingredients"
    assert mixed["recommendation"]["shared_ingredients"] == ["papa"]
    assert "Comparte papa con tu preferencia guardada de locro de papa." in mixed["text"]


@pytest.mark.parametrize("request_text", [
    "¿Cómo preparo la tortilla de papa y cebolla sin huevo?", "¿Cómo lo preparo sin huevo?",
])
def test_recipe_rechecks_ingredients_and_greeting_costs_nothing(tmp_path, request_text):
    trial = Trial(tmp_path)
    welcome = trial.say("¡Hola!")
    assert "¿Qué te gustaría comer hoy?" in welcome["text"] and welcome["trace"]["selected_action"] == "GREET"
    trial.say("Dame una cena.")
    recipe = trial.say(request_text)
    assert recipe["trace"]["source"] == "recipe"
    assert "lleva huevo" in recipe["text"] and "Mézclalas con huevo" not in recipe["text"]
    next_dish = trial.say("Dame otra opción.")["recommendation"]
    assert "huevo" in next_dish["excluded"] and "huevo" not in next_dish["ingredients"]
    assert not trial.state["memory"] and trial.state["questions_remaining"] == 1 and not trial.public
