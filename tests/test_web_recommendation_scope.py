import pytest

from web_demo.recommendations import meal_date, meal_request, preference_ingredients


DINNER = "dinner_preference"
BREAKFAST = "breakfast_preference"
TODAY = "2026-11-14"
TOMORROW = "2026-11-15"


def request(*texts, attribute=DINNER, date=TODAY, **extra):
    history = [{"role": "user", "text": text, "date": date} for text in texts]
    return meal_request(history, attribute, date=date, follow_up=len(texts) > 1, **extra)


@pytest.mark.parametrize("text", [
    "¿Qué puedo cenar sin arroz, lentejas y tomate?",
    "Sin arroz / lentejas / tommate, ¿qué ceno?",
    "Quiero una cena sin arroz, lentejas, fuera el tomate también. ¿Qué ceno?",
])
def test_list_punctuation_keeps_each_ingredient(text):
    assert request(text).excluded == {"arroz", "lenteja", "tomate"}


@pytest.mark.parametrize("text", [
    "Yo hoy no quiero que el plato lleve poyo, papa, ni tomate. ¿Qué puedo cenar?",
    "Para la cena no queremos que la receta contenga pollo, papas ni tomate. ¿Qué ceno?",
    "Quiero una cena que no incluya poyo, papa ni tomate.",
])
def test_subordinate_clause_conveys_a_food_exclusion(text):
    assert request(text).current_excluded == {"pollo", "papa", "tomate"}


def test_transposed_ingredient_in_list_and_imperative_keep_the_right_menu_state():
    menu = (("Pollo con papas", "pollo papa"), ("Arroz con lentejas", "arroz lenteja aceite"),
            ("Ensalada de quinua", "quinua pepino aceite"))
    first = "Hoy, para mi plato, sin papas, aciete, ni poyo. ¿Qué cena me propones?"
    flagged = request(first, dishes=menu)
    assert flagged.excluded == {"papa", "aceite", "pollo"}
    assert flagged.choices == []
    next_turn = "Ya encontré aceite. Para mi cena deja fuera todavía papa y pollo, y tampoco metas pepino. ¿Qué ceno?"
    reopened = request(first, next_turn, dishes=menu)
    assert reopened.excluded == {"papa", "pepino", "pollo"}
    assert reopened.choices == ["Arroz con lentejas"]


def test_memory_uncertainty_does_not_make_the_following_ingredient_list_ambiguous():
    parsed = request("¿Qué prefiero para cenar? Todavía no te he contado mis gustos.",
                     "No quiero que guardes un gusto ahora. Hoy para cenar nada de arrós, pepino, huevo, ni pollo. ¿Qué me recomiendas?")
    assert parsed.excluded == {"arroz", "pepino", "huevo", "pollo"}
    assert not parsed.unspecified
    assert "Pescado al horno con papas" in parsed.choices


def test_autocorrection_adds_and_removes_individual_restrictions():
    first = "¿Qué ceno sin tomate, cebolla y peskao?"
    second = "Tomate sí*, cebolla sigue fuera. Tampoco huevo. ¿Qué ceno?"
    third = "El pescado se puede usar ahora. Cebolla sigue fuera; huevo tampoco. ¿Qué ceno?"
    assert request(first, second).excluded == {"cebolla", "pescado", "huevo"}
    assert request(first, second, third).excluded == {"cebolla", "huevo"}


def test_nobody_inherits_someone_elses_restriction_until_they_share():
    separated = "Mi hermano pidió sin tomate, pero yo puedo comerlo. ¿Qué me recomiendas para cenar?"
    together = "Mi hermana no quiere pescado y yo no quiero tomate. ¿Qué cenamos juntos?"
    apart = "Se queda comiendo en su casa, es solo para mí. ¿Qué ceno?"
    assert request(separated).excluded == set()
    assert request(together).excluded == {"pescado", "tomate"}
    assert request(together, apart).excluded == {"tomate"}


def test_date_and_meal_changes_drop_the_previous_meals_restrictions():
    tonight = "¿Qué ceno hoy sin huevo?"
    different_date = request(tonight, "Para mañana quisiera que incluyera huevo, ¿qué puedo cenar?")
    breakfast = request(tonight, "¿Qué desayuno ahora?", attribute=BREAKFAST)
    assert different_date.date == TOMORROW
    assert different_date.excluded == set()
    assert breakfast.date == TODAY
    assert breakfast.excluded == set()


def test_dated_order_carries_over_to_its_date_and_not_the_recording_date():
    history = [{"role": "user", "date": TODAY, "text": "Para mañana sin tomate. ¿Qué puedo cenar mañana?"},
               {"role": "user", "date": TOMORROW, "text": "¿Qué ceno hoy, de lo de ayer?"}]
    assert meal_request(history, DINNER, date=TOMORROW, follow_up=True).excluded == {"tomate"}
    assert meal_date("¿Qué prefiero cenar mañana?", TODAY) == TOMORROW
    assert meal_date("¿Qué prefiero cenar el 2026-11-21?", TODAY) == "2026-11-21"
    assert meal_date("Ayer comí arroz. ¿Qué puedo cenar?", TODAY) == TODAY


def test_today_question_overrides_a_tomorrow_clause_but_lets_followup_keep_future_scope():
    current = request("Hoy cenamos sin papas; para mañana sin pescado. ¿Qué puedo cenar hoy?")
    follow_up = request("Mañana quiero una cena sin aceite ni papas, ¿cuál me recomiendas?",
                        "Ya conseguí papas. Todavía sin aceite. ¿Qué puedo cenar?")
    assert (current.date, current.excluded) == (TODAY, {"papa"})
    assert (follow_up.date, follow_up.excluded) == (TOMORROW, {"aceite"})


def test_unknown_ingredient_and_real_followup_are_separate_from_an_open_choice():
    first = "Quiero una cena sin uno de los ingredientes de la carta, aún no me dicen cuál. ¿Qué me recomiendas?"
    uncertain = request(first)
    open_choice = request("No sé cuál plato pedir. ¿Qué me recomiendas para cenar?")
    assert uncertain.unspecified and uncertain.excluded == set()
    assert uncertain.replacement() == "¿Qué alimento quieres evitar en esta comida?"
    assert not open_choice.unspecified
    confirmed = meal_request([{"role": "user", "text": first},
                              {"role": "user", "text": "El huevo", "answers_food_exclusion": True}], DINNER)
    assert not confirmed.unspecified and confirmed.excluded == {"huevo"}


def test_severe_restriction_does_not_modify_safety_behavior():
    severe = request("Tengo alergia al huevo. ¿Qué puedo cenar?")
    assert severe.severe and severe.current_excluded == {"huevo"}
    assert not severe.allows({"arroz"})
    assert severe.replacement().endswith("¿Qué opciones estás considerando?")


def test_memory_exclusions_and_catalog_override_have_separate_evidence():
    menu = (("Pollo con papas", "pollo papa"), ("Arroz con lentejas", {"arroz", "lenteja", "aceite"}),
            ("Pedir una aclaración", ()))
    first = request("¿Qué ceno sin aceite ni papas?", dishes=menu)
    assert first.choices == []
    again = request("¿Qué ceno sin aceite ni papas?", "Ya conseguí papas y sigo sin aceite. ¿Qué ceno?", dishes=menu)
    assert again.choices == ["Pollo con papas"]
    remembered = request("¿Qué ceno sin aceite ni papas?", "Ya conseguí papas y sigo sin aceite. ¿Qué ceno?",
                         dishes=menu, memory_excluded={"patata"})
    assert remembered.current_excluded == {"aceite"}
    assert remembered.memory_excluded == {"papa"}
    assert remembered.excluded == {"aceite", "papa"}
    assert remembered.choices == []
    assert preference_ingredients("POLLO CON PAPÁS", DINNER, dishes=menu) == {"pollo", "papa"}
    assert preference_ingredients("Me gustan las patatas", DINNER, dishes=menu) == {"papa"}
    assert preference_ingredients("Me gusta todo", DINNER, dishes=menu) == set()
def test_food_reference_can_reopen_a_menu_without_dropping_other_exclusions():
    first = "¿Qué puedo preparar sin pollo ni asiete? El pollo está aparte, tampoco lo voy a usar."
    second = "Me trajeron aceite, ese ya lo puedo usar. Para cenar sigo sin pollo, sin papa ni tomate. ¿Qué ceno?"
    response = request(first, second)
    assert response.current_excluded == {"pollo", "papa", "tomate"}
    assert response.accepts("Arroz con lentejas y zanahoria")
