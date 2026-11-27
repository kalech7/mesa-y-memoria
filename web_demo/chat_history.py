"""Public demo evidence and memory use in the live recommendation menu."""
from __future__ import annotations

from datetime import date, timedelta

from research.src.readers.symbolic import read
from research.tesis_meal_cases import prompt
from research.tesis_meal_varied_bank import prompt as varied_prompt

from .meal_policy import MealPolicy, option_tokens, recommendation_menu
from .recommendations import preference_ingredients


def simulated_history(today):
    today = date.fromisoformat(today)
    previous = (today - timedelta(days=2)).isoformat()
    following = (today + timedelta(days=1)).isoformat()
    today = today.isoformat()
    turns = (
        (1, previous, "dinner_preference", "pescado al horno con papas",
         "Para la cena, prefiero pescado al horno con papas de forma habitual."),
        (1, previous, "breakfast_preference", "avena con banano",
         "Para el desayuno, prefiero avena con banano de forma habitual."),
        (2, today, "dinner_preference", "arroz con lentejas y zanahoria",
         "Solo por hoy prefiero arroz con lentejas y zanahoria para la cena."),
    )
    sessions = []
    records = []
    for index, (session, when, attribute, value, utterance) in enumerate(turns, 1):
        if len(sessions) < session:
            sessions.append({"session": session, "simulated": True, "messages": []})
        source = f"demo-u{index}"
        sessions[-1]["messages"].append({"role": "user", "text": utterance, "turn_id": source, "date": when})
        sessions[-1]["messages"].append({"role": "assistant", "text": "Entendido, queda guardado."})
        records.append({
            "memory_id": f"demo-m{index}", "attribute": attribute, "value": value,
            "scope": {"kind": "interval" if session == 2 else "ongoing", "session_id": None,
                      "context": None, "valid_from": when, "valid_until_exclusive": following if session == 2 else None},
            "status": "confirmed", "recorded_at": when, "supersedes": [],
            "source_turn_ids": [source], "evidence_quotes": [{"turn_id": source, "text": utterance}],
        })
    return records, sessions


def recommendation_memory(memory, query):
    result = read(memory, query)
    if result["mode"] != "ANSWER":
        return []
    evidence = set(result["evidence_ids"])
    return [{key: record[key] for key in ("memory_id", "value")}
            for record in memory if record["memory_id"] in evidence]


def memory_match(candidate, preferred, attribute, dishes):
    if not preferred or not candidate or not candidate["ingredients"]:
        return False
    value = preferred[0]["value"]
    names = {name.casefold() for name, _ in dishes}
    if value.casefold() in names:
        return value.casefold() == candidate["name"].casefold()
    wanted = preference_ingredients(value, attribute, dishes=dishes)
    return bool(wanted) and wanted <= set(candidate["ingredients"])


class MemoryMealPolicy(MealPolicy):
    def choose_with_memory(self, turns, attribute, value):
        varied = self.adapter.startswith("comidas-variado-")
        recipes = recommendation_menu(turns, attribute, shuffle=varied)
        render = varied_prompt if varied else prompt
        description = ("Preferencia vigente de SQLite. Puedes usarla si es compatible con el pedido actual: "
                       + " ".join(value.split()))
        tokens = option_tokens(self.actor.tokenizer)
        instruction, separator, dialogue = render(turns, recipes).partition("\nUsuario: ")
        ids = self.actor.encode_prompt(f"{instruction}\n{description}{separator}{dialogue}")
        return self._predict(ids, recipes, tokens)
