"""Persistent food conversations with the two locked v6 clarification policies."""
from __future__ import annotations

from pathlib import Path
import re

from research.src.env.user_simulator import QUESTIONS
from .chat_history import memory_match, recommendation_memory
from .food_recipes import folded, preparation, preparation_request, recipe_dish
from .profiles_chat import MAX_MESSAGE, ProfilesChats, citations, confirmed_write, context_value, iso_date
from .profiles_evidence import query_attribute, query_context, query_date
from .profiles_runtime import ProfilesError
from .recommendations import DISHES, EXCLUSION, meal_request, preference_ingredients


STORAGE = Path(__file__).resolve().parents[1] / "outputs/final_food_chats"
ARMS = ("SFT-ramas", "GRPO-continuacion")
RECALL = re.compile(r"\b(?:recuerd\w*|acuerdas|guardad\w*|habitual\w*|suel[eo]|prefiero|preferencias?)\b", re.I)
ALTERNATIVE = re.compile(r"\b(?:otr[ao]\s+(?:opci[oó]n|alternativa|idea|plato|cena|bebida|desayuno)|"
                         r"algo\s+(?:diferente|distinto))\b", re.I)
DO_NOT_STORE = re.compile(r"\bno\s+(?:quiero\s+que\s+)?(?:(?:te|l[oa])\s+)?"
                          r"(?:lo\s+|la\s+)?(?:guardes|anotes|registres|recuerdes|conserves|almacenes)\b", re.I)
EXCLUSION_ONLY = re.compile(r"^\s*(?:evit[oa]|sin|no (?:puedo|quiero|me gusta)|soy al[eé]rgic[oa])\b", re.I)
UNSUPPORTED = re.compile(r"\b(?:receta(?:s)?|paso a paso|calor[ií]as|plan nutricional)\b", re.I)
CAPABILITIES = ("Puedo sugerirte una cena, un desayuno o una bebida a partir del menú, "
                "y recordar las preferencias que me confirmes.")


class FinalFoodChats(ProfilesChats):
    question_budget = 1

    def __init__(self, lab, storage=None):
        self.lab = lab
        super().__init__(storage=storage or STORAGE, runtime=lab)

    def resolve(self, arm=None):
        arm = "GRPO-continuacion" if arm is None else arm
        if arm not in ARMS:
            raise ProfilesError("Elige Qwen SFT o Qwen GRPO longitudinal.")
        return self.lab.describe_model(arm)

    @staticmethod
    def topic(text, default):
        # The agenda may mention breakfasts after a present request for dinner.
        # Ground the current topic in the preference-bearing clause in that case.
        for clause in re.split(r"[.!?;]\s*|,\s*(?=(?:estoy|mañana|luego|después)\b)", text, flags=re.I):
            if re.search(r"\b(?:quiero|prefiero|suelo|habitual (?:es|será)|anota|me apetece)\b", clause, re.I):
                return query_attribute(clause, default)
        return query_attribute(text, default)

    def choose(self, state, observation, legal, messages):
        if "TEMPORAL_SCOPE" not in legal:
            return "PROCEED", {"source": "memory", "model_used": False}
        if state["visit"] > 2:
            return "PROCEED", {"source": "forced", "action": "WAIT", "model_used": False,
                               "reason": "outside_two_decision_visits"}
        public_messages = [m for visit in state["past_visits"] for m in visit["messages"]] + messages
        public = {
            "visit": state["visit"], "questions_remaining": observation["questions_remaining"],
            "messages": [{"role": m["role"], "text": m["text"]} for m in public_messages],
            "question": QUESTIONS["TEMPORAL_SCOPE"], "memory": observation["memory"],
        }
        trace = self.lab.score_public(state["model"]["arm"], public)
        selected = "TEMPORAL_SCOPE" if trace["action"] == "ASK" else "PROCEED"
        return selected, {**trace, "model_used": trace["source"] == "qwen"}

    def _send(self, state, store, body):
        text = body.get("text")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= MAX_MESSAGE:
            raise ProfilesError(f"Escribe entre 1 y {MAX_MESSAGE} caracteres.")
        text = text.strip()
        when = iso_date(body.get("date", state["date"]))
        context = context_value(body.get("context", state["context"]))
        query_text = text.rsplit("¿", 1)[-1]
        attribute = self.topic(query_text, state["attribute"])
        pending = state["pending"] is not None
        number = f"{state['visit']}_{state['revision'] + 1}"
        turn = {"role": "user", "turn_id": "u" + number, "date": when, "text": text}
        history = [*state["messages"], turn]
        previous = history[-2].get("recommendation") if len(history) > 1 else None
        last_suggestion = next((message["recommendation"] for message in reversed(history[:-1])
                                if message.get("recommendation", {}).get("chosen")), None)
        if previous and previous.get("requested_food"):
            turn["answers_food_exclusion"] = True
        is_question = "¿" in text or "?" in text
        recall = is_question and RECALL.search(query_text)
        # Recipe questions can fall between a suggestion and "otra opción".
        # Keep them in the real chat but outside the meal-specific constraints.
        menu_history = [m for m in history[:-1] if not (m["role"] == "user" and preparation_request(m["text"])
                                                        and not m.get("recipe_constraint"))]
        alternative = bool(ALTERNATIVE.search(text) and last_suggestion)
        request = None if recall else meal_request([*menu_history, turn], attribute, follow_up=bool(previous) or alternative, date=when)
        recommendation = None
        refs = []
        source = "conversation"
        action = "HELP"
        greeting = bool(re.fullmatch(r"(?:hola(?:[, ]+que tal)?|buen[oa]s(?: dias| tardes| noches)?|que tal)",
                                     folded(text).strip(" !¡¿?.")))
        if greeting:
            answer = ("¡Hola! ¿Qué te gustaría comer hoy? Puedes pedirme una cena sin huevo "
                      "o decirme qué sueles desayunar.")
            action = "GREET"
        elif preparation_request(text):
            last = last_suggestion["chosen"] if last_suggestion else None
            dish = recipe_dish(text, last)
            answer = preparation(dish) if dish else None
            if answer:
                recipes = dict(DISHES[attribute])
                if dish not in recipes:
                    attribute = next(meal for meal, dishes in DISHES.items() if dish in dict(dishes))
                    recipes = dict(DISHES[attribute])
                turn["recommendation_attribute"] = attribute
                turn["recipe_constraint"] = bool(EXCLUSION.search(folded(text)))
                constraints = meal_request([*menu_history, turn], attribute, date=when, follow_up=bool(last),
                                           classified_request=True)
                ingredients = recipes[dish].split()
                if constraints.severe or constraints.unspecified:
                    answer = constraints.replacement()
                elif not constraints.allows(ingredients):
                    conflict = ", ".join(i for i in ingredients if not constraints.allows([i]))
                    answer = f"La receta de {dish.lower()} lleva {conflict}. Puedes pedirme otra opción."
            else:
                answer = "No encuentro esa preparación en el recetario. Puedes pedirme una sugerencia del menú."
            source, action = "recipe", "PREPARE"
        elif request:
            source, action = "catalog", "SUGGEST"
            query = {
                "attribute": attribute, "session_id": f"S{state['visit']}", "date": query_date(text, when),
                "context": query_context(query_text, context),
            }
            if request.date and query["date"] == when:
                query["date"] = request.date
            if query["date"] != when:
                query["session_id"] = "future_query"
            preferred = recommendation_memory(store.list_records(), query)
            seen = {
                item["chosen"].casefold()
                for message in history[:-1] if (item := message.get("recommendation")) and item.get("chosen")
                and item["attribute"] == attribute and item["date"] == query["date"]
            } if alternative else set()
            compatible = [
                {"name": name, "ingredients": sorted(ingredients)}
                for name, ingredients in request.suggestions
                if request.allows(ingredients) and name.casefold() not in seen
            ]
            chosen = None
            reason = None
            shared = []
            if preferred:
                value = preferred[0]["value"]
                chosen = next((item for item in compatible if item["name"].casefold() == value.casefold()), None)
                if not (chosen or alternative or request.excluded or request.severe or request.unspecified):
                    chosen = {"name": value, "ingredients": []}
                if chosen:
                    reason = "remembered_dish"
                else:
                    chosen = next((item for item in compatible if memory_match(item, preferred, attribute, DISHES[attribute])), None)
                    if chosen:
                        reason = "shared_ingredients"
                        shared = sorted(preference_ingredients(value, attribute) & set(chosen["ingredients"]))
            chosen = chosen or next(iter(compatible), None)
            if chosen:
                answer = f"Para esta comida te sugiero {chosen['name'].lower()}."
                if reason:
                    refs = citations(store, [], [item["memory_id"] for item in preferred])
                    if reason == "remembered_dish":
                        answer += " Es la preferencia que tienes guardada."
                    else:
                        answer += f" Comparte {', '.join(shared)} con tu preferencia guardada de {value}."
            elif alternative and not (request.severe or request.unspecified) and request.suggestions:
                answer = "Ya te mostré todas las opciones de este menú que cumplen tus condiciones."
            else:
                answer = request.replacement()
            turn["recommendation_attribute"] = attribute
            recommendation = {
                "attribute": attribute, "date": query["date"], "excluded": sorted(request.excluded),
                "chosen": chosen["name"] if chosen else None, "ingredients": chosen["ingredients"] if chosen else [],
                "requested_food": request.unspecified, "memory_used": [r["memory_id"] for r in refs],
                "reason": reason, "shared_ingredients": shared,
            }
        elif DO_NOT_STORE.search(text):
            answer, action = "Entendido. No guardaré esa indicación como una preferencia.", "NO_WRITE"
        elif not pending and EXCLUSION_ONLY.search(text):
            answer, action = "Tendré en cuenta esa indicación si me pides una opción para esta comida.", "NO_WRITE"
        elif (is_question and not recall) or (not pending and UNSUPPORTED.search(text)):
            answer = CAPABILITIES
        else:
            super()._send(state, store, body)
            response = state["messages"][-1]
            writes = [operation["record"] for operation in response["operations"] if "record" in operation]
            if writes and not is_question:
                response.update(text=" ".join(confirmed_write(record) for record in writes), response=None,
                                citations=citations(store, [], [r["memory_id"] for r in writes]))
            elif response["response"] and response["response"]["mode"] == "DEFER":
                response["text"] = ("Con lo que tengo, no puedo confirmar esa preferencia. " + CAPABILITIES)
            elif not is_question and response["trace"].get("action") == "WAIT" and response["response"]["mode"] == "ANSWER":
                response["text"] = f"Para esta visita, {response['response']['value']}."
            return
        state.update(date=when, context=context, attribute=attribute)
        if not greeting:
            state.update(pending=None, asked_slots=[])
        reply = {
            "role": "assistant", "turn_id": "a" + number, "text": answer, "date": when,
            "trace": {"source": source, "selected_action": action, "model_used": False},
            "response": None, "citations": refs, "operations": [], "verification": None,
        }
        if recommendation:
            reply["recommendation"] = recommendation
        state["messages"] = [*history, reply]
