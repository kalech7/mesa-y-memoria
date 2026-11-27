"""Free-form conversations with the local LoRA checkpoints and SQLite memory."""
from __future__ import annotations

from contextlib import closing
from datetime import date, datetime
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from threading import Lock
from uuid import uuid4
from zoneinfo import ZoneInfo

from research.src.env.user_simulator import QUESTIONS, response_availability
from research.src.memory.executor import Executor
from research.src.memory.store import Store
from research.src.policy.dialogue import attribute_in, current_context, preference_turn
from research.src.policy.llm_controller import InvalidOutput, MAX_INPUT_TOKENS
from research.src.policy.rule_controller import RuleController, evidence_model
from research.src.readers.symbolic import applicability, read
from research.tesis_memory_guard import plan_write

from .cases import MODELS
from .chat_history import memory_match, recommendation_memory, simulated_history
from .meal_models import meal_models
from .meal_policy import recommendation_menu
from .recommendations import asks_for_meal, meal_date, meal_request


MAX_MESSAGE = 800
REUSE = "Puede haber una conversación posterior. Conserva los datos confirmados y sus límites de vigencia."
MEMORY_QUERY = re.compile(r"\b(?:qu[eé] (?:\w+ ){0,3}prefiero|qu[eé] recuerdas|recuerdas qu[eé]|"
                          r"te acuerdas|(?:cu[aá]l es|cu[aá]l era) mi preferencia)\b", re.I)
LOCAL_ZONE = ZoneInfo("America/Guayaquil")
TIME_QUESTION = "¿Es solo para esta conversación, por hoy, por otro plazo o un cambio habitual?"


class ChatError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _meal(text, default):
    _, marker, question = text.rpartition("¿")
    return attribute_in(question, attribute_in(text, default)) if marker and question else attribute_in(text, default)


def _visible(controller, memory, history, query, input_limit=MAX_INPUT_TOKENS):
    """Fit whole previous exchanges before reducing the retrieved memory."""
    exchanges = []
    for message in history:
        if message["role"] == "user" and not message.get("in_reply_to"):
            exchanges.append([])
        exchanges[-1].append({key: message[key] for key in
                              ("role", "text", "turn_id", "date", "in_reply_to", "about_attribute") if key in message})
    keys = {query["attribute"], *(_meal(m["text"], query["attribute"]) for m in exchanges[-1] if m["role"] == "user")}
    ranked = sorted(range(len(memory)), key=lambda i: (
        memory[i]["attribute"] in keys,
        applicability(memory[i], {**query, "attribute": memory[i]["attribute"]}) is not False, i))
    observation = {"memory": memory, "user_messages": [turn for exchange in exchanges for turn in exchange],
                   "query": query, "reuse_regime": REUSE}

    def fits():
        return all(len(controller.backend.encode_prompt(controller.prompt(phase, observation))) <= input_limit
                   for phase in ("select", "respond", "write"))

    while not fits():
        if len(exchanges) > 1:
            exchanges.pop(0)
            observation["user_messages"] = [turn for exchange in exchanges for turn in exchange]
        elif ranked:
            ranked.pop(0)
            observation["memory"] = [memory[index] for index in sorted(ranked)]
        else:
            raise ChatError("El mensaje y su aclaración superan el contexto del modelo. Prueba con frases más breves.")
    limited = len(observation["user_messages"]) < len(history) or len(observation["memory"]) < len(memory)
    return observation, limited


def _call(controller, phase, observation, *args):
    try:
        result = getattr(controller, phase)(observation, *args)
    except InvalidOutput as error:
        record = controller.phase_records[-1]
        return None, {"phase": phase, "error": str(error), "raw": record["raw_output"]}
    return result, {"phase": phase, "action": result, "raw": controller.phase_records[-1]["raw_output"]}


def _render_answer(result):
    if result is None:
        return "El modelo no produjo una respuesta con el formato esperado. Puedes ver su salida en el detalle del turno."
    if result["mode"] == "ANSWER":
        return result["value"]
    if result["mode"] == "CONDITIONALIZE":
        return "\n".join(f"{branch['condition']} · {branch['value']}" for branch in result["branches"])
    return "No tengo suficiente información para darte una respuesta única."


class Conversation:
    def __init__(self, directory, model, clock, simulated):
        self.id = uuid4().hex
        self.model = model
        self.clock = clock
        self.simulated = simulated
        self.path = directory / f"{self.id}.sqlite"
        self.lock = Lock()
        self.revision = 0
        self.session = 1
        self.attribute = "dinner_preference"
        self.history = []
        self.past_sessions = []
        self.pending = None
        self.last_request = None
        self.last_response = None

    def view(self, store, **details):
        return {"id": self.id, "model": self.model.public(), "revision": self.revision, "session": self.session,
                "attribute": self.attribute, "date": self.clock(), "simulated": self.simulated,
                "messages": self.history, "past_sessions": self.past_sessions, "memory": store.list_records(),
                "pending": self.pending is not None, **details}

    def send(self, store, controller, text, verify):
        serial = f"{self.session}_{len(self.history) + 1}"
        date = self.clock()
        turn = {"turn_id": f"u{serial}", "role": "user", "text": text, "date": date}
        recall = bool(MEMORY_QUERY.search(text) and not preference_turn(turn) and not asks_for_meal(text))
        if recall:
            self.pending = None
        attribute = self.attribute if self.pending else _meal(text, self.attribute)
        if self.pending:
            turn.update(in_reply_to=self.pending, about_attribute=attribute)
        query = {"attribute": attribute, "date": date, "session_id": f"S{self.session}", "context": None}
        if recall:
            query["date"] = meal_date(text, date)
        history = [*self.history, turn]
        query["context"] = current_context({
            "user_messages": [m for m in history if m.get("date") == date], "query": query,
        })
        previous = self.history[-1].get("recommendation") if self.history else None
        if previous and previous.get("requested_food"):
            turn["answers_food_exclusion"] = True
        same_day = [m for m in history if m.get("date") == date or m["role"] == "assistant"]
        follow_up = bool(previous)
        if self.pending:
            # Recover open exchanges where a meal request had reached the
            # memory controller before the recommendation route recognized it.
            opening = next((m for m in reversed(same_day[:-1])
                            if m["role"] == "user" and not m.get("in_reply_to")), None)
            follow_up = follow_up or bool(opening and asks_for_meal(opening["text"]))
        policy = getattr(controller, "meal_policy", None)
        menu = recommendation_menu((), attribute, shuffle=False) if policy else None
        dishes = [(item["name"], item["ingredients"]) for item in menu if item["ingredients"]] if menu else None
        recommendation = None if recall else meal_request(
            history, attribute, follow_up=follow_up, date=date, dishes=dishes)
        direct_menu = (policy is not None and not verify and not self.pending
                       and not preference_turn(turn) and not recall)
        if recommendation or direct_menu:
            observation, limited = _visible(controller, store.list_records(), history, query,
                                            input_limit=MAX_INPUT_TOKENS - 256)
            trace = []
            source = "catalog"
            reply = recommendation.replacement() if recommendation else ""
            model_choice = None
            chosen = None
            raw = False
            reason = None
            meal_query = {**query, "date": recommendation.date or date} if recommendation else query
            preferred = recommendation_memory(store.list_records(), meal_query)
            if policy is not None and (not verify or not recommendation.severe):
                todays_ids = {m["turn_id"] for m in same_day if m["role"] == "user"}
                turns = [m["text"] for m in observation["user_messages"] if m["turn_id"] in todays_ids]
                remembered = next((item for item in menu
                                   if memory_match(item, preferred, attribute, dishes)
                                   and recommendation and recommendation.allows(item["ingredients"])), None)
                model_choice, record = (policy.choose_with_memory(turns, attribute, preferred[0]["value"])
                                        if preferred and hasattr(policy, "choose_with_memory")
                                        else policy.choose(turns, attribute))
                trace.append(record)
                raw = not verify
                if verify and remembered and not memory_match(model_choice, preferred, attribute, dishes):
                    chosen = remembered
                    reason = "remembered_preference"
                elif model_choice["ingredients"] and (raw or recommendation.allows(model_choice["ingredients"])):
                    chosen = model_choice
                    source = "model"
                else:
                    compatible = (next((item for item in menu
                                        if item["ingredients"] and recommendation.allows(item["ingredients"])), None)
                                  if verify else None)
                    if compatible:
                        chosen = compatible
                        reason = "incompatible_choice" if model_choice["ingredients"] else "unnecessary_clarification"
                    elif verify and recommendation and recommendation.unspecified:
                        reply = "¿Qué alimento quieres evitar en esta comida?"
                    else:
                        reply = "¿Qué ingredientes tienes a mano para esta comida?"
                    if not compatible and not model_choice["ingredients"]:
                        source = "model"
            elif recommendation and recommendation.choices and not (recommendation.unspecified or recommendation.severe):
                response, record = _call(controller, "recommendation", observation, recommendation.choices)
                trace.append(record)
                if response and response["mode"] == "ANSWER" and recommendation.accepts(response["value"]):
                    reply = f"Podrías preparar {response['value'].lower().strip(' .')}."
                    source = "model"
            if chosen:
                reply = f"Podrías preparar {chosen['name'].lower()}."
            self.pending = None
            details = {"source": source, "excluded": sorted(recommendation.excluded) if recommendation else []}
            if model_choice:
                details.update(model_choice=model_choice["name"], raw=raw, chosen=chosen["name"] if chosen else None)
                if reason:
                    details["reason"] = reason
                if preferred and hasattr(policy, "choose_with_memory"):
                    details["memory_offered"] = preferred
                if memory_match(chosen, preferred, attribute, dishes):
                    details["memory_used"] = preferred
            if recommendation and recommendation.unspecified and reply.startswith("¿Qué alimento quieres evitar"):
                details["requested_food"] = True
            return self.finish(store, history, attribute, f"a{serial}", reply, trace, context_limited=limited,
                               recommendation=details)
        observation, limited = _visible(controller, store.list_records(), history, query)
        trace = []
        selection_guard = None
        if not self.pending:
            action, record = _call(controller, "select", observation)
            trace.append(record)
            selected_slot = action.get("slot") if action and action["action"] == "ASK" else None
            if verify and not recall and preference_turn(turn):
                evidence = evidence_model(observation)
                fact = evidence["new_record"]
                if evidence["missing_slot"] is None and fact and fact["status"] == "confirmed":
                    if not action or action["action"] == "ASK":
                        selection_guard = "explicit_scope"
                        selected_slot = None
                elif evidence["missing_slot"] in {"TEMPORAL_SCOPE", "VALID_UNTIL"}:
                    if selected_slot != evidence["missing_slot"]:
                        selection_guard = "missing_scope"
                        selected_slot = evidence["missing_slot"]
            if selected_slot and not (verify and recall):
                self.pending = f"a_tool_{serial}"
                question = TIME_QUESTION if selected_slot == "TEMPORAL_SCOPE" else QUESTIONS[selected_slot]
                return self.finish(store, history, attribute, self.pending, question, trace,
                                   context_limited=limited, selected_slot=selected_slot, selection_guard=selection_guard)
            if not action and not (selection_guard or (verify and recall)):
                return self.finish(store, history, attribute, f"a{serial}",
                                   "El modelo no produjo una decisión con el formato esperado.", trace,
                                   context_limited=limited)
        response, record = _call(controller, "respond", observation)
        trace.append(record)
        if recall:
            answer = read(store.list_records(), query) if verify else response
            memory_read = {**(answer or {"mode": None, "value": None, "branches": [], "evidence_ids": []}),
                           "source": "sqlite" if verify else "model"}
            return self.finish(store, history, attribute, f"a{serial}", _render_answer(answer), trace,
                               memory_read=memory_read, context_limited=limited)
        if verify and self.pending and response_availability(turn) == "unavailable":
            self.pending = None
            answer = {"mode": "DEFER", "value": None, "branches": [], "evidence_ids": [], "source": "sqlite"}
            reply = "Dejo pendiente ese cambio hasta saber su vigencia. Los recuerdos confirmados siguen disponibles."
            return self.finish(store, history, attribute, f"a{serial}", reply, trace,
                               memory_read=answer, context_limited=limited)
        exchange = [turn]
        if self.pending:
            for prior in reversed(self.history):
                exchange.append(prior)
                if prior["role"] == "user" and not prior.get("in_reply_to"):
                    break
        write = decision = None
        if any(m["role"] == "user" and preference_turn(m) for m in exchange):
            proposed, record = _call(controller, "write", observation)
            trace.append(record)
            decision = plan_write(observation, proposed) if verify else None
            operations = decision["operations"] if decision else (proposed or {}).get("operations")
            write = Executor(store).execute(operations, recorded_at=date) if operations else None
        write_applied = (verify and write and write["valid"]
                         and any(op["op"] != "NOOP" for op in write["executed_operations"]))
        fallback = verify and not write_applied and response is None
        if fallback:
            response = RuleController().respond(observation)
        if write_applied:
            response = read(store.list_records(), query)
            if response["mode"] == "DEFER":
                reply = "Guardé la preferencia con la vigencia que indicaste."
            else:
                reply = _render_answer(response)
        else:
            reply = _render_answer(response)
        self.pending = None
        return self.finish(store, history, attribute, f"a{serial}", reply, trace,
                           write=write, verification=decision, context_limited=limited, fallback=fallback,
                           selection_guard=selection_guard, confirmed_write=bool(write_applied))

    def finish(self, store, history, attribute, turn_id, reply, trace, **details):
        self.history = [*history, {"turn_id": turn_id, "role": "assistant", "text": reply, "trace": trace,
                                   **details}]
        self.attribute = attribute
        return self.view(store)


class Chats:
    def __init__(self, live, *, clock=None):
        self.live = live
        self.clock = clock or (lambda: datetime.now(LOCAL_ZONE).date())
        self.directory = TemporaryDirectory(prefix="tesis-chat-")
        self.sessions = {}

    def today(self):
        value = self.clock()
        return date.fromisoformat(value).isoformat() if isinstance(value, str) else value.isoformat()

    def start(self, model_id, *, initial_memory=None, simulated=False):
        models = {**MODELS, **meal_models()[0]}
        if not isinstance(model_id, str) or model_id not in models:
            raise ChatError("Elige un modelo disponible.")
        available, reason = self.live.status()
        if not available:
            raise ChatError(reason, 503)
        chat = Conversation(Path(self.directory.name), models[model_id], self.today, simulated)
        if simulated:
            initial_memory, chat.past_sessions = simulated_history(self.today())
            chat.session = len(chat.past_sessions) + 1
        with closing(Store(chat.path)) as store:
            if initial_memory:
                store.initialize(initial_memory)
            result = chat.view(store)
        self.sessions[chat.id] = chat
        return result

    def handle(self, body):
        if not isinstance(body, dict):
            raise ChatError("Se esperaba una solicitud de conversación.")
        action = body.get("action")
        chat_id = body.get("id")
        if action == "start":
            simulated = body.get("simulated", False)
            if type(simulated) is not bool:
                raise ChatError("El historial de ejemplo debe ser booleano.")
            return self.start(body.get("model"), simulated=simulated)
        chat = self.sessions.get(chat_id) if isinstance(chat_id, str) else None
        if chat is None:
            raise ChatError("La conversación ya no está disponible en este servidor.", 404)
        with chat.lock, closing(Store(chat.path)) as store:
            if action == "resume":
                return chat.view(store)
            request_id = body.get("request_id")
            signature = (request_id, action, body.get("revision"), body.get("text"), body.get("verify"))
            if not isinstance(request_id, str) or not 1 <= len(request_id) <= 80:
                raise ChatError("Identificador de turno inválido.")
            if chat.last_request == signature:
                return chat.last_response
            if type(body.get("revision")) is not int or body["revision"] != chat.revision:
                raise ChatError("Hay una versión más reciente de esta conversación. Vuelve a cargarla.", 409)
            if action == "new_session":
                if chat.history:
                    messages = [{key: message[key] for key in ("role", "text", "turn_id", "date") if key in message}
                                for message in chat.history]
                    chat.past_sessions = [*chat.past_sessions, {"session": chat.session, "messages": messages,
                                                               "simulated": False}]
                chat.session += 1
                chat.history, chat.pending = [], None
                result = chat.view(store)
            elif action == "send":
                text, verify = body.get("text"), body.get("verify", True)
                if not isinstance(text, str) or not 1 <= len(text.strip()) <= MAX_MESSAGE:
                    raise ChatError(f"Escribe un mensaje de 1 a {MAX_MESSAGE} caracteres.")
                if type(verify) is not bool:
                    raise ChatError("El control de verificación debe ser booleano.")
                with self.live.chat_controller(chat.model) as controller:
                    result = chat.send(store, controller, text.strip(), verify)
            else:
                raise ChatError("Acción de conversación desconocida.")
            chat.revision += 1
            result["revision"] = chat.revision
            chat.last_request, chat.last_response = signature, result
            return result
