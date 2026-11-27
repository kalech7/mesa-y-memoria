"""Persistent, owner-scoped live conversations using the locked v5 selector."""
from __future__ import annotations

from contextlib import closing
from copy import deepcopy
from datetime import date, datetime
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets
from threading import RLock
from uuid import uuid4
from zoneinfo import ZoneInfo

from research.src.artifacts import canonical
from research.src.env.user_simulator import QUESTIONS
from research.src.memory.executor import Executor
from research.src.memory.store import Store
from research.src.policy.dialogue import current_context
from research.src.readers.symbolic import applicability, read
from research.tesis_profiles_parser_v5 import (
    _attribute, _original_id,
)
from research.tesis_shared_budget_env import ACTIONS, QUESTION_ACTIONS

from .chat import _render_answer
from .profiles_runtime import DEFAULT_RUN, ProfilesError, ProfilesRuntime
from .profiles_evidence import chat_question, decision, evidence, query_attribute, query_context, query_date, respond_to_query


DEFAULT_STORAGE = Path(__file__).resolve().parents[1] / "outputs/profiles_v5_chats"
ID = re.compile(r"^[a-f0-9]{32}$")
ATTRIBUTES = ("dinner_preference", "breakfast_preference", "drink_preference")
MAX_MESSAGE = 1200
LOCAL_ZONE = ZoneInfo("America/Guayaquil")
REUSE = "Puede haber visitas posteriores. Conserva solo preferencias y límites confirmados por la persona."


def iso_date(value):
    if not isinstance(value, str):
        raise ProfilesError("Indica una fecha ISO con formato YYYY-MM-DD.")
    try:
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError()
    except ValueError as error:
        raise ProfilesError("La fecha debe tener formato YYYY-MM-DD.") from error
    return value


def context_value(value):
    if value not in (None, "", "casa", "oficina"):
        raise ProfilesError("El contexto debe ser casa, oficina o quedar vacío.")
    return value or None


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def initialize(store):
    store.conn.executescript("""
        CREATE TABLE IF NOT EXISTS p5_state (
          singleton INTEGER PRIMARY KEY CHECK(singleton=1),
          owner_hash TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS p5_requests (
          request_id TEXT PRIMARY KEY, signature TEXT NOT NULL, response TEXT NOT NULL);
    """)
    store.conn.commit()


def legal_actions(observation):
    parsed = evidence(observation)
    _, candidates, _, _ = decision(observation)
    closing = ("PROCEED",)
    if any(approved and operation["record"]["scope"]["kind"] == "session" for operation, approved in candidates):
        closing += ("PROVISIONAL",)
    if not parsed["current_resolved"]:
        closing += ("DEFER",)
    slot = parsed["missing_slot"]
    if (not observation["questions_remaining"] or len(observation["asked_slots"]) >= 2
            or slot not in QUESTION_ACTIONS or parsed["clarification_state"] == "unavailable"
            or slot in observation["asked_slots"] and parsed["clarification_state"] != "partial"):
        return closing
    return tuple(action for action in ACTIONS if action in (*closing, slot))


def citations(store, messages, identifiers):
    turns = {turn["turn_id"]: turn for turn in messages}
    memories = {record["memory_id"]: record for record in store.list_records()}
    result = []
    for identifier in dict.fromkeys(_original_id(item, turns) for item in identifiers):
        if identifier in turns:
            result.append({"turn_id": identifier, "text": turns[identifier]["text"], "source": "user_turn"})
        elif identifier in memories:
            record = memories[identifier]
            result.append({"memory_id": identifier, "scope": record["scope"],
                           "quotes": record["evidence_quotes"], "source": "sqlite"})
    return result


def view(state, store):
    query = {"attribute": state["attribute"], "date": state["date"], "context": state["context"],
             "session_id": f"S{state['visit']}"}
    memory = [{**record, "applicable": applicability(record, {**query, "attribute": record["attribute"]})}
              for record in store.list_records()]
    return {**deepcopy(state), "memory": memory, "source": "sqlite",
            "questions_remaining": max(0, state["question_budget"] - state["questions_used"]),
            "extrapolation": state["visit"] > 3}


def confirmed_write(record):
    scope = record["scope"]
    if scope["kind"] == "session":
        description = "solo para esta visita"
    elif scope.get("valid_until_exclusive"):
        end = date.fromisoformat(scope["valid_until_exclusive"]).toordinal() - 1
        description = f"del {scope['valid_from']} al {date.fromordinal(end).isoformat()}, inclusive"
    else:
        description = f"como preferencia habitual desde {scope['valid_from']}"
    if scope.get("context"):
        description += f" en {scope['context']}"
    return f"Guardé {record['value']} {description}."


class ProfilesChats:
    question_budget = 2
    topic = staticmethod(query_attribute)

    def __init__(self, device="cpu", run=DEFAULT_RUN, storage=DEFAULT_STORAGE, *, runtime=None, clock=None):
        self.runtime = runtime or ProfilesRuntime(device, run)
        self.storage = Path(storage)
        self.storage.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.clock = clock or (lambda: datetime.now(LOCAL_ZONE).date().isoformat())
        self.lock = RLock()

    def catalog(self):
        return self.runtime.catalog()

    def resolve(self, model):
        return self.runtime.resolve(model)

    def start(self, body):
        descriptor = self.resolve(body.get("model"))
        token, identity = secrets.token_urlsafe(32), uuid4().hex
        state = {
            "id": identity, "model": descriptor, "revision": 0, "visit": 1,
            "date": iso_date(body.get("date", self.clock())), "context": context_value(body.get("context")),
            "attribute": "dinner_preference", "question_budget": self.question_budget, "questions_used": 0,
            "messages": [], "past_visits": [], "pending": None, "action_history": [], "asked_slots": [],
        }
        path = self.storage / f"{identity}.sqlite"
        with closing(Store(path)) as store:
            initialize(store)
            with store.conn:
                store.conn.execute("INSERT INTO p5_state VALUES(1,?,?)", (digest(token), canonical(state)))
            path.chmod(0o600)
            return {**view(state, store), "owner_token": token}

    def _access(self, store, body):
        row = store.conn.execute("SELECT owner_hash,payload FROM p5_state WHERE singleton=1").fetchone()
        token = body.get("owner_token")
        if row is None or not isinstance(token, str) or not hmac.compare_digest(row[0], digest(token)):
            raise ProfilesError("La conversación no está disponible con esta identidad.", 404)
        return json.loads(row[1])

    def handle(self, body):
        if not isinstance(body, dict):
            raise ProfilesError("Se esperaba una solicitud de conversación.")
        action = {"message": "send", "next_visit": "new_visit", "resume": "load"}.get(body.get("action"), body.get("action"))
        if action == "start":
            return self.start(body)
        if action not in ("load", "send", "new_visit"):
            raise ProfilesError("Acción de conversación desconocida.")
        identity = body.get("id")
        if not isinstance(identity, str) or not ID.fullmatch(identity):
            raise ProfilesError("Conversación desconocida.", 404)
        path = self.storage / f"{identity}.sqlite"
        if not path.is_file():
            raise ProfilesError("Conversación desconocida.", 404)
        with self.lock, closing(Store(path)) as store:
            state = self._access(store, body)
            if action == "load":
                return view(state, store)
            request_id = body.get("request_id")
            if not isinstance(request_id, str) or not 1 <= len(request_id) <= 80:
                raise ProfilesError("El identificador de la solicitud es obligatorio.")
            signature = canonical({key: value for key, value in body.items() if key != "owner_token"})
            cached = store.conn.execute("SELECT signature,response FROM p5_requests WHERE request_id=?", (request_id,)).fetchone()
            if cached:
                if cached[0] != signature:
                    raise ProfilesError("El identificador del turno ya corresponde a otro mensaje.", 409)
                return json.loads(cached[1])
            if type(body.get("revision")) is not int or body["revision"] != state["revision"]:
                raise ProfilesError("Hay una versión más reciente de esta conversación. Vuelve a cargarla.", 409)
            with closing(Store(":memory:")) as staged:
                staged.initialize(store.list_records())
                if action == "new_visit":
                    self._new_visit(state, body)
                else:
                    self._send(state, staged, body)
                state["revision"] += 1
                result = view(state, staged)
                store.conn.execute("BEGIN IMMEDIATE")
                try:
                    replay = store.conn.execute(
                        "SELECT signature,response FROM p5_requests WHERE request_id=?", (request_id,)).fetchone()
                    if replay:
                        if replay[0] != signature:
                            raise ProfilesError("El identificador del turno ya corresponde a otro mensaje.", 409)
                        store.conn.rollback()
                        return json.loads(replay[1])
                    latest = self._access(store, body)
                    if latest["revision"] != body["revision"]:
                        raise ProfilesError("Otra solicitud actualizó la conversación. Vuelve a cargarla.", 409)
                    store.conn.execute("DELETE FROM cm_records")
                    store.conn.executemany("INSERT INTO cm_records(memory_id,payload) VALUES(?,?)",
                                           [(record["memory_id"], canonical(record)) for record in staged.list_records()])
                    store.conn.executemany("INSERT INTO cm_executions(payload) VALUES(?)",
                                           staged.conn.execute("SELECT payload FROM cm_executions").fetchall())
                    store.conn.execute("UPDATE p5_state SET payload=? WHERE singleton=1", (canonical(state),))
                    store.conn.execute("INSERT INTO p5_requests VALUES(?,?,?)", (request_id, signature, canonical(result)))
                    store.conn.commit()
                except Exception:
                    store.conn.rollback()
                    raise
                return result

    def _new_visit(self, state, body):
        state["past_visits"].append({
            "visit": state["visit"], "date": state["date"], "context": state["context"],
            "messages": state["messages"], "pending": state["pending"],
        })
        state["visit"] += 1
        state["date"] = iso_date(body.get("date", state["date"]))
        state["context"] = context_value(body.get("context"))
        state["messages"], state["pending"], state["asked_slots"] = [], None, []

    def choose(self, state, observation, legal, messages):
        return self.runtime.decide(state["model"], observation, legal)

    def _send(self, state, store, body):
        text = body.get("text")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= MAX_MESSAGE:
            raise ProfilesError(f"Escribe entre 1 y {MAX_MESSAGE} caracteres.")
        when = iso_date(body.get("date", state["date"]))
        context = context_value(body.get("context", state["context"]))
        pending = state["pending"]
        question = chat_question(text)
        independent_query = question is not None
        if pending and independent_query:
            pending = None
        if pending is None:
            state["asked_slots"] = []
        attribute = pending["attribute"] if pending else self.topic(question or text, state["attribute"])
        if attribute not in ATTRIBUTES:
            raise ProfilesError("No se pudo identificar cena, desayuno o bebida.")
        turn = {"turn_id": f"u{state['visit']}_{state['revision'] + 1}", "role": "user",
                "text": text.strip(), "date": when}
        if pending:
            turn.update(in_reply_to=pending["turn_id"], about_attribute=attribute)
        messages = [*state["messages"], turn]
        opening = pending["opening"] if pending else len(messages) - 1
        active = messages[opening:]
        try:
            target_date = query_date(question, when) if independent_query else when
        except ValueError as error:
            raise ProfilesError(str(error)) from error
        query = {"attribute": attribute, "date": target_date,
                 "session_id": f"S{state['visit']}", "context": context}
        acquisition = {**query, "date": when}
        visible = {"query": acquisition, "memory": store.list_records(), "user_messages": active, "reuse_regime": REUSE}
        _, _, parsed, adaptations = decision(visible)
        visit_context = current_context(parsed)
        acquisition["context"] = visit_context
        query["context"] = query_context(question, visit_context) if independent_query else visit_context
        if query["date"] != when:
            query["session_id"] = "future_query"
        observation = {**visible, "query": query, "visit": state["visit"],
                       "questions_remaining": max(0, state["question_budget"] - state["questions_used"]),
                       "asked_slots": state["asked_slots"], "action_history": tuple(state["action_history"])}
        legal = legal_actions(observation)
        action, trace = self.choose(state, observation, legal, messages)
        if action not in legal:
            raise ProfilesError("La inferencia devolvió una acción fuera del menú.", 503)
        state["date"], state["context"], state["attribute"] = when, visit_context, attribute
        state["action_history"].append(action)
        state["pending"] = None
        operations, verification, refs = [], None, []
        if action in QUESTION_ACTIONS:
            state["questions_used"] += 1
            state["asked_slots"].append(action)
            identity = f"a_tool_{state['visit']}_{state['revision'] + 1}"
            state["pending"] = {"turn_id": identity, "attribute": attribute, "opening": opening, "slot": action}
            answer = QUESTIONS[action]
            response = None
        else:
            identity = f"a{state['visit']}_{state['revision'] + 1}"
            response, candidates, parsed, adaptations = decision(visible)
            if independent_query:
                # Answer the requested attribute. Other assertions can still
                # authorize their own writes, with their original citations.
                _, _, recalled, earlier_adaptations = decision({**visible, "user_messages": messages})
                adaptations = list({canonical(item): item for item in (*adaptations, *earlier_adaptations)}.values())
                relevant = [turn for turn in recalled["user_messages"]
                            if _attribute(turn["text"], attribute) == attribute]
                response = respond_to_query({**recalled, "user_messages": relevant}, query)
            response = deepcopy(response)
            if action == "DEFER":
                response = {"mode": "DEFER", "value": None, "branches": [], "evidence_ids": []}
            submitted = candidates if action == "PROCEED" else ()
            approved = [deepcopy(operation) for operation, justified in submitted if justified]
            applied = Executor(store).execute(approved or [{"op": "NOOP"}], recorded_at=when)
            if not applied["valid"]:
                raise ProfilesError("Las operaciones de memoria no pudieron aplicarse. " + str(applied["error"]), 500)
            operations = applied["executed_operations"]
            verification = {"mode": "veto", "proposed": len(submitted), "accepted": len(approved),
                            "rejected": len(submitted) - len(approved), "source": "v5_public_evidence"}
            all_messages = [item for visit in state["past_visits"] for item in visit["messages"]] + messages
            refs = citations(store, all_messages, response["evidence_ids"])
            original_ids = {message["turn_id"] for message in all_messages}
            original_ids.update(record["memory_id"] for record in store.list_records())
            response["evidence_ids"] = list(dict.fromkeys(
                _original_id(item, original_ids) for item in response["evidence_ids"]))
            answer = _render_answer(response)
            if response["mode"] == "ANSWER":
                answer = f"Para esta consulta, tu preferencia es {response['value']}."
            writes = [operation["record"] for operation in operations if "record" in operation]
            if writes:
                answer += " " + confirmed_write(writes[-1])
            if action == "PROVISIONAL" and response["mode"] == "ANSWER":
                answer += ". Lo tomo para esta visita."
        state["messages"] = [*messages, {
            "turn_id": identity, "role": "assistant", "text": answer, "date": when,
            "trace": {**trace, "selected_action": action, "legal_actions": list(legal),
                      "query": query, "live_wording_adaptations": adaptations},
            "response": response, "operations": operations, "verification": verification, "citations": refs,
        }]
