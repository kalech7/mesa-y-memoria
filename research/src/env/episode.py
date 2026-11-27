"""Three-session environment with phase-specific evidence availability.

Only observe/ask are actor-facing tools. private_trace is an evaluator artifact,
never serialized into a prompt. S3 opens a new SQLite connection and builds an
observation from persisted records and its explicit query only.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from ..memory.store import Store
from ..memory.executor import Executor
from .user_simulator import UserSimulator, QUESTIONS
from .imperfections import VERSION as IMPERFECTION_VERSION, normalize_config, omit_memory


class EpisodeEnvironment:
    def __init__(self, episode, db_path, *, nonresponse_rate=0.0, imperfections=None):
        if str(db_path) == ":memory:":
            raise ValueError("PERSISTENT_SQLITE_PATH_REQUIRED")
        self.db_path = Path(db_path)
        self._initial_episode = deepcopy(episode)
        self.nonresponse_rate = nonresponse_rate
        self.imperfections = normalize_config(imperfections)
        self._imperfection_root_id = episode.get("root_id", "")
        self.store = None
        self.private_trace = {}
        self.question_count = 0
        self._closed = False
        self._reopened = False
        self.reset()

    def reset(self, episode_id=None, rollout_seed=0):
        if self.store is not None:
            self.store.close()
        episode = deepcopy(self._initial_episode)
        if episode_id is not None and episode_id != episode.get("root_id"):
            raise ValueError("UNKNOWN_EPISODE")
        if self.db_path.exists():
            # A reset may reopen pristine S1, but must not destroy an existing rollout.
            probe = Store(self.db_path)
            existing = probe.list_records()
            probe.close()
            if existing != episode["initial_memory"]:
                raise ValueError("ROLLOUT_PATH_ALREADY_USED")
        self.store = Store(self.db_path)
        self.store.initialize(episode["initial_memory"])
        self._messages = []
        for message in episode["user_messages"]:
            if message.get("role", "user") not in {"user", "assistant"}:
                raise ValueError("INVALID_CONVERSATION_ROLE")
            self._messages.append({key: deepcopy(message[key]) for key in
                                   ("turn_id", "text", "role", "in_reply_to", "about_attribute") if key in message})
        if len({m["turn_id"] for m in self._messages}) != len(self._messages):
            raise ValueError("DUPLICATE_CONVERSATION_TURN_ID")
        self._dialogue_mode = any("role" in m for m in self._messages)
        self._current_query = deepcopy(episode["current_query"])
        self._future_query = deepcopy(episode["future_query"])
        self._simulator = UserSimulator(episode.get("replies", {}), root_id=episode.get("root_id", ""),
                                        nonresponse_rate=self.nonresponse_rate, imperfections=self.imperfections)
        self._reply = None
        self._closed = False
        self._reopened = False
        self.question_count = 0
        self.private_trace = {"observations": {}, "initial_memory": self.store.snapshot(),
                              "initial_memory_hash": self.store.hash(), "rollout_seed": rollout_seed,
                              "user_messages": deepcopy(self._messages), "user_reply": None,
                              "user_response_status": "not_asked",
                              "executed_operations": [], "write_result": None}
        if self.imperfections.active:
            self.private_trace["imperfections"] = {"version": IMPERFECTION_VERSION,
                "config": self.imperfections.to_dict(), "retrieval": {}, "reply": None,
                "reuse_memory": "complete_persisted_store_no_retrieval_fault"}
        return self.observe("select")

    def observe(self, phase, condition="O2", intervention=None):
        phase = phase.lower()
        if self._closed:
            raise RuntimeError("SESSION_CLOSED")
        if phase not in {"select", "respond", "write"}:
            raise ValueError("INVALID_PHASE")
        if condition not in {"O1", "O2"}:
            raise ValueError("INVALID_OBSERVATION_CONDITION")
        if intervention not in {None, "I0", "I1", "I2"}:
            raise ValueError("INVALID_INTERVENTION")
        memory = self.store.list_records()
        if condition == "O1":
            relevant = [r for r in memory if r["attribute"] == self._current_query["attribute"]]
            memory = [max(enumerate(relevant), key=lambda item: (item[1].get("recorded_at", ""), item[0]))[1]] if relevant else []
        if self.imperfections.active:
            memory, audit = omit_memory(memory, self.imperfections, self._imperfection_root_id)
            self.private_trace["imperfections"]["retrieval"][phase + ":" + condition] = audit
        messages = deepcopy(self._messages)
        visible_reply = self._reply is not None and phase != "select"
        if intervention == "I0" or (intervention == "I1" and phase == "write"):
            visible_reply = False
        if visible_reply:
            messages.append(deepcopy(self._reply))
        observation = {"memory": memory, "user_messages": messages,
                       "query": deepcopy(self._current_query),
                       "reuse_regime": "Habrá una consulta posterior del mismo atributo sin posibilidad de preguntar al usuario."}
        self.private_trace["observations"][phase] = deepcopy(observation)
        return observation

    def ask(self, slot):
        if self._closed:
            raise RuntimeError("SESSION_CLOSED")
        if self.question_count >= 1:
            raise ValueError("QUESTION_BUDGET_EXCEEDED")
        reply = self._simulator.reply(slot)
        if reply["turn_id"] in {m["turn_id"] for m in self._messages}:
            raise ValueError("SIMULATED_REPLY_TURN_COLLISION")
        self.question_count += 1
        if self._dialogue_mode:
            question_id = "a_tool_" + str(self.question_count)
            if question_id in {m["turn_id"] for m in self._messages}:
                raise ValueError("SIMULATED_QUESTION_TURN_COLLISION")
            attribute = self._current_query["attribute"]
            names = {"dinner_preference": "la cena", "drink_preference": "la bebida", "breakfast_preference": "el desayuno"}
            self._messages.append({"turn_id": question_id, "role": "assistant",
                "text": "Sobre " + names.get(attribute, attribute) + ": " + QUESTIONS[slot],
                "about_attribute": attribute})
            reply = {**reply, "role": "user", "in_reply_to": question_id, "about_attribute": attribute}
        self._reply = deepcopy(reply)
        self.private_trace["asked_slot"] = slot
        self.private_trace["user_reply"] = deepcopy(reply)
        self.private_trace["user_response_status"] = self._simulator.last_response_status
        if self.imperfections.active:
            self.private_trace["imperfections"]["reply"] = deepcopy(self._simulator.last_reply_audit)
        return deepcopy(reply)

    def apply_memory_operations(self, operations):
        if self._closed:
            raise RuntimeError("SESSION_CLOSED")
        result = Executor(self.store).execute(operations, recorded_at=self._current_query["date"])
        self.private_trace["write_result"] = deepcopy(result)
        self.private_trace["executed_operations"].extend(deepcopy(result["executed_operations"]))
        return result

    def close_session(self):
        if self._closed:
            return
        self.private_trace["final_memory"] = self.store.snapshot()
        self.private_trace["final_memory_hash"] = self.store.hash()
        self.store.close()
        self.store = None
        self._messages.clear()
        self._reply = None
        self._current_query = None
        self._simulator.clear()
        self._initial_episode = None
        self._closed = True

    def reopen_for_reuse(self):
        if not self._closed:
            raise RuntimeError("CLOSE_SESSION_BEFORE_REUSE")
        if self.store is not None:
            self.store.close()
        self.store = Store(self.db_path)
        self._reopened = True
        observation = {"memory": self.store.list_records(), "query": deepcopy(self._future_query)}
        self.private_trace["reuse_observation"] = deepcopy(observation)
        return observation

    def dispose(self):
        if self.store is not None:
            self.store.close()
            self.store = None


Environment = EpisodeEnvironment
