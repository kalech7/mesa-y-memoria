"""Profiles that can answer, clarify partially, or recontact in a later visit."""

from __future__ import annotations

from copy import deepcopy
import re

from research.tesis_extended_env import (
    ExtendedEpisode, evaluate, execute as execute_extended, outcome as extended_outcome, rule_action,
)
from research.tesis_interactive_env import public_evidence
from research.tesis_shared_budget_planner import Decision, WorldTree
from research.tesis_shared_budget_qwen_context import _plain, _text_cues


RETURN_HINT = re.compile(
    r"\b(?:te (?:contesto|respondo|digo|confirmo|escribo)|"
    r"vuelvo a (?:contestar|responder|escribirte)|"
    r"retomo (?:el chat|la charla|la conversacion))\b"
)
DEFERRED_REPLY = "Ahora me voy; mejor dejarlo ahí. Te respondo cuando vuelva."


def _visible_timing(observation):
    messages = observation["user_messages"]
    availability, later = _text_cues(messages)
    user_text = _plain(" ".join(turn["text"] for turn in messages if turn["role"] == "user"))
    return {
        "availability": availability, "later_plan": later,
        "expressed_return": bool(RETURN_HINT.search(user_text)),
        "observed_recontact": any(turn.get("recontact") is True and turn["role"] == "user"
                                  for turn in messages),
    }


def public_features(observation, legal=()):
    """Describe timing and evidence solely from messages already shown."""
    evidence = public_evidence(observation)
    return {
        **_visible_timing(observation),
        "visit": observation["visit"], "questions_remaining": observation["questions_remaining"],
        "clarification_state": evidence["clarification_state"],
        "current_resolved": evidence["current_resolved"], "missing_slot": evidence["missing_slot"],
        "legal_actions": tuple(legal),
    }


class ProfilesEpisode(ExtendedEpisode):
    """Queue a later user's message only after the policy buys that opportunity."""

    def __init__(self, story, db_path=":memory:", *, question_budget=None, verifier="veto"):
        self._pending = {}
        self._delivered = []
        super().__init__(story, db_path, question_budget=question_budget, verifier=verifier)

    def _open(self, index, records):
        super()._open(index, records)
        for clarification in self._pending.pop(index + 1, ()):
            turn = {
                "turn_id": f"return_v{index + 1}_{clarification['origin']}",
                "role": "user", "text": clarification["message"], "recontact": True,
            }
            # A recontact is fresh evidence in this session, before the question.
            self._env._messages.insert(-1, deepcopy(turn))
            self._opening.insert(-1, turn)
            knowledge_index = clarification["knowledge_index"]
            if knowledge_index is not None:
                self._knowledge[knowledge_index] = (clarification["fact"],)
            self._delivered.append({"origin": clarification["origin"], "visit": index + 1,
                                    "turn_id": turn["turn_id"]})

    def observe(self):
        view = super().observe()
        features = _visible_timing(view)
        # Existing planning tables use this key; it carries observed cues only.
        view["strategy_cue"] = (
            features["availability"], features["later_plan"],
            features["expressed_return"], features["observed_recontact"],
        )
        return view

    def _ask(self, action):
        if self._visit["style"] != "deferred" or action != self._visit["slot"]:
            super()._ask(action)
            return
        visit = self._visit
        origin = len(self._visits) + 1
        delayed = visit["recontact"]
        target, message = delayed["visit"], delayed["message"]
        if type(target) is not int or not origin < target <= len(self.story["visits"]):
            raise ValueError("A recontact must belong to a later visit")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("A recontact must contain a visible user message")
        answer = visit.get("deferred_reply", DEFERRED_REPLY)
        question, reply = self._env.deliver(action, answer)
        self._attempts[action] += 1
        self._events.append({"action": action, "question": question, "reply": reply,
                             "delivery": "deferred", "informative": False})
        self._pending.setdefault(target, []).append({
            "origin": origin, "message": message, "fact": visit["fact"],
            "knowledge_index": len(self._knowledge) if visit["fact"] is not None else None,
        })

    def summary(self):
        return {**super().summary(), "recontacts": deepcopy(self._delivered)}


def outcome(trace):
    events = [event for visit in trace["visits"] for event in visit["events"]]
    return {
        **extended_outcome(trace),
        "deferred_replies": sum(event["delivery"] == "deferred" for event in events),
        "delivered_recontacts": len(trace["recontacts"]),
        "context_questions": sum(event["action"] == "CURRENT_CONTEXT" for event in events),
    }


def execute(story, prefix, *, episode_class=ProfilesEpisode, **episode_options):
    return execute_extended(story, prefix, episode_class=episode_class, **episode_options)


def simulate(story, prefix, *, episode_class=ProfilesEpisode, **episode_options):
    result = execute(story, prefix, episode_class=episode_class, **episode_options)
    return result if isinstance(result, Decision) else outcome(result)


def enumerate_world(story):
    pending = [()]
    decisions, outcomes = {}, {}
    steps = 0
    while pending:
        prefix = pending.pop()
        result = simulate(story, prefix)
        steps += len(prefix)
        if isinstance(result, Decision):
            decisions[prefix] = result
            pending.extend(prefix + (action,) for action in result.legal)
        else:
            outcomes[prefix] = result
    return WorldTree(story, decisions, outcomes, steps)
