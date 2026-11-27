"""A two-session episode with one question budget and a fixed memory controller."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json

from research.src.artifacts import canonical
from research.src.policy.rule_controller import RuleController
from research.src.readers.symbolic import read
from research.src.scoring.evidence import audit_operation, score_response
from research.tesis_interactive_env import (
    ACTIONS as STOCK_ACTIONS,
    MAX_QUESTIONS,
    InteractiveEpisode,
    _address,
    _native,
    _possible_values,
    public_evidence,
    reference_action,
)


ACTIONS = (*STOCK_ACTIONS, "PROVISIONAL", "DEFER")
QUESTION_ACTIONS = STOCK_ACTIONS[1:]
CLOSE_ACTIONS = ("PROCEED", "PROVISIONAL", "DEFER")
VERIFIERS = ("veto", "off")


def decision_state(observation):
    """Return the compact public representation for a decision policy."""
    return {
        **public_evidence(observation),
        "strategy_cue": observation["strategy_cue"],
        "visit": observation["visit"],
        "questions_remaining": observation["questions_remaining"],
        "asked_slots": tuple(observation["asked_slots"]),
    }


def rule_action(observation, legal_actions):
    """Use the established evidence rule, subject to the shared budget."""
    action = reference_action(observation)
    return action if action in legal_actions else "PROCEED"


@lru_cache(maxsize=8192)
def _fixed_decision(observation_text):
    """Tag each confirmed proposal with its evidence audit result."""
    observation = json.loads(observation_text)
    controller = RuleController()
    response = controller.respond(observation)
    proposed = [operation for operation in controller.write(observation)["operations"]
                if operation.get("record", {}).get("status") == "confirmed"]
    return response, tuple((operation, audit_operation(operation, observation)) for operation in proposed)


def _expired_citations(records, response, query):
    cited = set(response["evidence_ids"])
    return [record["memory_id"] for record in records if record["memory_id"] in cited
            and _expired(record["scope"], query)]


def _expired(scope, query):
    expiry = scope.get("valid_until_exclusive")
    return ((expiry is not None and query["date"] >= expiry)
            or (scope["kind"] == "session" and scope.get("session_id") != query.get("session_id")))


class SharedBudgetEpisode(InteractiveEpisode):
    """Run native interactions and SQLite writes with questions shared across visits.

    PROCEED applies the fixed response and writer, and carries the remaining
    budget forward. PROVISIONAL gives the fixed recommendation for this visit
    without saving a new preference. DEFER postpones the recommendation and
    leaves memory alone. An optional write-veto ablation uses the same proposed
    operations, including confirmation status, with the evidence veto disabled.
    """

    def __init__(self, story, db_path=":memory:", *, question_budget=1, verifier="veto"):
        if len(story["visits"]) != 2:
            raise ValueError("The shared-budget pilot requires two visits")
        if type(question_budget) is not int or not 0 <= question_budget <= 2 * MAX_QUESTIONS:
            raise ValueError("Invalid shared question budget")
        if verifier not in VERIFIERS:
            raise ValueError("Invalid write verifier mode")
        self.question_budget = question_budget
        self.verifier = verifier
        self._questions_used = 0
        self._unsupported_memory_ids = set()
        super().__init__(story, db_path)

    def observe(self):
        observation = super().observe()
        return {
            **observation,
            "question_budget": self.question_budget,
            "questions_remaining": self.question_budget - self._questions_used,
            "strategy_cue": self._visit.get("strategy_cue"),
        }

    @property
    def legal_actions(self):
        stock = super().legal_actions
        if not stock:
            return ()
        if len(self._events) >= MAX_QUESTIONS or len(stock) == 1:
            return CLOSE_ACTIONS
        return ACTIONS

    def step(self, action):
        if action not in self.legal_actions:
            raise ValueError(f"Illegal shared-budget action: {action}")
        self._actions.append(action)
        if action in CLOSE_ACTIONS:
            self._finish_visit(action)
        else:
            self._ask(action)
            self._questions_used += 1
        return (None, True) if self.done else (self.observe(), False)

    def _finish_visit(self, action):
        visible = _native(self.observe())
        fixed_response, candidates = _fixed_decision(canonical(visible))
        response = (deepcopy(fixed_response) if action != "DEFER" else
                    {"mode": "DEFER", "value": None, "branches": [], "evidence_ids": []})
        proposed = candidates if action == "PROCEED" else ()
        accepted = [(operation, supported) for operation, supported in proposed
                    if supported or self.verifier == "off"]
        operations = [deepcopy(operation) for operation, _ in accepted]
        write = self._env.apply_memory_operations(operations or [{"op": "NOOP"}])
        self._visits.append(self._visit_trace(action, visible, response, proposed, accepted, write))
        if len(self._visits) == 1:
            self._open(1, self._visits[-1]["memory_after"])
        else:
            self._finish_episode()

    def _visit_trace(self, action, visible, response, proposed, accepted, write):
        visit = self._visit
        if visit["fact"] is not None:
            alternatives = visit["alternatives"] if not self._resolved else None
            self._knowledge.append(tuple(alternatives or [visit["fact"]]))
        current = score_response(response, visible, visit["expected"])
        query = _address(visible["query"], visit, self._resolved)
        expired = _expired_citations(visible["memory"], response, query)
        bad_citations = self._unsupported_memory_ids.intersection(response["evidence_ids"])
        supported = current["success"] and not expired and not bad_citations
        if self._visits and self.story["second_visit_kind"] == "reuse":
            supported = supported and _possible_values(self._knowledge, visit["query"]) == {response["value"]}
        if visit["slot"] == "CURRENT_CONTEXT" and not self._resolved:
            supported = False
        self._env.close_session()
        unsupported = {applied["record"]["memory_id"]
                       for applied, (_, justified) in zip(write["executed_operations"], accepted)
                       if not justified}
        self._unsupported_memory_ids.update(unsupported)
        return {
            "query": visible["query"],
            "effective_query": query,
            "opening_messages": self._opening,
            "events": self._events,
            "closing_action": action,
            "session_only_recommendation": action == "PROVISIONAL" and response["mode"] == "ANSWER",
            "expected": visit["expected"],
            "outcome": response,
            "current_supported": int(supported),
            "current_correct": current["factual"],
            "unsupported_commitments": int(response["mode"] == "ANSWER" and not supported),
            "expired_reuse": int(bool(expired)),
            "questions": len(self._events),
            "questions_remaining": self.question_budget - self._questions_used,
            "invalid_outputs": int(not write["valid"] or not current["valid"]),
            "unsupported_writes": len(unsupported),
            "unsupported_write_ids": sorted(unsupported),
            "rejected_proposals": len(proposed) - len(accepted),
            "omitted_confirmed_updates": (len(proposed) - len(accepted)
                                         + int(not write["valid"])),
            "informative_questions": sum(event["informative"] for event in self._events),
            "failed_questions": sum(event["delivery"] in ("unavailable", "irrelevant", "redundant")
                                    for event in self._events),
            "redundant_questions": sum(event["delivery"] in ("irrelevant", "redundant")
                                       for event in self._events),
            "executed_operations": write["executed_operations"],
            "memory_before": visible["memory"],
            "memory_after": self._env.private_trace["final_memory"],
        }

    def _finish_episode(self):
        self._final = self._env.reopen_for_reuse()["memory"]
        self._future = []
        for probe in self.story["future_queries"]:
            query, expected = probe["query"], probe["expected"]
            response = read(self._final, query)
            answerable = response["mode"] == "ANSWER"
            expired = _expired_citations(self._final, response, query)
            bad_citations = self._unsupported_memory_ids.intersection(response["evidence_ids"])
            correct = answerable and response["value"] == expected
            supported = (correct and not expired and not bad_citations
                         and _possible_values(self._knowledge, query) == {expected})
            self._future.append({
                "kind": probe["kind"],
                "query": query,
                "response": response,
                "expected": expected,
                "correct": int(correct),
                "supported": int(supported),
                "unsupported_commitment": int(answerable and not supported),
                "expired_reuse": int(bool(expired)),
                "expired_citation_ids": expired,
                "unsupported_citation_ids": sorted(bad_citations),
            })

    def summary(self):
        return {**super().summary(), "question_budget": self.question_budget,
                "questions_remaining": self.question_budget - self._questions_used,
                "verifier": self.verifier}
