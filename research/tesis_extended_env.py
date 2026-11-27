"""Three consecutive SQLite visits with a shared budget for clarification."""

from __future__ import annotations

from copy import deepcopy

from research.src.artifacts import canonical
from research.src.readers.symbolic import read
from research.tesis_interactive_env import InteractiveEpisode, MAX_QUESTIONS, _native, _possible_values, public_evidence
from research.tesis_shared_budget_env import (
    ACTIONS, QUESTION_ACTIONS, VERIFIERS, SharedBudgetEpisode, _expired_citations, _fixed_decision,
)
from research.tesis_shared_budget_planner import (
    Decision, WorldTree, group_trees, outcome as base_outcome, traverse, weighted,
)


TEMPORAL_QUESTIONS = ("TEMPORAL_SCOPE", "VALID_UNTIL", "CURRENT_CONTEXT")
PARTIAL_REPLIES = {
    "TEMPORAL_SCOPE": "Por ahora me sirve. Déjame mirar mi agenda para poder precisar el alcance.",
    "VALID_UNTIL": "Es por unos días. Tengo que mirar cuándo termina.",
    "CURRENT_CONTEXT": "Aún no llego. Déjame ver dónde estaré.",
}


def rule_action(observation, legal):
    """Clarify the visible ambiguity at its first opportunity."""
    slot = public_evidence(observation)["missing_slot"]
    return slot if slot in QUESTION_ACTIONS and slot in legal else "PROCEED"


class ExtendedEpisode(SharedBudgetEpisode):
    """Reuse the native writer and reader over three decision-bearing visits."""

    def __init__(self, story, db_path=":memory:", *, question_budget=None, verifier="veto"):
        if len(story["visits"]) != 3:
            raise ValueError("Three visits are required")
        budget = story["question_budget"] if question_budget is None else question_budget
        if type(budget) is not int or not 0 <= budget <= 2:
            raise ValueError("The extended pilot accepts from zero to two shared questions")
        if verifier not in VERIFIERS:
            raise ValueError("Invalid write verifier mode")
        self.question_budget = budget
        self.verifier = verifier
        self._questions_used = 0
        self._unsupported_memory_ids = set()
        InteractiveEpisode.__init__(self, story, db_path)

    @property
    def legal_actions(self):
        if self.done:
            return ()
        visible = self.observe()
        evidence = public_evidence(visible)
        slot = evidence["missing_slot"]
        # Expose a provisional alternative only for an observable session-only
        # write. This retains the choice when it actually changes SQLite.
        _, candidates = _fixed_decision(canonical(_native(visible)))
        provisional = any(
            approved and operation["record"]["scope"]["kind"] == "session"
            for operation, approved in candidates
        )
        closing = ("PROCEED",)
        if provisional:
            closing += ("PROVISIONAL",)
        if not evidence["current_resolved"]:
            closing += ("DEFER",)
        if not visible["questions_remaining"] or len(self._events) >= MAX_QUESTIONS:
            return closing
        if slot not in TEMPORAL_QUESTIONS or evidence["clarification_state"] == "unavailable":
            return closing
        if slot in visible["asked_slots"] and evidence["clarification_state"] != "partial":
            return closing
        return tuple(action for action in ACTIONS if action in (*closing, slot))

    def _ask(self, action):
        visit = self._visit
        self._attempts[action] += 1
        if action != visit["slot"]:
            text, delivery = "Ese detalle sigue sin aclarar lo que necesito.", "irrelevant"
        elif visit["style"] == "refused":
            text, delivery = "No lo puedo precisar; mejor dejarlo ahí por ahora.", "unavailable"
        elif visit["style"] == "partial" and self._attempts[action] == 1:
            text, delivery = visit.get("partial_answer") or PARTIAL_REPLIES[action], "partial"
        else:
            text, delivery = visit["answer"], "resolved"
            if not isinstance(text, str) or not text.strip():
                raise ValueError("A resolved clarification must contain public evidence")
            self._resolved = True
        question, reply = self._env.deliver(action, text)
        self._events.append({"action": action, "question": question, "reply": reply,
                             "delivery": delivery, "informative": delivery == "resolved"})

    def _propose_writes(self, candidates, visible):
        """Hook to intervene in the writer while holding the selected actions fixed."""
        return candidates

    def _finish_visit(self, action):
        visible = _native(self.observe())
        fixed_response, candidates = _fixed_decision(canonical(visible))
        response = (deepcopy(fixed_response) if action != "DEFER" else
                    {"mode": "DEFER", "value": None, "branches": [], "evidence_ids": []})
        proposed = self._propose_writes(deepcopy(candidates), visible) if action == "PROCEED" else ()
        accepted = [(operation, justified) for operation, justified in proposed
                    if justified or self.verifier == "off"]
        operations = [deepcopy(operation) for operation, _ in accepted]
        written = self._env.apply_memory_operations(operations or [{"op": "NOOP"}])
        trace = self._visit_trace(action, visible, response, proposed, accepted, written)
        if self._visit["fact"] is None and trace["current_supported"]:
            if _possible_values(self._knowledge, trace["effective_query"]) != {response["value"]}:
                trace["current_supported"] = 0
                trace["unsupported_commitments"] = 1
        self._visits.append(trace)
        if len(self._visits) < len(self.story["visits"]):
            self._env.dispose()
            self._open(len(self._visits), trace["memory_after"])
        else:
            self._finish_episode()

    def _read_future(self, query):
        """Hook to intervene in retrieval after the same SQLite operations."""
        return read(self._final, query)

    def _finish_episode(self):
        self._final = self._env.reopen_for_reuse()["memory"]
        self._future = []
        for probe in self.story["future_queries"]:
            query, expected = probe["query"], probe["expected"]
            response = self._read_future(query)
            answerable = response["mode"] == "ANSWER"
            expired = _expired_citations(self._final, response, query)
            bad_citations = self._unsupported_memory_ids.intersection(response["evidence_ids"])
            correct = answerable and response["value"] == expected
            supported = (correct and not expired and not bad_citations
                         and _possible_values(self._knowledge, query) == {expected})
            self._future.append({
                "kind": probe["kind"], "query": query, "response": response, "expected": expected,
                "correct": int(correct), "supported": int(supported),
                "unsupported_commitment": int(answerable and not supported),
                "expired_reuse": int(bool(expired)), "expired_citation_ids": expired,
                "unsupported_citation_ids": sorted(bad_citations),
            })

    def summary(self):
        return {**super().summary(), "plan_change": self.story.get("plan_change")}


def outcome(trace):
    events = [event for visit in trace["visits"] for event in visit["events"]]
    return {
        **base_outcome(trace),
        "questions_third": trace["visits"][2]["questions"],
        "plan_canceled": int(trace.get("plan_change") == "canceled"),
        "partial_replies": sum(event["delivery"] == "partial" for event in events),
        "partial_followups": sum(
            any(first["delivery"] == "partial" and second["delivery"] == "resolved"
                for first, second in zip(visit["events"], visit["events"][1:]))
            for visit in trace["visits"]
        ),
    }


def execute(story, prefix, *, episode_class=ExtendedEpisode, **episode_options):
    """Replay a public action path and return the raw trace or the next decision."""
    env = episode_class(story, **episode_options)
    try:
        for action in prefix:
            env.step(action)
        if env.done:
            return env.summary()
        return Decision({**env.observe(), "action_history": tuple(prefix)}, tuple(env.legal_actions))
    finally:
        env.close()


def simulate(story, prefix, *, episode_class=ExtendedEpisode, **episode_options):
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


def evaluate(trees, policy, *, method, seed=0):
    """Aggregate worlds within independent roots and audit comparable removals."""
    rows = []
    for root_id, worlds in group_trees(trees).items():
        first_actions, measured_worlds = {}, []
        for tree in worlds:
            path, measured = traverse(tree, policy)
            first_actions[path[0]] = first_actions.get(path[0], 0.) + tree.mass
            reductions = [
                tree.outcomes[path[:index] + path[index + 1:]]
                for index, action in enumerate(path)
                if action in QUESTION_ACTIONS and path[:index] + path[index + 1:] in tree.outcomes
            ]
            # Removal may make an evidence-dependent closure unavailable.
            # Track coverage before inferring that any question was needed.
            unnecessary = sum(
                reduced["future_supported"] >= measured["future_supported"]
                and reduced["current_supported"] >= measured["current_supported"]
                and reduced["future_unsupported_reuse"] <= measured["future_unsupported_reuse"]
                and reduced["expired_reuse"] <= measured["expired_reuse"]
                and reduced["unsupported_writes"] <= measured["unsupported_writes"]
                for reduced in reductions
            )
            measured_worlds.append((tree.mass, {
                **measured, "counterfactual_evaluable_questions": len(reductions),
                "counterfactually_unnecessary_questions": unnecessary,
            }))
        rows.append({
            "method": method, "seed": seed, "root_id": root_id,
            "scenario_family": worlds[0].story["scenario_family"], "first_actions": first_actions,
            "metrics": {name: weighted((mass, values[name]) for mass, values in measured_worlds)
                        for name in measured_worlds[0][1]},
        })
    return rows
