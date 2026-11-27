"""Three visits with competing opportunities to clarify a food preference.

The policy sees the conversation and its own remaining questions. Latent plan
changes, author labels and future query keys are available only to the simulator
and to the evaluator after a branch terminates.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
import re
from statistics import mean
from unicodedata import normalize

from research.src.data.build_bank import scope
from research.src.memory.executor import Executor
from research.src.memory.store import Store
from research.src.readers.symbolic import read
from research.tesis_shared_budget_planner import Decision, group_trees, traverse, weighted


ACTIONS = ("WAIT", "ASK")
QUESTION_COST = .08
FUTURE_WEIGHT = 1.5
UNSUPPORTED_COST = .35
ATTRIBUTES = {"dinner_preference": "cena", "breakfast_preference": "desayuno", "drink_preference": "bebida"}


@dataclass
class WorldTree:
    story: dict
    budget: int
    decisions: dict
    outcomes: dict
    environment_steps: int

    @property
    def mass(self):
        return self.story["mass"]


def _turn(role, identity, text):
    return {"role": role, "turn_id": identity, "text": text}


def _fold(text):
    return "".join(c for c in normalize("NFKD", text.casefold()) if not 0x300 <= ord(c) <= 0x36F)


def prompt_text(observation):
    """Serialize public evidence only, with no branch, root, or hidden plan IDs."""
    history = "\n".join(f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['text']}"
                        for m in observation["messages"])
    records = "\n".join(f"{ATTRIBUTES[m['attribute']]}: {m['value']} ({m['scope']['kind']}, "
                        + " / ".join(q["text"] for q in m["evidence_quotes"]) + ")"
                        for m in observation["memory"])
    return (f"Visita {observation['visit']} de 3. Preguntas disponibles: {observation['questions_remaining']}.\n"
            f"{history}\nMemoria confirmada: {records or 'vacía'}.\n"
            f"Si aclaras ahora, preguntarás: {observation['question']}")


def _resolve_scope(reply):
    """Controlled reply language supplied by the simulator, after a paid ASK."""
    response = _fold(reply)
    if re.search(r"\bsolo por hoy\b", response):
        return "session"
    if re.search(r"\bhabitual\b", response) and not re.search(r"\bno\b[^.!?]{0,30}\bhabitual\b", response):
        return "ongoing"
    return None


def _usable_value(visit):
    # The normalized food mention is a public structured field for the fixed
    # responder. Ground it in the utterance before replying or writing.
    value = visit["value"]
    return value if value and _fold(value) in _fold(visit["utterance"]) else None


def _apply(store, visit, number, action, messages):
    value = _usable_value(visit)
    written = []
    source_turn = messages[-1]
    reply = f"Para esta visita, {value}." if value else "Necesito más información para recomendar algo."
    if action == "ASK":
        messages.append(_turn("assistant", f"a{number}-q", visit["question"]))
        answered = _turn("user", f"u{number}-r", visit["reply"])
        messages.append(answered)
        kind = _resolve_scope(answered["text"])
        if kind and value:
            record = {
                "attribute": visit["attribute"], "value": value, "status": "confirmed",
                "scope": scope(kind, start=visit["date"], session=f"S{number}" if kind == "session" else None),
                "source_turn_ids": [source_turn["turn_id"], answered["turn_id"]],
                "evidence_quotes": [{"turn_id": turn["turn_id"], "text": turn["text"]}
                                    for turn in (source_turn, answered)],
            }
            executed = Executor(store).execute([{"op": "ADD", "record": record}], recorded_at=visit["date"])
            if not executed["valid"]:
                raise ValueError(f"Invalid v6 memory: {executed['error']}")
            written = executed["executed_operations"]
    messages.append(_turn("assistant", f"a{number}-response", reply))
    return {
        "visit": number, "action": action, "current_value": value,
        "current_evidence_ids": [source_turn["turn_id"]] if value else [],
        "current_supported": int(value is not None), "executed_operations": written,
    }


def _close(story, records, messages, visits, budget):
    terminal = story["visits"][2]
    messages.append(_turn("user", "u3", terminal["utterance"]))
    visible_text = {turn["turn_id"]: turn["text"] for turn in messages if turn["role"] == "user"}
    by_id = {record["memory_id"]: record for record in records}
    future = []
    for item in story["future_queries"]:
        query = {"attribute": item["attribute"], "date": item["date"], "session_id": "S3",
                 "context": item.get("context")}
        result = read(records, query)
        cited = [by_id[identity] for identity in result["evidence_ids"]]
        grounded = bool(cited) and all(
            source["value"] == result["value"]
            and all(visible_text.get(q["turn_id"]) == q["text"] for q in source["evidence_quotes"])
            for source in cited)
        answered = result["mode"] == "ANSWER"
        gold = item["expected_value"]
        supported_answer = answered and grounded and result["value"] == gold
        abstained_correctly = result["mode"] == "DEFER" and gold is None
        future.append({"query": query, "expected_value": gold, "response": result,
                       "correct": int(supported_answer or abstained_correctly),
                       "positive_supported": int(supported_answer), "correct_abstention": int(abstained_correctly),
                       "unsupported": int(answered and not supported_answer)})
    if not future:
        raise ValueError("V6 needs actual future queries")
    questions = [int(visit["action"] == "ASK") for visit in visits]
    current = mean(visit["current_supported"] for visit in visits)
    score = mean(probe["correct"] for probe in future)
    reuse = mean(probe["unsupported"] for probe in future)
    assert len(visits) == 2 and sum(questions) <= budget
    metrics = {
        "current_supported": current, "current_first_supported": visits[0]["current_supported"],
        "current_second_supported": visits[1]["current_supported"], "future_supported": score,
        "future_unsupported_reuse": reuse, "future_queries": len(future),
        "future_positive_queries": sum(row["expected_value"] is not None for row in future),
        "future_positive_supported": sum(row["positive_supported"] for row in future),
        "future_correct_abstentions": sum(row["correct_abstention"] for row in future),
        "questions": sum(questions), "questions_first": questions[0], "questions_second": questions[1],
        "applied_writes": len(records), "immediate": current - QUESTION_COST * sum(questions),
        "longitudinal": current + FUTURE_WEIGHT * score - QUESTION_COST * sum(questions) - UNSUPPORTED_COST * reuse,
    }
    return {"metrics": metrics, "visits": visits, "future": future, "memory": records,
            "messages": messages, "question_budget": budget}


def replay(story, prefix, budget=1):
    """Return the next public decision, or the full evidence and terminal outcomes."""
    if type(budget) is not int or budget not in (1, 2) or len(prefix) > 2:
        raise ValueError("Use a one- or two-question budget across two decision points")
    if len(story["visits"]) != 3:
        raise ValueError("Each story must span three visits")
    messages, visited = [], []
    with closing(Store(":memory:")) as store:
        for index, visit in enumerate(story["visits"][:2]):
            number = index + 1
            messages.append(_turn("user", f"u{number}", visit["utterance"]))
            remaining = budget - prefix[:index].count("ASK")
            legal = ACTIONS if remaining else ("WAIT",)
            if index == len(prefix):
                return Decision({"visit": number, "date": visit["date"], "messages": messages,
                                 "cue": story["cue"], "question": visit["question"],
                                 "questions_remaining": remaining, "memory": store.list_records()}, legal)
            action = prefix[index]
            if action not in legal:
                raise ValueError(f"Illegal action {action!r} at visit {number} with {remaining} questions")
            visited.append(_apply(store, visit, number, action, messages))
        return _close(story, store.list_records(), messages, visited, budget)


def enumerate_world(story, budget=1):
    pending, decisions, outcomes, steps = [()], {}, {}, 0
    while pending:
        prefix = pending.pop()
        result = replay(story, prefix, budget)
        steps += len(prefix)
        if isinstance(result, Decision):
            decisions[prefix] = result
            pending.extend(prefix + (action,) for action in result.legal)
        else:
            outcomes[prefix] = result["metrics"]
    return WorldTree(story, budget, decisions, outcomes, steps)


# Topic names come only from what the user said, never the authored variant tag.
_TOPIC = {
    "dinner": re.compile(r"\b(?:cena\w*|noches?|nocturn\w*)\b"),
    "morning": re.compile(r"\b(?:desayun\w*|matutin\w*|meriendas?|bebidas?|beber|hidrat\w*)\b"),
}
_REPEAT = re.compile(r"\b(?:semana\w*|cinco|varios|varias|repet\w*|repit\w*|cada|todos|todas|turnos|de lunes a viernes|diari\w*)\b")
_ONCE = re.compile(r"\b(?:sol[oa]s?|unic[oa]s?|una vez|un dia|una noche)\b")


def _agenda_topic(cue):
    scores = Counter()
    for clause in re.split(r"[.,;:!?]|\b(?:pero|mientras|y)\b", _fold(cue)):
        mentioned = [topic for topic, pattern in _TOPIC.items() if pattern.search(clause)]
        if len(mentioned) == 1:
            scores[mentioned[0]] += int(bool(_REPEAT.search(clause))) - int(bool(_ONCE.search(clause)))
    return "dinner" if scores["dinner"] > scores["morning"] else (
        "morning" if scores["morning"] > scores["dinner"] else "unknown")


def _state(obs):
    return obs["visit"], obs["questions_remaining"], _agenda_topic(obs["cue"])


class PublicPlanner:
    """Backward induction on TRAIN returns grouped by visible agenda and capacity."""

    def __init__(self):
        self.values = {}

    def margin(self, observation, legal):
        if "ASK" not in legal:
            return float("-inf")
        visit, remaining, agenda = _state(observation)
        row = self.values.get((visit, remaining, agenda), self.values.get((visit, remaining, None)))
        if row is None:
            return 0.
        return row["ASK"] - row["WAIT"]

    def __call__(self, observation, legal):
        return "ASK" if self.margin(observation, legal) > 1e-12 else "WAIT"


def fit_public_planner(train_trees):
    trees = list(train_trees)
    if not trees or any(tree.story["split"] != "train" for tree in trees):
        raise ValueError("The public planner fits TRAIN only")
    group_trees(trees)
    planner = PublicPlanner()
    for depth in (1, 0):
        buckets = defaultdict(lambda: defaultdict(list))
        for tree in trees:
            for path, decision in tree.decisions.items():
                if len(path) != depth or len(decision.legal) < 2:
                    continue
                key = _state(decision.observation)
                for action in decision.legal:
                    _, outcome = traverse(tree, planner, path + (action,))
                    for state in (key, (*key[:2], None)):
                        buckets[state][action].append((tree.mass, outcome["longitudinal"]))
        for state, by_action in buckets.items():
            planner.values[state] = {action: weighted(values) for action, values in by_action.items()}
    return planner


def _summarize(trees_by_root, policy):
    root_metrics = []
    for worlds in trees_by_root.values():
        results = [(tree.mass, traverse(tree, policy)[1]) for tree in worlds]
        root_metrics.append({key: weighted((mass, metrics[key]) for mass, metrics in results)
                             for key in results[0][1]})
    return {key: mean(root[key] for root in root_metrics) for key in root_metrics[0]}


def cpu_audit(train_trees, eval_trees):
    """Spend exactly one question in each forced branch and average latent worlds first."""
    train, evaluation = list(train_trees), list(eval_trees)
    if len({t.budget for t in [*train, *evaluation]}) != 1:
        raise ValueError("Compare fitting and evaluation at a single common budget")
    planner = fit_public_planner(train)
    by_root = group_trees(evaluation)
    if not by_root:
        raise ValueError("Cannot audit an empty evaluation")
    policies = {"planificador-publico": planner,
                "preguntar-primero": lambda obs, legal: "ASK" if obs["visit"] == 1 else "WAIT",
                "preguntar-segundo": lambda obs, legal: "ASK" if obs["visit"] == 2 else "WAIT",
                "preguntar-siempre": lambda obs, legal: "ASK" if "ASK" in legal else "WAIT",
                "nunca": lambda obs, legal: "WAIT"}
    roots, pairs, public_calls = [], defaultdict(dict), Counter()
    for root_id, worlds in by_root.items():
        first = worlds[0].story
        now = weighted((t.mass, t.outcomes[("ASK", "WAIT")]["future_supported"]) for t in worlds)
        later = weighted((t.mass, t.outcomes[("WAIT", "ASK")]["future_supported"]) for t in worlds)
        if first["variant"] not in ("ask_now", "save") or first["variant"] in pairs[first["pair_id"]]:
            raise ValueError("Each timing pair needs two different public agendas")
        difference = now - later
        pairs[first["pair_id"]][first["variant"]] = difference
        roots.append({"root_id": root_id, "pair_id": first["pair_id"], "variant": first["variant"],
                      "ask_first_future": now, "ask_second_future": later, "ask_first_minus_second": difference})
        for tree in worlds:
            path, _ = traverse(tree, planner)
            for index in range(len(path)):
                decision = tree.decisions[path[:index]]
                if len(decision.legal) > 1:
                    state = _state(decision.observation)
                    public_calls["visited_decisions"] += 1
                    public_calls["unseen_state_decisions"] += int(state not in planner.values)
                    public_calls["unknown_agenda_decisions"] += int(state[-1] == "unknown")
    if any(set(diffs) != {"ask_now", "save"} for diffs in pairs.values()):
        raise ValueError("Pair members must remain together in this audit")
    now_mean, later_mean = (mean(row[key] for row in roots) for key in ("ask_first_future", "ask_second_future"))
    best_by_root = mean(max(row["ask_first_future"], row["ask_second_future"]) for row in roots)
    return {"question_budget": evaluation[0].budget, "roots": len(roots), "worlds": len(evaluation),
            "policies": {label: _summarize(by_root, policy) for label, policy in policies.items()},
            "public_coverage": dict(public_calls),
            "paired_timing": {
                "pairs": len(pairs), "robust_flip_rate": mean(
                    diff["ask_now"] > 1e-12 and diff["save"] < -1e-12 for diff in pairs.values()),
                "mean_gain_over_best_constant": best_by_root - max(now_mean, later_mean),
                "by_root": roots, "always_first": now_mean, "always_second": later_mean,
                "root_conditional_upper_bound": best_by_root}}
