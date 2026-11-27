"""Two-slot, four-session memory environment with no model or judge dependency.

Only actual utterances enter observations. Private worlds are used exclusively by
the simulated user and the deterministic outcome/support checks after decisions.
The full dialogue is an audit artifact, never a retrieval source for the manager
or the frozen reader. STORE copies an entire chronological exchange verbatim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sqlite3
from typing import Any

VERSION = "budget_memory_env_v1"
ASK_COST = .10
WRITE_COST = .02
MAX_EXCHANGE_TOKENS = 96
ASK_ACTIONS = ("PROCEED", "ASK_TEMPORAL_SCOPE", "ASK_CURRENT_CONTEXT")
WRITE_ACTIONS = ("NOOP", "STORE_SLOT_1", "STORE_SLOT_2")
VALUE_MENUS = {
    "breakfast_preference": ("avena con fruta", "pan integral con queso", "yogur natural",
                             "huevos revueltos", "tortilla de maíz", "arepa con queso"),
    "dinner_preference": ("pollo al horno", "sopa de verduras", "salmón a la plancha",
                          "arroz con lentejas", "ensalada de garbanzos", "pasta con tomate"),
    "drink_preference": ("agua", "té de menta", "café con leche", "zumo de naranja",
                         "batido de fresa", "agua de coco"),
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


class BudgetMemoryStore:
    """Exactly two replaceable rows, with no version/history retrieval table."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS slots (slot_id INTEGER PRIMARY KEY "
                       "CHECK(slot_id IN (1,2)), exchange TEXT NOT NULL, "
                       "source_session INTEGER, stored_at TEXT)")
            for slot_id in (1, 2):
                db.execute("INSERT OR IGNORE INTO slots VALUES (?, '[]', NULL, NULL)", (slot_id,))
        self._check()

    def _check(self):
        with sqlite3.connect(self.path) as db:
            if db.execute("SELECT slot_id FROM slots ORDER BY slot_id").fetchall() != [(1,), (2,)]:
                raise ValueError("Memory must contain exactly slots 1 and 2")

    def snapshot(self):
        # Every call opens a new connection: no cached Python memory serves READ.
        self._check()
        with sqlite3.connect(self.path) as db:
            return [dict(slot_id=row[0], exchange=json.loads(row[1]),
                         source_session=row[2], stored_at=row[3])
                    for row in db.execute("SELECT * FROM slots ORDER BY slot_id")]

    def store(self, slot_id, exchange, session_index, date):
        if slot_id not in (1, 2) or not exchange:
            raise ValueError("STORE needs a legal slot and a complete nonempty exchange")
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE slots SET exchange=?, source_session=?, stored_at=? WHERE slot_id=?",
                       (canonical(exchange), session_index, date, slot_id))
        self._check()


def _applies(fact, query):
    if fact["attribute"] != query["attribute"] or fact["valid_from"] > query["date"]:
        return False
    until = fact.get("valid_until_exclusive")
    if until is not None and query["date"] >= until:
        return False
    kind = fact["scope_kind"]
    return (kind in ("ongoing", "interval") or
            kind == "session" and fact.get("session_id") == query.get("session_id") or
            kind == "contextual" and fact.get("context") == query.get("context"))


def _resolve_fact(world, query):
    applicable = [f for f in world["facts"] if _applies(f, query)]
    if not applicable:
        return None
    return max(applicable, key=lambda f: (f["scope_kind"] != "ongoing", f["sequence"]))


def resolve_world(world, query):
    """Private user preference, never exposed as a reader or actor feature."""
    fact = _resolve_fact(world, query)
    return fact["value"] if fact else None


def _compatible_worlds(episode, dialogue):
    by_id = {turn["turn_id"]: turn for turn in dialogue}
    compatible = []
    for world in episode["private"]["worlds"]:
        matches = True
        for index in range(4):
            question = by_id.get(f"s{index+1}a_question")
            reply = by_id.get(f"s{index+1}u_reply")
            if question is not None:
                if reply is None:
                    raise ValueError("Audit has a question without its actual user reply")
                actions = [a for a in ASK_ACTIONS[1:] if _question(index, a)["text"] == question["text"]]
                if len(actions) != 1:
                    raise ValueError("Unknown emitted clarification question")
                expected = _reply(world, index, actions[0])
                matches &= all(reply.get(k) == expected[k] for k in ("turn_id", "role", "text"))
        if matches:
            compatible.append(world)
    if not compatible:
        raise ValueError("Emitted evidence is incompatible with every authored world")
    return compatible


def _supported_in(world, query, answer, visible_user_ids, emitted):
    fact = _resolve_fact(world, query)
    return bool(fact and fact["value"] == answer and any(
        bool(group) and set(group) <= visible_user_ids for group in fact.get("support_sets", []))
        and all(tid in visible_user_ids and emitted[tid]["text"] == text
                for tid, text in fact.get("required_utterances", {}).items()))


def score_response(episode, dialogue, slots, query, answer):
    """Outcome verifier; complete user provenance is required in retained slots.

    Each supporting ID must match the emitted turn's exact role and literal text.
    Facts are resolved with date/context/correction precedence before checking
    provenance. Correct guesses from empty/obsolete/assistant memory do not pass.
    """
    if isinstance(answer, dict):
        answer = answer.get("value") if answer.get("mode") == "ANSWER" else "DEFER"
    worlds = _compatible_worlds(episode, dialogue)
    actual = next(w for w in episode["private"]["worlds"]
                  if w["world_id"] == episode["private"]["actual_world_id"])
    emitted = {t["turn_id"]: t for t in dialogue}
    public_ids = {tid for tid, turn in emitted.items() if turn["role"] == "user"}
    memory_ids = set()
    for slot in slots:
        for turn in slot["exchange"]:
            original = emitted.get(turn["turn_id"])
            if (original is not None and turn.get("role") == "user" and
                    all(turn.get(k) == original.get(k) for k in ("turn_id", "role", "text"))):
                memory_ids.add(turn["turn_id"])
    deferred = answer in (None, "DEFER")
    return dict(accepted_by_user=not deferred and answer == resolve_world(actual, query),
                publicly_supported=not deferred and all(_supported_in(w, query, answer, public_ids, emitted) for w in worlds),
                memory_supported=not deferred and all(_supported_in(w, query, answer, memory_ids, emitted) for w in worlds),
                compatible_world_count=len(worlds))


def reader_candidates(attribute):
    return [*VALUE_MENUS[attribute], "DEFER"]


def _reply(world, session_index, action):
    return _copy(world["replies"][str(session_index)][action])


def _with_context(turn, session):
    return {**_copy(turn), "date": session["date"], "session_id": session["session_id"]}


def exchange_variants(episode):
    """Enumerate all potentially stored literal exchanges, without inference."""
    for index, session in enumerate(episode["sessions"]):
        base = _copy(session["exchange"])
        yield base
        for action in ASK_ACTIONS[1:]:
            for world in episode["private"]["worlds"]:
                yield [*base, _question(index, action), _reply(world, index, action)]


def _question(index, action):
    text = {"ASK_TEMPORAL_SCOPE": "¿Esto es solo por esta sesión o de forma habitual?",
            "ASK_CURRENT_CONTEXT": "¿En qué lugar se aplica lo que comentas aquí?"}[action]
    return {"turn_id": f"s{index+1}a_question", "role": "assistant", "text": text}


def check_token_budget(episodes, tokenizer, limit=MAX_EXCHANGE_TOKENS):
    """CPU preflight for every role/world/ASK, using the actual model tokenizer.

    The slot capacity is literal dialogue text; provenance is retained separately
    and included in the independent full-prompt token check. Never truncate text.
    """
    maximum = 0
    count = 0
    for episode in episodes:
        for exchange in exchange_variants(episode):
            tokens = len(tokenizer.encode("\n".join(t["text"] for t in exchange),
                                          add_special_tokens=False))
            maximum = max(maximum, tokens)
            count += 1
            if tokens > limit:
                raise ValueError(f"Literal exchange exceeds {limit} tokens: "
                                 f"{episode['episode_id']} {tokens}")
    return {"checked_variants": count, "max_exchange_tokens": maximum, "limit": limit,
            "counting": "literal_text_joined_by_newlines", "model_calls": 0, "scoring_calls": 0}


def validate_episode(episode):
    if len(episode["sessions"]) != 4 or len(episode["queries"]) != 2:
        raise ValueError("An episode must contain exactly four sessions and two future queries")
    private = episode["private"]
    worlds = private["worlds"]
    if not worlds or private["actual_world_id"] not in {w["world_id"] for w in worlds}:
        raise ValueError("Actual world must belong to the compatible-world prior")
    ids = []
    for index, session in enumerate(episode["sessions"]):
        if not session["exchange"]:
            raise ValueError("Empty session")
        for turn in session["exchange"]:
            if set(turn) != {"turn_id", "role", "text"} or turn["role"] not in ("user", "assistant"):
                raise ValueError("Public dialogue must consist solely of literal turns")
            ids.append(turn["turn_id"])
        for world in worlds:
            for action in ASK_ACTIONS[1:]:
                reply = world["replies"][str(index)][action]
                if reply["role"] != "user":
                    raise ValueError("Clarification must be a user utterance")
    if len(set(ids)) != len(ids):
        raise ValueError("Turn IDs must be unique across sessions")
    for query in episode["queries"]:
        if query["attribute"] not in VALUE_MENUS:
            raise ValueError("Query uses an unknown public answer menu")
        if query["date"] <= episode["sessions"][-1]["date"]:
            raise ValueError("The two measured queries must occur after session four")
    public_user_ids = {turn["turn_id"] for session in episode["sessions"]
                       for turn in session["exchange"] if turn["role"] == "user"}
    for world in worlds:
        reply_ids = {r["turn_id"] for by_action in world["replies"].values() for r in by_action.values()}
        for fact in world["facts"]:
            groups = fact.get("support_sets")
            if not groups or any(not group or not set(group) <= (public_user_ids | reply_ids) for group in groups):
                raise ValueError("Every fact needs nonempty authored user support sets")
            if not set(fact.get("required_utterances", {})) <= reply_ids:
                raise ValueError("Required clarification evidence must be an authored user reply")
    return True


def _call_manager(policy, observation, candidates, seed, sample, decision_id):
    fn = policy.decide if hasattr(policy, "decide") else policy
    result = fn(_copy(observation), list(candidates), seed=seed,
                sample=sample, decision_id=decision_id)
    if isinstance(result, str):
        result = {"action": result, "metadata": {}}
    if not isinstance(result, dict) or result.get("action") not in candidates:
        raise ValueError("Manager action is outside the public legal catalogue")
    # Preserve all fields, including exact token IDs/distributions for RLOO.
    canonical(result)
    return _copy(result)


def _call_reader(reader, observation, candidates, seed, decision_id):
    fn = reader.answer if hasattr(reader, "answer") else reader
    result = fn(_copy(observation), list(candidates), seed=seed, decision_id=decision_id)
    if isinstance(result, str):
        result = {"answer": result, "metadata": {}}
    if not isinstance(result, dict) or result.get("answer") not in candidates:
        raise ValueError("Reader answer is outside its fixed public menu")
    canonical(result)
    return _copy(result)


def score_trace(trace):
    """Deterministic VAL/test utility, independent of external TRAIN feedback."""
    utilities = [0 if row["response"]["mode"] == "DEFER" else
                 (1 if row["accepted_by_user"] and row["publicly_supported"] and row["memory_supported"] else -1)
                 for row in trace["queries"]]
    if len(utilities) != 2:
        raise ValueError("Exactly two final queries are required")
    ask_count, write_count = trace["ask_count"], trace["write_count"]
    if not 0 <= ask_count <= 1 or not 0 <= write_count <= 4:
        raise ValueError("Episode interaction budgets were violated")
    utility = sum(utilities) / 2
    costs = ASK_COST * ask_count + WRITE_COST * write_count
    return dict(utility_mean=utility, mean_utility=utility, net_return=utility-costs,
                query_success=sum(u == 1 for u in utilities)/2,
                joint_success=int(all(u == 1 for u in utilities)),
                query_utilities=utilities, ask_count=ask_count, write_count=write_count,
                ask_cost=ASK_COST*ask_count, write_cost=WRITE_COST*write_count,
                costs=costs, accepted_rate=sum(q["accepted_by_user"] for q in trace["queries"])/2,
                supported_rate=sum(q["publicly_supported"] for q in trace["queries"])/2,
                memory_supported_rate=sum(q["memory_supported"] for q in trace["queries"])/2)


def full_history_reader_observation(trace, query):
    """Diagnostic higher-budget ceiling, using only actually emitted dialogue."""
    if "query" in query:
        query = query["query"]
    grouped = {}
    for turn in trace["dialogue"]:
        grouped.setdefault(turn["session_id"], []).append(turn)
    slots = []
    for index, turns in enumerate(grouped.values()):
        slots.append(dict(slot_id=index+1, source_session=index, stored_at=turns[0]["date"],
                          exchange=[{k: t[k] for k in ("turn_id", "role", "text")} for t in turns]))
    return {"slots": slots, "query": _copy(query), "answer_menu": reader_candidates(query["attribute"])}


def iter_public_observations(episode):
    """All reachable public manager/reader states, for tokenizer-only preflight.

    Branch over every legal action, both possible replies where applicable and
    both slot replacements. No answer, reward or model scoring is performed.
    Equal states are deduplicated; histories have no hidden retrieval channel.
    """
    validate_episode(episode)
    empty = [dict(slot_id=i, exchange=[], source_session=None, stored_at=None) for i in (1, 2)]
    states = [(empty, 0)]
    for index, session in enumerate(episode["sessions"]):
        next_states = {}
        for slots, asks in states:
            exchange = _copy(session["exchange"])
            observation = dict(decision_type="ask", session_index=index,
                               session_id=session["session_id"], date=session["date"],
                               exchange=exchange, slots=_copy(slots), questions_remaining=1-asks)
            candidates = list(ASK_ACTIONS if not asks else ("PROCEED",))
            yield "action", observation, candidates
            for action in candidates:
                outcomes = [(exchange, asks)]
                if action != "PROCEED":
                    unique = {canonical(_reply(w, index, action)): _reply(w, index, action)
                              for w in episode["private"]["worlds"]}
                    outcomes = [([*exchange, _question(index, action), reply], 1) for reply in unique.values()]
                for current, count in outcomes:
                    write_observation = dict(observation, decision_type="write", exchange=_copy(current),
                                             questions_remaining=1-count)
                    yield "action", write_observation, list(WRITE_ACTIONS)
                    for write in WRITE_ACTIONS:
                        updated = _copy(slots)
                        if write != "NOOP":
                            updated[int(write[-1])-1] = dict(slot_id=int(write[-1]), exchange=_copy(current),
                                                            source_session=index, stored_at=session["date"])
                        next_states[canonical([updated, count])] = (updated, count)
        states = list(next_states.values())
    for slots, _ in states:
        for query in episode["queries"]:
            candidates = reader_candidates(query["attribute"])
            yield "answer", dict(slots=_copy(slots), query=_copy(query), answer_menu=candidates), candidates


def run_episode(policy, reader, episode, seed, output_dir, sample=False, *, namespace=None):
    """Collect eight manager decisions; return evidence, costs and user feedback.

    Strict validation utility is diagnostic here; TRAIN's separate reward module
    combines user acceptance with the external judge after collection. No model,
    scorer labels, private fact or future query reaches a manager observation.
    """
    validate_episode(episode)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    db_path = destination / "memory.sqlite"
    if db_path.exists() or (destination / "trace.json").exists():
        raise FileExistsError("A rollout output directory cannot overwrite an earlier rollout")
    store = BudgetMemoryStore(db_path)
    namespace = str(destination.resolve()) if namespace is None else str(namespace)
    worlds = _copy(episode["private"]["worlds"])
    actual = next(w for w in worlds if w["world_id"] == episode["private"]["actual_world_id"])
    compatible = list(worlds)
    decisions, dialogue = [], []
    asks = writes = 0
    rng = random.Random(seed)

    for index, session in enumerate(episode["sessions"]):
        # Fresh local exchange on each iteration; previous history is not merged.
        exchange = _copy(session["exchange"])
        dialogue.extend([_with_context(turn, session) for turn in exchange])
        for phase in ("ask", "write"):
            remaining = 1 - asks
            observation = dict(decision_type=phase, session_index=index,
                               session_id=session["session_id"], date=session["date"],
                               exchange=_copy(exchange), slots=store.snapshot(),
                               questions_remaining=remaining)
            candidates = list((ASK_ACTIONS if remaining else ("PROCEED",))
                              if phase == "ask" else WRITE_ACTIONS)
            step_id = len(decisions)
            decision_id = f"{namespace}:{episode['episode_id']}:{seed}:{index}:{phase}:{step_id}"
            call_seed = rng.randrange(2**31)
            result = _call_manager(policy, observation, candidates, call_seed, sample, decision_id)
            action = result["action"]
            before = store.snapshot()
            if phase == "ask" and action != "PROCEED":
                asks += 1
                question = _question(index, action)
                reply = _reply(actual, index, action)
                exchange.extend([question, reply])
                dialogue.extend([_with_context(turn, session) for turn in (question, reply)])
                compatible = [w for w in compatible
                              if canonical(_reply(w, index, action)) == canonical(_reply(actual, index, action))]
            elif phase == "write" and action != "NOOP":
                store.store(int(action[-1]), exchange, index, session["date"])
                writes += 1
            decisions.append(dict(decision_id=decision_id, phase=phase, decision_type=phase,
                                  step_id=step_id, session_index=index, seed=call_seed,
                                  observation=observation, observation_sha256=fingerprint(observation),
                                  candidates=candidates, action=action, metadata=result.get("metadata", {}),
                                  policy_output=result, accepted=True,
                                  slots_before_sha256=fingerprint(before),
                                  slots_after_sha256=fingerprint(store.snapshot())))
        # Delete the local dialogue before constructing the next observation.
        exchange.clear()

    # Reconstruct from a newly opened store; never use the dialogue audit as READ.
    reopened = BudgetMemoryStore(db_path)
    final_slots = reopened.snapshot()
    queries = []
    for index, query in enumerate(episode["queries"]):
        query = _copy(query)
        query_id = f"q{index+1}"
        observation = {"slots": _copy(final_slots), "query": query,
                       "answer_menu": reader_candidates(query["attribute"])}
        reader_result = _call_reader(reader, observation, observation["answer_menu"],
                                     rng.randrange(2**31), f"{namespace}:{episode['episode_id']}:{seed}:{query_id}")
        answer = reader_result["answer"]
        deferred = answer == "DEFER"
        flags = score_response(episode, dialogue, final_slots, query, answer)
        accepted, supported = flags["accepted_by_user"], flags["publicly_supported"]
        utility = 0 if deferred else (1 if accepted and supported and flags["memory_supported"] else -1)
        queries.append(dict(query_id=query_id, query=query,
                            response={"mode": "DEFER" if deferred else "ANSWER",
                                      "value": None if deferred else answer},
                            accepted_by_user=accepted, publicly_supported=supported,
                            memory_supported=flags["memory_supported"],
                            compatible_world_count=flags["compatible_world_count"], deterministic_utility=utility,
                            reader_observation=observation, reader_output=reader_result,
                            reader_metadata=reader_result.get("metadata", {})))
    mean_utility = sum(row["deterministic_utility"] for row in queries) / len(queries)
    costs = ASK_COST * asks + WRITE_COST * writes
    judge_payload = dict(episode_id=episode["episode_id"], dialogue=_copy(dialogue),
                         queries=[{key: _copy(row[key]) for key in ("query_id", "query", "response")}
                                  for row in queries], final_slots=_copy(final_slots))
    trace = dict(version=VERSION, episode_id=episode["episode_id"],
                 root_id=episode.get("root_id", episode["episode_id"]),
                 family=episode.get("family"), rollout_seed=seed,
                 sample=bool(sample), decisions=decisions, dialogue=dialogue, queries=queries,
                 future_answers=_copy(queries), ask_count=asks, write_count=writes,
                 final_slots=final_slots, public_judge_payload=judge_payload,
                 metrics=dict(ask_count=asks, write_count=writes, ask_cost=ASK_COST*asks,
                              write_cost=WRITE_COST*writes, costs=costs,
                              mean_deterministic_utility=mean_utility,
                              deterministic_reward=mean_utility-costs,
                              accepted_rate=sum(q["accepted_by_user"] for q in queries)/2,
                              supported_rate=sum(q["publicly_supported"] for q in queries)/2),
                 persistence=dict(database=str(db_path.resolve()), reopened_before_reader=True,
                                  table="slots", row_count=len(final_slots),
                                  slots_sha256=fingerprint(final_slots)))
    trace["metrics"].update(score_trace(trace))
    (destination / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
    return trace
