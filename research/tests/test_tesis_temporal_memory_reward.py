from research.tesis_temporal_memory_reward import action_reward, recall_reward, write_reward


DAY = "2026-12-03"
TOMORROW = "2026-12-04"
ATTRIBUTE = "dinner_preference"


def query(day=DAY, session="S1"):
    return {"attribute": ATTRIBUTE, "date": day, "session_id": session, "context": None}


def scope(kind, *, start=DAY, end=None, session=None):
    return {"kind": kind, "valid_from": start, "valid_until_exclusive": end, "session_id": session, "context": None}


def record(value, lifetime, turn_id, text, **kwargs):
    return {"attribute": ATTRIBUTE, "value": value, "scope": lifetime, "status": "confirmed",
            "source_turn_ids": [turn_id], "evidence_quotes": [{"turn_id": turn_id, "text": text}], **kwargs}


def response(value, *refs):
    return {"mode": "ANSWER" if value else "DEFER", "value": value, "branches": [], "evidence_ids": list(refs)}


def test_reward_penalizes_restricting_a_day_to_a_chat_or_extending_it_past_expiration():
    text = "Para la cena prefiero sopa solo por hoy."
    user = {"role": "user", "turn_id": "u1", "text": text}
    unrelated = {"role": "user", "turn_id": "u2", "text": "Mi compañera prefiere sopa."}
    initial = record("quinua", scope("ongoing", start="2026-12-01"), "prev", "Prefiero quinua siempre.",
                     memory_id="m1")
    correct = record("sopa", scope("interval", end=TOMORROW), "u1", text)
    session_only = {**correct, "scope": scope("session", session="S1")}
    ongoing = {**correct, "scope": scope("ongoing")}
    bad_citation = {**correct, "source_turn_ids": ["u2"],
                    "evidence_quotes": [{"turn_id": "u2", "text": unrelated["text"]}]}
    observation = {"memory": [initial], "user_messages": [user, unrelated], "query": query()}
    probes = [
        {"query": query(), "expected": response("sopa")},
        {"query": query(session="S2"), "expected": response("sopa")},
        {"query": query(TOMORROW, "S3"), "expected": response("quinua")},
    ]

    def reward(added):
        return write_reward(observation, [{"op": "ADD", "record": added}], [correct], probes)

    best = reward(correct)
    assert best["joint"] and best["matched_probes"] == 3
    shortened = reward(session_only)
    assert not shortened["joint"] and shortened["matched_probes"] == 2
    widened = reward(ongoing)
    assert widened["harmful_future"] > 0 and widened["total"] < best["total"]
    unsupported = reward(bad_citation)
    assert unsupported["matched_probes"] == 3
    assert unsupported["unsupported_writes"] == unsupported["missing_writes"] == 1
    assert unsupported["total"] < best["total"]


def test_recall_requires_grounded_own_value_and_abstains_when_none_is_applicable():
    memory = [record("sopa", scope("session", session="S1"), "u1", "Prefiero sopa en este chat.",
                     memory_id="temporary")]
    expected = response("sopa", "temporary")
    assert recall_reward(memory, query(), expected, expected)["grounded"]
    assert recall_reward(memory, query(), response("sopa"), expected)["total"] < 0
    wrong = recall_reward(memory, query(session="S2"), expected, response(None))
    abstained = recall_reward(memory, query(session="S2"), response(None), response(None))
    assert not wrong["grounded"] and abstained["total"] > wrong["total"]


def test_correct_conditional_values_do_not_excuse_swapped_contexts():
    memory = [
        record("café", {**scope("contextual"), "context": "casa"}, "u1", "En casa prefiero café.",
               memory_id="house"),
        record("té", {**scope("contextual"), "context": "oficina"}, "u2", "En la oficina prefiero té.",
               memory_id="office"),
    ]
    gold = {"mode": "CONDITIONALIZE", "value": None, "evidence_ids": ["house", "office"],
            "branches": [{"condition": "casa", "value": "café"}, {"condition": "oficina", "value": "té"}]}
    reversed_order = {**gold, "branches": list(reversed(gold["branches"]))}
    swapped = {**gold, "branches": [{"condition": "casa", "value": "té"},
                                   {"condition": "oficina", "value": "café"}]}
    assert recall_reward(memory, query(), reversed_order, gold)["total"] > 0
    assert recall_reward(memory, query(), swapped, gold)["total"] < 0


def test_only_the_needed_question_is_rewarded():
    row = {"phase": "write", "correct_index": 0, "actions": [{"slot": "TEMPORAL_SCOPE"}]}
    assert action_reward(row, {"slot": "TEMPORAL_SCOPE"})["total"] > 0
    assert action_reward(row, {"slot": "VALID_UNTIL"})["total"] < 0
    assert action_reward(row, {"operations": [{"op": "NOOP"}]})["total"] < 0
