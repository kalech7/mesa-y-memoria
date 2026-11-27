"""Fresh user/world-disjoint bank for the two-slot memory experiment.

No historical episodes, action targets, model scores or prior heldout contents
are read. The same five prospective recipes are used in TRAIN/VAL/EVALUATION.
Private possible worlds provide outcome supervision, never SFT demonstrations.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import itertools
from pathlib import Path
import random

from research.budget_memory_env import (
    ASK_ACTIONS, MAX_EXCHANGE_TOKENS, VALUE_MENUS, VERSION as ENV_VERSION,
    check_token_budget, fingerprint, validate_episode,
)

VERSION = "budget_memory_data_v2_three_temporal_worlds"
FAMILIES = ("retention", "correction", "temporal", "context", "temporary_distractor")
ATTRIBUTE_WORDS = {"breakfast_preference": "el desayuno", "dinner_preference": "la cena",
                   "drink_preference": "beber"}
NEUTRAL = ("Hoy he dado un paseo y he ordenado unas fotos.",
           "Al final llegué bien. Gracias por acompañarme un rato.",
           "Estuve escuchando música mientras arreglaba la mesa.",
           "La tarde estuvo tranquila; solo quería conversar un momento.",
           "Esta mañana encontré una foto que no veía hace años.")


def _turn(session, text):
    return {"turn_id": f"s{session+1}u1", "role": "user", "text": text}


def _usual(attribute, value):
    return f"Para {ATTRIBUTE_WORDS[attribute]} prefiero {value} de forma habitual."


def _fact(attribute, value, session, start, kind="ongoing", context=None):
    return dict(attribute=attribute, value=value, scope_kind=kind, context=context,
                session_id=f"session_{session+1}" if kind == "session" else None,
                valid_from=start, valid_until_exclusive=None, sequence=session)


def make_episode(family, index, seed):
    if family not in FAMILIES:
        raise ValueError("Unknown prospective recipe")
    rng = random.Random(f"{seed}:{family}:{index}")
    uid = "bu" + fingerprint([VERSION, seed, index, family])[:20]
    start = date(2035, 1, 1) + timedelta(days=rng.randrange(1000))
    dates = [(start + timedelta(days=i)).isoformat() for i in range(4)]
    attrs = list(VALUE_MENUS); rng.shuffle(attrs)
    a, b, c = attrs
    va, anew = rng.sample(VALUE_MENUS[a], 2)
    vb, vc = rng.choice(VALUE_MENUS[b]), rng.choice(VALUE_MENUS[c])
    neutral = rng.sample(NEUTRAL, 2)
    # Event order is prospective and uniformly sampled among valid topological
    # orders. Neutral dialogue can occur anywhere, including the last useful fact.
    events = [{"kind": "base_a", "text": _usual(a, va)},
              {"kind": "base_b", "text": _usual(b, vb)},
              {"kind": "neutral", "text": neutral[0]},
              {"kind": "neutral", "text": neutral[1]}]
    dependent = family in ("correction", "temporal", "context")
    if family == "correction":
        events[2] = {"kind": "change", "text": f"Corrijo lo de antes: para {ATTRIBUTE_WORDS[a]} prefiero {anew} de forma habitual."}
    elif family in ("temporal", "context"):
        suffix = f"ahora prefiero {anew}." if family == "temporal" else f"aquí prefiero {anew}."
        events[2] = {"kind": "change", "text": f"Habitualmente, para {ATTRIBUTE_WORDS[a]}, {va}; {suffix}"}
    elif family == "temporary_distractor":
        events[2] = {"kind": "temporary", "text": f"Solo durante esta sesión, para {ATTRIBUTE_WORDS[c]} prefiero {vc}."}
    orders = [p for p in itertools.permutations(range(4)) if not dependent or p.index(0) < p.index(2)]
    order = rng.choice(orders)
    position = {event_id: order.index(event_id) for event_id in range(4)}
    change_index = position[2]
    sessions = []
    for session_index, event_id in enumerate(order):
        event = events[event_id]
        exchange = [_turn(session_index, event["text"])]
        # A proposal from the assistant is not an affirmed user preference.
        if event["kind"] == "neutral" and rng.random() < .5:
            proposed = rng.choice(VALUE_MENUS[rng.choice(attrs)])
            exchange = [{"turn_id": f"s{session_index+1}a1", "role": "assistant",
                         "text": f"¿Te apetecería {proposed}?"},
                        _turn(session_index, "No he confirmado ninguna preferencia nueva. " + event["text"])]
        sessions.append(dict(session_id=f"session_{session_index+1}", date=dates[session_index], exchange=exchange))
    def fact(attribute, value, event_id, kind="ongoing", context=None, until=None, reply_text=None):
        pos = position[event_id]
        row = _fact(attribute, value, pos, dates[pos], kind=kind, context=context)
        row["valid_until_exclusive"] = until
        group = [f"s{pos+1}u1"]
        if reply_text is not None:
            group.append(f"s{pos+1}u_reply")
            row["required_utterances"] = {f"s{pos+1}u_reply": reply_text}
        row["support_sets"] = [group]
        return row
    common_facts = [fact(a, va, 0), fact(b, vb, 1)]
    if family in ("temporal", "context"):
        # The updated exchange repeats the general baseline literally. This is
        # additional genuine user evidence, not an injected structured memory.
        common_facts[0]["support_sets"].append([f"s{change_index+1}u1"])
    until = (start + timedelta(days=9)).isoformat()
    variants = [("explicit", [])]
    if family == "correction":
        variants = [("corrected", [fact(a, anew, 2)])]
    elif family == "temporal":
        variants = [("interval", [fact(a, anew, 2, kind="interval", until=until,
                    reply_text=f"Solo hasta el {(start + timedelta(days=8)).isoformat()}, incluido.")]),
                    ("ongoing", [fact(a, anew, 2, reply_text="Como preferencia habitual desde ahora.")]),
                    ("session", [fact(a, anew, 2, kind="session", reply_text="Solo durante esta sesión.")])]
    elif family == "context":
        variants = [(ctx, [fact(a, anew, 2, kind="contextual", context=ctx, reply_text="Me refiero a "+ctx+".")])
                    for ctx in ("casa", "oficina")]
    elif family == "temporary_distractor":
        variants = [("temporary", [fact(c, vc, 2, kind="session")])]
    available = rng.random() >= .15
    worlds = []
    for variant, extra in variants:
        replies = {}
        for session_index in range(4):
            temporal = "No hay nada más que precisar sobre la duración."
            context = "No me refería a un lugar concreto."
            if session_index == change_index and family == "temporal":
                temporal = (f"Solo hasta el {(start + timedelta(days=8)).isoformat()}, incluido."
                            if variant == "interval" else "Solo durante esta sesión."
                            if variant == "session" else "Como preferencia habitual desde ahora.")
            if session_index == change_index and family == "context":
                context = "Me refiero a " + variant + "."
            if not available:
                temporal = context = "Ahora no te sé decir."
            replies[str(session_index)] = {
                "ASK_TEMPORAL_SCOPE": {"turn_id": f"s{session_index+1}u_reply", "role": "user", "text": temporal},
                "ASK_CURRENT_CONTEXT": {"turn_id": f"s{session_index+1}u_reply", "role": "user", "text": context}}
        worlds.append(dict(world_id="bw" + fingerprint([uid, variant])[:20],
                           facts=[*common_facts, *extra], replies=replies))
    actual = rng.randrange(len(worlds))
    query_attrs = [a, b]; rng.shuffle(query_attrs)
    queries = []
    for q, attribute in enumerate(query_attrs):
        # Both valid and expired interval futures occur after all four sessions.
        offset = (rng.randrange(5, 9) if index % 2 == 0 else rng.randrange(10, 15)) if family == "temporal" and attribute == a else rng.randrange(5, 15)
        queries.append(dict(attribute=attribute, date=(start+timedelta(days=offset)).isoformat(),
                            context=rng.choice(("casa", "oficina")), session_id=f"future_{q+1}"))
    episode = dict(version=VERSION, episode_id="be"+fingerprint([uid,"episode"])[:20],
                   root_id="br"+fingerprint([uid,"root"])[:20], user_id=uid, family=family,
                   sessions=sessions, queries=queries,
                   private=dict(actual_world_id=worlds[actual]["world_id"], worlds=worlds))
    validate_episode(episode)
    return episode


def build_bank(seed=202611059, partition_seed=202611060, train_roots=60,
               validation_roots=15, evaluation_roots=25):
    counts = {"train": train_roots, "validation": validation_roots, "evaluation": evaluation_roots}
    if any(n <= 0 or n % len(FAMILIES) for n in counts.values()):
        raise ValueError("Every split must be positive and exactly balanced over five families")
    bank = {role: [] for role in counts}
    rng = random.Random(partition_seed)
    # Generate independent users first; only then assign shuffled IDs to roles.
    total_each = sum(counts.values()) // len(FAMILIES)
    for family in FAMILIES:
        episodes = [make_episode(family, index, seed) for index in range(total_each)]
        rng.shuffle(episodes)
        offset = 0
        for role, n in counts.items():
            take = n // len(FAMILIES)
            bank[role].extend(episodes[offset:offset+take])
            offset += take
    for episodes in bank.values():
        rng.shuffle(episodes)
    assert_disjoint(bank)
    return bank


def assert_disjoint(bank):
    seen_users, seen_episodes, seen_worlds, seen_dialogues = set(), set(), set(), set()
    for episodes in bank.values():
        for episode in episodes:
            validate_episode(episode)
            user, eid = episode["user_id"], episode["episode_id"]
            worlds = {w["world_id"] for w in episode["private"]["worlds"]}
            dialogue = fingerprint(episode["sessions"])
            if user in seen_users or eid in seen_episodes or seen_worlds & worlds or dialogue in seen_dialogues:
                raise ValueError("User/world/episode/dialogue leakage between bank rows")
            seen_users.add(user); seen_episodes.add(eid)
            seen_worlds.update(worlds); seen_dialogues.add(dialogue)
    return True


def prepare(output, seed=202611059, partition_seed=202611060, train_roots=60,
            validation_roots=15, evaluation_roots=25, tokenizer=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Fresh banks must not overwrite a previous frozen bank")
    bank = build_bank(seed, partition_seed, train_roots, validation_roots, evaluation_roots)
    preflight = (check_token_budget([row for rows in bank.values() for row in rows], tokenizer)
                 if tokenizer is not None else {"status": "pending_real_tokenizer", "model_calls": 0, "scoring_calls": 0})
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for role, rows in bank.items():
        path = output / f"{role}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
        files[role] = dict(path=str(path.resolve()), sha256=__import__('hashlib').sha256(path.read_bytes()).hexdigest(),
                           roots=len(rows), users=len({r["user_id"] for r in rows}),
                           private_worlds=sum(len(r["private"]["worlds"]) for r in rows),
                           families={family: sum(r["family"] == family for r in rows) for family in FAMILIES})
    manifest = dict(version=VERSION, env_version=ENV_VERSION, seed=seed, partition_seed=partition_seed,
                    files=files, costs={"ask": .10, "write": .02}, memory_slots=2, sessions=4,
                    future_queries=2, max_exchange_tokens=MAX_EXCHANGE_TOKENS,
                    exchange_token_preflight=preflight, split_by="independent_user_and_world",
                    historical_data_read=False, model_calls=0, scoring_calls=0,
                    public_observations="literal current exchange, two slots, date and question budget only")
    (output / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def build_dataset(destination, seed=202611059, partition_seed=202611060, **kwargs):
    """Runner entry point; writes three JSONL files plus MANIFEST.json."""
    return prepare(destination, seed, partition_seed, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202611059)
    parser.add_argument("--partition-seed", type=int, default=202611060)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    print(json.dumps(prepare(args.output, args.seed, args.partition_seed, tokenizer=tokenizer),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
