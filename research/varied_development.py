"""Independent, scripted multi-turn development recipes; never model evaluation.

Gold follows literal construction recipes. This module does not import the
production parser, reader, evaluator, B1 or any inference backend. Supplied
assistant questions are context, not model-generated trajectories or user facts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import random
import shutil

from .src.data.semantic_partition_v3 import assert_semantic_split_isolation_v3

VERSION = "varied_scripted_development_v3.0"
NAMESPACE = "development_varied_v3_20261025_001"
STRATA = ("ongoing_correction", "separate_interval", "contextual_scope", "ambiguous_duration", "unrelated_attribute_control")
MEMORY_COUNTS = (1,) * 4 + (2,) * 6 + (3,) * 15 + (4,) * 15 + (5,) * 6 + (6,) * 4
DISHES = (
    "ensalada de garbanzos", "arroz con verduras", "pollo al horno", "sopa de fideos",
    "tortilla de patatas", "pasta con pesto", "lentejas con arroz", "tacos de frijoles",
    "cuscús con verduras", "crema de calabaza", "pescado al vapor", "berenjenas al horno",
    "guiso de lentejas", "lasaña de verduras", "quinoa con vegetales", "garbanzos con espinacas",
    "salmón a la plancha", "puré de patatas", "fideos con champiñones", "paella de verduras",
    "hamburguesa de lentejas", "empanadas de queso", "arroz con pollo", "sopa de tomate",
)
DRINKS = ("agua con limón", "té de menta", "café con leche", "infusión de manzanilla",
          "zumo de naranja", "batido de fresa", "agua de coco", "chocolate caliente")
BREAKFASTS = ("tostadas con aguacate", "avena con frutas", "yogur con granola", "huevos con espinacas",
              "pan con queso", "pudín de chía", "arepa con queso", "cereal con leche")
VOCABULARY = {"dinner_preference": DISHES, "drink_preference": DRINKS, "breakfast_preference": BREAKFASTS}
CLAUSE = {"dinner_preference": "Para la cena", "drink_preference": "Para beber", "breakfast_preference": "Para el desayuno"}
QUERY = {"dinner_preference": "¿Qué me recomiendas para cenar hoy?", "drink_preference": "¿Qué bebida me recomiendas hoy?",
         "breakfast_preference": "¿Qué me recomiendas para desayunar hoy?"}
SLOTS = ("TEMPORAL_SCOPE", "CURRENT_CONTEXT", "VALID_UNTIL", "GENERIC")
ATTRIBUTES = ("dinner_preference", "dinner_preference", "drink_preference", "dinner_preference", "breakfast_preference",
              "dinner_preference", "drink_preference", "dinner_preference", "breakfast_preference", "dinner_preference")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scope(kind, start, *, end=None, context=None, session=None):
    return {"kind": kind, "valid_from": start.isoformat(), "valid_until_exclusive": end.isoformat() if end else None,
            "context": context, "session_id": session}


class _Deck:
    def __init__(self, rng):
        self.rng, self.remaining = rng, {}

    def take(self, attribute, excluding=()):
        for _ in range(len(VOCABULARY[attribute]) * 3):
            if not self.remaining.get(attribute):
                self.remaining[attribute] = list(VOCABULARY[attribute]);self.rng.shuffle(self.remaining[attribute])
            value = self.remaining[attribute].pop()
            if value not in excluding:
                return value
        raise ValueError("Vocabulary exhausted by exclusions")


def _memories(count, attribute, now, deck, *, contextual=False, prior_count=1):
    sessions = [{"session_id": f"P{i+1}", "date": (now-timedelta(days=(prior_count-i)*8)).isoformat(), "messages": []}
                for i in range(prior_count)]
    records, previous = [], {}
    unrelated = [a for a in VOCABULARY if a != attribute]
    for index in range(count):
        session_index = min(prior_count - 1, index * prior_count // count)
        session = sessions[session_index]
        at = date.fromisoformat(session["date"])
        attr = attribute if index == 0 or (contextual and index == 1) else unrelated[(index - 1) % len(unrelated)]
        context = "oficina" if contextual and index == 1 else None
        value = deck.take(attr)
        mid, tid = f"m{index+1}", f"p{session_index+1}u{len(session['messages'])+1}"
        text = (f"En la oficina, {CLAUSE[attr].lower()} prefiero {value} de forma habitual." if context else
                f"{CLAUSE[attr]} prefiero {value} de forma habitual.")
        key = (attr, context)
        supersedes = [previous[key]] if key in previous else []
        if supersedes:
            text = "Corrijo: " + text[0].lower() + text[1:]
        session["messages"].append({"role": "user", "turn_id": tid, "text": text})
        records.append({"memory_id": mid, "attribute": attr, "value": value,
                        "scope": _scope("contextual" if context else "ongoing", at, context=context),
                        "recorded_at": at.isoformat(), "status": "confirmed", "source_turn_ids": [tid],
                        "evidence_quotes": [{"turn_id": tid, "text": text}], "supersedes": supersedes})
        previous[key] = mid
    return records, sessions


def _dialogue(core, length, rng, *, attribute, fake_value, style):
    messages = [{"role": role, "text": text} for role, text in core]
    remaining = length - len(messages)
    if remaining < 0:
        raise ValueError("Dialogue plan shorter than its semantic recipe")
    if remaining:
        scope_positions = [i for i,m in enumerate(messages[:-1]) if m["text"].casefold().startswith(
            ("como preferencia habitual", "hasta el", "corrijo el plazo", "esa preferencia"))]
        # Ask before the fact or before its clarification, without fabricating
        # a model response or moving the final principal user query.
        position = rng.choice([0] + scope_positions) if scope_positions else rng.choice([0, len(messages)-1])
        question = (f"¿Quieres que anote «{CLAUSE[attribute].lower()} prefiero {fake_value} de forma habitual»?"
                    if style == "unrelated_attribute_control" else
                    "¿La preferencia es habitual o una excepción?" if scope_positions and position else
                    rng.choice(("¿Qué preferencia quieres revisar?", "¿Qué necesitas que tengamos en cuenta?")))
        messages.insert(position, {"role": "assistant", "text": question})
        remaining -= 1
    prefix = []
    if remaining:
        prefix.append({"role":"user", "text":"Hola, revisemos mis preferencias."});remaining -= 1
    if remaining:
        prefix.append({"role":"assistant", "text":"Hola, te escucho."});remaining -= 1
    messages = prefix + messages
    fillers = [("user", "Gracias, continuemos."), ("assistant", "De acuerdo, revisemos la información."),
               ("user", "Eso es lo que quería revisar.")]
    rng.shuffle(fillers)
    for i in range(remaining):
        role, text = fillers[i % len(fillers)]
        messages.insert(rng.randrange(len(prefix), len(messages)), {"role": role, "text": text})
    counters = Counter()
    for message in messages:
        counters[message["role"]] += 1
        message["turn_id"] = ("u" if message["role"] == "user" else "a") + str(counters[message["role"]])
    return messages, f"u{counters['user']+1}"


def build_varied_development(*, seed=53017):
    rng, episodes = random.Random(seed), []
    deck = _Deck(rng)
    memory_counts = list(MEMORY_COUNTS);rng.shuffle(memory_counts)
    length_plans = ((4, 5, 6, 7, 8, 4, 5, 6, 7, 8), (3, 4, 5, 6, 7, 8, 3, 4, 5, 6),
                    (3, 4, 5, 6, 7, 8, 4, 5, 6, 7), (2, 3, 4, 5, 6, 7, 8, 3, 4, 5),
                    (2, 3, 4, 5, 6, 7, 8, 3, 4, 5))
    for stratum_index, stratum in enumerate(STRATA):
        for index in range(10):
            ordinal = stratum_index * 10 + index
            now = date(2028, 3, 1) + timedelta(days=ordinal * 2 + rng.randrange(2))
            future, attribute = now + timedelta(days=7), ATTRIBUTES[index]
            count = memory_counts[ordinal]
            memories, prior_sessions = _memories(count, attribute, now, deck,
                contextual=stratum == "contextual_scope", prior_count=1 + ordinal % min(3, count))
            baseline = memories[0]["value"]
            chosen = deck.take(attribute, excluding=[baseline])
            original_choice = deck.take(attribute, excluding=[baseline, chosen])
            other_attribute = next(a for a in VOCABULARY if a != attribute)
            unrelated_choice = deck.take(other_attribute)
            target_length = length_plans[stratum_index][index]
            clause = CLAUSE[attribute]
            current_context = "casa" if stratum == "contextual_scope" else None
            future_context = ("oficina" if index % 3 == 1 else "casa") if stratum == "contextual_scope" else None
            query_text = ("Hoy estoy en casa. " if current_context else "") + QUERY[attribute]
            boundary, known_scope = "ongoing", None
            if stratum == "ongoing_correction":
                family = "F1"
                core = [("user", f"{clause} prefiero {original_choice}."),
                        ("user", f"Corrijo: {clause.lower()} prefiero {chosen}."),
                        ("user", "Como preferencia habitual desde ahora."), ("user", query_text)]
                if target_length >= 6 and index % 2 == 0:
                    core.insert(1, ("user", "Solo en esta conversación."))
                known_scope = _scope("ongoing", now)
                worlds = [("ongoing", known_scope, chosen, chosen)]
            elif stratum == "separate_interval":
                family = "F2"
                end = now + timedelta(days=(9, 7, 5)[index % 3])
                inclusive = end - timedelta(days=1)
                boundary = "inside" if future < end else "exclusive_end" if future == end else "after"
                core = [("user", f"{clause} prefiero {chosen}."),
                        ("user", f"Hasta el {inclusive.isoformat()} inclusive."), ("user", query_text)]
                if target_length >= 5 and index % 2 == 0:
                    core.insert(1, ("user", "Solo en esta conversación."))
                    core[2] = ("user", f"Corrijo el plazo: hasta el {inclusive.isoformat()} inclusive.")
                known_scope = _scope("interval", now, end=end)
                worlds = [("interval", known_scope, chosen, chosen if future < end else baseline)]
            elif stratum == "contextual_scope":
                family = "F3"
                bounded = index % 2 == 1
                end = now + timedelta(days=(9, 7, 5)[index % 3]) if bounded else None
                scope_text = f"Hasta el {(end-timedelta(days=1)).isoformat()} inclusive." if end else "Como preferencia habitual desde ahora."
                core = [("user", f"En casa, {clause.lower()} prefiero {chosen}."), ("user", scope_text), ("user", query_text)]
                if target_length >= 5 and index % 3 == 0:
                    core[0] = ("user", f"{clause} prefiero {chosen}.")
                    core.insert(1, ("user", "Esa preferencia se aplica en casa."))
                known_scope = _scope("contextual", now, end=end, context="casa")
                if future_context == "oficina":
                    future_value = memories[1]["value"] if count >= 2 else baseline
                else:
                    future_value = chosen if end is None or future < end else baseline
                boundary = "ongoing" if end is None else "inside" if future < end else "exclusive_end" if future == end else "after"
                worlds = [("contextual", known_scope, chosen, future_value)]
            elif stratum == "ambiguous_duration":
                family, boundary = "F4", "unknown_duration"
                core = [("user", f"{clause}, ahora prefiero {chosen}."), ("user", query_text)]
                if target_length >= 4:
                    core.insert(1, ("user", f"{CLAUSE[other_attribute]} prefiero {unrelated_choice} de forma habitual."))
                worlds = [("session", _scope("session", now, session="S2"), chosen, baseline),
                          ("ongoing", _scope("ongoing", now), chosen, chosen)]
            else:
                family, boundary = "F0", "unrelated_attribute"
                core = [("user", f"{CLAUSE[other_attribute]} prefiero {unrelated_choice} de forma habitual."), ("user", query_text)]
                worlds = [("stable", None, baseline, baseline)]
            messages, reply_id = _dialogue(core, target_length, rng, attribute=attribute, fake_value=original_choice, style=stratum)
            rid = "vr" + _digest({"memories": memories, "prior_sessions": prior_sessions, "messages": messages,
                                  "current": now.isoformat(), "future": future.isoformat(), "stratum": stratum})[:16]
            for world_name, real_scope, current_value, future_value in worlds:
                replies = {slot: {"turn_id": reply_id, "text": "No puedo precisar ese dato ahora.", "availability": "unavailable"} for slot in SLOTS}
                decisive = "TEMPORAL_SCOPE" if stratum == "ambiguous_duration" else None
                if decisive:
                    replies[decisive] = {"turn_id": reply_id, "text": "Solo en esta conversación." if world_name == "session" else "Como preferencia habitual desde ahora.", "availability": "available"}
                episode = {"root_id": rid, "latent_world_id": "vw" + _digest([rid, world_name])[:16], "realization_id": "v0",
                    "split": "validation", "data_namespace": NAMESPACE, "data_role": "prospective_development_only",
                    "domain": "non_clinical_food_preferences", "generator_version": VERSION,
                    "scenario_family": family, "template_id": f"v3:{stratum}:style{index}",
                    "composition_id": f"v3:{stratum}:m{count}:p{len(prior_sessions)}:t{target_length}:{boundary}:{attribute}",
                    "initial_memory": deepcopy(memories), "prior_sessions": deepcopy(prior_sessions),
                    "user_messages": deepcopy(messages),
                    "current_query": {"attribute": attribute, "date": now.isoformat(), "session_id": "S2", "context": current_context},
                    "future_query": {"attribute": attribute, "date": future.isoformat(), "session_id": "S3", "context": future_context},
                    "replies": replies,
                    "truth": {"current_value": current_value, "future_value": future_value,
                              "initial_value": baseline, "new_value": chosen if stratum != "unrelated_attribute_control" else unrelated_choice,
                              "scope": deepcopy(real_scope), "scope_kind": world_name,
                              "scope_attribute": attribute if stratum != "unrelated_attribute_control" else other_attribute,
                              "decisive_slot": decisive, "current_context": current_context,
                              "current_date": now.isoformat(), "future_exclusive_benefit": decisive is not None},
                    "opportunity": {"eligible": decisive is not None, "negative_control": decisive is None, "slot": decisive},
                    "authoring": {"source": VERSION, "reviewer_type": "ai_authored_development", "human_authored": False,
                                  "coverage_source": "User structure-variation request and synthetic-data SPEC coverage, not B1 outcomes.",
                                  "stratum": stratum, "boundary": boundary, "model_results_used_for_design": False,
                                  "gold_source": "Literal authored recipe and independent date/context/version comparisons; no production parser/reader/scorer.",
                                  "provided_history_model_generated": False, "principal_queries_evaluated": 1,
                                  "max_new_preferences": 2, "persistence_required_for": attribute,
                                  "seed_effect": "Controls vocabulary decks, prior-memory allocation, dates, neutral turn ordering and final row order; does not imply conceptual OOD."}}
                public = {"memory": episode["initial_memory"], "user_messages": episode["user_messages"], "query": episode["current_query"]}
                episode["observable_state_id"] = _digest(public)
                episodes.append(episode)
    rng.shuffle(episodes)
    assert len(episodes) == 60 and len({e["root_id"] for e in episodes}) == 50
    return episodes


def diversity_plan(episodes):
    roots = {e["root_id"]:e for e in episodes}
    values = {r["value"] for e in episodes for r in e["initial_memory"]}
    values.update(e["truth"]["new_value"] for e in episodes)
    for e in episodes:
        for message in e["user_messages"]:
            if message.get("role") != "user":
                continue
            for vocabulary in VOCABULARY.values():
                values.update(v for v in vocabulary if v in message["text"])
    return {"roots": len(roots), "episodes": len(episodes), "worlds":len({e["latent_world_id"] for e in episodes}),
            "memory_counts_by_root": dict(sorted(Counter(len(e["initial_memory"]) for e in roots.values()).items())),
            "turn_counts_by_root": dict(sorted(Counter(len(e["user_messages"]) for e in roots.values()).items())),
            "prior_sessions_by_root": dict(sorted(Counter(len(e["prior_sessions"]) for e in roots.values()).items())),
            "strata_by_root":dict(Counter(e["authoring"]["stratum"] for e in roots.values())),
            "query_attributes_by_root":dict(Counter(e["current_query"]["attribute"] for e in roots.values())),
            "dishes_observed": sorted(set(DISHES)&values), "drinks_observed":sorted(set(DRINKS)&values),
            "breakfasts_observed":sorted(set(BREAKFASTS)&values),
            "scripted_assistant_turns":sum(m["role"]=="assistant" for e in roots.values() for m in e["user_messages"]),
            "template_ids":sorted({e["template_id"] for e in episodes}),
            "scope": "Variation inside five explicit construction recipes; no free dialogue, conceptual OOD or new learning claim."}


def prepare_varied_development(data_directory, report_directory, *, historical_directory="research/data/bank_1_1",
                               previous_development="research/data/development_v2_20261024_001/validation.jsonl", seed=53017):
    data_directory, report_directory = Path(data_directory), Path(report_directory)
    data_directory.mkdir(parents=True, exist_ok=True);report_directory.mkdir(parents=True, exist_ok=True)
    bank = data_directory/"validation.jsonl"
    if bank.exists() or (data_directory/"manifest.json").exists():
        raise FileExistsError("Refusing to overwrite an existing development version")
    inputs = {f"legacy_{split}":Path(historical_directory)/f"{split}.jsonl"
              for split in ("dev_smoke", "train", "validation", "test_id", "test_ood", "test_external")}
    inputs["development_v2"] = Path(previous_development)
    source_hashes = {name:_sha(path) for name,path in inputs.items()}
    snapshots=report_directory/"source_snapshot";snapshots.mkdir(exist_ok=True)
    for name,path in inputs.items():
        target=snapshots/f"{name}.jsonl"
        if target.exists() and _sha(target)!=source_hashes[name]:raise ValueError("Snapshot mismatch")
        if not target.exists():shutil.copyfile(path,target)
    target=snapshots/Path(__file__).name
    if target.exists() and _sha(target)!=_sha(__file__):raise ValueError("Producer snapshot changed; use a new report directory")
    if not target.exists():shutil.copyfile(__file__,target)
    historical={name:[json.loads(line) for line in path.open()] for name,path in inputs.items()}
    episodes=build_varied_development(seed=seed)
    isolation=assert_semantic_split_isolation_v3({"validation":episodes},historical_splits=historical)
    plan=diversity_plan(episodes)
    assert plan["memory_counts_by_root"]==dict(Counter(MEMORY_COUNTS))
    assert len(plan["dishes_observed"])>=24 and len(plan["drinks_observed"])>=8 and len(plan["breakfasts_observed"])>=8
    with bank.open("x") as stream:
        for episode in episodes:stream.write(json.dumps(episode,ensure_ascii=False,sort_keys=True)+"\n")
    after={name:_sha(path) for name,path in inputs.items()};assert after==source_hashes
    manifest={"version":VERSION,"data_namespace":NAMESPACE,"data_role":"prospective_development_only","split":"validation",
              **plan,"bank_path":str(bank.resolve()),"bank_sha256":_sha(bank),"seed":seed,
              "seed_effect":"Generation, allocation and row order; not evidence of conceptual OOD.",
              "historical_hashes_before":source_hashes,"historical_hashes_after":after,"isolation":isolation,
              "producer_sha256":_sha(__file__),"reviewer_type":"ai_authored_development","human_review":False,
              "model_generated_context":False,"principal_queries_per_episode":1,"max_new_preferences_per_dialogue":2,
              "unrelated_preferences_persistence_required":False,"training_steps":0,"new_Qwen_calls":0,"confirmatory_test_created":False,
              "limits":["Previous assistant questions are scripted context, not evidence of user preferences or outputs of B1.",
                        "One current principal query and one same-attribute future query; multiple user questions are not separate scored rollouts.",
                        "Only the last relevant new preference and its latest clarification are required for the queried attribute.",
                        "Short IDs reduce an identifier-copying confound; changes versus v2 are not a causal ablation.",
                        "Independent gold construction is AI-authored and still requires semantic verification and model evaluation."]}
    for path in (data_directory/"manifest.json",report_directory/"diversity_plan.json"):
        path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory",required=True);parser.add_argument("--report-directory",required=True)
    parser.add_argument("--historical-directory",default="research/data/bank_1_1")
    parser.add_argument("--previous-development",default="research/data/development_v2_20261024_001/validation.jsonl")
    parser.add_argument("--seed",type=int,default=53017)
    args=parser.parse_args(argv)
    result=prepare_varied_development(args.data_directory,args.report_directory,historical_directory=args.historical_directory,
                                     previous_development=args.previous_development,seed=args.seed)
    print(json.dumps({k:result[k] for k in ("version","roots","episodes","bank_path","bank_sha256","memory_counts_by_root","turn_counts_by_root")},ensure_ascii=False))


if __name__=="__main__":main()
