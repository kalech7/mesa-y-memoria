"""Generate prospective synthetic meal chats with grounded extra history."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path
import random

from . import competence_data as base
from .src.data.semantic_partition_v3 import assert_semantic_split_isolation_v3, observable_episode_id_v3
from .varied_development import CLAUSE, QUERY, VOCABULARY, diversity_plan

VERSION = "natural_meal_chats_v1"
SIZES = {"train": 50, "validation": 15, "evaluation": 25}
CHAT_ASIDES = (
    ("Uf, acabo de llegar del trabajo y vengo con prisa.", "Entiendo. Vamos con lo práctico."),
    ("Hoy salí tarde de clases. Ni me dio tiempo de hacer la compra.", "Vale, te sigo."),
    ("Estoy organizando el menú mientras espero el bus.", "Dale, cuéntame."),
    ("Me escribieron del grupo familiar y me distraje, jaja.", "Tranqui, seguimos."),
    ("Un segundo, estoy guardando las cosas del súper.", "Claro, tómate tu tiempo."),
    ("Perdón, estoy escribiendo esto desde el móvil.", "Sin problema, te leo."),
    ("Ando con mil cosas hoy. Mejor lo vemos rapidito.", "De acuerdo, dime."),
    ("Al final se canceló el plan que tenía con mi hermana.", "Ya veo. ¿Qué hacemos entonces?"),
)
QUERY_LEADS = (
    "Bueno, ya que estamos, ",
    "Me ayudas con una cosa más: ",
    "Entonces, para decidirme, ",
    "Y de paso, ",
    "A ver, para no darle más vueltas: ",
    "Volviendo a lo mío, ",
)
QUERY_VARIANTS = {
    "dinner_preference": ("¿qué me sugieres para cenar?", "¿qué podría preparar para la cena hoy?", "¿qué hacemos de cena?"),
    "drink_preference": ("¿qué me aconsejas tomar hoy?", "¿qué bebida me sugieres para hoy?", "¿qué me recomiendas para beber?"),
    "breakfast_preference": ("¿qué podría desayunar hoy?", "¿qué se te ocurre para el desayuno?", "¿qué me sugieres para desayunar?"),
}
NEUTRAL_QUESTIONS = {
    "¿Qué preferencia quieres revisar?": ("¿Por dónde empezamos?", "A ver, ¿qué quieres mirar primero?"),
    "¿Qué necesitas que tengamos en cuenta?": ("Dime, ¿hay algo que deba tener presente?", "¿Qué te viene bien que recordemos?"),
    "¿La preferencia es habitual o una excepción?": ("¿Eso vale en general o es por esta vez?", "¿Te pasa normalmente o solo esta vez?"),
    "Hola, te escucho.": ("Buenas, dime.", "Hola, cuéntame."),
    "De acuerdo, revisemos la información.": ("Vale, vamos por partes.", "Bien, te sigo."),
}
UNAVAILABLE = ("Uy, ahora mismo no lo tengo claro.", "No puedo precisar eso ahora.",
               "No tengo ese dato a mano.", "Ahora no te sé decir.")
SESSION_REPLIES = ("Sí, solo en esta conversación.", "Solo para esta sesión.", "Es solo para esta ocasión, eso sí.")
ONGOING_REPLIES = ("Sí, como preferencia habitual desde ahora.", "Apúntalo como preferencia habitual.",
                   "Me quedo con eso de forma habitual.")


def _make_extra_memories(episode, rng):
    current = episode["current_query"]
    attributes = [attribute for attribute in VOCABULARY if attribute != current["attribute"]]
    rng.shuffle(attributes)
    # These dated historical exceptions were already over before the current
    # interaction. Their values add plausible retrieval distractors.
    memories = episode["initial_memory"]
    date_now = date.fromisoformat(current["date"])
    for ordinal, attribute in enumerate(attributes[:rng.choice((1, 2))]):
        day = date_now - timedelta(days=50 + 9 * ordinal + rng.randrange(10))
        stop = day + timedelta(days=rng.choice((1, 3, 4)))
        day_text, stop_text = day.isoformat(), stop.isoformat()
        inclusive = (stop - timedelta(days=1)).isoformat()
        value = rng.choice(VOCABULARY[attribute])
        tid = f"old{ordinal+1}u1"
        context = rng.choice(("la visita", "el viaje", "la semana de mudanza"))
        quote = f"Por {context}. {CLAUSE[attribute]} prefiero {value}. Hasta el {inclusive} inclusive."
        memories.append({
            "memory_id": f"m{len(memories)+1}", "attribute": attribute, "value": value,
            "scope": {"kind": "interval", "valid_from": day_text, "valid_until_exclusive": stop_text,
                      "context": None, "session_id": None},
            "recorded_at": day_text, "status": "confirmed", "source_turn_ids": [tid],
            "evidence_quotes": [{"turn_id": tid, "text": quote}], "supersedes": [],
        })
        episode["prior_sessions"].append({
            "session_id": f"Pold{ordinal+1}", "date": day_text,
            "messages": [{"role": "assistant", "turn_id": f"old{ordinal+1}a1", "text": "¿Qué quieres dejar anotado?"},
                         {"role": "user", "turn_id": tid, "text": quote}],
        })
    episode["prior_sessions"].sort(key=lambda session: session["date"])


def _dialogue(episode, rng):
    messages = episode["user_messages"]
    query = episode["current_query"]
    final = messages[-1]
    question = QUERY[query["attribute"]]
    if not final["text"].endswith(question):
        raise ValueError("Expected the authored principal query at the end")
    final["text"] = final["text"][:-len(question)] + rng.choice(QUERY_LEADS) + rng.choice(QUERY_VARIANTS[query["attribute"]])
    for message in messages:
        if message["role"] == "assistant" and message["text"] in NEUTRAL_QUESTIONS:
            message["text"] = rng.choice(NEUTRAL_QUESTIONS[message["text"]])
    selected_asides = rng.sample(CHAT_ASIDES, rng.choice((1, 1, 2)))
    for index, (user, assistant) in enumerate(selected_asides):
        position = rng.randrange(len(messages))
        messages[position:position] = [
            {"role": "user", "turn_id": f"aside{index+1}u", "text": user},
            {"role": "assistant", "turn_id": f"aside{index+1}a", "text": assistant},
        ]
    scope = episode["truth"]["scope_kind"]
    for slot, reply in episode["replies"].items():
        if reply["availability"] == "unavailable":
            reply["text"] = rng.choice(UNAVAILABLE)
        elif scope == "session":
            reply["text"] = rng.choice(SESSION_REPLIES)
        elif scope == "ongoing":
            reply["text"] = rng.choice(ONGOING_REPLIES)
        else:
            raise ValueError(f"Unknown positive-reply scope for {slot}: {scope}")


def enrich(original, *, seed, namespace):
    """All public changes use root-level randomness shared by its latent worlds."""
    episode = deepcopy(original)
    rng = random.Random(f"{seed}:{original['root_id']}")
    _make_extra_memories(episode, rng)
    _dialogue(episode, rng)
    observable = observable_episode_id_v3(episode)
    episode["root_id"] = "nr" + observable[:20]
    episode["latent_world_id"] = "nw" + base._digest([observable, episode["truth"]["scope_kind"]])[:20]
    episode["observable_state_id"] = observable
    episode["template_id"] = "natural:" + original["template_id"]
    episode["composition_id"] = (f"natural:{episode['authoring']['stratum']}:m{len(episode['initial_memory'])}"
                                 f":p{len(episode['prior_sessions'])}:t{len(episode['user_messages'])}"
                                 f":{episode['authoring']['boundary']}:{episode['current_query']['attribute']}")
    episode.update(generator_version=VERSION, data_namespace=namespace)
    episode["authoring"].update(source=VERSION, base_root_id=original["root_id"],
                                naturalism="scripted chat variation; no actual user data")
    return episode


def audit_shape(splits, originals):
    failures = []
    for split, rows in splits.items():
        for episode, original in zip(rows, originals[split], strict=True):
            def check(condition, label):
                if not condition:
                    failures.append({"split": split, "root_id": episode["root_id"], "check": label})
            messages, records = episode["user_messages"], episode["initial_memory"]
            check(len(records) > len(original["initial_memory"]), "added_memories")
            check(len(messages) > len(original["user_messages"]), "added_chat_turns")
            check(episode["truth"] == original["truth"], "gold_unchanged")
            check(episode["current_query"] == original["current_query"], "current_query_unchanged")
            check(episode["future_query"] == original["future_query"], "future_query_unchanged")
            check(len(messages) == len({m["turn_id"] for m in messages}), "unique_chat_turns")
            prior = {m["turn_id"]: (session["date"], m["role"], m["text"])
                     for session in episode["prior_sessions"] for m in session["messages"]}
            check(all(prior.get(q["turn_id"]) == (record["recorded_at"], "user", q["text"])
                      for record in records for q in record["evidence_quotes"]), "historical_user_provenance")
    return failures


def _write_jsonl(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare(output, *, seed, partition_seed, sizes=None):
    output = Path(output)
    sizes = base._sizes(sizes or SIZES)
    historical_paths = base.discover_previous_banks(exclude=output)
    historical, adaptations = base._historical_views(historical_paths)
    source_paths = {**base._sources(), "research/natural_conversation_data.py": Path(__file__)}
    historical_hashes = {name: base._sha(path) for name, path in historical_paths.items()}
    source_hashes = {name: base._sha(path) for name, path in source_paths.items()}
    originals = base.partition_candidates(base.build_candidates(seed=seed, sizes=sizes), sizes, seed=partition_seed)
    base_isolation = assert_semantic_split_isolation_v3(originals, historical_splits=historical)
    splits = {split: [enrich(episode, seed=seed, namespace=output.name) for episode in rows]
              for split, rows in originals.items()}
    isolation = assert_semantic_split_isolation_v3(splits, historical_splits=historical)
    failures = base._static_audit(originals) + audit_shape(splits, originals)
    if failures:
        raise ValueError(f"Data contract audit failed: {failures[:10]}")

    output.mkdir(parents=True)
    partitions, teachers = {}, {}
    for split, rows in splits.items():
        path = output / f"{split}.jsonl"
        _write_jsonl(path, rows)
        partitions[split] = {"path": path.name, "sha256": base._sha(path), **diversity_plan(rows)}
        if split in {"train", "validation"}:
            path = output / f"teacher_{split}.jsonl"
            examples = list(base.teacher_examples(rows))
            _write_jsonl(path, examples)
            teachers[split] = {"path": path.name, "sha256": base._sha(path), "examples": len(examples),
                               "phases": dict(Counter(row["phase"] for row in examples)),
                               "routes": list(base.DEFAULT_ROUTES)}
    audit = base._audit_training_contracts(splits, output)
    audit.update(static_failures=failures, isolation=isolation, unenriched_isolation=base_isolation)
    base._write_json(output / "audit_contracts.json", audit)
    if historical_hashes != {name: base._sha(path) for name, path in historical_paths.items()}:
        raise RuntimeError("Historical corpus changed during generation")
    if source_hashes != {name: base._sha(path) for name, path in source_paths.items()}:
        raise RuntimeError("Source changed during generation")
    manifest = {"version": VERSION, "data_namespace": output.name, "human_review": False, "confirmatory_test": False,
                "seed": seed, "partition_seed": partition_seed, "sizes": sizes, "partitions": partitions,
                "teachers": teachers, "isolation": isolation, "unenriched_isolation": base_isolation,
                "historical_paths": {name: str(path.resolve()) for name, path in historical_paths.items()},
                "historical_hashes_before": historical_hashes, "historical_hashes_after": historical_hashes,
                "historical_identity_adaptations": adaptations,
                "source_hashes_before": source_hashes, "source_hashes_after": source_hashes,
                "audit_contracts": {"path": "audit_contracts.json", "status": audit["status"],
                                    "sha256": base._sha(output / "audit_contracts.json")},
                "training_steps": 0, "Qwen_calls": 0,
                "evaluation_use": "Held-back synthetic episodes, excluded from training and calibration.",
                "limits": ["Scripted colloquial chat, not a recording of actual users or unbounded conversation.",
                           "Established five gold recipes and evaluator grammar. Text enrichment is not a novel-task claim.",
                           "Exact root/behavioral/observable isolation; no claim of universal paraphrase independence.",
                           "Teacher targets generated on train and validation only; never on the held-back split."]}
    base._write_json(output / "manifest.json", manifest)
    if audit["status"] != "verified":
        raise ValueError(f"Symbolic consistency audit failed: {audit['failures'][:4]}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=202611041)
    parser.add_argument("--partition-seed", type=int, default=202611042)
    args = parser.parse_args()
    manifest = prepare(args.output, seed=args.seed, partition_seed=args.partition_seed)
    print(json.dumps({name: {key: item[key] for key in ("roots", "episodes", "memory_counts_by_root", "turn_counts_by_root")}
                      for name, item in manifest["partitions"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
