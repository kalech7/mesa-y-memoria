"""Synthetic development episodes; private annotations never belong in prompts.

The grammar is a transparent development generator, not independent OOD or human
authorship. Root partitioning precedes wording and all counterfactual worlds.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import random

FAMILIES = ("F0", "F1", "F2", "F3", "F4")
FOODS = ("ensalada", "pasta", "arroz", "sopa", "lentejas", "tacos", "cuscús", "pizza")
SLOTS = ("TEMPORAL_SCOPE", "CURRENT_CONTEXT", "VALID_UNTIL", "GENERIC")


def _id(seed: int, *parts: object) -> str:
    return hashlib.sha256(json.dumps([seed, *parts]).encode()).hexdigest()[:16]


def scope(kind="ongoing", *, start=None, end=None, context=None, session=None):
    return {"kind": kind, "session_id": session, "context": context,
            "valid_from": start, "valid_until_exclusive": end}


def _record(mid, value, at, text, *, attribute="dinner_preference", record_scope=None):
    return {"memory_id": mid, "attribute": attribute, "value": value,
            "scope": record_scope or scope(start=at), "status": "confirmed",
            "source_turn_ids": ["s1_" + mid],
            "evidence_quotes": [{"turn_id": "s1_" + mid, "text": text}],
            "supersedes": [], "recorded_at": at}


def build_bank(split="dev_smoke", roots=40, seed=4107, realizations=1, *, root=None):
    """Construct balanced roots and paired worlds, never generate external cases.

    Full training/validation/ID generation requires the mechanism gate G1.
    Independent OOD and external writing must be imported and audited separately.
    """
    if split not in {"dev_smoke", "train", "validation", "test_id"}:
        raise ValueError("INDEPENDENT_AUTHORING_REQUIRED: OOD/external are not cosmetic variants")
    if roots < 5 or roots % 5 or realizations < 1:
        raise ValueError("Root count must be a positive multiple of five; realizations >= 1")
    if split != "dev_smoke":
        from research.src.protocol import require_gate
        require_gate("G1", root=root)
    elif roots > 40:
        raise ValueError("dev_smoke is capped at 40 roots")
    rng = random.Random(seed)
    # Each cycle contains all families; foods rotate independently of family.
    result = []
    for i in range(roots):
        family = FAMILIES[i % 5]
        root_id = "r" + _id(seed, split, i)
        base, new, other = [FOODS[j % len(FOODS)] for j in (i, i + 3, i + 5)]
        when = date(2026, 1, 12) + timedelta(days=i % 25)
        future = when + timedelta(days=7)
        old = when - timedelta(days=25)
        recent = when - timedelta(days=2)
        mids = ["m" + _id(seed, root_id, n)[:10] for n in range(5)]
        initial = [_record(mids[0], base, old.isoformat(),
                           f"En general prefiero {base} para cenar.")]
        if family == "F3":
            initial[0]["scope"] = scope("contextual", start=old.isoformat(), context="casa")
            initial[0]["evidence_quotes"][0]["text"] = f"En casa prefiero {base} para cenar."
            initial.append(_record(mids[1], new, (old + timedelta(days=1)).isoformat(),
                           f"En la oficina prefiero {new} para cenar.",
                           record_scope=scope("contextual", start=old.isoformat(), context="oficina")))
        else:
            # A recently recorded but expired exception makes O1 genuinely lossy.
            initial.append(_record(mids[1], other, recent.isoformat(),
                           f"Solo aquel día preferí {other} para cenar.",
                           record_scope=scope("interval", start=recent.isoformat(),
                                              end=(recent + timedelta(days=1)).isoformat())))
        initial.extend([
            _record(mids[2], "agua", (old + timedelta(days=3)).isoformat(),
                    "Para beber prefiero agua.", attribute="drink_preference"),
            _record(mids[3], "fruta", (old + timedelta(days=5)).isoformat(),
                    "Para desayunar prefiero fruta.", attribute="breakfast_preference"),
        ])
        current_query = {"attribute": "dinner_preference", "date": when.isoformat(),
                         "session_id": "S2", "context": None}
        future_query = {"attribute": "dinner_preference", "date": future.isoformat(),
                        "session_id": "S3", "context": None}
        decisive = None
        if family == "F0":
            texts = ("¿Qué me recomiendas para cenar hoy?", "¿Qué cenaría hoy según mis preferencias?")
            worlds = [("stable", base, base, None)]
        elif family == "F1":
            texts = (f"Desde hoy prefiero {new} de forma habitual. ¿Qué me recomiendas para cenar?",
                     f"Mi preferencia habitual desde hoy es {new}. ¿Qué debería cenar?")
            worlds = [("ongoing", new, new, scope(start=when.isoformat()))]
        elif family == "F2":
            end = when + timedelta(days=1 + i % 3)
            texts = (f"Solo hasta el {end.isoformat()} inclusive prefiero {new}. ¿Qué ceno hoy?",
                     f"Como excepción hasta el {end.isoformat()} inclusive prefiero {new}. ¿Qué me sugieres hoy?")
            worlds = [("interval", new, base,
                       scope("interval", start=when.isoformat(), end=(end + timedelta(days=1)).isoformat()))]
        elif family == "F3":
            explicit = (i // 5) % 2 == 0
            if explicit:
                current_query["context"] = "casa"
                texts = ("Hoy ceno en casa. ¿Qué me recomiendas?", "¿Qué cenaría hoy en casa?")
                worlds = [("casa", base, new, None)]
            else:
                texts = ("¿Qué me recomiendas para cenar hoy?", "¿Qué debería cenar hoy?")
                worlds = [("casa", base, new, None), ("oficina", new, new, None)]
                decisive = "CURRENT_CONTEXT"
            future_query["context"] = "oficina"
        else:
            decisive = "TEMPORAL_SCOPE"
            texts = (f"Ahora prefiero {new}. ¿Qué me recomiendas para la cena de hoy?",
                     f"Para este momento prefiero {new}. ¿Qué me sugieres para cenar hoy?")
            short_kind = "interval" if (i // 5) % 3 == 2 else "session"
            short_scope = (scope("interval", start=when.isoformat(),
                                 end=(when + timedelta(days=2)).isoformat())
                           if short_kind == "interval" else
                           scope("session", start=when.isoformat(), session="S2"))
            worlds = [(short_kind, new, base, short_scope),
                      ("ongoing", new, new, scope(start=when.isoformat()))]
        for realization in range(realizations):
            wording = texts[realization % len(texts)]
            for world_name, current_value, future_value, true_scope in worlds:
                replies = {slot: {"turn_id": "u3", "text": "No tengo más detalles sobre eso."}
                           for slot in SLOTS}
                if family == "F4":
                    answer = {"session": "Solo en esta conversación.",
                              "ongoing": "Como preferencia habitual desde ahora.",
                              "interval": f"Como excepción hasta el {(when + timedelta(days=1)).isoformat()} inclusive."}[world_name]
                    replies[decisive] = {"turn_id": "u3", "text": answer}
                elif family == "F3":
                    replies["CURRENT_CONTEXT"] = {"turn_id": "u3", "text": f"Hoy estoy en {world_name}."}
                elif family == "F1":
                    replies["TEMPORAL_SCOPE"] = {"turn_id": "u3", "text": "Como preferencia habitual desde ahora."}
                elif family == "F2":
                    replies["VALID_UNTIL"] = {"turn_id": "u3", "text": f"Hasta el {end.isoformat()} inclusive."}
                result.append({
                    "root_id": root_id, "latent_world_id": "w" + _id(seed, root_id, world_name),
                    "realization_id": "v" + str(realization), "split": split, "scenario_family": family,
                    "initial_memory": deepcopy(initial), "user_messages": [{"turn_id": "u2", "text": wording}],
                    "current_query": deepcopy(current_query), "future_query": deepcopy(future_query),
                    "replies": replies, "truth": {
                        "current_value": current_value, "future_value": future_value,
                        "initial_value": base, "new_value": new, "scope": true_scope,
                        "scope_kind": world_name, "decisive_slot": decisive,
                        "current_context": world_name if family == "F3" else None,
                        "future_exclusive_benefit": family == "F4", "current_date": when.isoformat(),
                    },
                    "opportunity": {"eligible": family == "F4", "slot": "TEMPORAL_SCOPE" if family == "F4" else None,
                                    "negative_control": family in {"F0", "F1", "F2"}},
                    "authoring": {"source": "synthetic_development_grammar_v1", "independent": False},
                })
                if family == "F4":
                    # Matched explicit controls remain in the same statistical
                    # root and partition. Their visible scope intentionally
                    # differs between worlds; T02 concerns ambiguous prefixes.
                    control = deepcopy(result[-1])
                    control["realization_id"] = "control_v" + str(realization)
                    if world_name == "ongoing":
                        statement = f"Desde hoy prefiero {new} de forma habitual. ¿Qué ceno hoy?"
                    elif world_name == "session":
                        statement = f"Solo en esta conversación prefiero {new}. ¿Qué ceno hoy?"
                    else:
                        inclusive = date.fromisoformat(true_scope["valid_until_exclusive"]) - timedelta(days=1)
                        statement = f"Solo hasta el {inclusive.isoformat()} inclusive prefiero {new}. ¿Qué ceno hoy?"
                    control["user_messages"] = [{"turn_id": "u2", "text": statement}]
                    control["truth"]["decisive_slot"] = None
                    control["truth"]["future_exclusive_benefit"] = False
                    control["opportunity"] = {"eligible": False, "slot": None, "negative_control": True}
                    result.append(control)
    rng.shuffle(result)
    return result


def build_full_bank(*, root=None, seed=4107, sizes=None):
    sizes = sizes or {"train": 600, "validation": 150, "test_id": 300}
    return {split: build_bank(split, count, seed, root=root) for split, count in sizes.items()}


def write_bank(episodes, destination):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve provenance and previous versions; callers choose a new directory.
    with path.open("x", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=False, sort_keys=True) + "\n")


def teacher_trajectories(episodes):
    """Observable-only demonstrations, including irrelevant/nonresponse prefixes.

    These are examples, not executed experimental outcomes. Hidden truth is never
    consulted by the teacher. The normal trajectory and the two robustness
    continuations have separate IDs and retain their original prefixes.
    """
    from research.src.policy.rule_controller import RuleController
    teacher = RuleController()
    for episode in episodes:
        observation = {"memory": deepcopy(episode["initial_memory"]),
                       "user_messages": deepcopy(episode["user_messages"]),
                       "query": deepcopy(episode["current_query"]),
                       "reuse_regime": "later_same_attribute_without_questions"}
        decision = teacher.select(observation)
        yield {"phase": "select", "observation": deepcopy(observation), "action": decision,
               "completion": json.dumps(decision, ensure_ascii=False, sort_keys=True), "split": episode["split"],
               "root_id": episode["root_id"], "source": "observable_rule_teacher"}
        continuations = [("normal", episode["replies"].get(decision["slot"]))
                         if decision["action"] == "ASK" else ("normal", None),
                         ("irrelevant", episode["replies"]["GENERIC"]),
                         ("nonresponse", {"turn_id": "u3", "text": "No sé precisar eso ahora."})]
        for label, reply in continuations:
            obs = deepcopy(observation)
            if reply:
                obs["user_messages"].append(deepcopy(reply))
            for phase in ("respond", "write"):
                action = getattr(teacher, phase)(obs)
                yield {"phase": phase, "observation": deepcopy(obs),
                       "action": action, "completion": json.dumps(action, ensure_ascii=False, sort_keys=True),
                       "split": episode["split"], "root_id": episode["root_id"],
                       "continuation": label, "source": "observable_rule_teacher"}


def import_independent_bank(path, split, *, generator_author_id=None):
    """Import real independently authored cases without certifying semantics.

    Independent authors supply the same episode schema and private evaluator
    annotations. This checks syntax/provenance only. A separate semantic review
    remains pending, so importing a file cannot by itself open the test gate.
    """
    from research.src.memory.executor import validate_record
    from research.src.data.audit_splits import audit_splits, digest
    if split not in {"test_ood", "test_external"}:
        raise ValueError("Independent import is for OOD/external cases")
    source = Path(path)
    if source.suffix == ".jsonl":
        episodes = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        episodes = payload.get("episodes") if isinstance(payload, dict) else payload
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("A nonempty independent episode list is required")
    identities = set()
    required = {"root_id", "latent_world_id", "realization_id", "split", "scenario_family",
                "initial_memory", "user_messages", "current_query", "future_query", "replies", "truth", "authoring"}
    for episode in episodes:
        if not isinstance(episode, dict) or not required <= episode.keys():
            raise ValueError("Incomplete independent episode schema")
        if episode["split"] != split or episode["scenario_family"] not in FAMILIES:
            raise ValueError("Independent split/family mismatch")
        authorship = episode["authoring"]
        if not isinstance(authorship, dict) or authorship.get("independent") is not True or not authorship.get("author_id"):
            raise ValueError("INDEPENDENT_AUTHORSHIP_ATTESTATION_REQUIRED")
        if generator_author_id is not None and authorship["author_id"] == generator_author_id:
            raise ValueError("External author must differ from generator author")
        if split == "test_ood" and not authorship.get("held_out_mechanisms"):
            raise ValueError("OOD requires documented held-out mechanisms, not cosmetic wording")
        identity = tuple(episode[k] for k in ("root_id", "latent_world_id", "realization_id"))
        if identity in identities:
            raise ValueError("Duplicate independent episode identity")
        identities.add(identity)
        for record in episode["initial_memory"]:
            validate_record(record, initial=True)
        for query_name in ("current_query", "future_query"):
            query = episode[query_name]
            if not isinstance(query, dict) or not {"attribute", "date", "session_id", "context"} <= query.keys():
                raise ValueError("Malformed independent query")
            date.fromisoformat(query["date"])
        messages = episode["user_messages"]
        if not isinstance(messages, list) or not messages or any(
            not isinstance(m, dict) or not isinstance(m.get("turn_id"), str) or not isinstance(m.get("text"), str)
            for m in messages):
            raise ValueError("Malformed independent user evidence")
        if not {"current_value", "future_value"} <= episode["truth"].keys():
            raise ValueError("Independent evaluator annotations required")
        if not set(SLOTS) <= episode["replies"].keys():
            raise ValueError("Every permitted slot needs a fixed simulator reply")
    audit = audit_splits({split: episodes})
    audit.update({"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "episode_sha256": digest(episodes), "source_file": source.name,
                  "semantic_audit_status": "pending_independent_semantic_audit",
                  "external_authorship": "attested_not_independently_verified"})
    return {"episodes": episodes, "audit": audit}
