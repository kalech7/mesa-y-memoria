"""Versioned development recipes authored from coverage requirements, not B1.

No production parser, reader, evaluator or model derives the gold. The recipes
state the expected outcomes using explicit dates, contexts and preference
versions. This is AI-authored synthetic development, never a confirmatory test.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import random
import shutil

from .semantic_partition_v2 import assert_semantic_split_isolation_v2, semantic_episode_id_v2, SEMANTIC_IDENTITY_VERSION

VERSION = "synthetic_development_repair_v2.0"
NAMESPACE = "development_v2_20261024_001"
SLOTS = ("TEMPORAL_SCOPE", "CURRENT_CONTEXT", "VALID_UNTIL", "GENERIC")
FOODS = ("ensalada", "pasta", "arroz", "sopa", "lentejas", "tacos", "cuscús", "pizza")
STRATA = ("temporal_boundaries", "new_context_interval", "successive_supersessions", "unaffected_future_attribute", "ambiguous_temporal_boundary")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _scope(kind, start, *, end=None, context=None, session=None):
    return {"kind": kind, "valid_from": start.isoformat(), "valid_until_exclusive": end.isoformat() if end else None,
            "context": context, "session_id": session}


def _record(mid, value, at, *, attribute="dinner_preference", record_scope=None, supersedes=(), text=None):
    source = "s1_" + mid
    return {"memory_id": mid, "attribute": attribute, "value": value, "recorded_at": at.isoformat(),
            "scope": record_scope or _scope("ongoing", at), "status": "confirmed", "supersedes": list(supersedes),
            "source_turn_ids": [source], "evidence_quotes": [{"turn_id": source, "text": text or f"Desde el {at.isoformat()} prefiero {value} de forma habitual."}]}


def _replies():
    return {slot: {"turn_id": "u3", "text": "No puedo precisar ese dato ahora.", "availability": "unavailable"} for slot in SLOTS}


def _reply(table, slot, text):
    table[slot] = {"turn_id": "u3", "text": text, "availability": "available"}


def build_development_v2(*, seed=4108):
    """Build exactly 50 new roots, with two equivalent phrasings per root.

    Seed permutes the completed bank only. IDs are content-derived; split/seed
    do not manufacture new stories from existing behavioral content.
    """
    result = []
    for stratum_index, stratum in enumerate(STRATA):
        for index in range(10):
            ordinal = stratum_index * 10 + index
            now = date(2027, 3, 1) + timedelta(days=ordinal)
            future = now + timedelta(days=7)
            old = now - timedelta(days=40)
            base, changed, other, fourth = [FOODS[(ordinal + offset) % len(FOODS)] for offset in (0, 3, 5, 7)]
            initial = [_record("base", base, old),
                       _record("expired", other, now - timedelta(days=3),
                               record_scope=_scope("interval", now - timedelta(days=3), end=now - timedelta(days=2)),
                               text=f"Solo el {(now-timedelta(days=3)).isoformat()} preferí {other}.")]
            drink = _record("drink", "agua", old, attribute="drink_preference", text="Para beber prefiero agua.")
            breakfast = _record("breakfast", "fruta", old, attribute="breakfast_preference", text="Para desayunar prefiero fruta.")
            current_attribute, current_context, future_context = "dinner_preference", None, None
            replies = _replies()
            decisive = None
            affected_attribute = "dinner_preference"
            if stratum == "temporal_boundaries":
                family = "F2"
                end = now + timedelta(days=(9, 7, 5)[index % 3])
                inclusive = end - timedelta(days=1)
                real_scope = _scope("interval", now, end=end)
                texts = [f"Solo hasta el {inclusive.isoformat()} inclusive prefiero {changed}. ¿Qué ceno hoy?",
                         f"Como excepción hasta el {inclusive.isoformat()} inclusive prefiero {changed}. ¿Qué me sugieres para cenar hoy?"]
                future_value = changed if future < end else base
                boundary = "inside" if future < end else "exclusive_end" if future == end else "after"
                worlds = [("interval", real_scope, changed, future_value)]
                _reply(replies, "VALID_UNTIL", f"Hasta el {inclusive.isoformat()} inclusive.")
            elif stratum == "new_context_interval":
                family, current_context = "F3", "casa"
                future_context = "casa" if index % 2 == 0 else "oficina"
                end = now + timedelta(days=(9, 7, 5)[index % 3])
                office_end = now + timedelta(days=(5, 7, 9)[index % 3])
                inclusive = end - timedelta(days=1)
                initial = [initial[0], _record("office", other, old + timedelta(days=1),
                           record_scope=_scope("contextual", old, end=office_end, context="oficina"),
                           text=f"En la oficina prefiero {other} hasta el {(office_end-timedelta(days=1)).isoformat()} inclusive.")]
                texts = [f"Solo en casa hasta el {inclusive.isoformat()} inclusive prefiero {changed}. ¿Qué ceno hoy?",
                         f"Como excepción en casa hasta el {inclusive.isoformat()} inclusive prefiero {changed}. ¿Qué me recomiendas para cenar hoy?"]
                real_scope = _scope("contextual", now, end=end, context="casa")
                applicable_end = end if future_context == "casa" else office_end
                applicable_value = changed if future_context == "casa" else other
                future_value = applicable_value if future < applicable_end else base
                boundary = "inside" if future < applicable_end else "exclusive_end" if future == applicable_end else "after"
                worlds = [("contextual_interval", real_scope, changed, future_value)]
                _reply(replies, "CURRENT_CONTEXT", "Hoy estoy en casa.")
                _reply(replies, "VALID_UNTIL", f"Hasta el {inclusive.isoformat()} inclusive.")
            elif stratum == "successive_supersessions":
                family, boundary = "F1", "ongoing"
                intermediate_at, active_at = now - timedelta(days=25), now - timedelta(days=10)
                initial = [initial[0], _record("middle", other, intermediate_at, supersedes=["base"]),
                           _record("active", fourth, active_at, supersedes=["middle"])]
                texts = [f"Desde hoy prefiero {changed} de forma habitual. ¿Qué me recomiendas para cenar?",
                         f"Mi preferencia habitual desde hoy es {changed}. ¿Qué debería cenar?"]
                real_scope = _scope("ongoing", now)
                worlds = [("ongoing", real_scope, changed, changed)]
                _reply(replies, "TEMPORAL_SCOPE", "Como preferencia habitual desde ahora.")
            elif stratum == "unaffected_future_attribute":
                family, boundary, current_attribute = "F4", "unaffected_attribute", "drink_preference"
                texts = [f"Para la cena, ahora prefiero {changed}. ¿Qué me aconsejas beber?",
                         f"Para la cena, para este momento prefiero {changed}. ¿Qué me recomiendas beber hoy?"]
                worlds = [("unaffected_attribute_control", None, "agua", "agua")]
            else:
                family, boundary, decisive = "F4", "exclusive_end" if index % 2 == 0 else "after", "TEMPORAL_SCOPE"
                end = now + timedelta(days=7 if index % 2 == 0 else 5)
                inclusive = end - timedelta(days=1)
                texts = [f"Ahora prefiero {changed}. ¿Qué me recomiendas para la cena de hoy?",
                         f"Para este momento prefiero {changed}. ¿Qué me sugieres para cenar hoy?"]
                worlds = [("interval", _scope("interval", now, end=end), changed, base),
                          ("ongoing", _scope("ongoing", now), changed, changed)]
            initial.extend([drink, breakfast])
            variants = []
            for world_name, world_scope, current_value, future_value in worlds:
                world_replies = deepcopy(replies)
                if stratum == "ambiguous_temporal_boundary":
                    _reply(world_replies, "TEMPORAL_SCOPE", "Como preferencia habitual desde ahora." if world_name == "ongoing"
                           else f"Como excepción hasta el {inclusive.isoformat()} inclusive.")
                    if world_name == "interval":
                        _reply(world_replies, "VALID_UNTIL", f"Hasta el {inclusive.isoformat()} inclusive.")
                for realization, text in enumerate(texts):
                    truth = {"current_value": current_value, "future_value": future_value,
                             "initial_value": "agua" if current_attribute == "drink_preference" else fourth if stratum == "successive_supersessions" else base,
                             "new_value": changed, "scope": deepcopy(world_scope), "scope_kind": world_name,
                             "scope_attribute": affected_attribute, "decisive_slot": decisive,
                             "current_context": current_context, "current_date": now.isoformat(),
                             "future_exclusive_benefit": decisive is not None}
                    variants.append({"root_id": "pending", "latent_world_id": world_name, "realization_id": f"v{realization}",
                        "split": "validation", "scenario_family": family, "data_namespace": NAMESPACE,
                        "data_role": "prospective_development_only", "generator_version": VERSION,
                        "domain": "non_clinical_food_preferences", "template_id": f"v2:{stratum}:v{realization}",
                        "composition_id": f"v2:{stratum}:{boundary}:{future_context or 'global'}",
                        "initial_memory": deepcopy(initial), "user_messages": [{"turn_id": "u2", "text": text}],
                        "current_query": {"attribute": current_attribute, "date": now.isoformat(), "session_id": "S2", "context": current_context},
                        "future_query": {"attribute": current_attribute, "date": future.isoformat(), "session_id": "S3", "context": future_context},
                        "replies": deepcopy(world_replies), "truth": truth,
                        "opportunity": {"eligible": decisive is not None, "negative_control": decisive is None,
                                        "slot": decisive, "affected_attribute": affected_attribute, "tested_attribute": current_attribute},
                        "authoring": {"source": VERSION, "human_authored": False, "independent_human_review": False,
                                      "coverage_source": "SPEC_CODEX_AUDITORIA_DATOS_SINTETICOS_EDGE_CASES_V1 sections 3–4",
                                      "selection_uses_B1_results": False, "gold_source": "Independent construction recipes and literal date/context/version comparisons; no production evaluator/parser/reader.",
                                      "stratum": stratum, "boundary": boundary, "no_confirmatory_test": True}})
            rid = "v2r" + _digest(sorted(semantic_episode_id_v2(e) for e in variants))[:20]
            for episode in variants:
                world_name = episode["latent_world_id"]
                episode["root_id"] = rid
                episode["latent_world_id"] = "v2w" + _digest([rid, world_name])[:20]
                # Rename IDs and every existing reference, without changing data.
                mapping = {m["memory_id"]: rid + "_" + m["memory_id"] for m in episode["initial_memory"]}
                for memory in episode["initial_memory"]:
                    old_id = memory["memory_id"]
                    memory["memory_id"] = mapping[old_id]
                    memory["supersedes"] = [mapping[s] for s in memory["supersedes"]]
                    memory["source_turn_ids"] = ["s1_" + mapping[old_id]]
                    for quote in memory["evidence_quotes"]:
                        quote["turn_id"] = "s1_" + mapping[old_id]
                public = {"memory": episode["initial_memory"], "user_messages": episode["user_messages"],
                          "query": episode["current_query"], "reuse_regime": "later_same_attribute_without_questions"}
                episode["observable_state_id"] = _digest(public)
                result.append(episode)
    random.Random(seed).shuffle(result)
    assert len({e["root_id"] for e in result}) == 50 and len(result) == 120
    return result


def load_historical(directory):
    names = ("dev_smoke", "train", "validation", "test_id", "test_ood", "test_external")
    return {name: [json.loads(line) for line in (Path(directory) / f"{name}.jsonl").open()] for name in names}


def prepare_development_v2(data_directory, report_directory, *, historical_directory, seed=4108):
    data_directory, report_directory = Path(data_directory), Path(report_directory)
    data_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)
    bank = data_directory / "validation.jsonl"
    if bank.exists() or (data_directory / "manifest.json").exists():
        raise FileExistsError("Versioned development already exists; use a new directory or the check command")
    historical = load_historical(historical_directory)
    historical_hashes = {name: _sha(Path(historical_directory)/f"{name}.jsonl") for name in historical}
    snapshots = report_directory / "data_source_snapshot"
    snapshots.mkdir(exist_ok=True)
    for name, expected in historical_hashes.items():
        source, target = Path(historical_directory)/f"{name}.jsonl", snapshots/f"{name}.jsonl"
        if target.exists():
            if _sha(target) != expected:
                raise ValueError("Historical snapshot differs: " + str(target))
        else:
            shutil.copyfile(source, target)
    code_snapshots = report_directory / "data_generator_source_snapshot"
    code_snapshots.mkdir(exist_ok=True)
    for source in (Path(__file__), Path(__file__).with_name("semantic_partition_v2.py")):
        target = code_snapshots/source.name
        if target.exists() and _sha(target) != _sha(source):
            raise ValueError("Source snapshot already exists with different code; use new report version")
        if not target.exists():
            shutil.copyfile(source, target)
    episodes = build_development_v2(seed=seed)
    isolation = assert_semantic_split_isolation_v2({"validation": episodes}, historical_splits=historical)
    with bank.open("x") as stream:
        for episode in episodes:
            stream.write(json.dumps(episode, ensure_ascii=False, sort_keys=True) + "\n")
    current_hashes = {name: _sha(Path(historical_directory)/f"{name}.jsonl") for name in historical}
    assert historical_hashes == current_hashes
    manifest = {"version": VERSION, "data_namespace": NAMESPACE, "data_role": "prospective_development_only",
                "split": "validation", "roots": 50, "episodes": len(episodes), "seed_controls": "Output order only; root/world IDs derive from content.",
                "seed": seed, "bank_path": str(bank.resolve()), "bank_sha256": _sha(bank),
                "strata_roots": {s: len({e["root_id"] for e in episodes if e["authoring"]["stratum"] == s}) for s in STRATA},
                "worlds": len({e["latent_world_id"] for e in episodes}), "realizations": dict(Counter(e["realization_id"] for e in episodes)),
                "templates": sorted({e["template_id"] for e in episodes}), "compositions": sorted({e["composition_id"] for e in episodes}),
                "semantic_identity_version": SEMANTIC_IDENTITY_VERSION, "isolation": isolation,
                "historical_bank_hashes_before": historical_hashes, "historical_bank_hashes_after": current_hashes,
                "producer_source_hashes": {s.name: _sha(s) for s in code_snapshots.iterdir()},
                "training": False, "new_Qwen_calls": 0, "model_results_used_for_design": False,
                "confirmatory_test_created": False, "reviewer_type": "ai_authored_development",
                "limits": ["Fixed synthetic food grammar; no human-realism or broad OOD claim.",
                           "Exact identity isolation does not detect every paraphrase or structural shortcut.",
                           "Dates/food renaming are not counted as new compositions.",
                           "Unrelated future attribute means beverage queries while the new preference concerns dinner; S2/S3 query attributes remain equal.",
                           "Explicit boundary/context/supersession recipes are new development requirements, not confirmed learning improvements."]}
    (data_directory/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    (report_directory/"data_preparation_v2.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--data-directory", required=True)
    prepare.add_argument("--report-directory", required=True)
    prepare.add_argument("--historical-directory", default="research/data/bank_1_1")
    prepare.add_argument("--seed", type=int, default=4108)
    verify = sub.add_parser("check")
    verify.add_argument("--bank", required=True)
    verify.add_argument("--historical-directory", default="research/data/bank_1_1")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_development_v2(args.data_directory, args.report_directory,
                                        historical_directory=args.historical_directory, seed=args.seed)
        result = {k:result[k] for k in ("version", "roots", "episodes", "bank_path", "bank_sha256")}
    else:
        episodes = [json.loads(line) for line in Path(args.bank).open()]
        result = assert_semantic_split_isolation_v2({"validation": episodes}, historical_splits=load_historical(args.historical_directory))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
