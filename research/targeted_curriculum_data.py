"""Prospective conversations and a TRAIN-only curriculum of grounded actions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path
import random

from . import competence_data as base
from . import natural_conversation_data as natural
from .src.artifacts import canonical, digest, file_hash, write_json
from .src.data.semantic_partition_v3 import assert_semantic_split_isolation_v3, observable_episode_id_v3
from .src.memory.executor import validate_record
from .src.policy.dialogue import declarations, preference_turn
from .src.policy.llm_controller import LLMController, validate_action
from .src.scoring.evidence import audit_operation
from .varied_development import CLAUSE, VOCABULARY, diversity_plan

VERSION = "targeted_curriculum_v2_causal_reuse"
DEFAULT_SIZES = {"train": 60, "validation": 15, "evaluation": 25}
DEFAULT_SEED = 202611057
DEFAULT_PARTITION_SEED = 202611058


def _write_rows(path, rows):
    with Path(path).open("x") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")


def _identity(episode):
    return tuple(episode[key] for key in ("root_id", "latent_world_id", "realization_id"))


def _no_new_preference(episode):
    """A stable F0 recipe can answer from memory without any new fact to save."""
    users = episode["user_messages"]
    for message in users:
        if message["role"] == "user" and preference_turn(message):
            message["text"] = "Gracias, sigamos con lo que ya habíamos hablado."
    episode["authoring"]["targeted_case"] = "no_new_fact"


def _cross_attribute_tail(episode, rng):
    """Add a grounded second preference after the queried attribute's update."""
    query_attribute = episode["current_query"]["attribute"]
    existing = [attribute for _, attribute in declarations({"user_messages": episode["user_messages"],
                                                            "query": episode["current_query"]})
                if attribute != query_attribute]
    other = existing[-1] if existing else rng.choice([attribute for attribute in VOCABULARY if attribute != query_attribute])
    value = rng.choice(VOCABULARY[other])
    episode["user_messages"][-1:-1] = [
        {"role": "user", "turn_id": "curriculum_extra_u", "text":
         f"Ah, otra cosa. {CLAUSE[other]} prefiero {value} de forma habitual."},
        {"role": "assistant", "turn_id": "curriculum_extra_a", "text": "De acuerdo. Volvamos a tu consulta."},
    ]
    episode["authoring"]["targeted_case"] = "query_differs_from_last_declaration"


def _context_question(episode):
    """Current place is private until the user answers the requested question."""
    episode["current_query"]["context"] = None
    for message in episode["user_messages"]:
        if message["role"] == "user":
            message["text"] = message["text"].replace("Hoy estoy en casa. ", "")
    reply = episode["replies"]["CURRENT_CONTEXT"]
    reply.update(text="Ahora estoy en casa.", availability="available")
    episode["truth"]["decisive_slot"] = "CURRENT_CONTEXT"
    episode["opportunity"].update(eligible=True, negative_control=False, slot="CURRENT_CONTEXT")
    episode["authoring"]["context_question"] = True


def _future_reuse(episode, *, interval_inside):
    """Author delayed reuse from the declared scope, without invoking a reader."""
    stratum = episode["authoring"]["stratum"]
    current = date.fromisoformat(episode["current_query"]["date"])
    if stratum == "contextual_scope":
        end = episode["truth"]["scope"].get("valid_until_exclusive")
        future = current + timedelta(days=1 if end else 7)
        if end and future >= date.fromisoformat(end):
            raise ValueError("Contextual reuse must fall inside its authored validity")
        episode["future_query"].update(date=future.isoformat(), context="casa")
        episode["truth"]["future_value"] = episode["truth"]["new_value"]
        episode["authoring"].update(reuse_recipe="same_context_new_fact_required",
                                    boundary="inside" if end else "ongoing")
    elif stratum == "separate_interval":
        end = date.fromisoformat(episode["truth"]["scope"]["valid_until_exclusive"])
        future = current + timedelta(days=1) if interval_inside else end + timedelta(days=1)
        if interval_inside and future >= end:
            raise ValueError("Inside-interval reuse must precede its authored exclusive end")
        episode["future_query"]["date"] = future.isoformat()
        episode["truth"]["future_value"] = (episode["truth"]["new_value"] if interval_inside
                                             else episode["truth"]["initial_value"])
        episode["authoring"].update(reuse_recipe=("interval_inside_new_fact_required" if interval_inside
                                                 else "interval_expired_initial_memory"),
                                    boundary="inside" if interval_inside else "after")
    else:
        episode["authoring"]["reuse_recipe"] = {
            "ongoing_correction": "ongoing_new_fact_required",
            "unrelated_attribute_control": "stable_no_new_fact",
            "ambiguous_duration": "clarified_ongoing_new_fact_required" if episode["truth"]["scope_kind"] == "ongoing"
                                  else "clarified_session_preserve_baseline",
        }[stratum]


def enrich(episode, *, seed, namespace, interval_inside=None):
    result = natural.enrich(episode, seed=seed, namespace=namespace)
    rng = random.Random(f"{VERSION}:{seed}:{episode['root_id']}")
    # The no-new-fact control is authored from a stable queried-attribute recipe.
    # Assignment is shared by all worlds and fixed before any model is sampled.
    if episode["authoring"]["stratum"] == "unrelated_attribute_control":
        _no_new_preference(result)
    else:
        _cross_attribute_tail(result, rng)
    if episode["authoring"]["stratum"] == "contextual_scope":
        _context_question(result)
    if interval_inside is None:
        interval_inside = int(digest([VERSION, seed, episode["root_id"]])[:8], 16) % 2 == 0
    _future_reuse(result, interval_inside=interval_inside)
    observable = observable_episode_id_v3(result)
    result.update(root_id="tr" + observable[:20],
                  latent_world_id="tw" + digest([observable, result["truth"]["scope_kind"]])[:20],
                  observable_state_id=observable, generator_version=VERSION,
                  template_id="targeted:" + episode["template_id"])
    result["authoring"].update(source=VERSION, base_root_id=episode["root_id"],
                               target_selection="authored task coverage, independent of model outcomes")
    return result


def _shape_audit(splits, originals):
    expected = deepcopy(originals)
    for split, rows in splits.items():
        for episode, original in zip(rows, expected[split], strict=True):
            if episode["authoring"].get("context_question"):
                original["current_query"]["context"] = None
                original["truth"]["decisive_slot"] = "CURRENT_CONTEXT"
            _future_reuse(original, interval_inside=episode["authoring"].get("reuse_recipe") == "interval_inside_new_fact_required")
    failures = natural.audit_shape(splits, expected)
    for split, rows in splits.items():
        for episode in rows:
            public = {"user_messages": episode["user_messages"], "query": episode["current_query"]}
            entries = declarations(public)
            case = episode["authoring"]["targeted_case"]
            if case == "no_new_fact" and entries:
                failures.append({"split": split, "root_id": episode["root_id"], "check": "unexpected_new_fact"})
            if case == "query_differs_from_last_declaration" and (
                    not entries or entries[-1][1] == episode["current_query"]["attribute"]):
                failures.append({"split": split, "root_id": episode["root_id"], "check": "query_tail_attribute"})
            if len({attribute for _, attribute in entries}) > 2:
                failures.append({"split": split, "root_id": episode["root_id"], "check": "two_operation_budget"})
    return failures


def prepare(output, *, seed=DEFAULT_SEED, partition_seed=DEFAULT_PARTITION_SEED, sizes=None):
    """Create fresh isolated episode files, with EVAL untouched by teacher/scoring."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new targeted-data destination")
    sizes = base._sizes(sizes or DEFAULT_SIZES)
    historical_paths = base.discover_previous_banks(exclude=output)
    historical, adaptations = base._historical_views(historical_paths)
    sources = {**base._sources(), "research/natural_conversation_data.py": Path(natural.__file__),
               "research/targeted_curriculum_data.py": Path(__file__)}
    historical_hashes = {name: file_hash(path) for name, path in historical_paths.items()}
    source_hashes = {name: file_hash(path) for name, path in sources.items()}
    originals = base.partition_candidates(base.build_candidates(seed=seed, sizes=sizes), sizes, seed=partition_seed)
    base_isolation = assert_semantic_split_isolation_v3(originals, historical_splits=historical)
    interval_assignments = {}
    for role, rows in originals.items():
        roots = sorted({row["root_id"] for row in rows if row["authoring"]["stratum"] == "separate_interval"},
                       key=lambda rid: digest([VERSION, seed, role, rid]))
        interval_assignments.update({root: index % 2 == 0 for index, root in enumerate(roots)})
    splits = {role: [enrich(episode, seed=seed, namespace=output.name,
                           interval_inside=interval_assignments.get(episode["root_id"])) for episode in rows]
              for role, rows in originals.items()}
    isolation = assert_semantic_split_isolation_v3(splits, historical_splits=historical)
    failures = base._static_audit(originals) + _shape_audit(splits, originals)
    if failures:
        raise ValueError(f"Targeted authored-data audit failed: {failures[:5]}")
    output.mkdir(parents=True)
    partitions, teachers = {}, {}
    for role, rows in splits.items():
        path = output / f"{role}.jsonl"
        _write_rows(path, rows)
        partitions[role] = {"path": path.name, "sha256": file_hash(path), **diversity_plan(rows),
                            "targeted_cases_by_root": dict(Counter(row["authoring"]["targeted_case"]
                                for row in {episode["root_id"]: episode for episode in rows}.values())),
                            "reuse_recipes_by_episode": dict(Counter(row["authoring"]["reuse_recipe"] for row in rows))}
        if role in {"train", "validation"}:
            path = output / f"teacher_{role}.jsonl"
            examples = list(base.teacher_examples(rows))
            _write_rows(path, examples)
            teachers[role] = {"path": path.name, "sha256": file_hash(path), "examples": len(examples),
                              "phases": dict(Counter(row["phase"] for row in examples)),
                              "routes": list(base.DEFAULT_ROUTES)}
    audit = base._audit_training_contracts(splits, output)
    audit.update(static_failures=failures, isolation=isolation, unenriched_isolation=base_isolation)
    write_json(output / "audit_contracts.json", audit)
    if source_hashes != {name: file_hash(path) for name, path in sources.items()}:
        raise RuntimeError("Targeted-data source changed during generation")
    if historical_hashes != {name: file_hash(path) for name, path in historical_paths.items()}:
        raise RuntimeError("Historical input bank changed during generation")
    manifest = {"version": VERSION, "data_namespace": output.name, "human_review": False,
                "confirmatory_test": False, "seed": seed, "partition_seed": partition_seed,
                "sizes": sizes, "partitions": partitions, "teachers": teachers,
                "isolation": isolation, "unenriched_isolation": base_isolation,
                "historical_paths": {name: str(path.resolve()) for name, path in historical_paths.items()},
                "historical_hashes_before": historical_hashes, "historical_hashes_after": historical_hashes,
                "historical_identity_adaptations": adaptations,
                "source_hashes_before": source_hashes, "source_hashes_after": source_hashes,
                "audit_contracts": {"path": "audit_contracts.json", "status": audit["status"],
                                    "sha256": file_hash(output / "audit_contracts.json")},
                "training_steps": 0, "Qwen_calls": 0,
                "evaluation_use": "Static isolation/provenance only until validation-only checkpoint selection.",
                "limits": ["Scripted synthetic conversations and finite public grammar.",
                           "Literal source isolation does not establish structural or human-distribution generalization.",
                           "Teacher targets use public evidence and are produced on TRAIN/VALIDATION only."]}
    write_json(output / "manifest.json", manifest)
    if audit["status"] != "verified":
        raise ValueError(f"Targeted symbolic contract audit failed: {audit['failures'][:4]}")
    return manifest


def _check_target(example):
    action = validate_action(example["phase"], example["action"])
    if json.loads(example["completion"]) != action:
        raise ValueError("Teacher completion differs from its action")
    LLMController(None).prompt(example["phase"], example["observation"])
    if example["phase"] == "write":
        for operation in action["operations"]:
            if operation["op"] != "NOOP":
                validate_record(operation["record"])
            if not audit_operation(operation, example["observation"]):
                raise ValueError("A curriculum operation is unsupported by its public evidence")


def curriculum_tags(example):
    tags = {"phase_" + example["phase"], "route_" + example["route"]}
    observation = example["observation"]
    declared = declarations(observation)
    if declared and declared[-1][1] != observation["query"]["attribute"]:
        tags.add("query_differs_from_last_declaration")
    if len(observation["memory"]) > 3:
        tags.add("multiple_memory_distractors")
    if example["phase"] == "select" and example["action"]["action"] == "ASK":
        tags.add("select_ASK_" + example["action"]["slot"])
    if example["phase"] == "write":
        for operation in example["action"]["operations"]:
            tags.add("operation_" + operation["op"])
            if "record" in operation:
                tags.add("scope_" + operation["record"]["scope"]["kind"])
    return sorted(tags)


def build_curriculum(bank_dir, output, *, roots_per_operation=4, max_examples=180, min_examples=120):
    """Select whole TRAIN roots by deterministic teacher-operation coverage."""
    if isinstance(roots_per_operation, bool) or not isinstance(roots_per_operation, int) or roots_per_operation < 1:
        raise ValueError("roots_per_operation must be a positive integer")
    if isinstance(max_examples, bool) or not isinstance(max_examples, int) or max_examples < 1:
        raise ValueError("max_examples must be a positive integer")
    if isinstance(min_examples, bool) or not isinstance(min_examples, int) or not 1 <= min_examples <= max_examples:
        raise ValueError("min_examples must be within the positive example cap")
    bank_dir, output = Path(bank_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh TRAIN curriculum destination")
    splits, manifest = base.load_and_check(bank_dir)
    teacher_info = manifest["teachers"]["train"]
    teacher_path = bank_dir / teacher_info["path"]
    if file_hash(teacher_path) != teacher_info["sha256"]:
        raise ValueError("TRAIN teacher file changed")
    examples = [json.loads(line) for line in teacher_path.read_text().splitlines() if line.strip()]
    identities = {_identity(episode) for episode in splits["train"]}
    by_root, bucket_roots = defaultdict(list), defaultdict(set)
    for example in examples:
        if example["split"] != "train" or _identity(example) not in identities:
            raise ValueError("Curriculum accepts targets belonging to TRAIN only")
        _check_target(example)
        tags = curriculum_tags(example)
        by_root[example["root_id"]].append({**example, "curriculum_tags": tags})
        for tag in tags:
            bucket_roots[tag].add(example["root_id"])
    required = ("operation_NOOP", "operation_MARK_UNRESOLVED", "operation_ADD", "operation_SUPERSEDE",
                "query_differs_from_last_declaration", "multiple_memory_distractors",
                "select_ASK_CURRENT_CONTEXT", "select_ASK_TEMPORAL_SCOPE")
    missing = [tag for tag in required if not bucket_roots[tag]]
    if missing:
        raise ValueError(f"Missing required TRAIN teacher coverage: {missing}")
    ranked = {tag: sorted(bucket_roots[tag], key=lambda rid: digest([VERSION, tag, rid])) for tag in required}
    selected = set()
    # First cover every class, then expand root quotas while respecting the
    # actual number of targets across worlds and routes of each whole root.
    for quota in range(1, roots_per_operation + 1):
        for tag in required:
            if len(selected & bucket_roots[tag]) >= quota:
                continue
            for root in ranked[tag]:
                if root in selected:
                    continue
                proposed = selected | {root}
                if sum(len(by_root[rid]) for rid in proposed) <= max_examples:
                    selected = proposed
                    break
    uncovered = [tag for tag in required if not selected & bucket_roots[tag]]
    if uncovered:
        raise ValueError(f"Curriculum budget cannot cover required classes: {uncovered}")
    for root in sorted(by_root, key=lambda rid: digest([VERSION, "coverage_fill", rid])):
        count = sum(len(by_root[rid]) for rid in selected)
        if count >= min_examples:
            break
        if root not in selected and count + len(by_root[root]) <= max_examples:
            selected.add(root)
    coverage_selection = {tag: sorted(selected & bucket_roots[tag]) for tag in required}
    selected_examples = [example for root in sorted(selected) for example in by_root[root]]
    output.mkdir(parents=True)
    _write_rows(output / "examples.jsonl", selected_examples)
    counts = Counter(tag for example in selected_examples for tag in example["curriculum_tags"])
    report = {"version": VERSION, "status": "verified", "source_bank": str(bank_dir),
              "source_manifest_sha256": file_hash(bank_dir / "manifest.json"),
              "source_train_sha256": manifest["partitions"]["train"]["sha256"],
              "source_teacher_train_sha256": file_hash(teacher_path),
              "source_module_sha256": file_hash(Path(__file__)),
              "examples_path": str(output / "examples.jsonl"),
              "examples_sha256": file_hash(output / "examples.jsonl"),
              "examples": len(selected_examples), "roots": len(selected), "selected_roots": sorted(selected),
              "requested_roots_per_bucket": roots_per_operation, "max_examples": max_examples, "min_examples": min_examples,
              "coverage_selection": coverage_selection,
              "achieved_roots_per_bucket": {tag: len(roots) for tag, roots in coverage_selection.items()},
              "coverage_counts": dict(counts), "phase_counts": dict(Counter(row["phase"] for row in selected_examples)),
              "all_examples_split": "train", "heldback_targets_used": False,
              "selection_basis": "Observable teacher-operation coverage on TRAIN; no model outcomes.",
              "limitations": ["Root quotas expand subject to the total example cap; achieved coverage is recorded.",
                              "Whole roots retain every latent world and all recorded teacher routes."]}
    write_json(output / "CURRICULUM.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare_parser.add_argument("--partition-seed", type=int, default=DEFAULT_PARTITION_SEED)
    curriculum_parser = sub.add_parser("curriculum")
    curriculum_parser.add_argument("--bank", required=True)
    curriculum_parser.add_argument("--output", required=True)
    curriculum_parser.add_argument("--roots-per-operation", type=int, default=4)
    curriculum_parser.add_argument("--max-examples", type=int, default=180)
    curriculum_parser.add_argument("--min-examples", type=int, default=120)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output, seed=args.seed, partition_seed=args.partition_seed)
        print(canonical({"version": result["version"], "sizes": result["sizes"], "audit": result["audit_contracts"]}))
    else:
        result = build_curriculum(args.bank, args.output, roots_per_operation=args.roots_per_operation, max_examples=args.max_examples, min_examples=args.min_examples)
        print(canonical({key: result[key] for key in ("status", "examples", "roots", "phase_counts", "coverage_counts")}))


if __name__ == "__main__":
    main()
