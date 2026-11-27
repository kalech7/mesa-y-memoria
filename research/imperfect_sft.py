"""Auditable imperfect SFT demonstrations; truthful episodes/evaluator stay intact.

The configured rate is the proportion of distinct TRAIN roots receiving one
imperfect natural-route demonstration, not an expected accuracy or target-token
error rate. Four forced-route examples per episode remain clean. No model loads.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil

from .src.artifacts import canonical, digest, file_hash, write_json
from .src.policy.llm_controller import LLMController, validate_action

VERSION = "imperfect_sft_v1"


def _configuration(rate, seed):
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("Imperfect SFT rate must be finite and in [0,1]")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Imperfect SFT seed must be an integer")
    return float(rate), seed


def _rank(root_id, seed):
    return hashlib.sha256(f"{VERSION}:{seed}:{root_id}".encode()).hexdigest()


def _validate_examples(examples):
    if not examples or any(row.get("split") != "train" for row in examples):
        raise ValueError("Only nonempty TRAIN demonstrations may be loaded")
    for row in examples:
        action = validate_action(row["phase"], row["action"])
        if json.loads(row["completion"]) != action:
            raise ValueError("Completion/action mismatch in SFT demonstration")
        LLMController(None).prompt(row["phase"], row["observation"])


def _bad_write(row, requested):
    """Choose a public-evidence mistake, preserving executor-valid structure."""
    from .src.memory.executor import validate_record
    from .src.scoring.evidence import audit_operation
    original = row["action"]
    writes = [operation for operation in original["operations"] if operation["op"] != "NOOP"]
    if not writes:
        return None
    if requested == "temporary_as_permanent":
        for source in writes:
            if source["record"]["scope"]["kind"] == "ongoing":
                continue
            operation = deepcopy(source)
            operation["record"]["scope"] = {
                "kind": "ongoing", "valid_from": row["observation"]["query"]["date"],
                "valid_until_exclusive": None, "session_id": None, "context": None,
            }
            operation["record"]["status"] = "confirmed"
            if operation["op"] == "MARK_UNRESOLVED":
                operation["op"] = "ADD"
            validate_record(operation["record"])
            # A broader scope is used only when the unchanged evidence auditor
            # actually rejects it; some harmless rewrites would be equivalent.
            if not audit_operation(operation, row["observation"]):
                return {"operations": [operation]}, "temporary_as_permanent"
    return {"operations": [{"op": "NOOP"}]}, "omit_supported_update"


def prepare_teacher_examples(episodes, work_directory, *, rate=0., seed=7331):
    """Return seven examples per episode and a separate mutation audit.

    Input episodes are never altered. Assignment uses root identity and a seed,
    never latent truth, method, validation outcomes or policy scores. SELECT
    mistakes replay an actual environment route before downstream targets are
    generated. Unselected and forced-route rows are byte-equivalent as objects
    to the clean producer, and rate zero is exactly the historical producer.
    """
    from .competence_training import load_episodes, prepare_teacher_examples as clean_teacher
    from .competence_data import teacher_for_episode
    rate, seed = _configuration(rate, seed)
    episodes = load_episodes(episodes, "train")
    before = digest(episodes)
    roots = sorted({row["root_id"] for row in episodes}, key=lambda rid: (_rank(rid, seed), rid))
    selected = set(roots[:math.floor(rate * len(roots) + .5)])
    clean = clean_teacher(episodes, Path(work_directory) / "clean")
    examples = deepcopy(clean)
    mutations = []
    for index, episode in enumerate(episodes):
        root_id = episode["root_id"]
        if root_id not in selected:
            continue
        offset = index * 7
        original_rows = deepcopy(clean[offset:offset + 3])
        choice = int(_rank(root_id, seed)[:8], 16) % 3
        replacement = None
        if choice:
            candidate = _bad_write(original_rows[2], "temporary_as_permanent" if choice == 2 else "omit_supported_update")
            if candidate is not None:
                action, kind = candidate
                replacement = deepcopy(original_rows)
                replacement[2]["action"] = action
                replacement[2]["completion"] = json.dumps(action, ensure_ascii=False, sort_keys=True)
        if replacement is None:
            old_selection = original_rows[0]["action"]
            if old_selection["action"] == "ASK":
                action, route, kind = {"action": "PROCEED", "slot": None}, "PROCEED", "skip_clarification"
            else:
                action, route, kind = {"action": "ASK", "slot": "TEMPORAL_SCOPE"}, "ASK:TEMPORAL_SCOPE", "unnecessary_question"
            replay = teacher_for_episode(episode, condition="O2", route=route,
                work_directory=Path(work_directory) / "mutated" / f"episode_{index:06d}")
            replacement = []
            for old, replayed in zip(original_rows, replay):
                row = {**deepcopy(old), **deepcopy(replayed), "route": "natural",
                       "source": "observable_rule_teacher_v5_real_environment"}
                replacement.append(row)
            replacement[0]["action"] = action
            replacement[0]["completion"] = json.dumps(action, ensure_ascii=False, sort_keys=True)
        changed_targets = []
        for phase_index, (old, new) in enumerate(zip(original_rows, replacement)):
            if old["action"] != new["action"]:
                changed_targets.append(new["phase"])
            examples[offset + phase_index] = new
        if not changed_targets:
            raise ValueError("Selected root did not produce an imperfect target")
        mutations.append({"root_id": root_id, "latent_world_id": episode["latent_world_id"],
            "realization_id": episode["realization_id"], "episode_index": index, "kind": kind,
            "example_indices": list(range(offset, offset + 3)), "changed_target_phases": changed_targets,
            "clean": original_rows, "imperfect": deepcopy(replacement)})
    _validate_examples(examples)
    if digest(episodes) != before:
        raise RuntimeError("TRAIN episode truth changed during SFT preparation")
    changed_target_count = sum(a["action"] != b["action"] for a, b in zip(clean, examples))
    audit = {"version": VERSION, "rate": rate, "seed": seed, "rate_unit": "distinct_train_roots",
        "train_episode_sha256": before, "train_episode_sha256_after": digest(episodes),
        "root_count": len(roots), "selected_root_count": len(selected), "selected_roots": sorted(selected),
        "realized_root_fraction": len(selected) / len(roots), "examples": len(examples),
        "changed_target_count": changed_target_count, "changed_target_fraction": changed_target_count / len(examples),
        "mutation_counts": dict(Counter(row["kind"] for row in mutations)), "mutations": mutations,
        "clean_examples_sha256": digest(clean), "examples_sha256": digest(examples),
        "gold_modified": False, "evaluator_modified": False, "heldout_modified": False,
        "limitations": ["Synthetic controlled training mistakes; error frequencies are experimental settings, not estimates of human behavior.",
                       "An omitted write or unnecessary question need not reduce factual accuracy in every episode."]}
    return examples, audit


def _write_rows(path, rows):
    with Path(path).open("x") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")


def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prepare_bundle(train, output, *, rate, seed=7331, heldout_paths=None):
    """Create one new immutable arm directory; optionally copy heldout bytes."""
    from .competence_training import load_episodes
    rate, seed = _configuration(rate, seed)
    episodes = load_episodes(train, "train")
    heldout_paths = {role: Path(path).resolve() for role, path in (heldout_paths or {}).items()}
    if any(role not in {"validation", "evaluation", "test"} for role in heldout_paths):
        raise ValueError("Heldout roles must be validation/evaluation/test")
    train_roots = {row["root_id"] for row in episodes}
    partitions = {"train": episodes}
    for role, path in heldout_paths.items():
        rows = _read_rows(path)
        if not rows or any(row.get("split") != role for row in rows):
            raise ValueError("Heldout partition label mismatch: " + role)
        if train_roots & {row["root_id"] for row in rows}:
            raise ValueError("TRAIN/heldout root overlap")
        partitions[role] = rows
    from .src.data.semantic_partition_v3 import assert_semantic_split_isolation_v3
    isolation = assert_semantic_split_isolation_v3(partitions)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new imperfect SFT bundle destination")
    output.mkdir(parents=True)
    examples, audit = prepare_teacher_examples(episodes, output / "teacher_sqlite", rate=rate, seed=seed)
    # Recover the exact clean source for every mutation, retaining a full baseline.
    clean = deepcopy(examples)
    for mutation in audit["mutations"]:
        for index, row in zip(mutation["example_indices"], mutation["clean"]):
            clean[index] = row
    _write_rows(output / "examples.jsonl", examples)
    _write_rows(output / "clean_examples.jsonl", clean)
    write_json(output / "mutations.json", audit)
    files = {name: file_hash(output / name) for name in ("examples.jsonl", "clean_examples.jsonl", "mutations.json")}
    heldout = {}
    for role, source in heldout_paths.items():
        destination = output / (role + ".jsonl")
        before = file_hash(source)
        shutil.copyfile(source, destination)
        if file_hash(source) != before or file_hash(destination) != before:
            raise RuntimeError("Heldout bytes changed")
        files[destination.name] = before
        heldout[role] = {"source": str(source), "path": destination.name, "sha256": before}
    manifest = {"version": VERSION, "research_status": "exploratory", "rate": rate, "seed": seed,
        "rate_unit": "distinct_train_roots", "train_episode_sha256": audit["train_episode_sha256"],
        "files": files, "heldout": heldout, "root_count": audit["root_count"],
        "selected_root_count": audit["selected_root_count"], "changed_target_count": audit["changed_target_count"],
        "source_sha256": file_hash(__file__), "isolation": isolation, "training_steps": 0, "model_calls": 0}
    write_json(output / "manifest.json", manifest)
    return manifest


def load_examples(bundle_path, *, train=None, expected_rate=None, expected_seed=None):
    """Verify the frozen bundle before returning targets to the SFT optimizer."""
    from .competence_training import load_episodes
    path = Path(bundle_path)
    manifest = json.loads((path / "manifest.json").read_text())
    rate, seed = _configuration(manifest["rate"], manifest["seed"])
    if manifest.get("version") != VERSION or manifest.get("rate_unit") != "distinct_train_roots":
        raise ValueError("Unknown imperfect SFT bundle contract")
    if expected_rate is not None and _configuration(expected_rate, seed)[0] != rate:
        raise ValueError("Imperfect SFT rate mismatch")
    if expected_seed is not None and _configuration(rate, expected_seed)[1] != seed:
        raise ValueError("Imperfect SFT seed mismatch")
    if manifest.get("source_sha256") != file_hash(__file__):
        raise ValueError("Imperfect SFT producer source changed")
    if not {"examples.jsonl", "clean_examples.jsonl", "mutations.json"} <= manifest["files"].keys():
        raise ValueError("Incomplete imperfect SFT bundle")
    for name, expected in manifest["files"].items():
        if Path(name).is_absolute() or len(Path(name).parts) != 1 or file_hash(path / name) != expected:
            raise ValueError("Imperfect SFT artifact hash mismatch: " + name)
    if train is not None and manifest["train_episode_sha256"] != digest(load_episodes(train, "train")):
        raise ValueError("Imperfect SFT TRAIN episode digest mismatch")
    examples = _read_rows(path / "examples.jsonl")
    clean = _read_rows(path / "clean_examples.jsonl")
    audit = json.loads((path / "mutations.json").read_text())
    _validate_examples(examples)
    _validate_examples(clean)
    if (audit["train_episode_sha256"] != manifest["train_episode_sha256"] or audit["rate"] != rate
            or audit["seed"] != seed or digest(examples) != audit["examples_sha256"]
            or digest(clean) != audit["clean_examples_sha256"]):
        raise ValueError("Imperfect SFT audit mismatch")
    return examples, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Create auditable arms without training")
    prepare.add_argument("--train", required=True)
    prepare.add_argument("--validation")
    prepare.add_argument("--evaluation")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--rates", type=float, nargs="+", default=[0., .1, .3])
    prepare.add_argument("--seed", type=int, default=7331)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("bundle")
    args = parser.parse_args(argv)
    if args.command == "inspect":
        examples, audit = load_examples(args.bundle)
        print(canonical({key: value for key, value in audit.items() if key != "mutations"}))
        return 0
    rates = [_configuration(rate, args.seed)[0] for rate in args.rates]
    if len(set(rates)) != len(rates):
        raise ValueError("Duplicate imperfect SFT rates")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Choose a new experiment destination")
    heldout = {role: getattr(args, role) for role in ("validation", "evaluation") if getattr(args, role)}
    for rate in rates:
        result = prepare_bundle(args.train, output / f"rate_{format(rate, '.12g').replace('.', '_')}",
                                rate=rate, seed=args.seed, heldout_paths=heldout)
        print(canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
