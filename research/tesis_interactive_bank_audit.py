"""Audit provenance and data splits for the original and interactive thesis banks."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from research.src.artifacts import canonical, digest, file_hash, write_json
from research.src.data.semantic_partition_v3 import observable_episode_id_v3
from research.src.memory.executor import validate_record
from research.tesis_bank import audit_isolation
from research.tesis_interactive_env import BANK, DEFAULT_SEED, InteractiveEpisode, build_bank


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prior_provenance(memory, history):
    provenance = {m["turn_id"]: m["text"] for session in history for m in session["messages"]}
    for row in memory:
        validate_record(row, initial=True)
        _check(len({q["turn_id"] for q in row["evidence_quotes"]}) == len(row["source_turn_ids"]),
               "Memory record has incomplete evidence citations")
        for quote in row["evidence_quotes"]:
            _check(quote["text"] == provenance.get(quote["turn_id"]), "Memory quote differs from prior user text")
            _check(row["value"].casefold() in quote["text"].casefold(), "Prior evidence omits the archived choice")


def _core(family, query, memory, new_value):
    """A deliberately coarse overlap check ignoring dates, wording and later events."""
    attribute = query["attribute"]
    baseline = next(m["value"] for m in memory if m["attribute"] == attribute and m["scope"]["kind"] == "ongoing")
    return family, attribute, baseline, new_value


def audit(seed=DEFAULT_SEED, *, old_path=BANK):
    manifest_path = old_path / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    old, old_checks = {}, {}
    old_core = defaultdict(set)
    for split, specification in manifest["partitions"].items():
        path = old_path / specification["path"]
        _check(file_hash(path) == specification["sha256"], "Original partition differs from its frozen hash")
        old[split] = _rows(path)
        checks = Counter()
        for row in old[split]:
            _prior_provenance(row["initial_memory"], row["prior_sessions"])
            _check(row["split"] == split and row["root_id"] == "tb" + observable_episode_id_v3(row)[:20],
                   "The original public identity cannot be reproduced")
            checks.update(episodes=1, historical_records=len(row["initial_memory"]), public_identities_reproduced=1)
            old_core[_core(row["scenario_family"], row["current_query"], row["initial_memory"],
                           row["truth"].get("new_value") if row["scenario_family"] != "F3" else None)].add(split)
        old_checks[split] = dict(checks)
    old_isolation = audit_isolation(old)

    new = build_bank(seed=seed)
    old_roots = {row["root_id"] for rows in old.values() for row in rows}
    locations, core = defaultdict(set), defaultdict(set)
    groups, checks = defaultdict(list), defaultdict(Counter)
    public_fields = {"user_messages", "query", "memory", "reuse_regime", "visit", "questions_remaining", "asked_slots"}
    for split, rows in new.items():
        for story in rows:
            _prior_provenance(story["initial_memory"], story["prior_sessions"])
            _check(story["split"] == split and story["origin_root"] not in old_roots,
                   "Interactive split label changed or old root reused")
            first, second = story["visits"]
            facts = first["fact"]
            core[_core(story["scenario_family"], first["query"], story["initial_memory"],
                       facts["value"] if facts else None)].add(split)
            for name in ("root_id", "world_id", "origin_root", "recipe_id", "template"):
                locations[(name, story[name])].add(split)
            _check(first["query"]["session_id"] != second["query"]["session_id"], "The two visits share a session")
            _check(first["query"]["date"] <= second["query"]["date"] and all(
                p["query"]["date"] >= second["query"]["date"] for p in story["future_queries"]),
                "A reuse probe predates the archived conversation")
            counter = checks[split]
            counter.update(episodes=1, historical_records=len(story["initial_memory"]),
                           opening_turns=sum(len(v["messages"]) for v in story["visits"]))
            counter["same_day_different_session"] += first["query"]["date"] == second["query"]["date"]
            counter.update("first_visit_" + v["style"] for v in [first] if v["slot"])
            counter.update("second_visit_" + story["second_visit_kind"] for _ in [second])
            counter.update("probe_" + p["kind"] for p in story["future_queries"])
            env = InteractiveEpisode(story)
            try:
                observation = env.observe()
                _check(set(observation) == public_fields, "Native observation received private fields")
                groups[story["root_id"]].append((canonical(observation), env.legal_actions, story))
            finally:
                env.close()
    _check(all(len(parts) == 1 for parts in locations.values()), "Interactive partition identity crosses splits")
    for entries in groups.values():
        _check(len(entries) in (1, 2), "Unexpected worlds per interactive history")
        if len(entries) == 2:
            _check(len({prefix for prefix, _, _ in entries}) == len({mask for _, mask, _ in entries}) == 1,
                   "A paired world changed the pre-question public observation or action mask")
            worlds = [story for _, _, story in entries]
            _check(len({story["visits"][0]["answer"] for story in worlds}) == 2, "Paired replies collapsed")
            checks[worlds[0]["split"]]["paired_public_prefixes_verified"] += 1
    for split in new:
        checks[split]["roots"] = len({row["root_id"] for row in new[split]})
        checks[split]["bank_sha256"] = digest(new[split])
    coarse_cross_split = Counter(key[0] for key, splits in core.items() if len(splits) > 1)
    common_core = Counter(key[0] for key in old_core.keys() & core.keys())
    return {"status": "verified", "original": {"manifest_sha256": file_hash(manifest_path),
            "counts": old_checks, "isolation": old_isolation}, "interactive": {"seed": seed,
            "counts": checks, "interactive_roots_overlapping_original": 0,
            "recipe_world_and_template_overlap_between_splits": 0,
            "coarse_preference_overlap_between_splits_by_family": coarse_cross_split,
            "coarse_preference_overlap_with_original_by_family": common_core},
            "limits": ["A coarse key shares a food, attribute and family while ignoring wording, dates, context and later events.",
                       "Food and scope concepts are shared across splits. Identity checks do not prove unrestricted paraphrase generalization.",
                       "The audit checks evaluator-only fields and initial state identities. It does not evaluate or rank reserved policies.",
                       "Conversations are deterministic simulations with authored variation and fixed model-independent reply trajectories."]}


def compare_seeds(left_seed, right_seed):
    """Check authored underlying recipes plus exact public prefixes across two experiments."""
    _check(left_seed != right_seed, "Use two different seeds")
    fields = ("root_id", "world_id", "origin_root", "recipe_id", "normalized_public")

    def keys(seed):
        registries = {name: defaultdict(set) for name in fields}
        bank = build_bank(seed=seed)
        for split, stories in bank.items():
            for story in stories:
                first = story["visits"][0]
                equivalent = observable_episode_id_v3({"initial_memory": story["initial_memory"],
                    "user_messages": first["messages"], "current_query": first["query"],
                    "future_query": first["query"], "opportunity": {}, "replies": {}, "truth": {}})
                for name in fields:
                    registries[name][equivalent if name == "normalized_public" else story[name]].add(split)
        return registries

    left = keys(left_seed)
    right = keys(right_seed)
    overlap = {}
    for field in fields:
        shared = left[field].keys() & right[field].keys()
        splits = Counter(f"{a}:{b}" for identity in shared for a in left[field][identity] for b in right[field][identity])
        overlap[field] = {"count": len(shared), "split_pairs": dict(splits)}
    return {"compared_seeds": [left_seed, right_seed], "cross_experiment_identity_overlap": overlap,
            "scope": "Identity and authored recipe checks; food vocabulary and scenario concepts shared."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--compare-seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(seed=args.seed)
    if args.compare_seed is not None:
        result["comparison"] = compare_seeds(args.seed, args.compare_seed)
    write_json(args.output, result)
    print(canonical(result["interactive"]["counts"]))


if __name__ == "__main__":
    main()
