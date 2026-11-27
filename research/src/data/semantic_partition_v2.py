"""Prospective partitions by behavioral identity, preserving legacy bank files.

This identity is deliberately narrower than general semantic equivalence: only
IDs are alpha-renamed. Text, food values, dates, scopes, replies and golds remain
unchanged. It detects the exact content leakage found in the historical bank;
it does not certify linguistic independence or structural generalization.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import random

SEMANTIC_IDENTITY_VERSION = "behavioral_identity_v2"
BEHAVIOR_FIELDS = ("initial_memory", "user_messages", "current_query", "future_query", "replies", "truth", "opportunity")


class SemanticSplitLeakage(ValueError):
    def __init__(self, findings):
        self.findings = findings
        super().__init__("SEMANTIC_SPLIT_LEAKAGE: " + json.dumps(findings[:3], ensure_ascii=False, sort_keys=True))


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def semantic_episode_payload_v2(episode):
    """Preserve all behavior fields and replace only identifiers/references.

    Record/message order remains significant. Unknown source references remain
    unchanged; they are never silently repaired. Root, split, world and author
    metadata are intentionally outside the behavioral identity.
    """
    missing = set(BEHAVIOR_FIELDS) - episode.keys()
    if missing:
        raise ValueError("Incomplete behavioral episode: " + ",".join(sorted(missing)))
    identifiers = {}
    for index, record in enumerate(episode["initial_memory"]):
        identifiers[record["memory_id"]] = f"memory:{index}"
        for j, reference in enumerate(record.get("source_turn_ids", [])):
            identifiers.setdefault(reference, f"source:{index}:{j}")
        for j, quote in enumerate(record.get("evidence_quotes", [])):
            identifiers.setdefault(quote["turn_id"], f"source:{index}:{j}")
    for index, message in enumerate(episode["user_messages"]):
        identifiers.setdefault(message["turn_id"], f"user:{index}")
    for slot, reply in sorted(episode["replies"].items()):
        if isinstance(reply, dict) and reply.get("turn_id"):
            # Reply tables use one alternative turn for the single ASK.
            identifiers.setdefault(reply["turn_id"], "reply:" + slot)

    reference_keys = {"memory_id", "turn_id", "source_ref", "source_id", "source_turn_ids", "evidence_ids", "target_ids", "supersedes"}

    def rename(value, key=None):
        if isinstance(value, str):
            return identifiers.get(value, value) if key in reference_keys else value
        if isinstance(value, list):
            return [rename(v, key) for v in value]
        if isinstance(value, dict):
            return {k: rename(v, k) for k, v in value.items()}
        return value
    return rename({key: deepcopy(episode[key]) for key in BEHAVIOR_FIELDS})


def semantic_episode_id_v2(episode):
    return hashlib.sha256(_canonical(semantic_episode_payload_v2(episode)).encode()).hexdigest()


def observable_episode_id_v2(episode):
    """Link compatible worlds even if they carry different replies or golds."""
    payload = semantic_episode_payload_v2(episode)
    public = {key: payload[key] for key in ("initial_memory", "user_messages", "current_query")}
    return hashlib.sha256(_canonical(public).encode()).hexdigest()


def assert_semantic_split_isolation_v2(splits, *, historical_splits=None):
    """Reject root or behavioral identity crossing partitions, before training.

    ``splits`` maps partition names to loaded episodes. Historical comparisons
    are separate namespaces: any new root/content overlap with them is rejected.
    No exception for renamed root IDs or for a particular training method.
    """
    splits = {name: list(rows) for name, rows in splits.items()}
    historical_splits = {name: list(rows) for name, rows in (historical_splits or {}).items()}
    if any("role" in message for rows in [*splits.values(), *historical_splits.values()]
           for episode in rows for message in episode.get("user_messages", [])):
        from .semantic_partition_v3 import assert_semantic_split_isolation_v3
        return assert_semantic_split_isolation_v3(splits, historical_splits=historical_splits)
    owners, observable_owners, roots, counts, findings = {}, {}, {}, {}, []
    current_roots, current_signatures, current_observables = set(), set(), set()
    for split, episodes in splits.items():
        episodes = list(episodes)
        if not episodes:
            raise ValueError("Partition must contain episodes: " + str(split))
        for episode in episodes:
            rid = episode.get("root_id")
            if not isinstance(rid, str) or not rid:
                raise ValueError("Nonempty root_id required before partition validation")
            signature = semantic_episode_id_v2(episode)
            observable = observable_episode_id_v2(episode)
            for kind, key, registry in (("root", rid, roots), ("behavioral", signature, owners), ("observable", observable, observable_owners)):
                if key in registry and registry[key]["split"] != split:
                    findings.append({"kind": kind, "left": registry[key],
                                     "right": {"split": split, "root_id": rid}, "identity": key})
                registry.setdefault(key, {"split": split, "root_id": rid})
            current_roots.add(rid)
            current_signatures.add(signature)
            current_observables.add(observable)
        counts[split] = {"episodes": len(episodes), "roots": len({e["root_id"] for e in episodes}),
                         "behavioral_identities": len({semantic_episode_id_v2(e) for e in episodes}),
                         "observable_identities": len({observable_episode_id_v2(e) for e in episodes})}
    for split, episodes in (historical_splits or {}).items():
        for episode in episodes:
            rid, signature = episode["root_id"], semantic_episode_id_v2(episode)
            observable = observable_episode_id_v2(episode)
            if rid in current_roots or signature in current_signatures or observable in current_observables:
                kind = "historical_root" if rid in current_roots else "historical_behavioral" if signature in current_signatures else "historical_observable"
                findings.append({"kind": kind,
                                 "historical_split": split, "historical_root_id": rid,
                                 "current": owners.get(signature, observable_owners.get(observable, roots.get(rid))),
                                 "identity": observable if kind == "historical_observable" else signature})
    if findings:
        raise SemanticSplitLeakage(findings)
    return {"status": "verified", "identity_version": SEMANTIC_IDENTITY_VERSION,
            "partitions": counts, "historical_overlap": 0,
            "limits": "Exact behavior after ID renaming; no claim of paraphrase/structural independence."}


def partition_roots_v2(episodes, fractions, *, seed=4108):
    """Assign complete equivalence components before writing split labels.

    Roots sharing even one world/realization are unioned transitively. Fractions
    are targets, not permission to break a component to meet an exact quota.
    Seed affects assignment/order only; it never generates new content or IDs.
    """
    if not fractions or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in fractions.values()):
        raise ValueError("Positive partition fractions required")
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["root_id"]].append(deepcopy(episode))
    if not groups:
        raise ValueError("Cannot partition empty candidate roots")
    identity, observable_identity = semantic_episode_id_v2, observable_episode_id_v2
    if any("role" in message for variants in groups.values() for episode in variants
           for message in episode.get("user_messages", [])):
        from .semantic_partition_v3 import semantic_episode_id_v3, observable_episode_id_v3
        identity, observable_identity = semantic_episode_id_v3, observable_episode_id_v3
    parent = {rid: rid for rid in groups}

    def find(root):
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        return root

    owner = {}
    for rid, variants in groups.items():
        for episode in variants:
            for signature in (("behavioral", identity(episode)), ("observable", observable_identity(episode))):
                if signature in owner:
                    a, b = find(rid), find(owner[signature])
                    parent[a] = b
                else:
                    owner[signature] = rid
    components = defaultdict(list)
    for rid in groups:
        components[find(rid)].append(rid)
    # Content-only sorting makes input order and arbitrary root names immaterial.
    ordered = sorted(components.values(), key=lambda rs: _canonical(sorted({identity(e) for r in rs for e in groups[r]})))
    random.Random(seed).shuffle(ordered)
    names = sorted(fractions)
    total = sum(fractions.values())
    targets = {name: len(groups) * fractions[name] / total for name in names}
    used = {name: 0 for name in names}
    result = {name: [] for name in names}
    for component in ordered:
        destination = min(names, key=lambda name: (used[name] / targets[name], name))
        for rid in sorted(component):
            for episode in groups[rid]:
                episode["split"] = destination
                result[destination].append(episode)
        used[destination] += len(component)
    # Components can make a tiny target empty. Do not fabricate extra roots.
    result = {name: rows for name, rows in result.items() if rows}
    assert_semantic_split_isolation_v2(result)
    return result


def legacy_build_bank_v1(*args, **kwargs):
    """Explicit reproduction entrypoint: delegate unchanged historical builder."""
    from .build_bank import build_bank
    return build_bank(*args, **kwargs)
