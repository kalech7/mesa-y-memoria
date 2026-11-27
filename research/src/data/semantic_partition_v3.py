"""Role-aware exact identity; missing role and explicit user are equivalent."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from .semantic_partition_v2 import semantic_episode_payload_v2, SemanticSplitLeakage

SEMANTIC_IDENTITY_VERSION = "behavioral_identity_v3_roles"


def semantic_episode_payload_v3(episode):
    normalized = deepcopy(episode)
    for message in normalized.get("user_messages", []):
        if message.get("role") == "user":
            message.pop("role")
    for reply in normalized.get("replies", {}).values():
        if isinstance(reply, dict) and reply.get("role") == "user":
            reply.pop("role")
    payload = semantic_episode_payload_v2(normalized)
    identifiers = {m["turn_id"]: p["turn_id"] for m, p in zip(normalized["user_messages"], payload["user_messages"])}
    def rename(value):
        if isinstance(value, list):
            return [rename(v) for v in value]
        if isinstance(value, dict):
            return {k: identifiers.get(v, v) if k == "in_reply_to" and isinstance(v, str) else rename(v)
                    for k, v in value.items()}
        return value
    return rename(payload)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def semantic_episode_id_v3(episode):
    return _digest(semantic_episode_payload_v3(episode))


def observable_episode_id_v3(episode):
    payload = semantic_episode_payload_v3(episode)
    return _digest({k: payload[k] for k in ("initial_memory", "user_messages", "current_query")})


def assert_semantic_split_isolation_v3(splits, *, historical_splits=None):
    registries = {"root": {}, "behavioral": {}, "observable": {}}
    counts, findings = {}, []
    for split, rows in splits.items():
        rows = list(rows)
        if not rows:
            raise ValueError("Partition must contain episodes: " + split)
        for episode in rows:
            rid = episode.get("root_id")
            if not isinstance(rid, str) or not rid:
                raise ValueError("Nonempty root_id required")
            keys = {"root": rid, "behavioral": semantic_episode_id_v3(episode), "observable": observable_episode_id_v3(episode)}
            for kind, key in keys.items():
                prior = registries[kind].get(key)
                current = {"split": split, "root_id": rid}
                if prior and prior["split"] != split:
                    findings.append({"kind": kind, "left": prior, "right": current, "identity": key})
                registries[kind].setdefault(key, current)
        counts[split] = {"episodes": len(rows), "roots": len({e["root_id"] for e in rows}),
            "behavioral_identities": len({semantic_episode_id_v3(e) for e in rows}),
            "observable_identities": len({observable_episode_id_v3(e) for e in rows})}
    for split, rows in (historical_splits or {}).items():
        for episode in rows:
            keys = {"root": episode["root_id"], "behavioral": semantic_episode_id_v3(episode), "observable": observable_episode_id_v3(episode)}
            for kind, key in keys.items():
                if key in registries[kind]:
                    findings.append({"kind": "historical_" + kind, "historical_split": split,
                        "historical_root_id": episode["root_id"], "current": registries[kind][key], "identity": key})
                    break
    if findings:
        raise SemanticSplitLeakage(findings)
    return {"status": "verified", "identity_version": SEMANTIC_IDENTITY_VERSION, "partitions": counts,
            "historical_overlap": 0, "limits": "Exact identity with alpha-renamed references and user-role normalization; not universal paraphrase equivalence."}
