"""Root-level leakage audits and explicit, independently supplied human review."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
import random


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def canonical_signature(episode):
    """Ignore roots, food lexemes, absolute dates and surface wording.

    Repeated mechanism signatures are reported, not silently equated to new
    structural generalization. Root disjointness alone does not remove them.
    """
    start = date.fromisoformat(episode["current_query"]["date"])
    values, contexts = {}, {}
    def ordinal(mapping, key):
        if key is None:
            return None
        if key not in mapping:
            mapping[key] = len(mapping)
        return mapping[key]
    records = []
    for record in sorted(episode["initial_memory"], key=lambda r: (r.get("recorded_at", ""), r["attribute"])):
        sc = record["scope"]
        relative = lambda value: (date.fromisoformat(value) - start).days if value else None
        records.append({"attribute": record["attribute"], "value_position": ordinal(values, record["value"]),
                        "scope_kind": sc["kind"], "context_position": ordinal(contexts, sc.get("context")),
                        "from_offset": relative(sc.get("valid_from")),
                        "until_offset": relative(sc.get("valid_until_exclusive")),
                        "supersedes_count": len(record.get("supersedes", []))})
    truth = episode.get("truth", {})
    return digest({"records": records, "current_context": ordinal(contexts, episode["current_query"].get("context")),
                   "family": episode.get("scenario_family"), "scope_kind": truth.get("scope_kind"),
                   "current_equals_future": truth.get("current_value") == truth.get("future_value"),
                   "question_variable": truth.get("decisive_slot")})


def actor_prefix(episode):
    """Canonical prior-to-question public bytes, with no experimental IDs."""
    return json.dumps({"memory": episode["initial_memory"], "user_messages": episode["user_messages"],
                       "query": episode["current_query"],
                       "reuse_regime": "later_same_attribute_without_questions"},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def audit_splits(splits):
    """Validate root ownership and content before certifying partition counts."""
    if isinstance(splits, list):
        grouped = defaultdict(list)
        for episode in splits:
            grouped[episode["split"]].append(episode)
        splits = dict(grouped)
    owner, roots, signatures, counts = {}, {}, {}, {}
    for split, episodes in splits.items():
        if any(e["split"] != split for e in episodes):
            raise ValueError("Split label differs from containing partition")
        roots[split] = set()
        family_by_root = {}
        signatures[split] = Counter()
        paired = defaultdict(list)
        for episode in episodes:
            rid = episode["root_id"]
            if rid in owner and owner[rid] != split:
                raise ValueError(f"ROOT_SPLIT_LEAKAGE: {rid}")
            owner[rid] = split
            roots[split].add(rid)
            family_by_root[rid] = episode["scenario_family"]
            signatures[split][canonical_signature(episode)] += 1
            paired[(rid, episode["realization_id"])].append(episode)
            if not 4 <= len(episode["initial_memory"]) <= 8:
                raise ValueError("Narrative must have four to eight records")
        pair_count = 0
        for pair in paired.values():
            f4 = [e for e in pair if e["scenario_family"] == "F4"
                  and e.get("opportunity", {}).get("eligible", True)]
            if len(f4) > 1:
                if len({actor_prefix(e) for e in f4}) != 1:
                    raise ValueError("COUNTERFACTUAL_PREFIX_LEAKAGE")
                if len({e["replies"]["TEMPORAL_SCOPE"]["text"] for e in f4}) < 2:
                    raise ValueError("Decisive question must distinguish counterfactual worlds")
                pair_count += 1
        counts[split] = {"roots": len(roots[split]), "episodes": len(episodes),
                         "families": dict(Counter(family_by_root.values())), "f4_pairs": pair_count,
                         "canonical_signatures": dict(signatures[split]), "sha256": digest(episodes)}
    names = sorted(splits)
    overlaps = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            shared = set(signatures[left]) & set(signatures[right])
            overlaps[left + ":" + right] = {
                "root_overlap": sorted(roots[left] & roots[right]),
                "canonical_overlap_count": len(shared),
                "right_episodes_seen_canonical": sum(signatures[right][s] for s in shared),
                "interpretation": "Shared synthetic mechanisms; this is not independent structural OOD."}
    from .semantic_partition_v2 import assert_semantic_split_isolation_v2
    semantic_isolation = assert_semantic_split_isolation_v2(splits)
    return {"status": "passed", "partitions": counts, "overlap": overlaps,
            "semantic_isolation": semantic_isolation,
            "human_audit_status": "pending_human_audit", "external_authorship": "not_supplied"}


def audit_human_reviews(reviews, adjudications=()):
    """Validate actual independent review records; empty input never passes G2.

    Each entry: root_id, reviewer_id, family, information_sufficient (bool),
    decisive_slot (str/null), admissible_answers, writable_scope, f4_both_plausible.
    An adjudication: root_id, resolved (bool), rationale (nonempty str).
    """
    if not reviews:
        return {"status": "pending_human_audit", "roots": 0, "reason": "No human reviews supplied"}
    groups = defaultdict(list)
    for review in reviews:
        required = {"root_id", "reviewer_id", "family", "information_sufficient", "decisive_slot",
                    "admissible_answers", "writable_scope"}
        if not required <= review.keys() or type(review["information_sufficient"]) is not bool:
            raise ValueError("Malformed human review record")
        groups[review["root_id"]].append(review)
    if any(len(pair) != 2 or pair[0]["reviewer_id"] == pair[1]["reviewer_id"] for pair in groups.values()):
        return {"status": "pending_human_audit", "roots": len(groups), "reason": "Exactly two independent reviews required per root"}
    families = Counter(pair[0]["family"] for pair in groups.values())
    agreement = []
    mismatches = []
    for rid, pair in groups.items():
        equal = all(pair[0][k] == pair[1][k] for k in ("information_sufficient", "decisive_slot"))
        agreement.append(equal)
        if not equal or any(pair[0][k] != pair[1][k] for k in ("admissible_answers", "writable_scope")):
            mismatches.append(rid)
    adjudicated = {a["root_id"] for a in adjudications if a.get("resolved") is True and a.get("rationale", "").strip()}
    unresolved = sorted(set(mismatches) - adjudicated)
    sufficient = len(groups) >= 100 and all(families.get(f, 0) >= 20 for f in ("F0", "F1", "F2", "F3", "F4"))
    f4_valid = all(all(r.get("f4_both_plausible") is True for r in pair)
                   for pair in groups.values() if pair[0]["family"] == "F4")
    rate = sum(agreement) / len(agreement)
    # Cohen's kappa for the exact pair (sufficiency, decisive variable).
    labels_a = Counter((p[0]["information_sufficient"], p[0]["decisive_slot"]) for p in groups.values())
    labels_b = Counter((p[1]["information_sufficient"], p[1]["decisive_slot"]) for p in groups.values())
    chance = sum(labels_a[k] * labels_b[k] for k in labels_a.keys() | labels_b.keys()) / len(groups) ** 2
    kappa = (rate - chance) / (1 - chance) if chance < 1 else None
    passed = sufficient and f4_valid and rate >= .9 and not unresolved
    return {"status": "passed" if passed else "pending_human_audit", "roots": len(groups),
            "families": dict(families), "initial_exact_agreement": rate, "cohens_kappa": kappa,
            "unresolved_roots": unresolved, "f4_plausibility_passed": f4_valid,
            "reviews_sha256": digest(reviews), "adjudications_sha256": digest(adjudications)}


def prepare_human_audit(episodes, *, roots=100, seed=4107):
    """Prepare two blind packets and a separate private adjudication key.

    Samples only development/training/validation roots, twenty per family for
    the primary 100-root audit. F4 roots must provide both plausible latent-world
    reply candidates. Returned packets contain no expected answer/family label.
    This creates review material, never reviewer decisions or approval.
    """
    if roots < 100 or roots % 5:
        raise ValueError("Human audit needs at least 100 balanced roots")
    groups = defaultdict(list)
    for episode in episodes:
        if episode["split"] in {"dev_smoke", "train", "validation"}:
            groups[episode["root_id"]].append(episode)
    by_family = defaultdict(list)
    for rid, variants in groups.items():
        family = variants[0]["scenario_family"]
        if family == "F4":
            variants = [v for v in variants if v.get("opportunity", {}).get("eligible", True)]
            if len({v["latent_world_id"] for v in variants}) < 2 or not all(
                    v["truth"].get("decisive_slot") == "TEMPORAL_SCOPE" for v in variants):
                continue
            groups[rid] = variants
        by_family[family].append(rid)
    quota = roots // 5
    if any(len(by_family[f]) < quota for f in ("F0", "F1", "F2", "F3", "F4")):
        return {"status": "pending_human_audit", "reason": "Insufficient eligible development roots",
                "available_by_family": {f: len(by_family[f]) for f in ("F0", "F1", "F2", "F3", "F4")},
                "reviewer_packets": {}, "private_key": {}}
    rng = random.Random(seed)
    selected = [rid for f in ("F0", "F1", "F2", "F3", "F4")
                for rid in rng.sample(sorted(by_family[f]), quota)]
    rng.shuffle(selected)
    packets, private = [], {}
    for index, rid in enumerate(selected):
        variants = sorted(groups[rid], key=lambda e: (e["realization_id"], e["latent_world_id"]))
        first = variants[0]
        realization = first["realization_id"]
        variants = [e for e in variants if e["realization_id"] == realization]
        blind_id = "audit_" + digest([seed, index, rid])[:12]
        candidate_replies = list({e["replies"]["TEMPORAL_SCOPE"]["text"] for e in variants}) if first["scenario_family"] == "F4" else []
        rng.shuffle(candidate_replies)
        packets.append({"audit_id": blind_id, "initial_memory": first["initial_memory"],
                        "user_messages": first["user_messages"], "current_query": first["current_query"],
                        "candidate_scope_replies": candidate_replies,
                        "review_fields": ["information_sufficient", "decisive_slot", "admissible_answers",
                                          "writable_scope", "f4_both_plausible", "rationale"]})
        private[blind_id] = {"root_id": rid, "family": first["scenario_family"]}
    second_order = list(packets)
    rng.shuffle(second_order)
    return {"status": "pending_human_audit", "roots": roots,
            "reviewer_packets": {"reviewer_a": packets, "reviewer_b": second_order},
            "private_key": private, "seed": seed,
            "instructions": "Review independently without generator labels; identify admissible answers and writable scope. Preserve original decisions before adjudication."}
