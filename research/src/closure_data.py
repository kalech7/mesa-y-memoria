"""Development-only packets and fail-closed audits for a V4 protocol lock.

No new held-out bank is generated here. Template labels derived from historical
surface patterns and mechanism signatures are explicitly approximations, not an
independent authoring claim or human annotation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import re

from .artifacts import canonical, digest, file_hash, write_json
from .data.audit_splits import actor_prefix, canonical_signature

REQUIRED_METADATA = ("root_id", "template_id", "composition_id", "generator_version", "domain", "observable_state_id")
CLAIM_IDS = ("PI1_v4", "PI2_v4", "PI3a_v4", "PI3b_v4", "PI3c_v4",
             "PI3a_current_error", "PI3a_unsupported_write_episode", "PI3a_scope_error",
             "PI3b_current_error", "PI3b_unsupported_write_episode", "PI3b_scope_error")
PARTITIONS = {"train_bank": {"train"}, "validation_bank": {"validation"},
              "test_id_bank": {"test_id"}, "test_composition_bank": {"test_composition", "test_ood"}}


def _read_bank(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def historical_bank_paths(parent_run):
    parent = Path(parent_run).resolve()
    context = parent / "context.json"
    if not context.exists():
        return {}
    paths = json.loads(context.read_text()).get("paths", {})
    result = {}
    for split, name in paths.items():
        recorded = Path(name)
        candidates = [recorded if recorded.is_absolute() else parent / recorded,
                      parent.parents[1] / "data" / recorded.parent.name / recorded.name]
        result[split] = next((p.resolve() for p in candidates if p.is_file()), candidates[0])
    return result


def _template_pattern(episode):
    values = {str(r.get("value", "")) for r in episode.get("initial_memory", [])}
    truth = episode.get("truth", {})
    values.update(str(truth[k]) for k in ("current_value", "future_value", "initial_value", "new_value") if truth.get(k) is not None)
    def normalize(text):
        text = str(text).casefold()
        for value in sorted(values, key=len, reverse=True):
            if value:
                text = re.sub(r"(?<!\w)" + re.escape(value.casefold()) + r"(?!\w)", "{value}", text)
        return re.sub(r"\s+", " ", re.sub(r"\d{4}-\d{2}-\d{2}", "{date}", text)).strip()
    return digest({"memory_texts": [normalize(q.get("text", "")) for r in episode.get("initial_memory", [])
                                      for q in r.get("evidence_quotes", [])],
                   "user_messages": [normalize(m.get("text", "")) for m in episode.get("user_messages", [])]})


def _instance_signature(episode):
    records = [{"attribute": r.get("attribute"), "value": r.get("value"), "scope": r.get("scope"),
                "status": r.get("status"), "recorded_at": r.get("recorded_at"),
                "evidence_texts": [q.get("text") for q in r.get("evidence_quotes", [])]}
               for r in episode.get("initial_memory", [])]
    return digest({"memory": records, "messages": [m.get("text") for m in episode.get("user_messages", [])],
                   "current_query": episode.get("current_query"), "future_query": episode.get("future_query"),
                   "truth": episode.get("truth")})


def _annotate(episode):
    result = deepcopy(episode)
    result.setdefault("template_id", "derived_surface_" + _template_pattern(episode))
    result.setdefault("composition_id", "derived_mechanism_" + canonical_signature(episode))
    result.setdefault("generator_version", episode.get("authoring", {}).get("source", "unrecorded_historical_generator"))
    result.setdefault("domain", "non_clinical_food_preferences")
    result.setdefault("observable_state_id", digest(actor_prefix(episode).decode()))
    result["closure_metadata_provenance"] = {
        "template_id": "Derived from visible text after substituting observed lexical values and dates",
        "composition_id": "Derived from existing canonical_signature; restricted mechanism features",
        "generator_version": "Historical authoring.source label, not runtime generator attestation",
        "domain": "Existing protocol non_clinical_food_preferences",
        "observable_state_id": "SHA256 canonical existing actor_prefix, excludes hidden truth",
        "limitation": "Derived IDs do not establish independent templates, semantic equivalence or structural diversity."}
    return result


def prepare_development(parent_run, output, roots=50):
    if roots < 5 or roots % 5:
        raise ValueError("Development roots must be a positive multiple of five")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "DEVELOPMENT_V4.jsonl"
    if destination.exists():
        raise FileExistsError("Development packet already exists")
    groups, sources = defaultdict(list), []
    paths = historical_bank_paths(parent_run)
    # Validation first is a fixed rule. All worlds and realizations of a chosen
    # root are retained; no test episode is used as new development material.
    for split in ("validation", "train", "dev_smoke"):
        path = paths.get(split)
        if path is None or not path.is_file():
            continue
        sources.append({"split": split, "path": str(path), "sha256": file_hash(path)})
        for episode in _read_bank(path):
            if episode.get("split") not in {"validation", "train", "dev_smoke"}:
                raise ValueError("Historical development source contains a test split")
            groups[episode["root_id"]].append(episode)
    by_family = defaultdict(list)
    for rid, variants in groups.items():
        if len({e["scenario_family"] for e in variants}) != 1:
            raise ValueError("Family changes within a root")
        # Prefer validation roots, then train, then smoke; independent of score.
        by_family[variants[0]["scenario_family"]].append(rid)
    quota = roots // 5
    priority = {"validation": 0, "train": 1, "dev_smoke": 2}
    selected = []
    for family in ("F0", "F1", "F2", "F3", "F4"):
        candidates = sorted(by_family[family], key=lambda rid: (priority[groups[rid][0]["split"]], rid))
        if len(candidates) < quota:
            result = {"status": "blocked_missing_evidence", "reason": "Insufficient balanced historical development roots",
                      "available_by_family": {k: len(v) for k, v in by_family.items()}, "sources": sources}
            write_json(output / "DEVELOPMENT_MANIFEST_V4.json", result)
            return result
        selected.extend(candidates[:quota])
    episodes = [_annotate(e) for rid in selected for e in sorted(groups[rid], key=lambda e: (e["latent_world_id"], e["realization_id"]))]
    destination.write_text("".join(canonical(e) + "\n" for e in episodes))
    human_roots = []
    # Six roots per family, or all available if the packet has fewer than thirty.
    for family in ("F0", "F1", "F2", "F3", "F4"):
        human_roots.extend([rid for rid in selected if groups[rid][0]["scenario_family"] == family][:6])
    private, blind_rows = {}, []
    for rid in human_roots:
        blind_id = "review_" + digest(["closure_v4", rid])[:16]
        variants = groups[rid]
        private[blind_id] = {"root_id": rid, "scenario_family": variants[0]["scenario_family"],
                             "annotations": [{"latent_world_id": e["latent_world_id"], "realization_id": e["realization_id"],
                                              "truth": e.get("truth"), "opportunity": e.get("opportunity")} for e in variants]}
        visible = [{"variant_id": "v" + str(index), "initial_memory": e["initial_memory"],
                    "user_messages": e["user_messages"], "current_query": e["current_query"],
                    "future_query": e["future_query"], "simulator_replies_by_slot": e.get("replies", {})}
                   for index, e in enumerate(variants)]
        blind_rows.append({"blind_id": blind_id, "visible_variants": canonical(visible)})
    review_fields = ["blind_id", "visible_variants", "reviewer_id", "need_now", "need_future", "valid_slots",
                     "simulator_reply_faithful", "writable_scope", "rationale", "adjudication"]
    with (output / "HUMAN_REVIEW_V4.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        writer.writerows(sorted(blind_rows, key=lambda r: r["blind_id"]))
    write_json(output / "private/HUMAN_REVIEW_KEY_V4.json", private)
    (output / "private/HUMAN_REVIEW_KEY_V4.json").chmod(0o600)
    result = {"status": "passed", "roots": len(selected), "episodes": len(episodes),
              "roots_by_family": dict(Counter(groups[rid][0]["scenario_family"] for rid in selected)),
              "selected_root_ids": selected, "sources": sources, "development_bank": str(destination),
              "development_bank_sha256": file_hash(destination), "human_review_roots": len(human_roots),
              "human_audit": "not_executed", "human_review_status": "prepared_only",
              "limitations": ["Historical development data; no independent test generated or opened.",
                              "Template/composition IDs are documented approximations, not new structural diversity.",
                              "Blank reviewer fields are not human review or approval."]}
    write_json(output / "DEVELOPMENT_MANIFEST_V4.json", result)
    return result


def _positive_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate_protocol_plan(plan, historical_root=None):
    """Validate precision first, then inspect explicitly supplied banks.

    Protocol approval is evidence based; a new RNG seed is never accepted as a
    composition holdout. This is a finite feature audit, not a semantic oracle.
    """
    errors, hashes, audit = [], {}, {"test_banks_opened": False, "artifacts_by_role": {},
        "precision_temporal_order": "not_identifiable_from_current_artifacts",
        "confirmatory_temporal_provenance": False,
        "temporal_limitation": "Artifact timestamps and plan declarations alone do not prove precision was fixed before test creation/opening; lock remains exploratory without independent provenance."}
    precision = plan.get("precision_plan", {})
    if not isinstance(precision, dict):
        precision = {}
    required_precision = ("minimum_relevant_effect", "target_half_width", "planned_test_id_roots", "planned_test_composition_roots")
    if precision.get("source") != "development" or not precision.get("method"):
        errors.append("precision_plan must document a development-only precision method")
    for name in required_precision:
        if not _positive_number(precision.get(name)):
            errors.append("precision_plan missing positive " + name)
    artifact = Path(str(precision.get("artifact_path", "")))
    if not artifact.is_file():
        errors.append("precision_plan requires a real development precision artifact_path")
    else:
        hashes[str(artifact.resolve())] = file_hash(artifact)
        audit["artifacts_by_role"]["precision_artifact"] = str(artifact.resolve())
        try:
            evidence = json.loads(artifact.read_text())
            if not isinstance(evidence, dict) or evidence.get("source") != "development" or evidence.get("status") not in {"executed", "passed"}:
                errors.append("precision artifact must identify executed development-only estimation")
            if not _positive_number(evidence.get("estimated_half_width")):
                errors.append("precision artifact lacks estimated_half_width")
        except (ValueError, OSError):
            errors.append("precision artifact is not a readable JSON evidence object")
    claims = plan.get("claims", [])
    claim_ids = [c.get("id") if isinstance(c, dict) else c for c in claims] if isinstance(claims, list) else []
    if len(claim_ids) != 11 or set(str(x) for x in claim_ids) != set(CLAIM_IDS):
        errors.append("Exactly the five planned contrast IDs and six planned guard IDs are required: " + ",".join(CLAIM_IDS))
    if plan.get("training_seeds") != [101, 202, 303]:
        errors.append("training_seeds must be [101, 202, 303]")
    confidence = plan.get("individual_confidence")
    if plan.get("family_alpha") != .05 or not _positive_number(confidence) or not math.isclose(confidence, 1 - .05 / 11, abs_tol=1e-12):
        errors.append("Confidence must be 1 - 0.05/11 for the eleven-claim family")
    if not _positive_number(plan.get("harm_margin")):
        errors.append("A positive harm_margin must be fixed before test")
    elif plan["harm_margin"] >= 1:
        errors.append("harm_margin must be smaller than one for proportion guards")
    selection = plan.get("selection_validity_guards", {})
    if not isinstance(selection, dict) or not _positive_number(selection.get("utility_tolerance")):
        errors.append("selection_validity_guards requires a positive numerical utility_tolerance")
    guards = selection.get("validity_guards", {}) if isinstance(selection, dict) else {}
    if not isinstance(guards, dict) or not {"invalid_output_rate", "unsupported_writes"} <= guards.keys():
        errors.append("selection_validity_guards must declare invalid_output_rate and unsupported_writes limits")
    elif any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in guards.values()):
        errors.append("selection validity limits must be finite proportions in [0,1]")
    budgets = plan.get("budgets", {})
    budgets = budgets if isinstance(budgets, dict) else {}
    for name in ("training_seconds_per_arm_seed", "validation_seconds_per_arm_seed", "save_seconds_per_arm_seed", "max_new_llm_calls"):
        if not _positive_number(budgets.get(name)):
            errors.append("budgets missing positive " + name)
    if errors:
        return {"status": "failed", "errors": errors, "artifacts_sha256": hashes, "audit": audit}
    banks, roots, signatures, templates, compositions, instances = {}, {}, {}, {}, {}, {}
    for key, valid_splits in PARTITIONS.items():
        path = Path(str(plan.get(key, "")))
        if not path.is_file():
            errors.append(key + " does not exist")
            continue
        try:
            episodes = _read_bank(path)
            if not episodes:
                raise ValueError("empty bank")
            hashes[str(path.resolve())] = file_hash(path)
            audit["artifacts_by_role"][key] = str(path.resolve())
            for e in episodes:
                if any(e.get(k) in (None, "") for k in REQUIRED_METADATA):
                    errors.append(key + " missing required metadata on episode")
                if e.get("split") not in valid_splits:
                    errors.append(key + " contains incompatible split " + str(e.get("split")))
            identities = [(e.get("root_id"), e.get("latent_world_id"), e.get("realization_id")) for e in episodes]
            if len(identities) != len(set(identities)):
                errors.append(key + " contains duplicate episode identities")
            banks[key] = episodes
            roots[key] = {e["root_id"] for e in episodes}
            signatures[key] = {canonical_signature(e) for e in episodes}
            templates[key] = {e.get("template_id") for e in episodes}
            compositions[key] = {e.get("composition_id") for e in episodes}
            instances[key] = {_instance_signature(e) for e in episodes}
            if key.startswith("test_"):
                audit["test_banks_opened"] = True
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append(key + ": invalid bank: " + str(exc))
    keys = list(roots)
    overlaps = {}
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            shared = roots[left] & roots[right]
            if shared:
                errors.append("ROOT_SPLIT_LEAKAGE " + left + ":" + right)
            if instances[left] & instances[right]:
                errors.append("EXACT_CONTENT_OVERLAP " + left + ":" + right)
            overlaps[left + ":" + right] = {"root_overlap_count": len(shared),
                "mechanism_overlap_count": len(signatures[left] & signatures[right]),
                "declared_template_overlap_count": len(templates[left] & templates[right])}
    test_key = "test_composition_bank"
    for reference in ("train_bank", "validation_bank"):
        if test_key not in banks or reference not in banks:
            continue
        for label, mapping in (("mechanism_signature", signatures), ("template_id", templates), ("composition_id", compositions)):
            if mapping[test_key] & mapping[reference]:
                errors.append("COMPOSITION_NOT_HELD_OUT " + label + " overlaps " + reference)
        if {_template_pattern(e) for e in banks[test_key]} & {_template_pattern(e) for e in banks[reference]}:
            errors.append("COMPOSITION_NOT_HELD_OUT derived surface template overlaps " + reference)
    historical = historical_bank_paths(historical_root) if historical_root else {}
    if not historical:
        errors.append("Historical bank manifest unavailable; cannot verify new-test independence")
    historical_roots, historical_instances = set(), set()
    for split, path in historical.items():
        if not path.is_file():
            errors.append("Historical bank missing: " + str(path))
            continue
        try:
            episodes = _read_bank(path)
            historical_roots.update(e["root_id"] for e in episodes)
            historical_instances.update(_instance_signature(e) for e in episodes)
            hashes[str(path.resolve())] = file_hash(path)
            audit["artifacts_by_role"]["historical_" + split] = str(path.resolve())
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append("Invalid historical bank " + split + ": " + str(exc))
    for test in ("test_id_bank", "test_composition_bank"):
        if roots.get(test, set()) & historical_roots or instances.get(test, set()) & historical_instances:
            errors.append("HISTORICAL_TEST_REUSE " + test + " intersects already available historical data")
        planned = precision["planned_" + ("test_id_roots" if test == "test_id_bank" else "test_composition_roots")]
        if test in roots and len(roots[test]) != planned:
            errors.append(test + " root count differs from precision plan")
    audit.update(partitions={k: {"roots": len(roots[k]), "episodes": len(v), "mechanisms": len(signatures[k])} for k, v in banks.items()},
                 overlaps=overlaps, historical_roots=len(historical_roots),
                 template_conditioning="ID inference is conditional on any reused declared or derived templates.",
                 limitations=["Structural and content fingerprints are finite checks; no semantic-equivalence certification.",
                              "New random seeds alone do not establish composition diversity.",
                              "No human validation is implied by this protocol audit."], human_audit="not_executed")
    return {"status": "passed" if not errors else "failed", "errors": sorted(set(errors)),
            "artifacts_sha256": hashes, "audit": audit}
