"""Explicitly nonhuman development audit for authorized exploratory runs.

This checks executable consistency, not natural-language plausibility or novelty.
It shares public semantics with the evaluator and cannot replace independent
human review in confirmatory claims. Test gold is never accepted by this API.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date
from pathlib import Path
import xml.etree.ElementTree as ET

from ..artifacts import ROOT, digest, file_hash, now
from .audit_splits import actor_prefix, audit_splits

AUDIT_SOURCE_FILES = (
    "src/data/automatic_audit.py", "src/data/build_bank.py", "src/data/audit_splits.py",
    "src/data/semantic_partition_v2.py", "src/data/development_v2.py", "src/scoring/conditions.py",
    "src/data/semantic_partition_v3.py", "src/policy/dialogue.py",
    "src/scoring/evidence.py", "src/scoring/reward.py", "src/policy/rule_controller.py",
    "src/policy/llm_controller.py", "src/env/episode.py", "src/env/user_simulator.py",
    "src/memory/store.py", "src/memory/executor.py", "src/readers/symbolic.py",
)
ALLOWED_SPLITS = {"dev_smoke", "train", "validation"}


def test_evidence(path, root):
    """Read actual JUnit output, preserving skipped counts and its source hash."""
    if path is None:
        return {"status": "not_executed", "reason": "JUnit report required"}
    root = Path(root).resolve()
    source = Path(path)
    source = source.resolve() if source.is_absolute() else (root / source).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        return {"status": "failed", "reason": "JUnit report must exist within research root"}
    try:
        document = ET.parse(source).getroot()
        suites = [document] if document.tag == "testsuite" else list(document.findall("testsuite"))
        counts = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
                  for key in ("tests", "failures", "errors", "skipped")}
        passed = counts["tests"] > 0 and counts["failures"] == counts["errors"] == 0
        return {"status": "passed" if passed else "failed", "format": "junit_xml",
                "file": str(source.relative_to(root)), "sha256": file_hash(source), **counts}
    except (OSError, ValueError, ET.ParseError):
        return {"status": "failed", "reason": "Invalid JUnit report"}


def _private_reference(episode):
    memory = deepcopy(episode["initial_memory"])
    truth = episode["truth"]
    if truth.get("scope") is not None:
        scope = deepcopy(truth["scope"])
        memory.append({"memory_id": "private_truth_record", "attribute": episode["current_query"]["attribute"],
                       "value": truth["new_value"], "scope": scope, "status": "confirmed",
                       "supersedes": [m["memory_id"] for m in memory if m["attribute"] == episode["current_query"]["attribute"]
                                      and m["scope"]["kind"] == "ongoing"] if scope["kind"] == "ongoing" else []})
    return memory


def audit_development(episodes, *, root=None, tests_report=None):
    """Audit every provided development episode; never select on model outcomes.

    Returns an auditable report. Caller saves it to manifests/automatic_audit.json
    and hashes it in G2 evidence. No files, approvals or labels are auto-written.
    """
    from .build_bank import teacher_trajectories
    from ..memory.executor import validate_record
    from ..policy.llm_controller import validate_action
    from ..policy.rule_controller import RuleController, expected_memory, compatible_worlds
    from ..readers.symbolic import read
    from ..scoring.evidence import audit_operation, probe_queries, score_response
    root = Path(root or ROOT)
    episodes = list(episodes)
    if any(e.get("split") not in ALLOWED_SPLITS for e in episodes):
        raise ValueError("AUTOMATIC_AUDIT_DEVELOPMENT_ONLY: test gold cannot enter audit/tuning")
    issues, checks = [], Counter()
    def failure(code, episode=None, detail=None):
        issues.append({"code": code, "root_id": episode.get("root_id") if episode else None,
                       "realization_id": episode.get("realization_id") if episode else None,
                       "detail": detail})
    try:
        split_report = audit_splits(episodes)
        checks["root_partitions_and_counterfactual_prefixes"] += 1
    except (ValueError, TypeError, KeyError) as exc:
        split_report = {"status": "failed", "partitions": {}}
        failure("split_or_counterfactual_inconsistency", detail=str(exc))
    roots, paired = {}, defaultdict(list)
    phases = Counter()
    for episode in episodes:
        try:
            rid = episode["root_id"]
            if rid in roots and roots[rid] != episode["scenario_family"]:
                failure("root_family_changes", episode)
            roots[rid] = episode["scenario_family"]
            if episode["scenario_family"] == "F4" and episode.get("opportunity", {}).get("eligible", True):
                paired[(rid, episode["realization_id"])].append(episode)
            for record in episode["initial_memory"]:
                validate_record(record, initial=True)
                if record.get("recorded_at", "") > episode["current_query"]["date"]:
                    failure("future_record_in_initial_memory", episode)
                checks["initial_record_schema"] += 1
            for name in ("current_query", "future_query"):
                query = episode[name]
                if set(query) != {"attribute", "date", "session_id", "context"}:
                    failure("private_or_extra_query_fields", episode)
                date.fromisoformat(query["date"])
            if episode["future_query"]["date"] <= episode["current_query"]["date"]:
                failure("future_session_not_after_acquisition", episode)
            if any(set(message) != {"turn_id", "text"} for message in episode["user_messages"]):
                failure("private_or_extra_message_fields", episode)
            changed_future = deepcopy(episode)
            changed_future["future_query"] = {"private_future": "changed"}
            changed_future["truth"] = {"private_truth": "changed"}
            if actor_prefix(changed_future) != actor_prefix(episode):
                failure("future_influences_actor_prefix", episode)
            checks["future_and_label_boundary"] += 1
            observation = {"memory": deepcopy(episode["initial_memory"]),
                           "user_messages": deepcopy(episode["user_messages"]),
                           "query": deepcopy(episode["current_query"]),
                           "reuse_regime": "later_same_attribute_without_questions"}
            controller = RuleController()
            decision = controller.select(observation)
            if decision["action"] == "ASK":
                observation["user_messages"].append(deepcopy(episode["replies"][decision["slot"]]))
            current = score_response(controller.respond(observation), observation, episode["truth"]["current_value"])
            if current["success"] != 1:
                failure("normal_teacher_current_not_correct_and_justified", episode, current)
            future = read(expected_memory(observation), episode["future_query"])
            if future["mode"] != "ANSWER" or future["value"] != episode["truth"]["future_value"]:
                failure("visible_evidence_future_differs_from_truth", episode)
            private_memory = _private_reference(episode)
            for query in probe_queries(observation, *private_memory):
                expected = read(private_memory, query)
                # Only singleton private-world answers impose a probe target;
                # context-free questions can legitimately have alternatives.
                visible_values = {world["value"] for world in compatible_worlds(observation, query=query)}
                if expected["mode"] == "ANSWER" and visible_values != {expected["value"]}:
                    failure("truth_and_visible_probe_disagree", episode, query)
                    break
                checks["semantic_probes"] += 1
            checks["normal_trajectory_truth_alignment"] += 1
            for example in teacher_trajectories([episode]):
                phases[example["phase"]] += 1
                validate_action(example["phase"], example["action"])
                prefix = example["observation"]
                if set(prefix) != {"memory", "user_messages", "query", "reuse_regime"}:
                    failure("teacher_private_fields", episode)
                if example["phase"] == "write":
                    for operation in example["action"]["operations"]:
                        if operation["op"] != "NOOP":
                            validate_record(operation["record"])
                        if not audit_operation(operation, prefix):
                            failure("teacher_write_unsupported", episode, example.get("continuation"))
                if example["phase"] == "respond":
                    response = example["action"]
                    # Truth can remain hidden after irrelevant/nonresponse
                    # replies, so test justification rather than require success.
                    scored = score_response(response, prefix, episode["truth"]["current_value"])
                    if response["mode"] == "ANSWER" and not scored["justified"]:
                        failure("teacher_unjustified_commitment", episode, example.get("continuation"))
                checks["teacher_phase_validation"] += 1
        except (ValueError, TypeError, KeyError) as exc:
            failure("episode_validation_error", episode, str(exc))
    pair_roots = set()
    for (rid, realization), pair in paired.items():
        if len({item["latent_world_id"] for item in pair}) >= 2:
            if len({actor_prefix(item) for item in pair}) != 1:
                failure("F4_prefix_not_identical", pair[0])
            elif len({item["replies"]["TEMPORAL_SCOPE"]["text"] for item in pair}) < 2:
                failure("F4_relevant_reply_not_selective", pair[0])
            else:
                pair_roots.add(rid)
    families = dict(Counter(roots.values()))
    if len(roots) < 100 or any(families.get(f"F{i}", 0) < 20 for i in range(5)):
        failure("minimum_100_roots_20_per_family_not_met")
    if len(pair_roots) < 20:
        failure("minimum_20_F4_pairs_not_met")
    evidence = test_evidence(tests_report, root)
    if evidence["status"] != "passed":
        failure("automated_test_evidence_missing_or_failed")
    hashes = {}
    for relative in AUDIT_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            failure("audit_source_missing", detail=relative)
        else:
            hashes[relative] = file_hash(path)
    return {"status": "passed" if not issues else "failed", "created_at": now(),
            "research_status": "exploratory", "not_human": True,
            "T07": {"human_status": "not_executed_human", "disposition": "substituted_for_exploration"},
            "human_audit_status": "pending_human_audit", "roots_checked": len(roots),
            "episodes_checked": len(episodes), "families": families, "f4_pairs": len(pair_roots),
            "audited_splits": sorted({e["split"] for e in episodes}), "episodes_sha256": digest(episodes),
            "data_partitions": split_report.get("partitions", {}), "source_hashes": hashes,
            "checks": dict(checks), "teacher_phases": dict(phases), "tests_report": evidence,
            "unresolved_inconsistencies": issues, "independent_human_review": False,
            "limitations": ["Automatic audit shares the evaluator's semantics; it is not an independent plausibility judgment.",
                            "No human agreement or confirmation is claimed; test gold was excluded."]}
