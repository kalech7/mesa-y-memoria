"""Prepare a reviewable, explicitly exploratory local execution without test access."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import xml.etree.ElementTree as ET

from .artifacts import ROOT, canonical, digest, file_hash, load_config, now, write_json
from .protocol import gate_status, require_gate


def read_bank(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def mark_gate(gate, evidence, reason, *, root=None):
    root = Path(root or ROOT)
    gates = load_config(root / "manifests/gates.json")
    gates[gate] = {"status": "passed", "research_status": "exploratory", "completed_at": now(),
                   "reason": reason, "evidence": {str(Path(p).resolve().relative_to(root.resolve())): file_hash(p) for p in evidence}}
    write_json(root / "manifests/gates.json", gates)
    require_gate(gate, root)


def prepare_exploration(destination, *, root=None, run_tests=True):
    from .data.build_bank import build_bank, build_full_bank, write_bank, teacher_trajectories
    from .data.independent_bank import build_independent_bank, build_ai_external_proxy, bank_manifest
    from .data.audit_splits import audit_splits, prepare_human_audit
    from .data.automatic_audit import audit_development
    from .execution_state import freeze_protocol
    root = Path(root or ROOT).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    config = load_config(root / "configs/protocol.yaml")
    if config.get("execution_mode") != "exploratory":
        raise ValueError("This execution requires explicit exploratory authorization")
    require_gate("G0", root)
    context_path = destination / "context.json"
    if context_path.exists():
        require_gate("G2", root)
        return load_config(context_path)
    tests_xml = destination / "acceptance_tests.xml"
    if run_tests:
        print(canonical({"stage": "automated_acceptance_tests", "status": "running"}), flush=True)
        result = subprocess.run(["/usr/bin/python3", "-m", "pytest", "-q", "tests", "research/tests",
                                 "--junitxml=" + str(tests_xml)], cwd=root.parent,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                env={**os.environ, "PYTHONPATH": str(root.parent) + os.pathsep + str(root.parent / "tests")})
        (destination / "acceptance_tests.log").write_text(result.stdout)
        if result.returncode:
            raise RuntimeError("ACCEPTANCE_TESTS_FAILED: " + str(destination / "acceptance_tests.log"))
    suite = ET.parse(tests_xml).getroot().find("testsuite")
    if suite is None or int(suite.attrib["failures"]) or int(suite.attrib["errors"]):
        raise RuntimeError("Missing successful real acceptance evidence")
    evidence = {"status": "passed", "tests": int(suite.attrib["tests"]),
                "failures": 0, "errors": 0, "skipped": int(suite.attrib["skipped"]), "checked_at": now()}
    write_json(destination / "acceptance_summary.json", evidence)
    mark_gate("G1", [tests_xml, destination / "acceptance_summary.json"],
              "Real environment, evaluator, leakage, persistence and trainer checks passed", root=root)
    data = root / "data/bank_1_1"
    data.mkdir(parents=True, exist_ok=True)
    paths = {name: data / (name + ".jsonl") for name in ("dev_smoke", "train", "validation", "test_id", "test_ood", "test_external")}
    if not paths["dev_smoke"].exists():
        write_bank(build_bank(), paths["dev_smoke"])
    if not all(paths[name].exists() for name in ("train", "validation", "test_id")):
        for name, rows in build_full_bank(root=root).items():
            if not paths[name].exists():
                write_bank(rows, paths[name])
    if not paths["test_ood"].exists():
        write_bank(build_independent_bank(roots=config["sample"]["test_ood_roots"]), paths["test_ood"])
    if not paths["test_external"].exists():
        write_bank(build_ai_external_proxy(roots=config["sample"]["external_roots"]), paths["test_external"])
    banks = {name: read_bank(path) for name, path in paths.items()}
    split_report = audit_splits(banks)
    # This audit checks root IDs/construction, not model outputs or test tuning.
    write_json(destination / "split_integrity.json", split_report)
    manifest = {"status": "constructed_pending_audit", "protocol_version": config["protocol_version"],
                "research_status": "exploratory", "data_seed": config["seeds"]["data_partition"],
                "partitions": split_report["partitions"],
                "data_file_hashes": {str(path.relative_to(root)): file_hash(path) for path in paths.values()},
                "authorship": {name: bank_manifest(banks[name]) for name in ("test_ood", "test_external")},
                "human_authored_external_status": "not_executed", "human_audit": "pending_human_audit", "test_opened": False}
    for name in ("primary_codex_review.json", "astra_high_review.json", "adjudication.json"):
        review = root / "reports/ai_reviews" / name
        if not review.is_file() or load_config(review).get("status") != "executed":
            raise RuntimeError("USER_REQUESTED_AI_REVIEW_PENDING: " + str(review))
        manifest["data_file_hashes"][str(review.relative_to(root))] = file_hash(review)
    adjudication = load_config(root / "reports/ai_reviews/adjudication.json")
    if adjudication.get("unresolved_decision_differences"):
        raise RuntimeError("AI_REVIEW_HAS_UNRESOLVED_DECISIONS")
    manifest["ai_review_status"] = "executed_primary_and_astra_high_not_human"
    write_json(root / "manifests/data_manifest.json", manifest)
    # Fixed balanced robustness roots are selected before any test policy result.
    rng = random.Random(config["seeds"]["data_partition"])
    root_families = {e["root_id"]: e["scenario_family"] for e in banks["test_id"]}
    robust = [rid for family in ("F0", "F1", "F2", "F3", "F4")
              for rid in rng.sample(sorted(r for r, f in root_families.items() if f == family), 20)]
    write_json(root / "manifests/robustness_roots.json", {"root_ids": robust, "seed": config["seeds"]["data_partition"],
               "selected_before_test": True, "families": {f: 20 for f in ("F0", "F1", "F2", "F3", "F4")}})
    development = banks["dev_smoke"] + banks["train"] + banks["validation"]
    packet = prepare_human_audit(development)
    review_dir = root / "reports/human_review_materials"
    for reviewer, records in packet["reviewer_packets"].items():
        write_json(review_dir / (reviewer + ".json"), {"status": "pending_human_audit", "stories": records})
    write_json(root / "manifests/private/human_audit_key.json", packet["private_key"])
    print(canonical({"stage": "automatic_development_audit", "status": "running",
                     "independent_roots": len({e["root_id"] for e in development})}), flush=True)
    audit = audit_development(development, root=root, tests_report=tests_xml)
    write_json(root / "manifests/automatic_audit.json", audit)
    if audit["status"] != "passed":
        raise RuntimeError("AUTOMATIC_AUDIT_FAILED: " + str(root / "manifests/automatic_audit.json"))
    examples = list(teacher_trajectories(banks["train"]))
    corpus = data / "sft_teacher.jsonl"
    if not corpus.exists():
        with corpus.open("x") as stream:
            for item in examples:
                stream.write(canonical(item) + "\n")
    manifest["data_file_hashes"][str(corpus.relative_to(root))] = file_hash(corpus)
    manifest["status"] = "automatically_audited_exploratory"
    manifest["sft_examples"] = len(examples)
    manifest["sft_examples_by_phase"] = dict(Counter(e["phase"] for e in examples))
    write_json(root / "manifests/data_manifest.json", manifest)
    local_campaign = root / "configs/campaign.local.json"
    if local_campaign.exists():
        local = load_config(local_campaign)
        local["data_status"] = "automatically_audited_exploratory"
        write_json(local_campaign, local)
    # Do not freeze empty placeholders that G4 will replace with real artifacts.
    selected = root / "manifests/selected_checkpoints.json"
    if selected.exists() and load_config(selected).get("status") != "passed":
        archive = root / "archive/pre_execution/selected_checkpoints.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            selected.rename(archive)
        else:
            selected.unlink()
    shared = {str(path.relative_to(root.parent)): file_hash(path) for path in
              (root.parent / "memory/store.py", root.parent / "llm_local_001/model.py")}
    write_json(root / "manifests/shared_dependencies.json", {"files": shared, "checked_at": now()})
    freeze_protocol(root)
    mark_gate("G2", [root / "manifests/automatic_audit.json", root / "manifests/exploratory_authorization.json", tests_xml],
              "Authorized exploratory substitution only: nonhuman audit passes, protocol/data frozen; human T07 not executed", root=root)
    context = {"root": str(root), "destination": str(destination), "config": config,
               "paths": {name: str(path) for name, path in paths.items()},
               "model_path": str(root.parent / config["runtime"]["model_path"]),
               "robustness_root_ids": robust, "research_status": "exploratory", "prepared_at": now()}
    write_json(context_path, context)
    print(canonical({"stage": "preparation", "status": "executed", "gates": {g: d["status"] for g, d in gate_status(root).items()}}), flush=True)
    return context
