"""Synthetic unit fixtures verify gates; they are never real audit approvals."""
from copy import deepcopy
import json

import pytest

from research.src.artifacts import digest, file_hash, write_json
from research.src.protocol import FROZEN_REQUIRED_FILES, GateError, gate_status, require_gate


def save(root, relative, value):
    write_json(root / relative, value)


def load(root, relative):
    return json.loads((root / relative).read_text())


def hash_entry(root, relative):
    return {relative: file_hash(root / relative)}


def reviews():
    result = []
    for index in range(100):
        family = f"F{index % 5}"
        for reviewer in ("test_fixture_a", "test_fixture_b"):
            result.append({"root_id": f"fixture{index}", "reviewer_id": reviewer, "family": family,
                           "information_sufficient": True, "decisive_slot": None,
                           "admissible_answers": ["fixture"], "writable_scope": "ongoing",
                           "f4_both_plausible": True})
    return result


@pytest.fixture
def gated_root(tmp_path):
    root = tmp_path
    for relative in FROZEN_REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic unit fixture only\n")
    config = {"model": {"preferred": "Qwen/Qwen2.5-3B-Instruct",
                        "technical_fallback": "Qwen/Qwen2.5-1.5B-Instruct",
                        "selected_id": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "pinned_revision",
                        "tokenizer_revision": "pinned_tokenizer", "precision": "bfloat16", "quantization": "none"},
              "budget": {"post_sft_accelerator_seconds_cap": 300},
              "sample": {"test_id_locked_roots": 300}}
    save(root, "configs/protocol.yaml", config)
    update = {"status": "executed", "trainable_before": "a" * 64, "trainable_after": "b" * 64,
              "frozen_before": "c" * 64, "frozen_after": "c" * 64}
    smoke = {"status": "passed", "g0_eligible": True, "protocol_candidate": True,
             "load": "executed", "generation": {"status": "executed"},
             "sft": deepcopy(update), "grpo": {**update, "reference_before": "d" * 64, "reference_after": "d" * 64}}
    lock = {"status": "passed", "protocol_hash": digest(config), "model": deepcopy(config["model"]),
            "hardware": {"fixture": True}, "versions": {v: "fixture_version" for v in ("python", "torch", "transformers", "peft", "trainer")},
            "training_plan": {"batch": 8, "microbatch": 1, "accumulation": 8, "rollouts": 4, "accelerator_seconds_cap": 300},
            "technical_smoke": smoke}
    save(root, "configs/environment.lock.yaml", lock)
    save(root, "manifests/data_manifest.json", {"status": "fixture"})
    save(root, "manifests/human_audit.json", {"status": "passed", "reviews": reviews(), "adjudications": []})
    save(root, "manifests/gates.json", {g: {"status": "passed", "evidence": hash_entry(root, "manifests/data_manifest.json")}
                                         for g in ("G1", "G2", "G3", "G4", "G5")})
    save(root, "manifests/frozen_protocol.json", {"files": {relative: file_hash(root / relative) for relative in FROZEN_REQUIRED_FILES}})
    entries, b1s = [], {}
    for method in ("B1", "B2-L", "B3", "B2-I"):
        for seed in (101, 202, 303):
            adapter = f"checkpoints/{method}/{seed}/adapter_model.safetensors"
            path = root / adapter
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"fixture {method} {seed}".encode())
            sha = file_hash(path)
            if method == "B1":
                b1s[seed] = sha
            report = f"checkpoints/{method}/{seed}/training.json"
            save(root, report, {"status": "executed", "seed": seed, "best_step": 1,
                                "history": [{"step": 0, "validation": {"utility": 0}},
                                            {"step": 1, "validation": {"utility": 1}}]})
            entries.append({"method": method, "seed": seed, "adapter_file": adapter, "adapter_hash": sha,
                            "selected_step": 1, "selection_split": "validation", "step_zero_evaluated": True,
                            "validation_report": report, "validation_report_hash": file_hash(root / report),
                            "parent_checkpoint_hash": b1s[seed] if method != "B1" else None,
                            "initial_adapter_hash": b1s[seed] if method != "B1" else None,
                            "kl_reference_hash": b1s[seed] if method in {"B2-L", "B2-I"} else None})
    save(root, "manifests/selected_checkpoints.json", {"status": "passed", "selection_split": "validation",
                                                       "step_zero_eligible": True, "checkpoints": entries})
    return root


def test_all_artifacts_are_required_to_release_gates(gated_root):
    assert all(status["status"] == "passed" for status in gate_status(gated_root).values())


def test_empty_repository_fails_closed(tmp_path):
    with pytest.raises(GateError, match="TECHNICAL_LOCK_REQUIRED"):
        require_gate("G4", tmp_path)


@pytest.mark.parametrize("key,value", [("revision", ""), ("tokenizer_revision", "  "),
                                       ("selected_id", "other/model"), ("quantization", None)])
def test_G0_invalid_model_strings_block(gated_root, key, value):
    config = load(gated_root, "configs/protocol.yaml")
    config["model"][key] = value
    save(gated_root, "configs/protocol.yaml", config)
    assert gate_status(gated_root)["G0"]["status"] == "blocked"


def test_G0_discordant_locked_model_blocks(gated_root):
    lock = load(gated_root, "configs/environment.lock.yaml")
    lock["model"]["revision"] = "different_revision"
    save(gated_root, "configs/environment.lock.yaml", lock)
    assert "lock.model_disagrees_with_protocol" in gate_status(gated_root)["G0"]["reasons"]


@pytest.mark.parametrize("section,key", [("versions", "torch"), ("versions", "trainer"),
                                          ("training_plan", "accumulation")])
def test_G0_missing_runtime_or_plan_fields_blocks(gated_root, section, key):
    lock = load(gated_root, "configs/environment.lock.yaml")
    lock[section][key] = None
    save(gated_root, "configs/environment.lock.yaml", lock)
    assert f"lock.{section}.{key}" in gate_status(gated_root)["G0"]["reasons"]


def test_G0_status_without_real_updates_is_insufficient(gated_root):
    lock = load(gated_root, "configs/environment.lock.yaml")
    lock["technical_smoke"]["sft"]["trainable_after"] = lock["technical_smoke"]["sft"]["trainable_before"]
    save(gated_root, "configs/environment.lock.yaml", lock)
    assert "technical_smoke.sft.verified_update" in gate_status(gated_root)["G0"]["reasons"]


def test_G2_cannot_freeze_only_a_readme(gated_root):
    (gated_root / "README.md").write_text("fixture")
    save(gated_root, "manifests/frozen_protocol.json", {"files": hash_entry(gated_root, "README.md")})
    assert any("required frozen artifact missing" in reason for reason in gate_status(gated_root)["G2"]["reasons"])


def test_G2_original_human_reviews_are_required(gated_root):
    save(gated_root, "manifests/human_audit.json", {"status": "passed", "roots": 100,
                                                   "initial_exact_agreement": 1, "unresolved_roots": []})
    assert "pending_human_audit" in gate_status(gated_root)["G2"]["reasons"]


def test_G2_recomputes_actual_human_disagreement(gated_root):
    audit = load(gated_root, "manifests/human_audit.json")
    audit["reviews"][0]["writable_scope"] = "session"
    save(gated_root, "manifests/human_audit.json", audit)
    assert "pending_human_audit" in gate_status(gated_root)["G2"]["reasons"]


def test_frozen_file_path_cannot_escape_root(gated_root, tmp_path):
    frozen = load(gated_root, "manifests/frozen_protocol.json")
    frozen["files"]["../outside"] = "a" * 64
    save(gated_root, "manifests/frozen_protocol.json", frozen)
    assert "frozen artifact changed: ../outside" in gate_status(gated_root)["G2"]["reasons"]


def test_G4_checkpoint_seed_matrix_is_mandatory(gated_root):
    manifest = load(gated_root, "manifests/selected_checkpoints.json")
    manifest["checkpoints"].pop()
    save(gated_root, "manifests/selected_checkpoints.json", manifest)
    assert "checkpoint_method_seed_matrix_incomplete" in gate_status(gated_root)["G4"]["reasons"]


def test_G4_parent_and_KL_reference_are_selected_same_seed_B1(gated_root):
    manifest = load(gated_root, "manifests/selected_checkpoints.json")
    entry = next(e for e in manifest["checkpoints"] if e["method"] == "B2-L" and e["seed"] == 101)
    entry["parent_checkpoint_hash"] = "a" * 64
    entry["kl_reference_hash"] = "b" * 64
    save(gated_root, "manifests/selected_checkpoints.json", manifest)
    reasons = gate_status(gated_root)["G4"]["reasons"]
    assert "continuation_parent_not_same_seed_selected_B1: B2-L/101" in reasons
    assert "KL_reference_not_same_seed_B1: B2-L/101" in reasons


def test_G4_selection_on_test_is_blocked(gated_root):
    manifest = load(gated_root, "manifests/selected_checkpoints.json")
    manifest["checkpoints"][0]["selection_split"] = "test_id"
    save(gated_root, "manifests/selected_checkpoints.json", manifest)
    assert "checkpoint_selection_not_validation_with_step_zero: B1/101" in gate_status(gated_root)["G4"]["reasons"]


def test_G4_adapters_and_validation_reports_are_hashed(gated_root):
    manifest = load(gated_root, "manifests/selected_checkpoints.json")
    entry = manifest["checkpoints"][0]
    (gated_root / entry["adapter_file"]).write_bytes(b"changed")
    (gated_root / entry["validation_report"]).write_text("{}")
    reasons = gate_status(gated_root)["G4"]["reasons"]
    assert "changed or missing adapter: B1/101" in reasons
    assert "changed or missing validation report: B1/101" in reasons


def test_G4_step_zero_must_actually_be_eligible_for_selection(gated_root):
    manifest = load(gated_root, "manifests/selected_checkpoints.json")
    entry = manifest["checkpoints"][0]
    report = load(gated_root, entry["validation_report"])
    report["history"][0]["validation"]["utility"] = 2
    save(gated_root, entry["validation_report"], report)
    entry["validation_report_hash"] = file_hash(gated_root / entry["validation_report"])
    save(gated_root, "manifests/selected_checkpoints.json", manifest)
    assert "checkpoint_is_not_common_validation_best: B1/101" in gate_status(gated_root)["G4"]["reasons"]


def exploratory_fixture(root):
    """Unit-only artifact fixtures; none leave the temporary test directory."""
    from research.src.data.automatic_audit import AUDIT_SOURCE_FILES
    config = load(root, "configs/protocol.yaml")
    config["execution_mode"] = "exploratory"
    save(root, "configs/protocol.yaml", config)
    lock = load(root, "configs/environment.lock.yaml")
    lock["protocol_hash"] = digest(config)
    save(root, "configs/environment.lock.yaml", lock)
    source = root / "src/data/automatic_audit.py"
    source.write_text("unit fixture automatic audit source")
    junit = root / "reports/tests.xml"
    junit.parent.mkdir()
    junit.write_text('<testsuites><testsuite tests="4" failures="0" errors="0" skipped="0" /></testsuites>')
    families = {f"F{i}": 20 for i in range(5)}
    partition = {"roots": 100, "families": families, "sha256": "a" * 64}
    save(root, "manifests/data_manifest.json", {"partitions": {"train": partition}})
    save(root, "manifests/human_audit.json", {"status": "pending_human_audit", "reviews": [], "adjudications": []})
    save(root, "manifests/exploratory_authorization.json", {
        "scope": "automatic_audit_instead_of_human", "research_status": "exploratory", "not_human": True,
        "authorized_at": "2026-10-23T15:00:00+00:00", "user_instruction": "UNIT FIXTURE authorization, not a human approval"})
    report = {"status": "passed", "research_status": "exploratory", "not_human": True,
              "independent_human_review": False, "human_audit_status": "pending_human_audit",
              "T07": {"human_status": "not_executed_human", "disposition": "substituted_for_exploration"},
              "roots_checked": 100, "families": families, "f4_pairs": 20,
              "unresolved_inconsistencies": [], "audited_splits": ["train"], "data_partitions": {"train": partition},
              "source_hashes": {relative: file_hash(root / relative) for relative in AUDIT_SOURCE_FILES},
              "tests_report": {"status": "passed", "format": "junit_xml", "tests": 4,
                               "failures": 0, "errors": 0, "skipped": 0, "file": "reports/tests.xml", "sha256": file_hash(junit)},
              "checks": {key: 1 for key in ("root_partitions_and_counterfactual_prefixes", "initial_record_schema", "future_and_label_boundary",
                                            "semantic_probes", "normal_trajectory_truth_alignment", "teacher_phase_validation")},
              "teacher_phases": {"select": 1, "respond": 1, "write": 1}}
    save(root, "manifests/automatic_audit.json", report)
    required = set(FROZEN_REQUIRED_FILES) | {"src/data/automatic_audit.py", "manifests/automatic_audit.json", "manifests/exploratory_authorization.json"}
    save(root, "manifests/frozen_protocol.json", {"files": {relative: file_hash(root / relative) for relative in required}})
    gates = load(root, "manifests/gates.json")
    for gate in gates.values():
        gate["evidence"] = hash_entry(root, "manifests/data_manifest.json")
    gates["G5"]["research_status"] = "exploratory"
    save(root, "manifests/gates.json", gates)
    return root


@pytest.mark.parametrize("relative", ["src/scoring/conditions.py", "src/data/semantic_partition_v2.py", "src/data/development_v2.py"])
def test_changing_repaired_semantic_dependencies_invalidates_the_audit(gated_root, relative):
    exploratory_fixture(gated_root)
    assert gate_status(gated_root)["G2"]["status"] == "passed"
    (gated_root / relative).write_text("changed semantic dependency")
    assert gate_status(gated_root)["G2"]["status"] == "blocked"


def test_authorized_exploration_can_proceed_without_fabricating_human_T07(gated_root):
    exploratory_fixture(gated_root)
    statuses = gate_status(gated_root)
    assert all(value["status"] == "passed" for value in statuses.values())
    assert all(value["research_status"] == "exploratory" for value in statuses.values())
    assert load(gated_root, "manifests/human_audit.json")["status"] == "pending_human_audit"


def test_exploratory_config_alone_never_bypasses_human_gate(gated_root):
    exploratory_fixture(gated_root)
    (gated_root / "manifests/exploratory_authorization.json").unlink()
    assert "EXPLORATORY_AUTHORIZATION_REQUIRED" in gate_status(gated_root)["G2"]["reasons"]


def test_automatic_audit_cannot_release_confirmatory_mode(gated_root):
    exploratory_fixture(gated_root)
    config = load(gated_root, "configs/protocol.yaml")
    config["execution_mode"] = "confirmatory"
    save(gated_root, "configs/protocol.yaml", config)
    assert "pending_human_audit" in gate_status(gated_root)["G2"]["reasons"]


def test_exploratory_unresolved_semantics_and_test_gold_block(gated_root):
    exploratory_fixture(gated_root)
    report = load(gated_root, "manifests/automatic_audit.json")
    report["unresolved_inconsistencies"] = [{"code": "contradiction"}]
    report["audited_splits"] = ["test_id"]
    save(gated_root, "manifests/automatic_audit.json", report)
    reasons = gate_status(gated_root)["G2"]["reasons"]
    assert "automatic_audit_has_unresolved_inconsistencies" in reasons
    assert "automatic_audit_must_exclude_test_gold" in reasons


def test_exploratory_source_or_junit_change_invalidates_audit(gated_root):
    exploratory_fixture(gated_root)
    (gated_root / "src/scoring/evidence.py").write_text("changed semantics")
    (gated_root / "reports/tests.xml").write_text('<testsuite tests="4" failures="1" />')
    reasons = gate_status(gated_root)["G2"]["reasons"]
    assert "automatic_audit_source_changed: src/scoring/evidence.py" in reasons
    assert "automatic_audit_test_evidence_required" in reasons


def test_exploratory_G5_cannot_claim_confirmatory_completion(gated_root):
    exploratory_fixture(gated_root)
    gates = load(gated_root, "manifests/gates.json")
    gates["G5"]["research_status"] = "confirmatory"
    save(gated_root, "manifests/gates.json", gates)
    assert "exploratory_completion_must_not_claim_confirmatory" in gate_status(gated_root)["G5"]["reasons"]
