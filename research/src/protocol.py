"""Fail closed at experimental gates; implementation smoke never opens test.

Approval labels alone cannot release an experiment: locks, original human
reviews, frozen source files and selected checkpoint artifacts are verified.
"""
from __future__ import annotations

import math
from pathlib import Path
from .artifacts import ROOT, file_hash, load_config, digest


class GateError(RuntimeError):
    pass


FROZEN_REQUIRED_FILES = frozenset({
    "configs/protocol.yaml", "configs/environment.lock.yaml", "manifests/data_manifest.json",
    "prompts/select.txt", "prompts/respond.txt", "prompts/write.txt", "prompts/reader.txt", "prompts/extractor.txt",
    "src/scoring/evidence.py", "src/scoring/reward.py", "src/policy/rule_controller.py",
    "src/policy/llm_controller.py", "src/readers/symbolic.py", "src/readers/frozen_llm.py",
    "src/env/episode.py", "src/env/user_simulator.py", "src/memory/store.py", "src/memory/executor.py",
    "src/protocol.py", "src/artifacts.py", "src/data/build_bank.py", "src/data/audit_splits.py", "src/evaluation/runner.py",
    "src/data/semantic_partition_v2.py", "src/data/development_v2.py", "src/scoring/conditions.py",
    "src/data/semantic_partition_v3.py", "src/policy/dialogue.py",
    "src/training/campaign.py", "src/closure_training.py",
    "src/evaluation/persistence_interventions.py", "src/training/sft.py", "src/training/episodic_grpo.py",
    "src/training/resources.py",
})


def _load(path):
    if not path.exists():
        return {}
    try:
        value = load_config(path)
        return value if isinstance(value, dict) else {}
    except (ValueError, OSError):
        return {}


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _checked_file(root, relative, expected):
    if not _nonempty(relative) or Path(relative).is_absolute() or not _sha(expected):
        return False
    path = (root / relative).resolve()
    return path.is_relative_to(root.resolve()) and path.is_file() and file_hash(path) == expected


def _human_audit_errors(root, audit):
    """Recompute the existing audit helper from original reviews, not counters.

    human_audit.json contains reviews/adjudications lists or reviews_file plus
    reviews_sha256 (and optional adjudications_file/adjudications_sha256). Raw
    records retain both independent decisions before adjudication.
    """
    from .data.audit_splits import audit_human_reviews
    import json
    values = {}
    for field in ("reviews", "adjudications"):
        if field in audit:
            values[field] = audit[field]
        elif audit.get(field + "_file"):
            relative = audit[field + "_file"]
            expected = audit.get(field + "_sha256")
            if not _checked_file(root, relative, expected):
                return ["invalid human audit artifact: " + field]
            try:
                values[field] = json.loads((root / relative).read_text())
            except (ValueError, OSError):
                return ["invalid human audit JSON: " + field]
        else:
            values[field] = []
    if not isinstance(values["reviews"], list) or not isinstance(values["adjudications"], list):
        return ["pending_human_audit: original records required"]
    try:
        summary = audit_human_reviews(values["reviews"], values["adjudications"])
    except (ValueError, TypeError, KeyError):
        return ["pending_human_audit: malformed original records"]
    errors = []
    if audit.get("status") != "passed" or summary.get("status") != "passed":
        errors.append("pending_human_audit")
    # If summary fields are supplied, they must match the actual helper output.
    for key, actual in summary.items():
        if key in audit and key != "status" and audit[key] != actual:
            errors.append("human audit summary mismatch: " + key)
    return errors



def _exploratory_authorization_errors(root, authorization):
    from datetime import datetime
    errors = []
    if (authorization.get("scope") != "automatic_audit_instead_of_human"
            or authorization.get("research_status") != "exploratory"
            or authorization.get("not_human") is not True
            or not _nonempty(authorization.get("user_instruction"))):
        errors.append("EXPLORATORY_AUTHORIZATION_REQUIRED")
    try:
        datetime.fromisoformat(authorization["authorized_at"].replace("Z", "+00:00"))
    except (KeyError, ValueError, TypeError, AttributeError):
        errors.append("exploratory_authorization_date_required")
    return errors


def _automatic_audit_errors(root):
    from .data.automatic_audit import AUDIT_SOURCE_FILES, ALLOWED_SPLITS
    audit = _load(root / "manifests/automatic_audit.json")
    errors = []
    if (audit.get("status") != "passed" or audit.get("research_status") != "exploratory"
            or audit.get("not_human") is not True or audit.get("independent_human_review") is not False
            or audit.get("human_audit_status") != "pending_human_audit"
            or audit.get("T07") != {"human_status": "not_executed_human", "disposition": "substituted_for_exploration"}):
        errors.append("exploratory_automatic_audit_required")
    roots = audit.get("roots_checked", 0)
    families = audit.get("families", {})
    pairs = audit.get("f4_pairs", 0)
    if (not isinstance(roots, int) or isinstance(roots, bool) or roots < 100
            or not isinstance(families, dict)
            or any(not isinstance(families.get(f"F{i}"), int) or families.get(f"F{i}", 0) < 20 for i in range(5))
            or not isinstance(pairs, int) or isinstance(pairs, bool) or pairs < 20):
        errors.append("automatic_audit_sample_insufficient")
    if audit.get("unresolved_inconsistencies") != []:
        errors.append("automatic_audit_has_unresolved_inconsistencies")
    splits = audit.get("audited_splits", [])
    if not isinstance(splits, list) or not splits or not set(splits) <= ALLOWED_SPLITS:
        errors.append("automatic_audit_must_exclude_test_gold")
    source_hashes = audit.get("source_hashes", {})
    if not isinstance(source_hashes, dict):
        source_hashes = {}
    for relative in AUDIT_SOURCE_FILES:
        if not _checked_file(root, relative, source_hashes.get(relative)):
            errors.append("automatic_audit_source_changed: " + relative)
    tests = audit.get("tests_report", {})
    if (not isinstance(tests, dict) or tests.get("status") != "passed" or tests.get("format") != "junit_xml"
            or not isinstance(tests.get("tests"), int) or tests.get("tests", 0) <= 0
            or tests.get("failures") != 0 or tests.get("errors") != 0
            or not _checked_file(root, tests.get("file"), tests.get("sha256"))):
        errors.append("automatic_audit_test_evidence_required")
    else:
        from .data.automatic_audit import test_evidence
        actual = test_evidence(tests["file"], root)
        if any(actual.get(key) != tests.get(key) for key in ("status", "tests", "failures", "errors", "skipped", "sha256")):
            errors.append("automatic_audit_test_summary_mismatch")
    checks = audit.get("checks", {})
    for key in ("root_partitions_and_counterfactual_prefixes", "initial_record_schema", "future_and_label_boundary",
                "semantic_probes", "normal_trajectory_truth_alignment", "teacher_phase_validation"):
        if not isinstance(checks, dict) or not _positive(checks.get(key)):
            errors.append("automatic_audit_check_not_executed: " + key)
    phases = audit.get("teacher_phases", {})
    if not isinstance(phases, dict) or any(not _positive(phases.get(key)) for key in ("select", "respond", "write")):
        errors.append("automatic_audit_teacher_phases_incomplete")
    partitions = audit.get("data_partitions", {})
    manifest = _load(root / "manifests/data_manifest.json")
    declared = manifest.get("partitions", {})
    if not isinstance(partitions, dict) or not isinstance(declared, dict):
        errors.append("automatic_audit_data_manifest_missing")
    else:
        declared_development = {split for split in declared if split in ALLOWED_SPLITS}
        if not partitions or not declared_development or set(partitions) != set(splits) or set(partitions) != declared_development:
            errors.append("automatic_audit_development_partition_coverage_incomplete")
        try:
            if sum(data["roots"] for data in partitions.values()) != roots:
                errors.append("automatic_audit_root_denominator_mismatch")
            combined_families = {f"F{i}": sum(data["families"].get(f"F{i}", 0) for data in partitions.values()) for i in range(5)}
            if any(combined_families[f"F{i}"] != families.get(f"F{i}") for i in range(5)):
                errors.append("automatic_audit_family_denominator_mismatch")
        except (ValueError, TypeError, KeyError):
            errors.append("automatic_audit_partition_counts_invalid")
        for split, data in declared.items():
            if split in ALLOWED_SPLITS and (split not in partitions or data.get("sha256") != partitions[split].get("sha256")):
                errors.append("automatic_audit_data_hash_mismatch: " + split)
    return errors


def _checkpoint_errors(root):
    manifest = _load(root / "manifests/selected_checkpoints.json")
    errors = []
    if manifest.get("status") != "passed" or manifest.get("selection_split") != "validation" or manifest.get("step_zero_eligible") is not True:
        errors.append("selected_checkpoints_not_frozen_on_validation")
    entries = manifest.get("checkpoints", [])
    if not isinstance(entries, list):
        return errors + ["malformed selected checkpoints"]
    required = {(method, seed) for method in ("B1", "B2-L", "B3", "B2-I") for seed in (101, 202, 303)}
    indexed = {}
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("malformed checkpoint entry")
            continue
        if not _nonempty(entry.get("method")) or not isinstance(entry.get("seed"), int) or isinstance(entry.get("seed"), bool):
            errors.append("malformed checkpoint identity")
            continue
        key = (entry.get("method"), entry.get("seed"))
        if key in indexed:
            errors.append("duplicate selected checkpoint")
        indexed[key] = entry
    if set(indexed) != required:
        errors.append("checkpoint_method_seed_matrix_incomplete")
    for (method, seed), entry in indexed.items():
        label = f"{method}/{seed}"
        if not _checked_file(root, entry.get("adapter_file"), entry.get("adapter_hash")):
            errors.append("changed or missing adapter: " + label)
        if entry.get("selection_split") != "validation" or entry.get("step_zero_evaluated") is not True:
            errors.append("checkpoint_selection_not_validation_with_step_zero: " + label)
        selected_step = entry.get("selected_step")
        if not isinstance(selected_step, int) or isinstance(selected_step, bool) or selected_step < 0:
            errors.append("invalid selected checkpoint step: " + label)
        report_file, report_hash = entry.get("validation_report"), entry.get("validation_report_hash")
        if not _checked_file(root, report_file, report_hash):
            errors.append("changed or missing validation report: " + label)
        else:
            report = _load(root / report_file)
            history = report.get("history", [])
            evaluated = {item.get("step") for item in history if isinstance(item, dict) and isinstance(item.get("validation"), dict)} if isinstance(history, list) else set()
            if (report.get("status") != "executed" or report.get("seed") != seed
                    or report.get("best_step") != selected_step or 0 not in evaluated or selected_step not in evaluated):
                errors.append("validation report lacks selected/zero checkpoint: " + label)
            else:
                try:
                    from .training.sft import selection_key
                    candidates = [(item["step"], selection_key(item["validation"], item["step"]))
                                  for item in history if isinstance(item, dict) and isinstance(item.get("validation"), dict)]
                    expected_step = max(candidates, key=lambda candidate: candidate[1])[0]
                    if expected_step != selected_step:
                        errors.append("checkpoint_is_not_common_validation_best: " + label)
                except (ValueError, TypeError, KeyError):
                    errors.append("invalid validation selection metrics: " + label)
        if method != "B1":
            b1 = indexed.get(("B1", seed), {})
            parent = b1.get("adapter_hash")
            if not _sha(parent) or entry.get("parent_checkpoint_hash") != parent or entry.get("initial_adapter_hash") != parent:
                errors.append("continuation_parent_not_same_seed_selected_B1: " + label)
            if selected_step == 0 and entry.get("adapter_hash") != parent:
                errors.append("step_zero_adapter_differs_from_B1: " + label)
            if method in {"B2-L", "B2-I"} and entry.get("kl_reference_hash") != parent:
                errors.append("KL_reference_not_same_seed_B1: " + label)
    return errors


def gate_status(root=None):
    root = Path(root or ROOT)
    config = _load(root / "configs/protocol.yaml")
    lock = _load(root / "configs/environment.lock.yaml")
    gates = _load(root / "manifests/gates.json")
    reasons = {}
    missing = []
    model = config.get("model", {})
    if not isinstance(model, dict):
        model = {}
    for key in ("selected_id", "revision", "tokenizer_revision", "precision", "quantization"):
        if not _nonempty(model.get(key)):
            missing.append("model." + key)
    if model.get("selected_id") not in {model.get("preferred"), model.get("technical_fallback")} or not _nonempty(model.get("selected_id")):
        missing.append("model.selected_id_not_approved")
    cap = config.get("budget", {}).get("post_sft_accelerator_seconds_cap")
    if not _positive(cap):
        missing.append("budget.post_sft_accelerator_seconds_cap")
    smoke = lock.get("technical_smoke", {})
    if not isinstance(smoke, dict):
        smoke = {}
    if (lock.get("status") != "passed" or smoke.get("status") != "passed"
            or smoke.get("g0_eligible") is not True or smoke.get("protocol_candidate") is not True
            or smoke.get("load") != "executed" or not isinstance(smoke.get("generation"), dict)
            or smoke.get("generation", {}).get("status") != "executed"):
        missing.append("technical_smoke")
    for algorithm in ("sft", "grpo"):
        result = smoke.get(algorithm, {})
        if (not isinstance(result, dict) or result.get("status") != "executed"
                or not _sha(result.get("trainable_before")) or not _sha(result.get("trainable_after"))
                or result.get("trainable_before") == result.get("trainable_after")
                or not _sha(result.get("frozen_before")) or result.get("frozen_before") != result.get("frozen_after")):
            missing.append("technical_smoke." + algorithm + ".verified_update")
    rl_result = smoke.get("grpo", {})
    if not isinstance(rl_result, dict) or not _sha(rl_result.get("reference_before")) or rl_result.get("reference_before") != rl_result.get("reference_after"):
        missing.append("technical_smoke.grpo.frozen_B1_reference")
    if config.get("runtime", {}).get("device") == "cuda":
        capacity = smoke.get("context_capacity", {})
        provenance = smoke.get("candidate_provenance", {})
        if (smoke.get("device") != "cuda" or smoke.get("precision") != model.get("precision")
                or smoke.get("gradient_checkpointing") is not True
                or provenance.get("model_id") != model.get("selected_id")
                or provenance.get("revision") != model.get("revision")
                or provenance.get("tokenizer_revision") != model.get("tokenizer_revision")):
            missing.append("technical_smoke.runtime_or_identity_mismatch")
        if (capacity.get("status") != "passed" or capacity.get("finite_adapter_gradients") is not True
                or capacity.get("input_tokens") != config.get("limits", {}).get("max_input_tokens_per_phase")
                or capacity.get("output_tokens") != config.get("limits", {}).get("max_write_tokens")
                or not _positive(smoke.get("cuda_peak_reserved_bytes"))):
            missing.append("technical_smoke.gpu_context_capacity")
    if not isinstance(lock.get("hardware"), dict) or not lock.get("hardware"):
        missing.append("lock.hardware")
    versions = lock.get("versions", {})
    for key in ("python", "torch", "transformers", "peft", "trainer"):
        if not isinstance(versions, dict) or not _nonempty(versions.get(key)):
            missing.append("lock.versions." + key)
    if lock.get("model") != model:
        missing.append("lock.model_disagrees_with_protocol")
    plan = lock.get("training_plan", {})
    for key in ("batch", "microbatch", "accumulation", "rollouts"):
        if not isinstance(plan, dict) or not isinstance(plan.get(key), int) or isinstance(plan.get(key), bool) or plan[key] < 1:
            missing.append("lock.training_plan." + key)
    if not isinstance(plan, dict) or plan.get("accelerator_seconds_cap") != cap or not _positive(plan.get("accelerator_seconds_cap")):
        missing.append("lock.training_plan.accelerator_seconds_cap")
    if lock.get("protocol_hash") != digest(config):
        missing.append("lock.protocol_hash")
    reasons["G0"] = missing
    for gate in ("G1", "G2", "G3", "G4", "G5"):
        item = gates.get(gate, {})
        if not isinstance(item, dict):
            item = {}
        errors = [] if item.get("status") == "passed" else ["gate evidence pending"]
        evidence = item.get("evidence", {})
        if not isinstance(evidence, dict) or not evidence:
            errors.append("missing hashed evidence")
            evidence = {}
        for relative, expected in evidence.items():
            if not _checked_file(root, relative, expected):
                errors.append("changed or missing evidence: " + str(relative))
        reasons[gate] = errors
    exploratory = config.get("execution_mode") == "exploratory"
    if exploratory:
        reasons["G2"].extend(_exploratory_authorization_errors(root, _load(root / "manifests/exploratory_authorization.json")))
        reasons["G2"].extend(_automatic_audit_errors(root))
        # G5 can close the authorized exploratory campaign, never certify the
        # original confirmatory study or claim that human T07 was performed.
        if gates.get("G5", {}).get("research_status") != "exploratory":
            reasons["G5"].append("exploratory_completion_must_not_claim_confirmatory")
    else:
        reasons["G2"].extend(_human_audit_errors(root, _load(root / "manifests/human_audit.json")))
    frozen = _load(root / "manifests/frozen_protocol.json")
    files = frozen.get("files", {})
    if not isinstance(files, dict):
        files = {}
    required_frozen = set(FROZEN_REQUIRED_FILES)
    if exploratory:
        required_frozen.update({"src/data/automatic_audit.py", "manifests/automatic_audit.json", "manifests/exploratory_authorization.json"})
    for relative in sorted(required_frozen - files.keys()):
        reasons["G2"].append("required frozen artifact missing: " + relative)
    for relative, expected in files.items():
        if not _checked_file(root, relative, expected):
            reasons["G2"].append("frozen artifact changed: " + str(relative))
    size = config.get("sample", {}).get("test_id_locked_roots")
    if not isinstance(size, int) or isinstance(size, bool) or not 300 <= size <= 1200 or size % 5:
        reasons["G4"].append("test_sample_size_not_locked")
    reasons["G4"].extend(_checkpoint_errors(root))
    statuses, prerequisites = {}, []
    for gate, errors in reasons.items():
        statuses[gate] = {"status": "passed" if not errors and not prerequisites else "blocked",
                          "reasons": errors, "blocked_by": list(prerequisites),
                          "research_status": "exploratory" if exploratory else "confirmatory",
                          "human_audit_status": "pending_human_audit" if exploratory else "required_independent_review"}
        if errors:
            prerequisites.append(gate)
    return statuses


def require_gate(gate, root=None):
    status = gate_status(root)
    if gate not in status:
        raise ValueError("Unknown gate: " + gate)
    if status[gate]["status"] != "passed":
        code = "TECHNICAL_LOCK_REQUIRED" if status["G0"]["status"] != "passed" else "GATE_REQUIRED"
        raise GateError(f"{code}: {gate}: {status[gate]}")


def assert_split_allowed(split, *, training=False, root=None):
    from .closure_protocol import active_split_authorization
    if active_split_authorization(split, training):
        return
    if split.startswith("test"):
        if training:
            raise GateError("CLOSED_TEST: test examples cannot receive gradients")
        require_gate("G4", root)
    elif training:
        if split != "train":
            raise GateError("Training requires train partition")
        require_gate("G2", root)
