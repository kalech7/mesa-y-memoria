"""Lossless episode identification for the V4 audit of historical traces.

This module never deduplicates rows, repairs model outputs or recomputes historical
scores. Missing values are the literal JSON ``null`` in CSV, including a missing
nonresponse rate. Original JSONL line references remain the source of raw tokens.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from .artifacts import canonical, digest, file_hash, write_json

PRIMARY_KEY = ("run_id", "condition_id", "method", "checkpoint_id", "training_seed",
               "root_id", "latent_world_id", "realization_id", "future_probe_id")
CONDITION_FIELDS = ("experiment", "split", "observation_condition", "intervention",
                    "control_group", "reader_id", "extraction_track", "nonresponse_rate",
                    "decoding_config_hash", "model_revision", "tokenizer_revision",
                    "parser_version", "evaluator_version", "simulator_version",
                    "generation_seed", "stochastic_replica_id", "max_questions_s3")
LEGACY_KEY = ("experiment", "split", "method", "seed", "root_id", "latent_world_id",
              "realization_id", "scenario_family", "observation_condition", "intervention",
              "control_group", "reader_id", "extraction_track")


def _control(row):
    if "control_group" in row:
        return row["control_group"]
    if row.get("experiment") != "E3":
        return None
    episode = row.get("trace", {}).get("episode", {})
    opportunity = episode.get("opportunity", {})
    if opportunity.get("eligible"):
        return "primary_F4"
    if episode.get("current_query", {}).get("attribute") == "drink_preference":
        return "unaffected_attribute"
    return "clear_case" if opportunity.get("negative_control") else "other"


def _cell(value):
    return "null" if value is None else canonical(value) if isinstance(value, (dict, list, bool)) else value


def _normal_record(record):
    scope = record.get("scope", {})
    kind = scope.get("kind")
    boundary = scope.get("valid_until_exclusive")
    # Preserve the actual scope as well: a missing boundary has different
    # meanings for ongoing, session-bound and unknown facts.
    boundary_semantics = ("explicit_exclusive_boundary" if boundary is not None else
                          "until_superseded" if kind == "ongoing" else
                          "session_bound" if kind == "session" else
                          "context_bound_no_explicit_end" if kind == "contextual" else "unknown")
    return {"memory_id": record.get("memory_id"), "attribute": record.get("attribute"),
            "value": record.get("value"), "context_scope": scope.get("context"),
            "valid_from": scope.get("valid_from"), "valid_until": boundary,
            "valid_until_semantics": boundary_semantics, "duration_mode": kind or "unknown",
            "session_id": scope.get("session_id"),
            "source_evidence_ids": record.get("source_turn_ids", []),
            "evidence_quotes": record.get("evidence_quotes", []),
            "supersedes": record.get("supersedes", []), "raw_record": record}


def flatten_episode(row, run_id, *, source_path=None, source_line=None, line_sha256=None):
    """Export observed fields only; derived J is a per-episode product."""
    trace = row.get("trace", {})
    episode = trace.get("episode", {})
    metrics = row.get("metrics", {})
    rewards = row.get("rewards_by_component", {})
    flat = {k: v for k, v in row.items() if v is None or isinstance(v, (str, int, float, bool))}
    flat.update(metrics)
    flat.update({k: v for k, v in rewards.items() if k != "components"})
    flat.update(run_id=run_id, evidence_origin="historical_reexport", control_group=_control(row),
                training_seed=row.get("training_seed", row.get("seed")),
                nonresponse_rate=row.get("nonresponse_rate"),
                generation_seed=row.get("generation_seed", trace.get("rollout_seed")),
                stochastic_replica_id=row.get("stochastic_replica_id"),
                parser_version=row.get("parser_version"),
                evaluator_version=row.get("evaluator_version", row.get("evaluator_hash")),
                simulator_version=row.get("simulator_version"),
                source_path=str(source_path) if source_path else None, source_line=source_line,
                source_line_sha256=line_sha256, metrics_original=metrics,
                rewards_by_component_original=rewards,
                resource_counters=row.get("resource_counters", {}))
    for name in ("template_id", "composition_id", "generator_version", "domain", "observable_state_id"):
        flat[name] = row.get(name, episode.get(name))
    checkpoint = row.get("checkpoint_id", row.get("adapter_hash"))
    if checkpoint is None and row.get("model_revision"):
        checkpoint = "base_revision:" + row["model_revision"]
    flat["checkpoint_id"] = checkpoint
    flat["checkpoint_identity_status"] = "recorded" if checkpoint is not None else "not_identifiable_from_current_artifacts"
    flat["future_probe_id"] = row.get("future_probe_id", "primary" if "FJS" in metrics else "current_only")
    flat["FJS_primary"] = metrics.get("FJS_primary", metrics.get("FJS"))
    c2, fjs = metrics.get("C2"), flat["FJS_primary"]
    flat["J_primary"] = c2 * fjs if c2 is not None and fjs is not None else None
    flat["current_error"] = 1 - c2 if c2 is not None else None
    flat["unsupported_write_count"] = metrics.get("unsupported_write_count", metrics.get("unsupported_operations"))
    flat["invalid_output_count"] = metrics.get("invalid_output_count", metrics.get("invalid_outputs"))
    invalid = flat["invalid_output_count"]
    flat["any_invalid_output"] = int(invalid > 0) if invalid is not None else None
    for name, value in rewards.get("components", {}).items():
        flat["reward_" + name] = value
    selection = row.get("parsed_actions", trace.get("parsed_actions", {})).get("select")
    flat["action_emitted"] = selection.get("action") if isinstance(selection, dict) else None
    flat["slot"] = trace.get("asked_slot", selection.get("slot") if isinstance(selection, dict) else None)
    # Historical logs do not distinguish requested, emitted and delivered in
    # every branch. Do not turn a missing delivery event into a success flag.
    for name in ("ask_requested", "ask_emitted", "ask_delivered", "user_response_received", "response_informative"):
        flat[name] = row.get(name, trace.get(name, metrics.get(name)))
    flat["question_count_recorded"] = metrics.get("questions")
    flat["phase_errors"] = trace.get("errors", [])
    flat["raw_output_reference"] = {"path": flat["source_path"], "line": source_line,
                                    "line_sha256": line_sha256, "field": "raw_outputs"}
    flat["condition_factors"] = {k: flat.get(k) for k in CONDITION_FIELDS}
    flat["condition_id"] = "historical_v4_" + digest(flat["condition_factors"])
    return flat


def discover_trace_sources(parent_run):
    """Use the historical manifest; do not include training or merged copies."""
    parent_run = Path(parent_run).resolve()
    manifest = parent_run / "reports" / "ANALISIS.json"
    if manifest.exists():
        entries = json.loads(manifest.read_text()).get("source_traces", [])
        sources = []
        for entry in entries:
            recorded = Path(entry["path"])
            local = parent_run / "traces" / recorded.name
            path = local if local.exists() else recorded if recorded.is_absolute() else parent_run / recorded
            sources.append({"path": path, "expected_sha256": entry.get("sha256"),
                            "manifest_path": str(manifest), "recorded_path": str(recorded)})
        return sources
    return [{"path": p, "expected_sha256": None} for p in sorted((parent_run / "traces").glob("*.jsonl"))]


def _supplementary_csv(parent_run, output, source_name, target_name, run_id):
    source = parent_run / "reports" / source_name
    if not source.exists():
        return {"status": "blocked_missing_evidence", "source": str(source)}
    with source.open(newline="") as handle, (output / target_name).open("w", newline="") as target:
        reader = csv.DictReader(handle)
        fields = ["run_id", "evidence_origin", "source_path", "source_sha256"] + list(reader.fieldnames or [])
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        count, sha = 0, file_hash(source)
        for row in reader:
            writer.writerow({"run_id": run_id, "evidence_origin": "historical_report_not_new_training",
                             "source_path": str(source), "source_sha256": sha, **row})
            count += 1
    return {"status": "passed", "rows": count, "source": str(source), "sha256": sha,
            "limitation": "Missing historical telemetry stays missing; no new rollout or optimizer step."}


def reexport_results(parent_run: Path, output: Path, run_id: str) -> dict:
    """Stream historical JSONL; preserve collisions as failures, never delete rows.

    Memory audits reference the source line for full prompts and generations and
    contain actual before/after states and delivered evidence, without hidden gold.
    """
    parent_run, output = Path(parent_run).resolve(), Path(output).resolve()
    if output == parent_run or parent_run in output.parents:
        raise ValueError("Closure output must be outside the historical parent run")
    output.mkdir(parents=True, exist_ok=True)
    names = ("RESULTADOS_V4.csv", "MEMORIAS_AUDITADAS.jsonl", "PRESUPUESTOS_V4.csv", "CURVAS_RL_V4.csv", "EXPORTACION_V4.json")
    if any((output / name).exists() for name in names):
        raise FileExistsError("V4 export already exists; choose a fresh output directory")
    spool = output / ".closure_export_rows.jsonl"
    counts, legacy, fields = Counter(), Counter(), set()
    sources, failures, rows, memory_rows = [], [], 0, 0
    null_rates, rate_counts, missing_checkpoint, missing_critical_ids = 0, Counter(), 0, 0
    manifests = discover_trace_sources(parent_run)
    seen_sources = set()
    with spool.open("w") as normalized, (output / "MEMORIAS_AUDITADAS.jsonl").open("w") as audit:
        for entry in manifests:
            path = entry["path"].resolve()
            if path in seen_sources:
                failures.append({"path": str(path), "error": "duplicate_source_manifest_entry"})
                # Do not read a manifest alias twice; fail the source manifest.
                continue
            seen_sources.add(path)
            if not path.is_file():
                failures.append({"path": str(path), "error": "missing_source_trace"})
                continue
            source_hash, source_rows = hashlib.sha256(), 0
            with path.open("rb") as handle:
                for line_no, line in enumerate(handle, 1):
                    source_hash.update(line)
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError("episode must be an object")
                        flat = flatten_episode(row, run_id, source_path=path, source_line=line_no,
                                               line_sha256=hashlib.sha256(line).hexdigest())
                    except (ValueError, TypeError, AttributeError) as exc:
                        failures.append({"path": str(path), "line": line_no, "error": str(exc)})
                        continue
                    normalized.write(canonical(flat) + "\n")
                    fields.update(flat)
                    counts[tuple(canonical(flat.get(k)) for k in PRIMARY_KEY)] += 1
                    legacy[tuple(canonical(flat.get(k)) for k in LEGACY_KEY)] += 1
                    missing_checkpoint += flat["checkpoint_id"] is None
                    missing_critical_ids += any(flat.get(k) is None for k in ("method", "training_seed", "root_id", "latent_world_id", "realization_id"))
                    null_rates += flat["nonresponse_rate"] is None
                    rate_counts[canonical(flat["nonresponse_rate"])] += 1
                    rows += 1
                    source_rows += 1
                    trace = row.get("trace", {})
                    before, after = trace.get("initial_memory"), trace.get("final_memory")
                    if before is not None or after is not None:
                        evidence = trace.get("observations", {}).get("write")
                        audit_row = {k: flat.get(k) for k in PRIMARY_KEY}
                        audit_row.update(source_path=str(path), source_line=line_no,
                            source_line_sha256=flat["source_line_sha256"],
                            before=[_normal_record(r) for r in before] if isinstance(before, list) else None,
                            after=[_normal_record(r) for r in after] if isinstance(after, list) else None,
                            operations=trace.get("executed_operations", row.get("executed_operations")),
                            writer_visible_evidence=evidence,
                            user_reply_delivered=trace.get("user_reply", row.get("user_reply")),
                            semantic_support_status="not_reaudited_by_exporter",
                            initial_memory_hash=row.get("initial_memory_hash", trace.get("initial_memory_hash")),
                            final_memory_hash=row.get("final_memory_hash", trace.get("final_memory_hash")))
                        audit.write(canonical(audit_row) + "\n")
                        memory_rows += 1
            actual = source_hash.hexdigest()
            expected = entry.get("expected_sha256")
            verified = actual == expected if expected else None
            sources.append({"path": str(path), "sha256": actual, "expected_sha256": expected,
                            "verified": verified, "rows": source_rows})
            if verified is False:
                failures.append({"path": str(path), "error": "source_hash_mismatch", "expected": expected, "observed": actual})
    ordered_fields = list(PRIMARY_KEY) + sorted(fields - set(PRIMARY_KEY))
    with (output / "RESULTADOS_V4.csv").open("w", newline="") as target, spool.open() as normalized:
        writer = csv.DictWriter(target, fieldnames=ordered_fields)
        writer.writeheader()
        for line in normalized:
            flat = json.loads(line)
            writer.writerow({k: _cell(flat.get(k)) for k in ordered_fields})
    spool.unlink()
    duplicate_keys = sum(n > 1 for n in counts.values())
    duplicate_rows = sum(n for n in counts.values() if n > 1)
    supplements = {
        "budgets": _supplementary_csv(parent_run, output, "PRESUPUESTOS.csv", "PRESUPUESTOS_V4.csv", run_id),
        "training_curves": _supplementary_csv(parent_run, output, "CURVAS_ENTRENAMIENTO.csv", "CURVAS_RL_V4.csv", run_id)}
    if not manifests:
        failures.append({"error": "no_source_traces"})
    checks = [
        {"id": "EXP-01", "status": "passed" if sources and not failures else "blocked_missing_evidence" if not sources else "failed",
         "evidence": "EXPORTACION_V4.json", "description": "Historical trace provenance and source hashes", "failures": failures},
        {"id": "EXP-02", "status": "passed" if rows and not duplicate_keys and not missing_critical_ids else "failed" if rows else "blocked_missing_evidence",
         "evidence": "RESULTADOS_V4.csv", "description": "Full condition key, no deduplication", "duplicate_keys": duplicate_keys,
         "missing_critical_ids": missing_critical_ids},
        {"id": "EXP-03", "status": "passed" if rows else "blocked_missing_evidence",
         "evidence": "RESULTADOS_V4.csv", "description": "Explicit nonresponse_rate, original metrics and denominators retained"},
        {"id": "EXP-04", "status": "blocked_missing_evidence" if missing_checkpoint else "passed" if rows else "blocked_missing_evidence",
         "evidence": "RESULTADOS_V4.csv", "description": "Recorded checkpoint identity, not a new weights verification",
         "missing_checkpoint_rows": missing_checkpoint}]
    summary = {"status": "failed" if duplicate_keys or missing_critical_ids or failures and sources else "passed" if rows else "blocked_missing_evidence",
        "run_id": run_id, "parent_run": str(parent_run), "rows": rows, "memory_rows": memory_rows,
        "primary_key": list(PRIMARY_KEY), "condition_factors": list(CONDITION_FIELDS),
        "duplicate_primary_keys": duplicate_keys, "duplicate_primary_rows": duplicate_rows,
        "legacy_collision_keys": sum(n > 1 for n in legacy.values()),
        "legacy_collision_rows": sum(n for n in legacy.values() if n > 1),
        "nonresponse_rate_counts": dict(rate_counts), "null_nonresponse_rows": null_rates,
        "null_csv_semantics": "literal null = unrecorded/not applicable, never rate zero",
        "sources": sources, "checks": checks, "supplementary": supplements,
        "paths": {name: str(output / name) for name in names if (output / name).exists()},
        "human_audit": "not_executed", "limitations": [
            "Historical scores preserved; export does not certify evaluator semantics or current infrastructure.",
            "Missing historical parser/simulator versions and action delivery stages remain null.",
            "Checkpoint identifiers are recorded provenance, not runtime weight/logit identity checks.",
            "J_primary is newly derived per episode and is exploratory for historical data.",
            "Memory support is not verified merely by retaining evidence IDs."]}
    summary["paths"]["EXPORTACION_V4.json"] = str(output / "EXPORTACION_V4.json")
    write_json(output / "EXPORTACION_V4.json", summary)
    return summary
