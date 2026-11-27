"""Prospective competence evaluation, with selection confined to validation.

The actor receives only real environment observations. Production parsing,
scoring, rewards and SQLite are reused without repairs or partial task credit.
The observable rule teacher is a separately labelled positive control, not an
independent human gold or a learned actor. This module opens no global bank.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
from statistics import mean
import time

from .src.artifacts import canonical, digest, file_hash, now, write_json
from .src.env.episode import EpisodeEnvironment
from .src.policy.llm_controller import InvalidOutput, LLMController, validate_action
from .src.policy.rule_controller import RuleController
from .src.readers.symbolic import read
from .src.scoring.evidence import Evaluator, EVALUATOR_VERSION
from .src.scoring.reward import rewards

VERSION = "competence_evaluation_v1"
DEFAULT_THRESHOLDS = {
    "minimum_roots": 10,
    "select_format": .95,
    "select_decision": .70,
    "respond_format": .95,
    "C2": .80,
    "write_format": .95,
    "write_faithful_episode": .90,
    "FJS": .75,
}
SEEDS = (101, 202, 303)
SELECT_REFERENCE = "observable_rule_teacher_v5"


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _source_hashes(prompt_root):
    paths = [Path(__file__), *Path(__file__).parent.joinpath("src/env").glob("*.py"),
             *Path(__file__).parent.joinpath("src/scoring").glob("*.py"),
             *Path(__file__).parent.joinpath("src/memory").glob("*.py"),
             *Path(__file__).parent.joinpath("src/policy").glob("*.py"),
             *Path(__file__).parent.joinpath("src/readers").glob("*.py")]
    paths.extend(Path(__file__).parent.glob("competence_*.py"))
    paths.append(Path(__file__).resolve().parents[1] / "llm_local_001/model.py")
    if prompt_root is not None:
        paths.extend(Path(prompt_root).glob("*.txt"))
    return {str(p.resolve()): file_hash(p) for p in sorted(set(paths)) if p.is_file()}


def _controller(policy, prompt_root):
    if all(callable(getattr(policy, phase, None)) for phase in ("select", "respond", "write")):
        controller = policy
        if prompt_root is not None and hasattr(controller, "prompt_root") and Path(controller.prompt_root).resolve() != Path(prompt_root).resolve():
            raise ValueError("Controller prompt root differs from the predeclared prompt root")
    else:
        controller = LLMController(policy, prompt_root=prompt_root, sample=False)
    if getattr(controller, "sample", False):
        raise ValueError("Selection and evaluation require greedy decoding; stochastic trajectories cannot be reused")
    return controller


def _adapter_fingerprint(controller):
    backend = getattr(controller, "backend", controller)
    model = getattr(backend, "model", None)
    if model is None:
        return {"scope": "scripted_or_test_control_without_model", "sha256": None}
    hasher, count = hashlib.sha256(), 0
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            continue
        value = parameter.detach().cpu().contiguous()
        hasher.update(name.encode())
        hasher.update(str(value.dtype).encode())
        hasher.update(canonical(list(value.shape)).encode())
        hasher.update(value.view(backend.torch.uint8).numpy().tobytes())
        count += 1
    if not count:
        raise ValueError("A real competence actor must expose its integrated LoRA parameters")
    return {"scope": "loaded_LoRA_tensors_only; base identity is verified by the campaign manifest",
            "sha256": hasher.hexdigest(), "adapter_tensors": count}


def _phase(controller, phase, observation):
    records = getattr(controller, "phase_records", None)
    before = len(records) if records is not None else 0
    try:
        action = getattr(controller, phase)(deepcopy(observation))
        error = None
    except InvalidOutput as exc:
        action, error = {}, {"phase": phase, "kind": "invalid_output", "error": str(exc)}
    if error is None:
        try:
            validate_action(phase, action)
        except (ValueError, TypeError, KeyError) as exc:
            action, error = {}, {"phase": phase, "kind": "invalid_output", "error": str(exc)}
    if hasattr(controller, "serialized_records"):
        actual_records = controller.serialized_records()[before:]
    else:
        actual_records = _jsonable(getattr(controller, "phase_records", [])[before:])
    return {"action": deepcopy(action), "error": error, "records": actual_records,
            "observation_sha256": digest(observation), "source": "actor_output"}


def _verify_sqlite(path, trace):
    with sqlite3.connect("file:" + str(Path(path).resolve()) + "?mode=ro", uri=True) as connection:
        records = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM cm_records ORDER BY seq")]
    if records != trace["final_memory"] or records != trace["reuse_observation"]["memory"]:
        raise RuntimeError("Persisted SQLite state differs from the reader input")
    if set(trace["reuse_observation"]) != {"memory", "query"}:
        raise RuntimeError("Future reader received transient or latent state")
    return {"status": "verified", "sqlite_path": str(Path(path).resolve()), "sqlite_sha256": file_hash(path),
            "memory_sha256": digest(records), "reuse_input_keys": ["memory", "query"]}


def _run_trajectory(controller, episode, sqlite_path, *, condition="O2", route="natural",
                    fixed_response=None, method="B1c", training_seed=101):
    """Execute one immutable branch; fixed_response is an exact paired output."""
    if route not in {"natural", "PROCEED", "I1", "I2"}:
        raise ValueError("Unregistered competence route")
    if Path(sqlite_path).exists():
        raise FileExistsError("Do not overwrite an existing rollout SQLite file")
    backend = getattr(controller, "backend", None)
    if backend is not None and hasattr(backend, "seed"):
        backend.seed(training_seed)
    if hasattr(controller, "phase_records"):
        controller.phase_records.clear()
    environment = EpisodeEnvironment(episode, sqlite_path)
    started, phases = time.monotonic(), {}
    intervention = route if route in {"I1", "I2"} else None
    try:
        select_obs = environment.observe("select", condition, intervention)
        reference_select = RuleController().select(deepcopy(select_obs))
        if route == "natural":
            phases["select"] = _phase(controller, "select", select_obs)
        else:
            action = {"action": "PROCEED", "slot": None} if route == "PROCEED" else {"action": "ASK", "slot": "TEMPORAL_SCOPE"}
            phases["select"] = {"action": action, "error": None, "records": [],
                                "observation_sha256": digest(select_obs), "source": "predeclared_intervention"}
        action = phases["select"]["action"]
        if phases["select"]["error"] is None and action.get("action") == "ASK":
            environment.ask(action["slot"])
        respond_obs = environment.observe("respond", condition, intervention)
        if fixed_response is None:
            phases["respond"] = _phase(controller, "respond", respond_obs)
        else:
            if fixed_response["observation_sha256"] != digest(respond_obs):
                raise RuntimeError("PI2 requires byte-identical RESPOND observations")
            phases["respond"] = deepcopy(fixed_response)
            phases["respond"]["source"] = "exact_same_paired_response"
        response = phases["respond"]["action"]
        write_obs = environment.observe("write", condition, intervention)
        phases["write"] = _phase(controller, "write", write_obs)
        if phases["write"]["error"] is None:
            result = environment.apply_memory_operations(phases["write"]["action"]["operations"])
            if not result["valid"]:
                phases["write"]["error"] = {"phase": "write", "kind": "invalid_output", "error": result["error"]}
        questions = environment.question_count
        environment.close_session()
        reuse = environment.reopen_for_reuse()
        # Reader is stateless and sees persisted records plus its query only.
        if hasattr(controller, "clear_transient"):
            controller.clear_transient()
        elif hasattr(controller, "phase_records"):
            controller.phase_records.clear()
        future_response = read(deepcopy(reuse["memory"]), deepcopy(reuse["query"]))
        trace = deepcopy(environment.private_trace)
        trace.update(episode=deepcopy(episode), response=deepcopy(response), future_response=future_response,
                     interaction_observation={"memory": deepcopy(episode["initial_memory"]),
                         "user_messages": deepcopy(respond_obs["user_messages"]),
                         "query": deepcopy(episode["current_query"]), "reuse_regime": respond_obs["reuse_regime"]},
                     parsed_actions={p: deepcopy(v["action"]) for p, v in phases.items()},
                     errors=[v["error"] for v in phases.values() if v["error"]],
                     phase_records=[r for v in phases.values() for r in v["records"]])
        scorer = Evaluator()
        current, memory, future = scorer.score_current(trace), scorer.score_memory(trace), scorer.score_future(trace)
        invalid = sum(v["error"] is not None for v in phases.values())
        reward = rewards(current["utility"], future["utility"], questions, memory["unsupported_operations"], invalid)
        metrics = {"select_format": int(phases["select"]["error"] is None),
                   "select_decision": int(phases["select"]["error"] is None and action == reference_select),
                   "select_actor_evaluated": int(route == "natural"),
                   "respond_format": int(phases["respond"]["error"] is None),
                   "respond_value_correct": current["factual"],
                   "respond_citations_supported": int(current["justified"]),
                   "respond_success": current["success"], "C2": current["success"],
                   "current_incorrect_or_unsupported": int(current["utility"] < 0),
                   "unsupported_current": current["unsupported_commitment"],
                   "write_format": int(phases["write"]["error"] is None),
                   "write_faithful_episode": int(phases["write"]["error"] is None and memory["unsupported_operations"] == 0),
                   "unsupported_write_episode": int(memory["unsupported_operations"] > 0),
                   "unsupported_write_count": memory["unsupported_operations"],
                   "write_operation_count": memory["operations"],
                   "FJS": future["success"], "FJS_primary": future["success"],
                   "J_primary": current["success"] * future["success"],
                   "future_invalid": int(not future["valid"]),
                   "questions": questions, "invalid_output_count": invalid, "any_invalid_output": int(invalid > 0),
                   "invalid_output_rate": invalid / 3., "R_long": reward["R_long"], "R_short": reward["R_short"],
                   "utility": reward["R_long"], "unsupported_writes": int(memory["unsupported_operations"] > 0),
                   "future_success": future["success"], "current_success": current["success"]}
        metrics["phase_format"] = mean(metrics[key] for key in ("select_format", "respond_format", "write_format"))
        metrics["selection_competence_utility"] = mean(metrics[key] for key in ("C2", "FJS", "write_faithful_episode", "phase_format"))
        row = {"version": VERSION, "evaluator_version": EVALUATOR_VERSION, "method": method,
               "training_seed": training_seed, "seed": training_seed,
               **{key: episode[key] for key in ("root_id", "latent_world_id", "realization_id", "split", "scenario_family")},
               "condition": condition, "route": route, "reader_id": "fixed_symbolic",
               "pi2_cohort": ("eligible_ambiguous" if episode.get("opportunity", {}).get("eligible") else
                              "unaffected_attribute" if episode.get("authoring", {}).get("stratum") == "unrelated_attribute_control" else
                              "clear_control"),
               "select_reference_source": SELECT_REFERENCE, "select_gold_independent": False,
               "reference_select": reference_select, "metrics": metrics, "phases": phases,
               "scoring": {"current": current, "memory": memory, "future": future},
               "rewards_by_component": reward, "trace": trace, "technical_status": "completed",
               "elapsed_seconds": time.monotonic() - started}
        environment.dispose()
        row["persistence_verification"] = _verify_sqlite(sqlite_path, trace)
        return row
    except Exception as exc:
        records = (controller.serialized_records() if hasattr(controller, "serialized_records")
                   else _jsonable(getattr(controller, "phase_records", [])))
        write_json(Path(sqlite_path).with_suffix(".partial.json"), _jsonable({
            "status": "technical_failure", "method": method, "training_seed": training_seed,
            "condition": condition, "route": route, "episode": episode,
            "trace_at_failure": deepcopy(environment.private_trace), "completed_phases": phases,
            "actor_phase_records_at_failure": records, "error": f"{type(exc).__name__}: {exc}",
            "scores": None, "retry_count": 0}))
        raise
    finally:
        environment.dispose()


def aggregate_rows(rows, *, route="natural", condition="O2"):
    """Realizations within world, worlds within root, equally weighted families."""
    selected = [r for r in rows if r["route"] == route and r["condition"] == condition]
    if not selected:
        return {"episodes": 0, "roots": 0, "metrics": {}, "by_root": {}, "by_family": {}}
    worlds, families, identities = defaultdict(lambda: defaultdict(dict)), {}, set()
    for row in selected:
        key = (row["root_id"], row["latent_world_id"], row["realization_id"])
        if key in identities:
            raise ValueError("Duplicate episode within a condition; separate training seeds before aggregation")
        identities.add(key)
        rid, world, realization = key
        if rid in families and families[rid] != row["scenario_family"]:
            raise ValueError("A root cannot change family")
        families[rid] = row["scenario_family"]
        worlds[rid][world][realization] = row["metrics"]
    metrics = sorted(set.intersection(*(set(r["metrics"]) for r in selected)))
    by_root = {rid: {m: mean(mean(v[m] for v in realizations.values()) for realizations in world.values()) for m in metrics}
               for rid, world in worlds.items()}
    by_family = {f: {m: mean(v[m] for rid, v in by_root.items() if families[rid] == f) for m in metrics}
                 for f in sorted(set(families.values()))}
    return {"episodes": len(selected), "roots": len(by_root), "metrics": {m: mean(v[m] for v in by_family.values()) for m in metrics},
            "root_weighted": {m: mean(v[m] for v in by_root.values()) for m in metrics},
            "by_root": by_root, "by_family": by_family, "families": families,
            "aggregation": "realizations_then_worlds_then_roots_then_equal_families"}


def capability_gate(summary, thresholds=None):
    """A preregistered validation gate, never an evaluation-set selector."""
    if summary.get("role") != "validation":
        return {"status": "not_applicable", "passed": False, "reason": "Capability selection is validation-only"}
    thresholds = deepcopy(thresholds or summary.get("thresholds", DEFAULT_THRESHOLDS))
    preregistered = summary.get("thresholds")
    if preregistered is None or thresholds != preregistered:
        raise ValueError("Capability thresholds must equal those frozen before generation")
    metrics, roots = summary.get("metrics", {}), summary.get("roots", 0)
    checks = []
    for metric, cutoff in thresholds.items():
        if type(cutoff) not in {int, float} or not math.isfinite(cutoff):
            raise ValueError("Invalid capability threshold")
        observed = roots if metric == "minimum_roots" else metrics.get(metric)
        checks.append({"metric": metric, "minimum": cutoff, "observed": observed,
                       "passed": observed is not None and observed >= cutoff})
    completed = summary.get("status") == "completed" and summary.get("completed_episodes") == summary.get("scheduled_episodes")
    passed = completed and bool(checks) and all(c["passed"] for c in checks)
    return {"status": "passed" if passed else "failed" if completed else "blocked_missing_evidence", "passed": passed,
            "role": "validation", "checks": checks, "thresholds": thresholds,
            "research_status": "exploratory", "data_sha256": summary.get("bank_content_sha256"),
            "protocol_sha256": summary.get("protocol_sha256"), "observed_root_count": roots,
            "interpretation": "Operational capability under this synthetic validation bank, not a general correctness or human-validity certificate",
            "select_reference_source": SELECT_REFERENCE, "select_gold_independent": False}


def _validate_heldback_authorization(protocol, episodes, controller, method, training_seed):
    authorization = protocol.get("heldback_authorization", {})
    selection = Path(str(authorization.get("selection_manifest_path", "")))
    if not selection.is_file() or file_hash(selection) != authorization.get("selection_manifest_sha256"):
        raise ValueError("Heldback evaluation requires unchanged validation-selection evidence")
    if authorization.get("locked_bank_sha256") != digest(episodes):
        raise ValueError("Heldback episode bank differs from the authorized frozen content")
    manifest = json.loads(selection.read_text())
    if (manifest.get("version") != "competence_selection_v1" or manifest.get("selection_complete") is not True
            or manifest.get("selection_split") != "validation" or manifest.get("selected_before_evaluation") is not True
            or manifest.get("heldback_episodes_sha256") != digest(episodes)):
        raise ValueError("Heldback requires a completed validation-only selection manifest bound to these episodes")
    matches = [r for r in manifest.get("rows", []) if r.get("method") == method and r.get("training_seed") == training_seed]
    if len(matches) != 1:
        raise ValueError("Heldback policy has no unique validation-selected method/seed record")
    selected = matches[0]
    backend = getattr(controller, "backend", controller)
    origin = getattr(backend, "origin_checkpoint", None)
    if origin is None:
        if protocol.get("test_only") is not True or not selected.get("fake_actor_fingerprint") or selected["fake_actor_fingerprint"] != protocol.get("fake_actor_fingerprint"):
            raise ValueError("Heldback actor lacks a verifiable selected checkpoint")
        return
    checkpoint = Path(origin).resolve()
    if checkpoint != Path(selected["checkpoint_path"]).resolve():
        raise ValueError("Loaded actor differs from the validation-selected checkpoint")
    if (file_hash(checkpoint / "local_checkpoint.json") != selected["checkpoint_manifest_sha256"]
            or file_hash(checkpoint / "adapter_model.safetensors") != selected["adapter_sha256"]):
        raise ValueError("Selected checkpoint artifacts changed")
    if any(p.requires_grad for p in backend.model.parameters()):
        raise ValueError("Heldback evaluation must load frozen selected weights")


class _RecordedTeacher:
    """Execute supplied observable teacher records only on identical inputs."""
    def __init__(self, records):
        self.expected = {record["phase"]: deepcopy(record) for record in records}
        if set(self.expected) != {"select", "respond", "write"}:
            raise ValueError("Teacher factory must supply SELECT/RESPOND/WRITE")
        self.phase_records = []

    def _call(self, phase, observation):
        expected = self.expected[phase]
        if expected["observation"] != observation:
            raise RuntimeError("Teacher control observation differs from its independently executed environment route")
        action = deepcopy(expected["action"])
        self.phase_records.append({"phase": phase, "raw_output": expected.get("completion", canonical(action)),
                                   "parsed_action": action, "source": expected.get("source", SELECT_REFERENCE),
                                   "parse_status": "passed", "schema_status": "passed", "truncated": False,
                                   "fallback_used": False, "fallback_reason": None})
        return action

    def select(self, observation):
        return self._call("select", observation)

    def respond(self, observation):
        return self._call("respond", observation)

    def write(self, observation):
        return self._call("write", observation)


def evaluate_selected(policy, episodes, output, *, prompt_root=None, method="B1c", training_seed=101,
                      role="validation", protocol=None, include_interventions=True, include_teacher=True,
                      teacher_factory=None):
    """Evaluate one already-selected policy without opening any bank itself.

    Training callbacks should disable intervention/teacher panels. Final panels
    require validation-selection evidence and a frozen heldback content hash.
    Any technical failure stops generation, preserves partial rows and blocks the
    capability gate; malformed model outputs remain scored episodes, with no retry.
    """
    episodes = list(episodes)
    if not episodes:
        raise ValueError("A nonempty explicitly supplied episode bank is required")
    if role not in {"validation", "evaluation"}:
        raise ValueError("role must be validation or evaluation")
    splits = {e.get("split") for e in episodes}
    if role == "validation" and splits != {"validation"}:
        raise ValueError("Selection/capability evaluation requires validation-only episodes")
    if role == "evaluation" and splits != {"evaluation"}:
        raise ValueError("Heldback evaluation received a development or training episode")
    identities = [tuple(e[k] for k in ("root_id", "latent_world_id", "realization_id")) for e in episodes]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate episode identity in supplied bank")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Competence evaluation requires a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    protocol = deepcopy(protocol or {})
    if "thresholds" not in protocol:
        raise ValueError("Explicit capability thresholds are required in the predeclared protocol")
    thresholds = deepcopy(protocol["thresholds"])
    if not set(DEFAULT_THRESHOLDS) <= set(thresholds):
        raise ValueError("All capability thresholds must be predeclared")
    for name, value in thresholds.items():
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise ValueError("Malformed capability threshold: " + name)
        if name != "minimum_roots" and value > 1:
            raise ValueError("Proportion threshold must be within [0,1]")
    controller = _controller(policy, prompt_root)
    if role == "evaluation":
        _validate_heldback_authorization(protocol, episodes, controller, method, training_seed)
    actor_before = _adapter_fingerprint(controller)
    actual_prompt_root = prompt_root or getattr(controller, "prompt_root", None)
    predeclared = {"version": VERSION, "created_at": now(), "role": role, "method": method,
                   "training_seed": training_seed, "thresholds": thresholds,
                   "bank_content_sha256": digest(episodes), "source_files_sha256": _source_hashes(actual_prompt_root),
                   "protocol": protocol, "include_interventions": include_interventions,
                   "actor_before": actor_before, "test_only": protocol.get("test_only", False),
                   "include_teacher": include_teacher, "decoding": "greedy",
                   "selection_reference": SELECT_REFERENCE,
                   "technical_failure_rule": "stop_without_retry_preserve_partial_rows_and_block_gate",
                   "PI1": "forced PROCEED O1/O2; format/scorer/query fixed",
                   "PI2": "fixed ASK TEMPORAL_SCOPE; one identical RESPOND output for I1/I2; writer alone loses/gains reply"}
    write_json(output / "predeclaration.json", predeclared)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in predeclared["source_files_sha256"]:
        source = Path(path)
        try:
            relative = source.relative_to(Path(__file__).resolve().parents[1])
        except ValueError:
            relative = Path("external_sources") / (digest(str(source))[:16] + "_" + source.name)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    states = output / "states"
    states.mkdir()
    rows, teachers, failure = [], [], None
    scheduled = len(episodes) * (5 if include_interventions else 1)
    with (output / "trajectories.jsonl").open("x") as stream, (output / "teacher_controls.jsonl").open("x") as teacher_stream:
        try:
            for index, episode in enumerate(episodes):
                natural = _run_trajectory(controller, episode, states / f"{index:05d}_natural.sqlite", method=method, training_seed=training_seed)
                natural["evaluation_role"] = role
                rows.append(natural)
                stream.write(canonical(_jsonable(natural)) + "\n")
                stream.flush()
                if include_interventions:
                    for condition in ("O1", "O2"):
                        row = _run_trajectory(controller, episode, states / f"{index:05d}_pi1_{condition}.sqlite", condition=condition,
                                              route="PROCEED", method=method, training_seed=training_seed)
                        row["evaluation_role"] = role
                        rows.append(row)
                        stream.write(canonical(_jsonable(row)) + "\n")
                        stream.flush()
                    first = _run_trajectory(controller, episode, states / f"{index:05d}_pi2_I1.sqlite", route="I1", method=method, training_seed=training_seed)
                    second = _run_trajectory(controller, episode, states / f"{index:05d}_pi2_I2.sqlite", route="I2",
                                             fixed_response=first["phases"]["respond"], method=method, training_seed=training_seed)
                    if first["trace"]["response"] != second["trace"]["response"] or first["metrics"]["C2"] != second["metrics"]["C2"]:
                        raise RuntimeError("PI2 changed the current response instead of only writer information")
                    for row in (first, second):
                        row["evaluation_role"] = role
                        row["paired_response_sha256"] = digest(first["phases"]["respond"])
                        rows.append(row)
                        stream.write(canonical(_jsonable(row)) + "\n")
                    stream.flush()
                if include_teacher:
                    teacher_controller = (_RecordedTeacher(teacher_factory(episode, condition="O2", route="natural"))
                                          if teacher_factory is not None else RuleController())
                    teacher = _run_trajectory(teacher_controller, episode, states / f"{index:05d}_teacher.sqlite",
                                              method="observable_rule_teacher_control", training_seed=training_seed)
                    teacher["positive_control_only"] = True
                    teacher["evaluation_role"] = role
                    teacher["independent_gold"] = False
                    teachers.append(teacher)
                    teacher_stream.write(canonical(_jsonable(teacher)) + "\n")
                    teacher_stream.flush()
        except Exception as exc:
            failure = {"type": type(exc).__name__, "error": str(exc), "completed_episodes": len(rows),
                       "no_retries": True, "partial_evidence_preserved": True}
    actor_after = _adapter_fingerprint(controller)
    if actor_before != actor_after:
        failure = {"type": "WeightMutation", "error": "Adapter tensor contents changed during frozen evaluation",
                   "partial_evidence_preserved": True}
    natural_summary = aggregate_rows(rows)
    summary = {"status": "completed" if failure is None and len(rows) == scheduled else "failed", "version": VERSION,
               "role": role, "method": method, "training_seed": training_seed,
               "scheduled_episodes": scheduled, "completed_episodes": len(rows), "natural_scheduled_episodes": len(episodes),
               "protocol_sha256": file_hash(output / "predeclaration.json"), "thresholds": thresholds,
               "bank_content_sha256": digest(episodes),
               "actor_before": actor_before, "actor_after": actor_after,
               "weights_unchanged": actor_before == actor_after,
               **natural_summary, "panels": {"natural": natural_summary}, "failure": failure,
               "teacher_control": aggregate_rows(teachers), "teacher_source": SELECT_REFERENCE,
               "teacher_independent_gold": False, "human_audit": "not_executed", "data_scope": "synthetic",
               "selection_eligible": role == "validation", "partial_credit": False,
               "paths": {"trajectories": str(output / "trajectories.jsonl"), "predeclaration": str(output / "predeclaration.json")},
               "limitations": ["SELECT agreement uses an observable rule teacher, not independent human decision labels.",
                               "Separate factual/citation diagnostics do not award partial task reward.",
                               "Faithful NOOP can still omit a useful update; FJS and operation count remain separate.",
                               "Symbolic future reader measures persisted-memory functionality, not LLM reader competence.",
                               "Heldback results are descriptive synthetic evidence; no external validity is certified."]}
    if include_interventions:
        summary["panels"].update({"PI1_" + c: aggregate_rows(rows, route="PROCEED", condition=c) for c in ("O1", "O2")})
        summary["panels"].update({"PI2_" + r: aggregate_rows(rows, route=r) for r in ("I1", "I2")})
    gate = capability_gate(summary)
    summary["capability_gate"] = gate
    write_json(output / "capability_gate.json", gate)
    write_json(output / "summary.json", summary)
    metric_fields = sorted({key for row in rows for key in row["metrics"]})
    with (output / "phase_metrics.csv").open("w", newline="") as handle:
        id_fields = ["method", "training_seed", "root_id", "latent_world_id", "realization_id", "split", "scenario_family", "condition", "route"]
        writer = csv.DictWriter(handle, fieldnames=id_fields + metric_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{key: row[key] for key in id_fields}, **row["metrics"]})
    completion = {"status": summary["status"], "role": role, "training": False,
                  "selection_eligible": role == "validation", "artifacts_sha256": {
                      str(p.relative_to(output)): file_hash(p) for p in output.rglob("*") if p.is_file()},
                  "new_actor_output_repairs": 0, "human_audit": "not_executed"}
    write_json(output / "completion.json", completion)
    return summary


def descriptive_contrasts(rows):
    """Five PI contrasts and a parent control; incomplete never becomes zero."""
    rows = list(rows)
    specifications = [
        ("PI1_B1c_O2_minus_O1", "C2", "B1c", "PROCEED", "O2", "B1c", "PROCEED", "O1"),
        ("PI2_B1c_I2_minus_I1", "FJS", "B1c", "I2", "O2", "B1c", "I1", "O2"),
        ("PI3_RLL_minus_moreSFT", "J_primary", "B2-L", "natural", "O2", "B3", "natural", "O2"),
        ("PI3_RLL_minus_RLI", "J_primary", "B2-L", "natural", "O2", "B2-I", "natural", "O2"),
        ("PI3_RLL_minus_B1c", "J_primary", "B2-L", "natural", "O2", "B1c", "natural", "O2"),
        ("COMPETENCE_B1c_minus_parent", "J_primary", "B1c", "natural", "O2", "B1-parent", "natural", "O2"),
    ]
    results = []
    for name, metric, lm, lr, lc, rm, rr, rc in specifications:
        maps = []
        error = None
        for method, route, condition in ((lm, lr, lc), (rm, rr, rc)):
            selected = [r for r in rows if (r["method"], r["route"], r["condition"]) == (method, route, condition)]
            identity = {(r["training_seed"], r["root_id"], r["latent_world_id"], r["realization_id"]) for r in selected}
            if len(identity) != len(selected):
                error = "duplicate episode identities"
            if any(r.get("evaluation_role") != "evaluation" for r in selected):
                error = "Final descriptive contrasts require heldback evaluation rows, not validation-selection outputs"
            maps.append({(r["training_seed"], r["root_id"], r["latent_world_id"], r["realization_id"]): r for r in selected})
        result = {"contrast": name, "metric": metric, "scope": "descriptive_synthetic_only", "confirmatory": False}
        if not maps[0] or maps[0].keys() != maps[1].keys() or error:
            result.update(status="incomplete", reason=error or "Missing fully paired worlds/realizations; no intersection-only estimate")
            results.append(result)
            continue
        seeds = {k[0] for k in maps[0]}
        root_profiles = defaultdict(set)
        for seed, rid, world, realization in maps[0]:
            root_profiles[(rid, world, realization)].add(seed)
        if seeds != set(SEEDS) or any(profile != set(SEEDS) for profile in root_profiles.values()):
            result.update(status="incomplete", reason="Three complete training seeds 101,202,303 required")
            results.append(result)
            continue
        differences = []
        for key, left in maps[0].items():
            right = maps[1][key]
            if left["scenario_family"] != right["scenario_family"]:
                raise ValueError("Paired family changed")
            value = left["metrics"][metric] - right["metrics"][metric]
            differences.append({**left, "metrics": {"difference": value}, "route": "contrast", "condition": "paired"})
        per_seed = {str(seed): aggregate_rows([r for r in differences if r["training_seed"] == seed], route="contrast", condition="paired") for seed in SEEDS}
        estimate = mean(s["metrics"]["difference"] for s in per_seed.values())
        result.update(status="completed", estimate=estimate, per_seed={seed: s["metrics"]["difference"] for seed, s in per_seed.items()},
                      per_family={f: mean(s["by_family"][f]["difference"] for s in per_seed.values()) for f in next(iter(per_seed.values()))["by_family"]},
                      roots=len(next(iter(per_seed.values()))["by_root"]),
                      outcome="positive_descriptive" if estimate > 0 else "negative_descriptive" if estimate < 0 else "no_difference_observed",
                      guards="not_evaluated_by_descriptive_contrast; no global advantage claim")
        if name.startswith("PI2_"):
            result["pooled_cohorts_diagnostic_only"] = {key: result[key] for key in ("estimate", "per_seed", "per_family", "roots")}
            result["by_cohort"] = {}
            for cohort in sorted({r.get("pi2_cohort", "unclassified") for r in differences}):
                entries = [r for r in differences if r.get("pi2_cohort", "unclassified") == cohort]
                summaries = {str(seed): aggregate_rows([r for r in entries if r["training_seed"] == seed], route="contrast", condition="paired") for seed in SEEDS}
                result["by_cohort"][cohort] = {
                    "estimate": mean(v["metrics"]["difference"] for v in summaries.values()),
                    "per_seed": {seed: v["metrics"]["difference"] for seed, v in summaries.items()},
                    "roots": len(next(iter(summaries.values()))["by_root"]),
                    "per_family": {f: mean(v["by_family"][f]["difference"] for v in summaries.values())
                                   for f in next(iter(summaries.values()))["by_family"]}}
            result["primary_cohort"] = "eligible_ambiguous"
            primary = result["by_cohort"].get("eligible_ambiguous")
            if primary is None:
                result.update(status="incomplete", estimate=None, reason="No preregistered eligible ambiguous cohort for PI2")
            else:
                result.update(primary)
                result["outcome"] = "positive_descriptive" if primary["estimate"] > 0 else "negative_descriptive" if primary["estimate"] < 0 else "no_difference_observed"
            result["cohort_limitation"] = "Cohorts are declared by generator metadata before outcomes; overlapping roots are not summed as distinct users."
        results.append(result)
    return {"status": "completed" if all(r["status"] == "completed" for r in results) else "incomplete", "contrasts": results,
            "human_audit": "not_executed", "confirmation": "not_claimed"}
