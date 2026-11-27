"""Long-running, frozen exploratory campaign with auditable completed-unit reuse.

This module never prepares or authorizes an experiment. ``run_full(context)``
requires the prepared G2 context and executes all later stages without prompt
selection, reward changes or test-time optimization. Human work remains absent.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import gc
import json
from pathlib import Path
import random
import time

from .artifacts import canonical, digest, file_hash, load_config, now
from .protocol import require_gate
from .training.sft import _write_json, parameter_hash

METHODS = ("B0", "B0-P", "B1", "B2-L", "B3", "B2-I", "B4", "C-NEVER", "C-ALWAYS")
BASE_METHODS = frozenset({"B0", "B0-P", "B4"})
TRAINED_METHODS = ("B1", "B2-L", "B3", "B2-I")
SEEDS = (101, 202, 303)


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def semantic_config(config):
    """The preregistered sample-size-only lock may occur between campaign stages."""
    result = deepcopy(config)
    result.setdefault("sample", {})["test_id_locked_roots"] = None
    return result


def episode_identity(row):
    return tuple(row[key] for key in ("root_id", "latent_world_id", "realization_id"))


def power_report_for_lock(report):
    """Bridge the richer five-contrast planning report to the gate's old schema."""
    executed = [contrast for contrast in report["contrasts"] if contrast["status"] == "executed"]
    if not executed:
        raise RuntimeError("NO_EXECUTED_VALIDATION_POWER_SCENARIO")
    grouped = defaultdict(list)
    for contrast in executed:
        for scenario in contrast["scenarios"]:
            grouped[(scenario["candidate_total_id_roots"], scenario["variance_multiplier"])].append(scenario["estimated_power"])
    return {"status": "planning_only", "method": "paired_root_and_seed_resampling_normal_test",
            "input_split": "validation", "scenarios": [
                {"roots": roots, "variance_scale": multiplier, "estimated_power": min(values),
                 "aggregation": "minimum over executed prespecified contrasts", "executed_contrasts": len(values)}
                for (roots, multiplier), values in sorted(grouped.items())],
            "native_five_contrast_report": report, "input_split_assertion": "validation_only",
            "precision_limited": report["precision_limited"],
            "choice_reason": "Fixed resource-feasible 300 roots before test; no claim of sufficient power and no outcome-driven extension"}


def plan_secondary_cases(episodes, robustness_root_ids, *, seed=4107):
    """Select controls using construction metadata before any test actor runs."""
    families = {episode["root_id"]: episode["scenario_family"] for episode in episodes}
    robust = list(robustness_root_ids)
    if len(robust) != 100 or len(set(robust)) != 100 or not set(robust) <= families.keys():
        raise ValueError("Exactly 100 distinct preregistered ID robustness roots are required")
    if any(sum(families[root] == family for root in robust) != 20 for family in ("F0", "F1", "F2", "F3", "F4")):
        raise ValueError("Robustness roots must remain balanced across the five families")
    candidates = defaultdict(set)
    eligible = set()
    for episode in episodes:
        opportunity = episode.get("opportunity", {})
        if opportunity.get("eligible"):
            eligible.add(episode["root_id"])
        if opportunity.get("negative_control"):
            candidates[episode["scenario_family"]].add(episode["root_id"])
    rng = random.Random(seed)
    controls = []
    for family in ("F0", "F1", "F2", "F4"):
        if len(candidates[family]) < 5:
            raise ValueError("At least five clear negative-control roots are required per declared family")
        controls.extend(rng.sample(sorted(candidates[family]), 5))
    return {"status": "preregistered", "selected_before_test": True,
            "robustness_root_ids": robust, "eligible_f4_root_ids": sorted(eligible),
            "negative_control_root_ids": controls, "negative_control_families": {f: 5 for f in ("F0", "F1", "F2", "F4")},
            "controlled_question_slot": "TEMPORAL_SCOPE", "nonresponse_rates": [.2, .4],
            "unaffected_attribute_root_ids": sorted(families)[:20],
            "unaffected_attribute_controls": "derived drink_preference queries retain parent root IDs",
            "reader": "same_frozen_base_for_all_methods", "seeds": list(SEEDS),
            "external_status": "AI_AUTHORED_PROXY_not_human", "selection_seed": seed}


def verify_intervention_invariants(paths):
    """Check actual phase bytes; do not erase information from final memories."""
    arms = {arm: {episode_identity(row): row for row in read_rows(path)} for arm, path in paths.items()}
    if set(arms) != {"I0", "I1", "I2"} or any(arms[arm].keys() != arms["I0"].keys() for arm in arms):
        raise RuntimeError("INTERVENTION_PAIR_MATRIX_INCOMPLETE")
    counts = {"episodes": 0, "respond_I1_I2_identical": 0, "write_I0_I1_identical": 0,
              "respond_outputs_identical": 0}
    effects = []
    for identity in sorted(arms["I0"]):
        rows = {arm: arms[arm][identity] for arm in arms}
        for phase, left, right, label in (("respond", "I1", "I2", "respond_I1_I2_identical"),
                                         ("write", "I0", "I1", "write_I0_I1_identical")):
            observations = [rows[arm]["trace"]["observations"][phase] for arm in (left, right)]
            if canonical(observations[0]) != canonical(observations[1]):
                raise RuntimeError("INTERVENTION_INFORMATION_BOUNDARY_FAILED: " + label)
            records = [{r["phase"]: r for r in rows[arm]["trace"]["phase_records"]} for arm in (left, right)]
            if phase in records[0] and phase in records[1]:
                if records[0][phase]["prompt"] != records[1][phase]["prompt"]:
                    raise RuntimeError("INTERVENTION_PROMPT_BYTES_DIFFER: " + label)
                first = records[0][phase]["generation"]["prompt_ids"]
                second = records[1][phase]["generation"]["prompt_ids"]
                if first != second:
                    raise RuntimeError("INTERVENTION_PROMPT_TOKENS_DIFFER: " + label)
            counts[label] += 1
        counts["respond_outputs_identical"] += int(rows["I1"]["parsed_actions"]["respond"] == rows["I2"]["parsed_actions"]["respond"])
        counts["episodes"] += 1
        effects.append({"root_id": identity[0], "latent_world_id": identity[1], "realization_id": identity[2],
                        "negative_control": rows["I0"].get("negative_control", False),
                        "Delta_actual": rows["I1"]["metrics"]["C2"] - rows["I0"]["metrics"]["C2"],
                        "Delta_persistencia": rows["I2"]["metrics"]["FJS"] - rows["I1"]["metrics"]["FJS"],
                        "Delta_total_futuro": rows["I2"]["metrics"]["FJS"] - rows["I0"]["metrics"]["FJS"],
                        "Delta_utilidad": rows["I2"]["rewards_by_component"]["R_long"] - rows["I0"]["rewards_by_component"]["R_long"]})
    return {"status": "passed", "invariants": counts, "effects": effects}


class FullCampaign:
    def __init__(self, context):
        self.context = deepcopy(context)
        self.root = Path(context["root"]).resolve()
        self.destination = Path(context["destination"]).resolve()
        if not self.destination.is_relative_to(self.root):
            raise ValueError("Full campaign artifacts must remain under research for gate evidence")
        self.config = context["config"]
        if self.config.get("execution_mode") != "exploratory" or context.get("research_status") != "exploratory":
            raise ValueError("Explicit prepared exploratory context is required")
        self.paths = {name: Path(path).resolve() for name, path in context["paths"].items()}
        self.model_path = Path(context["model_path"]).resolve()
        self.batch_size = self.config.get("runtime", {}).get("evaluation_batch_size", 8)
        self.device = self.config.get("runtime", {}).get("device", "cuda")
        self.started = time.monotonic()
        self.active_stage = None
        self.trace_paths = []
        self.common_inputs = {"protocol_semantic_hash": digest(semantic_config(self.config)),
                              "executor_source_sha256": file_hash(Path(__file__))}
        self.destination.mkdir(parents=True, exist_ok=True)
        (self.destination / "stages").mkdir(exist_ok=True)
        (self.destination / "traces").mkdir(exist_ok=True)

    def emit(self, stage, status="running", **details):
        row = {"stage": stage, "status": status, "updated_at": now(),
               "elapsed_seconds": time.monotonic() - self.started,
               "research_status": "exploratory", "human_audit_status": "not_executed", **details}
        _write_json(self.destination / "status.json", row)
        with (self.destination / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical(row) + "\n")
            stream.flush()
        print(canonical({"campaign_progress": row}), flush=True)

    def stage(self, name, function, *, inputs=None):
        self.active_stage = name
        marker = self.destination / "stages" / (name + ".json")
        declared = {**self.common_inputs, **(inputs or {})}
        if marker.exists():
            previous = load_config(marker)
            if previous.get("status") == "executed":
                if previous.get("inputs") != declared:
                    raise RuntimeError("COMPLETED_STAGE_SOURCE_CHANGED: " + name)
                for path, expected in previous.get("artifacts", {}).items():
                    if not Path(path).is_file() or file_hash(path) != expected:
                        raise RuntimeError("COMPLETED_STAGE_ARTIFACT_CHANGED: " + path)
                self.emit(name, "cached_completed", result=previous["result"])
                return previous["result"]
        self.emit(name)
        _write_json(marker, {"status": "running", "inputs": declared, "started_at": now()})
        started = time.monotonic()
        try:
            result = function()
            files = [Path(path).resolve() for path in result.get("artifacts", [])]
            hashes = {str(path): file_hash(path) for path in files}
            _write_json(marker, {"status": "executed", "inputs": declared, "result": result,
                                "artifacts": hashes, "elapsed_seconds": time.monotonic() - started,
                                "completed_at": now()})
            self.emit(name, "executed", elapsed_stage_seconds=time.monotonic() - started, artifacts=list(hashes))
            return result
        except BaseException as exc:
            _write_json(marker, {"status": "failed", "inputs": declared, "error_type": type(exc).__name__,
                                "error": str(exc), "failed_at": now(), "partial_outputs_preserved": True})
            self.emit(name, "failed", error_type=type(exc).__name__, error=str(exc))
            raise

    def unit_path(self, name):
        return self.destination / "traces" / (name + ".jsonl")

    def trace_unit(self, name, iterator_factory, *, inputs, expected_rows=None):
        path = self.unit_path(name)
        marker = path.with_suffix(".complete.json")
        declared = {**self.common_inputs, **inputs}
        if marker.exists():
            previous = load_config(marker)
            if previous.get("status") != "executed" or previous.get("inputs") != declared:
                raise RuntimeError("COMPLETED_TRACE_UNIT_SOURCE_CHANGED: " + name)
            if not path.is_file() or file_hash(path) != previous["trace_sha256"]:
                raise RuntimeError("COMPLETED_TRACE_UNIT_CHANGED: " + name)
            if expected_rows is not None and previous["rows"] != expected_rows:
                raise RuntimeError("COMPLETED_TRACE_UNIT_DENOMINATOR_CHANGED: " + name)
            self.emit(self.active_stage, unit=name, cached_completed=True, episodes=previous["rows"])
            return path
        if path.exists():
            raise RuntimeError("PARTIAL_TRACE_REQUIRES_EXPLICIT_RECOVERY: " + str(path))
        started, count, last_update = time.monotonic(), 0, time.monotonic()
        with path.open("x", encoding="utf-8") as stream:
            for count, row in enumerate(iterator_factory(), start=1):
                stream.write(canonical(row) + "\n")
                stream.flush()
                if count % self.batch_size == 0 or time.monotonic() - last_update > 45:
                    elapsed = time.monotonic() - started
                    self.emit(self.active_stage, unit=name, episodes=count, scheduled_episodes=expected_rows,
                              unit_elapsed_seconds=elapsed,
                              estimated_unit_remaining_seconds=(elapsed / count * (expected_rows - count)) if expected_rows else None)
                    last_update = time.monotonic()
        if expected_rows is not None and count != expected_rows:
            raise RuntimeError(f"SCHEDULED_DENOMINATOR_MISMATCH: {name}: {count} != {expected_rows}")
        _write_json(marker, {"status": "executed", "inputs": declared, "trace_sha256": file_hash(path),
                            "rows": count, "elapsed_seconds": time.monotonic() - started, "completed_at": now()})
        return path

    def clone_unit(self, name, source, seed, *, inputs=None):
        def cloned():
            source_hash = file_hash(source)
            for original in read_rows(source):
                row = deepcopy(original)
                row["seed"] = seed
                row["cache_reuse"] = {"status": "reused_exact_greedy_base_or_rule_output",
                    "source_trace": str(source), "source_trace_sha256": source_hash,
                    "source_seed": original["seed"], "target_seed": seed,
                    "not_independent_training_run": True, "original_resource_counters": row.get("resource_counters")}
                row["resource_counters"] = {"cache_reused": True, "generated_tokens": 0, "elapsed_seconds": 0.}
                yield row
        return self.trace_unit(name, cloned, inputs={"source_trace_sha256": file_hash(source), "seed": seed, **(inputs or {})},
                               expected_rows=load_config(Path(source).with_suffix(".complete.json"))["rows"])

    def _selected(self):
        manifest = load_config(self.root / "manifests/selected_checkpoints.json")
        return {(entry["method"], entry["seed"]): (self.root / entry["adapter_file"]).parent for entry in manifest["checkpoints"]}

    def _checkpoint_identity(self, method, seed):
        if method in BASE_METHODS:
            return {"model_id": self.config["model"]["selected_id"], "revision": self.config["model"]["revision"], "method": method}
        checkpoint_method = "B1" if method.startswith("C-") else method
        checkpoint = self._selected()[(checkpoint_method, seed)]
        return {"adapter_sha256": file_hash(checkpoint / "adapter_model.safetensors"), "method": method, "seed": seed}

    def backend(self, method, seed):
        from .policy.llm_controller import SpecLocalPolicy
        if method == "B4":
            return None
        if method in BASE_METHODS:
            model = SpecLocalPolicy(self.model_path, device=self.device, precision=self.config["model"]["precision"], lora=False, seed=seed)
        else:
            checkpoint_method = "B1" if method.startswith("C-") else method
            model = SpecLocalPolicy.from_checkpoint(self._selected()[(checkpoint_method, seed)], trainable=False, device=self.device)
        if any(parameter.requires_grad for parameter in model.model.parameters()):
            raise RuntimeError("TEST_TIME_WEIGHT_UPDATES_FORBIDDEN")
        model.model.eval()
        return model

    @staticmethod
    def release(model):
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def model_unit(self, name, method, seed, iterator_factory, *, episodes=None, inputs=None, expected_rows=None):
        declared = {"checkpoint": self._checkpoint_identity(method, seed), **(inputs or {})}
        if episodes is not None:
            declared["episode_schedule_sha256"] = digest([episode_identity(e) for e in episodes])
            expected_rows = len(episodes) if expected_rows is None else expected_rows
        def evaluated():
            model = self.backend(method, seed)
            try:
                before = parameter_hash(model) if model is not None else None
                for row in iterator_factory(model):
                    row["research_status"] = "exploratory"
                    row["human_audit_status"] = "not_executed"
                    row["human_authored_external_status"] = "not_executed"
                    if row.get("split") == "test_external":
                        row["external_authorship"] = "AI_AUTHORED_PROXY"
                    yield row
                if model is not None and (any(p.requires_grad for p in model.model.parameters()) or parameter_hash(model) != before):
                    raise RuntimeError("FROZEN_EVALUATION_WEIGHTS_CHANGED")
            finally:
                del model
                self.release(None)
        return self.trace_unit(name, evaluated, inputs=declared, expected_rows=expected_rows)

    def e2_rows(self, episodes, model, method, seed, **options):
        from .evaluation.batched import evaluate_batched
        from .evaluation.runner import run_episode
        from .policy.rule_controller import RuleController
        if method == "B4":
            generated = (run_episode(episode, RuleController(), method=method, rollout_seed=seed, root=self.root, **options) for episode in episodes)
        else:
            generated = evaluate_batched(episodes, model, method=method, seed=seed, batch_size=self.batch_size, root=self.root, **options)
        for row in generated:
            row["experiment"] = "E3" if options.get("intervention") else "E2"
            row["negative_control"] = bool(row.get("trace", {}).get("episode", {}).get("opportunity", {}).get("negative_control"))
            yield row

    def e1_rows(self, episodes, model, method, seed, condition):
        from .evaluation.batched import generate_batch
        from .policy.rule_controller import RuleController
        from .scoring.evidence import score_response
        for start in range(0, len(episodes), self.batch_size):
            group = episodes[start:start + self.batch_size]
            observations = []
            for episode in group:
                memory = deepcopy(episode["initial_memory"])
                relevant = [record for record in memory if record["attribute"] == episode["current_query"]["attribute"]]
                if condition == "O1":
                    memory = [max(enumerate(relevant), key=lambda pair: (pair[1].get("recorded_at", ""), pair[0]))[1]] if relevant else []
                observations.append({"memory": memory, "user_messages": deepcopy(episode["user_messages"]),
                    "query": deepcopy(episode["current_query"]), "reuse_regime": "Habrá una consulta posterior sin canal de aclaración."})
            outputs = generate_batch(model, "respond", observations) if method != "B4" else [
                {"parsed_action": RuleController().respond(observation)} for observation in observations]
            for episode, observation, output in zip(group, observations, outputs):
                response = output.get("parsed_action") or {}
                score = score_response(response, observation, episode["truth"]["current_value"])
                yield {"experiment": "E1", "method": method, "seed": seed,
                    **{key: episode[key] for key in ("root_id", "latent_world_id", "realization_id", "scenario_family", "split")},
                    "observation_condition": condition, "reader_id": "current_snapshot_no_reuse", "extraction_track": "REF",
                    "intervention": None, "technical_status": "executed", "retry_count": 0,
                    "metrics": {"C2": score["success"], "unsupported_current": score["unsupported_commitment"],
                                "questions": 0, "invalid_outputs": int(not score["valid"])},
                    "score": score, "response": response, "raw_outputs": [output],
                    "input_characters": len(canonical(observation)),
                    "input_tokens": len(output.get("generation", {}).get("prompt_ids", [])) if method != "B4" else None,
                    "silent_truncation": False, "trace": {"episode": deepcopy(episode), "observations": {"respond": observation}}}

    def _add_traces(self, paths):
        for path in paths:
            if str(path) not in self.trace_paths:
                self.trace_paths.append(str(path))

    def mechanism(self):
        from research.run import smoke
        directory = self.destination / "E0"
        summary_path = directory / "summary.json"
        if summary_path.exists():
            summary = load_config(summary_path)
            if file_hash(directory / "TRAZAS.jsonl") != summary["traces_sha256"]:
                raise RuntimeError("E0_PREVIOUS_TRACE_CHANGED")
        else:
            summary = smoke(directory)
        if not summary["intervention_invariants_passed"] or summary["invalid_phase_outputs"]:
            raise RuntimeError("E0_MECHANISM_FAILED")
        return {"status": "executed", "artifacts": [str(summary_path), str(directory / "TRAZAS.jsonl"), str(directory / "INTERVENCIONES.jsonl")]}

    def baselines(self):
        validation = list(read_rows(self.paths["validation"]))
        files = []
        for method in ("B0", "B4"):
            for condition in ("O1", "O2"):
                source = self.model_unit(f"E1_validation_{method}_{condition}_s101", method, 101,
                    lambda model, m=method, c=condition: self.e1_rows(validation, model, m, 101, c), episodes=validation,
                    inputs={"bank_sha256": file_hash(self.paths["validation"]), "condition": condition})
                files.append(source)
                for seed in SEEDS[1:]:
                    files.append(self.clone_unit(f"E1_validation_{method}_{condition}_s{seed}", source, seed))
        for method in ("B0", "B0-P", "B4"):
            source = self.model_unit(f"E2_validation_{method}_s101", method, 101,
                lambda model, m=method: self.e2_rows(validation, model, m, 101), episodes=validation,
                inputs={"bank_sha256": file_hash(self.paths["validation"])})
            files.append(source)
            for seed in SEEDS[1:]:
                files.append(self.clone_unit(f"E2_validation_{method}_s{seed}", source, seed))
        summary = self.destination / "baseline_summary.json"
        _write_json(summary, {"status": "executed", "split": "validation", "prompt_variants_per_base": 1,
                             "source_traces": {str(path): file_hash(path) for path in files},
                             "baseline_seed_reuse": "Exact greedy base/rules; not independent trained models"})
        return {"status": "executed", "traces": [str(p) for p in files], "artifacts": [str(p) for p in files] + [str(summary)]}

    def sft(self):
        from .training.campaign import run_sft_campaign
        directory = self.destination / "sft"
        run_sft_campaign(self.model_path, self.paths["train"], self.paths["validation"], directory, device=self.device, root=self.root)
        return {"status": "executed", "campaign": str(directory / "sft_campaign.json"),
                "artifacts": [str(directory / "sft_campaign.json"), str(directory / "campaign_plan.json")]}

    def release_g3(self, baselines, sft):
        from .prepare import mark_gate
        budget = self.destination / "G3_budget.json"
        _write_json(budget, {"status": "frozen_before_continuations", "accelerator_seconds_per_branch":
            self.config["budget"]["post_sft_accelerator_seconds_cap"], "methods": ["B2-L", "B3", "B2-I"],
            "seeds": list(SEEDS), "reference_execution": self.config["runtime"].get("reference_execution"),
            "config_semantic_hash": self.common_inputs["protocol_semantic_hash"],
            "checkpoint_selection": "common_longitudinal_validation_with_step_zero"})
        mark_gate("G3", [self.destination / "baseline_summary.json", Path(sft["campaign"]), budget],
                  "Real validation baselines and complete three-seed SFT exist; continuation budget fixed before running", root=self.root)
        return {"status": "executed", "artifacts": [str(budget)], "gate": "G3"}

    def continuations(self, sft):
        from .training.campaign import run_continuations
        directory = self.destination / "continuations"
        run_continuations(sft["campaign"], self.paths["train"], self.paths["validation"], directory, device=self.device, root=self.root)
        return {"status": "executed", "campaign": str(directory / "continuations_campaign.json"),
                "artifacts": [str(directory / "continuations_campaign.json"), str(directory / "campaign_plan.json")]}

    def _selected_validation_trace(self, result, log_directory, fallback=None):
        reused = result.get("plan", {}).get("initial_validation_reused", False)
        if result["best_step"] == 0 and reused:
            if fallback is None:
                raise RuntimeError("SELECTED_STEP_ZERO_VALIDATION_SOURCE_MISSING")
            return Path(fallback)
        actual = [entry for entry in result["history"] if "validation" in entry and not (entry["step"] == 0 and reused)]
        indices = [i for i, entry in enumerate(actual, 1) if entry["step"] == result["best_step"]]
        if len(indices) != 1:
            raise RuntimeError("SELECTED_VALIDATION_TRACE_INDEX_AMBIGUOUS")
        path = Path(log_directory) / f"validation_{indices[0]:04d}.jsonl"
        if not path.is_file():
            raise RuntimeError("SELECTED_VALIDATION_TRACE_MISSING: " + str(path))
        return path

    def selection_and_g4(self, sft, continuation):
        from .execution_state import make_selected_checkpoint_manifest, freeze_protocol, lock_test_size
        from .analysis.power_from_validation import plan_from_traces
        from .prepare import mark_gate
        require_gate("G3", self.root)
        sft_runs, continuation_runs = load_config(sft["campaign"]), load_config(continuation["campaign"])
        make_selected_checkpoint_manifest(sft_runs, continuation_runs, root=self.root)
        selected_paths = []
        for seed in SEEDS:
            parent = sft_runs[str(seed)]
            b1_source = self._selected_validation_trace(parent, self.destination / "sft" / str(seed) / "B1_validation")
            for method in TRAINED_METHODS:
                result = parent if method == "B1" else continuation_runs[str(seed)][method]
                source = b1_source if method == "B1" else self._selected_validation_trace(result,
                    self.destination / "continuations" / str(seed) / (method + "_validation"), fallback=b1_source)
                def relabeled(source=source, method=method, seed=seed, result=result):
                    for row in read_rows(source):
                        row.update(method=method, seed=seed, experiment="E2", selection_split="validation",
                            selected_step=result["best_step"], selected_validation_source={"path": str(source), "sha256": file_hash(source)})
                        yield row
                selected_paths.append(self.trace_unit(f"selected_validation_{method}_s{seed}", relabeled,
                    inputs={"source_validation_sha256": file_hash(source), "method": method, "seed": seed,
                            "selected_checkpoint_sha256": self._checkpoint_identity(method, seed)},
                    expected_rows=sum(1 for _ in read_rows(self.paths["validation"]))))
        power_directory = self.destination / "power"
        power_directory.mkdir(exist_ok=True)
        validation = list(read_rows(self.paths["validation"]))
        validation_opportunities = [e for e in validation if e.get("opportunity", {}).get("eligible")]
        power_sources = list(selected_paths)
        for condition in ("O1", "O2"):
            for seed in SEEDS:
                power_sources.append(self.unit_path(f"E1_validation_B0_{condition}_s{seed}"))
        for seed in SEEDS:
            for arm in ("I0", "I1", "I2"):
                power_sources.append(self.model_unit(f"E3_validation_B1_{arm}_s{seed}", "B1", seed,
                    lambda model, s=seed, a=arm: self.e2_rows(validation_opportunities, model, "B1", s,
                        intervention=a, forced_slot="TEMPORAL_SCOPE"), episodes=validation_opportunities,
                    inputs={"validation_bank_sha256": file_hash(self.paths["validation"]), "arm": arm}))
        power = plan_from_traces(power_sources, power_directory, feasible_id_roots=300)
        power_report = power_report_for_lock(power)
        power_report["resource_feasible_locked_roots"] = 300
        power_report["choice_reason"] = "Fixed feasible minimum before opening test; precision limits are reported rather than extending on outcomes"
        power_path = self.destination / "power_for_lock.json"
        _write_json(power_path, power_report)
        test_id = list(read_rows(self.paths["test_id"]))
        plan = plan_secondary_cases(test_id, self.context["robustness_root_ids"], seed=self.config["seeds"]["data_partition"])
        plan_path = self.destination / "secondary_plan.json"
        if plan_path.exists() and load_config(plan_path) != plan:
            raise RuntimeError("PREREGISTERED_SECONDARY_ROOTS_CHANGED")
        _write_json(plan_path, plan)
        current = load_config(self.root / "configs/protocol.yaml")
        if current["sample"]["test_id_locked_roots"] is None:
            lock_test_size(300, power_report, root=self.root)
        elif current["sample"]["test_id_locked_roots"] != 300:
            raise RuntimeError("FULL_CAMPAIGN_EXPECTS_PREREGISTERED_300_ID_ROOTS")
        freeze_protocol(self.root)
        mark_gate("G4", [self.root / "manifests/selected_checkpoints.json", self.root / "manifests/test_size_power.json",
                          power_path, plan_path],
                  "All checkpoints selected on validation; paired power limitations and 300 ID roots fixed before one test opening", root=self.root)
        return {"status": "executed", "secondary_plan": str(plan_path), "selected_validation_traces": [str(p) for p in selected_paths],
                "artifacts": [str(self.root / "manifests/selected_checkpoints.json"), str(power_path), str(plan_path)]}

    def open_test(self):
        require_gate("G4", self.root)
        marker = self.root / "manifests/test_opening.json"
        identity = {"owner": str(self.destination), "executor": "full_campaign", "methods": list(METHODS), "seeds": list(SEEDS),
                    "selected_checkpoints_sha256": file_hash(self.root / "manifests/selected_checkpoints.json"),
                    "protocol_semantic_hash": self.common_inputs["protocol_semantic_hash"]}
        if marker.exists():
            previous = load_config(marker)
            if any(previous.get(key) != value for key, value in identity.items()):
                raise RuntimeError("TEST_ALREADY_OPENED_BY_ANOTHER_OR_CHANGED_CAMPAIGN")
            return
        with marker.open("x", encoding="utf-8") as stream:
            stream.write(canonical({**identity, "opened_at": now(), "status": "opened", "research_status": "exploratory",
                                    "human_external_status": "not_executed"}) + "\n")

    def closed_test(self):
        self.open_test()
        test_id = list(read_rows(self.paths["test_id"]))
        episodes = test_id + list(read_rows(self.paths["test_ood"])) + list(read_rows(self.paths["test_external"]))
        if len({e["root_id"] for e in test_id}) != 300:
            raise RuntimeError("LOCKED_ID_SAMPLE_DENOMINATOR_MISMATCH")
        traces = []
        bank_hashes = {name: file_hash(self.paths[name]) for name in ("test_id", "test_ood", "test_external")}
        for method in ("B0", "B4"):
            for condition in ("O1", "O2"):
                source = self.model_unit(f"E1_test_id_{method}_{condition}_s101", method, 101,
                    lambda model, m=method, c=condition: self.e1_rows(test_id, model, m, 101, c), episodes=test_id,
                    inputs={"test_banks": bank_hashes, "condition": condition})
                traces.append(source)
                for seed in SEEDS[1:]:
                    traces.append(self.clone_unit(f"E1_test_id_{method}_{condition}_s{seed}", source, seed))
        e2 = {}
        for method in METHODS:
            for seed in SEEDS:
                name = f"E2_all_test_{method}_s{seed}"
                if method in BASE_METHODS and seed != 101:
                    path = self.clone_unit(name, e2[(method, 101)], seed)
                else:
                    path = self.model_unit(name, method, seed,
                        lambda model, m=method, s=seed: self.e2_rows(episodes, model, m, s), episodes=episodes,
                        inputs={"test_banks": bank_hashes, "reader": "fixed_symbolic"})
                traces.append(path)
                e2[(method, seed)] = path
        completion = self.destination / "test_completion.json"
        _write_json(completion, {"status": "executed", "completed_at": now(), "opening_marker": str(self.root / "manifests/test_opening.json"),
                    "methods": list(METHODS), "seeds": list(SEEDS), "all_test_partitions": list(bank_hashes),
                    "trace_hashes": {str(path): file_hash(path) for path in traces}})
        return {"status": "executed", "traces": [str(p) for p in traces],
                "e2": {f"{method}:{seed}": str(path) for (method, seed), path in e2.items()},
                "artifacts": [str(p) for p in traces] + [str(completion)]}

    def interventions(self, selection):
        plan = load_config(selection["secondary_plan"])
        eligible, negative = set(plan["eligible_f4_root_ids"]), set(plan["negative_control_root_ids"])
        episodes = [episode for episode in read_rows(self.paths["test_id"]) if
            (episode["root_id"] in eligible and episode["opportunity"].get("eligible")) or
            (episode["root_id"] in negative and episode["opportunity"].get("negative_control"))]
        from .data.negative_controls import unaffected_attribute_controls
        controls = unaffected_attribute_controls(list(read_rows(self.paths["test_id"])), roots=20)
        if sorted({e["root_id"] for e in controls}) != plan["unaffected_attribute_root_ids"]:
            raise RuntimeError("UNAFFECTED_ATTRIBUTE_CONTROL_ROOTS_CHANGED")
        episodes.extend(controls)
        paths, checks = [], []
        for method in ("B1", "B2-L"):
            for seed in SEEDS:
                arms = {}
                for arm in ("I0", "I1", "I2"):
                    arms[arm] = self.model_unit(f"E3_{method}_{arm}_s{seed}", method, seed,
                        lambda model, m=method, s=seed, a=arm: self.e2_rows(episodes, model, m, s,
                            intervention=a, forced_slot=plan["controlled_question_slot"]), episodes=episodes,
                        inputs={"secondary_plan_sha256": file_hash(selection["secondary_plan"]), "arm": arm})
                    paths.append(arms[arm])
                result = verify_intervention_invariants(arms)
                result.update(method=method, seed=seed)
                checks.append(result)
        path = self.destination / "persistence_invariants.json"
        _write_json(path, {"status": "passed", "methods": ["B1", "B2-L"], "seeds": list(SEEDS), "checks": checks})
        return {"status": "executed", "traces": [str(p) for p in paths], "artifacts": [str(p) for p in paths] + [str(path)]}

    def common_reader_experiment(self, closed, *, experiment, robustness_only=False):
        from .evaluation.secondary import reread_frozen, reuse_with_interaction
        transform = reread_frozen if experiment == "E4" else reuse_with_interaction
        robust = set(self.context["robustness_root_ids"])
        traces, baseline_sources = [], {}
        for method in METHODS:
            for seed in SEEDS:
                source = Path(closed["e2"][f"{method}:{seed}"])
                name = f"{experiment}_{method}_s{seed}"
                if method in BASE_METHODS and seed != 101:
                    path = self.clone_unit(name, baseline_sources[method], seed,
                        inputs={"actual_controller_trace_sha256": file_hash(source)})
                else:
                    selected = [row for row in read_rows(source) if not robustness_only or
                                (row["split"] == "test_id" and row["root_id"] in robust)]
                    path = self.model_unit(name, "B0", 101,
                        lambda model, rows=selected: transform(rows, model, batch_size=self.batch_size),
                        inputs={"actual_controller_trace_sha256": file_hash(source), "experiment": experiment,
                                "controller_method": method, "controller_seed": seed, "common_frozen_base": True},
                        expected_rows=len(selected))
                    if method in BASE_METHODS:
                        baseline_sources[method] = path
                traces.append(path)
        return {"status": "executed", "traces": [str(p) for p in traces], "artifacts": [str(p) for p in traces]}

    def robustness(self, selection):
        from .evaluation.secondary import build_extraction_cache, evaluate_extracted, evaluate_nonresponse
        robust = set(self.context["robustness_root_ids"])
        episodes = [e for e in read_rows(self.paths["test_id"]) if e["root_id"] in robust]
        cache_path = self.destination / "extraction_cache.json"
        base = self.backend("B0", 101)
        try:
            initial = parameter_hash(base)
            representatives = {e["root_id"]: e for e in episodes}
            for index, episode in enumerate(representatives.values(), 1):
                build_extraction_cache([episode], base, cache_path)
                self.emit(self.active_stage, unit="frozen_S1_extraction", roots=index, scheduled_roots=100)
            if parameter_hash(base) != initial:
                raise RuntimeError("FROZEN_EXTRACTOR_WEIGHTS_CHANGED")
        finally:
            del base
            self.release(None)
        cache = load_config(cache_path)
        traces, bases = [], {}
        for method in METHODS:
            for seed in SEEDS:
                for track in ("EXTRACT", "NONRESPONSE_20", "NONRESPONSE_40"):
                    name = f"E5_{track}_{method}_s{seed}"
                    if method in BASE_METHODS and seed != 101:
                        path = self.clone_unit(name, bases[(method, track)], seed)
                    else:
                        def generated(model, method=method, seed=seed, track=track):
                            options = {"method": method, "seed": seed, "batch_size": self.batch_size, "root": self.root}
                            if track == "EXTRACT":
                                yield from evaluate_extracted(episodes, model, cache, **options)
                            else:
                                rate = .2 if track == "NONRESPONSE_20" else .4
                                yield from evaluate_nonresponse(episodes, model, rates=(rate,), **options)
                        path = self.model_unit(name, method, seed, generated, episodes=episodes,
                            inputs={"secondary_plan_sha256": file_hash(selection["secondary_plan"]),
                                    "extraction_cache_sha256": file_hash(cache_path), "robustness_track": track})
                        if method in BASE_METHODS:
                            bases[(method, track)] = path
                    traces.append(path)
        return {"status": "executed", "traces": [str(p) for p in traces], "artifacts": [str(cache_path)] + [str(p) for p in traces]}

    def reports(self, stages):
        from .analysis.executed_reports import generate_executed_reports
        traces = []
        for result in stages:
            for path in result.get("traces", []):
                if path not in traces:
                    traces.append(path)
        metadata = {"scope": "authorized_exploratory_full_campaign", "protocol_version": self.config["protocol_version"],
                    "model": self.config["model"], "seeds": list(SEEDS), "execution": "local_only",
                    "human_audit": "not_executed", "human_authored_external": "not_executed",
                    "external_proxy": "AI_AUTHORED_PROXY", "automatic_audit": "executed_nonhuman",
                    "test_id_roots": 300, "test_size_planning": str(self.destination / "power_for_lock.json"),
                    "SFT_campaign": str(self.destination / "sft/sft_campaign.json"),
                    "continuations_campaign": str(self.destination / "continuations/continuations_campaign.json"),
                    "resource_budget_seconds_per_continuation": self.config["budget"]["post_sft_accelerator_seconds_cap"],
                    "baseline_seed_cache": "reused exact greedy outputs, never extra independent trainings",
                    "actual_campaign_elapsed_seconds": time.monotonic() - self.started}
        report_dir = self.destination / "reports"
        summary = generate_executed_reports(traces, report_dir,
            resamples=self.config["evaluation"]["bootstrap_resamples"], metadata=metadata)
        return {"status": "executed", "scheduled_rows": summary["scheduled_rows"],
                "artifacts": [str(path) for path in sorted(report_dir.iterdir()) if path.is_file()]}

    def run(self):
        from .prepare import mark_gate
        require_gate("G2", self.root)
        try:
            self.stage("E0_mechanism", self.mechanism)
            baseline = self.stage("baseline_validation", self.baselines,
                                  inputs={"validation_bank_sha256": file_hash(self.paths["validation"])})
            sft = self.stage("SFT_three_seeds", self.sft,
                inputs={"train_bank_sha256": file_hash(self.paths["train"]), "validation_bank_sha256": file_hash(self.paths["validation"])})
            self.stage("G3_release", lambda: self.release_g3(baseline, sft), inputs={"sft_campaign_sha256": file_hash(sft["campaign"])})
            continuation = self.stage("continuations_three_seeds", lambda: self.continuations(sft),
                                      inputs={"sft_campaign_sha256": file_hash(sft["campaign"])})
            selection = self.stage("selection_power_G4", lambda: self.selection_and_g4(sft, continuation),
                inputs={"sft_campaign_sha256": file_hash(sft["campaign"]), "continuations_campaign_sha256": file_hash(continuation["campaign"])})
            closed = self.stage("closed_test_E1_E2", self.closed_test,
                inputs={"selected_checkpoints_sha256": file_hash(self.root / "manifests/selected_checkpoints.json")})
            intervention = self.stage("E3_persistence", lambda: self.interventions(selection),
                inputs={"secondary_plan_sha256": file_hash(selection["secondary_plan"])})
            reader = self.stage("E4_common_frozen_reader", lambda: self.common_reader_experiment(closed, experiment="E4"))
            robustness = self.stage("E5_extraction_nonresponse", lambda: self.robustness(selection),
                inputs={"secondary_plan_sha256": file_hash(selection["secondary_plan"])})
            reuse = self.stage("E6_reuse_with_interaction", lambda: self.common_reader_experiment(closed, experiment="E6", robustness_only=True))
            reports = self.stage("executed_reports", lambda: self.reports([baseline, closed, intervention, reader, robustness, reuse]))
            evidence = [self.destination / "stages" / (name + ".json") for name in
                        ("closed_test_E1_E2", "E3_persistence", "E4_common_frozen_reader", "E5_extraction_nonresponse", "E6_reuse_with_interaction", "executed_reports")]
            evidence += [self.destination / "reports/ANALISIS.json", self.destination / "reports/RESULTADOS.csv"]
            mark_gate("G5", evidence, "All authorized exploratory stages E0-E6 completed with real local traces; human review/external authorship not executed", root=self.root)
            result = {"status": "executed", "research_status": "exploratory", "human_audit_status": "not_executed",
                      "human_external_status": "not_executed", "reports": str(self.destination / "reports"),
                      "gates": "G0-G5_passed_for_exploratory_scope_only", "scheduled_rows": reports["scheduled_rows"]}
            self.emit("completed", **result)
            _write_json(self.destination / "completion.json", result)
            return result
        except BaseException as exc:
            self.emit(self.active_stage or "initialization", "failed", error_type=type(exc).__name__, error=str(exc),
                      partial_outputs_preserved=True, confirmatory_research_complete=False)
            raise


def run_full(context):
    """Execute the already authorized, prepared and frozen exploratory protocol."""
    return FullCampaign(context).run()
