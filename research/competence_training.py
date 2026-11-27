"""Isolated, explicitly exploratory competence adaptation and continuations.

This module does not grant historical G2/G3 permission. Its entry points verify
the existing explicit closure authorization, a frozen plan and observable-only teacher
targets *before* loading a model. No command runs on import.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
from dataclasses import asdict, is_dataclass
import gc
import json
import math
from pathlib import Path
import time

from research.src.artifacts import digest, file_hash, write_json
from research.src.policy.llm_controller import LLMController, PHASE_LIMITS, MAX_INPUT_TOKENS, SpecLocalPolicy, validate_action
from research.src.training.sft import train_sft, parameter_hash, selection_is_better
from research.src.training.episodic_grpo import train_grpo, verify_reference, EpisodeRollout, PhaseSample, group_telemetry, validate_group
from research.src.training.rollouts import make_rollout_collector

ROOT = Path(__file__).resolve().parent
PROMPTS = ROOT / "prompts/conversation_v3"
REVIEW = ROOT.parent / "reports/varied_protocol_v3_20261025_001/TRAINING_CHECKPOINT_REVIEW.json"
REWARD_SHA256 = "5b2fc887a1b047ba60fc4e567ddcd3ba9b2be21beb84c4fac865e319872fdb99"
METHODS = {"B1c", "B3", "B2-I", "B2-L"}


def _checked_imperfect_examples(specification, train, method):
    """Bind an opt-in train-only demonstration bundle before loading weights."""
    from research.imperfect_sft import load_examples
    if method not in {"B1c", "B3"}:
        raise ValueError("Imperfect demonstrations apply only to SFT, never RL rewards")
    if not isinstance(specification, dict) or set(specification) != {
            "bundle_path", "manifest_sha256", "rate", "seed"}:
        raise ValueError("Explicit imperfect SFT bundle identity required")
    path = Path(specification["bundle_path"]).resolve()
    if file_hash(path / "manifest.json") != specification["manifest_sha256"]:
        raise ValueError("Imperfect SFT manifest changed")
    return load_examples(path, train=train, expected_rate=specification["rate"],
                         expected_seed=specification["seed"])


def make_training_plan(method, parent_checkpoint, train, validation, *, seed,
                       authorization_dir, prompt_root=PROMPTS, frozen_files=None, config=None):
    """Resolve the preregistered design to concrete files, without authorizing it.

    ``config`` supplies ``preregistration={path,sha256}`` and, for a continuation,
    ``b1c_selection_evidence={path,sha256}``. RL additionally requires
    ``capability_thresholds``, ``capability_evidence_sha256`` and
    ``signal_evidence_sha256``. Root persists the result, then seals its hash in
    authorization/PILOT_PLAN.json["competence_plans"].
    """
    if method not in METHODS or seed not in {101, 202, 303}:
        raise ValueError("Unknown competence method/seed")
    if Path(prompt_root).resolve() != PROMPTS.resolve():
        raise ValueError("All methods must use common conversation_v3 prompts")
    config = deepcopy(config or {})
    from research.competence_evaluation import DEFAULT_THRESHOLDS
    preregistration = config.get("preregistration", {})
    if not Path(preregistration.get("path", "")).is_file() or file_hash(preregistration["path"]) != preregistration.get("sha256"):
        raise ValueError("A frozen global preregistration is required")
    frozen = dict(frozen_files or {})
    for path in [Path(__file__).resolve(), ROOT / "src/scoring/reward.py",
                 ROOT / "competence_data.py", ROOT / "competence_evaluation.py",
                 *(PROMPTS / f"{phase}.txt" for phase in ("select", "respond", "write"))]:
        frozen[str(path.resolve())] = file_hash(path)
    plan = {"schema_version": "competence_training_v1", "research_status": "exploratory", "not_human": True,
        "method": method, "training_seed": seed, "authorization_dir": str(Path(authorization_dir).resolve()),
        "parent_checkpoint": str(Path(parent_checkpoint).resolve()), "parent_evidence": checkpoint_evidence(parent_checkpoint),
        "preregistration": preregistration, "frozen_files": frozen,
        "data_sha256": {"train": digest(load_episodes(train, "train")), "validation": digest(load_episodes(validation, "validation"))},
        "limits": {"max_input_tokens": 2048, "max_select_tokens": 64, "max_respond_tokens": 256, "max_write_tokens": 512},
        "decoding": {"temperature": 1., "top_k": 0, "top_p": 1., "selection_sample": False, "rollout_sample": True},
        "learning_rate": 1e-4 if method in {"B1c", "B3"} else 5e-6,
        "epochs": 1 if method == "B1c" else None, "max_updates": 20 if method.startswith("B2-") else None,
        "checkpoint_time_fractions": None if method == "B1c" else [.5, 1.],
        "selection": {"score": "mean_C2_FJS_writefaithful_phaseformat", "utility_tolerance": 1e-6,
                      "terminal_checkpoint_eligible": True},
        "budgets": {"training_seconds": 1800 if method == "B1c" else 450,
                    "validation_seconds": 900, "save_seconds": 120, "max_validation_evaluations": 4},
        "roots_per_batch": 2, "rollouts_per_root": 4, "kl_coefficient": .02, "clip_epsilon": .2,
        "signal_minimum_groups": 40, "signal_minimum_task_varying_fraction": .10,
        "signal_minimum_integrated_improvements": 1}
    plan["capability_thresholds"] = deepcopy(config.get("capability_thresholds", DEFAULT_THRESHOLDS))
    for key in ("b1c_selection_evidence", "capability_thresholds", "capability_evidence_sha256", "signal_evidence_sha256"):
        if key in config:
            plan[key] = config[key]
    if "budgets" in config:
        plan["budgets"].update(config["budgets"])
    if "imperfect_sft" in config:
        _checked_imperfect_examples(config["imperfect_sft"], train, method)
        plan["imperfect_sft"] = deepcopy(config["imperfect_sft"])
        source = ROOT / "imperfect_sft.py"
        plan["frozen_files"][str(source.resolve())] = file_hash(source)
    return plan


def load_episodes(source, split):
    """Reject every closed-test or other split before teacher/optimizer use."""
    if split not in {"train", "validation"}:
        raise ValueError("Competence training accepts only train/validation")
    if isinstance(source, (str, Path)):
        rows = [json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
    else:
        rows = deepcopy(list(source))
    if not rows or any(row.get("split") != split for row in rows):
        raise ValueError("Wrong or empty competence partition: " + split)
    return rows


def historical_b1(seed, review=REVIEW):
    """Use the selected historical B1, particularly step 7140 for seed 202."""
    rows = json.loads(Path(review).read_text())["rows"]
    matches = [r for r in rows if r["method"] == "B1" and r["training_seed"] == seed]
    if len(matches) != 1:
        raise ValueError("No unique selected historical B1")
    row = matches[0]
    path = Path(row["checkpoint_path"]).resolve()
    if file_hash(path / "local_checkpoint.json") != row["checkpoint_manifest_sha256"]:
        raise ValueError("Historical B1 checkpoint manifest changed")
    summary = Path(row["training_summary_path"])
    if file_hash(summary) != row["training_summary_sha256"]:
        raise ValueError("Historical B1 selection evidence changed")
    if Path(json.loads(summary.read_text())["selected_checkpoint"]).resolve() != path:
        raise ValueError("Historical B1 is not the selected checkpoint")
    return path


def checkpoint_evidence(checkpoint):
    """Validate adapter/tokenizer artifacts without loading torch or the GPU."""
    from llm_local_001.model import artifact_hashes
    path = Path(checkpoint).resolve()
    manifest = json.loads((path / "local_checkpoint.json").read_text())
    actual = artifact_hashes(path)
    if actual != manifest["artifacts_sha256"]:
        raise ValueError("Checkpoint artifact manifest mismatch")
    return {"path": str(path), "manifest_sha256": file_hash(path / "local_checkpoint.json"),
            "artifacts_sha256": actual}


def prepare_teacher_examples(episodes, work_directory):
    """Seven exact targets per train episode, balanced by the frozen route plan.

    The sole teacher implementation is the data module's real-environment
    observable RuleController route. Hidden annotations never reach prompts.
    """
    from research.competence_data import teacher_for_episode
    examples = []
    for index, episode in enumerate(load_episodes(episodes, "train")):
        records = []
        for route_index, route in enumerate(("natural", "PROCEED", "ASK:TEMPORAL_SCOPE")):
            trajectory = teacher_for_episode(episode, condition="O2", route=route,
                work_directory=Path(work_directory) / f"episode_{index:06d}" / f"route_{route_index}")
            if [row["phase"].lower() for row in trajectory] != ["select", "respond", "write"]:
                raise ValueError("Teacher must preserve exactly three real phases")
            records.extend((row, route) for row in trajectory if route == "natural" or row["phase"].lower() != "select")
        for row, route in records:
            phase = row["phase"].lower()
            action = validate_action(phase, row["action"])
            completion = row["completion"]
            if json.loads(completion) != action:
                raise ValueError("Teacher completion differs from its recorded observable action")
            # LLMController.prompt rejects all non-public observation fields.
            observation = deepcopy(row["observation"])
            LLMController(None, prompt_root=PROMPTS).prompt(phase, observation)
            examples.append({**deepcopy(row), "phase": phase, "observation": observation, "action": deepcopy(action),
                "completion": completion, "split": "train", "root_id": episode["root_id"],
                "latent_world_id": episode["latent_world_id"], "realization_id": episode["realization_id"],
                "source": "observable_rule_teacher_v5_real_environment", "route": route})
    return examples


def profile_teacher(policy, examples, *, prompt_root=PROMPTS):
    """Tokenize only. Fail before the first update; never truncate or repair."""
    controller = LLMController(policy, prompt_root=prompt_root)
    records = []
    for example in examples:
        phase = example["phase"]
        generation = policy.supervised_generation(controller.prompt(phase, example["observation"]), example["completion"])
        n_input, n_target = len(generation.prompt_ids), len(generation.completion_ids)
        if n_input > MAX_INPUT_TOKENS or not 0 < n_target <= PHASE_LIMITS[phase]:
            raise ValueError("TEACHER_TOKEN_LIMIT: no silent truncation")
        records.append({"phase": phase, "input_tokens": n_input, "completion_tokens": n_target})
    return {"examples": len(records), "max_input_tokens": max(r["input_tokens"] for r in records),
            "max_completion_by_phase": {p: max(r["completion_tokens"] for r in records if r["phase"] == p)
                                        for p in ("select", "respond", "write")}, "records": records}


def inspect_plan(plan_path, *, method, seed, parent_checkpoint, train, validation, authorization_dir):
    """Fail-closed CPU preflight. No mutation of historical protocol evidence."""
    from research.src.closure_protocol import require_training_authorization
    if method not in METHODS or seed not in {101, 202, 303}:
        raise ValueError("Unknown competence method/seed")
    authorization_dir = Path(authorization_dir).resolve()  # None is invalid.
    require_training_authorization(authorization_dir)
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    authorization_plan = json.loads((authorization_dir / "PILOT_PLAN.json").read_text())
    if authorization_plan.get("competence_plans", {}).get(str(plan_path)) != file_hash(plan_path):
        raise ValueError("Competence plan is not linked to sealed training authorization")
    if (plan.get("research_status") != "exploratory" or plan.get("not_human") is not True
            or plan.get("training_seed") != seed or plan.get("method") != method):
        raise ValueError("Exploratory competence plan identity mismatch")
    if plan.get("authorization_dir") != str(authorization_dir):
        raise ValueError("Competence plan authorization directory mismatch")
    preregistration = plan.get("preregistration", {})
    if not Path(preregistration.get("path", "")).is_file() or file_hash(preregistration["path"]) != preregistration.get("sha256"):
        raise ValueError("Global preregistration changed")
    if Path(plan["parent_checkpoint"]).resolve() != Path(parent_checkpoint).resolve():
        raise ValueError("Plan parent checkpoint mismatch")
    if method == "B1c" and Path(parent_checkpoint).resolve() != historical_b1(seed):
        raise ValueError("Adaptation must start at true selected historical B1")
    if method != "B1c":
        selection = plan.get("b1c_selection_evidence", {})
        selection_path = Path(selection.get("path", ""))
        if not selection_path.is_file() or file_hash(selection_path) != selection.get("sha256"):
            raise ValueError("Selected B1c evidence is missing or changed")
        selected = json.loads(selection_path.read_text())
        if (selected.get("method") != "B1c" or selected.get("seed") != seed
                or Path(selected["selected_checkpoint"]).resolve() != Path(parent_checkpoint).resolve()):
            raise ValueError("Continuation parent is not this seed's selected B1c")
    for key, source, split in (("train", train, "train"), ("validation", validation, "validation")):
        rows = load_episodes(source, split)
        if plan.get("data_sha256", {}).get(key) != digest(rows):
            raise ValueError("Frozen competence data mismatch: " + key)
    if {e["root_id"] for e in load_episodes(train, "train")} & {e["root_id"] for e in load_episodes(validation, "validation")}:
        raise ValueError("Train/validation root overlap")
    for name, limit in (("max_input_tokens", 2048), ("max_select_tokens", 64), ("max_respond_tokens", 256), ("max_write_tokens", 512)):
        if plan.get("limits", {}).get(name) != limit:
            raise ValueError("Common decoder/token limits changed: " + name)
    if plan.get("decoding") != {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "selection_sample": False, "rollout_sample": True}:
        raise ValueError("Frozen decoding configuration mismatch")
    if plan.get("checkpoint_time_fractions") != ([.5, 1.] if method != "B1c" else None):
        raise ValueError("Selection schedule differs from declared competence design")
    if plan.get("selection") != {"score": "mean_C2_FJS_writefaithful_phaseformat", "utility_tolerance": 1e-6,
                                  "terminal_checkpoint_eligible": True}:
        raise ValueError("Predeclared competence selection required")
    if plan.get("learning_rate") != (1e-4 if method in {"B1c", "B3"} else 5e-6):
        raise ValueError("Competence learning rate differs from frozen design")
    if method.startswith("B2-") and (plan.get("roots_per_batch") != 2 or plan.get("kl_coefficient") != .02
                                     or plan.get("clip_epsilon") != .2):
        raise ValueError("GRPO batch/KL/clip differs from frozen design")
    for key in ("training_seconds", "validation_seconds", "save_seconds", "max_validation_evaluations"):
        value = plan.get("budgets", {}).get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Explicit positive budget required: " + key)
    if plan["budgets"]["training_seconds"] > (1800 if method == "B1c" else 450) or plan["budgets"]["save_seconds"] > 120:
        raise ValueError("Frozen competence training/save cap exceeded")
    if plan["budgets"]["max_validation_evaluations"] < 4:
        raise ValueError("Budget initial + half/final + terminal evaluation")
    frozen = plan.get("frozen_files", {})
    required = {str(Path(__file__).resolve()), str(ROOT / "src/scoring/reward.py"),
                *(str((PROMPTS / f"{phase}.txt").resolve()) for phase in ("select", "respond", "write"))}
    if not required <= frozen.keys():
        raise ValueError("Plan must freeze trainer, unchanged reward and common prompts")
    for path, expected in frozen.items():
        if file_hash(path) != expected:
            raise ValueError("Frozen competence source changed: " + path)
    if file_hash(ROOT / "src/scoring/reward.py") != REWARD_SHA256:
        raise ValueError("Original reward changed")
    if "imperfect_sft" in plan:
        source = str((ROOT / "imperfect_sft.py").resolve())
        if source not in frozen:
            raise ValueError("Imperfect SFT producer must be frozen in the training plan")
        _checked_imperfect_examples(plan["imperfect_sft"], train, method)
    evidence = checkpoint_evidence(parent_checkpoint)
    if plan.get("parent_evidence") != evidence:
        raise ValueError("Frozen parent checkpoint evidence mismatch")
    return {"status": "passed", "plan_path": str(plan_path), "plan_sha256": file_hash(plan_path), "plan": plan,
            "authorization_dir": str(authorization_dir), "parent": evidence, "research_status": "exploratory", "not_human": True}


class SelectionValidation:
    """Complete frozen cohort evaluations; preserve partials on budget failure."""
    def __init__(self, episodes, destination, plan, method, seed):
        self.episodes, self.destination, self.plan = episodes, Path(destination), plan
        self.method, self.seed = method, seed
        self.calls, self.elapsed_seconds = 0, 0.

    def __call__(self, policy):
        from research.competence_evaluation import evaluate_selected
        budget = self.plan["budgets"]
        if self.calls >= budget["max_validation_evaluations"] or self.elapsed_seconds >= budget["validation_seconds"]:
            raise RuntimeError("COMPETENCE_VALIDATION_BUDGET_EXHAUSTED")
        index = self.calls
        self.calls += 1
        started = time.monotonic()
        try:
            summary = evaluate_selected(policy, self.episodes, self.destination / f"evaluation_{index:02d}",
                prompt_root=PROMPTS, method=self.method, training_seed=self.seed, role="validation",
                protocol={"thresholds": self.plan["capability_thresholds"], "training_plan_sha256": digest(self.plan)},
                include_interventions=False, include_teacher=False)
            if summary.get("status") != "completed" or summary.get("completed_episodes") != summary.get("scheduled_episodes"):
                raise RuntimeError("INCOMPLETE_SELECTION_VALIDATION: preserve partials; no checkpoint selection")
            return competence_metrics(summary["metrics"])
        finally:
            self.elapsed_seconds += time.monotonic() - started


def verify_capability(evidence_path, expected_sha256, parent_checkpoint, plan):
    """RL requires a separately recorded, predeclared B1c capability result."""
    if file_hash(evidence_path) != expected_sha256:
        raise ValueError("Capability evidence hash mismatch")
    value = json.loads(Path(evidence_path).read_text())
    if value.get("status") != "passed" or value.get("research_status") != "exploratory":
        raise ValueError("Capability gate has not passed")
    if value.get("checkpoint_evidence") != checkpoint_evidence(parent_checkpoint):
        raise ValueError("Capability gate evaluated another checkpoint")
    if value.get("thresholds") != plan.get("capability_thresholds") or not value.get("thresholds"):
        raise ValueError("Capability thresholds differ from frozen plan")
    if value.get("data_role") != "capability_validation" or not value.get("data_sha256"):
        raise ValueError("Independent capability-validation evidence required")
    return value


def competence_metrics(metrics):
    """A new predeclared selection score; original task rewards remain intact."""
    result = dict(metrics)
    keys = ("C2", "FJS", "write_faithful_episode", "select_format", "respond_format", "write_format")
    for key in keys:
        if key not in result or not math.isfinite(float(result[key])) or not 0 <= float(result[key]) <= 1:
            raise ValueError("Missing or invalid competence metric: " + key)
    result["reward_utility"] = result.get("utility")
    result["phase_format"] = sum(float(result[k]) for k in ("select_format", "respond_format", "write_format")) / 3
    result["utility"] = (result["C2"] + result["FJS"] + result["write_faithful_episode"] + result["phase_format"]) / 4
    return result


def verify_signal(evidence_path, expected_sha256, parent_checkpoint):
    if file_hash(evidence_path) != expected_sha256:
        raise ValueError("Signal evidence hash mismatch")
    value = json.loads(Path(evidence_path).read_text())
    if value.get("checkpoint_evidence") != checkpoint_evidence(parent_checkpoint):
        raise ValueError("Signal diagnostic used another checkpoint")
    if (value.get("parameter_sha256_before") is None
            or value.get("parameter_sha256_before") != value.get("parameter_sha256_after")):
        raise ValueError("Signal diagnostic did not preserve frozen weights")
    groups = value.get("groups", [])
    if len(groups) < 40 or any(len(g.get("task_returns", [])) != 4 for g in groups):
        raise ValueError("At least forty complete four-rollout groups are required")
    if any(not isinstance(r, (int, float)) or not math.isfinite(r) for g in groups for r in g["task_returns"]):
        raise ValueError("Nonfinite task signal")
    # A penalty-only or KL-only difference cannot authorize task learning.
    varying = sum(len(set(group["task_returns"])) > 1 for group in groups)
    if (value.get("status") != "passed" or varying / len(groups) < .10 or value.get("updates") != 0 or value.get("split") != "train"
            or value.get("integrated_improvements", 0) < 1):
        raise ValueError("Insufficient independent train task signal before RL")
    return value


def integrated_group_comparison(baseline, samples):
    """A valid sampled J improvement over the same-state natural greedy actor."""
    if len(samples) != 4:
        raise ValueError("Integrated comparison requires four independent samples")
    baseline_j = baseline["metrics"]["C2"] * baseline["metrics"]["FJS"]
    improvements = []
    for sample in samples:
        metrics = sample["metrics"]
        sample_j = metrics["C2"] * metrics["FJS"]
        valid = metrics["invalid_outputs"] == 0 and metrics["unsupported_operations"] == 0
        improvements.append(bool(valid and sample_j > baseline_j))
    return {"baseline_J": baseline_j, "sample_J": [row["metrics"]["C2"] * row["metrics"]["FJS"] for row in samples],
            "sample_valid_improvements": improvements, "integrated_improvement": any(improvements),
            "improving_samples": sum(improvements)}


def _json_safe(value):
    if is_dataclass(value):
        return {key: _json_safe(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def collect_signal_diagnostic(checkpoint, train, output, *, authorization_dir,
                              preregistration_path, selection_evidence_path, seed, groups=40):
    """Frozen TRAIN diagnosis; 40 fixed states, natural greedy + 4 full draws.

    No optimizer exists in this function. The exact cohort, 600-generation cap,
    checkpoint and global preregistration are checked before model loading.
    Root explicitly calls this API only after its CUDA/environment preflight.
    """
    from research.src.closure_protocol import require_training_authorization, execution_context
    from research.src.evaluation.runner import run_episode
    from llm_local_001.model import Generation
    require_training_authorization(authorization_dir)
    train = load_episodes(train, "train")
    preregistration_path = Path(preregistration_path).resolve()
    prereg = json.loads(preregistration_path.read_text())
    if (prereg.get("research_status") != "exploratory" or groups != 40
            or prereg.get("signal_max_new_generations") != 600
            or prereg.get("data_sha256", {}).get("train") != digest(train)):
        raise ValueError("Signal preregistration/data/budget mismatch")
    identities = [tuple(row[key] for key in ("root_id", "latent_world_id", "realization_id")) for row in train]
    episodes = dict(zip(identities, train))
    selected_ids = [tuple(identity) for identity in prereg.get("signal_episode_ids", [])]
    if len(episodes) != len(train) or len(selected_ids) != groups or len(set(selected_ids)) != groups:
        raise ValueError("Signal cohort must contain forty unique fixed episodes")
    if any(identity not in episodes for identity in selected_ids):
        raise ValueError("Signal cohort includes an episode outside TRAIN")
    families = Counter(episodes[identity]["scenario_family"] for identity in selected_ids)
    if len(families) != 5 or set(families.values()) != {8}:
        raise ValueError("Signal cohort must balance eight episodes in each of five families")
    selection_evidence_path = Path(selection_evidence_path).resolve()
    selection = json.loads(selection_evidence_path.read_text())
    if (selection.get("method") != "B1c" or selection.get("seed") != seed
            or Path(selection["selected_checkpoint"]).resolve() != Path(checkpoint).resolve()
            or selection.get("status") != "executed"):
        raise ValueError("Signal must diagnose this seed's selected B1c")
    original_authorization = json.loads((selection_evidence_path.parent / "AUTHORIZATION.json").read_text())
    original_plan = original_authorization["plan"]
    expected_prereg = {"path": str(preregistration_path), "sha256": file_hash(preregistration_path)}
    if original_plan.get("preregistration") != expected_prereg:
        raise ValueError("Signal preregistration differs from selected B1c adaptation")
    for path, expected in original_plan["frozen_files"].items():
        if file_hash(path) != expected:
            raise ValueError("Frozen signal source changed: " + path)
    evidence = checkpoint_evidence(checkpoint)
    if selection.get("selected_checkpoint_evidence") != evidence:
        raise ValueError("Selected B1c checkpoint changed before signal diagnosis")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh signal diagnostic destination")
    output.mkdir(parents=True)
    input_plan = {"research_status": "exploratory", "not_human": True, "training_seed": seed,
        "checkpoint_evidence": evidence, "updates": 0, "split": "train", "groups": groups,
        "preregistration": expected_prereg, "selection_evidence": {"path": str(selection_evidence_path), "sha256": file_hash(selection_evidence_path)},
        "data_sha256": digest(train), "signal_episode_ids": [list(identity) for identity in selected_ids],
        "max_new_generations": 600, "decoding": {"greedy_baseline": True, "sample_draws": 4, "temperature": 1., "top_k": 0, "top_p": 1.},
        "frozen_files": original_plan["frozen_files"], "comparison": "valid_sample_J_greater_than_same_episode_natural_greedy_J"}
    write_json(output / "signal_input_plan.json", input_plan)
    policy = controller = None
    generated_calls, sampled_groups, improving_samples = 0, [], 0
    started = time.monotonic()
    try:
        policy = SpecLocalPolicy.from_checkpoint(checkpoint, trainable=False, device="cuda")
        if any(p.requires_grad for p in policy.model.parameters()):
            raise ValueError("Signal actor must be entirely frozen")
        policy.config.temperature = 1.
        policy.model.eval()
        before = parameter_hash(policy)
        class CountedBackend:
            def __getattr__(self, name):
                return getattr(policy, name)
            def generate(self, *args, **kwargs):
                nonlocal generated_calls
                if generated_calls >= 600:
                    raise RuntimeError("SIGNAL_GENERATION_BUDGET_EXCEEDED")
                generated_calls += 1  # Count attempted physical calls, including failures.
                return policy.generate(*args, **kwargs)
        counted_backend = CountedBackend()
        with execution_context(authorization_dir, "training"), (output / "trajectories.jsonl").open("x") as stream:
            for group_index, identity in enumerate(selected_ids):
                episode = episodes[identity]
                physical_rows, rollouts = [], []
                for draw in range(5):
                    sample = draw > 0
                    rollout_seed = seed * 1_000_000 + group_index * 10 + draw
                    db_path = output / "sqlite" / f"group_{group_index:03d}_draw_{draw}.sqlite"
                    db_path.parent.mkdir(exist_ok=True)
                    controller = LLMController(counted_backend, prompt_root=PROMPTS, sample=sample)
                    with policy.torch.no_grad():
                        row = run_episode(episode, controller, condition="O2", rollout_seed=rollout_seed,
                            method="B1c-signal-sample" if sample else "B1c-signal-greedy", db_path=db_path)
                    records = row["trace"]["phase_records"]
                    if [record["phase"] for record in records] != ["select", "respond", "write"]:
                        raise RuntimeError("Signal episode lacks three real phase prefixes")
                    row.update(signal_group=group_index, signal_draw=draw, sample=sample,
                               physical_sqlite_path=str(db_path.resolve()))
                    stream.write(json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    physical_rows.append(row)
                    if sample:
                        components = row["rewards_by_component"]["components"]
                        phases = [PhaseSample(record["phase"], record["generation"] if isinstance(record["generation"], Generation)
                                              else Generation(**record["generation"])) for record in records]
                        metrics = row["metrics"]
                        rollouts.append(EpisodeRollout(episode["root_id"], episode["latent_world_id"], rollout_seed, phases,
                            components["current"], components["future"], questions=metrics["questions"],
                            unsupported_operations=metrics["unsupported_operations"], invalid_outputs=metrics["invalid_outputs"],
                            memory_path=str(db_path.resolve()), split="train"))
                validate_group(rollouts)
                comparison = integrated_group_comparison(physical_rows[0], physical_rows[1:])
                improving_samples += comparison["improving_samples"]
                group = {**group_telemetry(rollouts, "long"), **comparison, "group_index": group_index,
                    "realization_id": episode["realization_id"], "rollout_seeds": [r.rollout_seed for r in rollouts],
                    "short_telemetry": group_telemetry(rollouts, "short"),
                    "trajectory_lines": list(range(group_index * 5 + 1, group_index * 5 + 6))}
                sampled_groups.append(group)
                write_json(output / "progress.json", {"groups_completed": len(sampled_groups), "new_generations": generated_calls,
                    "integrated_improvements": sum(g["integrated_improvement"] for g in sampled_groups)})
                print(json.dumps({"competence_signal_progress": {"groups": len(sampled_groups), "of": groups, "new_generations": generated_calls}}), flush=True)
        after = parameter_hash(policy)
        if before != after:
            raise RuntimeError("SIGNAL_FROZEN_WEIGHTS_CHANGED")
        varying = sum(group["std_task"] > 0 for group in sampled_groups)
        integrated = sum(group["integrated_improvement"] for group in sampled_groups)
        passed = varying >= 4 and integrated >= 1
        report = {"status": "passed" if passed else "insufficient", "research_status": "exploratory", "not_human": True,
            "split": "train", "updates": 0, "checkpoint_evidence": evidence, "training_seed": seed,
            "groups": sampled_groups, "task_varying_groups": varying, "integrated_improvements": integrated,
            "improving_samples": improving_samples, "new_generations": generated_calls,
            "parameter_sha256_before": before, "parameter_sha256_after": after,
            "preregistration": expected_prereg, "input_plan_sha256": file_hash(output / "signal_input_plan.json"),
            "trajectories_sha256": file_hash(output / "trajectories.jsonl"),
            "sqlite_sha256": {str(path.resolve()): file_hash(path) for path in sorted((output / "sqlite").glob("*.sqlite"))},
            "wall_seconds": time.monotonic() - started,
            "interpretation": "Frozen-policy TRAIN sampling diagnostic; no optimizer, no demonstrated RL learning"}
        write_json(output / "signal_report.json", report)
        return report
    except BaseException as error:
        if controller is not None:
            write_json(output / "PARTIAL_PHASE_RECORDS.json", _json_safe(controller.phase_records))
        write_json(output / "FAILURE.json", {"status": "failed", "error": repr(error), "groups_completed": len(sampled_groups),
            "new_generations": generated_calls, "updates": 0, "automatic_resume": False})
        raise
    finally:
        controller = None
        _release(policy)


def select_competence_checkpoint(result):
    """Include the declared terminal snapshot, including update/time caps."""
    candidates = [entry for entry in result["history"] if "validation" in entry and
                  (entry.get("checkpoint_eligible") or entry["step"] == 0 or entry is result["history"][-1])]
    if not candidates:
        raise ValueError("No competence checkpoint selection evidence")
    chosen = candidates[0]
    config = {"utility_tolerance": 1e-6, "validity_guards": {"utility": 1.}}
    for candidate in candidates[1:]:
        if selection_is_better(candidate["validation"], candidate["step"], chosen["validation"], chosen["step"], config):
            chosen = candidate
    path = Path(result["selected_checkpoint"]).parent / f"step_{chosen['step']:06d}"
    if not path.is_dir():
        raise ValueError("Selected checkpoint was not physically saved")
    result["core_selected_checkpoint"] = result["selected_checkpoint"]
    result["selected_checkpoint"], result["best_step"] = str(path), chosen["step"]
    result["selection_metric"] = "mean_C2_FJS_writefaithful_phaseformat"
    return result


def verify_initialization(actor, parent_checkpoint, reference=None):
    """No new adapter, stacking or reference drift is accepted."""
    if Path(actor.origin_checkpoint).resolve() != Path(parent_checkpoint).resolve():
        raise ValueError("Actor origin differs from declared selected parent")
    if reference is not None:
        verify_reference(actor, reference)
    return {"parent_checkpoint": str(Path(parent_checkpoint).resolve()),
            "actor_effective_parameter_sha256": parameter_hash(actor),
            "reference_effective_parameter_sha256": parameter_hash(reference) if reference is not None else None,
            "reference_all_frozen": all(not p.requires_grad for p in reference.model.parameters()) if reference is not None else None}


def _release(policy):
    if policy is not None:
        policy.model.to("cpu")
    gc.collect()
    if policy is not None and policy.torch.cuda.is_available():
        policy.torch.cuda.empty_cache()


def run_training(method, parent_checkpoint, train, validation, destination, *, seed,
                 authorization_dir, plan_path, device="cuda", capability_path=None, signal_path=None,
                 policy_factory=None):
    """Execute only after root orchestration explicitly invokes this function.

    Methods: B1c common adaptation; B3 extra SFT; B2-I/B2-L short/long
    episodic GRPO. Each invocation resets AdamW and loads the exact same selected
    B1c for that seed. No evaluation partition is accepted by this API.
    """
    train, validation = load_episodes(train, "train"), load_episodes(validation, "validation")
    authorization = inspect_plan(plan_path, method=method, seed=seed, parent_checkpoint=parent_checkpoint,
                                train=train, validation=validation, authorization_dir=authorization_dir)
    plan = authorization["plan"]
    if method.startswith("B2-"):
        if capability_path is None:
            raise ValueError("Capability evidence required before RL model load")
        verify_capability(capability_path, plan["capability_evidence_sha256"], parent_checkpoint, plan)
        if signal_path is None:
            raise ValueError("Forty-group task signal evidence required before RL model load")
        verify_signal(signal_path, plan["signal_evidence_sha256"], parent_checkpoint)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Use a new competence training destination")
    destination.mkdir(parents=True)
    write_json(destination / "AUTHORIZATION.json", authorization)
    factory = policy_factory or SpecLocalPolicy.from_checkpoint
    actor = reference = collector = validation_callback = None
    rollout_seconds, rollout_calls = 0., 0
    started = time.monotonic()
    try:
        examples = None
        if method in {"B1c", "B3"}:
            if "imperfect_sft" in plan:
                examples, demonstration_audit = _checked_imperfect_examples(plan["imperfect_sft"], train, method)
                write_json(destination / "IMPERFECT_DEMONSTRATIONS.json", demonstration_audit)
            else:
                examples = prepare_teacher_examples(train, destination / "teacher_sqlite")
            write_json(destination / "teacher_examples.json", examples)
        actor = factory(parent_checkpoint, trainable=True, device=device)
        actor.config.temperature = 1.
        validation_callback = SelectionValidation(validation, destination / "validation", plan, method, seed)
        if examples is not None:
            write_json(destination / "TEACHER_TOKEN_PROFILE.json", profile_teacher(actor, examples))
            initialization = verify_initialization(actor, parent_checkpoint)
        else:
            reference = factory(parent_checkpoint, trainable=False, device="cpu")
            reference.config.temperature = 1.
            initialization = verify_initialization(actor, parent_checkpoint, reference)
        write_json(destination / "INITIALIZATION.json", initialization)
        common = dict(seed=seed, smoke=False, budget_seconds=plan["budgets"]["training_seconds"],
            closure_authorization=authorization_dir, checkpoint_time_fractions=plan["checkpoint_time_fractions"],
            closure_selection={"utility_tolerance": 1e-6, "validity_guards": {"utility": 1.}},
            checkpoint_save_budget_seconds=plan["budgets"]["save_seconds"])
        # Core optimizer insists on an empty directory; metadata lives beside it.
        model_destination = destination / "training"
        from research.src.closure_protocol import execution_context
        with execution_context(authorization_dir, "training"):
            if examples is not None:
                result = train_sft(actor, examples, model_destination, validation_callback, continuation=method != "B1c",
                    epochs=1, checkpoint_every=max(1, len(examples) // 2),
                    learning_rate=plan["learning_rate"], prompt_root=PROMPTS, **common)
            else:
                underlying = make_rollout_collector(destination / "rollouts", condition="O2", method=method,
                                                    prompt_root=PROMPTS)
                def collector(policy, episode, rollout_seed):
                    nonlocal rollout_seconds, rollout_calls
                    before = time.monotonic()
                    try:
                        return underlying(policy, episode, rollout_seed)
                    finally:
                        rollout_seconds += time.monotonic() - before
                        rollout_calls += 1
                result = train_grpo(actor, reference, train, collector, model_destination, validation_callback,
                    horizon="short" if method == "B2-I" else "long", updates=20,
                    roots_per_batch=plan["roots_per_batch"], rollouts_per_root=4,
                    learning_rate=plan["learning_rate"], kl_coefficient=plan["kl_coefficient"],
                    clip_epsilon=plan["clip_epsilon"], reference_execution="cpu_storage_gpu_compute", **common)
        result = select_competence_checkpoint(result)
        counters = result["resource_counters"]
        result.update(method=method, research_status="exploratory", not_human=True,
            parent_evidence=authorization["parent"], plan_sha256=authorization["plan_sha256"],
            selected_checkpoint_evidence=checkpoint_evidence(result["selected_checkpoint"]),
            timing={"rollout_seconds": rollout_seconds, "rollout_trajectories": rollout_calls,
                "optimization_and_training_bookkeeping_seconds": max(0., counters["training_seconds"] - rollout_seconds),
                "validation_seconds": validation_callback.elapsed_seconds,
                "validation_evaluations": validation_callback.calls,
                "save_seconds": counters["checkpoint_save_seconds"],
                "wall_seconds": time.monotonic() - started,
                "optimization_scope": "Includes likelihood/reference forwards, backward, optimizer and training bookkeeping; excludes rollout execution, validation and checkpoint saves"})
        write_json(destination / "COMPETENCE_TRAINING.json", result)
        return result
    except BaseException as error:
        write_json(destination / "FAILURE.json", {"status": "failed", "error": repr(error),
            "research_status": "exploratory", "not_human": True, "automatic_resume": False,
            "wall_seconds": time.monotonic() - started})
        raise
    finally:
        # Clear callback/controller ownership before releasing the resident model.
        collector = validation_callback = None
        _release(reference)
        _release(actor)


def train_adaptation(parent_checkpoint, train, validation, destination, **options):
    return run_training("B1c", parent_checkpoint, train, validation, destination, **options)


def train_continuation(method, b1c_checkpoint, train, validation, destination, **options):
    if method not in METHODS - {"B1c"}:
        raise ValueError("Continuation must be B3/B2-I/B2-L")
    return run_training(method, b1c_checkpoint, train, validation, destination, **options)
