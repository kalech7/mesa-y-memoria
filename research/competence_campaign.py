"""Frozen, exploratory competence adaptation before a new RL comparison.

This campaign owns new data, new authorizations and new checkpoints. It never
changes historical gates or trains on an evaluation partition. Human review is
not fabricated; the existing explicit exploratory authorization is recorded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import gc
import io
import json
from pathlib import Path
import shutil
from statistics import mean
import time
import xml.etree.ElementTree as ET

from .src.artifacts import canonical, digest, file_hash, now, write_json

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPO / "research/runs/competence_campaign_v1_20261025_001"
DEFAULT_DATA = REPO / "research/data/competence_v1_20261025_001"
REPORTS = REPO / "reports/competence_campaign_20261025_001"
PROMPTS = REPO / "research/prompts/conversation_v3"
SEEDS = (101, 202, 303)
METHODS = ("B1c", "B3", "B2-I", "B2-L")
REWARD_SHA = "5b2fc887a1b047ba60fc4e567ddcd3ba9b2be21beb84c4fac865e319872fdb99"


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Preserve the previous artifact: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def progress(run, stage, **details):
    value = {"stage": stage, "recorded_at": now(), **details}
    write_json(Path(run) / "progress.json", value)
    print(canonical(value), flush=True)


def checkpoint_subset(episodes, per_family=2):
    """Choose complete roots before model inference; keep all their worlds."""
    if not episodes or any(e["split"] != "validation" for e in episodes):
        raise ValueError("Checkpoint selection accepts validation only")
    families = defaultdict(set)
    for episode in episodes:
        families[episode["scenario_family"]].add(episode["root_id"])
    if set(families) != {f"F{i}" for i in range(5)} or any(len(v) < per_family for v in families.values()):
        raise ValueError("Five balanced validation families are required")
    roots = {rid for values in families.values() for rid in sorted(values)[:per_family]}
    return [e for e in episodes if e["root_id"] in roots]


def signal_episode_ids(episodes):
    """Forty fixed train worlds, eight per family, selected without outcomes."""
    if not episodes or any(e["split"] != "train" for e in episodes):
        raise ValueError("Signal probe accepts train only")
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["scenario_family"]].append(episode)
    if set(groups) != {f"F{i}" for i in range(5)}:
        raise ValueError("Signal coverage requires five families")
    result = []
    for family in sorted(groups):
        selected = sorted(groups[family], key=lambda e: tuple(e[k] for k in ("root_id", "latent_world_id", "realization_id")))[:8]
        if len(selected) != 8:
            raise ValueError("Eight train worlds per signal family required")
        result.extend([[e[k] for k in ("root_id", "latent_world_id", "realization_id")] for e in selected])
    return result


def branch_schedule(capability_passed, signal_passed):
    """Extra SFT is a useful control even when RL is not justified."""
    return ["B3", "B2-I", "B2-L"] if capability_passed and signal_passed else ["B3"]


def _junit(path):
    tree = ET.parse(path).getroot()
    suites = [tree] if tree.tag == "testsuite" else tree.findall("testsuite")
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    if counts["tests"] <= 0 or counts["failures"] or counts["errors"]:
        raise ValueError("Actual passing JUnit evidence is required")
    return {"status": "passed", "path": str(Path(path).resolve()), "sha256": file_hash(path), **counts}


def _sources():
    from .src.closure_protocol import source_files
    paths = set(source_files())
    paths.update((REPO / "research").glob("competence_*.py"))
    paths.update((REPO / "research/tests").glob("test_competence_*.py"))
    return {str(p.resolve()): file_hash(p) for p in sorted(paths)}


def _token_preflight(data, parent, *, sft_examples=None):
    """No model load. Check every training/validation target, including EOS."""
    from transformers import AutoTokenizer
    from .src.policy.llm_controller import LLMController, PHASE_LIMITS, MAX_INPUT_TOKENS
    tokenizer = AutoTokenizer.from_pretrained(parent, local_files_only=True, trust_remote_code=False)
    builder = LLMController(None, prompt_root=PROMPTS)
    phases, maximums, max_input = Counter(), {}, 0
    rows = []
    for split in ("train", "validation"):
        path = Path(data) / f"teacher_{split}.jsonl"
        examples = sft_examples if split == "train" and sft_examples is not None else read_rows(path)
        for example in examples:
            phase = example["phase"].lower()
            prompt = builder.prompt(phase, example["observation"])
            inputs = (tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
                      if tokenizer.chat_template else tokenizer.encode(prompt, add_special_tokens=True))
            text = example["completion"]
            if not isinstance(text, str):
                text = canonical(text)
            targets = tokenizer.encode(text, add_special_tokens=False)
            if not targets or targets[-1] != tokenizer.eos_token_id:
                targets.append(tokenizer.eos_token_id)
            if len(inputs) > MAX_INPUT_TOKENS or not 0 < len(targets) <= PHASE_LIMITS[phase]:
                raise ValueError("TEACHER_TOKEN_LIMIT: neither truncate nor discard a difficult example")
            max_input = max(max_input, len(inputs))
            maximums[phase] = max(maximums.get(phase, 0), len(targets))
            phases[(split, phase)] += 1
            rows.append({"split": split, "phase": phase, "prompt_sha256": digest(prompt),
                         "input_tokens": len(inputs), "completion_tokens": len(targets)})
    return {"status": "passed", "model_inference": False, "examples": len(rows), "maximum_input": max_input,
            "maximum_completion_by_phase": maximums,
            "phase_counts": {f"{split}:{phase}": n for (split, phase), n in phases.items()}, "prefixes": rows}


def prepare(data, output, *, tests_report, technical_report, imperfect_sft_bundle=None):
    from .competence_data import load_and_check
    from .competence_evaluation import DEFAULT_THRESHOLDS
    from .competence_training import historical_b1
    from .src.closure_protocol import source_hashes
    output, data = Path(output).resolve(), Path(data).resolve()
    if output.exists():
        raise FileExistsError("Choose a new campaign directory")
    splits, manifest = load_and_check(data)
    if set(splits) != {"train", "validation", "evaluation"}:
        raise ValueError("Three independently grouped partitions required")
    audit_path = data / "audit_contracts.json"
    audit = read_json(audit_path)
    if audit.get("status") != "verified" or audit.get("failures") or audit.get("static_failures"):
        raise ValueError("Current data contract audit must pass before training")
    tests = _junit(tests_report)
    technical = read_json(technical_report)
    if (technical.get("status") != "passed" or technical.get("device") != "cuda"
            or technical.get("context_capacity", {}).get("status") != "passed"):
        raise ValueError("A real successful CUDA SFT/GRPO/context-capacity check is required")
    for algorithm in ("sft", "grpo"):
        result = technical[algorithm]
        if result["trainable_before"] == result["trainable_after"] or result["frozen_before"] != result["frozen_after"]:
            raise ValueError("Technical probe lacks actual adapter updates or changed the base")
    if technical["grpo"]["reference_before"] != technical["grpo"]["reference_after"]:
        raise ValueError("Technical reference changed")
    if file_hash(REPO / "research/src/scoring/reward.py") != REWARD_SHA:
        raise ValueError("Original reward changed")
    parents = {str(seed): str(historical_b1(seed)) for seed in SEEDS}
    imperfect_specification, sft_examples = None, None
    if imperfect_sft_bundle is not None:
        from .imperfect_sft import load_examples
        bundle = Path(imperfect_sft_bundle).resolve()
        bundle_manifest = read_json(bundle / "manifest.json")
        imperfect_specification = {"bundle_path": str(bundle), "manifest_sha256": file_hash(bundle / "manifest.json"),
                                   "rate": bundle_manifest["rate"], "seed": bundle_manifest["seed"]}
        sft_examples, _ = load_examples(bundle, train=splits["train"],
            expected_rate=imperfect_specification["rate"], expected_seed=imperfect_specification["seed"])
    profile = _token_preflight(data, parents["101"], sft_examples=sft_examples)
    thresholds = {**DEFAULT_THRESHOLDS, "minimum_roots": 30}
    selection = checkpoint_subset(splits["validation"])
    frozen = _sources()
    if imperfect_specification is not None:
        for source in (REPO / "research/imperfect_sft.py",):
            frozen[str(source.resolve())] = file_hash(source)
    evidence = {str(Path(p).resolve()): file_hash(p) for p in (tests_report, technical_report, audit_path, data / "manifest.json")}
    for split in ("train", "validation", "evaluation"):
        evidence[str(data / f"{split}.jsonl")] = file_hash(data / f"{split}.jsonl")
    for split in ("train", "validation"):
        evidence[str(data / f"teacher_{split}.jsonl")] = file_hash(data / f"teacher_{split}.jsonl")
    for item in audit["artifacts"].values():
        evidence[str(data / item["path"])] = item["sha256"]
    for name, path in manifest["historical_paths"].items():
        evidence[str(Path(path).resolve())] = manifest["historical_hashes_after"][name]
    if imperfect_specification is not None:
        for path in Path(imperfect_specification["bundle_path"]).rglob("*"):
            if path.is_file():
                evidence[str(path.resolve())] = file_hash(path)
    plan = {"version": "competence_campaign_v1", "created_at": now(), "research_status": "exploratory",
            "not_human": True, "human_audit": "not_executed", "confirmatory": False,
            "user_instruction": "implementa esos 4 puntos usa subagentes",
            "data": str(data), "data_sha256": {s: digest(rows) for s, rows in splits.items()},
            "data_file_sha256": {s: file_hash(data / f"{s}.jsonl") for s in splits},
            "training_seeds": list(SEEDS), "parent_checkpoints": parents, "prompt_root": str(PROMPTS),
            "checkpoint_validation_root_ids": sorted({e["root_id"] for e in selection}),
            "checkpoint_thresholds": {**DEFAULT_THRESHOLDS, "minimum_roots": 10},
            "capability_thresholds": thresholds,
            "adaptation": {"epochs": 1, "learning_rate": 1e-4, "max_training_seconds": 1800},
            "continuations": {"methods": ["B3", "B2-I", "B2-L"], "training_seconds_per_branch": 450,
                              "sft_learning_rate": 1e-4, "rl_learning_rate": 5e-6, "maximum_rl_updates": 20,
                              "group_size": 4, "roots_per_batch": 2, "kl": .02, "clip": .2,
                              "checkpoint_time_fractions": [.5, 1.], "terminal_checkpoint_eligible": True},
            "selection_score": "mean_C2_FJS_writefaithful_phaseformat",
            "selection_rule": "validation only, step zero eligible, earliest ties within 1e-6",
            "signal_episode_ids": signal_episode_ids(splits["train"]), "signal_max_new_generations": 600,
            "signal_groups_per_seed": 40, "signal_minimum_task_varying_fraction": .10,
            "signal_minimum_integrated_improvements": 1,
            "continuation_gate": "B3 always runs; RL requires all three B1c seeds to pass capability and their own train signal probe",
            "heldback_rule": "Seal every selected checkpoint before any evaluation-model call; never select from evaluation",
            "frozen_files": frozen, "evidence_sha256": evidence, "core_source_digest": digest(source_hashes()),
            "limitations": ["Finite scripted grammar, AI teacher and audit; no independent human validation.",
                            "A short prospective continuation pilot is not a definitive RL efficacy comparison.",
                            "Equal declared training time does not imply equal updates or token/FLOP budgets."]}
    if imperfect_specification is not None:
        plan["imperfect_sft"] = imperfect_specification
        plan["demonstration_comparison"] = "B1c and B3 use the same frozen imperfect TRAIN demonstrations; RL rewards and heldout labels stay clean"
    plan["sha256"] = digest(plan)
    output.mkdir(parents=True)
    write_new(output / "PREREGISTRATION.json", plan)
    write_new(output / "TOKEN_PREFLIGHT.json", profile)
    write_new(output / "TEST_EVIDENCE.json", tests)
    for source in frozen:
        src = Path(source)
        target = output / "source_snapshot" / src.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    for seed in SEEDS:
        _make_plan(output, "B1c", seed, parents[str(seed)], splits["train"], selection,
                   output / "authorization_adaptation")
    _authorization(output, "authorization_adaptation", "adaptation")
    progress(output, "prepared", roots={s: len({e["root_id"] for e in rows}) for s, rows in splits.items()},
             maximum_input=profile["maximum_input"], training=False)
    return plan


def verify_plan(run):
    run = Path(run).resolve()
    plan = read_json(run / "PREREGISTRATION.json")
    if digest({k: v for k, v in plan.items() if k != "sha256"}) != plan["sha256"]:
        raise ValueError("Preregistration changed")
    for path, expected in {**plan["frozen_files"], **plan["evidence_sha256"]}.items():
        if file_hash(path) != expected:
            raise ValueError("Frozen source/data/evidence changed: " + path)
    return plan


def _make_plan(run, method, seed, parent, train, validation, authorization, *, b1c_evidence=None, capability=None, signal=None):
    from .competence_training import make_training_plan
    global_plan = read_json(Path(run) / "PREREGISTRATION.json")
    config = {"preregistration": {"path": str(Path(run) / "PREREGISTRATION.json"),
                                  "sha256": file_hash(Path(run) / "PREREGISTRATION.json")},
              "evaluation_thresholds": global_plan["checkpoint_thresholds"]}
    if method in {"B1c", "B3"} and "imperfect_sft" in global_plan:
        config["imperfect_sft"] = global_plan["imperfect_sft"]
    if b1c_evidence is not None:
        config["b1c_selection_evidence"] = {"path": str(b1c_evidence), "sha256": file_hash(b1c_evidence)}
    if capability is not None:
        config.update(capability_thresholds=global_plan["capability_thresholds"], capability_evidence_sha256=file_hash(capability))
    if signal is not None:
        config["signal_evidence_sha256"] = file_hash(signal)
    plan = make_training_plan(method, parent, train, validation, seed=seed, authorization_dir=authorization,
                              frozen_files=global_plan["frozen_files"], config=config)
    path = Path(run) / "training_plans" / f"{method}_{seed}.json"
    write_new(path, plan)
    return path


def _authorization(run, folder, phase):
    """Use actual, hashed evidence through the existing separate-run gates."""
    from .src.closure_protocol import record_stage, source_hashes
    run, directory = Path(run), Path(run) / folder
    plan = verify_plan(run)
    directory.mkdir(parents=True)
    write_new(directory / "MANIFIESTO_V4.json", {"protocol_version": "closure_v4",
        "purpose": "competence_campaign_v1_" + phase, "research_status": "exploratory", "not_human": True,
        "created_at": now(), "code_snapshot_sha256": digest(source_hashes()), "stages": {}})
    tests = run / "TEST_EVIDENCE.json"
    tests_xml = Path(read_json(tests)["path"])
    audit = Path(plan["data"]) / "audit_contracts.json"
    technical = next(Path(p) for p in plan["evidence_sha256"] if Path(p).name == "technical_smoke.json")
    prereg = run / "PREREGISTRATION.json"
    record_stage(directory, "inventory", {"status": "passed", "scope": "actual local CUDA runtime and pinned historical parents"},
                 [prereg, technical])
    record_stage(directory, "test-environment", {"status": "passed", "evidence_kind": "executed tests and new corpus contracts"},
                 [tests, tests_xml, audit])
    record_stage(directory, "test-training", {"status": "passed", "evidence_kind": "real CUDA adapter update and frozen backbone/reference"},
                 [technical, tests, tests_xml])
    record_stage(directory, "diagnose", {"status": "passed", "critical_checks_passed": True,
        "scope": "new data contracts, observable teacher targets, token limits, actual technical tests; not an RL efficacy claim"},
                 [audit, run / "TOKEN_PREFLIGHT.json", tests, tests_xml, technical])
    methods = {"adaptation": {"B1c"}, "more_sft": {"B3"}, "rl": {"B2-I", "B2-L"}}[phase]
    plans = {str(p.resolve()): file_hash(p) for p in (run / "training_plans").glob("*.json") if read_json(p)["method"] in methods}
    expected = 6 if phase == "rl" else 3
    if len(plans) != expected:
        raise ValueError("Complete predeclared plan matrix required before authorizing training")
    pilot = {"scope": "competence_exploratory_" + phase, "preregistration": str(prereg),
             "preregistration_sha256": file_hash(prereg), "competence_plans": plans,
             "automatic_audit_instead_of_human": True, "human_review_executed": False}
    write_new(directory / "PILOT_PLAN.json", pilot)
    manifest = read_json(directory / "MANIFIESTO_V4.json")
    manifest["pilot_plan_sha256"] = file_hash(directory / "PILOT_PLAN.json")
    write_json(directory / "MANIFIESTO_V4.json", manifest)
    from .src.closure_protocol import require_training_authorization
    require_training_authorization(directory)


def _release(policy):
    if policy is not None:
        policy.model.to("cpu")
        policy.torch.cuda.empty_cache()
    gc.collect()


def _load_inputs(run):
    plan = verify_plan(run)
    data = Path(plan["data"])
    return plan, {s: read_rows(data / f"{s}.jsonl") for s in ("train", "validation")}


def adapt(run):
    from .competence_training import train_adaptation, checkpoint_evidence
    from .competence_evaluation import evaluate_selected
    from .src.policy.llm_controller import SpecLocalPolicy
    run = Path(run).resolve()
    if (run / "ADAPTATION_COMPLETION.json").exists():
        verify_plan(run)
        return read_json(run / "ADAPTATION_COMPLETION.json")
    plan, splits = _load_inputs(run)
    subset = [e for e in splits["validation"] if e["root_id"] in plan["checkpoint_validation_root_ids"]]
    results = {}
    for seed in SEEDS:
        progress(run, "adaptation", seed=seed)
        target = run / "adaptation" / str(seed)
        result = train_adaptation(plan["parent_checkpoints"][str(seed)], splits["train"], subset, target,
            seed=seed, authorization_dir=run / "authorization_adaptation",
            plan_path=run / "training_plans" / f"B1c_{seed}.json", device="cuda")
        checkpoint = result["selected_checkpoint"]
        progress(run, "capability_validation", seed=seed, selected_step=result["best_step"])
        policy = SpecLocalPolicy.from_checkpoint(checkpoint, trainable=False, device="cuda")
        try:
            summary = evaluate_selected(policy, splits["validation"], run / "capability" / str(seed), prompt_root=PROMPTS,
                method="B1c", training_seed=seed, role="validation", protocol={"thresholds": plan["capability_thresholds"]},
                include_interventions=False, include_teacher=False)
        finally:
            _release(policy)
            del policy
        if summary["status"] != "completed":
            raise RuntimeError("Capability evaluation technically failed; preserve partial evidence")
        capability = {"status": summary["capability_gate"]["status"], "research_status": "exploratory",
            "data_role": "capability_validation", "data_sha256": digest(splits["validation"]),
            "checkpoint_evidence": checkpoint_evidence(checkpoint), "thresholds": plan["capability_thresholds"],
            "metrics": summary["metrics"], "checks": summary["capability_gate"],
            "summary_path": str(run / "capability" / str(seed) / "summary.json"),
            "summary_sha256": file_hash(run / "capability" / str(seed) / "summary.json")}
        capability_path = run / "capability" / str(seed) / "CAPABILITY_EVIDENCE.json"
        write_new(capability_path, capability)
        results[str(seed)] = {"selected_checkpoint": checkpoint, "best_step": result["best_step"],
            "training_evidence": str(target / "COMPETENCE_TRAINING.json"),
            "training_evidence_sha256": file_hash(target / "COMPETENCE_TRAINING.json"),
            "capability_evidence": str(capability_path), "capability_evidence_sha256": file_hash(capability_path),
            "capability_status": capability["status"], "metrics": summary["metrics"]}
        progress(run, "capability_completed", seed=seed, capability_status=capability["status"], metrics=summary["metrics"])
    verify_plan(run)
    completion = {"status": "completed", "created_at": now(), "results": results,
                  "all_seeds_capable": all(r["capability_status"] == "passed" for r in results.values()),
                  "evaluation_opened": False}
    write_new(run / "ADAPTATION_COMPLETION.json", completion)
    return completion


def signal(run):
    from .competence_training import collect_signal_diagnostic
    run = Path(run).resolve()
    plan, splits = _load_inputs(run)
    if (run / "SIGNAL_COMPLETION.json").exists():
        return read_json(run / "SIGNAL_COMPLETION.json")
    adaptation = read_json(run / "ADAPTATION_COMPLETION.json")
    if not adaptation["all_seeds_capable"]:
        value = {"status": "not_executed", "reason": "At least one B1c seed failed the predeclared capability gate",
                 "all_seeds_signal_passed": False, "results": {}, "new_training": False}
    else:
        results = {}
        for seed in SEEDS:
            selected = adaptation["results"][str(seed)]
            progress(run, "frozen_train_signal", seed=seed, groups=40)
            results[str(seed)] = collect_signal_diagnostic(selected["selected_checkpoint"], splits["train"],
                run / "signal" / str(seed), authorization_dir=run / "authorization_adaptation",
                preregistration_path=run / "PREREGISTRATION.json", selection_evidence_path=selected["training_evidence"],
                seed=seed, groups=40)
        value = {"status": "completed", "results": results,
                 "all_seeds_signal_passed": all(r["status"] == "passed" for r in results.values()), "new_training": False}
    write_new(run / "SIGNAL_COMPLETION.json", value)
    return value


def continuations(run):
    from .competence_training import train_continuation
    run = Path(run).resolve()
    plan, splits = _load_inputs(run)
    if (run / "CONTINUATIONS_COMPLETION.json").exists():
        return read_json(run / "CONTINUATIONS_COMPLETION.json")
    adaptation, signals = read_json(run / "ADAPTATION_COMPLETION.json"), read_json(run / "SIGNAL_COMPLETION.json")
    subset = [e for e in splits["validation"] if e["root_id"] in plan["checkpoint_validation_root_ids"]]
    schedule = branch_schedule(adaptation["all_seeds_capable"], signals["all_seeds_signal_passed"])
    results = {str(seed): {} for seed in SEEDS}
    for phase, methods in (("more_sft", ["B3"]), ("rl", [m for m in schedule if m != "B3"])):
        if not methods:
            continue
        auth = run / ("authorization_" + phase)
        for seed in SEEDS:
            selected = adaptation["results"][str(seed)]
            for method in methods:
                _make_plan(run, method, seed, selected["selected_checkpoint"], splits["train"], subset, auth,
                    b1c_evidence=selected["training_evidence"],
                    capability=selected["capability_evidence"] if phase == "rl" else None,
                    signal=run / "signal" / str(seed) / "signal_report.json" if phase == "rl" else None)
        _authorization(run, "authorization_" + phase, phase)
        for seed in SEEDS:
            selected = adaptation["results"][str(seed)]
            for method in methods:
                progress(run, "continuation", seed=seed, method=method)
                result = train_continuation(method, selected["selected_checkpoint"], splits["train"], subset,
                    run / "continuations" / str(seed) / method, seed=seed, authorization_dir=auth,
                    plan_path=run / "training_plans" / f"{method}_{seed}.json", device="cuda",
                    capability_path=selected["capability_evidence"] if phase == "rl" else None,
                    signal_path=run / "signal" / str(seed) / "signal_report.json" if phase == "rl" else None)
                results[str(seed)][method] = {"selected_checkpoint": result["selected_checkpoint"], "best_step": result["best_step"],
                    "training_evidence": str(run / "continuations" / str(seed) / method / "COMPETENCE_TRAINING.json")}
    value = {"status": "completed" if len(schedule) == 3 else "partial_RL_not_executed", "results": results,
             "more_sft_status": "completed", "RL_status": "completed" if len(schedule) == 3 else "not_executed",
             "RL_block_reason": None if len(schedule) == 3 else "Predeclared competence/task-signal prerequisite not satisfied",
             "scope": "short exploratory prospective continuations"}
    write_new(run / "CONTINUATIONS_COMPLETION.json", value)
    return value


def evaluate(run):
    from .competence_evaluation import evaluate_selected
    from .competence_training import checkpoint_evidence
    from .src.policy.llm_controller import SpecLocalPolicy
    run = Path(run).resolve()
    plan = verify_plan(run)
    if (run / "EVALUATION_COMPLETION.json").exists():
        return read_json(run / "EVALUATION_COMPLETION.json")
    # Selection is now finished; no further validation or training occurs.
    adaptation, continuation = read_json(run / "ADAPTATION_COMPLETION.json"), read_json(run / "CONTINUATIONS_COMPLETION.json")
    episodes = read_rows(Path(plan["data"]) / "evaluation.jsonl")
    if digest(episodes) != plan["data_sha256"]["evaluation"]:
        raise ValueError("Heldback bank changed")
    selected = []
    for seed in SEEDS:
        for method, checkpoint in (("B1-parent", plan["parent_checkpoints"][str(seed)]),
                                   ("B1c", adaptation["results"][str(seed)]["selected_checkpoint"])):
            selected.append({"method": method, "training_seed": seed, "checkpoint_path": checkpoint})
        for method, result in continuation["results"][str(seed)].items():
            selected.append({"method": method, "training_seed": seed, "checkpoint_path": result["selected_checkpoint"]})
    for row in selected:
        checkpoint_evidence(row["checkpoint_path"])
        row["checkpoint_manifest_sha256"] = file_hash(Path(row["checkpoint_path"]) / "local_checkpoint.json")
        row["adapter_sha256"] = file_hash(Path(row["checkpoint_path"]) / "adapter_model.safetensors")
    selection = {"version": "competence_selection_v1", "created_at": now(), "selection_complete": True,
        "selection_split": "validation", "selected_before_evaluation": True,
        "heldback_episodes_sha256": digest(episodes), "heldback_bank_sha256": plan["data_file_sha256"]["evaluation"],
        "rows": selected, "continuation_status": continuation["status"]}
    path = run / "SELECTED_CHECKPOINTS.json"
    write_new(path, selection)
    protocol = {"thresholds": plan["capability_thresholds"], "heldback_authorization": {
        "selection_manifest_path": str(path), "selection_manifest_sha256": file_hash(path), "locked_bank_sha256": digest(episodes)}}
    results = {}
    for row in selected:
        seed, method = row["training_seed"], row["method"]
        progress(run, "heldback_evaluation", seed=seed, method=method)
        policy = SpecLocalPolicy.from_checkpoint(row["checkpoint_path"], trainable=False, device="cuda")
        try:
            summary = evaluate_selected(policy, episodes, run / "evaluation" / f"{method}_{seed}", prompt_root=PROMPTS,
                method=method, training_seed=seed, role="evaluation", protocol=protocol,
                include_interventions=method == "B1c", include_teacher=False)
        finally:
            _release(policy)
            del policy
        if summary["status"] != "completed":
            raise RuntimeError("Heldback evaluation technically failed; never fill missing outcomes")
        results[f"{method}_{seed}"] = {"metrics": summary["metrics"], "roots": summary["roots"],
            "summary_path": str(run / "evaluation" / f"{method}_{seed}" / "summary.json"),
            "summary_sha256": file_hash(run / "evaluation" / f"{method}_{seed}" / "summary.json")}
    verify_plan(run)
    value = {"status": "completed", "results": results, "selection_manifest_sha256": file_hash(path),
             "evaluation_performed_after_all_selection": True, "no_selection_after_evaluation": True}
    write_new(run / "EVALUATION_COMPLETION.json", value)
    return value


def report(run):
    from .competence_evaluation import descriptive_contrasts
    run = Path(run).resolve()
    plan = verify_plan(run)
    evaluation, adaptation = read_json(run / "EVALUATION_COMPLETION.json"), read_json(run / "ADAPTATION_COMPLETION.json")
    continuation = read_json(run / "CONTINUATIONS_COMPLETION.json")
    rows = []
    for value in evaluation["results"].values():
        rows.extend(read_rows(Path(value["summary_path"]).parent / "trajectories.jsonl"))
    contrasts = descriptive_contrasts(rows)
    decision = ("compare_completed_short_RL_pilots_without_claiming_general_superiority" if continuation["status"] == "completed"
                else "establish_missing_competence_or_train_signal_before_RL")
    outcome = {"status": "completed", "research_status": "exploratory", "decision": decision,
        "adaptation_completed": True, "all_seeds_capable": adaptation["all_seeds_capable"],
        "continuation_status": continuation["status"], "contrasts": contrasts,
        "evaluation": evaluation["results"], "capability_by_seed": adaptation["results"],
        "preregistration_sha256": file_hash(run / "PREREGISTRATION.json"),
        "human_audit": "not_executed", "confirmatory": False,
        "next_experiment": ("Replicate on an independently authored corpus with a predeclared sufficient learning budget."
                            if continuation["status"] == "completed" else
                            "Diagnose the failed predeclared capability or train-signal checks on separate development; do not tune with this evaluation bank.")}
    write_new(run / "FINAL_DECISION.json", outcome)
    text = ["# Respuesta a Vincent: adaptación común y evaluación nueva", "",
        f"Estado: adaptación completada en tres semillas; competencia suficiente en todas: **{adaptation['all_seeds_capable']}**. "
        f"Continuaciones: **{continuation['status']}**. No se sustituyen ramas ausentes por resultados de referencia.", "",
        "200 raíces de entrenamiento, 30 de validación y 50 de evaluación; todas nuevas respecto a los bancos observados. "
        "Se conserva la gramática sintética y sus limitaciones: esto no acredita validación humana ni un estudio confirmatorio.", "",
        "| Modelo seleccionado | Respuesta actual C2 | Acierto futuro FJS | Éxito conjunto J |",
        "|---|---:|---:|---:|"]
    methods = sorted({key.rsplit("_", 1)[0] for key in evaluation["results"]})
    for method in methods:
        metrics = [evaluation["results"][f"{method}_{seed}"]["metrics"] for seed in SEEDS]
        text.append(f"| {method} | {100*mean(m['C2'] for m in metrics):.2f}% | {100*mean(m['FJS'] for m in metrics):.2f}% | {100*mean(m['J_primary'] for m in metrics):.2f}% |")
    text += ["", "Los porcentajes promedian mundos dentro de raíces y luego las tres semillas; las semillas no son usuarios adicionales.",
             "J=C2×FJS; la validez de formato y las escrituras sin sustento permanecen como métricas separadas.", "", "## Preguntas de Vincent", ""]
    for contrast in contrasts["contrasts"]:
        if contrast["status"] == "completed":
            text.append(f"- {contrast['contrast']}: {100*contrast['estimate']:+.2f} puntos porcentuales; resultado descriptivo, sin afirmación confirmatoria.")
        else:
            text.append(f"- {contrast['contrast']}: no estimable — {contrast.get('reason', 'comparación incompleta')}.")
    text += ["", "## Decisión", "", decision, "", outcome["next_experiment"], "",
        "La recompensa original y los prompts fueron comunes a todos los métodos. B1c y sus continuaciones se seleccionaron exclusivamente en validación. "
        "Si falló la puerta de competencia o de señal, RL no se ejecutó; más SFT sí se conserva como control. "
        "No se bajaron umbrales después de observar resultados.", "",
        "[Decisión y resultados completos](FINAL_DECISION.json). [Planes de entrenamiento](training_plans/). "
        "[Selección previa a evaluación](SELECTED_CHECKPOINTS.json). [Prerregistro](PREREGISTRATION.json)."]
    target = run / "RESPUESTA_VINCENT.md"
    with target.open("x") as stream:
        stream.write("\n".join(text) + "\n")
    completion = {"status": "completed", "completed_at": now(), "plan_sha256": plan["sha256"],
        "training_was_executed": True, "human_review_was_executed": False,
        "artifacts_sha256": {str(p.relative_to(run)): file_hash(p) for p in run.rglob("*")
                              if p.is_file() and p.name not in {"progress.json", "COMPLETION.json"}}}
    write_new(run / "COMPLETION.json", completion)
    progress(run, "completed", decision=decision, continuation_status=continuation["status"])
    return outcome


def execute(run):
    run = Path(run).resolve()
    if (run / "COMPLETION.json").exists():
        raise FileExistsError("This campaign already completed; preserve its results")
    try:
        adapt(run)
        signal(run)
        continuations(run)
        evaluate(run)
        return report(run)
    except BaseException as error:
        write_json(run / "FAILURE.json", {"status": "failed", "error": repr(error), "recorded_at": now(),
                   "automatic_resume": False, "preserve_partial_evidence": True})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--output", default=str(DEFAULT_RUN))
    p.add_argument("--tests-report", required=True)
    p.add_argument("--technical-report", required=True)
    p.add_argument("--imperfect-sft-bundle", help="Frozen train-only demonstration arm; B1c and more-SFT share it")
    for command in ("execute", "adapt", "signal", "continuations", "evaluate", "report", "status"):
        p = sub.add_parser(command)
        p.add_argument("--run", default=str(DEFAULT_RUN))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = prepare(args.data, args.output, tests_report=args.tests_report, technical_report=args.technical_report,
                        imperfect_sft_bundle=args.imperfect_sft_bundle)
        print(canonical({"status": "prepared", "sha256": value["sha256"]}))
    elif args.command == "status":
        value = read_json(Path(args.run) / "progress.json")
        print(canonical(value))
    else:
        value = globals()[args.command](args.run)
        print(canonical({"status": value.get("status", "completed")}))


if __name__ == "__main__":
    main()
