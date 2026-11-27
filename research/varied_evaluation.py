"""Prospective, frozen-checkpoint evaluation of varied synthetic development.

Prepare freezes data, source, prompts, twelve selections and a strict generation
budget without loading a model. Execute consumes that plan once; no training,
checkpoint selection, test opening, hidden-information prompt or retry exists.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sqlite3
from statistics import mean
import struct
import tempfile
import time

from llm_local_001.model import Generation
from .src.artifacts import ROOT, canonical, digest, file_hash, now, write_json
from .src.closure_protocol import source_hashes
from .src.env.episode import EpisodeEnvironment
from .src.env.user_simulator import QUESTIONS
from .src.evaluation.runner import run_episode, select_valid
from .src.policy.llm_controller import LLMController, MAX_INPUT_TOKENS, PHASE_LIMITS, SpecLocalPolicy
from .src.policy.rule_controller import RuleController, evidence_model
from .src.scoring.evidence import Evaluator
from .src.training.episodic_grpo import EpisodeRollout, PhaseSample, group_telemetry
from .src.training.sft import parameter_hash
from .signal_diagnostic import independent_read, analyze as analyze_actions, compact as compact_action, enrich

VERSION = "varied_conversation_v3_frozen_selection"
ACTIONS = ("PROCEED", *("ASK:" + slot for slot in sorted(QUESTIONS)))
METHODS = ("B1", "B3", "B2-I", "B2-L")
SEEDS = (101, 202, 303)


def _sources():
    values = source_hashes()
    for path in ROOT.glob("*.py"):
        values[str(path.relative_to(ROOT.parent))] = file_hash(path)
    return dict(sorted(values.items()))


def _runtime():
    values = {"python": platform.python_version()}
    for package in ("torch", "transformers", "peft", "tokenizers"):
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            values[package] = None
    return values


def _tensor_fingerprint(path):
    raw = Path(path).read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + length])
    values = []
    for name, item in sorted(header.items()):
        if name == "__metadata__":
            continue
        start, end = item["data_offsets"]
        blob = raw[8 + length + start:8 + length + end]
        values.append({"name": name, "dtype": item["dtype"], "shape": item["shape"],
                       "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)})
    return digest(values)


def verify_catalog(review_path):
    """Verify actual selected files; equal scores or filenames never alias."""
    review = json.loads(Path(review_path).read_text())
    rows = review["rows"]
    expected = {(seed, method) for seed in SEEDS for method in METHODS}
    if len(rows) != len(expected) or {(r["training_seed"], r["method"]) for r in rows} != expected:
        raise ValueError("Twelve historically selected checkpoints are required")
    files, hashed = {}, {}
    def verify(path, expected_hash):
        path = Path(path).resolve()
        if path not in hashed:
            hashed[path] = file_hash(path)
        if hashed[path] != expected_hash:
            raise ValueError("Selected artifact changed: " + str(path))
        files[str(path)] = expected_hash
    for model_path, model in review["base_models"].items():
        for name, expected_hash in model["actual_artifacts_sha256"].items():
            verify(Path(model_path) / name, expected_hash)
    for row in rows:
        checkpoint = Path(row["checkpoint_path"])
        verify(row["training_summary_path"], row["training_summary_sha256"])
        summary = json.loads(Path(row["training_summary_path"]).read_text())
        if Path(summary["selected_checkpoint"]).resolve() != checkpoint.resolve() or summary["best_step"] != row["selected_step"]:
            raise ValueError("Historical selection changed")
        verify(checkpoint / "local_checkpoint.json", row["checkpoint_manifest_sha256"])
        manifest = json.loads((checkpoint / "local_checkpoint.json").read_text())
        present = {str(p.relative_to(checkpoint)) for p in checkpoint.rglob("*") if p.is_file() and p.name != "local_checkpoint.json"}
        if present != set(manifest["artifacts_sha256"]):
            raise ValueError("Selected checkpoint artifact set changed")
        for name, expected_hash in manifest["artifacts_sha256"].items():
            verify(checkpoint / name, expected_hash)
        tensor = _tensor_fingerprint(checkpoint / "adapter_model.safetensors")
        if tensor != row["selected_adapter_tensor_content_sha256"]:
            raise ValueError("Selected adapter tensors changed")
        recipe = deepcopy(row["effective_policy_fingerprint_recipe"])
        base = Path(manifest["config"]["model_path"])
        base_files = review["base_models"][str(base)]["actual_artifacts_sha256"]
        recipe["base_model_weights"] = {name: hashed[(base / name).resolve()] for name in base_files
                                        if name.endswith((".safetensors", ".bin", ".index.json"))}
        recipe["base_architecture_sha256"] = hashed[(base / "config.json").resolve()]
        recipe["precision"] = manifest["config"]["precision"]
        recipe["adapter_tensor_content_sha256"] = tensor
        recipe["adapter_config"] = json.loads((checkpoint / "adapter_config.json").read_text())
        recipe["tokenizer_artifacts_sha256"] = {name: file_hash(checkpoint / name) for name in recipe["tokenizer_artifacts_sha256"]}
        if digest(recipe) != row["effective_policy_fingerprint_sha256"]:
            raise ValueError("Effective policy equivalence proof changed")
    files[str(Path(review_path).resolve())] = file_hash(review_path)
    ordered = sorted(rows, key=lambda r: (r["training_seed"], METHODS.index(r["method"])))
    classes = []
    for row in ordered:
        fingerprint = row["effective_policy_fingerprint_sha256"]
        found = next((group for group in classes if group["fingerprint"] == fingerprint), None)
        label = {key: row[key] for key in ("method", "training_seed", "checkpoint_path", "selected_step")}
        if found is None:
            found = {"fingerprint": fingerprint, "representative": label, "labels": []}
            classes.append(found)
        found["labels"].append(label)
    return classes, files


def historical_banks():
    paths = {name: ROOT / "data/bank_1_1" / (name + ".jsonl")
             for name in ("dev_smoke", "train", "validation", "test_id", "test_ood", "test_external")}
    paths["development_v2"] = ROOT / "data/development_v2_20261024_001/validation.jsonl"
    return {name: [json.loads(line) for line in path.open() if line.strip()] for name, path in paths.items()}, paths


def load_bank(path, *, expected_roots=50, historical_splits=None):
    path = Path(path).resolve()
    manifest_path = path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("data_role") != "prospective_development_only" or manifest.get("bank_sha256") != file_hash(path):
        raise ValueError("A frozen prospective development manifest is required")
    episodes = [json.loads(line) for line in path.open() if line.strip()]
    identities = [tuple(e[k] for k in ("root_id", "latent_world_id", "realization_id")) for e in episodes]
    if len(identities) != len(set(identities)) or any(e["split"] != "validation" for e in episodes):
        raise ValueError("Duplicate or non-development episode")
    if len({e["root_id"] for e in episodes}) != expected_roots:
        raise ValueError("Unexpected independent-root count")
    if any(e["realization_id"] != "v0" for e in episodes):
        raise ValueError("This plan freezes exactly one linguistic realization per world")
    for episode in episodes:
        if not 1 <= len(episode["initial_memory"]) <= 6 or not 2 <= len(episode["user_messages"]) <= 8:
            raise ValueError("Varied v3 requires 1..6 memory records and 2..8 dialogue turns")
        if any(m.get("role", "user") not in {"user", "assistant"} for m in episode["user_messages"]):
            raise ValueError("Invalid public dialogue role")
        if not episode.get("authoring", {}).get("stratum"):
            raise ValueError("Frozen authoring stratum is required")
    if expected_roots == 50 and not {3, 4} <= {len(e["initial_memory"]) for e in episodes}:
        raise ValueError("The full v3 bank must include both 3-record and 4-record histories")
    from .src.data.semantic_partition_v3 import assert_semantic_split_isolation_v3
    paths = {}
    if historical_splits is None:
        historical_splits, paths = historical_banks()
    isolation = assert_semantic_split_isolation_v3({"validation": episodes}, historical_splits=historical_splits)
    manifest = {**manifest, "runtime_isolation_verification": isolation,
                "runtime_historical_bank_sha256": {str(p.resolve()): file_hash(p) for p in paths.values()}}
    return episodes, manifest


def action_id(action):
    if not select_valid(action):
        return "INVALID"
    return "PROCEED" if action["action"] == "PROCEED" else "ASK:" + action["slot"]


def action_options(action):
    if action == "PROCEED":
        return {"method": "forced_PROCEED", "condition": "O2"}
    if action not in ACTIONS:
        raise ValueError("Illegal action")
    return {"method": "forced_ASK", "condition": "O2", "intervention": "I2", "forced_slot": action.split(":", 1)[1]}


def enumerate_cases(episodes, prompt_root, encoder):
    """Derive every allowed prefix through the current real environment."""
    builder = LLMController(object(), prompt_root=prompt_root)
    registry, cases = {}, []
    def add(phase, observation):
        prompt = builder.prompt(phase, observation)
        tokens = list(encoder(prompt))
        if not tokens or len(tokens) > MAX_INPUT_TOKENS:
            raise ValueError("Input token limit exceeded before inference")
        request = digest({"prompt": prompt, "max_new_tokens": PHASE_LIMITS[phase]})
        entry = {"phase": phase, "prompt": prompt, "prompt_ids": tokens, "max_new_tokens": PHASE_LIMITS[phase]}
        if request in registry and registry[request] != entry:
            raise ValueError("Prefix identity collision")
        registry[request] = entry
        return prompt
    with tempfile.TemporaryDirectory(prefix="varied_v3_prefixes_") as work:
        for index, episode in enumerate(episodes):
            cid = digest([episode[k] for k in ("root_id", "latent_world_id", "realization_id")])[:24]
            prior = None
            for condition in ("O1", "O2"):
                for ai, action in enumerate(ACTIONS):
                    env = EpisodeEnvironment(episode, Path(work) / f"{index}_{condition}_{ai}.sqlite")
                    try:
                        select = env.observe("select", condition)
                        add("select", select)
                        if condition == "O2":
                            prior = select
                        if action.startswith("ASK:"):
                            env.ask(action.split(":", 1)[1])
                        for phase in ("respond", "write"):
                            add(phase, env.observe(phase, condition, "I2" if action != "PROCEED" else None))
                        if condition == "O2" and action != "PROCEED":
                            add("write", env.observe("write", condition, "I1"))
                    finally:
                        env.dispose()
            # Role retention is an environment contract, not a learned behavior.
            if any(a.get("role", "user") != b.get("role", "user") for a, b in zip(prior["user_messages"], episode["user_messages"], strict=True)):
                raise ValueError("Environment discarded a public dialogue role")
            altered = deepcopy(episode)
            altered.update(root_id="PRIVATE_ROOT", latent_world_id="PRIVATE_WORLD", scenario_family="PRIVATE_FAMILY")
            altered["truth"] = {"secret": "PRIVATE_GOLD"}
            altered["future_query"] = {"secret": "PRIVATE_FUTURE_QUERY"}
            env = EpisodeEnvironment(altered, Path(work) / f"metadata_{index}.sqlite")
            try:
                if builder.prompt("select", env.observe("select")) != builder.prompt("select", prior):
                    raise ValueError("Private metadata changed the actor prefix")
            finally:
                env.dispose()
            cases.append({"case_id": cid, "state_id": digest(builder.prompt("select", prior)),
                          "select_observation": prior, "episode": episode,
                          "reference_slot": evidence_model(prior)["missing_slot"] or "TEMPORAL_SCOPE"})
    for root in {e["root_id"] for e in episodes}:
        if len({c["state_id"] for c in cases if c["episode"]["root_id"] == root}) != 1:
            raise ValueError("Compatible worlds expose different pre-ASK prompts")
    return cases, registry


def sampled_episodes(episodes):
    strata = defaultdict(set)
    for episode in episodes:
        strata[episode["authoring"]["stratum"]].add(episode["root_id"])
    roots = {root for group in strata.values() for root in sorted(group)[:2]}
    return sorted((e for e in episodes if e["root_id"] in roots), key=lambda e: (e["root_id"], e["latent_world_id"], e["realization_id"]))


def prepare(bank, checkpoint_review, prompt_root, output, *, maximum_new_generations=4000,
            device="cuda", encoder=None, expected_roots=50):
    output, bank, prompt_root = Path(output).resolve(), Path(bank).resolve(), Path(prompt_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh evaluation directory; preserve all prior evidence")
    if output.is_relative_to(ROOT / "runs/full_exploratory_001"):
        raise ValueError("Historical campaign is read-only")
    classes, artifacts = verify_catalog(checkpoint_review)
    review = json.loads(Path(checkpoint_review).read_text()) if Path(checkpoint_review).is_file() else None
    if review is not None and len({digest(row["effective_policy_fingerprint_recipe"]["tokenizer_artifacts_sha256"]) for row in review["rows"]}) != 1:
        raise ValueError("This shared-prefix plan requires identical tokenizer artifacts across selected policies")
    episodes, manifest = load_bank(bank, expected_roots=expected_roots)
    if expected_roots != 50 and encoder is None:
        raise ValueError("Nonprotocol sizes are allowed only in an injected tokenizer fixture")
    tokenizer_fixture = encoder is not None
    if encoder is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(classes[0]["representative"]["checkpoint_path"], local_files_only=True)
        def encoder(prompt):
            return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True) if tokenizer.chat_template else tokenizer.encode(prompt, add_special_tokens=True)
    cases, registry = enumerate_cases(episodes, prompt_root, encoder)
    selected_samples = sampled_episodes(episodes)
    states = len({c["state_id"] for c in cases})
    # Conservative request bound; exact-prompt caching can only reduce it.
    bound = len(classes) * len(episodes) * 3 + len(episodes) * (5 * 2 + 3 + 2 + 2 * 2) + states * 4 * 2 + len(selected_samples) * 4 * 2 * 3
    if maximum_new_generations <= 0 or bound > maximum_new_generations:
        raise ValueError(f"Predeclared generation budget insufficient: bound={bound}, cap={maximum_new_generations}")
    sources = _sources()
    artifacts.update({str(bank): file_hash(bank), str(bank.with_name("manifest.json")): file_hash(bank.with_name("manifest.json"))})
    artifacts.update(manifest["runtime_historical_bank_sha256"])
    for phase in ("select", "respond", "write", "reader"):
        p = prompt_root / (phase + ".txt")
        artifacts[str(p)] = file_hash(p)
    output.mkdir(parents=True, exist_ok=True)
    for name, expected in sources.items():
        src = ROOT.parent / name
        dst = output / "source_snapshot" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    with (output / "allowed_prefixes.jsonl").open("x") as stream:
        for key, value in sorted(registry.items()):
            stream.write(canonical({"request": key, **value}) + "\n")
    plan = {"version": VERSION, "created_at": now(), "bank": str(bank), "bank_sha256": file_hash(bank),
        "bank_manifest": manifest, "checkpoint_review": str(Path(checkpoint_review).resolve()),
        "prompt_root": str(prompt_root), "training": False, "optimizer_steps": 0,
        "checkpoint_selection": False, "confirmatory_test_opened": False, "human_audit": "not_executed",
        "episodes": len(episodes), "roots": expected_roots, "observable_states": states,
        "policy_classes": classes, "maximum_new_generations": maximum_new_generations,
        "conservative_new_generation_bound": bound, "runtime": _runtime(), "device": device,
        "generation": {"batch_size": 1, "greedy": True, "sample_temperatures": [1., 1.3],
            "draws_per_state": 4, "rollouts_per_group": 4, "top_p": 1., "top_k": 0,
            "phase_limits": PHASE_LIMITS, "no_retry": True, "reader": "fixed_symbolic"},
        "sampled_episode_ids": [[e[k] for k in ("root_id", "latent_world_id", "realization_id")] for e in selected_samples],
        "source_files_sha256": sources, "artifacts_sha256": artifacts,
        "allowed_prefixes_sha256": file_hash(output / "allowed_prefixes.jsonl"),
        "maximum_input_tokens": max(len(value["prompt_ids"]) for value in registry.values()),
        "tokenizer_fixture": tokenizer_fixture,
        "PI1": "B1-101 forced PROCEED O2 minus forced PROCEED O1 in C2; separate natural-policy O2 minus O1 contrast. WRITE still executes after the current response.",
        "PI2": "B1-101 I2 minus I1, same forced question/response; only writer reply availability changes",
        "PI3": "12 historical selections, 8 effective policies, natural O2; no reselection using new bank"}
    plan["sha256"] = digest(plan)
    write_json(output / "plan.json", plan)
    return plan


def verify_plan(output):
    output = Path(output).resolve()
    plan = json.loads((output / "plan.json").read_text())
    if plan["sha256"] != digest({key: value for key, value in plan.items() if key != "sha256"}):
        raise ValueError("Frozen evaluation plan changed")
    if plan["source_files_sha256"] != _sources() or plan["runtime"] != _runtime():
        raise ValueError("Frozen evaluation source/runtime changed")
    for path, expected in plan["artifacts_sha256"].items():
        if file_hash(path) != expected:
            raise ValueError("Frozen evaluation artifact changed: " + path)
    if file_hash(output / "allowed_prefixes.jsonl") != plan["allowed_prefixes_sha256"]:
        raise ValueError("Frozen actor prefix registry changed")
    return plan


class GenerationBudget:
    def __init__(self, maximum, progress=None):
        self.maximum, self.calls, self.hits, self.requests = maximum, 0, 0, 0
        self.progress = progress

    def before_call(self):
        if self.calls >= self.maximum:
            raise RuntimeError("NEW_GENERATION_BUDGET_EXHAUSTED")
        self.calls += 1  # failed inference attempts still consume the budget
        if self.progress:
            self.progress()


class MultiPolicyCache:
    """Disk-backed greedy outputs, namespaced by verified effective policy."""
    def __init__(self, backend, fingerprint, runtime, directory, registry, budget):
        self.original, self.fingerprint, self.runtime = backend, fingerprint, runtime
        self.directory, self.registry, self.budget = Path(directory), registry, budget
        self.directory.mkdir(parents=True, exist_ok=True)
        self.last_seed = None

    def __getattr__(self, name):
        return getattr(self.original, name)

    def seed(self, seed):
        self.last_seed = seed
        return self.original.seed(seed)

    def generate(self, prompt, sample=False):
        request = digest({"prompt": prompt, "max_new_tokens": self.config.max_new_tokens})
        entry = self.registry.get(request)
        if entry is None or self.original.encode_prompt(prompt) != entry["prompt_ids"]:
            raise RuntimeError("UNPLANNED_OR_RETOKENIZED_PREFIX")
        if sample and self.last_seed is None:
            raise RuntimeError("Explicit stochastic generation seed required")
        namespace = {"effective_policy": self.fingerprint, "runtime": self.runtime,
            "precision": self.config.precision, "generation_batch_size": 1,
            "decoding": {"sample": sample, "temperature": self.config.temperature if sample else None,
                "top_k": 0, "top_p": 1., "max_new_tokens": self.config.max_new_tokens,
                "seed": self.last_seed if sample else None}}
        key = digest({"namespace": namespace, "request": request, "prompt_ids": entry["prompt_ids"]})
        path = self.directory / (key + ".json")
        self.budget.requests += 1
        hit = not sample and path.is_file()
        if hit:
            cached = json.loads(path.read_text())
            if (cached["namespace"] != namespace or cached["prompt"] != prompt
                    or cached.get("generation_sha256") != digest(cached["generation"])
                    or cached["generation"]["prompt_ids"] != entry["prompt_ids"]):
                raise RuntimeError("Corrupt greedy cache namespace")
            generated = Generation(**cached["generation"])
            self.budget.hits += 1
        else:
            if path.exists():
                raise FileExistsError("Stochastic seed/prefix reuse is forbidden")
            self.budget.before_call()
            generated = self.original.generate(prompt, sample=sample)
            if generated.prompt_ids != entry["prompt_ids"]:
                raise RuntimeError("Model generation prefix does not match frozen tokenizer")
            write_json(path, {"key": key, "namespace": namespace, "prompt": prompt,
                "generation": asdict(generated), "generation_sha256": digest(asdict(generated)),
                "input_token_ids_sha256": digest(generated.prompt_ids)})
        generated = deepcopy(generated)
        generated.metadata = {**(generated.metadata or {}), "varied_cache_key": key,
            "cache_hit": hit, "effective_policy_fingerprint": self.fingerprint,
            "stochastic_cache_reused": False}
        return generated


class ReferenceWriter(RuleController):
    def __init__(self, action, response):
        self.action, self.response = deepcopy(action), deepcopy(response)

    def select(self, observation):
        return deepcopy(self.action)

    def respond(self, observation):
        return deepcopy(self.response)


def _metrics(row):
    return {"C2": row["metrics"]["C2"], "FJS": row["metrics"]["FJS"],
        "J_primary": row["metrics"]["C2"] * row["metrics"]["FJS"],
        "unsupported_operations": row["metrics"]["unsupported_operations"],
        "unsupported_write_episode": row["metrics"]["unsupported_write_episode"],
        "invalid_outputs": row["metrics"]["invalid_outputs"], "questions": row["metrics"]["questions"],
        "R_long": row["rewards_by_component"]["R_long"], "R_short": row["rewards_by_component"]["R_short"]}


def summarize(rows, episodes):
    lookup = {e["root_id"]: e["authoring"]["stratum"] for e in episodes}
    roots = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        roots[row["root_id"]][row["latent_world_id"]][row["realization_id"]].append(_metrics(row))
    if not roots:
        return {"episodes": 0, "roots": 0, "root_weighted": None}
    metrics = tuple(_metrics(rows[0]))
    values = {rid: {m: mean(mean(mean(r[m] for r in replicas) for replicas in realizations.values()) for realizations in worlds.values())
                    for m in metrics} for rid, worlds in roots.items()}
    strata = sorted({lookup[r] for r in values})
    by_stratum = {s: {m: mean(v[m] for r, v in values.items() if lookup[r] == s) for m in metrics} for s in strata}
    return {"episodes": len(rows), "roots": len(values), "by_root": values, "by_stratum": by_stratum,
        "root_weighted": {m: mean(v[m] for v in values.values()) for m in metrics},
        "family_equal_weighted": {m: mean(v[m] for v in by_stratum.values()) for m in metrics},
        "episode_counts": {m: sum(_metrics(r)[m] for r in rows) for m in metrics}}


def _release():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def execute(output, *, device="cuda", backend_factory=None):
    """Execute a prepared plan once. No partial run is implicitly resumed."""
    output = Path(output).resolve()
    plan = verify_plan(output)
    if device != plan["device"]:
        raise ValueError("Device differs from frozen plan")
    if plan.get("tokenizer_fixture") and backend_factory is None:
        raise ValueError("A tokenizer fixture cannot authorize real model execution")
    if any((output / name).exists() for name in ("execution_started.json", "trajectories.jsonl", "completion.json")):
        raise FileExistsError("Execution already started; preserve partial evidence and use a new destination")
    episodes, _ = load_bank(plan["bank"], expected_roots=plan["roots"])
    registry = {r["request"]: r for r in (json.loads(line) for line in (output / "allowed_prefixes.jsonl").open())}
    # Stored prefix entries include their request field, which is not part of
    # tokenization identity; strip it before reconstructing the case registry.
    encoder_map = {entry["prompt"]: entry["prompt_ids"] for entry in registry.values()}
    cases, _ = enumerate_cases(episodes, plan["prompt_root"], lambda p: encoder_map[p])
    cases_by_id = {case["case_id"]: case for case in cases}
    budget = GenerationBudget(plan["maximum_new_generations"])
    started, rows_written = time.monotonic(), 0
    stage = {"name": "starting"}
    last_print = [started]
    def progress(force=False):
        value = {"status": "executed" if stage["name"] == "completed" else "running", "stage": stage, "new_calls": budget.calls,
            "cache_hits": budget.hits, "requests": budget.requests,
            "trajectories": rows_written, "elapsed_seconds": time.monotonic() - started}
        write_json(output / "progress.json", value)
        if force or time.monotonic() - last_print[0] >= 15.:
            print(canonical(value), flush=True)
            last_print[0] = time.monotonic()
    budget.progress = progress
    write_json(output / "execution_started.json", {"started_at": now(), "plan_sha256": plan["sha256"],
        "training": False, "optimizer_steps": 0, "execution_scope": "injected_test_backend" if backend_factory else "real_frozen_local_LLM"})
    states = output / "states"
    states.mkdir()
    primary, contrasts, diagnostic_rows, selector_rows, group_rows, weight_checks = defaultdict(list), defaultdict(list), [], [], [], []
    natural_b1 = {}
    def emit(stream, row, case, arm, intended, representative, *, writer="model", action="NATURAL"):
        nonlocal rows_written
        row.update(evaluation_version=VERSION, review_arm=arm, writer=writer,
            diagnostic_version=VERSION,
            diagnostic_case_id=case["case_id"], observable_state_id=case["state_id"],
            intended_model=intended, loaded_representative=representative,
            J_primary=row["metrics"]["C2"] * row["metrics"]["FJS"], action_id=action,
            training=False, optimizer_steps=0, checkpoint_reselected=False)
        evaluator = Evaluator()
        row["scores"] = {"current": evaluator.score_current(row["trace"]),
                         "future": evaluator.score_future(row["trace"]),
                         "memory": evaluator.score_memory(row["trace"])}
        rows_written += 1
        row["trace_line"] = rows_written
        payload = canonical(row)
        stream.write(payload + "\n")
        stream.flush()
        row["trace_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
        return row
    def trajectory(case, controller, identifier, **options):
        path = states / (identifier + ".sqlite")
        if path.exists():
            raise FileExistsError("A trajectory must have its own physical SQLite state")
        row = run_episode(case["episode"], controller, db_path=path, **options)
        with sqlite3.connect("file:" + str(path.resolve()) + "?mode=ro", uri=True) as connection:
            persisted = [json.loads(item[0]) for item in connection.execute("SELECT payload FROM cm_records ORDER BY seq")]
        if persisted != row["trace"]["final_memory"] or persisted != row["trace"]["reuse_observation"]["memory"]:
            raise RuntimeError("PERSISTED_SESSION3_MEMORY_MISMATCH")
        row.update(sqlite_path=str(path.resolve()), sqlite_sha256=file_hash(path))
        return row
    try:
        with (output / "trajectories.jsonl").open("x") as stream, (output / "selector_samples.jsonl").open("x") as selectors:
            for policy_index, policy in enumerate(plan["policy_classes"]):
                representative = policy["representative"]
                stage.update(name="primary", policy_index=policy_index, representative=representative)
                progress(True)
                factory = backend_factory or (lambda checkpoint, device: SpecLocalPolicy.from_checkpoint(checkpoint, trainable=False, device=device))
                backend = factory(representative["checkpoint_path"], device)
                if any(p.requires_grad for p in backend.model.parameters()):
                    raise RuntimeError("Frozen evaluation requires every parameter frozen")
                before = backend.weights_fingerprint() if backend_factory else parameter_hash(backend)
                cached = MultiPolicyCache(backend, policy["fingerprint"], plan["runtime"], output / "generation_cache", registry, budget)
                backend.config.temperature = 1.
                for label_index, intended in enumerate(policy["labels"]):
                    arm = f'{intended["method"]}_{intended["training_seed"]}'
                    for ci, case in enumerate(cases):
                        row = trajectory(case, LLMController(cached, prompt_root=plan["prompt_root"]),
                            f"primary_{policy_index}_{label_index}_{ci}", method=intended["method"], rollout_seed=intended["training_seed"])
                        emit(stream, row, case, arm, intended, representative)
                        primary[arm].append(row)
                        if intended["method"] == "B1" and intended["training_seed"] == 101:
                            natural_b1[case["case_id"]] = row
                        stage.update(arm=arm, cases_completed=ci + 1, cases_scheduled=len(cases))
                        progress(True)
                if any(l["method"] == "B1" and l["training_seed"] == 101 for l in policy["labels"]):
                    intended = next(l for l in policy["labels"] if l["method"] == "B1" and l["training_seed"] == 101)
                    stage.update(name="B1_101_action_diagnosis")
                    for ci, case in enumerate(cases):
                        natural = natural_b1[case["case_id"]]
                        natural_action = action_id(natural["parsed_actions"]["select"])
                        # Existing analysis helpers only receive new v3 rows,
                        # generated from the new real environment and prompts.
                        enrich(natural, case, "NATURAL", "B1", natural_action)
                        diagnostic_rows.append(compact_action(natural))
                        for ai, action in enumerate(ACTIONS):
                            opts = action_options(action)
                            if action == "PROCEED":
                                opts["method"] = "C-NEVER"
                            model = trajectory(case, LLMController(cached, prompt_root=plan["prompt_root"]),
                                f"action_{ci}_{ai}", rollout_seed=101, **opts)
                            enrich(model, case, action, "B1", natural_action)
                            emit(stream, model, case, "B1_101_action_matrix", intended, representative, action=action, writer="B1")
                            diagnostic_rows.append(compact_action(model))
                            if action == "PROCEED":
                                contrasts["B1_101_O2_no_question"].append(model)
                            reference = trajectory(case, ReferenceWriter(model["parsed_actions"]["select"], model["trace"]["response"]),
                                f"reference_{ci}_{ai}", rollout_seed=101, **opts)
                            for part in ("initial_memory", "user_reply", "response"):
                                if model["trace"][part] != reference["trace"][part]:
                                    raise ValueError("Reference writer changed pre-WRITE evidence")
                            if model["trace"]["observations"]["write"] != reference["trace"]["observations"]["write"]:
                                raise ValueError("Reference writer received different evidence")
                            enrich(reference, case, action, "reference", natural_action)
                            reference["model_response_source_line"] = model["trace_line"]
                            emit(stream, reference, case, "B1_101_reference_matrix", intended, representative, action=action, writer="reference")
                            diagnostic_rows.append(compact_action(reference))
                        reference_natural = trajectory(case, ReferenceWriter(natural["parsed_actions"]["select"], natural["trace"]["response"]),
                            f"reference_natural_{ci}", rollout_seed=101, method="B1_natural_reference_writer")
                        if natural["trace"]["observations"]["write"] != reference_natural["trace"]["observations"]["write"] or natural["trace"]["response"] != reference_natural["trace"]["response"]:
                            raise ValueError("Natural writer replacement changed delivered evidence or response")
                        reference_natural["model_response_source_line"] = natural["trace_line"]
                        emit(stream, reference_natural, case, "B1_101_reference_writer_natural", intended, representative, writer="reference")
                        contrasts["B1_101_reference_writer_natural"].append(reference_natural)
                        o1 = trajectory(case, LLMController(cached, prompt_root=plan["prompt_root"]),
                            f"O1_{ci}", condition="O1", rollout_seed=101, method="B1")
                        emit(stream, o1, case, "B1_101_O1", intended, representative)
                        contrasts["B1_101_O1"].append(o1)
                        o1_fixed = trajectory(case, LLMController(cached, prompt_root=plan["prompt_root"]),
                            f"O1_no_question_{ci}", condition="O1", rollout_seed=101, method="C-NEVER")
                        emit(stream, o1_fixed, case, "B1_101_O1_no_question", intended, representative, action="PROCEED")
                        contrasts["B1_101_O1_no_question"].append(o1_fixed)
                        pairs = {}
                        for intervention in ("I1", "I2"):
                            paired = trajectory(case, LLMController(cached, prompt_root=plan["prompt_root"]),
                                f"{intervention}_{ci}", intervention=intervention, forced_slot=case["reference_slot"], rollout_seed=101, method="B1")
                            emit(stream, paired, case, "B1_101_" + intervention, intended, representative)
                            contrasts["B1_101_" + intervention].append(paired)
                            pairs[intervention] = paired
                        for part in ("response", "initial_memory", "user_reply"):
                            if pairs["I1"]["trace"][part] != pairs["I2"]["trace"][part]:
                                raise ValueError("PI2 pair changed a non-intervened component")
                        if pairs["I1"]["trace"]["observations"]["respond"] != pairs["I2"]["trace"]["observations"]["respond"]:
                            raise ValueError("PI2 RESPOND input mismatch")
                        stage.update(cases_completed=ci + 1, cases_scheduled=len(cases))
                        progress(True)
                    values = analyze_actions(diagnostic_rows)
                    write_json(output / "action_diagnostics.json", values)
                    by_state = {value["observable_state_id"]: value for value in values["states"]}
                    unique_states = {case["state_id"]: case for case in cases}
                    for ti, temperature in enumerate(plan["generation"]["sample_temperatures"]):
                        backend.config.temperature = temperature
                        stage.update(name="selector_sampling", temperature=temperature)
                        for si, (state_id, case) in enumerate(sorted(unique_states.items())):
                            for draw in range(4):
                                seed = 85100000 + ti * 100000 + si * 10 + draw
                                cached.seed(seed)
                                controller = LLMController(cached, prompt_root=plan["prompt_root"], sample=True)
                                error = None
                                try:
                                    selected = controller.select(deepcopy(case["select_observation"]))
                                except ValueError as exc:
                                    selected, error = {}, str(exc)
                                aid = action_id(selected)
                                av = by_state[state_id]["action_values"]
                                chosen = av.get(aid)
                                baseline = av.get(by_state[state_id]["B1_action"])
                                item = {"temperature": temperature, "observable_state_id": state_id,
                                    "draw": draw, "seed": seed, "action": aid, "error": error,
                                    "phase_records": controller.serialized_records(),
                                    "action_values_greedy_continuation": chosen,
                                    "better_than_B1_greedy": bool(chosen and baseline and chosen["R_long"] > baseline["R_long"] + 1e-9),
                                    "regret": max(value["R_long"] for value in av.values()) - chosen["R_long"] if chosen else None,
                                    "scope": "SELECT sample only; frozen greedy action values, not full sampled return"}
                                selector_rows.append(item)
                                selectors.write(canonical(item) + "\n")
                                selectors.flush()
                            stage.update(states_completed=si + 1, states_scheduled=len(unique_states))
                            progress(True)
                    for ti, temperature in enumerate(plan["generation"]["sample_temperatures"]):
                        backend.config.temperature = temperature
                        stage.update(name="full_sampled_groups", temperature=temperature)
                        for ei, episode in enumerate(sampled_episodes(episodes)):
                            cid = digest([episode[k] for k in ("root_id", "latent_world_id", "realization_id")])[:24]
                            case, group = cases_by_id[cid], []
                            for draw in range(4):
                                seed = 95100000 + ti * 100000 + ei * 10 + draw
                                row = trajectory(case, LLMController(cached, prompt_root=plan["prompt_root"], sample=True),
                                    f"sampled_{ti}_{ei}_{draw}", rollout_seed=seed, method="frozen_GRPO_signal_probe")
                                row.update(temperature=temperature, stochastic_replica=draw, rollout_seed=seed)
                                emit(stream, row, case, "B1_101_full_sampling", intended, representative)
                                phases = [PhaseSample(record["phase"], Generation(**record["generation"])) for record in row["trace"]["phase_records"]]
                                if [phase.phase for phase in phases] != ["select", "respond", "write"]:
                                    raise ValueError("Incomplete sampled trajectory prefixes")
                                components = row["rewards_by_component"]["components"]
                                group.append(EpisodeRollout(episode["root_id"], episode["latent_world_id"], seed, phases,
                                    components["current"], components["future"], row["metrics"]["questions"],
                                    row["metrics"]["unsupported_operations"], row["metrics"]["invalid_outputs"],
                                    str(states / f"sampled_{ti}_{ei}_{draw}.sqlite"), "validation"))
                            if len({tuple(r.phases[0].generation.prompt_ids) for r in group}) != 1:
                                raise ValueError("A sampled group changed pre-ASK state")
                            group_rows.append({"temperature": temperature, "root_id": episode["root_id"],
                                "latent_world_id": episode["latent_world_id"], "stratum": episode["authoring"]["stratum"],
                                "long": group_telemetry(group, "long"), "short": group_telemetry(group, "short"),
                                "optimizer_steps": 0, "scope": "frozen complete on-policy diagnostic trajectories"})
                            write_json(output / "group_diagnostics.json", group_rows)
                            progress(True)
                    backend.config.temperature = 1.
                after = backend.weights_fingerprint() if backend_factory else parameter_hash(backend)
                if before != after or any(p.requires_grad for p in backend.model.parameters()):
                    raise RuntimeError("Frozen model weights changed")
                weight_checks.append({"representative": representative, "effective_fingerprint": policy["fingerprint"], "before": before, "after": after, "frozen": True})
                # The SELECT sampling controller otherwise retains cached ->
                # backend after the B1-101 branch and into the next GPU load.
                controller = None
                del cached, backend
                _release()
        verify_plan(output)
        summary = {"status": "executed", "training": False, "optimizer_steps": 0,
            "execution_scope": "injected_test_backend" if backend_factory else "real_frozen_local_LLM",
            "checkpoint_selection": False, "confirmatory_test_opened": False,
            "primary": {arm: summarize(rows, episodes) for arm, rows in primary.items()},
            "contrasts_101": {arm: summarize(rows, episodes) for arm, rows in contrasts.items()},
            "PI2_pairs_verified": len(episodes), "action_diagnostics_path": str(output / "action_diagnostics.json"),
            "selector_samples": len(selector_rows), "selector_summary": {str(t): {
                "samples": sum(row["temperature"] == t for row in selector_rows),
                "invalid": sum(row["temperature"] == t and row["action"] == "INVALID" for row in selector_rows),
                "states_visiting_improvement": len({row["observable_state_id"] for row in selector_rows if row["temperature"] == t and row["better_than_B1_greedy"]})} for t in (1., 1.3)},
            "full_sampled_groups": len(group_rows), "weights": weight_checks,
            "calls": {"actual_generations": budget.calls, "cache_hits": budget.hits,
                "requests": budget.requests, "cap": budget.maximum, "conservative_bound": plan["conservative_new_generation_bound"]},
            "trajectories": rows_written, "elapsed_seconds": time.monotonic() - started,
            "scope": "Synthetic prospective development, one realization, fixed symbolic reader; not confirmatory, human-validated or a new learning experiment"}
        write_json(output / "summary.json", summary)
        completion = {**{k: summary[k] for k in ("status", "execution_scope", "training", "optimizer_steps", "calls", "trajectories", "elapsed_seconds")},
            "completed_at": now(), "plan_sha256": plan["sha256"],
            "artifacts_sha256": {str(p.relative_to(output)): file_hash(p) for p in output.rglob("*") if p.is_file() and p.name not in {"progress.json", "completion.json"}}}
        write_json(output / "completion.json", completion)
        stage.update(name="completed")
        progress(True)
        return summary
    except Exception as exc:
        write_json(output / "failure.json", {"status": "failed", "error_type": type(exc).__name__,
            "error": str(exc), "stage": stage, "new_calls": budget.calls, "trajectories": rows_written,
            "recovery": "Preserve every partial artifact; no implicit retries or checkpoint reselection"})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--bank", required=True)
    prepare_parser.add_argument("--checkpoint-review", required=True)
    prepare_parser.add_argument("--prompt-root", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--maximum-new-generations", type=int, default=4000)
    prepare_parser.add_argument("--device", default="cuda")
    execute_parser = sub.add_parser("execute")
    execute_parser.add_argument("--output", required=True)
    execute_parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.bank, args.checkpoint_review, args.prompt_root, args.output,
                         maximum_new_generations=args.maximum_new_generations, device=args.device)
        print(canonical({"status": "prepared_no_inference", "bound": result["conservative_new_generation_bound"], "maximum_input_tokens": result["maximum_input_tokens"]}))
    else:
        execute(args.output, device=args.device)


if __name__ == "__main__":
    main()
