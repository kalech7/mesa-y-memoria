"""Compare supervised and GRPO continuations of the same food-memory selector."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from itertools import cycle, zip_longest
import json
from pathlib import Path
import random
import shutil
from statistics import mean
import time

from research.competence_data import teacher_examples
from research.competence_evaluation import DEFAULT_THRESHOLDS, _run_trajectory, aggregate_rows, evaluate_selected
from research.competence_training import _release
from research.grpo_select_checkpoint import load_selector
from research.grpo_select_pilot import (
    BETA, CLIP, DRAWS, LR, MODEL_PATH, PROMPTS, REPO, SplitController, categorical_loss,
    choose, policy_return, preflight, scores,
)
from research.legal_selection_policy import ACTION_CATALOG, LegalSelectionPolicy
from research.src.analysis.paired_statistics import paired_bootstrap
from research.src.artifacts import canonical, digest, file_hash, now, write_json
from research.src.env.episode import EpisodeEnvironment
from research.src.policy.llm_controller import LLMController
from research.src.training.episodic_grpo import group_advantages
from research.src.training.sft import TrainingBudget, parameter_hash
from research.weak_sft_experiment import selection_record


SEEDS = (101, 202, 303)
METHODS = ("SFT", "SFT-adicional-SELECT", "GRPO-SELECT")
PASSES = 2
CHECKPOINT_FRACTIONS = (.5, 1.)
CLEAN_CAMPAIGN = (REPO / "research/runs/imperfections_campaign_20261101_001"
                  / "campaigns/arm_00/adaptation")
PARENT_MANIFESTS = {
    101: "e08fba2332b9397464a2dc9ced715f315cd20a507a93c54b51dc7ad2fec4a4af",
    202: "5c5f7070d21468d69d8700de49a4afd30ce7a5a81cff6f58e07dac386444a61e",
    303: "3f686b725a2cefc07d5b9e4c6657d2ddd816a51b9f1154c4d30363ebe12665a6",
}


@dataclass(frozen=True)
class TrainingCase:
    episode: dict
    observation_sha256: str
    prompt_ids: tuple[int, ...]
    teacher_action: int


class FrozenOutputCache:
    """Reuse greedy responses from the same frozen model and exact prompt."""

    def __init__(self, policy):
        self.policy = policy
        self.generations = {}
        self.hits = 0

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def generate(self, prompt, sample=False):
        if sample:
            raise ValueError("Frozen branch caching requires greedy decoding")
        key = (prompt, self.config.max_new_tokens)
        if key in self.generations:
            self.hits += 1
        else:
            self.generations[key] = self.policy.generate(prompt, sample=False)
        return deepcopy(self.generations[key])


def episode_id(episode):
    return tuple(episode[key] for key in ("root_id", "latent_world_id", "realization_id"))


def ordered_train(episodes, seed):
    """One shared, root-balanced pass per seed, keeping worlds of a root together."""
    if not episodes or any(row["split"] != "train" for row in episodes):
        raise ValueError("Comparison training requires nonempty TRAIN-only episodes")
    identities = [episode_id(row) for row in episodes]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate TRAIN episode")
    grouped = defaultdict(lambda: defaultdict(list))
    root_families = {}
    for row in episodes:
        root, family = row["root_id"], row["scenario_family"]
        if root_families.setdefault(root, family) != family:
            raise ValueError("TRAIN root has conflicting families")
        grouped[family][root].append(row)
    rng = random.Random(seed)
    families = sorted(grouped)
    rng.shuffle(families)
    if "F4" in families:
        families.remove("F4")
        families.insert(0, "F4")
    buckets = []
    for family in families:
        roots = sorted(grouped[family])
        rng.shuffle(roots)
        buckets.append([sorted(grouped[family][root], key=episode_id) for root in roots])
    return [episode for block in zip_longest(*buckets)
            for root in block if root is not None for episode in root]


def prepare_training_cases(episodes, tokenizer):
    """Bind the clean observable teacher to the exact SELECT prefix shared by both methods."""
    controller = LLMController(tokenizer, prompt_root=PROMPTS)
    catalog = [json.loads(action) for action in ACTION_CATALOG]
    examples = {}
    labels_by_prefix = {}
    for row in teacher_examples(episodes, routes=("natural",)):
        if row["phase"] != "select":
            continue
        observation = row["observation"]
        action = catalog.index(row["action"])
        prefix = digest(observation)
        prompt = tuple(tokenizer.encode_prompt(controller.prompt("select", observation)))
        if labels_by_prefix.setdefault(prompt, action) != action:
            raise ValueError("Identical SELECT inputs have conflicting teacher labels")
        examples[episode_id(row)] = (prefix, prompt, action)
    if set(examples) != {episode_id(row) for row in episodes}:
        raise ValueError("Missing or unexpected observable SELECT teacher examples")
    return {episode_id(row): TrainingCase(row, *examples[episode_id(row)]) for row in episodes}


def checkpoint_record(method, checkpoint, step, seed):
    return {**selection_record(method, checkpoint, step), "training_seed": seed}


def parent_records(seeds):
    records = {}
    for seed in seeds:
        parent = CLEAN_CAMPAIGN / str(seed) / "training/step_000840"
        record = checkpoint_record("SFT", parent, 0, seed)
        if record["checkpoint_manifest_sha256"] != PARENT_MANIFESTS[seed]:
            raise ValueError(f"Historical clean SFT parent changed for seed {seed}")
        records[seed] = record
    return records


def evaluate(selector, downstream, episodes, output, method, seed, *, authorization=None, cache=None):
    previous_hits = cache.hits if cache is not None else 0
    summary = evaluate_selected(
        SplitController(selector, downstream if cache is None else cache), episodes, output, prompt_root=PROMPTS,
        method=method, training_seed=seed, role="evaluation" if authorization else "validation",
        protocol={"thresholds": DEFAULT_THRESHOLDS, "heldback_authorization": authorization},
        include_interventions=False, include_teacher=False,
    )
    if summary["status"] != "completed":
        raise RuntimeError(f"{method} evaluation failed: {summary['failure']}")
    traces = output / "trajectories.jsonl"
    with traces.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    for row in rows:
        row["metrics"] = scores(row)
    return {**aggregate_rows(rows), "trajectories": str(traces), "trajectories_sha256": file_hash(traces),
            "frozen_output_cache_hits": cache.hits - previous_hits if cache is not None else 0}


def train_sft_group(actor, optimizer, parameters, case, _rollout):
    """Negative log likelihood over the same five actions used by GRPO."""
    optimization = []
    for _ in range(PASSES):
        with actor.likelihood_mode():
            _, current = actor._action_distribution(case.prompt_ids)
            optimization.append(update(actor, optimizer, parameters, -current[case.teacher_action]))
    return {"target_action": json.loads(ACTION_CATALOG[case.teacher_action]), "optimization": optimization}


def update(actor, optimizer, parameters, loss):
    if not actor.torch.isfinite(loss):
        raise RuntimeError("Nonfinite categorical training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = actor.torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    return {"loss": loss.item(), "gradient_norm": gradient.item()}


def make_grpo_step(downstream, seed, output):
    torch = downstream.torch
    rng = torch.Generator(device=downstream.device).manual_seed(seed)
    branches = output / "train_branches"
    branches.mkdir()
    frozen_cache = FrozenOutputCache(downstream)

    def step(actor, optimizer, parameters, case, index):
        with torch.no_grad():
            with actor.likelihood_mode():
                _, old = actor._action_distribution(case.prompt_ids)
            with downstream.likelihood_mode():
                _, reference = downstream._action_distribution(case.prompt_ids)
        actions = torch.multinomial(old.exp(), DRAWS, replacement=True, generator=rng)
        cache, draws = {}, []
        for action in actions.tolist():
            if action not in cache:
                stem = branches / f"{index:06d}_{action}"
                controller = SplitController(downstream, frozen_cache, forced_action=action)
                row = _run_trajectory(controller, case.episode, stem.with_suffix(".sqlite"),
                                      method="GRPO-SELECT-TRAIN", training_seed=seed)
                if row["phases"]["select"]["observation_sha256"] != case.observation_sha256:
                    raise RuntimeError("GRPO rollout prefix differs from the observable SFT prefix")
                trace = stem.with_suffix(".json")
                write_json(trace, row)
                metrics = scores(row)
                task, reward = policy_return(metrics)
                cache[action] = {"action": json.loads(ACTION_CATALOG[action]), "metrics": metrics,
                                 "task_return": task, "return": reward, "trajectory": str(trace)}
            draws.append(cache[action])
        advantages = old.new_tensor(group_advantages([draw["return"] for draw in draws]))
        optimization = []
        for _ in range(PASSES):
            with actor.likelihood_mode():
                _, current = actor._action_distribution(case.prompt_ids)
                loss, kl, ratios = categorical_loss(torch, current, old, reference, actions, advantages)
                optimization.append({**update(actor, optimizer, parameters, loss),
                                     "KL": kl.item(), "ratio_min": ratios.min().item(),
                                     "ratio_max": ratios.max().item()})
        return {"draws": draws, "semantic_signal": len({draw["task_return"] for draw in draws}) > 1,
                "advantages": advantages.tolist(), "optimization": optimization}

    return step


def train_continuation(actor, downstream, method, cases, validation_bank, baseline,
                       output, seed, seconds, validation_cache):
    """Spend the same allocation in each arm, with validation at the same fractions."""
    output.mkdir(parents=True)
    initial_hash = parameter_hash(actor)
    frozen_hash = parameter_hash(downstream)
    backbone_hash = parameter_hash(actor, trainable=False)
    adapter_before = parameter_hash(actor, trainable=True)
    if initial_hash != frozen_hash or any(p.requires_grad for p in downstream.model.parameters()):
        raise ValueError("Continuations must start at their identical frozen seed parent")
    parameters = [p for p in actor.model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable LoRA parameters")
    optimizer = actor.torch.optim.AdamW(parameters, lr=LR, weight_decay=0)
    step = make_grpo_step(downstream, seed, output) if method == METHODS[2] else train_sft_group
    validations = {0: baseline}
    paths = {0: Path(downstream.origin_checkpoint)}
    budget = TrainingBudget(seconds, CHECKPOINT_FRACTIONS)
    groups_with_task_signal = 0
    budget.start()
    with (output / "training.jsonl").open("x", encoding="utf-8") as stream:
        for index, case in enumerate(cycle(cases), 1):
            record = step(actor, optimizer, parameters, case, index)
            if actor.device.type == "cuda":
                actor.torch.cuda.synchronize(actor.device)
            groups_with_task_signal += int(record.get("semantic_signal", False))
            stream.write(canonical({"group": index, "episode": episode_id(case.episode),
                                    "prompt_sha256": digest(case.prompt_ids),
                                    "training_seconds": budget.elapsed(0.), **record}) + "\n")
            stream.flush()
            if budget.checkpoint_due(0.):
                started = time.monotonic()
                checkpoint = output / f"step_{index:06d}"
                actor.save_checkpoint(checkpoint)
                paths[index] = checkpoint
                validations[index] = evaluate(actor, downstream, validation_bank,
                                              output / "validation" / checkpoint.name, method, seed,
                                              cache=validation_cache)
                budget.pause(time.monotonic() - started)
                print(f"{seed} {method} · grupo {index} · J estricto "
                      f"{validations[index]['metrics']['strict_joint']:.1%}", flush=True)
                if not budget.pending:
                    break
    budget.finish()
    if parameter_hash(actor, trainable=False) != backbone_hash or parameter_hash(downstream) != frozen_hash:
        raise RuntimeError("Training changed the backbone or frozen response/write policy")
    selected = choose(baseline, validations)
    result = {"method": method, "training_seed": seed, "allocated_seconds": seconds,
              "training_seconds": budget.elapsed(0.), "validation_and_save_seconds": budget.paused_seconds,
              "overrun_seconds": max(0., budget.elapsed(0.) - seconds), "groups": index,
              "optimizer_steps": index * PASSES, "groups_with_task_signal": groups_with_task_signal,
              "adapter_changed": adapter_before != parameter_hash(actor, trainable=True),
              "initial_policy_sha256": initial_hash, "frozen_policy_sha256": frozen_hash,
              "selection": checkpoint_record(method, paths[selected], selected, seed),
              "validation": {step: panel["metrics"] for step, panel in validations.items()}}
    write_json(output / "TRAINING.json", result)
    return result


def frozen_sources():
    from research.competence_evaluation import _source_hashes

    files = {Path(path) for path in _source_hashes(PROMPTS)}
    files.update(REPO / name for name in (
        "research/tesis_bank.py", "research/tesis_comparison.py", "research/grpo_select_pilot.py",
        "research/grpo_select_checkpoint.py", "research/legal_selection_policy.py",
        "research/phase_temperature_policy.py", "research/reward_review_experiment.py",
        "research/reward_memory_credit.py", "research/src/analysis/paired_statistics.py",
        "research/src/training/episodic_grpo.py", "research/src/training/sft.py",
    ))
    return sorted(files)


def freeze_plan(output, bank, model_path, splits, manifest, token_report, parents,
                cases, seconds, device, *, resamples):
    output.mkdir(parents=True)
    write_json(output / "TOKEN_PREFLIGHT.json", token_report)
    snapshots = {}
    for source in frozen_sources():
        relative = source.relative_to(REPO)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        snapshots[str(relative)] = file_hash(target)
    plan = {"created_at": now(), "status": "exploratory", "scope": "SELECT-only continuation",
            "bank": str(bank), "bank_manifest_sha256": file_hash(bank / "manifest.json"),
            "bank_review": manifest.get("human_review"), "device": device, "model": str(model_path),
            "split_hashes": {role: digest(rows) for role, rows in splits.items()},
            "split_roots": {role: len({row["root_id"] for row in rows}) for role, rows in splits.items()},
            "token_preflight_sha256": file_hash(output / "TOKEN_PREFLIGHT.json"),
            "clean_parents": parents, "seeds": list(parents), "methods": list(METHODS),
            "frozen_components": ["backbone", "RESPOND", "WRITE", "fixed_symbolic reader"],
            "train_prefixes": {seed: [{"episode": episode_id(case.episode),
                                       "observation_sha256": case.observation_sha256,
                                       "prompt_sha256": digest(case.prompt_ids),
                                       "teacher_action": case.teacher_action}
                                      for case in ordered] for seed, ordered in cases.items()},
            "continuation": {"allocation_seconds_per_arm_per_seed": seconds,
                             "included": "time to collect GRPO rollouts and optimize LoRA",
                             "excluded": "model loading, token preflight, validation and checkpoint saving",
                             "checkpoint_fractions": CHECKPOINT_FRACTIONS,
                             "atomic_overrun": "complete a group started before its budget threshold",
                             "optimizer": "AdamW", "learning_rate": LR, "weight_decay": 0.,
                             "passes_per_group": PASSES, "draws_per_grpo_group": DRAWS,
                             "grpo_kl_to_parent": BETA, "grpo_clip": CLIP,
                             "grpo_return": "2*strict_joint + .5*C2 + .5*FJS + .5*write_delta"
                                            " - .1*ASK - .5*unsupported_write - .2*invalid_output",
                             "frozen_greedy_cache": "exact RESPOND/WRITE prompt and output token cap, "
                                                    "separate by seed and partition",
                             "sft_target": "clean observable teacher, categorical negative log likelihood"},
            "selection": {"validation_only": True, "step_zero_eligible": True,
                          "guards": "no rise in unsupported writing or invalid output",
                          "order": ["strict_joint", "FJS", "fewer uninformative questions", "earlier step"]},
            "statistics": {"primary_metric": "strict_joint", "future_metric": "FJS",
                           "bootstrap": "paired roots stratified by family, conditional on trained seeds",
                           "confidence": .95, "resamples": resamples}, "source_sha256": snapshots}
    write_json(output / "PLAN.json", plan)
    return plan


def assert_plan_unchanged(plan, output, bank, splits):
    if file_hash(bank / "manifest.json") != plan["bank_manifest_sha256"]:
        raise ValueError("Bank manifest changed after training plan freeze")
    for role, rows in splits.items():
        if digest(rows) != plan["split_hashes"][role]:
            raise ValueError(f"Frozen {role} content changed")
    for relative, expected in plan["source_sha256"].items():
        if (file_hash(REPO / relative) != expected
                or file_hash(output / "source_snapshot" / relative) != expected):
            raise ValueError(f"Frozen comparison source changed: {relative}")


def selected_rows(seeds, parents, training):
    result = []
    for seed in seeds:
        result.append(parents[seed])
        for method in METHODS[1:]:
            row = training[seed][method]["selection"]
            if (row["method"], row["training_seed"]) != (method, seed):
                raise ValueError("Unexpected method or seed in validation selection")
            if checkpoint_record(method, row["checkpoint_path"], row["step"], seed) != row:
                raise ValueError("Validation-selected artifacts changed")
            result.append(row)
    return result


@contextmanager
def policy_loaded(checkpoint, device, model_path, *, trainable=False):
    policy = load_selector(checkpoint, device=device, model_path=model_path, trainable=trainable)
    try:
        yield policy
    finally:
        _release(policy)


def evaluate_reserved(rows, parents, episodes, output, selection, device, model_path):
    panels = {}
    authorization = {"selection_manifest_path": str(selection), "selection_manifest_sha256": file_hash(selection),
                     "locked_bank_sha256": digest(episodes)}
    for seed, parent in parents.items():
        panels[seed], evaluated = {}, {}
        with policy_loaded(parent["checkpoint_path"], device, model_path) as downstream:
            cache = FrozenOutputCache(downstream)
            for row in rows:
                if row["training_seed"] != seed:
                    continue
                method, fingerprint = row["method"], row["policy_fingerprint"]
                if fingerprint in evaluated:
                    alias = evaluated[fingerprint]
                    panels[seed][method] = {**panels[seed][alias], "same_policy_as": alias}
                    continue
                destination = output / "evaluation" / str(seed) / method
                if row["step"] == 0:
                    panel = evaluate(downstream, downstream, episodes, destination, method, seed,
                                     authorization=authorization, cache=cache)
                else:
                    with policy_loaded(row["checkpoint_path"], device, model_path) as selector:
                        panel = evaluate(selector, downstream, episodes, destination, method, seed,
                                         authorization=authorization, cache=cache)
                panels[seed][method] = panel
                evaluated[fingerprint] = method
    return panels


def summarize(panels, *, resamples):
    roots = []
    for seed, methods in panels.items():
        for method, panel in methods.items():
            roots.extend({"method": method, "seed": seed, "root_id": root,
                          "scenario_family": panel["families"][root], "metrics": metrics}
                         for root, metrics in panel["by_root"].items())
    comparisons = ((METHODS[2], METHODS[1]), (METHODS[2], METHODS[0]), (METHODS[1], METHODS[0]))
    return {
        "means": {method: {key: mean(methods[method]["metrics"][key] for methods in panels.values())
                           for key in next(iter(panels.values()))[method]["metrics"]}
                  for method in METHODS},
        "paired_root_contrasts": {
            f"{left} vs {right}": {
                key: paired_bootstrap(roots, left, right, key, resamples=resamples, confidence=.95)
                for key in ("strict_joint", "FJS", "uninformative_questions", "unsupported_write")
            }
            for left, right in comparisons
        },
    }


def run(bank, output, *, device, model_path, seconds, seeds, resamples, mode):
    from transformers import AutoTokenizer
    from research.tesis_bank import load_and_check

    if output.exists():
        raise FileExistsError("Comparison requires a fresh output directory")
    splits, manifest = load_and_check(bank)
    parents = parent_records(seeds)
    tokenizer = LegalSelectionPolicy.__new__(LegalSelectionPolicy)
    tokenizer.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    token_report, eligible_train = preflight(tokenizer, splits)
    prepared = prepare_training_cases(eligible_train, tokenizer)
    cases = {seed: [prepared[episode_id(row)] for row in ordered_train(eligible_train, seed)] for seed in seeds}
    plan = freeze_plan(output, bank, model_path, splits, manifest, token_report, parents, cases, seconds,
                       device, resamples=resamples)
    if any(not row["within_limit"] for row in token_report["episodes"] if row["role"] != "train"):
        raise ValueError("Frozen validation/evaluation includes an oversized branch; see TOKEN_PREFLIGHT.json")
    if mode == "plan":
        return {"status": "planned", "seeds": seeds, "output": str(output), "roots": plan["split_roots"],
                "excluded_train_roots": token_report["excluded_train_roots"]}
    training = {}
    for seed, parent in parents.items():
        training[seed] = {}
        checkpoint = parent["checkpoint_path"]
        print(f"{seed} · evaluando SFT limpio en validación", flush=True)
        with policy_loaded(checkpoint, device, model_path) as downstream:
            validation_cache = FrozenOutputCache(downstream)
            baseline = evaluate(downstream, downstream, splits["validation"],
                                output / "validation" / str(seed) / "SFT", "SFT", seed,
                                cache=validation_cache)
            for method in METHODS[1:]:
                print(f"{seed} · entrenando {method} por {seconds}s", flush=True)
                with policy_loaded(checkpoint, device, model_path, trainable=True) as actor:
                    training[seed][method] = train_continuation(
                        actor, downstream, method, cases[seed], splits["validation"], baseline,
                        output / "training" / str(seed) / method, seed, seconds, validation_cache)
    assert_plan_unchanged(plan, output, bank, splits)
    rows = selected_rows(seeds, parents, training)
    selection = output / "SELECTION.json"
    write_json(selection, {"version": "competence_selection_v1", "selection_complete": True,
                           "selection_split": "validation", "selected_before_evaluation": True,
                           "heldback_episodes_sha256": digest(splits["evaluation"]),
                           "plan_sha256": file_hash(output / "PLAN.json"), "rows": rows})
    if mode == "validation":
        return {"status": "validation_completed", "selection": str(selection)}
    panels = evaluate_reserved(rows, parents, splits["evaluation"], output, selection, device, model_path)
    result = {"status": "completed_exploratory", "scope": plan["scope"],
              "seeds": seeds, "selection_sha256": file_hash(selection), "training": training, "panels": panels,
              **summarize(panels, resamples=resamples)}
    write_json(output / "RESULTS.json", result)
    return {"status": result["status"], "seeds": seeds, "means": result["means"],
            "output": str(output / "RESULTS.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget-seconds", type=int, default=300)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=SEEDS)
    parser.add_argument("--resamples", type=int, default=2000)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--plan-only", action="store_true")
    stages.add_argument("--validation-only", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or args.budget_seconds < 1 or args.resamples < 1:
        parser.error("Use unique seeds and positive budget/resample counts")
    mode = "plan" if args.plan_only else "validation" if args.validation_only else "evaluation"
    print(canonical(run(args.bank.resolve(), args.output.resolve(), device=args.device,
                        model_path=args.model_path.resolve(), seconds=args.budget_seconds,
                        seeds=tuple(args.seeds), resamples=args.resamples, mode=mode)), flush=True)


if __name__ == "__main__":
    main()
