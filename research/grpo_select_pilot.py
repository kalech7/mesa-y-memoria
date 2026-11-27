"""Train the Qwen LoRA selector with GRPO while freezing RESPOND and WRITE."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import time

from research.competence_data import load_and_check
from research.competence_evaluation import (
    DEFAULT_THRESHOLDS, _run_trajectory, aggregate_rows, evaluate_selected,
)
from research.competence_training import _release
from research.grpo_select_checkpoint import load_selector
from research.legal_selection_policy import ACTION_CATALOG, LegalSelectionPolicy
from research.reward_memory_credit import memory_credit
from research.reward_review_experiment import informative_question
from research.src.artifacts import canonical, digest, file_hash, now, write_json
from research.src.env.episode import EpisodeEnvironment
from research.src.policy.llm_controller import LLMController, MAX_INPUT_TOKENS
from research.src.training.episodic_grpo import group_advantages
from research.src.training.sft import parameter_hash
from research.weak_sft_experiment import selection_record


REPO = Path(__file__).resolve().parents[1]
PARENT = REPO / ("research/runs/imperfections_campaign_20261101_001/campaigns/"
                 "arm_00/adaptation/101/training/step_000840")
MODEL_PATH = REPO / "models/Qwen2.5-0.5B-Instruct"
PROMPTS = REPO / "research/targeted_prompts_v2"
SEED = 101
DRAWS = 4
BETA = .02
CLIP = .20
LR = 3e-5


class SplitController(LLMController):
    """Same frozen Qwen policy for RESPOND and WRITE across both arms."""

    def __init__(self, selector, downstream, *, forced_action=None):
        super().__init__(selector, prompt_root=PROMPTS)
        self.downstream = downstream
        self.forced_action = forced_action

    def select(self, observation):
        if self.forced_action is not None:
            return json.loads(ACTION_CATALOG[self.forced_action])
        return super().select(observation)

    def call(self, phase, observation, instruction_suffix=None):
        if phase == "select":
            return super().call(phase, observation, instruction_suffix)
        selector, self.backend = self.backend, self.downstream
        try:
            return super().call(phase, observation, instruction_suffix)
        finally:
            self.backend = selector


def schedule(episodes, rounds):
    """Each round sees one ambiguous root in both worlds and two clear roots."""
    ambiguous, controls = defaultdict(list), defaultdict(list)
    for episode in episodes:
        if episode["split"] != "train":
            raise ValueError("GRPO only accepts TRAIN episodes")
        if episode["authoring"]["stratum"] == "ambiguous_duration":
            ambiguous[episode["root_id"]].append(episode)
        else:
            controls[episode["authoring"]["stratum"]].append(episode)
    rng = random.Random(SEED)
    roots = sorted(ambiguous)
    rng.shuffle(roots)
    buckets = [sorted(rows, key=lambda row: row["root_id"]) for _, rows in sorted(controls.items())]
    for bucket in buckets:
        rng.shuffle(bucket)
    clear = [row for group in zip(*buckets) for row in group]
    if len(roots) < rounds or len(clear) < 2 * rounds:
        raise ValueError("Insufficient roots for the fixed ambiguous/control coverage")
    rows = []
    for step, root in enumerate(roots[:rounds]):
        pair = {episode["truth"]["scope_kind"]: episode for episode in ambiguous[root]}
        if len(ambiguous[root]) != 2 or set(pair) != {"ongoing", "session"}:
            raise ValueError("An ambiguous TRAIN root needs one ongoing and one session world")
        rows.extend([pair["ongoing"], clear[2 * step], pair["session"], clear[2 * step + 1]])
    return rows


def preflight(policy, splits):
    """Tokenize every possible branch using only authentic public observations."""
    controller = LLMController(policy, prompt_root=PROMPTS)
    report, train = {"limit": MAX_INPUT_TOKENS, "episodes": []}, []
    actions = [json.loads(text) for text in ACTION_CATALOG]
    with TemporaryDirectory(prefix="grpo-selector-preflight-") as work:
        for role, episodes in splits.items():
            for index, episode in enumerate(episodes):
                values = {}
                for action_index, action in enumerate(actions):
                    path = Path(work) / f"{role}_{index:04d}_{action_index}.sqlite"
                    environment = EpisodeEnvironment(episode, path)
                    try:
                        prefixes = {"select": environment.observe("select", "O2")}
                        if action["action"] == "ASK":
                            environment.ask(action["slot"])
                        prefixes.update({phase: environment.observe(phase, "O2")
                                         for phase in ("respond", "write")})
                    finally:
                        environment.dispose()
                    key = action["slot"] or action["action"]
                    values[key] = {phase: len(policy.encode_prompt(controller.prompt(phase, obs)))
                                   for phase, obs in prefixes.items()}
                maximum = max(tokens for branch in values.values() for tokens in branch.values())
                report["episodes"].append({"role": role, "root_id": episode["root_id"],
                    "world_id": episode["latent_world_id"], "token_counts": values,
                    "max_input_tokens": maximum, "within_limit": maximum <= MAX_INPUT_TOKENS})
                if role == "train" and maximum <= MAX_INPUT_TOKENS:
                    train.append(episode)
    oversized_roots = {row["root_id"] for row in report["episodes"]
                       if row["role"] == "train" and not row["within_limit"]}
    report["excluded_train_roots"] = sorted(oversized_roots)
    return report, [episode for episode in train if episode["root_id"] not in oversized_roots]


def scores(row):
    metrics, trace = row["metrics"], row["trace"]
    memory = memory_credit(trace, {"success": metrics["FJS"]}, metrics["unsupported_write_count"])
    informative = bool(informative_question(trace))
    return {
        "C2": metrics["C2"], "FJS": metrics["FJS"],
        "strict_joint": float(bool(metrics["C2"] and metrics["FJS"] and
                                   not metrics["any_invalid_output"] and not metrics["unsupported_write_episode"])),
        "informative_questions": int(informative), "questions": metrics["questions"],
        "uninformative_questions": int(bool(metrics["questions"]) and not informative),
        "causal_write_delta": memory["benefit"], "invalid_output": metrics["any_invalid_output"],
        "unsupported_write": metrics["unsupported_write_episode"], "raw_return": metrics["R_long"],
    }


def policy_return(metrics):
    # The task signal separates semantic progress from the ASK penalty.
    task = (2 * metrics["strict_joint"] + .5 * metrics["C2"] + .5 * metrics["FJS"]
            + .5 * metrics["causal_write_delta"])
    penalties = (.10 * metrics["questions"] + .50 * metrics["unsupported_write"]
                 + .20 * metrics["invalid_output"])
    return task, task - penalties


def categorical_loss(torch, current, old, reference, actions, advantages):
    ratios = (current[actions] - old[actions]).exp()
    policy = -torch.minimum(ratios * advantages, ratios.clamp(1 - CLIP, 1 + CLIP) * advantages).mean()
    kl = (current.exp() * (current - reference)).sum()
    return policy + BETA * kl, kl, ratios


def evaluate(selector, downstream, episodes, output, method, *, authorization=None):
    result = evaluate_selected(SplitController(selector, downstream), episodes, output,
                               prompt_root=PROMPTS, method=method, training_seed=SEED,
                               role="evaluation" if authorization else "validation",
                               protocol={"thresholds": DEFAULT_THRESHOLDS,
                                         "heldback_authorization": authorization},
                               include_interventions=False, include_teacher=False)
    if result["status"] != "completed":
        raise RuntimeError(f"{method} evaluation failed: {result['failure']}")
    with (output / "trajectories.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream]
    for row in rows:
        row["metrics"] = scores(row)
    return aggregate_rows(rows)


def choose(baseline, checkpoints):
    initial = baseline["metrics"]
    eligible = [(step, report["metrics"]) for step, report in checkpoints.items()
                if report["metrics"]["invalid_output"] <= initial["invalid_output"] + 1e-9
                and report["metrics"]["unsupported_write"] <= initial["unsupported_write"] + 1e-9]
    best, _ = max(eligible, key=lambda item: (
        item[1]["strict_joint"], item[1]["FJS"],
        -item[1]["uninformative_questions"], -item[0],
    ))
    return best


def fit(actor, downstream, cohort, validation_bank, run, budget):
    torch = actor.torch
    params = [p for p in actor.model.parameters() if p.requires_grad]
    if not params or any(p.requires_grad for p in downstream.model.parameters()):
        raise RuntimeError("Expected trainable selector LoRA and frozen downstream Qwen")
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0)
    if downstream.device != actor.device:
        raise ValueError("Pilot uses the same device for actor and reference")
    started = time.monotonic()
    validation_seconds = 0.
    validation = {}
    controller = LLMController(actor, prompt_root=PROMPTS)
    branches = run / "train_branches"
    branches.mkdir()
    torch_rng = torch.Generator(device=actor.device).manual_seed(SEED)
    with (run / "grpo_groups.jsonl").open("x") as stream:
        for index, episode in enumerate(cohort, 1):
            if time.monotonic() - started - validation_seconds >= budget:
                raise TimeoutError("Declared training budget exhausted; partial traces preserved")
            inspection = EpisodeEnvironment(episode, branches / f"{index:02d}_prefix.sqlite")
            try:
                observation = inspection.observe("select", "O2")
            finally:
                inspection.dispose()
            prompt_ids = actor.encode_prompt(controller.prompt("select", observation))
            if len(prompt_ids) > MAX_INPUT_TOKENS:
                raise ValueError("SELECT input exceeds the protocol token limit")
            with torch.no_grad(), actor.likelihood_mode():
                _, old = actor._action_distribution(prompt_ids)
            with torch.no_grad(), downstream.likelihood_mode():
                _, reference = downstream._action_distribution(prompt_ids)
            actions = torch.multinomial(old.exp(), DRAWS, replacement=True, generator=torch_rng)
            cache = {}
            task_returns, rewards, evidence = [], [], []
            for action in actions.tolist():
                if action not in cache:
                    location = branches / f"{index:02d}_{action}"
                    branch_controller = SplitController(downstream, downstream, forced_action=action)
                    row = _run_trajectory(branch_controller, episode, location.with_suffix(".sqlite"),
                                          method="GRPO-SELECT-TRAIN", training_seed=SEED)
                    write_json(location.with_suffix(".json"), row)
                    if row["phases"]["select"]["observation_sha256"] != digest(observation):
                        raise RuntimeError("Branch SELECT input differs from the sampled action prefix")
                    cache[action] = (scores(row), str(location.with_suffix(".json")))
                metrics, evidence_path = cache[action]
                task, reward = policy_return(metrics)
                task_returns.append(task)
                rewards.append(reward)
                evidence.append({"action": json.loads(ACTION_CATALOG[action]),
                                 "metrics": metrics, "path": evidence_path})
            advantages = old.new_tensor(group_advantages(rewards))
            details = []
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                with actor.likelihood_mode():
                    _, current = actor._action_distribution(prompt_ids)
                    loss, kl, ratios = categorical_loss(torch, current, old, reference, actions, advantages)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite categorical GRPO loss")
                    loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                details.append({"loss": loss.item(), "KL": kl.item(), "grad_norm": gradient.item(),
                                "ratio_min": ratios.min().item(), "ratio_max": ratios.max().item()})
            record = {"group": index, "root": episode["root_id"], "world": episode["latent_world_id"],
                      "stratum": episode["authoring"]["stratum"], "prompt_sha256": digest(prompt_ids),
                      "task_returns": task_returns, "returns": rewards, "advantages": advantages.tolist(),
                      "semantic_signal": len(set(task_returns)) > 1, "rollouts": evidence, "optimization": details}
            stream.write(canonical(record) + "\n")
            stream.flush()
            print(f"GRPO {index}/{len(cohort)} · signal={record['semantic_signal']} · rewards={rewards}", flush=True)
            if index in {len(cohort) // 2, len(cohort)}:
                validation_start = time.monotonic()
                checkpoint = run / f"step_{index:06d}"
                actor.save_checkpoint(checkpoint)
                validation[index] = evaluate(actor, downstream, validation_bank,
                                             run / "validation" / checkpoint.name, "GRPO-SELECT")
                validation_seconds += time.monotonic() - validation_start
                print(f"validation step {index}: {validation[index]['metrics']}", flush=True)
    return validation


def run_pilot(bank, destination, *, rounds, budget, device, model_path):
    from transformers import AutoTokenizer

    splits, manifest = load_and_check(bank)
    destination.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    tokenization = LegalSelectionPolicy.__new__(LegalSelectionPolicy)
    tokenization.tokenizer = tokenizer
    token_report, eligible_train = preflight(tokenization, splits)
    preflight_path = destination / "TOKEN_PREFLIGHT.json"
    write_json(preflight_path, token_report)
    oversized_eval = [row for row in token_report["episodes"] if row["role"] != "train" and not row["within_limit"]]
    if oversized_eval:
        raise ValueError("A frozen validation/evaluation episode exceeds the full-branch token limit")
    cohort = schedule(eligible_train, rounds)
    source_paths = (Path(__file__), REPO / "research/grpo_select_checkpoint.py",
                    REPO / "research/legal_selection_policy.py", REPO / "research/phase_temperature_policy.py",
                    REPO / "research/competence_evaluation.py", REPO / "research/competence_data.py",
                    REPO / "research/reward_memory_credit.py", REPO / "research/reward_review_experiment.py",
                    REPO / "research/src/env/episode.py", REPO / "research/src/scoring/evidence.py",
                    REPO / "research/src/policy/llm_controller.py", REPO / "llm_local_001/model.py",
                    *sorted(PROMPTS.glob("*.txt")))
    snapshot = destination / "source_snapshot"
    for path in source_paths:
        copy = snapshot / path.relative_to(REPO)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
    write_json(destination / "PILOT_PLAN.json", {
        "created_at": now(), "scope": "exploratory SELECT-only GRPO, frozen Qwen RESPOND/WRITE, local verifier",
        "seed": SEED, "device": device, "parent": selection_record("SFT", PARENT, 0),
        "model_path": str(model_path), "token_preflight_sha256": file_hash(preflight_path),
        "train_candidate_filter": "Exclude complete TRAIN roots with any public branch above the existing token limit",
        "fresh_synthetic_bank": str(bank), "data_manifest_sha256": file_hash(bank / "manifest.json"),
        "data_hashes": {role: digest(rows) for role, rows in splits.items()},
        "cohort": [{k: row[k] for k in ("root_id", "latent_world_id", "realization_id")} for row in cohort],
        "human_review": manifest["human_review"], "draws": DRAWS, "passes": 2, "lr": LR, "beta": BETA,
        "clip": CLIP, "budget_seconds": budget,
        "reward": "2*strict_joint + .5*C2 + .5*FJS + .5*causal_write_delta - .10*ASK - .50*unsupported_write - .20*invalid_output",
        "selection": "validation only, include step 0, no increased invalid/unsupported rates; rank strict joint, future, fewer uninformative questions, earlier step",
        "source_sha256": {str(path.relative_to(REPO)): file_hash(path) for path in source_paths},
    })
    downstream = load_selector(PARENT, device=device, model_path=model_path)
    actor = None
    try:
        frozen_hash = parameter_hash(downstream)
        baseline = evaluate(downstream, downstream, splits["validation"],
                            destination / "validation" / "SFT", "SFT")
        actor = load_selector(PARENT, trainable=True, device=device, model_path=model_path)
        adapter_before = parameter_hash(actor, trainable=True)
        backbone_before = parameter_hash(actor, trainable=False)
        if parameter_hash(actor) != frozen_hash:
            raise ValueError("Initial actor and frozen SFT checkpoint differ")
        validations = {0: baseline, **fit(actor, downstream, cohort, splits["validation"], destination, budget)}
        adapter_changed = adapter_before != parameter_hash(actor, trainable=True)
        if parameter_hash(actor, trainable=False) != backbone_before or parameter_hash(downstream) != frozen_hash:
            raise RuntimeError("A frozen parameter changed")
        _release(actor)
        actor = None
        best = choose(baseline, validations)
        methods = {"SFT": (PARENT, 0), "GRPO-SELECT": (
            destination / f"step_{best:06d}" if best else PARENT, best),
            "GRPO-terminal": (destination / f"step_{len(cohort):06d}", len(cohort))}
        selection_path = destination / "SELECTION.json"
        write_json(selection_path, {
            "version": "competence_selection_v1", "selection_complete": True,
            "selection_split": "validation", "selected_before_evaluation": True,
            "heldback_episodes_sha256": digest(splits["evaluation"]),
            "rows": [selection_record(name, path, step) for name, (path, step) in methods.items()],
            "metrics": {step: row["metrics"] for step, row in validations.items()},
        })
        authorization = {"selection_manifest_path": str(selection_path), "selection_manifest_sha256": file_hash(selection_path),
                         "locked_bank_sha256": digest(splits["evaluation"])}
        panels, cache = {}, {}
        for name, (path, step) in methods.items():
            if path in cache:
                panels[name] = {**panels[cache[path]], "same_checkpoint_as": cache[path]}
                continue
            selector = downstream if step == 0 else load_selector(path, device=device, model_path=model_path)
            try:
                panels[name] = evaluate(selector, downstream, splits["evaluation"],
                                        destination / "evaluation" / name, name, authorization=authorization)
                cache[path] = name
            finally:
                if step:
                    _release(selector)
        with (destination / "grpo_groups.jsonl").open() as stream:
            groups = [json.loads(line) for line in stream]
        baseline_roots = panels["SFT"]["by_root"]
        paired = {name: dict(Counter(
            "improved" if row["strict_joint"] > baseline_roots[root]["strict_joint"]
            else "worse" if row["strict_joint"] < baseline_roots[root]["strict_joint"] else "tied"
            for root, row in panel["by_root"].items()
        )) for name, panel in panels.items()}
        result = {"status": "completed_exploratory", "completed_grpo_groups": len(groups),
                  "optimizer_steps": 2 * len(groups), "adapter_changed": adapter_changed,
                  "task_signal_groups": sum(g["semantic_signal"] for g in groups),
                  "informative_asks_in_draws": sum(r["metrics"]["informative_questions"] for g in groups for r in g["rollouts"]),
                  "distinct_train_branches": sum(1 for _ in (destination / "train_branches").glob("*.json")),
                  "selected_step": best, "paired_strict_joint_roots": paired, "panels": panels}
        write_json(destination / "RESULT.json", result)
        return {name: panel["metrics"] for name, panel in panels.items()}
    finally:
        _release(actor)
        _release(downstream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--budget-seconds", type=int, default=900)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    if args.rounds < 1 or args.budget_seconds < 1:
        parser.error("rounds and budget-seconds must be positive")
    print(canonical(run_pilot(args.bank.resolve(), args.output.resolve(),
                              rounds=args.rounds, budget=args.budget_seconds, device=args.device,
                              model_path=args.model_path.resolve())))


if __name__ == "__main__":
    main()
