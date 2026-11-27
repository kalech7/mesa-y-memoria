"""A balanced SFT adjustment followed by one monitored GRPO seed."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import random
from statistics import mean
import time

from research import ai_reward_judge
from research.ai_feedback_experiment import JUDGE, ai_task, request_verdicts
from research.competence_data import load_and_check
from research.competence_evaluation import DEFAULT_THRESHOLDS, aggregate_rows, evaluate_selected
from research.competence_training import PROMPTS, _release, checkpoint_evidence, profile_teacher, verify_initialization
from research.joint_reward_experiment import evaluation_summary, joint_success
from research.reward_review_experiment import components
from research.src.artifacts import canonical, digest, file_hash, now, write_json
from research.src.closure_protocol import execution_context
from research.src.policy.llm_controller import SpecLocalPolicy
from research.src.training.episodic_grpo import group_advantages, train_grpo
from research.src.training.rollouts import make_rollout_collector
from research.src.training.sft import train_sft
from research.weak_sft_experiment import SEED, authorize, evaluate, read, selection_record
from research.weak_sft_qa import _rows

SFT_STEPS = 60
RL_STEPS = 16


def strict_metrics(summary):
    rows = _rows(summary["paths"]["trajectories"])
    for row in rows:
        m = row["metrics"]
        m["J_strict"] = joint_success(m["C2"], m["FJS"], row["phases"]["write"]["error"] is None,
                                      m["unsupported_write_count"])
    metrics = aggregate_rows(rows)["metrics"]
    return {**metrics, "utility": metrics["J_strict"], "current_failure": 1 - metrics["C2"]}


class Validation:
    def __init__(self, episodes, destination, method):
        self.episodes, self.destination, self.method = episodes, destination, method
        self.calls = 0

    def __call__(self, policy):
        output = self.destination / f"evaluation_{self.calls:02d}"
        self.calls += 1
        summary = evaluate_selected(policy, self.episodes, output, prompt_root=PROMPTS, method=self.method,
            training_seed=SEED, role="validation", protocol={"thresholds": DEFAULT_THRESHOLDS},
            include_interventions=False, include_teacher=False)
        if summary["status"] != "completed":
            raise RuntimeError(f"Validation failed: {summary['failure']}")
        metrics = strict_metrics(summary)
        write_json(output / "strict_metrics.json", metrics)
        print("validation", self.method, "strict", metrics["J_strict"], "current", metrics["C2"], flush=True)
        return metrics


def selection(metrics):
    return {"utility_tolerance": 1e-6, "validity_guards": {
        "current_failure": min(1., 1 - metrics["C2"] + .05),
        "unsupported_write_episode": min(1., metrics["unsupported_write_episode"] + .02),
        "any_invalid_output": min(1., metrics["any_invalid_output"] + .02),
    }}


def cohort(rows, count):
    by_family = defaultdict(dict)
    for row in sorted(rows, key=lambda e: (e["root_id"], e["latent_world_id"], e["realization_id"])):
        by_family[row["scenario_family"]].setdefault(row["root_id"], row)
    rng = random.Random(7331)
    buckets = []
    for family in sorted(by_family):
        bucket = list(by_family[family].values())
        rng.shuffle(bucket)
        buckets.append(bucket)
    return [row for batch in zip(*buckets) for row in batch][:count]


def sft_adjust(run, parent, examples, validation, initial):
    actor = SpecLocalPolicy.from_checkpoint(parent, trainable=True, device="cuda")
    try:
        write_json(run / "SFT_TOKEN_PROFILE.json", profile_teacher(actor, examples))
        with execution_context(run, "training"):
            return train_sft(actor, examples, run / "sft/optimizer",
                Validation(validation, run / "sft/validation", "SFT-balanced"), seed=SEED,
                continuation=True, initial_validation_metrics=initial, updates=SFT_STEPS, epochs=1,
                checkpoint_every=30, learning_rate=5e-5, budget_seconds=180,
                checkpoint_save_budget_seconds=120, prompt_root=PROMPTS,
                closure_authorization=run, closure_selection=selection(initial))
    finally:
        _release(actor)


def rl_train(run, parent, episodes, validation, initial):
    actor = SpecLocalPolicy.from_checkpoint(parent, trainable=True, device="cuda")
    reference = None
    try:
        reference = SpecLocalPolicy.from_checkpoint(parent, trainable=False, device="cpu")
        actor.config.temperature = reference.config.temperature = 1.3
        write_json(run / "RL_INITIALIZATION.json", verify_initialization(actor, parent, reference))
        collect = make_rollout_collector(run / "rl/environment_rollouts", method="RL-live", prompt_root=PROMPTS)
        pending, groups = {}, 0
        with (run / "AI_FEEDBACK.jsonl").open("x") as stream, execution_context(run, "training"):
            def rollout(policy, episode, seed):
                nonlocal groups
                if not pending:
                    batch, samples = [], []
                    for draw in range(4):
                        policy.seed(seed + draw)
                        result = collect(policy, episode, seed + draw)
                        row = read(Path(result.memory_path).with_suffix(".json"))
                        batch.append((result, row))
                        samples.append({"id": str(seed + draw), "evidence": ai_reward_judge.judge_input(row)})
                    started = time.monotonic()
                    verdicts = request_verdicts(run, f"group_{seed}", samples)
                    group_parts = []
                    for (result, row), sample in zip(batch, samples, strict=True):
                        raw = verdicts[sample["id"]]
                        parts = components(row)
                        task, relief, points = ai_task(parts, raw)
                        verdict = {**raw, "model": JUDGE, "input_sha256": digest(sample["evidence"]),
                                   "raw_output": canonical(raw), "elapsed_seconds": (time.monotonic()-started)/4}
                        stream.write(canonical({"seed": result.rollout_seed, "root_id": result.root_id,
                            "verdict": verdict, "components": parts, "task": task, "ask_cost_waiver": relief,
                            "feedback_points": points}) + "\n")
                        stream.flush()
                        pending[result.rollout_seed] = replace(result, r2=task, r3=relief)
                        group_parts.append((parts, row["metrics"]))
                    groups += 1
                    values = {
                        "rollout/J_strict": mean(p["joint"] for p, _ in group_parts),
                        "rollout/C2": mean(m["C2"] for _, m in group_parts),
                        "rollout/FJS": mean(m["FJS"] for _, m in group_parts),
                        "rollout/unsupported_write_rate": mean(m["unsupported_operations"] > 0 for _, m in group_parts),
                        "rollout/helpful_write_rate": mean(p["benefit"] > 0 for p, _ in group_parts),
                    }
                    with (run / "rl/optimizer/training_metrics.jsonl").open("a") as metrics_stream:
                        metrics_stream.write(canonical({"step": groups, "values": values}) + "\n")
                return pending.pop(seed)
            return train_grpo(actor, reference, episodes, rollout, run / "rl/optimizer",
                Validation(validation, run / "rl/validation", "RL-live"), horizon="long", updates=RL_STEPS,
                seed=SEED, roots_per_batch=1, rollouts_per_root=4, learning_rate=1e-5,
                kl_coefficient=.02, clip_epsilon=.2, checkpoint_every=8, budget_seconds=3000,
                initial_validation_metrics=initial, closure_authorization=run,
                checkpoint_save_budget_seconds=120, closure_selection=selection(initial),
                reference_execution="cpu_storage_gpu_compute")
    finally:
        _release(reference)
        _release(actor)


def audit(run, result, examples):
    train = read(run / "sft/optimizer/training.json")
    if train["resource_counters"]["updates"] != SFT_STEPS:
        raise ValueError("Incomplete SFT")
    if train["frozen_before"] != train["frozen_after"]:
        raise ValueError("SFT changed base weights")
    updates = _rows(run / "sft/optimizer/updates.jsonl")
    if Counter(row["phase"] for row in updates) != Counter(row["phase"] for row in examples):
        raise ValueError("Unbalanced SFT schedule")
    if result["resource_counters"]["updates"] != RL_STEPS:
        raise ValueError("Incomplete RL")
    if result["frozen_before"] != result["frozen_after"] or result["reference_before"] != result["reference_after"]:
        raise ValueError("RL changed frozen weights")
    labels, rows = _rows(run / "AI_FEEDBACK.jsonl"), _rows(run / "rl/optimizer/rollouts.jsonl")
    groups, stats = defaultdict(list), Counter()
    for label, row in zip(labels, rows, strict=True):
        rollout = row["rollout"]
        evidence = read(Path(rollout["memory_path"]).with_suffix(".json"))
        verdict = label["verdict"]
        parts = components(evidence)
        task, relief, _ = ai_task(parts, verdict)
        m = evidence["metrics"]
        expected = task + relief - .1*m["questions"] - .5*m["unsupported_operations"] - .2*m["invalid_outputs"]
        if label["seed"] != rollout["rollout_seed"] or abs(row["return"]-expected)>1e-9:
            raise ValueError("Feedback/reward mismatch")
        if digest(ai_reward_judge.judge_input(evidence)) != verdict["input_sha256"] or label["components"] != parts:
            raise ValueError("Judge evidence mismatch")
        groups[row["group_id"]].append((row["return"], row["advantage"]))
        stats["helpful_writes"] += parts["benefit"] > 0
        stats["unsupported_writes"] += m["unsupported_operations"] > 0
        stats["joint_successes"] += parts["joint"]
    if len(rows) != RL_STEPS*4:
        raise ValueError("Missing rollouts")
    for group in groups.values():
        rewards, advantages = zip(*group)
        if any(abs(a-b)>1e-9 for a,b in zip(advantages,group_advantages(rewards),strict=True)):
            raise ValueError("Advantage mismatch")
    return {**stats, "seed":SEED,"sft_updates":SFT_STEPS,"rl_updates":RL_STEPS,"judge_calls":len(labels),
            "sft_adapter_changed":train["trainable_before"]!=train["trainable_after"],
            "rl_adapter_changed":result["trainable_before"]!=result["trainable_after"],
            "sft_selected_step":train["best_step"],"rl_selected_step":result["best_step"]}


def execute(fresh_dir, run):
    previous = Path("research/runs/weak_sft_natural_20261104_001").resolve()
    old_plan = read(previous / "PILOT_PLAN.json")
    original, original_manifest = load_and_check(Path(old_plan["bank_dir"]))
    fresh, fresh_manifest = load_and_check(fresh_dir)
    parent = Path("research/runs/reward_review_20261104_002/warmup/optimizer/last").resolve()
    training = cohort(original["train"], RL_STEPS)
    teaching = cohort(original["train"], 20)
    chosen_roots = {row["root_id"] for row in teaching}
    examples = [row for row in _rows(Path(old_plan["bank_dir"])/original_manifest["teachers"]["train"]["path"])
                if row["route"] == "natural" and row["root_id"] in chosen_roots]
    # Keep one world per root for this short, balanced supervised pass.
    identities = {(row["root_id"],row["latent_world_id"],row["realization_id"]) for row in teaching}
    examples = [row for row in examples if (row["root_id"],row["latent_world_id"],row["realization_id"]) in identities]
    if Counter(row["phase"] for row in examples) != {"select":20,"respond":20,"write":20}:
        raise ValueError("Expected sixty balanced TRAIN demonstrations")
    sources = {str(path.resolve()):file_hash(path) for path in (
        Path(__file__),Path(ai_reward_judge.__file__),Path(__file__).with_name("ai_feedback_experiment.py"),
        Path(__file__).with_name("reward_review_experiment.py"),Path(__file__).with_name("reward_memory_credit.py"))}
    plan = {"created_at":now(),"seed":SEED,"scope":"One monitored seed, brief balanced SFT, GRPO with AI feedback",
        "parent":checkpoint_evidence(parent),"source_sha256":sources,
        "train_sha256":digest(original["train"]),"examples_sha256":digest(examples),
        "rl_cohort_sha256":digest(training),"fresh_bank":str(fresh_dir),"fresh_data_seed":fresh_manifest["seed"],
        "validation_sha256":digest(fresh["validation"]),"heldback_sha256":digest(fresh["evaluation"]),
        "sft":{"updates":SFT_STEPS,"learning_rate":5e-5,"checkpoints":[0,30,60],"phase_counts":dict(Counter(r["phase"] for r in examples))},
        "rl":{"updates":RL_STEPS,"learning_rate":1e-5,"temperature":1.3,"checkpoints":[0,8,16],"judge":JUDGE},
        "selection":"Maximize strict joint validation success. Earliest ties. Current accuracy may fall <=5 pp; unsupported and invalid may rise <=2 pp relative to that branch's initial model."}
    run.mkdir(parents=True)
    authorize(run,fresh_dir,plan)
    write_json(run/"SFT_EXAMPLES.json",examples)
    baseline=SpecLocalPolicy.from_checkpoint(parent,trainable=False,device="cuda")
    try:
        initial=Validation(fresh["validation"],run/"baseline_validation","SFT-start")(baseline)
    finally:
        _release(baseline)
    warm=sft_adjust(run,str(parent),examples,fresh["validation"],initial)
    parent_selected=warm["selected_checkpoint"]
    sft_initial=next(row["validation"] for row in warm["history"] if row["step"]==warm["best_step"])
    write_json(run/"SFT_SELECTION.json",{"selected":checkpoint_evidence(parent_selected),
        "selected_step":warm["best_step"],"initial_metrics":initial,"selected_metrics":sft_initial})
    print("starting RL",parent_selected,flush=True)
    result=rl_train(run,parent_selected,training,fresh["validation"],sft_initial)
    checked=audit(run,result,examples)
    write_json(run/"QA_TRAINING.json",checked)
    if any(file_hash(path)!=sha for path,sha in sources.items()):
        raise ValueError("Sources changed during run")
    rows=[selection_record("SFT-start",parent,0),
        selection_record("SFT-balanced",parent_selected,warm["best_step"]),
        selection_record("RL-live",result["selected_checkpoint"],result["best_step"]),
        selection_record("RL-live-terminal",result["final_checkpoint"],RL_STEPS)]
    select_path=run/"SELECTION.json"
    write_json(select_path,{"version":"competence_selection_v1","created_at":now(),"selection_complete":True,
        "selection_split":"validation","selected_before_evaluation":True,"heldback_episodes_sha256":plan["heldback_sha256"],"rows":rows})
    summaries=evaluate(run,fresh["evaluation"],select_path)
    panels={method:evaluation_summary(Path(record["summary"]),fresh["evaluation"],select_path,
        read(record["summary"])["method"],SEED) for method,record in summaries.items()}
    comparisons={}
    for method in ("RL-live","RL-live-terminal"):
        diffs=[value-panels["SFT-balanced"]["by_root"][rid] for rid,value in panels[method]["by_root"].items()]
        comparisons[method]={"delta_J_pp":100*mean(diffs),"improved":sum(d>0 for d in diffs),
            "worse":sum(d<0 for d in diffs),"tied":sum(d==0 for d in diffs)}
    write_json(run/"COMPARISON.json",{"status":"verified","training":checked,"panels":panels,"paired_vs_SFT":comparisons,
        "evaluation_roots":len(panels["RL-live"]["by_root"]),"evaluation_episodes":len(fresh["evaluation"])})
    print("completed comparison",{k:v["metrics"]["J_primary"] for k,v in panels.items()},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    execute(args.fresh.resolve(),args.output.resolve())


if __name__=="__main__":
    main()
