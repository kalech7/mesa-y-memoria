"""One seed, no additional SFT: causal-reuse environment and phase-credit RL."""
import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import random
from statistics import mean

from research import targeted_ai_reward_safe as feedback
from research.competence_data import load_and_check
from research.competence_evaluation import aggregate_rows, evaluate_selected, DEFAULT_THRESHOLDS
from research.competence_training import _release, checkpoint_evidence, verify_initialization
from research.joint_reward_experiment import evaluation_summary, joint_success
from research.live_rl_experiment import Validation, selection, strict_metrics
from research.src.artifacts import canonical, digest, file_hash, now, write_json
from research.src.closure_protocol import execution_context
from research.legal_selection_policy import LegalSelectionPolicy as SpecLocalPolicy
from research.phase_temperature_policy import verify_phase_profiles
from research.src.policy.rule_controller import RuleController
from research.src.training import credit_grpo
from research.src.training.rollouts import make_rollout_collector
from research.targeted_rl_review import review_train
from research.weak_sft_experiment import authorize, read, selection_record
from research.weak_sft_qa import _rows

SEED=101
UPDATES=16
PASSES=2
PARENT=Path("research/runs/imperfections_campaign_20261101_001/campaigns/arm_00/adaptation/101/training/step_000840")
PROMPTS=Path(__file__).resolve().with_name("targeted_prompts_v2")
TEMPERATURE=1.0
PHASE_TEMPERATURES={"select":1.0,"respond":.6,"write":.35}
PROBE=Path("reports/reward_review_targeted_20261105_001/exploration_legal_probe/RESULT.json")


def unnecessary_question(row):
    observation=row["trace"]["observations"]["select"]
    return int(bool(row["metrics"]["questions"]) and RuleController().select(observation)["action"]=="PROCEED")


class PromptValidation(Validation):
    def __call__(self,policy):
        output=self.destination/f"evaluation_{self.calls:02d}"
        self.calls+=1
        summary=evaluate_selected(policy,self.episodes,output,prompt_root=PROMPTS,method=self.method,
            training_seed=SEED,role="validation",protocol={"thresholds":DEFAULT_THRESHOLDS},
            include_interventions=False,include_teacher=False)
        if summary["status"]!="completed":
            raise RuntimeError(f"Validation failed: {summary['failure']}")
        metrics=strict_metrics(summary)
        rows=_rows(summary["paths"]["trajectories"])
        for row in rows:
            row["metrics"]["unnecessary_questions"]=unnecessary_question(row)
        metrics["unnecessary_questions"]=aggregate_rows(rows)["metrics"]["unnecessary_questions"]
        # On fifteen validation roots, minimum nonzero J change is >.03;
        # at most .001 is used solely to break exact J ties by question friction.
        metrics["utility"]=metrics["J_strict"]-.001*metrics["unnecessary_questions"]
        write_json(output/"strict_metrics.json",metrics)
        print("validation",self.method,"strict",metrics["J_strict"],"current",metrics["C2"],flush=True)
        return metrics


def evaluate(run,bank,selection_path):
    protocol={"thresholds":DEFAULT_THRESHOLDS,"heldback_authorization":{
        "selection_manifest_path":str(selection_path),"selection_manifest_sha256":file_hash(selection_path),
        "locked_bank_sha256":digest(bank)}}
    summaries={};equivalent={}
    for record in read(selection_path)["rows"]:
        method,key=record["method"],record["policy_fingerprint"]
        if key in equivalent:
            summaries[method]={**summaries[equivalent[key]],"same_adapter_as":equivalent[key]}
            continue
        actor=SpecLocalPolicy.from_checkpoint(record["checkpoint_path"],trainable=False,device="cuda")
        try:
            output=run/"evaluation"/method
            result=evaluate_selected(actor,bank,output,prompt_root=PROMPTS,method=method,
                training_seed=SEED,role="evaluation",protocol=protocol,
                include_interventions=False,include_teacher=False)
            if result["status"]!="completed":
                raise RuntimeError(f"Incomplete evaluation for {method}: {result['failure']}")
            summaries[method]={"summary":str(output/"summary.json")}
            equivalent[key]=method
            print("evaluated",method,"episodes",result["completed_episodes"],flush=True)
        finally:
            _release(actor)
    return summaries


def curriculum(rows):
    """Independent roots, family balance, F4 world types alternate prospectively."""
    families=defaultdict(lambda:defaultdict(list))
    for row in rows:
        families[row["scenario_family"]][row["root_id"]].append(row)
    rng=random.Random(7331)
    buckets=[]
    for family,roots in sorted(families.items()):
        keys=sorted(roots);rng.shuffle(keys)
        bucket=[]
        for index,key in enumerate(keys):
            variants=sorted(roots[key],key=lambda e:e["latent_world_id"])
            if family=="F4":
                preferred="ongoing" if index%2==0 else "session"
                variants.sort(key=lambda e:e["truth"]["scope_kind"]!=preferred)
            bucket.append(variants[0])
        buckets.append(bucket)
    chosen=[row for batch in zip(*buckets) for row in batch][:UPDATES]
    if len(chosen)!=UPDATES or len({r["root_id"] for r in chosen})!=UPDATES:
        raise ValueError(f"Expected {UPDATES} distinct TRAIN roots")
    return chosen


def semantic_selection(method,checkpoint,step):
    """Ignore only order of PEFT target_modules; retain all inference artifacts."""
    row=selection_record(method,checkpoint,step)
    evidence=checkpoint_evidence(checkpoint)
    manifest=read(Path(checkpoint)/"local_checkpoint.json")
    artifacts=dict(evidence["artifacts_sha256"])
    artifacts.pop("README.md",None)
    config=read(Path(checkpoint)/"adapter_config.json")
    if isinstance(config.get("target_modules"),list):
        config["target_modules"]=sorted(config["target_modules"])
    artifacts["adapter_config.json"]=digest(config)
    row["policy_fingerprint"]=digest(dict(base=manifest["base_artifacts_sha256"],
        precision=manifest["config"]["precision"],artifacts=artifacts,
        prompt_sha256={p.name:file_hash(p) for p in sorted(PROMPTS.glob("*.txt"))},
        selection_policy_source_sha256=file_hash(Path("research/legal_selection_policy.py")),
        policy_profile=read(PROBE)["phase_profile"],decoding="greedy_legal_category_then_token"))
    row["equivalence_rule"]="Only canonicalize PEFT target_modules order; greedy inference for all models"
    return row


def train(run,parent,training,validation,initial):
    actor=SpecLocalPolicy.from_checkpoint(parent,trainable=True,device="cuda")
    reference=None
    try:
        reference=SpecLocalPolicy.from_checkpoint(parent,trainable=False,device="cpu")
        profile=verify_phase_profiles(actor,reference)
        if profile!=read(PROBE)["phase_profile"]:
            raise ValueError("Training profile differs from actual frozen feasibility probe")
        write_json(run/"RL_INITIALIZATION.json",verify_initialization(actor,parent,reference))
        write_json(run/"PHASE_EXPLORATION.json",actor.phase_profile())
        collector=make_rollout_collector(run/"rl/environment_rollouts",method="RL-credit",prompt_root=PROMPTS)
        pending={};groups=0
        with (run/"AI_FEEDBACK.jsonl").open("x") as stream,execution_context(run,"training"):
            def collect(policy,episode,seed):
                nonlocal groups
                if not pending:
                    batch=[];samples=[]
                    for draw in range(4):
                        policy.seed(seed+draw)
                        result=collector(policy,episode,seed+draw)
                        evidence=read(Path(result.memory_path).with_suffix(".json"))
                        batch.append((result,evidence))
                        samples.append(dict(id=str(seed+draw),evidence=feedback.judge_payload(evidence)))
                    print("awaiting target-aware AI feedback",seed,flush=True)
                    verdicts=feedback.request_feedback(run,f"group_{seed}",samples)
                    rewards=[]
                    for (result,evidence),sample in zip(batch,samples,strict=True):
                        verdict=verdicts[sample["id"]]
                        reward=feedback.reward(evidence,verdict)
                        updated=replace(result,r2=reward["optimizer_r2"],r3=reward["optimizer_r3"])
                        pending[result.rollout_seed]=credit_grpo.with_credit(updated,reward["phase_rewards"])
                        stream.write(canonical(dict(seed=result.rollout_seed,root_id=result.root_id,
                            input_sha256=digest(sample["evidence"]),verdict=verdict,reward=reward))+"\n")
                        stream.flush();rewards.append(reward)
                    groups+=1
                    values={
                        "rollout/J_strict":mean(r["program_joint"] for r in rewards),
                        "rollout/reward_joint":mean(r["joint"] for r in rewards),
                        "rollout/C2":mean(r["program_current"] for r in rewards),
                        "rollout/FJS":mean(r["program_future"] for r in rewards),
                        "rollout/helpful_write_rate":mean(r["benefit"]>0 for r in rewards),
                        "rollout/unsafe_rate":mean(r["unsafe"] for r in rewards),
                        "rollout/unsupported_write_rate":mean(r["unsupported_operations"]>0 for r in rewards),
                    }
                    for phase in credit_grpo.PHASES:
                        values[f"credit/{phase}_reward"]=mean(r["phase_rewards"][phase] for r in rewards)
                    with (run/"rl/optimizer/training_metrics.jsonl").open("a") as metrics:
                        metrics.write(canonical(dict(step=groups,values=values))+"\n")
                return pending.pop(seed)
            return credit_grpo.train_credit_grpo(actor,reference,training,collect,run/"rl/optimizer",
                PromptValidation(validation,run/"rl/validation","RL-credit"),updates=UPDATES,seed=SEED,
                optimization_passes=PASSES,learning_rate=1e-5,kl_coefficient=.03,clip_epsilon=.2,
                checkpoint_every=8,budget_seconds=3000,initial_validation_metrics=initial,
                closure_authorization=run,closure_selection=selection(initial),checkpoint_save_budget_seconds=120)
    finally:
        _release(reference);_release(actor)


def strict_panel(path):
    rows=_rows(read(path)["paths"]["trajectories"])
    for row in rows:
        m=row["metrics"]
        m["J_strict"]=joint_success(m["C2"],m["FJS"],row["phases"]["write"]["error"] is None,m["unsupported_write_count"])
        m["unnecessary_questions"]=unnecessary_question(row)
    panel=aggregate_rows(rows)
    return dict(metrics=panel["metrics"],by_root={r:m["J_strict"] for r,m in panel["by_root"].items()},
                by_root_unnecessary_questions={r:m["unnecessary_questions"] for r,m in panel["by_root"].items()})


def execute(bank,run):
    probe=read(PROBE)
    if not probe["feasible"]:
        raise RuntimeError("TRAIN exploration did not pass feasibility; do not launch blind optimization")
    if probe["phase_temperatures"]!=PHASE_TEMPERATURES:
        raise RuntimeError("Exploration profile differs from declared probe")
    if probe["parent"]!=checkpoint_evidence(PARENT):
        raise RuntimeError("Training parent differs from the approved frozen probe parent")
    splits,manifest=load_and_check(bank)
    training=curriculum(splits["train"])
    source_paths=[Path(__file__),Path(credit_grpo.__file__),Path(feedback.__file__),
        Path("research/targeted_curriculum_data.py"),Path("research/live_rl_experiment.py"),
        Path("research/reward_memory_credit.py"),Path("research/ai_reward_judge.py"),
        Path("research/targeted_rl_review.py"),Path("research/src/training/episodic_grpo.py"),
        Path("llm_local_001/model.py"),Path("research/phase_temperature_policy.py"),
        Path("research/legal_selection_policy.py"),*sorted(PROMPTS.glob("*.txt"))]
    sources={str(p.resolve()):file_hash(p) for p in source_paths}
    plan=dict(created_at=now(),seed=SEED,sft_updates=0,no_additional_sft=True,
        scope="No new SFT; clean existing parent, legal categorical SELECT shared by both models, phase-credit safe-floor RL",
        likelihood_units={"select":"categorical action logprob repeated for the mean-loss interface",
                          "respond":"autoregressive token","write":"autoregressive token"},
        previous_development_attempt="research/runs/reward_review_targeted_20261105_001",
        previous_attempt_stop_sha256=file_hash(Path("reports/reward_review_targeted_20261105_001/REWARD_STOP.json")),
        exploration_probe_sha256=file_hash(PROBE),phase_profile=probe["phase_profile"],prompt_root=str(PROMPTS),
        parent=checkpoint_evidence(PARENT),bank_dir=str(bank),bank_manifest_sha256=file_hash(bank/"manifest.json"),
        source_sha256=sources,heldback_sha256=digest(splits["evaluation"]),
        validation_sha256=digest(splits["validation"]),train_sha256=digest(splits["train"]),
        preflight_sha256={str(p.resolve()):file_hash(p) for p in (
            Path("reports/reward_review_targeted_20261105_001/GUIDED_TOKEN_PREFLIGHT.json"),
            Path("reports/reward_review_targeted_20261105_001/GUIDED_PROMPTS.json"),
            Path("reports/reward_review_targeted_20261105_001/TEST_RESULTS_FINAL.json"),
            Path("reports/reward_review_targeted_20261105_001/QA_PHASE_TEMPERATURE_POLICY.json"),
            Path("reports/reward_review_targeted_20261105_001/QA_LEGAL_SELECTION_POLICY.json"),
            Path("reports/reward_review_targeted_20261105_001/QA_EXPLORACION_LEGAL_TRAIN.json"),
            Path("reports/reward_review_targeted_20261105_001/QA_CLEAN_PARENT.json"),
            Path("reports/reward_review_targeted_20261105_001/QA_LEGAL_GPU_RESOURCE.json"),
            Path("reports/reward_review_targeted_20261105_001/QA_LEGAL_GPU_WORSTCASE.json"),
            Path("reports/reward_review_targeted_20261105_001/QA_REWARD_SAFE_CPU.json"),
            Path("reports/reward_review_targeted_20261105_001/exploration_legal_probe/PLAN.json"),
            Path("reports/reward_review_targeted_20261105_001/exploration_legal_probe.py"),
            Path("reports/reward_review_targeted_20261105_001/AUDITORIA_ENTORNO_TRAIN.json"))},
        rl_cohort_sha256=digest(training),rl_schedule=[dict(root_id=e["root_id"],world_id=e["latent_world_id"]) for e in training],
        counter_interpretation="SELECT token counts are the chosen action representation; each categorical distribution scores five candidates. Counters are not complete forward-compute exposure; use elapsed time and explicit protocol.",
        rl=dict(updates=UPDATES,optimization_passes=PASSES,optimizer_steps=UPDATES*PASSES,rollouts_per_root=4,
            learning_rate=1e-5,base_temperature=TEMPERATURE,phase_temperatures=PHASE_TEMPERATURES,
            kl_coefficient=.03,phase_weight=1/3,
            checkpoints=[0,8,16],judge=feedback.JUDGE),
        selection="Strict joint primary; exact J ties prefer fewer pre-ASK-reference-unnecessary questions (utility=J-.001*u); earliest full ties; current drop<=5pp, invalid/unsupported increase<=2pp",
        comparison="Frozen SFT vs validation-selected RL and preregistered terminal diagnostic, same decoder/prompts/reader")
    run.mkdir(parents=True)
    authorize(run,bank,plan)
    write_json(run/"TRAIN_COHORT.json",training)
    actor=SpecLocalPolicy.from_checkpoint(PARENT,trainable=False,device="cuda")
    try:
        initial=PromptValidation(splits["validation"],run/"baseline_validation","SFT-frozen")(actor)
    finally:
        _release(actor)
    result=train(run,str(PARENT.resolve()),training,splits["validation"],initial)
    audit=review_train(run/"rl/optimizer")
    if not (audit["frozen_unchanged"] and audit["reference_unchanged"] and audit["adapter_changed"]):
        raise ValueError("Training integrity failed")
    labels=_rows(run/"AI_FEEDBACK.jsonl")
    if len(labels)!=UPDATES*4:
        raise ValueError("Incomplete AI feedback")
    for label,row in zip(labels,_rows(run/"rl/optimizer/rollouts.jsonl"),strict=True):
        evidence=read(Path(row["rollout"]["memory_path"]).with_suffix(".json"))
        if feedback.reward(evidence,label["verdict"])!=label["reward"]:
            raise ValueError("Feedback reward reconstruction failed")
        if digest(feedback.judge_payload(evidence))!=label["input_sha256"]:
            raise ValueError("Feedback evidence changed")
    write_json(run/"QA_TRAINING.json",audit)
    if any(file_hash(p)!=sha for p,sha in sources.items()):
        raise ValueError("Frozen source changed")
    chosen=run/"SELECTION.json"
    write_json(chosen,dict(version="competence_selection_v1",created_at=now(),selection_complete=True,
        selection_split="validation",selected_before_evaluation=True,
        heldback_episodes_sha256=plan["heldback_sha256"],rows=[
            semantic_selection("SFT-frozen",PARENT,0),
            semantic_selection("RL-credit",result["selected_checkpoint"],result["best_step"]),
            semantic_selection("RL-credit-terminal",result["final_checkpoint"],UPDATES)]))
    summaries=evaluate(run,splits["evaluation"],chosen)
    panels={}
    for method,value in summaries.items():
        summary=evaluation_summary(Path(value["summary"]),splits["evaluation"],chosen,
            read(value["summary"])["method"],SEED)
        strict=strict_panel(value["summary"])
        summary["metrics"]["unnecessary_questions"]=strict["metrics"]["unnecessary_questions"]
        panels[method]={**summary,"strict":strict}
    contrasts={}
    for method in ("RL-credit","RL-credit-terminal"):
        changes=[v-panels["SFT-frozen"]["strict"]["by_root"][rid] for rid,v in panels[method]["strict"]["by_root"].items()]
        contrasts[method]=dict(delta_J_strict_pp=100*mean(changes),improved=sum(d>0 for d in changes),
            worse=sum(d<0 for d in changes),tied=sum(d==0 for d in changes))
        ask_differences=[v-panels["SFT-frozen"]["strict"]["by_root_unnecessary_questions"][rid]
            for rid,v in panels[method]["strict"]["by_root_unnecessary_questions"].items()]
        contrasts[method]["delta_unnecessary_questions_pp"]=100*mean(ask_differences)
    write_json(run/"COMPARISON.json",dict(status="verified",sft_updates=0,seed=SEED,
        training=dict(rl_updates=UPDATES,optimizer_steps=UPDATES*PASSES,rl_selected_step=result["best_step"],
                      judge_calls=len(labels)),panels=panels,paired_strict_vs_SFT=contrasts,
        evaluation_roots=25,evaluation_episodes=len(splits["evaluation"])))
    print("completed strict comparison",{k:v["metrics"]["J_faithful"] for k,v in panels.items()},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();execute(args.bank.resolve(),args.output.resolve())


if __name__=="__main__":
    main()
