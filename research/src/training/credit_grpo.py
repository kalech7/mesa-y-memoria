"""Prospective GRPO: separate phase credit, equal phase weights, fixed-old reuse.

Historical episodic_grpo.py stays unchanged. Outer updates collect fresh groups;
each optimization pass accumulates the entire group before an Adam step.
"""
from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
import time

from .episodic_grpo import (
    EpisodeRollout, clipped_token_loss, group_advantages, group_telemetry,
    reference_logprobs_on_actor_device, root_batch, validate_group, verify_reference,
)
from .sft import (
    _gate, _prepare, _optimizer_step, _initial_metrics, _validate,
    _save_training_state, _write_json, parameter_hash, RunJournal,
    TrainingBudget, CheckpointSaveBudget, selection_is_better, selection_guards_passed,
)

PHASES = ("select", "respond", "write")


@dataclass
class CreditRollout(EpisodeRollout):
    phase_rewards: dict = field(default_factory=dict)


def with_credit(rollout, rewards):
    if set(rewards) != set(PHASES) or not all(math.isfinite(float(v)) for v in rewards.values()):
        raise ValueError("Exactly three finite absolute phase rewards required")
    return CreditRollout(**{f.name: getattr(rollout, f.name) for f in fields(EpisodeRollout)},
                         phase_rewards=dict(rewards))


def phase_advantages(rollouts):
    return {phase: group_advantages([r.phase_rewards[phase] for r in rollouts]) for phase in PHASES}


def train_credit_grpo(policy, reference, roots, collect_rollout, destination, validation,
                      *, updates=24, seed=101, learning_rate=3e-5, kl_coefficient=.03,
                      clip_epsilon=.2, optimization_passes=2, checkpoint_every=12,
                      budget_seconds=3600, initial_validation_metrics=None,
                      closure_authorization=None, closure_selection=None,
                      checkpoint_save_budget_seconds=120, smoke=False):
    _gate(smoke, continuation=True, closure_authorization=closure_authorization)
    if updates < 1 or optimization_passes < 1 or checkpoint_every < 1 or learning_rate <= 0:
        raise ValueError("Invalid declared optimization schedule")
    if not 0 < clip_epsilon < 1 or kl_coefficient < 0 or budget_seconds <= 0:
        raise ValueError("Invalid GRPO configuration")
    if str(reference.device) != "cpu":
        raise ValueError("Reference must remain on CPU between likelihood calls")
    roots = list(roots)
    if not roots:
        raise ValueError("Empty TRAIN cohort")
    destination, parameters = _prepare(policy, destination, seed)
    plan = dict(seed=seed, max_updates=updates, roots_per_batch=1, rollouts_per_root=4,
                learning_rate=learning_rate, kl_coefficient=kl_coefficient, clip_epsilon=clip_epsilon,
                rollout_optimization_passes=optimization_passes, phase_weight=1/3,
                old_logprobs="computed once before all optimization passes of each fresh group",
                credit="independently normalized reward per phase; common safety/joint tiers",
                max_grad_norm=1., budget_seconds=budget_seconds, checkpoint_every=checkpoint_every,
                closure_selection=closure_selection,
                optimizer={"name":"AdamW","betas":[.9,.999],"eps":1e-8,"weight_decay":0.},
                generated_tokens="unique sampled completions", trained_tokens="includes both optimization passes",
                input_tokens="prompts used by optimization, including both passes",
                kl_interpretation="sampled k3 KL proxy on reused old-policy rollout tokens")
    clock, saves = TrainingBudget(budget_seconds, None, separate=True), CheckpointSaveBudget(checkpoint_save_budget_seconds)
    with RunJournal(destination, "phase_credit_clipped_grpo", plan) as journal:
        verify_reference(policy, reference, smoke=smoke)
        frozen_before, reference_before = parameter_hash(policy, trainable=False), parameter_hash(reference)
        adapter_before = parameter_hash(policy, trainable=True)
        optimizer = policy.torch.optim.AdamW([p for _, p in parameters], lr=learning_rate, weight_decay=0.)
        counters = dict(updates=0, optimizer_steps=0, episodes=0, questions=0, input_tokens=0,
                        generated_tokens=0, trained_tokens=0, validation_evaluations=0,
                        allocated_seconds=budget_seconds)
        def save(path, step):
            def operation():
                policy.save_checkpoint(path)
                _save_training_state(policy, optimizer, path, step, counters)
            saves.save(operation)
        began = time.monotonic()
        journal.event("validation_started", step=0, reused=initial_validation_metrics is not None)
        metrics, _ = _initial_metrics(policy, validation, initial_validation_metrics)
        best_metrics, best_step = metrics, 0
        history = [{"step":0,"validation":metrics,"selection_guard_passed":selection_guards_passed(metrics,closure_selection)}]
        journal.event("validation_completed",step=0,metrics=metrics)
        save(destination/"step_000000",0)
        clock.pause(time.monotonic()-began)
        clock.start()
        stop_reason = "max_updates"
        with (destination/"rollouts.jsonl").open("x") as trace, (destination/"group_diagnostics.jsonl").open("x") as diagnostic, \
             (destination/"optimizer_passes.jsonl").open("x") as passes:
            for outer in range(1, updates+1):
                saves.ensure_available()
                if clock.exhausted(time.monotonic()-began):
                    stop_reason="time_budget"
                    break
                root = root_batch(roots, outer-1, 1)[0]
                group, references = [], {}
                for draw in range(4):
                    sampled_seed = seed*10000000+(outer-1)*4+draw
                    policy.seed(sampled_seed)
                    rollout = collect_rollout(policy,root,sampled_seed)
                    if not isinstance(rollout,CreditRollout):
                        raise TypeError("Collector must return explicit CreditRollout")
                    for phase in rollout.phases:
                        with policy.torch.no_grad(), policy.likelihood_mode():
                            old = policy.sequence_logprobs(phase.generation,sampling_distribution=True)
                        if not bool(policy.torch.isfinite(old).all()):
                            raise RuntimeError("NONFINITE_OLD_LOGPROBS")
                        phase.old_logprobs = old.detach().cpu().tolist()
                        ref = reference_logprobs_on_actor_device(reference,policy,phase.generation)
                        if not bool(policy.torch.isfinite(ref).all()):
                            raise RuntimeError("NONFINITE_REFERENCE_LOGPROBS")
                        references[(sampled_seed,phase.phase)] = ref.detach().cpu()
                        counters["generated_tokens"] += len(phase.generation.completion_ids)
                    group.append(rollout)
                    counters["episodes"] += 1
                    counters["questions"] += rollout.questions
                    journal.event("collecting",step=outer-1,pending_update=outer,rollout_index=draw)
                validate_group(group,expected_group_size=4,smoke=smoke)
                advantages = phase_advantages(group)
                episode_advantages = group_advantages([r.reward("long") for r in group])
                telemetry = group_telemetry(group,"long")
                telemetry.update(optimizer_update=outer,group_id=f"{outer-1}:0",
                    phase_returns={p:[r.phase_rewards[p] for r in group] for p in PHASES},
                    phase_advantages=advantages,phase_std={p:pstdev([r.phase_rewards[p] for r in group]) for p in PHASES})
                diagnostic.write(json.dumps(telemetry,ensure_ascii=False)+"\n");diagnostic.flush()
                for index,rollout in enumerate(group):
                    trace.write(json.dumps(dict(update=outer,group_id=f"{outer-1}:0",rollout=asdict(rollout),
                        return_value=rollout.reward("long"),advantage=episode_advantages[index],
                        phase_advantages={p:advantages[p][index] for p in PHASES},
                        actor_masks=[p.actor_mask for p in rollout.phases]),ensure_ascii=False)+"\n")
                trace.flush()
                last = None
                for optimization_pass in range(1,optimization_passes+1):
                    optimizer.zero_grad(set_to_none=True)
                    total_loss, total_kl, ratios, clipping = 0.,0.,[],[]
                    phase_losses = {p:0. for p in PHASES}
                    for index,rollout in enumerate(group):
                        for phase in rollout.phases:
                            generated=phase.generation
                            with policy.likelihood_mode():
                                new=policy.sequence_logprobs(generated,sampling_distribution=True)
                                old=policy.torch.tensor(phase.old_logprobs,device=new.device)
                                ref=references[(rollout.rollout_seed,phase.phase)].to(new.device)
                                if not bool(policy.torch.isfinite(new).all()):
                                    raise RuntimeError("NONFINITE_ACTOR_LOGPROBS")
                                loss,kl=clipped_token_loss(new,old,ref,advantages[phase.phase][index],
                                                         clip_epsilon=clip_epsilon,kl_coefficient=kl_coefficient)
                                weight=1/(3*len(group))
                                weighted=weight*loss
                                if not bool(policy.torch.isfinite(weighted)):
                                    raise RuntimeError("NONFINITE_CREDIT_GRPO_LOSS")
                                ratio=(new.detach()-old).exp()
                                if not bool(policy.torch.isfinite(ratio).all()):
                                    raise RuntimeError("NONFINITE_IMPORTANCE_RATIO")
                                ratios.extend(ratio.cpu().tolist())
                                clipping.append(float(((ratio<1-clip_epsilon)|(ratio>1+clip_epsilon)).float().mean().cpu()))
                                weighted.backward()
                            total_loss+=float(weighted.detach().cpu());total_kl+=float(kl.detach().cpu())*weight
                            phase_losses[phase.phase]+=float(weighted.detach().cpu())
                            counters["input_tokens"]+=len(generated.prompt_ids)
                            counters["trained_tokens"]+=len(generated.completion_ids)
                    step_info={}
                    gradient=_optimizer_step(policy,optimizer,parameters,step_info)
                    if not math.isfinite(gradient):
                        raise RuntimeError("NONFINITE_CREDIT_GRADIENT")
                    counters["optimizer_steps"]+=1
                    last=dict(step=outer,outer_update=outer,optimization_pass=optimization_pass,
                        optimizer_step=counters["optimizer_steps"],loss=total_loss,kl=total_kl,grad_norm=gradient,
                        phase_losses=phase_losses,ratio_min=min(ratios),ratio_max=max(ratios),ratio_mean=fmean(ratios),
                        clip_fraction_phase_mean=fmean(clipping),**step_info)
                    passes.write(json.dumps(last)+"\n");passes.flush()
                    journal.event("optimization_pass_completed",**last)
                counters["updates"]=outer
                entry={**last,"resource_counters":dict(counters)}
                journal.progress(outer,updates,loss=last["loss"],resource_counters=dict(counters))
                if outer%checkpoint_every==0 or outer==updates:
                    pause=time.monotonic()
                    journal.event("validation_started",step=outer)
                    metrics,_=_validate(policy,validation,outer)
                    counters["validation_evaluations"]+=1
                    entry.update(validation=metrics,selection_guard_passed=selection_guards_passed(metrics,closure_selection))
                    save(destination/f"step_{outer:06d}",outer)
                    journal.event("validation_completed",step=outer,metrics=metrics)
                    clock.pause(time.monotonic()-pause)
                    if selection_is_better(metrics,outer,best_metrics,best_step,closure_selection):
                        best_step,best_metrics=outer,metrics
                history.append(entry)
            clock.finish()
        final_step=counters["updates"]
        if final_step!=updates:
            raise RuntimeError("Incomplete declared RL budget; do not evaluate heldback data")
        save(destination/"last",final_step)
        frozen_after,reference_after=parameter_hash(policy,trainable=False),parameter_hash(reference)
        if frozen_before!=frozen_after or reference_before!=reference_after:
            raise RuntimeError("FROZEN_PARAMETERS_CHANGED")
        result=dict(algorithm="phase_credit_clipped_grpo",seed=seed,status="executed",plan=plan,
            best_step=best_step,selected_checkpoint=str(destination/f"step_{best_step:06d}"),
            final_checkpoint=str(destination/"last"),stop_reason=stop_reason,
            parent_checkpoint=policy.origin_checkpoint,kl_reference=reference.origin_checkpoint,
            trainable_before=adapter_before,trainable_after=parameter_hash(policy,trainable=True),
            frozen_before=frozen_before,frozen_after=frozen_after,reference_before=reference_before,
            reference_after=reference_after,history=history,
            resource_counters={**counters,**saves.counters(),"elapsed_seconds":time.monotonic()-began,
                               "training_seconds":clock.elapsed(time.monotonic()-began),
                               "validation_and_checkpoint_seconds":clock.paused_seconds})
        _write_json(destination/"training.json",result)
        journal.event("completed",step=final_step,selected_step=best_step,resource_counters=result["resource_counters"])
        return result
