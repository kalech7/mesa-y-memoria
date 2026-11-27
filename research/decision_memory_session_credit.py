"""Session ASK×WRITE action enumeration with a full-support prefix mixture.

Choose t~Uniform(0..3) before roll-in. The prefix proposal is .5*pi + .5*defer,
where defer forces earlier ASK to PROCEED and samples WRITE under pi. If an
earlier ASK occurred, pi/mu=2. Otherwise r=prod pi(PROCEED) and pi/mu=2r/(1+r).
Weights are detached, never clipped or batch-normalized.

At t enumerate pi_A(a|h)*pi_B(b|h after the actual reply to a): nine legal pairs
when ASK is available, three when ASK is forced. Only the continuation AFTER
WRITE is sampled. The weighted local joint gradient estimates grad E[R]/4.
It is not the earlier mean-over-eight-actions estimator and uses no n/4 factor.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import random

from research.budget_memory_env import ASK_COST, WRITE_COST, WRITE_ACTIONS, canonical, fingerprint, run_episode
from research.decision_memory_credit import (
    _actor_fingerprint, _behavior_probabilities, _ForkPolicy, _RecordedReader,
    _seed, _unchanged, _write,
)
from research.decision_memory_rollin import _ForbiddenRollinReader, _RollinPolicy, _TargetReached

VERSION = "decision_memory_session_credit_v1"
SAMPLING_CONTRACT = "session_joint_prefix_mixture_IS_v1"


class _SessionFork:
    def __init__(self, policy, prefix, ask_target, ask_action, write_action):
        self.policy = policy
        self.ask_step = ask_target["step_id"]
        self.write_step = self.ask_step+1
        self.ask_action = ask_action
        self.write_action = write_action
        self.replayer = _ForkPolicy(policy,[*prefix,ask_target],self.ask_step,ask_action)
        self.write_target = None

    def decide(self, observation, candidates, *, seed, sample, decision_id):
        step = 2*observation["session_index"] + int(observation["decision_type"] == "write")
        if step != self.write_step:
            return self.replayer.decide(observation,candidates,seed=seed,sample=sample,decision_id=decision_id)
        if sample is not True or candidates != list(WRITE_ACTIONS):
            raise ValueError("Session WRITE must retain its full public catalogue")
        fn = self.policy.decide if hasattr(self.policy,"decide") else self.policy
        raw = fn(observation,candidates,seed=seed,sample=True,decision_id=decision_id)
        if not isinstance(raw,dict) or raw.get("action") not in candidates:
            raise TypeError("Session WRITE needs real conditional controller metadata")
        raw = deepcopy(raw)
        self.write_target = dict(decision_id=decision_id,step_id=step,phase="write",decision_type="write",
            session_index=observation["session_index"],seed=seed,observation=deepcopy(observation),
            observation_sha256=fingerprint(observation),candidates=list(candidates),action=raw["action"],
            metadata=deepcopy(raw["metadata"]),policy_output=raw,target_draw_not_executed=True)
        self.write_target["metadata"]["target_draw_not_executed"] = True
        return {"action":self.write_action,"metadata":{
            "credit_role":"forced_session_write","policy_evaluated":True,"sampled":False,
            "decision_id":decision_id,"forced_action":self.write_action,
            "conditional_distribution_recorded":True,"target_draw_not_executed":True,
            "source_policy_output":deepcopy(raw)}}


def collect_session_group(policy, reader, episode, seed, output_dir, session_index=None):
    """Enumerate one exogenous whole session, keeping the actor fixed throughout.

    Explicit session_index is intended for predeclared exogenous plans/CPU tests;
    otherwise the legacy target_index RNG samples Uniform(0..3) before roll-in.
    The mixture component cannot be overridden using an outcome or a callback.
    """
    visit_seed,selector_seed,fork_seed,mixture_seed = (_seed(seed,name) for name in
        ("manager_visitation","target_index","fork_future","rollin_mixture"))
    selection_mode = "preselected_by_caller" if session_index is not None else "uniform_rng_before_rollin"
    if session_index is None:
        session_index = random.Random(selector_seed).randrange(4)
    if type(session_index) is not int or not 0 <= session_index < 4:
        raise ValueError("Session must be independently preselected in 0..3")
    component = "defer_ask" if random.Random(mixture_seed).random() < .5 else "on_policy"
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("A session group cannot overwrite an earlier group")
    output.mkdir(parents=True,exist_ok=True)
    namespace = f"decision-session:{fingerprint([str(output.resolve()),seed])[:20]}"
    initial = _actor_fingerprint(policy)
    proposal = _RollinPolicy(policy,"ask",session_index,
                            defer_previous_ask=component=="defer_ask",allow_forced_target=True)
    try:
        run_episode(proposal,_ForbiddenRollinReader(),episode,visit_seed,output/"rollin",
                    sample=True,namespace=namespace+":rollin")
    except _TargetReached:
        pass
    else:
        raise ValueError("Session roll-in did not stop before ASK")
    _unchanged(policy,initial)
    ask_target,prefix = proposal.target,proposal.prefix
    if ask_target is None or ask_target["step_id"] != 2*session_index or len(prefix)!=2*session_index:
        raise ValueError("Session prefix/target length is inconsistent")
    asked_before = any(d["phase"]=="ask" and d["action"]!="PROCEED" for d in prefix)
    if asked_before and component=="defer_ask":
        raise ValueError("Deferred-ASK proposal produced an impossible spent-budget prefix")
    if asked_before:
        logs,log_r,r = [],None,None
        weight,log_weight = 2.,math.log(2.)
    else:
        logs = [row["logprob_proceed"] for row in proposal.ask_prefix_behavior]
        if len(logs)!=session_index:
            raise ValueError("No-ASK prefix requires every earlier PROCEED likelihood")
        log_r = math.fsum(logs)
        r = math.exp(log_r)
        log_weight = math.log(2.)+log_r-math.log1p(r)
        weight = math.exp(log_weight)
    if not math.isfinite(weight) or not 0 <= weight <= 2:
        raise ValueError("Exact prefix mixture ratio must be finite in [0,2]")
    ask_probs = _behavior_probabilities(ask_target)
    expected_ask_count = 1 if asked_before else 3
    if len(ask_target["candidates"])!=expected_ask_count:
        raise ValueError("Session ASK catalogue disagrees with the actual question budget")
    ask_target.update(session_selection_probability=.25,within_phase_selection_probability=None,
                      session_selection_mode=selection_mode,prefix_importance=weight)
    past_cost = math.fsum(ASK_COST if d["phase"]=="ask" and d["action"]!="PROCEED" else
                         WRITE_COST if d["phase"]=="write" and d["action"]!="NOOP" else 0.
                         for d in prefix)
    on_policy_source = component=="on_policy" or session_index==0
    rollin = dict(collection_role="session_prefix_only",outcome_not_evaluated=True,
        decisions=deepcopy(prefix),session_index=session_index,mixture_component=component,
        mixture_probability=.5,source_path_on_policy=on_policy_source,real_reader_calls=0,
        asked_before=asked_before,proceed_prefix_logprobs=list(logs),
        ask_prefix_behavior=deepcopy(proposal.ask_prefix_behavior),prefix_importance=weight)
    _write(output/"rollin"/"trace.json",rollin)
    branches,pairs,write_targets,write_probs,reader_calls = [],[],[],[],[]
    for ask_index,ask_action in enumerate(ask_target["candidates"]):
        first_write = None
        for write_index,write_action in enumerate(WRITE_ACTIONS):
            branch_index = len(branches)
            wrapper = _SessionFork(policy,prefix,ask_target,ask_action,write_action)
            recorded_reader = _RecordedReader(reader)
            branch = run_episode(wrapper,recorded_reader,episode,fork_seed,
                output/f"branch_{branch_index:02d}",sample=True,
                namespace=f"{namespace}:branch:{branch_index}")
            _unchanged(policy,initial)
            conditional = wrapper.write_target
            if conditional is None or len(recorded_reader.calls)!=2:
                raise ValueError("A session branch lacks its WRITE distribution or real outcomes")
            probabilities = _behavior_probabilities(conditional)
            if first_write is None:
                first_write = conditional
                write_targets.append(conditional)
                write_probs.append(probabilities)
            elif (canonical(conditional["observation"])!=canonical(first_write["observation"])
                  or conditional["candidates"]!=first_write["candidates"]
                  or any(abs(a-b)>1e-6 for a,b in zip(probabilities,write_probs[ask_index]))):
                raise ValueError("Conditional WRITE state/distribution changed across the same ASK branch")
            if (wrapper.replayer.replayed!=2*session_index or wrapper.replayer.forced!=1
                    or wrapper.replayer.sampled!=6-2*session_index):
                raise ValueError("Session branch must replay prefix, force ASK+WRITE, then sample its suffix")
            pair = dict(ask_index=ask_index,write_index=write_index,branch_index=branch_index,
                        ask_action=ask_action,write_action=write_action)
            pairs.append(pair)
            branch["counterfactual"] = dict(collection_role="session_action_enumeration",
                sampling_contract=SAMPLING_CONTRACT,**pair,session_index=session_index,
                ask_step=2*session_index,write_step=2*session_index+1,
                replayed_prefix_decisions=2*session_index,forced_decisions=2,
                sampled_future_decisions=6-2*session_index,fork_seed=fork_seed,
                past_cost=past_cost,future_cost=branch["metrics"]["costs"]-past_cost,
                prefix_importance=weight,real_reader_calls=2,actor_fingerprint=initial,
                source_path_on_policy=False,use_for_whole_episode_LOO=False)
            _write(output/f"branch_{branch_index:02d}"/"trace.json",branch)
            branches.append(branch)
            reader_calls.append(recorded_reader.calls)
    joint_probs = [ask_probs[p["ask_index"]]*write_probs[p["ask_index"]][p["write_index"]] for p in pairs]
    if not math.isclose(math.fsum(joint_probs),1.,rel_tol=0,abs_tol=1e-6):
        raise ValueError("Joint session behavior probabilities are not normalized")
    decision_seeds = [[d["seed"] for d in branch["decisions"]] for branch in branches]
    reader_seeds = [[c["seed"] for c in calls] for calls in reader_calls]
    if any(s!=decision_seeds[0] for s in decision_seeds) or any(s!=reader_seeds[0] for s in reader_seeds):
        raise ValueError("Session branches must use new common future randomness")
    diagnostics = dict(actor_before=initial,actor_after=_unchanged(policy,initial),actor_unchanged=True,
        session_index=session_index,session_selection_probability=.25,
        selected_session_before_rollin=True,session_selection_mode=selection_mode,
        mixture_component=component,mixture_probability=.5,asked_before=asked_before,
        proceed_prefix_logprobs=list(logs),no_ask_prefix_logprob=log_r,no_ask_prefix_probability=r,
        prefix_importance=weight,prefix_log_importance=log_weight,
        importance_detached=True,importance_clipped=False,importance_batch_renormalized=False,
        source_path_on_policy=on_policy_source,rollin_real_reader_calls=0,
        fork_real_reader_calls=sum(map(len,reader_calls)),outcome_label_episodes=len(branches),
        visit_seed=visit_seed,selector_seed=selector_seed,fork_future_seed=fork_seed,mixture_seed=mixture_seed,
        common_decision_seeds=decision_seeds[0],common_reader_seeds=reader_seeds[0],
        past_cost=past_cost,normalization="sessionUniform1/4; detached pi/(.5pi+.5defer); gradientJ/4",
        density_proof=("Earlier ASK spent: defer density0, mu=.5pi, ratio2. "
            "No earlier ASK: pi=r*Pwrite, defer=Pwrite, mu=.5*(1+r)*Pwrite, ratio2r/(1+r)."),
        teacher_actions=False,selected_by_return=False,optimizer_steps=0)
    group = dict(version=VERSION,sampling_contract=SAMPLING_CONTRACT,
        episode_id=episode["episode_id"],root_id=episode.get("root_id",episode["episode_id"]),
        family=episode.get("family"),seed=seed,phase="session",session_index=session_index,
        ask_target=ask_target,write_targets=write_targets,action_pairs=pairs,
        prefix=deepcopy(prefix),rollin=rollin,branches=branches,
        behavior_ask_probs=ask_probs,behavior_write_probs=write_probs,
        behavior_probs=joint_probs,selection_weight=weight,diagnostics=diagnostics)
    _write(output/"group.json",group)
    return group


def enumerate_session_credit(group,rewards):
    """Center actual session-pair returns; no greedy action target is created."""
    if group.get("sampling_contract")!=SAMPLING_CONTRACT:
        raise ValueError("A session-joint prefix-mixture group is required")
    if len(rewards)!=len(group["branches"]) or len(rewards)!=len(group["action_pairs"]):
        raise ValueError("Each legal session pair requires an evaluated future return")
    values=[]
    for branch,reward in zip(group["branches"],rewards):
        if branch.get("counterfactual",{}).get("sampling_contract")!=SAMPLING_CONTRACT:
            raise ValueError("Only real session outcome branches can receive credit")
        value=reward.get("episodic_reward") if isinstance(reward,dict) else reward
        if not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError("A finite real session-pair return is required")
        values.append(float(value))
    probabilities=group["behavior_probs"]
    baseline=math.fsum(p*q for p,q in zip(probabilities,values))
    past_cost=group["diagnostics"]["past_cost"]
    advantages=[q-baseline for q in values]
    return dict(version=VERSION,sampling_contract=SAMPLING_CONTRACT,
        action_pairs=deepcopy(group["action_pairs"]),behavior_probs=list(probabilities),
        full_returns=values,future_returns=[q+past_cost for q in values],past_cost=past_cost,
        future_costs=[b["counterfactual"]["future_cost"] for b in group["branches"]],
        baseline_full=baseline,baseline_future=baseline+past_cost,advantages=advantages,
        weighted_advantage_mean=math.fsum(p*a for p,a in zip(probabilities,advantages)),
        selection_weight=group["selection_weight"],prefix_importance=group["selection_weight"],
        task_loss="-detached_prefix_importance * sum(pi_ASK*pi_WRITE_conditional*stopgrad(advantages))",
        advantage_normalization="none",target_action=None,whole_episode_LOO=False)
