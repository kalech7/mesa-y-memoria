"""Prospective competence data, grouped partitions and observable v5 teachers.

Recipes reuse varied_v3, not its saved episodes. Literal gold is authored by
that producer and shifted consistently in time. The teacher receives only real
EpisodeEnvironment observations. Evaluation is a held-back development role,
not a confirmatory test or a human annotation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import tempfile

from .varied_development import build_varied_development, diversity_plan, STRATA, VOCABULARY, CLAUSE
from .varied_development_v3_003 import NEUTRAL_QUESTIONS
from .src.data.semantic_partition_v3 import (
    assert_semantic_split_isolation_v3, semantic_episode_id_v3, observable_episode_id_v3,
)

ROOT = Path(__file__).resolve().parents[1]
VERSION = "competence_data_v1_observable_teacher_v5"
DEFAULT_SIZES = {"train":200, "validation":30, "evaluation":50}
DEFAULT_ROUTES = ("natural", "PROCEED", "ASK:TEMPORAL_SCOPE")
PUBLIC_KEYS = {"memory", "user_messages", "query", "reuse_regime"}
DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")


def _load(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def _shift_dates(value, days):
    if isinstance(value, str):
        return DATE_PATTERN.sub(lambda match:(date.fromisoformat(match[0])+timedelta(days=days)).isoformat(),value)
    if isinstance(value,list):
        return [_shift_dates(item,days) for item in value]
    if isinstance(value,dict):
        return {key:_shift_dates(item,days) for key,item in value.items()}
    return value


def _support_loss_control(episode):
    """Literal recipe: latest same-attribute exception expired yesterday.

    Rebuild historical sessions before data are frozen. Other preferences retain
    their values and correction order; each record cites its actual dated user
    message. No existing corpus is an input to this construction.
    """
    records=episode["initial_memory"]
    if len(records)==1:
        episode["authoring"]["observability_control"]={"kind":"single_memory_negative_control","expected_O1_C2":1,"expected_O2_C2":1}
        return
    attribute=episode["current_query"]["attribute"]
    now=date.fromisoformat(episode["current_query"]["date"])
    yesterday=now-timedelta(days=1)
    old=records[-1]
    values=[v for v in VOCABULARY[attribute] if v!=records[0]["value"]]
    value=values[int(_digest([episode["root_id"],"expired-exception"])[:12],16)%len(values)]
    count=min(3,len(records))
    sessions=[{"session_id":f"P{i+1}","date":(yesterday if i==count-1 else now-timedelta(days=(count-i)*8)).isoformat(),"messages":[]}
              for i in range(count)]
    for index,record in enumerate(records):
        session_index=count-1 if index==len(records)-1 else min(count-2,index*(count-1)//(len(records)-1))
        session=sessions[session_index]
        tid=f"p{session_index+1}u{len(session['messages'])+1}"
        if index==len(records)-1:
            text=f"Solo el {yesterday.isoformat()}, {CLAUSE[attribute].lower()} prefiero {value}."
            record={"memory_id":old["memory_id"],"attribute":attribute,"value":value,
                    "scope":{"kind":"interval","valid_from":yesterday.isoformat(),"valid_until_exclusive":now.isoformat(),"context":None,"session_id":None},
                    "status":"confirmed","supersedes":[]}
            records[index]=record
        else:
            text=record["evidence_quotes"][0]["text"]
            record["scope"]["valid_from"]=session["date"]
        session["messages"].append({"role":"user","turn_id":tid,"text":text})
        record.update(recorded_at=session["date"],source_turn_ids=[tid],evidence_quotes=[{"turn_id":tid,"text":text}])
    episode["prior_sessions"]=sessions
    episode["authoring"]["observability_control"]={"kind":"expired_latest_same_attribute_record",
        "base_memory_id":"m1","expired_memory_id":records[-1]["memory_id"],"expected_O1_C2":0,"expected_O2_C2":1}


def _fresh_batch(seed, batch):
    # Fresh future calendar periods are a generation choice, not evidence of OOD.
    offset=(date(2031,1,1)-date(2028,3,1)).days+batch*150
    rows=_shift_dates(build_varied_development(seed=seed+batch*1009),offset)
    for episode in rows:
        for index,message in enumerate(episode["user_messages"]):
            if message["role"]=="assistant" and message["text"].startswith("¿Quieres que anote «"):
                message["text"]=NEUTRAL_QUESTIONS[(batch+index)%len(NEUTRAL_QUESTIONS)]
        episode["authoring"].update(source=VERSION,source_recipe="varied_v3 fresh generation",generation_seed=seed+batch*1009,
            generation_batch=batch,coverage_source="Prospective competence/PI1/PI2 requirements, not model outcomes.",
            gold_source="Literal varied_v3 recipe, consistently shifted dates and authored expired-exception control; no parser/scorer gold.")
        if episode["authoring"]["stratum"]=="unrelated_attribute_control":
            _support_loss_control(episode)
        episode.update(generator_version=VERSION,data_namespace=f"competence_v1_seed_{seed}",data_role="unassigned_prospective_pool")
        observable=observable_episode_id_v3(episode)
        episode["root_id"]="cr"+observable[:20]
        episode["latent_world_id"]="cw"+_digest([observable,episode["truth"]["scope_kind"]])[:20]
        episode["observable_state_id"]=observable
        episode["template_id"]="competence:"+episode["template_id"]
        episode["composition_id"]=(f"competence:{episode['authoring']['stratum']}:m{len(episode['initial_memory'])}"
            f":p{len(episode['prior_sessions'])}:t{len(episode['user_messages'])}:{episode['authoring']['boundary']}:{episode['current_query']['attribute']}")
    return rows


def _sizes(sizes):
    sizes=dict(sizes)
    if set(sizes)!={"train","validation","evaluation"} or any(type(n) is not int or n<=0 or n%len(STRATA) for n in sizes.values()):
        raise ValueError("Require positive train/validation/evaluation root counts divisible by five")
    return sizes


def build_candidates(*, seed, sizes=None):
    sizes=_sizes(sizes or DEFAULT_SIZES)
    target=sum(sizes.values())//len(STRATA)
    groups={};batch=0
    # Reserve one literal single-memory F0 control for each partition. This is a
    # coverage constraint fixed before methods run, never a score-based filter.
    while True:
        for episode in _fresh_batch(seed,batch):
            groups.setdefault(episode["root_id"],[]).append(episode)
        counts=Counter(v[0]["authoring"]["stratum"] for v in groups.values())
        singles=[rid for rid,v in groups.items() if v[0]["authoring"]["stratum"]=="unrelated_attribute_control" and len(v[0]["initial_memory"])==1]
        if min(counts.values())>=target and len(singles)>=3:break
        batch+=1
        if batch>=40:raise ValueError("Cannot meet preregistered coverage without changing recipe")
    rng=random.Random(seed+31);selected=[]
    for stratum in STRATA:
        available=sorted(rid for rid,v in groups.items() if v[0]["authoring"]["stratum"]==stratum)
        rng.shuffle(available)
        mandatory=sorted(singles)[:3] if stratum=="unrelated_attribute_control" else []
        if len(mandatory)>target:raise ValueError("Need at least three F0 roots across partitions")
        chosen=mandatory+[rid for rid in available if rid not in mandatory][:target-len(mandatory)]
        for rid in chosen:selected.extend(deepcopy(groups[rid]))
    return selected


def partition_candidates(episodes, sizes, *, seed):
    """Union complete canonical/observable components before split assignment."""
    sizes=_sizes(sizes);groups=defaultdict(list)
    for episode in episodes:groups[episode["root_id"]].append(deepcopy(episode))
    for variants in groups.values():variants.sort(key=semantic_episode_id_v3)
    if len(groups)!=sum(sizes.values()):raise ValueError("Candidate root count differs from partition quotas")
    parent={rid:rid for rid in groups}
    def find(rid):
        while parent[rid]!=rid:
            parent[rid]=parent[parent[rid]];rid=parent[rid]
        return rid
    owners={}
    for rid,variants in groups.items():
        for episode in variants:
            for signature in (("behavioral",semantic_episode_id_v3(episode)),("observable",observable_episode_id_v3(episode))):
                if signature in owners:parent[find(rid)]=find(owners[signature])
                else:owners[signature]=rid
    components=defaultdict(list)
    for rid in groups:components[find(rid)].append(rid)
    strata=defaultdict(list)
    for component in components.values():
        labels={groups[rid][0]["authoring"]["stratum"] for rid in component}
        if len(labels)!=1:raise ValueError("Equivalent roots cross recipe strata; review before partitioning")
        strata[next(iter(labels))].append(component)
    rng=random.Random(seed);result={name:[] for name in sizes}
    for stratum in STRATA:
        components=sorted(strata[stratum],key=lambda component:_digest(sorted(observable_episode_id_v3(groups[rid][0]) for rid in component)))
        rng.shuffle(components)
        remaining={name:n//len(STRATA) for name,n in sizes.items()}
        assigned=[]
        if stratum=="unrelated_attribute_control":
            singles=[c for c in components if len(c)==1 and len(groups[c[0]][0]["initial_memory"])==1]
            if len(singles)<3:raise ValueError("Three complete negative-control components required")
            for name,component in zip(sorted(sizes),singles[:3]):
                assigned.append((name,component));remaining[name]-=1;components.remove(component)
        for component in components:
            fits=[name for name,n in remaining.items() if n>=len(component)]
            if not fits:raise ValueError("An equivalence component cannot fit quotas; never split it")
            name=max(fits,key=lambda label:(remaining[label],label))
            assigned.append((name,component));remaining[name]-=len(component)
        if any(remaining.values()):raise ValueError("Incomplete stratum quotas")
        for name,component in assigned:
            for rid in sorted(component):
                for episode in groups[rid]:
                    episode.update(split=name,partition_role=name,data_role={"train":"prospective_supervised_training","validation":"prospective_development_selection","evaluation":"held_back_development_evaluation"}[name])
                    result[name].append(episode)
    for rows in result.values():rng.shuffle(rows)
    assert_semantic_split_isolation_v3(result)
    return result


def discover_previous_banks(*, exclude=None):
    paths=[*sorted((ROOT/"research/data").rglob("*.jsonl")),*sorted((ROOT/"research/reports").rglob("*.jsonl"))]
    result={}
    for path in paths:
        if exclude and path.resolve().is_relative_to(Path(exclude).resolve()):continue
        with path.open() as stream:
            line=next((s for s in stream if s.strip()),None)
        if line is None:continue
        row=json.loads(line)
        if {"root_id","initial_memory","user_messages","current_query"}.issubset(row):
            result[str(path.relative_to(ROOT))]=path
    if not result:raise ValueError("No previous episode banks found; historical isolation cannot be certified")
    return result


def _historical_views(paths):
    historical={};adaptations={}
    for name,path in paths.items():
        rows=_load(path);missing=Counter()
        for row in rows:
            # Very old smoke banks did not record opportunity. Public identity
            # is still checked exactly; this sentinel is never actor-facing.
            for field in ("future_query","replies","truth","opportunity"):
                if field not in row:row[field]={"historical_field_not_recorded":True};missing[field]+=1
        historical[name]=rows
        if missing:adaptations[name]=dict(missing)
    return historical,adaptations


def teacher_for_episode(episode, *, condition="O2", route="natural", work_directory=None):
    """Return three phase targets using public environment observations only.

    SELECT is always the natural teacher target. executed_selection records the
    route intervention separately. Only natural SELECT should enter SFT once;
    forced routes supply RESPOND/WRITE targets. A supplied work directory retains
    its uniquely named pristine-memory SQLite; no teacher write is executed.
    """
    from .src.env.episode import EpisodeEnvironment
    from .src.policy.rule_controller import RuleController
    from .src.policy.llm_controller import validate_action
    if condition not in {"O1","O2"}:raise ValueError("Unknown condition")
    if route!="natural" and route!="PROCEED" and route not in {"ASK:"+s for s in ("TEMPORAL_SCOPE","CURRENT_CONTEXT","VALID_UNTIL","GENERIC")}:
        raise ValueError("Unknown teacher route")
    temporary=tempfile.TemporaryDirectory(prefix="competence_teacher_") if work_directory is None else None
    work=Path(temporary.name) if temporary else Path(work_directory)
    work.mkdir(parents=True,exist_ok=True)
    filename=_digest([episode["root_id"],episode["latent_world_id"],episode["realization_id"],condition,route])[:32]+".sqlite"
    env=None
    try:
        env=EpisodeEnvironment(episode,work/filename)
        controller=RuleController();observations={};actions={}
        observations["select"]=env.observe("select",condition)
        actions["select"]=controller.select(deepcopy(observations["select"]))
        executed=deepcopy(actions["select"]) if route=="natural" else {"action":"PROCEED","slot":None} if route=="PROCEED" else {"action":"ASK","slot":route.split(":",1)[1]}
        if executed["action"]=="ASK":env.ask(executed["slot"])
        for phase in ("respond","write"):
            observations[phase]=env.observe(phase,condition)
            actions[phase]=getattr(controller,phase)(deepcopy(observations[phase]))
        result=[]
        for phase in ("select","respond","write"):
            if set(observations[phase])!=PUBLIC_KEYS:raise ValueError("Teacher observation contains unexpected keys")
            validate_action(phase,actions[phase])
            result.append({"phase":phase,"observation":deepcopy(observations[phase]),"action":actions[phase],
                "completion":json.dumps(actions[phase],ensure_ascii=False,sort_keys=True),"split":episode["split"],
                "root_id":episode["root_id"],"latent_world_id":episode["latent_world_id"],"realization_id":episode["realization_id"],
                "condition":condition,"route":route,"executed_selection":executed,"source":"observable_rule_teacher_v5",
                "teacher_db_file":filename,"teacher_database_retained":temporary is None})
        return result
    finally:
        if env is not None:env.dispose()
        if temporary is not None:temporary.cleanup()


def teacher_examples(episodes, *, routes=DEFAULT_ROUTES, work_directory=None):
    if not routes or routes[0]!="natural" or len(set(routes))!=len(routes):raise ValueError("Unique routes starting with natural required")
    for episode in episodes:
        for route in routes:
            for row in teacher_for_episode(episode,route=route,work_directory=work_directory):
                if route=="natural" or row["phase"]!="select":yield row


def _static_audit(splits):
    failures=[]
    for role,rows in splits.items():
        for episode in rows:
            prior={m["turn_id"]:(s["date"],m) for s in episode["prior_sessions"] for m in s["messages"]}
            if not 1<=len(episode["initial_memory"])<=6 or not 2<=len(episode["user_messages"])<=8:
                failures.append({"role":role,"root_id":episode["root_id"],"kind":"shape"})
            for record in episode["initial_memory"]:
                for quote in record["evidence_quotes"]:
                    at,message=prior[quote["turn_id"]]
                    if message["role"]!="user" or quote["text"]!=message["text"] or at!=record["recorded_at"]:
                        failures.append({"role":role,"root_id":episode["root_id"],"kind":"memory_provenance"})
            if any(m["role"]=="assistant" and "prefiero" in m["text"] for m in episode["user_messages"]):
                failures.append({"role":role,"root_id":episode["root_id"],"kind":"assistant_preference_proposal"})
            control=episode["authoring"].get("observability_control")
            if control and control["kind"]=="expired_latest_same_attribute_record":
                first,last=episode["initial_memory"][0],episode["initial_memory"][-1]
                if first["value"]!=episode["truth"]["current_value"] or first["value"]!=episode["truth"]["future_value"] or last["scope"]["valid_until_exclusive"]!=episode["current_query"]["date"]:
                    failures.append({"role":role,"root_id":episode["root_id"],"kind":"expired_control_recipe"})
    return failures


def _audit_training_contracts(splits, output):
    from .src.evaluation.runner import run_episode
    from .src.policy.rule_controller import RuleController
    work=output/"audit_work";work.mkdir()
    failures=[];counts=Counter();by_split=defaultdict(Counter);pi1=Counter()
    with (output/"audit_contract_traces.jsonl").open("x") as target, (output/"audit_pi1_traces.jsonl").open("x") as pi1_target, (output/"audit_counterfactual_prefixes.jsonl").open("x") as cf_target:
        for role in ("train","validation"):
            for index,episode in enumerate(splits[role]):
                row=run_episode(deepcopy(episode),RuleController(),method="B4",condition="O2",db_path=work/f"{role}_{index:04d}.sqlite")
                metrics=row["metrics"]
                if metrics["C2"]!=1 or metrics["FJS"]!=1 or metrics["unsupported_operations"]!=0 or metrics["invalid_outputs"]!=0:
                    failures.append({"kind":"literal_gold_or_write_contract_mismatch","split":role,"root_id":episode["root_id"],"world_id":episode["latent_world_id"],"metrics":metrics})
                control=episode["authoring"].get("observability_control")
                if control:
                    o1=run_episode(deepcopy(episode),RuleController(),method="C-NEVER",condition="O1",db_path=work/f"{role}_{index:04d}_O1.sqlite")
                    memory_ids=[record["memory_id"] for record in o1["trace"]["observations"]["select"]["memory"]]
                    expected_id=control.get("expired_memory_id","m1")
                    if o1["metrics"]["C2"]!=control["expected_O1_C2"] or memory_ids!=[expected_id] or metrics["C2"]!=control["expected_O2_C2"]:
                        failures.append({"kind":"PI1_support_loss_manipulation_mismatch","split":role,"root_id":episode["root_id"],"O1_C2":o1["metrics"]["C2"],"memory_ids":memory_ids})
                    pi1.update(cases=1,positive_benefits=int(metrics["C2"]>o1["metrics"]["C2"]),negative_controls=int(control["kind"]=="single_memory_negative_control"))
                    pi1_target.write(json.dumps(o1,ensure_ascii=False,sort_keys=True)+"\n");pi1_target.flush()
                original=teacher_for_episode(episode,work_directory=work/"prefix_original")
                counterfactual=deepcopy(episode)
                counterfactual.update(truth={"private":"MUST_NOT_APPEAR_IN_OBSERVATION"},future_query={"attribute":"private_future","date":"2099-12-30","session_id":"HIDDEN","context":"PRIVATE"},
                    opportunity={"private":True},authoring={"private":True},latent_world_id="PRIVATE_WORLD")
                changed=teacher_for_episode(counterfactual,work_directory=work/"prefix_counterfactual")
                for a,b in zip(original,changed):
                    cf_target.write(json.dumps({"split":role,"root_id":episode["root_id"],"latent_world_id":episode["latent_world_id"],
                        "phase":a["phase"],"original_observation_sha256":_digest(a["observation"]),"counterfactual_observation_sha256":_digest(b["observation"]),
                        "original_action_sha256":_digest(a["action"]),"counterfactual_action_sha256":_digest(b["action"]),
                        "matched":a["observation"]==b["observation"] and a["action"]==b["action"]},ensure_ascii=False,sort_keys=True)+"\n")
                if any(a["observation"]!=b["observation"] or a["action"]!=b["action"] for a,b in zip(original,changed)):
                    failures.append({"kind":"hidden_information_changes_public_teacher_prefix","split":role,"root_id":episode["root_id"]})
                counts.update(episodes=1,counterfactual_phase_prefixes=3)
                by_split[role].update(episodes=1,C2=metrics["C2"],FJS=metrics["FJS"],unsupported_operations=metrics["unsupported_operations"],invalid_outputs=metrics["invalid_outputs"])
                target.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n");target.flush()
            print(json.dumps({"data_audit_completed":role,"episodes":by_split[role]["episodes"],"failures":len(failures)}),flush=True)
    return {"reviewer_type":"ai_contract_audit","status":"verified" if not failures else "failed","counts":dict(counts),
        "by_split":{name:dict(values) for name,values in by_split.items()},"PI1_controls":dict(pi1),"failures":failures,
        "evaluation_policy":"Evaluation only static identity/shape/provenance coverage; no evaluation B4 or model scores computed.",
        "artifacts":{label:{"path":name,"sha256":_sha(output/name)} for label,name in
            (("natural_traces","audit_contract_traces.jsonl"),("PI1_traces","audit_pi1_traces.jsonl"),("counterfactual_prefixes","audit_counterfactual_prefixes.jsonl"))},
        "Qwen_calls":0,"training_steps":0,"limits":"B4 consistency and counterfactual public-prefix checks; not independent human truth or a confirmatory result."}


def _sources():
    paths=[Path(__file__),ROOT/"research/varied_development.py",ROOT/"research/varied_development_v3_003.py",
        ROOT/"research/src/data/semantic_partition_v2.py",ROOT/"research/src/data/semantic_partition_v3.py",
        ROOT/"research/src/policy/rule_controller.py",ROOT/"research/src/policy/dialogue.py",
        ROOT/"research/src/policy/llm_controller.py",ROOT/"research/src/memory/store.py",ROOT/"research/src/memory/executor.py",
        ROOT/"research/src/env/episode.py",ROOT/"research/src/env/user_simulator.py",ROOT/"research/src/scoring/evidence.py",
        ROOT/"research/src/scoring/reward.py",
        ROOT/"research/src/readers/symbolic.py",ROOT/"research/src/evaluation/runner.py"]
    return {str(path.relative_to(ROOT)):path for path in paths}


def prepare(output, *, seed, partition_seed, sizes=None, historical_paths=None, emit_teachers=True, audit_contracts=True):
    sizes=_sizes(sizes or DEFAULT_SIZES);output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError("Never overwrite a corpus or partial build; choose a new version")
    output.mkdir(parents=True,exist_ok=True)
    historical_paths={str(k):Path(v) for k,v in (discover_previous_banks(exclude=output) if historical_paths is None else historical_paths).items()}
    if not historical_paths:raise ValueError("Explicitly empty historical exclusion is not allowed")
    historical,adaptations=_historical_views(historical_paths)
    before={name:_sha(path) for name,path in historical_paths.items()};sources=_sources();source_before={name:_sha(path) for name,path in sources.items()}
    snapshots=output/"source_snapshot";snapshots.mkdir()
    snapshot_files={}
    for name,path in {**historical_paths,**sources}.items():
        target=snapshots/(_sha(path)[:16]+"_"+path.name);shutil.copyfile(path,target);snapshot_files[name]=str(target.relative_to(output))
    splits=partition_candidates(build_candidates(seed=seed,sizes=sizes),sizes,seed=partition_seed)
    isolation=assert_semantic_split_isolation_v3(splits,historical_splits=historical)
    static_failures=_static_audit(splits)
    if static_failures:
        _write_json(output/"failed_static_audit.json",static_failures);raise ValueError("Literal data contract failed; do not retouch gold to favor a method")
    partitions={}
    for name,rows in splits.items():
        path=output/(name+".jsonl")
        with path.open("x") as stream:
            for row in rows:stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
        partitions[name]={"path":path.name,"sha256":_sha(path),**diversity_plan(rows),
            "observability_controls_by_root":dict(Counter(e["authoring"].get("observability_control",{}).get("kind","not_control") for e in {e["root_id"]:e for e in rows}.values()))}
    teachers={}
    if emit_teachers:
        for name in ("train","validation"):
            path=output/("teacher_"+name+".jsonl");count=0;phases=Counter()
            with path.open("x") as stream:
                for row in teacher_examples(splits[name],work_directory=output/"teacher_work"):
                    stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n");count+=1;phases[row["phase"]]+=1
            teachers[name]={"path":str(path.relative_to(output)),"sha256":_sha(path),"examples":count,"phases":dict(phases),"routes":list(DEFAULT_ROUTES)}
    audit=_audit_training_contracts(splits,output) if audit_contracts else {"status":"not_run","reason":"Fixture-only build; cannot open training gates"}
    audit.update(static_failures=static_failures,isolation=isolation)
    _write_json(output/"audit_contracts.json",audit)
    after={name:_sha(path) for name,path in historical_paths.items()};source_after={name:_sha(path) for name,path in sources.items()}
    if before!=after or source_before!=source_after:raise ValueError("Frozen data/teacher source changed during build")
    manifest={"version":VERSION,"data_namespace":output.name,"reviewer_type":"ai_authored_prospective_data","human_review":False,
        "confirmatory_test":False,"seed":seed,"partition_seed":partition_seed,"sizes":sizes,"partitions":partitions,"teachers":teachers,
        "isolation":isolation,"historical_paths":{name:str(path.resolve()) for name,path in historical_paths.items()},
        "historical_hashes_before":before,"historical_hashes_after":after,"historical_identity_adaptations":adaptations,
        "source_hashes_before":source_before,"source_hashes_after":source_after,"source_snapshot_files":snapshot_files,
        "audit_contracts":{"path":"audit_contracts.json","sha256":_sha(output/"audit_contracts.json"),"status":audit["status"]},
        "model_inference_calls_before_creation":0,"Qwen_calls":0,"training_steps":0,
        "evaluation_use":"Held-back development; never checkpoint selection, tuning, teacher training or confirmatory test.",
        "teacher_contract":"RuleController v5 sees only real public observations. Natural SELECT once, RESPOND/WRITE from natural, PROCEED and ASK:TEMPORAL_SCOPE. No latent truth/future query in targets or prompts.",
        "limits":["Reuses five known v5 recipes; exact reference-normalized independence does not establish structural OOD, human realism or paraphrase independence.",
            "New dates/values/IDs are controlled generation variation, not conceptual diversity.",
            "One principal current query per episode and one same-attribute future query; earlier dialogue is scripted context.",
            "Rule teacher targets are a symbolic control, not independent human gold.",
            "Very old incomplete smoke banks retain public-prefix identity checks with missing-private-field sentinels documented in historical_identity_adaptations."]}
    # Return the same JSON-normalized schema that readers load (count keys are strings).
    manifest=json.loads(json.dumps(manifest,ensure_ascii=False))
    _write_json(output/"manifest.json",manifest)
    if audit["status"]=="failed":raise ValueError("B4 contract mismatch; preserve evidence and investigate without changing gold")
    return manifest


def load_and_check(output, *, historical_paths=None, verify_sources=True, require_audit=True):
    output=Path(output);manifest=json.loads((output/"manifest.json").read_text());splits={}
    if require_audit and manifest["audit_contracts"]["status"]!="verified":raise ValueError("Verified corpus contracts are required")
    for name,info in manifest["partitions"].items():
        path=output/info["path"]
        if _sha(path)!=info["sha256"]:raise ValueError("Bank SHA mismatch: "+name)
        splits[name]=_load(path)
        if any(e["split"]!=name for e in splits[name]):raise ValueError("Partition label mismatch")
    for info in [*manifest["teachers"].values(),manifest["audit_contracts"]]:
        if _sha(output/info["path"])!=info["sha256"]:raise ValueError("Teacher/audit SHA mismatch")
    audit=json.loads((output/manifest["audit_contracts"]["path"]).read_text())
    for info in audit.get("artifacts",{}).values():
        if _sha(output/info["path"])!=info["sha256"]:raise ValueError("Audit trace SHA mismatch")
    paths={k:Path(v) for k,v in (manifest["historical_paths"] if historical_paths is None else historical_paths).items()}
    if not paths:raise ValueError("Explicitly empty historical exclusion is not allowed")
    if any(_sha(paths[name])!=value for name,value in manifest["historical_hashes_after"].items()):raise ValueError("Previous bank changed after freeze")
    if verify_sources and any(_sha(ROOT/name)!=value for name,value in manifest["source_hashes_after"].items()):raise ValueError("Teacher/data source changed after freeze")
    historical,_=_historical_views(paths)
    assert_semantic_split_isolation_v3(splits,historical_splits=historical)
    return splits,manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    parser.add_argument("--seed",type=int,required=True);parser.add_argument("--partition-seed",type=int,required=True)
    args=parser.parse_args(argv)
    result=prepare(args.output,seed=args.seed,partition_seed=args.partition_seed)
    print(json.dumps({"output":args.output,"sizes":result["sizes"],"audit":result["audit_contracts"]},ensure_ascii=False))


if __name__=="__main__":main()
