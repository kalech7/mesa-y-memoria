"""Replacement revision of varied v3 with prospective PI1 support-loss controls.

Read the frozen 001 corpus; never rewrite it or its generator. Only F0 controls
with at least two initial memories replace the last distractor with a recent,
expired preference for the queried attribute. Literal current/future golds and
all current-dialogue turns remain unchanged. This is not independent of 001.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import shutil

from .src.data.semantic_partition_v3 import assert_semantic_split_isolation_v3, semantic_episode_id_v3

VERSION = "varied_scripted_development_v3.1_pi1_control"
NAMESPACE = "development_varied_v3_20261025_002"
CLAUSE = {"dinner_preference":"para la cena", "drink_preference":"para beber", "breakfast_preference":"para el desayuno"}
POOL_KEY = {"dinner_preference":"dishes_observed", "drink_preference":"drinks_observed", "breakfast_preference":"breakfasts_observed"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def revise_for_observability(episodes, parent_manifest):
    revised=deepcopy(episodes);changes=[]
    for episode in revised:
        episode.update(data_namespace=NAMESPACE,generator_version=VERSION)
        episode["authoring"]=deepcopy(episode["authoring"])
        episode["authoring"].update(replacement_revision="002",parent_namespace=parent_manifest["data_namespace"],
                                    lineage_root_id=episode["root_id"],revision_source="PI1 observability requirement before B1 varied-bank inference; not selection on model errors.")
        if episode["authoring"]["stratum"]!="unrelated_attribute_control":continue
        if len(episode["initial_memory"])==1:
            episode["authoring"]["observability_control"]={"kind":"single_memory_negative_control","expected_O1_C2":1,"expected_O2_C2":1}
            continue
        primary=episode["current_query"]["attribute"]
        base=episode["initial_memory"][0]
        old=episode["initial_memory"][-1]
        assert base["memory_id"]=="m1" and base["attribute"]==primary and base["scope"]["kind"]=="ongoing"
        assert old["attribute"]!=primary
        assert not any(old["memory_id"] in r.get("supersedes",[]) for r in episode["initial_memory"][:-1])
        source_id=old["source_turn_ids"][0]
        original_session=next(s for s in episode["prior_sessions"] if any(m["turn_id"]==source_id for m in s["messages"]))
        yesterday=date.fromisoformat(episode["current_query"]["date"])-timedelta(days=1)
        today=date.fromisoformat(episode["current_query"]["date"])
        values=[v for v in parent_manifest[POOL_KEY[primary]] if v!=base["value"]]
        # Fixed lexical choice is merely a distractor, not a new structural OOD claim.
        value=values[int(hashlib.sha256(episode["root_id"].encode()).hexdigest()[:8],16)%len(values)]
        text=f"Solo el {yesterday.isoformat()}, {CLAUSE[primary]} prefiero {value}."
        if len(original_session["messages"])==1:
            original_session["date"]=yesterday.isoformat()
            original_session["messages"][0]={"role":"user","turn_id":source_id,"text":text}
            new_source_id=source_id
        else:
            if len(episode["prior_sessions"])>=3:
                raise ValueError("This parent topology needs an explicit additional history recipe; do not fabricate a fourth prior session")
            original_session["messages"]=[m for m in original_session["messages"] if m["turn_id"]!=source_id]
            session_number=len(episode["prior_sessions"])+1
            new_source_id=f"p{session_number}u1"
            episode["prior_sessions"].append({"session_id":f"P{session_number}","date":yesterday.isoformat(),
                "messages":[{"role":"user","turn_id":new_source_id,"text":text}]})
        replacement={"memory_id":old["memory_id"],"attribute":primary,"value":value,
                     "scope":{"kind":"interval","valid_from":yesterday.isoformat(),"valid_until_exclusive":today.isoformat(),"context":None,"session_id":None},
                     "recorded_at":yesterday.isoformat(),"status":"confirmed","supersedes":[],
                     "source_turn_ids":[new_source_id],"evidence_quotes":[{"turn_id":new_source_id,"text":text}]}
        episode["initial_memory"][-1]=replacement
        episode["authoring"]["parent_composition_id"]=episode["composition_id"]
        episode["composition_id"]=(f"v3.1:unrelated_attribute_control:pi1_expired_latest:m{len(episode['initial_memory'])}"
                                     f":p{len(episode['prior_sessions'])}:t{len(episode['user_messages'])}:{primary}")
        episode["template_id"] += ":pi1_control"
        assert replacement["recorded_at"]>base["recorded_at"]
        episode["authoring"]["observability_control"]={"kind":"expired_latest_same_attribute_record",
            "base_memory_id":"m1","expired_memory_id":replacement["memory_id"],
            "expected_O1_C2":0,"expected_O2_C2":1,"current_future_gold_unchanged":True}
        public={"memory":episode["initial_memory"],"user_messages":episode["user_messages"],"query":episode["current_query"]}
        episode["observable_state_id"]=digest(public)
        changes.append({"root_id":episode["root_id"],"world_id":episode["latent_world_id"],"old_record":old,
                        "new_record":replacement,"gold_current":episode["truth"]["current_value"],
                        "gold_future":episode["truth"]["future_value"],"expected_O1_C2":0,"expected_O2_C2":1})
    return revised,changes


def prepare_revision(parent_directory,data_directory,report_directory,*,historical_directory="research/data/bank_1_1",
                     previous_development="research/data/development_v2_20261024_001/validation.jsonl",
                     parent_generator="research/varied_development.py"):
    parent_directory,data_directory,report_directory=Path(parent_directory),Path(data_directory),Path(report_directory)
    parent_bank=parent_directory/"validation.jsonl";parent_manifest_path=parent_directory/"manifest.json"
    parent_manifest=json.loads(parent_manifest_path.read_text())
    assert sha(parent_bank)==parent_manifest["bank_sha256"]
    assert sha(parent_generator)==parent_manifest["producer_sha256"]
    data_directory.mkdir(parents=True,exist_ok=True);report_directory.mkdir(parents=True,exist_ok=True)
    if (data_directory/"validation.jsonl").exists() or (data_directory/"manifest.json").exists():
        raise FileExistsError("Preserve existing revisions; use a new output version")
    inputs={f"legacy_{split}":Path(historical_directory)/f"{split}.jsonl" for split in ("dev_smoke","train","validation","test_id","test_ood","test_external")}
    inputs["development_v2"]=Path(previous_development)
    before={name:sha(path) for name,path in inputs.items()}
    protected={"parent_bank":sha(parent_bank),"parent_manifest":sha(parent_manifest_path),"parent_generator":sha(parent_generator)}
    snapshots=report_directory/"source_snapshot";snapshots.mkdir(exist_ok=True)
    # Preserve the 001 producer and data before applying the explicit revision.
    for name,path in {**inputs,"parent_bank":parent_bank,"parent_manifest":parent_manifest_path,
                      "parent_generator":Path(parent_generator),"revision_generator":Path(__file__)}.items():
        target=snapshots/(name+path.suffix)
        if target.exists() and sha(target)!=sha(path):raise ValueError("Snapshot already exists with different source")
        if not target.exists():shutil.copyfile(path,target)
    original=[json.loads(line) for line in parent_bank.open()]
    episodes,changes=revise_for_observability(original,parent_manifest)
    assert len(episodes)==60 and len({e["root_id"] for e in episodes})==50 and len(changes)==9
    for old,new in zip(original,episodes):
        assert old["truth"]==new["truth"] and old["user_messages"]==new["user_messages"]
        assert old["current_query"]==new["current_query"] and old["future_query"]==new["future_query"]
        assert old["root_id"]==new["root_id"] and old["latent_world_id"]==new["latent_world_id"]
    historical={name:[json.loads(line) for line in path.open()] for name,path in inputs.items()}
    isolation=assert_semantic_split_isolation_v3({"validation":episodes},historical_splits=historical)
    old_signatures={semantic_episode_id_v3(e) for e in original}
    repeated=sum(semantic_episode_id_v3(e) in old_signatures for e in episodes)
    bank=data_directory/"validation.jsonl"
    with bank.open("x") as stream:
        for episode in episodes:stream.write(json.dumps(episode,ensure_ascii=False,sort_keys=True)+"\n")
    roots={e["root_id"]:e for e in episodes}
    manifest=deepcopy(parent_manifest)
    manifest.update(version=VERSION,data_namespace=NAMESPACE,bank_path=str(bank.resolve()),bank_sha256=sha(bank),
        producer_sha256=sha(__file__),parent_bank_path=str(parent_bank.resolve()),parent_bank_sha256=protected["parent_bank"],
        lineage={"parent_bank":str(parent_bank.resolve()),"parent_bank_sha256":protected["parent_bank"],
                 "relationship":"revision_of_same_corpus_not_independent","model_inference_calls_before_revision":0},
        parent_generator_sha256=protected["parent_generator"],parent_manifest_sha256=protected["parent_manifest"],
        replaces_namespace=parent_manifest["data_namespace"],independent_of_parent=False,
        lineage_root_ids_preserved=True,revision_changes=9,changed_roots=9,observability_positive_controls=9,observability_negative_controls=1,
        expected_parent_semantic_overlap_episodes=repeated,root_overlap_with_parent=50,
        prior_sessions_by_root=dict(sorted(Counter(len(e["prior_sessions"]) for e in roots.values()).items())),
        template_ids=sorted({e["template_id"] for e in episodes}),composition_ids=sorted({e["composition_id"] for e in episodes}),
        historical_hashes_before=before,historical_hashes_after={name:sha(path) for name,path in inputs.items()},
        isolation=isolation,revision_selection_basis="PI1 observability support-loss requirement before any B1 varied-v3 inference; no method-result selection.",
        parent_protection_before=protected,parent_protection_after={"parent_bank":sha(parent_bank),"parent_manifest":sha(parent_manifest_path),"parent_generator":sha(parent_generator)})
    assert manifest["historical_hashes_before"]==manifest["historical_hashes_after"]
    assert manifest["parent_protection_before"]==manifest["parent_protection_after"]
    assert repeated==51
    manifest["limits"]=list(manifest["limits"])+["Revision002 replaces001, shares50lineage roots and51/60episode identities; never pool both as independent users.",
        "Nine PI1 controls alter only a historical distractor and its supporting user statement/session; all current/future golds and current dialogues are unchanged.",
        "O1 support-loss contrast is to be verified with B4 before model inference; it is not a reported B1 improvement."]
    (data_directory/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    (report_directory/"revision_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    (report_directory/"changes.json").write_text(json.dumps(changes,ensure_ascii=False,indent=2)+"\n")
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-directory",default="research/data/development_varied_v3_20261025_001")
    parser.add_argument("--data-directory",required=True);parser.add_argument("--report-directory",required=True)
    parser.add_argument("--historical-directory",default="research/data/bank_1_1")
    parser.add_argument("--previous-development",default="research/data/development_v2_20261024_001/validation.jsonl")
    parser.add_argument("--parent-generator",default="research/varied_development.py")
    args=parser.parse_args(argv)
    manifest=prepare_revision(args.parent_directory,args.data_directory,args.report_directory,historical_directory=args.historical_directory,
                              previous_development=args.previous_development,parent_generator=args.parent_generator)
    print(json.dumps({key:manifest[key] for key in ("version","roots","episodes","bank_path","bank_sha256","observability_positive_controls","observability_negative_controls","expected_parent_semantic_overlap_episodes")},ensure_ascii=False))


if __name__=="__main__":main()
