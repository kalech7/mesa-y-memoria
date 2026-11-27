"""Read-only result checks and a blind raw-text review package (no adjudication)."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import xml.etree.ElementTree as ET


def inspect(run):
    from .evaluate import write_json
    run=Path(run)
    frozen=json.loads((run/'freeze.json').read_text())
    hashes={name:Path(name).exists() and hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected
            for name,expected in frozen['sources'].items()}
    snapshot_hashes={name:hashlib.sha256((run/'source_snapshot'/name).read_bytes()).hexdigest()==expected
                     for name,expected in frozen['sources'].items()}
    metrics=json.loads((run/'longitudinal_metrics.json').read_text())
    all_bounds=[]
    for condition,policies in metrics.items():
        for name,summary in policies.items():
            for key,value in summary.items():
                if isinstance(value,dict) and 'value' in value and value['value'] is not None:
                    all_bounds.append(0<=value['value']<=1)
    alignment={}
    for directory,condition in (('test','ORACLE_STRUCTURED'),('pipeline','PIPELINE'),('shift','FAMILY_HOLDOUT')):
        ids=[]
        for path in sorted((run/directory).glob('*.jsonl')):
            observed=[]
            for line in path.open():
                row=json.loads(line)
                observed.append(row['episode_id'])
            ids.append(observed)
        alignment[condition]=bool(ids and all(len(a)==len(set(a)) and a==ids[0] for a in ids))
    sqlite_results={}
    for path in sorted(run.glob('test/sqlite/*/*.sqlite')):
        db=sqlite3.connect('file:'+str(path.resolve())+'?mode=ro',uri=True)
        try:
            integrity=db.execute('PRAGMA integrity_check').fetchone()[0]
            fact_count=db.execute('SELECT count(*) FROM temporal_facts').fetchone()[0]
            retrieval_count=db.execute('SELECT count(*) FROM temporal_retrievals').fetchone()[0]
            sqlite_results[str(path.relative_to(run))]={'integrity':integrity,'facts':fact_count,'retrieval_traces':retrieval_count}
        finally:
            db.close()
    svg={p.name:ET.parse(p).getroot().tag=='{http://www.w3.org/2000/svg}svg' for p in run.glob('*.svg')}
    broken=[]
    report=run/'INFORME_RL_LONGITUDINAL.md'
    for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
        if '://' in target or target.startswith('#'):
            continue
        target=re.sub(r':\d+$','',target.split('#')[0])
        if not (run/target).exists():
            broken.append(target)
    result={'current_sources_match_freeze':all(hashes.values()),'source_hash_checks':hashes,
            'archived_sources_match_freeze':all(snapshot_hashes.values()),
            'all_metric_ratios_within_zero_one':all(all_bounds),
            'exact_same_unique_episode_ids_across_policies':alignment,
            'retained_sqlite_databases':sqlite_results,
            'all_retained_sqlite_integrity_ok':all(v['integrity']=='ok' for v in sqlite_results.values()),
            'svg_parse_checks':svg,'report_broken_local_links':broken}
    write_json(run/'artifact_verification.json',result)
    return result


def prepare_review(run,per_family=2):
    """Raw evidence, without kind/polarity/gold/policy labels; reviewers stay blank."""
    run=Path(run)
    scenarios={e['episode_id']:e for e in json.loads((run/'evaluator_private'/'test.json').read_text())}
    counts=Counter()
    gold_rows=[]
    response_rows=[]
    selected_users=[]
    path=run/'test'/'CLARIFICATION_RULE_BASED.jsonl'
    for line in path.open():
        row=json.loads(line)
        if counts[row['family']]>=per_family:
            continue
        counts[row['family']]+=1
        selected_users.append(row['episode_id'])
        scenario=scenarios[row['episode_id']]
        history=[]
        for index,session in enumerate(row['sessions']):
            spec=scenario['sessions'][index]
            for event in spec['events']:
                history.append({'observed_at':event['at'],'recorded_at':spec['at'],
                                'role':'user','source':event['source'],'text':event['text']})
            review_id=f"review_{len(selected_users):03d}_{index+1:02d}"
            base={'review_id':review_id,'history':json.dumps(history,ensure_ascii=False),
                  'query':spec['query'],'query_at':spec['query_at'],'communicated_context':json.dumps(spec['context'],ensure_ascii=False)}
            gold_rows.append({**base,'reviewer':'','acceptable_actions':'','acceptable_foods':'',
                              'clarification_needed':'','rationale':'','uncertainty':''})
            responses=[]
            for step in session['steps']:
                if 'response' in step:
                    responses.append({'question_key':step['executed_action']['question_key'],
                                      'response':step['response']})
                    history.append({'observed_at':spec['at'],'recorded_at':spec['at'],
                                    'role':'user','source':'clarification','text':step['response']})
            response_rows.append({**base,'interaction':json.dumps(responses,ensure_ascii=False),
                                  'decision':json.dumps(session['action'],ensure_ascii=False),
                                  'reviewer':'','question_relevant':'','grounded':'',
                                  'memory_operation_appropriate':'','rationale':''})
    for name,rows in [('gold_review_blind_raw.csv',gold_rows),('response_review_blind_raw.csv',response_rows)]:
        with (run/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ['review_id'])
            writer.writeheader();writer.writerows(rows)
    value={'external_adjudication':'NOT_PERFORMED','users':len(selected_users),'query_rows':len(gold_rows),
           'per_family_quota':per_family,'selection_rule':'first two test users per family, no outcome-based selection',
           'reviewer_fields':'intentionally empty','gold_labels_or_policy_names_in_blind_package':False}
    (run/'review_package_manifest.json').write_text(json.dumps(value,indent=2)+'\n')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps({'verification':inspect(args.run),'review_package':prepare_review(args.run)},ensure_ascii=False))


if __name__=='__main__':
    main()
