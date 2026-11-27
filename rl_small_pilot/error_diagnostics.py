"""Private evaluator diagnostics. Never import this module into policy inference.

Field discrepancies are observations; cause labels are explicitly probable.
The paired extraction intervention identifies pipeline-sensitive failures but
cannot by itself prove that a neural policy could correct missing evidence.
"""
from collections import Counter
from copy import deepcopy
from rl_observability.policies import public_analysis

CAUSES=('historical_retrieval','linguistic_extraction','date_or_validity',
        'context_interpretation','conflict_detection','question_selection',
        'response_decision','memory_update','rule_repair','undetermined')

def diagnose_rows(rows):
    cases=[]
    for row in rows:
        records={r.get('evidence_id'):r for r in row.get('extraction_records',[])}
        for index,step in enumerate(row.get('steps',[])):
            if not step.get('terminal') or step.get('correct'):
                continue
            causes=[]; evidence=[]
            obs=step['observation']; analysis=public_analysis(obs)
            visible_ids={f['memory_id'] for f in obs.get('memory',[])}
            available=step.get('available_memories',[])+step.get('pending_evidence',[])
            missing=[f for f in available if f['food']==obs['target_food'] and f['memory_id'] not in visible_ids]
            if missing:
                causes.append('historical_retrieval'); evidence.append('Relevant available fact omitted from exposed history; may be deliberate O1 ablation.')
            visible_eids={f['evidence_id'] for f in obs.get('memory',[])}
            mismatches=[r for key,r in records.items() if key in visible_eids and r.get('extraction_error')]
            # Missing extracted facts are absent from visible_eids; retain those
            # that were actually communicated by the current session time.
            mismatches += [r for key,r in records.items() if key not in visible_eids and 'missing_fact' in r.get('fields',[]) and (r.get('declared') or {}).get('observed_at','')<=step.get('at','')]
            if row.get('condition',row.get('extraction'))=='PIPELINE' and mismatches:
                causes.append('linguistic_extraction'); evidence.append('Field discrepancy precedes this decision; causal impact requires paired ORACLE comparison.')
                if any(r.get('temporal_error') for r in mismatches):
                    causes.append('date_or_validity')
                if any(any(f.startswith('scope.') for f in r.get('fields',[])) for r in mismatches):
                    causes.append('context_interpretation')
            if analysis['missing']:
                causes.append('context_interpretation'); evidence.append('Public interpretation reports missing contextual slots.')
            if step.get('incorrect_updates'):
                causes.append('memory_update'); evidence.append('Unjustified update actually executed; additional penalty recorded.')
            if step.get('shield_reasons'):
                causes.append('rule_repair'); evidence.append('A shield intervened; proposed/executed actions retained separately.')
            prior=row.get('steps',[])[:index]
            if any(s.get('executed_action',{}).get('conversation')=='CLARIFY' and not s.get('necessary_question') for s in prior if s.get('session_index')==step.get('session_index')):
                causes.append('question_selection')
            if step.get('session_index')==2 and not obs.get('incoming_evidence'):
                prior_pending=[f for s in prior for f in s.get('pending_evidence',[]) if f.get('food')==obs['target_food']]
                if any(f['memory_id'] not in visible_ids for f in prior_pending):
                    causes.append('memory_update'); evidence.append('Earlier communicated target evidence is absent in session three; omission/indexing is a probable contributor.')
            if not causes:
                if analysis['conflict'] and step['executed_action']['conversation']=='ANSWER':
                    causes.append('conflict_detection'); evidence.append('Answer proposed despite public detectable disagreement; internal neural detection is unobserved.')
                else:
                    causes.append('response_decision'); evidence.append('No diagnosed upstream discrepancy in exposed evidence; probable action-selection error, not proven unique cause.')
            causes=list(dict.fromkeys(causes)) or ['undetermined']
            multiplicity='undetermined' if causes==['undetermined'] else 'single' if len(causes)==1 else 'multiple'
            cases.append(dict(user_id=row['user_id'],family=row['family'],
                representation=row['representation'],condition=row.get('condition',row.get('extraction')),
                session_index=step['session_index'],causes=causes,multiplicity=multiplicity,
                attribution='probable_not_independently_causally_identified',evidence=evidence,
                original_input=deepcopy(step.get('original_input',[])),
                available_memories=deepcopy(step.get('available_memories',[])),
                pending_evidence=deepcopy(step.get('pending_evidence',[])),
                retrieved_memories=deepcopy(step.get('retrieved_memories',[])),
                observation=deepcopy(obs),proposed_action=deepcopy(step['proposed_action']),
                executed_action=deepcopy(step['executed_action']),reward=step['reward'],
                final_correct=step['correct'],unsupported_answer=step.get('unsupported_answer',False)))
    counts=Counter(c for case in cases for c in case['causes'])
    total=len(cases)
    return dict(cases=cases,summary=dict(failed_decisions=total,
        categories={cause:dict(count=counts[cause],percent=100*counts[cause]/max(1,total)) for cause in CAUSES},
        multiplicity=dict(Counter(case['multiplicity'] for case in cases)),
        denominator='failed terminal decisions; cause percentages can overlap',
        limitations='Stage labels are probable; absence of a measured discrepancy is not proof of absence. Shared fixed symbolic interpretation is not neural language understanding.'))

def compare_paired(oracle_rows,pipeline_rows):
    oracle={r['user_id']:r for r in oracle_rows}; pipeline={r['user_id']:r for r in pipeline_rows}
    common=sorted(set(oracle)&set(pipeline))
    def outcomes(row):return {s['session_index']:s['correct'] for s in row.get('steps',[]) if s.get('terminal')}
    counter=Counter()
    for uid in common:
        a,b=outcomes(oracle[uid]),outcomes(pipeline[uid])
        for stage in sorted(set(a)&set(b)):
            counter['terminal_pairs']+=1
            counter['oracle_errors']+=not a[stage]
            counter['pipeline_errors']+=not b[stage]
            counter['pipeline_only_errors']+=a[stage] and not b[stage]
            counter['oracle_only_errors']+=not a[stage] and b[stage]
            counter['both_errors']+=not a[stage] and not b[stage]
    return dict(users=len(common),**counter,
        interpretation='Paired exposure intervention on the same communicated evidence. Pipeline-only failure is extraction/adapter-sensitive; it is not evidence that RL can reconstruct unavailable facts.')

# Explicit aliases for consumers with descriptive names.
analyze_errors=diagnose_rows
