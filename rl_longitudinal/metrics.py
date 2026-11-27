"""User-clustered metrics. Sessions are never treated as independent people."""
from collections import defaultdict
from rl_feasibility.statistics import bootstrap_mean, mean


def annotate(row, scenario):
    """Derive denominators from action traces and author-specified facts.

    This does not invoke the agent's temporal reconciler. Relevance for retrieval
    is deliberately broad: all communicated history about the target food plus
    restrictions, or all food history for an open query, including expired facts.
    This is retention/coverage, not validation of semantic retrieval ranking.
    """
    from .reward import independent_applicable, instant
    observations=row['policy_observations']
    steps=row['steps']
    if len(observations)!=len(steps):
        raise ValueError('Every action must have exactly one logged public observation')
    reference_history=[]
    def token(f):
        return (f['evidence_id'],f['food'],f['polarity'])
    for index, session in enumerate(row['sessions']):
        spec=scenario['sessions'][index]
        initial=next(o for o,s in zip(observations,steps) if s['session_index']==index)
        observed=[o for o,s in zip(observations,steps) if s['session_index']==index]
        terminal=observed[-1]
        traces=session['steps']
        questions=[s for s in traces if s['executed_action']['conversation']=='CLARIFY']
        session['asks']=len(questions)
        session['unnecessary_asks']=sum(not s.get('necessary_question',False) for s in questions)
        session['repeated_asks']=sum(bool(s.get('repeated_question')) for s in questions)
        session['restriction_violations']=int(session['restriction_violation'])
        forbidden=set(spec['gold'].get('restrictions',[]))
        session['attempted_restriction_violations']=sum(
            s['proposed_action'].get('food') in forbidden and s['proposed_action']['conversation']=='ANSWER'
            for s in traces)
        session['shield_interventions']=sum(bool(s.get('shield_reasons') or s.get('memory_violations')) for s in traces)
        session['actions_proposed']=len(traces)
        session['needed_asks']=int(spec['gold'].get('needs_clarification',False))
        session['needed_asks_performed']=int(any(s.get('necessary_question') for s in questions))
        session['unjustified_updates']=sum('UNJUSTIFIED_STABLE_UPDATE' in s.get('memory_violations',[]) for s in traces)
        session['memory_updates']=sum(s['proposed_action']['memory_operation']=='UPDATE_WITH_HISTORY' for s in traces)
        target=spec.get('target_food')
        history=[f for f in reference_history if not target or f['food']==target or f['kind']=='RESTRICTION']
        expected={token(f) for f in history}
        actual={token(f) for f in initial['historical_memory']}
        session['historical_relevant']=len(expected)
        session['historical_retrieved']=len(expected & actual)
        reference={f['memory_id']:f for f in reference_history+spec['incoming']}
        # Independent annotations for replies count ONLY after that question
        # produced the actual public response, including extractor omissions.
        reply_facts=[]
        for trace in questions:
            reply=spec.get('responses',{}).get(trace['executed_action'].get('question_key'),{})
            fact=reply.get('fact')
            if fact:
                value=dict(fact,memory_id='clar_'+fact['memory_id'],evidence_id='clar_'+fact['evidence_id'])
                reference[value['memory_id']]=value
                reply_facts.append(value)
        originals=list(reference.values())
        by_id={f['memory_id']:f for f in originals}
        effective_ends={f['memory_id']:f.get('valid_to') for f in originals}
        for new in originals:
            if new.get('kind') not in ('EXPLICIT_CHANGE','CLARIFICATION') or not new.get('valid_from'):
                continue
            for old_id in new.get('supersedes',[]):
                old=by_id.get(old_id)
                if old and old['food']==new['food'] and old['scope']==new['scope'] and old['kind']!='RESTRICTION':
                    current=effective_ends[old_id]
                    if current is None or instant(new['valid_from'])<instant(current):
                        effective_ends[old_id]=new['valid_from']
        expected_applicable=set()
        for original in originals:
            value=dict(original,valid_to=effective_ends[original['memory_id']])
            if (not target or value['food']==target or value['kind']=='RESTRICTION') and independent_applicable(value,spec['query_at'],terminal['context']):
                expected_applicable.add(token(value))
        public_facts=terminal['historical_memory']+terminal['incoming_evidence']
        public_byid={f['memory_id']:f for f in public_facts}
        predicted={token(public_byid[i]) for i in terminal['reconciliation']['applicable_ids'] if i in public_byid}
        universe={token(f) for f in originals}|{token(f) for f in public_facts}
        session['temporal_total']=len(universe)
        session['temporal_correct']=sum((t in expected_applicable)==(t in predicted) for t in universe)
        # Type/scope extraction accuracy is measured against communicated facts,
        # not a hidden preference or a policy's chosen action.
        parsed={token(f):f for f in public_facts}
        current_facts=spec['incoming']+reply_facts
        session['interpretation_total']=len(current_facts)
        session['interpretation_correct']=sum(
            token(f) in parsed and all(parsed[token(f)].get(k)==f.get(k)
                                      for k in ('kind','scope','valid_from','valid_to','source'))
            for f in current_facts)
        # Purely causal propagation is computed by matched interventions instead
        # of labelling every future mistake a propagated memory error.
        session['future_sessions']=0
        session['propagated_future_errors']=0
        # Initial exposures explicitly annotated as material source/temporal/
        # restriction conflicts. Resolution at the terminal step cannot erase a
        # true positive; unrelated open-query conflicts cannot inflate recall.
        eligible=bool(target and spec['gold'].get('needs_clarification') and scenario['family'] in
                      ('SOURCE_CHANGE','AMBIGUOUS_BEHAVIOR','RESTRICTION','UNANSWERED'))
        rec=initial['reconciliation']
        detected=bool(rec.get('conflicts') or rec.get('uncertain_foods') or target in rec.get('forbidden_foods',[]))
        session['relevant_conflicts']=int(eligible)
        session['conflicts_detected']=int(eligible and detected)
        reference_history=list(reference.values())
    row['metric_annotation']='independent_author_facts_plus_actual_public_replies_v1'
    return row


def ratio(numerator, denominator):
    return {'numerator':numerator, 'denominator':denominator,
            'value':numerator / denominator if denominator else None}


def _session_count(row, key):
    return sum(int(s.get(key, 0)) for s in row['sessions'])


def summarize(rows, repetitions=1000):
    """Consume independent evaluator diagnostics, not policy-chosen labels."""
    sessions = [s for row in rows for s in row['sessions']]
    users = len(rows)
    outcomes = [s for s in sessions if 'correct' in s]
    followups = [s for s in outcomes if s.get('session_index', 0) > 0]
    answers = [s for s in sessions if s.get('action', {}).get('conversation') == 'ANSWER']
    per_user_accuracy = [sum(bool(s['correct']) for s in row['sessions']) / len(row['sessions'])
                         for row in rows if row['sessions']]
    per_user_followup = [mean(float(s['correct']) for s in row['sessions']
                             if s.get('session_index', 0) > 0) for row in rows]
    per_user_followup = [v for v in per_user_followup if v is not None]
    def count(key):
        return sum(int(s.get(key, 0)) for s in sessions)
    results = {
        'users':users, 'sessions':len(sessions),
        'return_per_user':bootstrap_mean([r['return'] for r in rows], repetitions),
        'return_per_session':bootstrap_mean([r['return'] / len(r['sessions']) for r in rows], repetitions),
        'decision_accuracy_user_macro':bootstrap_mean(per_user_accuracy, repetitions),
        'decision_accuracy':ratio(sum(bool(s['correct']) for s in outcomes), len(outcomes)),
        'followup_accuracy':ratio(sum(bool(s['correct']) for s in followups), len(followups)),
        'followup_accuracy_user_macro':bootstrap_mean(per_user_followup, repetitions),
        'answer_evidence_fidelity':ratio(sum(bool(s.get('grounded')) for s in answers), len(answers)),
        'clarifications':count('asks'),
        'unnecessary_clarification_rate':ratio(count('unnecessary_asks'), count('asks')),
        'repeated_clarification_rate':ratio(count('repeated_asks'), count('asks')),
        'restriction_violation_rate':ratio(count('restriction_violations'), len(sessions)),
        'attempted_restriction_violations':count('attempted_restriction_violations'),
        'shield_interventions':count('shield_interventions'),
        'proposed_actions':count('actions_proposed'),
        'latency_per_user_ms':bootstrap_mean([r.get('elapsed_ms', 0) for r in rows], repetitions),
    }
    # Never manufacture component accuracy when no independent annotation exists.
    for metric, num, den in (
        ('historical_relevant_recall','historical_retrieved','historical_relevant'),
        ('temporal_applicability_accuracy','temporal_correct','temporal_total'),
        ('interpretation_accuracy','interpretation_correct','interpretation_total'),
        ('attempted_unjustified_update_rate','unjustified_updates','memory_updates'),
        ('relevant_conflict_recall','conflicts_detected','relevant_conflicts'),
        ('necessary_clarification_recall','needed_asks_performed','needed_asks'),
        ('memory_operation_accuracy','correct_memory_operations','evaluated_memory_operations'),
        ('proposed_memory_operation_accuracy','proposed_correct_memory_operations','evaluated_memory_operations'),
        ('propagated_future_error_rate','propagated_future_errors','future_sessions'),
    ):
        results[metric] = ratio(count(num), count(den))
    results['unjustified_overwrite_rate']=ratio(0,count('evaluated_memory_operations'))
    groups = defaultdict(list)
    for row in rows:
        groups[row['family']].append(row)
    results['by_family'] = {
        family:{'users':len(group),
                'return_per_session':mean(r['return']/len(r['sessions']) for r in group),
                'decision_accuracy':ratio(sum(bool(s['correct']) for r in group for s in r['sessions']),
                                          sum(len(r['sessions']) for r in group)),
                'followup_accuracy':ratio(sum(bool(s['correct']) for r in group for s in r['sessions']
                                             if s.get('session_index',0)>0),
                                         sum(s.get('session_index',0)>0 for r in group for s in r['sessions'])),
                'clarifications':sum(_session_count(r,'asks') for r in group)}
        for family,group in sorted(groups.items())}
    results['by_previous_session_count'] = {}
    for index in sorted({s.get('session_index',0) for s in sessions}):
        group = [s for s in sessions if s.get('session_index',0)==index]
        results['by_previous_session_count'][str(index)] = {
            'sessions':len(group), 'decision_accuracy':ratio(sum(bool(s['correct']) for s in group),len(group))}
    return results


def paired(left, right, field='return_per_session', repetitions=1000):
    a = {row['episode_id']:row for row in left}
    b = {row['episode_id']:row for row in right}
    if len(a)!=len(left) or len(b)!=len(right) or a.keys()!=b.keys():
        raise ValueError('Comparisons require identical unique users')
    def value(row):
        if field == 'return_per_session':
            return row['return'] / len(row['sessions'])
        selected = row['sessions'] if field == 'accuracy' else [s for s in row['sessions'] if s.get('session_index',0)>0]
        return mean(float(s['correct']) for s in selected)
    differences = [value(a[key])-value(b[key]) for key in sorted(a)]
    return bootstrap_mean(differences, repetitions)
