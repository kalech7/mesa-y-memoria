"""Public evidence adapters, with no import of private scenarios or evaluator.

`interpret_visible` is deliberately usable by every comparator. Giving O3 a cached
interpretation need not outperform an O2 policy capable of deriving it itself.
"""
from copy import deepcopy
from datetime import datetime
import re
import unicodedata

REPRESENTATIONS = {'O1':'RECENT_ONLY','O2':'TEMPORAL_HISTORY','O3':'TEMPORAL_ENRICHED'}


def _norm(value):
    return ''.join(c for c in unicodedata.normalize('NFD',str(value).lower()) if unicodedata.category(c)!='Mn')


def evidence_kind(fact):
    """Conservative interpretation of communicated text and source alone."""
    text=_norm(fact.get('text',fact.get('content','')))
    if fact.get('source')=='ORDER_HISTORY':
        return 'OBSERVED_BEHAVIOR'
    if fact.get('polarity')=='FORBID' or 'alergia' in text:
        return 'RESTRICTION'
    if any(word in text for word in ('solo hoy','solamente hoy','por hoy')):
        return 'TEMPORARY_EXCEPTION'
    if any(word in text for word in ('desde ahora','a partir de ahora','he cambiado de opinion','ahora me gusta','ahora no me gusta')):
        return 'CLARIFICATION' if fact.get('source')=='CLARIFICATION' else 'EXPLICIT_CHANGE'
    return 'STABLE_PREFERENCE'


def public_fact(fact, enriched=False):
    """Whitelist actual communicated evidence, not scenario author labels."""
    out={k:deepcopy(fact.get(k)) for k in ('memory_id','food','polarity','observed_at','recorded_at',
        'valid_from','valid_to','source','evidence_id','scope','supersedes','text')}
    out['content']=out['text']
    out['context']=deepcopy(out['scope'])
    if enriched:
        out['kind']=evidence_kind(out)
        out['interpretation_provenance']={'evidence_id':out['evidence_id'],
            'method':'visible_text_and_source_v1','fields':['content','source','observed_at','valid_from','valid_to','context']}
    return out


def _instant(value):
    return datetime.fromisoformat(value.replace('Z','+00:00'))


def interpret_visible(observation):
    """Derive temporal/conflict features solely from an observation's whitelist.

    No expected action is returned. Facts, potential conflict, and provenance are
    evidence claims; choosing to ask, preserve, answer or update remains a policy.
    """
    facts=deepcopy(observation.get('memory',[]))
    target=observation.get('target_food')
    at=observation.get('query_at')
    active=[]; expired=[]
    for f in facts:
        f['kind']=evidence_kind(f)
        lower=f.get('valid_from') or f.get('observed_at')
        upper=f.get('valid_to')
        if at and ((lower and _instant(at)<_instant(lower)) or (upper and _instant(at)>=_instant(upper))):
            expired.append(f)
        else:
            active.append(f)
    relevant=[f for f in active if f.get('food')==target]
    restrictions=[f for f in facts if f.get('polarity')=='FORBID']
    superseded={old for f in relevant for old in f.get('supersedes',[])}
    current=[f for f in relevant if f['memory_id'] not in superseded]
    # Explicit temporal evidence determines scoped precedence, independently of
    # whether the policy wrote a supersession edge to SQLite.
    changes=[f for f in current if f['kind'] in ('EXPLICIT_CHANGE','CLARIFICATION')]
    if changes:
        latest=max(changes,key=lambda f:(f['observed_at'],f['recorded_at'],f['memory_id']))
        current=[f for f in current if f['observed_at']>=latest['observed_at']]
        # A later communicated clarification resolves order-vs-statement evidence
        # from the same session; it is not a replacement inferred from hidden state.
        if latest['kind']=='CLARIFICATION' or latest.get('source')=='CLARIFICATION':
            current=[f for f in current if f['memory_id']==latest['memory_id'] or f['observed_at']>latest['observed_at']]
    temporary=[f for f in current if f['kind']=='TEMPORARY_EXCEPTION']
    if temporary:
        newest=max(f['observed_at'] for f in temporary)
        current=[f for f in current if f['observed_at']>=newest]
    polarities={f.get('polarity') for f in current if f.get('polarity') in ('LIKE','DISLIKE')}
    conflict=len(polarities)>1
    history=deepcopy(observation.get('clarification_history',[]))
    answered=any(h.get('status')=='RESOLVED' and h.get('question_key')=='preference' for h in history)
    unresolved=any(h.get('status')=='UNRESOLVED' and h.get('question_key')=='preference' for h in history)
    incoming_ids={f['memory_id'] for f in observation.get('incoming_evidence',[])}
    incoming=[f for f in facts if f['memory_id'] in incoming_ids]
    missing=[]
    for f in relevant:
        for slot,value in (f.get('scope') or {}).items():
            if value not in (None,'ANY') and not observation.get('context',{}).get(slot):
                missing.append(slot)
    return dict(facts=facts,active_facts=active,expired_facts=expired,relevant_facts=relevant,
        applicable_facts=current,incoming_facts=incoming,restrictions=restrictions,
        polarities=sorted(polarities),conflict=conflict,potential_conflicts=[dict(
            food=target,evidence_ids=[f['evidence_id'] for f in current]) ] if conflict else [],
        preference_types=sorted({f['kind'] for f in facts}),
        temporary_exception=bool(temporary),
        relations=[dict(relation='supersedes',subject=f['memory_id'],object=old,
                       evidence_id=f['evidence_id']) for f in facts for old in f.get('supersedes',[])],
        known_context=deepcopy(observation.get('context',{})),missing_context=sorted(set(missing)),
        clarification_answered=answered,clarification_unresolved=unresolved,
        clarification_history=history,
        provenance=[dict(evidence_id=f['evidence_id'],method='visible_text_temporal_v1',
                         fields=['content','source','query_at','valid_from','valid_to','context']) for f in facts])


def build_observation(store, pending, session, index, representation, question_count):
    if representation not in REPRESENTATIONS:
        raise ValueError('Unknown observation representation')
    retrieval=store.retrieve(session['query_at'],known_at=session['at'],context=session.get('context',{}),
                             food=session['target_food'])
    known={f.memory_id: f.__dict__ for f in retrieval.facts}
    known.update({f['memory_id']:deepcopy(f) for f in pending})
    all_facts=list(known.values())
    non_restrictions=[f for f in all_facts if f['polarity']!='FORBID']
    if representation=='O1' and non_restrictions:
        non_restrictions=[max(non_restrictions,key=lambda f:(f['observed_at'],f['recorded_at'],f['memory_id']))]
    # All explicit safety constraints remain accessible even in RECENT_ONLY.
    selected=non_restrictions+[f for f in all_facts if f['polarity']=='FORBID']
    memory=[public_fact(f,representation=='O3') for f in selected]
    pending_ids={f['memory_id'] for f in pending}
    observation=dict(representation=representation,representation_name=REPRESENTATIONS[representation],
        query=session['query'],query_at=session['query_at'],current_at=session['at'],
        target_food=session['target_food'],context=deepcopy(session.get('context',{})),
        session_index=index,remaining_clarifications=max(0,1-question_count) if index==1 else 0,
        memory=memory,incoming_evidence=[f for f in memory if f['memory_id'] in pending_ids],
        # Question bookkeeping is distinct from a persisted preference fact.
        # Raw reply text stays in the SQLite archive; its semantic evidence is
        # exposed through pending/selected memories. Otherwise NOOP would still
        # leak an omitted clarification back through this metadata side channel.
        clarification_history=[{k:deepcopy(row[k]) for k in ('seq','question_key','at','status') if k in row}
            for row in store.clarification_history(session['at'])])
    if representation=='O3':
        observation['enriched']=interpret_visible(observation)
    store.complete_trace(retrieval.trace_id,exposed_memory_ids=[f['memory_id'] for f in memory],
        pending_evidence_ids=[f['evidence_id'] for f in pending],representation=representation)
    return observation,retrieval.trace_id
