"""Shared public-only feature encoding; no scenario or evaluator imports.

This fixed symbolic interpreter is available to every small-policy comparator.
Its capability is not attributed to reinforcement learning. O3 caches public
interpretations; O2 permits the same computations from its visible history.
"""
from copy import deepcopy
from rl_observability.policies import public_analysis
from rl_observability.observations import evidence_kind

MEMORY_OPERATIONS = ('ADD_MEMORY', 'UPDATE_WITH_HISTORY', 'PRESERVE_CONFLICT', 'NOOP')
QUESTION_SLOTS = ('preference', 'meal', 'day', 'source')
ACTIONS = tuple(
    dict(conversation=conversation, memory_operation=operation, polarity=polarity, slot=slot)
    for operation in MEMORY_OPERATIONS
    for conversation, polarity, slot in (
        ('ANSWER','LIKE',None), ('ANSWER','DISLIKE',None),
        ('CONDITIONAL_ANSWER',None,None), ('PRESERVE_CONFLICT',None,None),
        *(('CLARIFY',None,slot) for slot in QUESTION_SLOTS)))

def action_index(action):
    value = {k: action.get(k) for k in ('conversation','memory_operation','polarity','slot')}
    if value['memory_operation'] == 'ADD':
        value['memory_operation'] = 'ADD_MEMORY'
    return ACTIONS.index(value)

def action_mask(observation):
    return [a['conversation'] != 'CLARIFY' or observation.get('remaining_clarifications',0)>0
            for a in ACTIONS]

# Feature names describe evidence, never an expected action. Absolute dates,
# identifiers, scenario families, split names and private labels are omitted.
FEATURE_NAMES = (
    'session_two','session_three','clarification_budget',
    'active_like','active_dislike','active_restriction','active_ambiguous',
    'visible_facts','incoming_facts','relevant_facts',
    'visible_like','visible_dislike','visible_restriction',
    'current_facts','expired_facts','future_facts',
    'stable_facts','explicit_change_facts','temporary_facts','clarification_facts',
    'behavior_facts','source_user','source_clarification','source_order',
    'incoming_like','incoming_dislike','incoming_explicit_change','incoming_temporary',
    'incoming_clarification','incoming_supersession','history_supersession',
    'potential_disagreement','safety_restriction','missing_day','missing_meal',
    'missing_company','missing_place','resolved_clarification','unresolved_clarification',
    'asked_preference','asked_meal','asked_day','asked_source',
    'known_day','known_meal','known_company','known_place',
    'scope_day','scope_meal','scope_company','scope_place')
FEATURE_DIM = len(FEATURE_NAMES)
FEATURE_VERSION = 'SPEC-RL-005-public-symbolic-v1'

def feature_vector(observation):
    """Bounded numeric evidence features, identical for M0/M1/M2.

    O3 does not confer privileged semantic computations: its supplied enrichment
    is reproducible by the shared fixed interpreter from O2's public fields.
    """
    analysis = public_analysis(observation)
    encoded = analysis['encoded_facts']
    facts = observation.get('memory',[])
    incoming = observation.get('incoming_evidence',[])
    history = observation.get('clarification_history',[])
    values = dict.fromkeys(FEATURE_NAMES,0.0)
    def count(name, n): values[name] = min(float(n),4.0)/4.0
    values['session_two'] = float(observation.get('session_index') == 1)
    values['session_three'] = float(observation.get('session_index') == 2)
    values['clarification_budget'] = float(observation.get('remaining_clarifications',0)>0)
    for polarity in ('LIKE','DISLIKE'):
        count('active_'+polarity.lower(),sum(f.get('polarity')==polarity for f in analysis['active']))
        count('visible_'+polarity.lower(),sum(f.get('polarity')==polarity for f in facts))
        count('incoming_'+polarity.lower(),sum(f.get('polarity')==polarity for f in incoming))
    count('active_restriction',sum(f.get('polarity')=='FORBID' for f in analysis['active']))
    count('active_ambiguous',sum(evidence_kind(f) in ('OBSERVED_BEHAVIOR','UNRESOLVED','AMBIGUOUS') for f in analysis['active']))
    count('visible_restriction',sum(f.get('polarity')=='FORBID' for f in facts))
    count('visible_facts',len(facts)); count('incoming_facts',len(incoming))
    count('relevant_facts',sum(e[0] for e in encoded))
    for label, temporal in (('current','CURRENT'),('expired','ENDED'),('future','FUTURE')):
        count(label+'_facts',sum(e[4]==temporal for e in encoded))
    for label, kind in (('stable','STABLE_PREFERENCE'),('explicit_change','EXPLICIT_CHANGE'),
                        ('temporary','TEMPORARY_EXCEPTION'),('clarification','CLARIFICATION'),
                        ('behavior','OBSERVED_BEHAVIOR')):
        count(label+'_facts',sum(evidence_kind(f)==kind for f in facts))
        if 'incoming_'+label in values:
            count('incoming_'+label,sum(evidence_kind(f)==kind for f in incoming))
    for label,source in (('user','USER_STATEMENT'),('clarification','CLARIFICATION'),('order','ORDER_HISTORY')):
        count('source_'+label,sum(f.get('source')==source for f in facts))
    count('incoming_supersession',sum(bool(f.get('supersedes')) for f in incoming))
    count('history_supersession',sum(bool(f.get('supersedes')) for f in facts))
    values['potential_disagreement']=float(analysis['conflict'])
    values['safety_restriction']=float(analysis['blocked'])
    for slot in ('day','meal','company','place'):
        values['missing_'+slot]=float(slot in analysis['missing'])
        values['known_'+slot]=float(observation.get('context',{}).get(slot) not in (None,'ANY'))
        count('scope_'+slot,sum((f.get('scope') or {}).get(slot) not in (None,'ANY') for f in facts))
    values['resolved_clarification']=float(any(h.get('status')=='RESOLVED' for h in history))
    values['unresolved_clarification']=float(any(h.get('status')=='UNRESOLVED' for h in history))
    for slot in QUESTION_SLOTS:
        values['asked_'+slot]=float(any(h.get('question_key')==slot for h in history))
    return [values[name] for name in FEATURE_NAMES]
