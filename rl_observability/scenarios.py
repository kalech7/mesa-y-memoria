"""Five controlled families. Gold and scripted future replies stay private.

Splits use disjoint users, dates and surface templates. The five abstract mechanisms
are deliberately shared: this is lexical/profile generalization, not held-out-family
or independent abstract-template generalization.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import random

FAMILIES = ('EXPLICIT_CHANGE', 'TEMPORARY_EXCEPTION', 'RELEVANT_RESOLVABLE',
            'IRRELEVANT_CONFLICT', 'RELEVANT_UNRESOLVABLE')
FOODS = ('arroz', 'sopa', 'pizza', 'ensalada', 'pollo', 'lentejas')
TEMPLATES = {
 'train': {'like':'Me gusta {food}.', 'dislike':'No me gusta {food}.',
           'change':'Desde ahora ya no me gusta {food}.',
           'temporary':'Solo hoy no me gusta {food}.',
           'order':'He pedido {food}.',
           'query':'¿Me gusta {food} para esta comida?',
           'reply':'Confirmo que desde ahora no me gusta {food}.'},
 'validation': {'like':'En general, me gusta {food}.', 'dislike':'En general, no me gusta {food}.',
           'change':'A partir de ahora ya no me gusta {food}.',
           'temporary':'Por hoy no me gusta {food}.',
           'order':'Hoy he pedido {food}.',
           'query':'Según mis gustos actuales, ¿me gusta {food}?',
           'reply':'A partir de ahora no me gusta {food}.'},
 'test': {'like':'Habitualmente me gusta {food}.', 'dislike':'Habitualmente no me gusta {food}.',
           'change':'He cambiado de opinión: desde ahora ya no me gusta {food}.',
           'temporary':'Solamente hoy no me gusta {food}.',
           'order':'En esta ocasión he pedido {food}.',
           'query':'Para la fecha indicada, ¿me gusta {food}?',
           'reply':'He cambiado de opinión: ahora no me gusta {food}.'},
}

def generate_scenarios(n_users=100, seed=42, split='train'):
    if split not in TEMPLATES:
        raise ValueError('Unknown split')
    if n_users < 1:
        raise ValueError('At least one user is required')
    rng = random.Random(seed)
    result = []
    for i in range(n_users):
        family = FAMILIES[i % len(FAMILIES)]
        uid = hashlib.sha256(f'observability:{split}:{seed}:{i}'.encode()).hexdigest()[:18]
        base = datetime({'train':2022,'validation':2024,'test':2026}[split],1,1,12,tzinfo=timezone.utc)+timedelta(days=rng.randrange(150))
        dates = [(base+timedelta(days=d)).isoformat().replace('+00:00','Z') for d in (0,15+rng.randrange(5),45+rng.randrange(8))]
        food, other = rng.sample(FOODS,2)
        templates = TEMPLATES[split]
        facts=[]
        def fact(food,polarity='LIKE',stage=0,kind='STABLE_PREFERENCE',text=None,source='USER_STATEMENT',valid_to=None):
            j=len(facts)
            value=dict(memory_id=f'm{j}',user_id=uid,food=food,polarity=polarity,
                scope={'day':'ANY','meal':'ANY','company':'ANY','place':'ANY'},
                kind=kind,observed_at=dates[stage],recorded_at=dates[stage],
                valid_from=dates[stage],valid_to=valid_to,source=source,evidence_id=f'e{j}',
                status='ACTIVE',supersedes=[],uncertainty='KNOWN',
                text=text or templates['like' if polarity=='LIKE' else 'dislike'].format(food=food))
            facts.append(value)
            return deepcopy(value)
        initial_polarity = 'DISLIKE' if family.startswith('RELEVANT_') else 'LIKE'
        initial=[fact(food,initial_polarity)]
        # A permanent explicit safety constraint is visible in every representation.
        initial.append(fact('maní','FORBID',kind='RESTRICTION',text='Tengo alergia al maní. No lo recomiendes.'))
        if family=='EXPLICIT_CHANGE':
            incoming=[fact(food,'DISLIKE',1,'EXPLICIT_CHANGE',templates['change'].format(food=food))]
            s2,s3='DISLIKE','DISLIKE'
        elif family=='TEMPORARY_EXCEPTION':
            end=(datetime.fromisoformat(dates[1].replace('Z','+00:00'))+timedelta(days=1)).replace(hour=0).isoformat().replace('+00:00','Z')
            incoming=[fact(food,'DISLIKE',1,'TEMPORARY_EXCEPTION',templates['temporary'].format(food=food),valid_to=end)]
            s2,s3='DISLIKE','LIKE'
        elif family=='IRRELEVANT_CONFLICT':
            incoming=[fact(other,'LIKE',1),fact(other,'DISLIKE',1,'AMBIGUOUS')]
            s2,s3='LIKE','LIKE'
        else:
            incoming=[fact(food,'LIKE',1,'AMBIGUOUS',templates['like'].format(food=food))]
            s2=s3='LIKE' if family=='RELEVANT_RESOLVABLE' else None
        # Opposite direct statements without an explicit change marker remain
        # unresolved until a clarification is actually communicated.
        response=None
        if family=='RELEVANT_RESOLVABLE':
            reply_text=templates['reply'].format(food=food).replace('no me gusta','me gusta')
            response=dict(text=reply_text,status='RESOLVED',
                fact=fact(food,'LIKE',1,'CLARIFICATION',reply_text))
        elif family=='RELEVANT_UNRESOLVABLE':
            response=dict(text='No lo sé; todavía no puedo aclarar mi preferencia.',status='UNRESOLVED',fact=None)
        sessions=[]
        for stage,inc,truth in ((0,initial,initial_polarity),(1,incoming,s2),(2,[],s3)):
            sessions.append(dict(at=dates[stage],query_at=dates[stage],
                query=templates['query'].format(food=food),target_food=food,context={},
                incoming=inc,events=[dict(message_id=f['evidence_id'],text=f['text'],at=f['observed_at'],
                    source=f['source']) for f in inc],
                # The response is private until the corresponding action is taken.
                responses={'preference':deepcopy(response)} if stage==1 and response else {},
                gold=dict(polarity=truth,needs_clarification=stage==1 and response is not None,
                    unresolved=truth is None, relevant_memory_ids=[initial[0]['memory_id']]+[
                        f['memory_id'] for f in incoming if f['food']==food],
                    conflict=family.startswith('RELEVANT_') and stage>0,
                    temporal=family in ('EXPLICIT_CHANGE','TEMPORARY_EXCEPTION'))))
        result.append(dict(episode_id=f'{split}-{uid}',user_id=uid,family=family,split=split,
                           template_id=f'{split}:{family}',sessions=sessions))
    return result

def generate_splits(train_n=100, validation_n=50, test_n=100, seed=42):
    return {name:generate_scenarios(n, seed+offset, name)
            for name,n,offset in (('train',train_n,0),('validation',validation_n,10000),('test',test_n,20000))}
