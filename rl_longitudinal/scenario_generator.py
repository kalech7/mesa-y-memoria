"""Seeded longitudinal vignettes; private labels never enter policy observations.

Expected decisions are authored here rather than computed by the reconciler.
These are controlled synthetic cases, not validated models of human diets.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import random

FOODS = ('ensalada', 'pizza', 'sopa', 'arroz', 'pollo', 'sushi', 'ramen', 'maní')
FAMILIES = ('STABLE_EXCEPTION', 'EXPLICIT_CHANGE', 'CPC_DAY', 'CPC_MEAL',
            'SOURCE_CHANGE', 'AMBIGUOUS_BEHAVIOR', 'RESTRICTION', 'UNANSWERED',
            'IRRELEVANT_CONFLICT', 'LATE_RECORDED_HISTORY')


def generate_episodes(n=200, seed=42, split='train', distribution='standard', families=None):
    if split not in ('train', 'validation', 'test', 'shift'):
        raise ValueError('Unknown split')
    rng = random.Random(seed)
    result = []
    year = {'train': 2023, 'validation': 2024, 'test': 2025, 'shift': 2026}[split]
    selected = tuple(families or (FAMILIES[:8] if split in ('train', 'validation') else FAMILIES[8:] if split == 'shift' else FAMILIES))
    if not selected or set(selected) - set(FAMILIES):
        raise ValueError('Unknown family')
    for index in range(n):
        family = selected[index % len(selected)]
        # Split-specific dates, users and temporal spacing. No IDs are observed.
        base = datetime(year, 1, 1, 12, tzinfo=timezone.utc) + timedelta(days=rng.randrange(20))
        gap = {'train': 30, 'validation': 37, 'test': 43, 'shift': 59}[split]
        dates = [(base + timedelta(days=gap * k)).isoformat().replace('+00:00', 'Z') for k in range(7)]
        uid = hashlib.sha256(f'{split}:{seed}:{index}'.encode()).hexdigest()[:18]
        facts = []

        def fact(food, polarity='LIKE', stage=0, kind='STABLE_PREFERENCE', scope=None,
                 source='USER_STATEMENT', valid_from=None, valid_to=None,
                 supersedes=(), observed_at=None):
            pos = len(facts)
            f = dict(memory_id=f'm{pos:03d}', user_id=uid, food=food, polarity=polarity,
                     scope=scope or {'day': 'ANY', 'meal': 'ANY', 'company': 'ANY', 'place': 'ANY'},
                     kind=kind, observed_at=observed_at or dates[stage], recorded_at=dates[stage],
                     valid_from=valid_from or observed_at or dates[stage], valid_to=valid_to,
                     source=source, evidence_id=f'e{pos:03d}', status='ACTIVE',
                     supersedes=list(supersedes), uncertainty='KNOWN')
            facts.append(f)
            return f

        sessions = []

        def session(stage, incoming=(), foods=(), context=None, target=None, responses=None,
                    acceptable=None, query_at=None, query=None, need=False, unresolved=False):
            ses = dict(at=dates[stage], query_at=query_at or dates[stage],
                       query=query or ('¿Qué me recomiendas?' if not target else f'¿Puedo recomendar {target} según mis gustos?'),
                       context=context or {}, target_food=target, user_available=True,
                       incoming=deepcopy(list(incoming)), responses=deepcopy(responses or {}),
                       gold=dict(foods=list(foods), acceptable_conversations=acceptable or (['ANSWER'] if foods else ['PRESERVE_CONFLICT', 'CONDITIONAL_ANSWER', 'ABSTAIN']),
                                 needs_clarification=need, unresolved=unresolved,
                                 clarification_keys=list((responses or {}).keys()) if need else [],
                                 restrictions=['maní'] if family == 'RESTRICTION' else []))
            sessions.append(ses)
            return ses

        def reply(key, food=None, stage=2, status='RESOLVED', value=None, supersedes=()):
            f = fact(food, stage=stage, kind='EXPLICIT_CHANGE', supersedes=supersedes) if food else None
            return {key: dict(text=(f'Desde ahora y en general, cualquier día y en cualquier comida, sin importar la compañía ni el lugar, prefiero {food} de forma estable.' if food else 'No lo sé; mi preferencia sigue sin estar clara.'),
                              status=status, value=value, fact=f)}

        a, b = rng.sample(('ensalada', 'pizza', 'sopa', 'arroz'), 2)
        if family == 'STABLE_EXCEPTION':
            a, b = 'ensalada', 'pollo'
            f0 = fact(a)
            session(0, [f0], [a])
            expiry = (datetime.fromisoformat(dates[1].replace('Z', '+00:00')) + timedelta(days=1)).isoformat().replace('+00:00', 'Z')
            f1 = fact(b, stage=1, kind='TEMPORARY_EXCEPTION', valid_to=expiry)
            session(1, [f1], [b], target=b, query='¿Qué elegí comer solamente en esta ocasión?')
            session(3, foods=[a], query='¿Qué me recomiendas normalmente?')
            session(4, foods=[b], target=b, query_at=dates[1], query='¿Qué elegí en aquella ocasión?')
        elif family == 'EXPLICIT_CHANGE':
            a, b = 'sushi', 'ramen'
            f0 = fact(a)
            session(0, [f0], [a])
            f1 = fact(a, 'DISLIKE', 2, 'EXPLICIT_CHANGE', supersedes=[f0['memory_id']])
            f2 = fact(b, stage=2, kind='EXPLICIT_CHANGE')
            session(2, [f1, f2], [b])
            session(3, foods=[b])
            session(4, foods=[a], query_at=dates[0])
        elif family in ('CPC_DAY', 'CPC_MEAL'):
            slot = 'day' if family == 'CPC_DAY' else 'meal'
            first, second = ('WEEKDAY', 'WEEKEND') if slot == 'day' else ('BREAKFAST', 'DINNER')
            s0 = {'day': 'ANY', 'meal': 'ANY', 'company': 'ANY', 'place': 'ANY', slot: first}
            s1 = {**s0, slot: second}
            f0, f1 = fact(a, scope=s0), fact(b, stage=1, scope=s1)
            session(0, [f0], [a], context={slot: first})
            session(1, [f1], [b], context={slot: second})
            phrase = {'WEEKEND':'el fin de semana','WEEKDAY':'un día laboral','BREAKFAST':'el desayuno','DINNER':'la cena'}[second]
            response = {slot: dict(text=f'Es para {phrase}.', status='RESOLVED', value=second, fact=None)}
            session(2, foods=[b], responses=response, need=True,
                    query='Para una próxima ocasión todavía sin fechar, ¿qué me recomiendas?' if slot=='day' else 'Sin haber elegido aún el tipo de comida, ¿qué me recomiendas?')
            session(3, foods=[b], context={slot: second})
            session(4, foods=[a], context={slot: first})
        elif family in ('SOURCE_CHANGE', 'AMBIGUOUS_BEHAVIOR'):
            f0, neg = fact(a), fact(b, 'DISLIKE')
            session(0, [f0, neg], [a])
            order = fact(b, stage=1, kind='OBSERVED_BEHAVIOR', source='ORDER_HISTORY')
            status = 'RESOLVED' if family == 'SOURCE_CHANGE' else 'UNRESOLVED'
            responses = reply('current:' + b, b if status == 'RESOLVED' else None, stage=1,
                              status=status, supersedes=[neg['memory_id']])
            session(1, [order], [b] if status == 'RESOLVED' else [], target=b,
                    responses=responses, need=True, unresolved=status != 'RESOLVED')
            session(3, foods=[b] if status == 'RESOLVED' else [], target=b, unresolved=status != 'RESOLVED')
            session(4, foods=[a, b] if status == 'RESOLVED' else [a])
        elif family == 'RESTRICTION':
            f0 = fact(a)
            restriction = fact('maní', 'DISLIKE', kind='RESTRICTION', source='USER_STATEMENT')
            session(0, [f0, restriction], [a])
            recent = fact('maní', stage=2, kind='AMBIGUOUS')
            response = {'restriction:maní': dict(text='Mi alergia al maní continúa; no lo recomiendes.', status='RESOLVED', value='RESTRICTED', fact=None)}
            session(2, [recent], target='maní', responses=response, need=True, unresolved=True)
            session(3, foods=[a])
            session(4, target='maní', unresolved=True)
        elif family == 'UNANSWERED':
            pos = fact(a)
            session(0, [pos], [a])
            neg = fact(a, 'DISLIKE', stage=2, kind='AMBIGUOUS')
            response = reply('current:' + a, stage=2, status='UNRESOLVED')
            session(2, [neg], target=a, responses=response, need=True, unresolved=True)
            session(3, target=a, unresolved=True)
            session(4, target=a, unresolved=True)
        elif family == 'IRRELEVANT_CONFLICT':
            clean = fact(a)
            session(0, [clean], [a])
            p, q = fact(b, stage=1), fact(b, 'DISLIKE', 1, 'AMBIGUOUS')
            session(1, [p, q], [a])
            session(3, foods=[a])
            session(4, foods=[a])
        else:
            # Evidence is registered late but explicitly describes an old taste.
            old = fact(a, stage=2, observed_at=dates[0], valid_to=dates[1])
            new = fact(b, stage=2, kind='EXPLICIT_CHANGE', valid_from=dates[2])
            session(2, [old, new], [b])
            session(3, foods=[a], query_at=dates[0])
            session(4, foods=[b])
            session(5, foods=[a], query_at=dates[0])

        # Public utterances have split-held wording; labels are not interpolated.
        for ses in sessions:
            ses['events'] = [_render_fact(f, split) for f in ses['incoming']]
            if distribution == 'shift':
                ses['query'] = 'Recordando lo que te había contado, ' + ses['query'].lower()
            # This annotation is authored independently of retrieval/resolution:
            # old facts remain relevant history even when no longer applicable.
            ses['gold']['relevant_memory_ids'] = [f['memory_id'] for f in facts
                if f in [z for ss in sessions for z in ss['incoming']]
                and f['recorded_at'] <= ses['at']
                and (ses.get('target_food') is None or f['food'] == ses['target_food'] or f['kind'] == 'RESTRICTION')]
        result.append(dict(episode_id=uid, user_id=uid, seed=seed, split=split,
                           family=family, temporal_family=f'{split}:gap{gap}', sessions=sessions))
    return result


def _render_fact(f, split):
    food = f['food']
    prefix = {'train': 'Me gusta', 'validation': 'Prefiero', 'test': 'Disfruto comer', 'shift': 'Me quedo con'}[split]
    if f['kind'] == 'RESTRICTION':
        text = f'Soy alérgico a {food}.'
    elif f['source'] == 'ORDER_HISTORY':
        text = f'He pedido {food} últimamente.'
    elif f['kind'] == 'TEMPORARY_EXCEPTION':
        text = f'Solo hoy prefiero {food}; es una excepción.'
    elif f['polarity'] == 'DISLIKE':
        text = f'Ya no me gusta {food}.' if f['kind'] == 'EXPLICIT_CHANGE' else f'No me gusta {food}.'
    else:
        text = f'{prefix} {food}.'
        if f['kind'] == 'EXPLICIT_CHANGE':
            text = f'Desde ahora prefiero {food}.'
    scope = f['scope']
    day = {'ANY': 'Cualquier día', 'WEEKDAY': 'Los días laborales', 'WEEKEND': 'Los fines de semana'}.get(scope.get('day'), '')
    meal = {'ANY': 'en cualquier comida', 'BREAKFAST': 'en el desayuno', 'LUNCH': 'en el almuerzo', 'DINNER': 'en la cena', 'SNACK': 'en la merienda'}.get(scope.get('meal'), '')
    all_other = 'sin importar la compañía ni el lugar' if scope.get('company') == scope.get('place') == 'ANY' else ''
    return dict(message_id=f['evidence_id'], text=f'{day}, {meal}, {all_other}: {text}', source=f['source'],
                at=f['observed_at'], valid_from=f['valid_from'], valid_to=f['valid_to'],
                supersedes=f['supersedes'])


generate_scenarios = generate_episodes
