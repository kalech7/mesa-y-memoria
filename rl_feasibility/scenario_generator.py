"""Author-defined latent food users; never imports resolver or reward labels.

The generator authors preference constraints and latent situations. Agent-visible
records and private replies are separate. Template families are split-specific;
native semantic state hashes are split into disjoint training/validation/test
buckets. Structural categories are shared deliberately, so this tests policy transfer,
not general-purpose natural-language or human preference generalization.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import itertools
import json
import random

from simulator.preferences import Preference, Restriction

FOODS = ('ensalada', 'pizza', 'sopa', 'arroz')
CATEGORIES = ('CLEAR', 'CPC_DAY', 'CPC_MEAL', 'CPC_COMPANY', 'CPC_PLACE',
              'CPC_TWO_SLOT', 'BOC_RESOLVABLE', 'BOC_UNRESOLVABLE',
              'SCC_RESOLVABLE', 'SCC_UNRESOLVABLE', 'SAME_SCOPE', 'RESTRICTION')
TIME = '2026-10-19T12:00:00Z'
QUERIES = {
    'train': ('¿Qué debería comer?', 'Recomiéndame algo para comer.'),
    'validation': ('Quiero elegir mi comida; ¿qué sugieres?', 'Ayúdame a decidir qué comer.'),
    'test': ('¿Qué opción alimentaria se ajusta a lo que sabes de mí?',
             'Con mis gustos conocidos, ¿qué elegirías para mi comida?'),
    'shift': ('Necesito una propuesta de comida fundamentada.',),
}
STYLE = {'train': 'En general me gusta {food}.',
         'validation': 'En general prefiero {food}.',
         'test': 'En general disfruto comer {food}.',
         'shift': 'Mi gusto general se inclina por {food}.'}
SLOT_TEXT = {
    'day': {'WEEKDAY': 'Es para un día laboral.', 'WEEKEND': 'Es para un fin de semana.'},
    'meal': {'BREAKFAST': 'Es para el desayuno.', 'LUNCH': 'Es para el almuerzo.',
             'DINNER': 'Es para la cena.', 'SNACK': 'Es para la merienda.'},
    'company': {'alone': 'Voy a comer solo.', 'friends': 'Voy a comer con amigos.'},
    'place': {'home': 'Voy a comer en casa.', 'work': 'Voy a comer en la oficina.'},
}


def _pref(index, food, polarity='LIKE', **kw):
    return asdict(Preference(preference_id=f'p{index}', user_id='user', food_item=food,
                            polarity=polarity, day_scope='ANY', meal_scope='ANY',
                            evidence_id=f'e{index}', timestamp=TIME, **kw))


def _preference(index, food, **kw):
    # Explicit overrides of the two scopes avoid pretending unknown means ANY.
    p = _pref(index, food)
    p.update(kw)
    return p


def _text(p, split):
    """Render exactly the authored scopes; omission never stands for ANY.

    ORDER_HISTORY describes observed ordering, while USER_STATEMENT describes a
    communicated taste. Fold-specific statement wording is a predefined language
    stressor, not an alternate latent preference or reward definition.
    """
    food, polarity = p['food_item'], p['polarity']
    parts = []
    day_phrase = {'ANY': 'Para cualquier día', 'WEEKDAY': 'Los días laborales',
                  'WEEKEND': 'Los fines de semana'}.get(p.get('day_scope'))
    if day_phrase:
        parts.append(day_phrase)
    meal_phrase = {'ANY': 'para cualquier tipo de comida',
                   'BREAKFAST': 'en el desayuno', 'LUNCH': 'en el almuerzo',
                   'DINNER': 'en la cena', 'SNACK': 'en la merienda'}.get(p.get('meal_scope'))
    if meal_phrase:
        parts.append(meal_phrase)
    for key, value in p.get('context', {}).items():
        parts.append({('company', 'alone'): 'cuando como solo',
                      ('company', 'friends'): 'cuando como con amigos',
                      ('place', 'home'): 'en casa',
                      ('place', 'work'): 'en la oficina'}.get((key, value), value))
    if p.get('stability') == 'OSCILLATING':
        parts.append('mis gustos cambian y no estoy seguro de mantenerlos')
    if p.get('source') == 'ORDER_HISTORY':
        # Ordering is evidence of behavior, never an inferred taste declaration.
        cue = 'pedí' if polarity == 'LIKE' else 'no pedí'
    elif polarity == 'DISLIKE':
        cue = 'no me gusta'
    else:
        cue = {'train': 'me gusta', 'validation': 'prefiero',
               'test': 'disfruto comer', 'shift': 'me quedo con'}[split]
    prefix = ', '.join(parts)
    return f'{prefix + ", " if prefix else ""}{cue} {food}.'


_PREFERENCE_SEMANTICS = ('food_item', 'polarity', 'day_scope', 'meal_scope',
    'context', 'valid_from', 'valid_to', 'source', 'timestamp', 'weekdays',
    'temporary', 'stability', 'scope_unknown', 'explicit_update')
_RESTRICTION_SEMANTICS = ('food_item', 'restriction_type', 'day_scope',
    'meal_scope', 'context', 'valid_from', 'valid_to', 'source', 'timestamp',
    'certainty', 'weekdays', 'scope_unknown')
SPLIT_STATE_BUCKETS = {'train': frozenset(range(6)),
                      'validation': frozenset((6, 7)),
                      'test': frozenset((8, 9)), 'shift': frozenset((8, 9))}


def semantic_fingerprint(scenario):
    """Hash an authored native state, excluding identities and surface wording.

    This is a partitioning instrument, never an agent observation or feature.
    Full latent context is included; distinct native states may still share the
    same decision-relevant abstraction (especially the simple CLEAR category).
    Test and distribution-shift sets intentionally share the held-out buckets.
    """
    def canonical(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    def records(items, keys):
        # Reordering or duplicating identical evidence does not create a new
        # preference state. Source and temporal evidence still remain distinct.
        return sorted({canonical({key: item.get(key) for key in keys}) for item in items})
    semantic = {'preferences': records(scenario['oracle_preferences'], _PREFERENCE_SEMANTICS),
                'restrictions': records(scenario['oracle_restrictions'], _RESTRICTION_SEMANTICS),
                'context': scenario.get('visible_context', {}),
                'latent': {'context': scenario['latent'].get('context', {}),
                           'acceptable_foods': sorted(scenario['latent']['acceptable_foods']),
                           'unresolvable': scenario['latent']['unresolvable'],
                           'current_choice': scenario['latent'].get('current_choice')}}
    return hashlib.sha256(canonical(semantic).encode()).hexdigest()


def state_bucket(scenario):
    return int(semantic_fingerprint(scenario), 16) % 10


def generate_scenarios(n=2000, seed=42, split='train', distribution='standard'):
    if split not in QUERIES:
        raise ValueError('Unknown split')
    rng = random.Random(seed)
    output = []
    # Disjoint food-to-role permutations, with uniform food marginals in every
    # split. Simple clear cases necessarily share semantics in a four-food world.
    orbits = []
    seen = set()
    for permutation in itertools.permutations(FOODS):
        if permutation in seen:
            continue
        orbit = [permutation[k:] + permutation[:k] for k in range(4)]
        orbits.append(orbit)
        seen.update(orbit)
    pool = sum(orbits[:3] if split == 'train' else orbits[3:4] if split == 'validation' else orbits[4:], [])
    for i in range(n):
        # Category assignment has no relationship with visible IDs or food identity.
        category = CATEGORIES[i % len(CATEGORIES)]
        if distribution != 'standard':
            category = rng.choices(CATEGORIES, weights=[1, 1, 1, 2, 2, 4, 2, 2, 2, 2, 2, 1])[0]
        for attempt in range(10000):
            foods = list(rng.choice(pool))
            a, b, c, d = foods
            prefs, restrictions = [], []
            context, response_map = {}, {}
            hidden_context = {'day': rng.choice(('WEEKDAY', 'WEEKEND')),
                              'meal': rng.choice(('BREAKFAST', 'LUNCH', 'DINNER', 'SNACK')),
                              'company': rng.choice(('alone', 'friends')),
                              'place': rng.choice(('home', 'work'))}
            # A user's uncommunicated situation is independent of the wording/template.
            for slot, value in hidden_context.items():
                response_map[slot] = {'value': value, 'text': SLOT_TEXT[slot][value]}
            unresolved = category in ('BOC_UNRESOLVABLE', 'SCC_UNRESOLVABLE', 'SAME_SCOPE')
            truth_foods = [a]
            current_choice = None
            if category in ('CLEAR', 'RESTRICTION'):
                prefs = [_preference(0, a)]
                if category == 'RESTRICTION':
                    prefs.append(_preference(1, b))
                    restrictions = [asdict(Restriction('r0', 'user', b, 'ALLERGY',
                                    day_scope='ANY', meal_scope='ANY', evidence_id='er0', timestamp=TIME))]
            elif category.startswith('CPC_') and category != 'CPC_TWO_SLOT':
                slot = category[4:].lower()
                values = {'day': ('WEEKDAY', 'WEEKEND'),
                          'meal': (('BREAKFAST', 'LUNCH') if split == 'train' else
                                   ('DINNER', 'SNACK') if split == 'validation' else
                                   ('LUNCH', 'DINNER')),
                          'company': ('alone', 'friends'), 'place': ('home', 'work')}[slot]
                hidden_context[slot] = rng.choice(values)
                response_map[slot] = {'value': hidden_context[slot], 'text': SLOT_TEXT[slot][hidden_context[slot]]}
                for j, (value, food) in enumerate(zip(values, (a, b))):
                    kw = {'day_scope': value} if slot == 'day' else {'meal_scope': value} if slot == 'meal' else {'context': {slot: value}}
                    prefs.append(_preference(j, food, **kw))
                truth_foods = [a if hidden_context[slot] == values[0] else b]
            elif category == 'CPC_TWO_SLOT':
                # Company plus meal: all four conjunctions are meaningful. A partial
                # answer leaves two live possibilities; two different questions matter.
                meals = ('BREAKFAST', 'LUNCH') if split == 'train' else ('DINNER', 'SNACK') if split == 'validation' else ('LUNCH', 'DINNER')
                hidden_context['meal'] = rng.choice(meals)
                response_map['meal'] = {'value': hidden_context['meal'], 'text': SLOT_TEXT['meal'][hidden_context['meal']]}
                for j, food in enumerate(foods):
                    prefs.append(_preference(j, food, meal_scope=meals[j // 2], context={'company': ('alone', 'friends')[j % 2]}))
                truth_foods = [foods[meals.index(hidden_context['meal']) * 2 + ('alone', 'friends').index(hidden_context['company'])]]
            elif category.startswith('BOC'):
                prefs = [_preference(0, a, stability='OSCILLATING', timestamp='2026-10-17T12:00:00Z'),
                         _preference(1, a, polarity='DISLIKE', stability='OSCILLATING', timestamp='2026-10-18T12:00:00Z'),
                         _preference(2, b, stability='OSCILLATING')]
                current_choice = None if unresolved else rng.choice((a, b))
                truth_foods = [] if unresolved else [current_choice]
            elif category.startswith('SCC'):
                prefs = [_preference(0, a), _preference(1, b, polarity='DISLIKE'),
                         _preference(2, b, source='ORDER_HISTORY')]
                current_choice = None if unresolved else rng.choice((a, b))
                # The independent declaration about a remains a justified open
                # recommendation despite disputed evidence about b.
                truth_foods = sorted({a} | ({current_choice} if current_choice else set()))
            else:
                prefs = [_preference(0, a), _preference(1, a, polarity='DISLIKE'), _preference(2, b)]
                truth_foods = [b]
            # Visible dates explain chronology; source is evidence provenance, never a
            # privileged reliability label. Whether user can clarify remains hidden.
            if unresolved:
                response_map['current_preference'] = {'value': None,
                    'text': 'No puedo aclarar cuál prefiero ahora; sigo sin poder decidir.'}
            elif category.startswith(('BOC', 'SCC')):
                food = current_choice
                response_map['current_preference'] = {'value': food,
                    'text': f'Para esta comida, actualmente prefiero {food}; esa es mi elección explícita ahora.'}
            else:
                response_map['current_preference'] = {'value': None,
                    'text': 'Mi preferencia depende de las condiciones que ya te he contado.'}
            events = [{'id': p['evidence_id'], 'user_id': 'user', 'text': _text(p, split),
                       'source': p['source'], 'timestamp': p['timestamp']} for p in prefs]
            for r in restrictions:
                events.append({'id': r['evidence_id'], 'user_id': 'user', 'text': f"Para cualquier día y cualquier tipo de comida, tengo alergia a {r['food_item']}.",
                               'source': 'USER_STATEMENT', 'timestamp': TIME})
            if rng.random() < .5 or distribution != 'standard':
                events.append({'id': 'ed', 'user_id': 'user', 'text': 'Mi color favorito es el azul.',
                               'source': 'USER_STATEMENT', 'timestamp': TIME})
            uid = hashlib.sha256(f'{split}/{seed}/{i}'.encode()).hexdigest()[:20]
            candidate = {'scenario_id': f'{split}-{uid}', 'user_id': uid, 'category': category,
                           'split': split, 'query': rng.choice(QUERIES[split]), 'events': events,
                           'oracle_preferences': prefs, 'oracle_restrictions': restrictions,
                           'visible_context': context, 'latent': {'context': hidden_context,
                           'acceptable_foods': truth_foods, 'unresolvable': unresolved,
                           'current_choice': current_choice},
                           'response_map': response_map, 'reference_time': TIME}
            if state_bucket(candidate) in SPLIT_STATE_BUCKETS[split]:
                output.append(candidate)
                break
        else:
            raise RuntimeError("Unable to sample a state in the assigned split buckets")
    rng.shuffle(output)
    return deepcopy(output)
