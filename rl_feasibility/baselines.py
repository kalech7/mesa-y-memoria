"""Policies operating only on the explicitly public observation.

No policy receives an environment, episode identifier, seed, latent state, gold
label or future response. The strong reference is a transparent information-seeking
rule, not an omniscient upper bound and not an optimal POMDP planner.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

FOODS = ('ensalada', 'pizza', 'sopa', 'arroz')
SLOTS = ('day', 'meal', 'company', 'place', 'current_preference')
SLOT_ACTIONS = {slot: 'CLARIFY_' + slot.upper() for slot in SLOTS}


def _support(observation):
    from rl_feasibility.environment import public_support
    return public_support(observation)


def _asked(observation):
    return {str(slot).lower().replace('clarify_', '') for slot in observation.get('asked_slots', [])}


def _available(observation, slots):
    if observation.get('remaining_questions', 0) <= 0 or observation.get('remaining_steps', 0) <= 1:
        return []
    return [slot for slot in slots if slot not in _asked(observation)]


def canonical_foods(observation):
    """Food renaming equivariance; record IDs, users and seeds are ignored."""
    ordered = []
    for preference in observation.get('preferences', []):
        food = preference.get('food_item')
        if food in FOODS and food not in ordered:
            ordered.append(food)
    return ordered + [food for food in FOODS if food not in ordered]


def _canon_action(action, observation):
    if action.startswith('ANSWER:'):
        return 'ANSWER@' + str(canonical_foods(observation).index(action.split(':', 1)[1]))
    return action


def _real_action(action, observation):
    if action.startswith('ANSWER@'):
        return 'ANSWER:' + canonical_foods(observation)[int(action.split('@')[1])]
    return action


def feature_key(observation):
    """Finite observable history abstraction used by BOTH SFT and Q-learning.

    It includes remaining horizon, asked slots, uncertainty and public applicability;
    it excludes free wording, evidence identifiers, user IDs, seed and category.
    This is a belief-relevant compression, not a proof of a sufficient statistic.
    """
    support = _support(observation)
    foods = canonical_foods(observation)
    def indices(name):
        return tuple(sorted(foods.index(f) for f in support.get(name, []) if f in foods))
    prefs = observation.get('preferences', [])
    response_uncertain = any(any(word in str(m.get('text', '')).lower()
                                  for word in ('no sé', 'no se', 'incierto', 'no puedo', 'no estoy', 'no cambia'))
                             for m in observation.get('messages', []) if m.get('role') == 'user')
    return json.dumps([
        indices('supported_foods'), indices('possible_foods'), indices('safe_foods'),
        sorted(support.get('missing_slots', [])), bool(support.get('conflict')),
        bool(support.get('unresolved')), sorted(_asked(observation)),
        int(observation.get('remaining_questions', 0)), int(observation.get('remaining_steps', 0)),
        sorted({p.get('source', 'UNKNOWN') for p in prefs}),
        any(p.get('stability') in ('oscillating', 'uncertain', 'OSCILLATING', 'UNCERTAIN') for p in prefs),
        response_uncertain,
    ], ensure_ascii=False, separators=(',', ':'))


def action_catalog(observation):
    actions = ['ANSWER@' + str(i) for i in range(len(FOODS))]
    actions += [SLOT_ACTIONS[s] for s in _available(observation, SLOTS)]
    return actions + ['PRESERVE_CONFLICT', 'ABSTAIN']


def _final_action(observation):
    support = _support(observation)
    supported = [f for f in canonical_foods(observation)
                 if f in support.get('supported_foods', []) and f in support.get('safe_foods', [])]
    # Open recommendation: another food's conflict does not disqualify a clean
    # and applicable option. No exclusivity is inferred from a LIKE statement.
    if supported:
        return 'ANSWER:' + supported[0]
    if support.get('conflict') or support.get('unresolved'):
        return 'PRESERVE_CONFLICT'
    return 'ABSTAIN'


class NeverAsk:
    name = 'NEVER_ASK'
    def select_action(self, observation):
        return _final_action(observation)


class AlwaysAsk:
    name = 'ALWAYS_ASK'
    def select_action(self, observation):
        context = observation.get('context', {})
        absent = [s for s in SLOTS if not context.get(s)]
        available = _available(observation, absent)
        return SLOT_ACTIONS[available[0]] if available else _final_action(observation)


class StrongReference:
    """Visible-only, selective questioning using communicated scope partitions.

    It asks only a missing scope that can distinguish public alternatives, or asks
    current preference once for conflicting evidence. It cannot know whether the
    synthetic user will answer. It stops after an unsuccessful current-preference
    response, preserving uncertainty. This is a strong reproducible reference,
    with no optimality guarantee or privileged model of the user's response.
    """
    name = 'STRONG_REFERENCE'
    def select_action(self, observation):
        support = _support(observation)
        final = _final_action(observation)
        if final.startswith('ANSWER:'):
            return final
        missing = support.get('missing_slots', [])
        useful = [s for s in SLOTS[:-1] if s in missing]
        if support.get('conflict') or support.get('unresolved'):
            useful.append('current_preference')
        available = _available(observation, useful)
        if available:
            return SLOT_ACTIONS[available[0]]
        return _final_action(observation)


class CurrentHeuristic:
    """Adapter to the historical deterministic resolver, without private labels."""
    name = 'CURRENT_HEURISTIC'
    def select_action(self, observation):
        from simulator.resolution import resolve_preferences
        ctx = observation.get('context', {})
        parts = [observation.get('query', '')]
        words = {'WEEKDAY':'día laboral', 'WEEKEND':'fin de semana', 'BREAKFAST':'desayuno',
                 'LUNCH':'almuerzo', 'DINNER':'cena', 'SNACK':'merienda'}
        for key in ('day', 'meal'):
            if ctx.get(key) in words:
                parts.append(words[ctx[key]])
        if ctx.get('company'):
            parts.append({'alone': 'como solo', 'friends': 'con amigos', 'family': 'con familia'}.get(
                ctx['company'], 'con ' + str(ctx['company'])))
        if ctx.get('place'):
            parts.append({'home': 'en casa', 'work': 'en la oficina'}.get(
                ctx['place'], 'en ' + str(ctx['place'])))
        result = resolve_preferences(observation.get('preferences', []), observation.get('restrictions', []),
                                     ' '.join(parts), reference_time=observation.get('reference_time'))
        selected = result['selected_action']
        foods = [f for f in result.get('selected_food_items', []) if f in FOODS]
        safe = _support(observation).get('safe_foods', [])
        if selected == 'ANSWER' and foods and foods[0] in safe:
            return 'ANSWER:' + foods[0]
        if selected == 'CLARIFY':
            # The current resolver emits a natural-language question; map it to
            # the corresponding semantic environment action, retaining its order.
            question = str(result.get('clarification_question', '')).lower()
            slots = []
            for slot, cues in [('day', ('día', 'dia', 'semana')), ('meal', ('comida', 'desayuno', 'cena', 'almuerzo')),
                               ('company', ('compañ', 'compan', 'quién', 'quien')), ('place', ('lugar', 'dónde', 'donde'))]:
                if any(cue in question for cue in cues):
                    slots.append(slot)
            available = _available(observation, slots)
            if available:
                return SLOT_ACTIONS[available[0]]
        if selected in ('PRESERVE_CONFLICT', 'VERIFY'):
            return 'PRESERVE_CONFLICT'
        return _final_action(observation)


class SupervisedPolicy:
    """Maximum-likelihood tabular behavior cloning of visible-only reference.

    Same observation abstraction/action catalogue as Q-learning. Training uses
    only public observations and teacher actions, never rewards or private labels.
    Unsupported states fall back to the auditable current heuristic (reported).
    """
    name = 'SUPERVISED_POLICY'
    def __init__(self, counts=None):
        self.counts = counts or {}
        self.fallback = CurrentHeuristic()

    def select_action(self, observation):
        counts = self.counts.get(feature_key(observation), {})
        eligible = [a for a in action_catalog(observation) if a in counts]
        if not eligible:
            return self.fallback.select_action(observation)
        selected = max(eligible, key=lambda a: (counts[a], -action_catalog(observation).index(a)))
        return _real_action(selected, observation)

    def to_dict(self):
        return {'policy_type':'tabular_behavior_cloning', 'counts':self.counts,
                'fallback':'CURRENT_HEURISTIC', 'feature_version':1}

    @classmethod
    def from_dict(cls, value):
        return cls(value['counts'])


def train_supervised(scenarios, condition='ORACLE_STRUCTURED', output_dir=None, teacher=None):
    from rl_feasibility.environment import FoodPreferenceEnv
    teacher = teacher or StrongReference()
    counts = defaultdict(Counter)
    examples = 0
    for scenario in scenarios:
        env = FoodPreferenceEnv(scenario, condition=condition)
        observation, _ = env.reset()
        while True:
            action = teacher.select_action(observation)
            counts[feature_key(observation)][_canon_action(action, observation)] += 1
            examples += 1
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    policy = SupervisedPolicy({key:dict(value) for key,value in counts.items()})
    result = {'policy':policy, 'examples':examples, 'unique_observation_keys':len(counts),
              'teacher':teacher.name, 'private_or_reward_labels_used':False}
    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path/'supervised_policy.json').write_text(json.dumps(policy.to_dict(), ensure_ascii=False, indent=2))
        (path/'supervised_training.json').write_text(json.dumps({k:v for k,v in result.items() if k!='policy'}, indent=2))
    return result


def make_baselines(supervised_policy=None):
    policies = [NeverAsk(), AlwaysAsk(), CurrentHeuristic(), StrongReference()]
    if supervised_policy is not None:
        policies.append(supervised_policy)
    return {policy.name:policy for policy in policies}
