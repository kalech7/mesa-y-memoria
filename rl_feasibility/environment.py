"""Restricted POMDP adapter for communicated food evidence.

Only ``observation`` goes to policies. Scenarios and ``diagnostic_metrics`` belong
to the evaluator. This is an interface boundary, not a malicious-code sandbox.
"""
from copy import deepcopy
from dataclasses import asdict, fields
from datetime import datetime, timezone

from simulator.preferences import Preference, Restriction, extract_preferences
from rl_feasibility.reward import RewardConfig, terminal_reward
from rl_feasibility.scenario_generator import FOODS, generate_scenarios

SLOTS = ('day', 'meal', 'company', 'place', 'current_preference')
QUESTION_ACTIONS = tuple('CLARIFY_' + s.upper() for s in SLOTS)
ACTIONS = tuple('ANSWER:' + f for f in FOODS) + QUESTION_ACTIONS + ('PRESERVE_CONFLICT', 'ABSTAIN')
QUESTIONS = {'day': '¿Comerás en un día laboral o en fin de semana?',
             'meal': '¿Para qué tipo de comida es: desayuno, almuerzo, cena o merienda?',
             'company': '¿Comerás solo o con amigos?',
             'place': '¿Comerás en casa o en la oficina?',
             'current_preference': '¿Puedes aclarar tu preferencia actual para esta comida, o sigue sin estar clara?'}


_PRIVATE_NAMES = {'latent', 'hidden', 'hidden_state', 'true_state', 'truth', 'gold',
                  'expected', 'acceptable_foods', 'acceptable_actions', 'response_map',
                  'category', 'scenario_id', 'seed', 'split', 'rationale', 'user_reply'}
_RECORD_CONTEXT = {'company', 'place', 'goal', 'occasion', 'budget', 'availability'}
_OBS_CONTEXT = {'day', 'meal', 'company', 'place', 'current_preference', 'current_uncertain'}
_EVENT_KEYS = {'id', 'message_id', 'event_id', 'user_id', 'text', 'source',
               'timestamp', 'at', 'role', 'context', 'valid_from', 'valid_to',
               'supersedes', 'confidence', 'food_aliases', 'meal_aliases'}


def _reject_private_keys(value):
    if isinstance(value, dict):
        if set(value) & _PRIVATE_NAMES:
            raise ValueError('Private metadata cannot enter an observation record')
        for item in value.values():
            _reject_private_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_private_keys(item)


def _validated_records(records, model):
    allowed = {f.name for f in fields(model)}
    normalized = []
    for record in records:
        if not isinstance(record, dict) or set(record) - allowed:
            raise ValueError('Unexpected structured memory field')
        _reject_private_keys(record)
        if set(record.get('context', {})) - _RECORD_CONTEXT:
            raise ValueError('Unexpected preference context field')
        try:
            normalized.append(asdict(model(**record)))
        except TypeError as exc:
            raise ValueError('Malformed structured memory') from exc
    return normalized


def _validate_scenario_boundary(scenario):
    scenario['oracle_preferences'] = _validated_records(scenario['oracle_preferences'], Preference)
    scenario['oracle_restrictions'] = _validated_records(scenario['oracle_restrictions'], Restriction)
    visible = scenario.get('visible_context', {})
    _reject_private_keys(visible)
    if set(visible) - _OBS_CONTEXT:
        raise ValueError('Unexpected visible context field')
    for event in scenario['events']:
        if set(event) - _EVENT_KEYS:
            raise ValueError('Unexpected visible event field')
        _reject_private_keys(event)
        if set(event.get('context', {})) - _RECORD_CONTEXT:
            raise ValueError('Unexpected event context field')
    for slot in SLOTS:
        value = scenario['response_map'][slot]['value']
        allowed = {'day': ('WEEKDAY', 'WEEKEND'), 'meal': ('BREAKFAST', 'LUNCH', 'DINNER', 'SNACK'),
                   'company': ('alone', 'friends', 'family'), 'place': ('home', 'work'),
                   'current_preference': FOODS}[slot]
        if value is not None and value not in allowed:
            raise ValueError('Invalid response slot value')
    return scenario


def _scope_status(record, context):
    """Independent matching of explicitly communicated finite constraints."""
    missing = []
    for key, obs_key in (('day_scope', 'day'), ('meal_scope', 'meal')):
        value = record.get(key)
        if value is None and obs_key in context:
            missing.append('unknown_condition')
        if value not in (None, 'ANY'):
            if obs_key not in context:
                missing.append(obs_key)
            elif context[obs_key] != value:
                return None
    for key, value in (record.get('context') or {}).items():
        if key not in context:
            missing.append(key)
        elif context[key] != value:
            return None
    if record.get('scope_unknown'):
        missing.append('unknown_condition')
    return missing


def _observed(timestamp, reference):
    if not timestamp:
        return True
    def instant(value):
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return instant(timestamp) <= instant(reference)


def public_support(observation):
    """Finite observable constraint analysis, NOT the latent reward oracle.

    Returned lists are based on observed records, not author labels. This helper
    can support baselines/features; the existing heuristic remains independent.
    """
    context = observation.get('context', {})
    preferences = observation.get('preferences', [])
    restrictions = observation.get('restrictions', [])
    current = context.get('current_preference')
    compatible, missing, support, forbidden = [], set(), set(), set()
    when = observation.get('reference_time', '2026-10-19T12:00:00Z')
    for r in restrictions:
        if r.get('valid_from') and str(r['valid_from']) > when:
            continue
        if r.get('valid_to') and str(r['valid_to']) < when[:10]:
            continue
        if _scope_status(r, context) == []:
            forbidden.add(r['food_item'])
    for p in preferences:
        if p.get('valid_from') and str(p['valid_from']) > when:
            continue
        if p.get('valid_to') and str(p['valid_to']) < when[:10]:
            continue
        status = _scope_status(p, context)
        if status is not None:
            compatible.append((p, status))
            missing.update(status)
    possible = {p['food_item'] for p, _ in compatible if p['polarity'] == 'LIKE'} - forbidden
    def partition_overlap(left, right):
        # This policy-side calculation is independent of the terminal auditor.
        def days(record):
            if record.get('weekdays'):
                return set(record['weekdays'])
            return {'WEEKDAY': set(range(5)), 'WEEKEND': {5, 6}}.get(
                record.get('day_scope'), set(range(7)))
        if not days(left) & days(right):
            return False
        first, second = left.get('meal_scope'), right.get('meal_scope')
        if first not in (None, 'ANY') and second not in (None, 'ANY') and first != second:
            return False
        return not any(key in right.get('context', {}) and right['context'][key] != value
                       for key, value in left.get('context', {}).items())

    conflict = False
    for p, status in compatible:
        food = p['food_item']
        local_records = [(q, qstatus) for q, qstatus in compatible if q['food_item'] == food]
        unstable = any(q.get('stability') == 'OSCILLATING' for q, _ in local_records)
        opposite_overlap = any(a['polarity'] == 'LIKE' and b['polarity'] == 'DISLIKE'
                               and partition_overlap(a, b)
                               for a, _ in local_records for b, _ in local_records)
        local_conflict = unstable or opposite_overlap
        conflict = conflict or local_conflict
        if p['polarity'] != 'LIKE' or status:
            continue
        # A contradiction about another food, or a missing condition for another
        # preference, cannot invalidate this independent applicable declaration.
        if not local_conflict and p.get('source') == 'USER_STATEMENT':
            support.add(food)
    # A current choice is evidence about this occasion and this food only. It
    # resolves that choice without deleting other clean tastes or old conflicts.
    if current in FOODS:
        explicit = any(p['food_item'] == current and p['polarity'] == 'LIKE'
                       and p.get('source') == 'USER_STATEMENT' and p.get('explicit_update')
                       and not status for p, status in compatible)
        if explicit:
            support.add(current)
            possible.add(current)
    support -= forbidden
    unresolved = bool(context.get('current_uncertain'))
    relevant_slots = set(missing)
    if conflict:
        relevant_slots.add('current_preference')
    return {'supported_foods': sorted(support), 'possible_foods': sorted(possible),
            'safe_foods': sorted(set(FOODS) - forbidden), 'forbidden_foods': sorted(forbidden),
            'missing_slots': sorted(relevant_slots), 'conflict': conflict,
            'unresolved': unresolved}


def audit_evidence(observation, food=None):
    """Independent terminal-evidence audit, deliberately not ``public_support``.

    This evaluator checks communicated assertions against the selected action;
    it does not select a policy action or inspect future user responses. Keeping
    it separate lets a faulty policy feature hurt measured performance.
    """
    ctx = observation.get('context', {})
    when = observation.get('reference_time', '2026-10-19T12:00:00Z')

    def timely(record):
        return (_observed(record.get('timestamp'), when)
                and (not record.get('valid_from') or str(record['valid_from']) <= when)
                and (not record.get('valid_to') or str(record['valid_to']) >= when[:10]))

    def entailed(record):
        if not timely(record) or record.get('scope_unknown'):
            return False
        for field, key in (('day_scope', 'day'), ('meal_scope', 'meal')):
            value = record.get(field)
            if value is None and key in ctx:
                return False
            if value not in (None, 'ANY') and ctx.get(key) != value:
                return False
        return all(ctx.get(key) == value for key, value in record.get('context', {}).items())

    def overlap(left, right):
        for field in ('day_scope', 'meal_scope'):
            a, b = left.get(field), right.get(field)
            if a not in (None, 'ANY') and b not in (None, 'ANY') and a != b:
                return False
        return not any(key in right.get('context', {}) and value != right['context'][key]
                       for key, value in left.get('context', {}).items())

    def compatible(record):
        for field, key in (('day_scope', 'day'), ('meal_scope', 'meal')):
            value = record.get(field)
            if key in ctx and value not in (None, 'ANY', ctx[key]):
                return False
        return not any(key in ctx and ctx[key] != value
                       for key, value in record.get('context', {}).items())

    records = [p for p in observation.get('preferences', []) if timely(p) and compatible(p)]
    forbidden = {r['food_item'] for r in observation.get('restrictions', []) if entailed(r)}
    def conflict_in(items):
        return (any(p.get('stability') == 'OSCILLATING' for p in items)
            or any(a['food_item'] == b['food_item']
                and {a['polarity'], b['polarity']} == {'LIKE', 'DISLIKE'} and overlap(a, b)
                for i, a in enumerate(items) for b in items[i + 1:]))

    conflicting = conflict_in(records)
    current = ctx.get('current_preference')
    current_evidence = any(p['food_item'] == current and p['polarity'] == 'LIKE'
                           and p.get('source') == 'USER_STATEMENT' and p.get('explicit_update')
                           and entailed(p) for p in records)
    food_records = [p for p in records if p['food_item'] == food]
    # The terminal audit checks the selected food locally. Global memory
    # conflict is reported separately and still supports honest preservation.
    if current_evidence and food == current:
        supported = food not in forbidden
    else:
        supported = bool(food and food not in forbidden and not conflict_in(food_records) and
            any(p['polarity'] == 'LIKE' and p.get('source') == 'USER_STATEMENT'
                and entailed(p) for p in food_records))
    return {'supported': supported, 'conflict': bool(conflicting),
            'forbidden': food in forbidden if food else False}


class FoodPreferenceEnv:
    def __init__(self, scenario=None, condition='ORACLE_STRUCTURED', reward_config=None,
                 max_steps=4, max_clarifications=2):
        if condition not in ('ORACLE_STRUCTURED', 'ORACLE-STRUCTURED', 'PIPELINE'):
            raise ValueError('Unknown condition')
        self._scenario = _validate_scenario_boundary(deepcopy(scenario)) if scenario is not None else None
        self._fixed_scenario = scenario is not None
        self.condition = condition.replace('-', '_')
        self.reward_config = reward_config or RewardConfig()
        self.max_steps = int(max_steps)
        self.max_clarifications = int(max_clarifications)
        if self.max_steps < 1 or self.max_clarifications < 0:
            raise ValueError('Invalid episode budgets')
        self._done = True

    def reset(self, seed=None, scenario_id=None):
        if not self._fixed_scenario:
            self._scenario = _validate_scenario_boundary(generate_scenarios(1, seed=seed or 0)[0])
        if scenario_id is not None and scenario_id != self._scenario['scenario_id']:
            raise ValueError('scenario_id does not identify the supplied scenario')
        scenario = self._scenario
        self._context = deepcopy(scenario.get('visible_context', {}))
        visible_events = [e for e in scenario['events'] if _observed(e.get('timestamp'), scenario['reference_time'])]
        self._visible_event_ids = {e.get('id', e.get('message_id', e.get('event_id'))) for e in visible_events}
        self._messages = [{'role': e.get('role', 'user'), 'text': e['text'],
                           'evidence_id': e.get('id', e.get('message_id', e.get('event_id'))),
                           'source': e.get('source', 'USER_STATEMENT'),
                           'timestamp': e.get('timestamp', e.get('at'))} for e in visible_events]
        self._messages.append({'role': 'user', 'text': scenario['query']})
        if self.condition == 'ORACLE_STRUCTURED':
            self._preferences = deepcopy([p for p in scenario['oracle_preferences']
                if _observed(p.get('timestamp'), scenario['reference_time'])
                and p.get('evidence_id') in self._visible_event_ids])
            self._restrictions = deepcopy([r for r in scenario['oracle_restrictions']
                if _observed(r.get('timestamp'), scenario['reference_time'])
                and r.get('evidence_id') in self._visible_event_ids])
        else:
            self._preferences, self._restrictions = [], []
            for e in visible_events:
                parsed = extract_preferences({**e, 'message_id': e.get('id', e.get('message_id', e.get('event_id'))),
                    'at': e.get('timestamp', e.get('at'))}, user_id='user')
                self._preferences.extend(asdict(x) for x in parsed.preferences)
                self._restrictions.extend(asdict(x) for x in parsed.restrictions)
        self._steps, self._questions, self._asked = 0, 0, []
        self._done, self._total_reward = False, 0.0
        self._transitions = []
        self._initial_support = public_support(self.observation())
        # Denominators describe the communicated task, not extractor success.
        # These evaluator-only records are never passed to a PIPELINE policy.
        oracle_observation = self.observation()
        oracle_observation['preferences'] = [p for p in scenario['oracle_preferences']
            if p.get('evidence_id') in self._visible_event_ids and _observed(p.get('timestamp'), scenario['reference_time'])]
        oracle_observation['restrictions'] = [r for r in scenario['oracle_restrictions']
            if r.get('evidence_id') in self._visible_event_ids and _observed(r.get('timestamp'), scenario['reference_time'])]
        self._initial_oracle_support = public_support(oracle_observation)
        self._initial_oracle_observation = deepcopy(oracle_observation)
        self._outcome = None
        return self.observation(), {}

    def observation(self):
        # Whitelist only: never copy the parent scenario or private response map.
        return deepcopy({'query': self._scenario['query'],
                         'preferences': self._preferences, 'restrictions': self._restrictions,
                         'context': self._context, 'messages': self._messages,
                         'asked_slots': self._asked,
                         'remaining_questions': max(0, self.max_clarifications - self._questions),
                         'remaining_steps': max(0, self.max_steps - self._steps),
                         'reference_time': self._scenario['reference_time']})

    def _terminal_outcome(self, action):
        latent = self._scenario['latent']
        food = action.split(':', 1)[1] if action.startswith('ANSWER:') else None
        audit = audit_evidence(self.observation(), food)
        grounded = audit['supported'] if food else bool(
            action == 'PRESERVE_CONFLICT' and audit['conflict'])
        blocked = audit['forbidden']
        # Ground-truth safety is evaluated against restrictions actually stated
        # in visible evidence, including ones that PIPELINE may fail to extract.
        # The execution shield itself uses only the public structured records.
        safety_observation = self.observation()
        safety_observation['restrictions'] = [r for r in self._scenario.get('oracle_restrictions', [])
            if r.get('evidence_id') in self._visible_event_ids]
        safety_observation['context'] = {**self._scenario['latent'].get('context', {}), **self._context}
        attempted_violation = audit_evidence(safety_observation, food)['forbidden']
        violation = bool(attempted_violation and not blocked)
        latent_correct = bool(food and food in latent['acceptable_foods'])
        preserve = bool(action == 'PRESERVE_CONFLICT' and latent['unresolvable'] and audit['conflict'])
        correct_abstention = bool(action == 'ABSTAIN' and not latent['acceptable_foods']
                                  and not latent['unresolvable'] and not any(audit_evidence(self.observation(), f)['supported'] for f in FOODS))
        return {'latent_correct': latent_correct, 'evidence_supported': grounded,
                'restriction_violation': violation,
                'attempted_restriction_violation': attempted_violation,
                'safety_blocked': blocked,
                'correct_conflict_preservation': preserve,
                'correct_abstention': correct_abstention,
                'final_correct': bool((latent_correct and grounded and not attempted_violation) or preserve or correct_abstention),
                'unjustified_answer': bool(food and not grounded),
                'selected_food': None if blocked else food, 'selected_action': action,
                'executed_action': 'SAFETY_BLOCK' if blocked else action}

    def step(self, action):
        if self._done:
            raise RuntimeError('reset required before step or after terminal')
        if action not in ACTIONS:
            raise ValueError('Action is not in the finite catalog')
        if action in QUESTION_ACTIONS and self._questions >= self.max_clarifications:
            raise ValueError('Clarification budget exhausted')
        self._steps += 1
        before = public_support(self.observation())
        terminated, truncated = False, False
        if action in QUESTION_ACTIONS:
            self._questions += 1
            slot = action[len('CLARIFY_'):].lower()
            self._messages.append({'role': 'assistant', 'text': QUESTIONS[slot]})
            reply = deepcopy(self._scenario['response_map'][slot])
            repeated = slot in self._asked
            if repeated:
                reply = {'value': None, 'text': 'Eso ya te lo he comunicado; no tengo información adicional.'}
            self._asked.append(slot)
            self._messages.append({'role': 'user', 'text': reply['text'],
                'evidence_id': f'ec{self._questions}', 'source': 'USER_STATEMENT',
                'timestamp': self._scenario['reference_time']})
            self._visible_event_ids.add(f'ec{self._questions}')
            if not repeated and reply['value'] is not None:
                self._context[slot] = reply['value']
                if slot == 'current_preference':
                    event = {'id': f'ec{self._questions}', 'text': reply['text'],
                             'timestamp': self._scenario['reference_time'], 'source': 'USER_STATEMENT'}
                    if self.condition == 'ORACLE_STRUCTURED':
                        from rl_feasibility.scenario_generator import _preference
                        self._preferences.append(_preference(len(self._preferences), reply['value'],
                            explicit_update=True, temporary=True, evidence_id=event['id']))
                    else:
                        parsed = extract_preferences({**event, 'message_id': event['id'],
                            'at': event['timestamp']}, user_id='user')
                        for p in parsed.preferences:
                            item = asdict(p)
                            item['explicit_update'] = True
                            item['temporary'] = True
                            self._preferences.append(item)
            elif slot == 'current_preference' and not repeated and audit_evidence(self.observation())['conflict']:
                self._context['current_uncertain'] = True
            after = public_support(self.observation())
            # This diagnostic is evaluator-only; the policy sees the reply itself.
            informative = bool(before['supported_foods'] != after['supported_foods']
                               or before['possible_foods'] != after['possible_foods']
                               or before['missing_slots'] != after['missing_slots']
                               or before['unresolved'] != after['unresolved'])
            reward = -self.reward_config.clarification_cost
            self._transitions.append({'action': action, 'information_changed': informative,
                                      'relevant_question': slot in before['missing_slots'],
                                      'sufficient_before': bool(before['supported_foods']),
                                      'reward': reward})
            if self._steps >= self.max_steps:
                truncated = True
                self._outcome = self._terminal_outcome('TRUNCATED')
                reward += self.reward_config.truncation
        else:
            terminated = True
            self._outcome = self._terminal_outcome(action)
            reward = terminal_reward(action, self._outcome, self.reward_config)
            self._transitions.append({'action': action, 'reward': reward})
        self._done = terminated or truncated
        self._total_reward += reward
        # Safe info deliberately does not expose correctness, latent answer,
        # category, future response, or reward components even after termination.
        return self.observation(), reward, terminated, truncated, {}

    @property
    def done(self):
        return self._done


Environment = FoodPreferenceEnv


def diagnostic_metrics(env):
    """Evaluator-only: do not pass to policy training inputs or select_action."""
    if not env.done:
        raise ValueError('Diagnostics are released only after episode termination')
    outcome = deepcopy(env._outcome or {})
    # Evaluate acquisition independently of the chosen terminal answer and the
    # PIPELINE extractor. Only replies actually communicated are added here.
    truth = env._scenario['latent']
    initial = env._initial_oracle_observation
    needed = bool(env._initial_oracle_support['missing_slots'] and not truth['unresolvable']
                  and not any(audit_evidence(initial, f)['supported'] for f in truth['acceptable_foods']))
    communicated = deepcopy(initial)
    communicated['context'] = deepcopy(env._context)
    current = env._context.get('current_preference')
    if 'current_preference' in env._asked and current in FOODS:
        reply_index = env._asked.index('current_preference') + 1
        communicated['preferences'].append(asdict(Preference(
            'audit_current', 'user', current, 'LIKE', day_scope='ANY', meal_scope='ANY',
            source='USER_STATEMENT', timestamp=env._scenario['reference_time'],
            evidence_id=f'ec{reply_index}', temporary=True, explicit_update=True)))
    acquired = bool(needed and any(audit_evidence(communicated, f)['supported'] for f in truth['acceptable_foods']))
    outcome.update({'return': env._total_reward, 'turns': env._steps,
                    'clarifications': env._questions,
                    'useful_clarifications': sum(t.get('information_changed', False) for t in env._transitions),
                    'task_useful_clarifications': sum(t.get('information_changed', False) and not t.get('sufficient_before', False) and t.get('relevant_question', False) for t in env._transitions),
                    'unnecessary_clarifications': sum(t.get('sufficient_before', False) or not t.get('information_changed', False) or not t.get('relevant_question', False) for t in env._transitions if t['action'] in QUESTION_ACTIONS),
                    'necessary_clarification_required': needed,
                    'necessary_information_acquired': acquired,
                    'necessary_clarification_omitted': bool(needed and not acquired),
                    'initial_need_without_final_success': bool(needed and not outcome.get('final_correct')),
                    'decision_correct': bool(outcome.get('final_correct')),
                    'unsupported_answer': bool(outcome.get('unjustified_answer')),
                    'has_restriction': bool(env._scenario.get('oracle_restrictions')),
                    'unresolvable': env._scenario['latent']['unresolvable'],
                    'conflict_unresolvable': env._scenario['latent']['unresolvable'],
                    'decision_unresolvable': bool(env._scenario['latent']['unresolvable'] and not truth['acceptable_foods']),
                    'evidence_fidelity': bool(outcome.get('evidence_supported') or outcome.get('correct_conflict_preservation') or outcome.get('selected_action') == 'ABSTAIN'),
                    'transitions': deepcopy(env._transitions)})
    return outcome


final_metrics = diagnostic_metrics
