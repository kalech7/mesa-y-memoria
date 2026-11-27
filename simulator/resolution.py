"""Deterministic, evidence-preserving food preference resolution.

This component consumes communicated evidence only. It never receives scenario gold
or a simulator's private state. Retrieval is deliberately exhaustive for a small
user memory; applicability and selection are separate, inspectable stages.
"""
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from time import perf_counter
import unicodedata

from simulator.preferences import extract_preferences, parse_context
from data_generation.schema import assert_no_private_keys


def _record(value):
    return asdict(value) if is_dataclass(value) else dict(value)


def _norm(value):
    return ''.join(c for c in unicodedata.normalize('NFD', str(value).lower())
                   if unicodedata.category(c) != 'Mn').strip()


def _date(value):
    return str(value)[:10] if value else None


def _instant(value):
    """ISO instants are compared in UTC; naive datetimes explicitly mean UTC."""
    if isinstance(value, datetime):
        instant = value
    else:
        instant = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(timezone.utc)


def _observed_at_or_before(timestamp, reference_time):
    if not timestamp or not reference_time:
        return True
    # A date-only reference represents the whole named day, preserving the
    # existing API. Time-bearing references are exact instants, including offsets.
    if len(str(reference_time)) <= 10 or len(str(timestamp)) <= 10:
        return _date(timestamp) <= _date(reference_time)
    return _instant(timestamp) <= _instant(reference_time)


def _scope_days(record):
    if record.get('weekdays'):
        return frozenset(record['weekdays'])
    return {'WEEKDAY': frozenset(range(5)), 'WEEKEND': frozenset((5, 6)),
            'ANY': frozenset(range(7))}.get(record.get('day_scope'))


def _context_match(condition, observed):
    if isinstance(condition, (tuple, list, set)):
        return observed in condition
    return condition == observed


def _status(record, context, when):
    """Return missing conditions, or None for a known-incompatible record."""
    missing = set()
    if when:
        if record.get('valid_from') and when < _date(record['valid_from']):
            return None
        if record.get('valid_to') and when > _date(record['valid_to']):
            return None
    elif record.get('valid_from') or record.get('valid_to'):
        missing.add('date')
    days, requested = _scope_days(record), _scope_days(context)
    if days is None:
        # An unqualified question can report the communicated taste without
        # asserting that it applies to a newly supplied day. None stays None.
        if context.get('day_scope') is not None or context.get('weekdays'):
            missing.add('day_scope')
    elif days != frozenset(range(7)):
        if requested is None:
            missing.add('day_scope')
        elif not requested <= days:
            if not requested & days:
                return None
            missing.add('day_scope')
    meal = record.get('meal_scope')
    if meal is None:
        if context.get('meal_scope') is not None:
            missing.add('meal_scope')
    elif meal != 'ANY':
        if not context.get('meal_scope') or context.get('meal_scope') == 'ANY':
            missing.add('meal_scope')
        elif context['meal_scope'] != meal:
            return None
    for key, value in (record.get('context') or {}).items():
        actual = (context.get('context') or {}).get(key)
        if actual is None:
            missing.add(key)
        elif not _context_match(value, actual):
            return None
    missing.update(record.get('scope_unknown') or ())
    return missing


def _same_scope(left, right):
    return (_scope_days(left) == _scope_days(right)
            and left.get('meal_scope') == right.get('meal_scope')
            and left.get('context', {}) == right.get('context', {}))


def _more_specific(left, right):
    """Partial ordering: each scope narrows or equals, at least one narrows."""
    ld, rd = _scope_days(left), _scope_days(right)
    if ld is None or rd is None or not ld <= rd:
        return False
    lm, rm = left.get('meal_scope'), right.get('meal_scope')
    if lm is None or rm is None or (lm != rm and rm != 'ANY'):
        return False
    lc, rc = left.get('context') or {}, right.get('context') or {}
    if any(key not in lc or lc[key] != val for key, val in rc.items()):
        return False
    # A bounded exception is narrower in time than an unbounded habit.
    lv = (left.get('valid_from'), left.get('valid_to'))
    rv = (right.get('valid_from'), right.get('valid_to'))
    for lval, rval, start in ((lv[0], rv[0], True), (lv[1], rv[1], False)):
        if rval and (not lval or (_date(lval) < _date(rval) if start else _date(lval) > _date(rval))):
            return False
    return ld != rd or lm != rm or lc != rc or lv != rv


def _statement(record):
    return record.get('source') == 'USER_STATEMENT'


def _superseded(record, records, when):
    for other in records:
        if other is record:
            continue
        if when and ((_date(other.get('timestamp')) and _date(other['timestamp']) > when)
                     or (other.get('valid_from') and _date(other['valid_from']) > when)
                     or (other.get('valid_to') and _date(other['valid_to']) < when)):
            continue
        # Corrections are scoped statements, not a license for observations or
        # unrelated contexts to invalidate a user's communicated preference.
        if not (_statement(other) and _statement(record) and _same_scope(other, record)):
            continue
        references = other.get('supersedes') or ()
        if record.get('preference_id') in references or record.get('evidence_id') in references:
            return True
        if (other.get('explicit_update')
                and _norm(other.get('food_item')) == _norm(record.get('food_item'))
                and str(other.get('timestamp', '')) > str(record.get('timestamp', ''))):
            return True
    return False


def _partitioned(records):
    for index, left in enumerate(records):
        for right in records[index + 1:]:
            ld, rd = _scope_days(left), _scope_days(right)
            if ld and rd and not ld & rd:
                return True
            lm, rm = left.get('meal_scope'), right.get('meal_scope')
            if lm not in (None, 'ANY') and rm not in (None, 'ANY') and lm != rm:
                return True
            lc, rc = left.get('context') or {}, right.get('context') or {}
            if any(key in rc and rc[key] != value for key, value in lc.items()):
                return True
    return False


def _oscillating(records):
    if any(r.get('stability') in ('uncertain', 'oscillating', 'UNCERTAIN', 'OSCILLATING') for r in records):
        return True
    groups = {}
    for record in records:
        if not _statement(record):
            continue
        key = (_norm(record.get('food_item')), str(_scope_days(record)),
               record.get('meal_scope'), str(sorted((record.get('context') or {}).items())))
        groups.setdefault(key, []).append(record)
    for group in groups.values():
        polarities = []
        for record in sorted(group, key=lambda r: (str(r.get('timestamp', '')), r.get('preference_id', ''))):
            polarity = record.get('polarity')
            if not polarities or polarities[-1] != polarity:
                polarities.append(polarity)
        if len(polarities) >= 3 and polarities[-1] == polarities[-3]:
            return True
    return False


# Small transparent lexical catalogue for synthetic safety checks only. No
# nutritional or clinical inference is made for an unknown ingredient.
_CATEGORIES = {
    'lacteos': {'leche', 'queso', 'yogur', 'mantequilla'},
    'gluten': {'trigo', 'pan', 'pasta', 'tostadas'},
    'frutos secos': {'mani', 'cacahuete', 'nuez', 'nueces', 'almendra', 'almendras'},
    'carne': {'carne', 'pollo', 'cerdo', 'res', 'hamburguesa'},
}


def _restricted(food, restriction):
    target = _norm(restriction.get('food_item') or restriction.get('substance') or restriction.get('category') or '')
    actual = _norm(food)
    if not target:
        return False
    return (target == actual or target in actual.split()
            or any(target == cat and any(item == actual or item in actual.split() for item in items)
                   for cat, items in _CATEGORIES.items()))


def resolve_preferences(preferences, restrictions, query, reference_time=None, scenario_id='', user_id=''):
    """Resolve visible records, retaining every original and unresolved alternative."""
    started = perf_counter()
    preferences = [_record(p) for p in preferences if not user_id or _record(p).get('user_id') == user_id]
    restrictions = [_record(r) for r in restrictions if not user_id or _record(r).get('user_id') == user_id]
    context = _record(parse_context(query, reference_time))
    when = context.get('date') or _date(reference_time)
    # Ignore future observations: unavailable at the query's reference time.
    visible = [p for p in preferences if _observed_at_or_before(p.get('timestamp'), reference_time)]
    statuses = {p['preference_id']: _status(p, context, when) for p in visible}
    compatible = [p for p in visible if statuses[p['preference_id']] is not None and not _superseded(p, visible, when)]
    applicable = [p for p in compatible if not statuses[p['preference_id']]]
    pending = [p for p in compatible if statuses[p['preference_id']]]
    active_restrictions = [r for r in restrictions
                           if _observed_at_or_before(r.get('timestamp'), reference_time)
                           and _status(r, context, when) is not None]
    result = {
        'scenario_id': scenario_id, 'user_id': user_id, 'query': query,
        'resolved_context': context,
        'retrieved_preference_ids': sorted(p['preference_id'] for p in visible),
        'applicable_preference_ids': sorted(p['preference_id'] for p in applicable),
        'retrieved_restriction_ids': sorted(r['restriction_id'] for r in active_restrictions),
        'applicable_restriction_ids': sorted(r['restriction_id'] for r in active_restrictions
                                             if not _status(r, context, when)),
        'detected_conflict_type': None,
        'selected_action': 'ABSTAIN', 'selected_food_items': [], 'evidence_ids': [],
        'clarification_question': None, 'explanation': '', 'execution_time_ms': 0.0,
    }
    def finish(action, reason, used=(), foods=(), question=None, conflict=None):
        result.update(selected_action=action, explanation=reason,
                      evidence_ids=sorted({p['evidence_id'] for p in used if p.get('evidence_id')}),
                      selected_food_items=sorted(set(foods)), clarification_question=question)
        if conflict:
            result['detected_conflict_type'] = conflict
        result['execution_time_ms'] = (perf_counter() - started) * 1000
        return result
    if context.get('unknown_conditions'):
        return finish('CLARIFY', 'La consulta contiene condiciones no identificadas o incompatibles.',
                      compatible, question='¿Puedes precisar la fecha, comida y condiciones de tu consulta?')
    if pending:
        missing = set().union(*(statuses[p['preference_id']] for p in pending))
        names = {'day_scope': 'día de la semana', 'meal_scope': 'tipo de comida', 'date': 'fecha',
                 'place': 'lugar', 'company': 'compañía', 'goal': 'objetivo alimentario',
                 'occasion': 'ocasión', 'budget': 'presupuesto',
                 'uncertain_preference': 'gusto que aún no has decidido',
                 'multiple_place': 'lugar', 'multiple_company': 'compañía',
                 'multiple_meals': 'tipo de comida', 'unparsed_condition': 'condición que limita ese gusto'}
        details = ', '.join(sorted({names.get(name, 'condición de aplicación') for name in missing}))
        return finish('CLARIFY', 'Hay evidencias cuyo alcance necesita precisar el contexto.',
                      applicable + pending, question=f'¿Para qué {details} quieres conocer la preferencia?',
                      conflict='CPC' if _partitioned(compatible) else None)
    if not applicable:
        return finish('ABSTAIN', 'No hay preferencias comunicadas aplicables al contexto solicitado.')
    if _oscillating(compatible):
        return finish('PRESERVE_CONFLICT', 'La trayectoria no permite suponer una preferencia estable.',
                      compatible, conflict='BOC')
    # Specificity may refine a general declaration only for a comparable context.
    effective = [p for p in applicable if not any(_statement(p) == _statement(q)
                 and _norm(p['food_item']) == _norm(q['food_item']) and _more_specific(q, p)
                 for q in applicable if q is not p)]
    for index, left in enumerate(effective):
        for right in effective[index + 1:]:
            if (_norm(left['food_item']) == _norm(right['food_item'])
                    and {left.get('polarity'), right.get('polarity')} == {'LIKE', 'DISLIKE'}):
                scc = left.get('source') != right.get('source')
                return finish('VERIFY' if scc else 'PRESERVE_CONFLICT',
                              'Las evidencias aplicables son incompatibles; se conservan ambas.',
                              effective, conflict='SCC' if scc else 'SAME_SCOPE')
    declared = [p for p in effective if _statement(p)]
    if not declared:
        return finish('VERIFY', 'El comportamiento observado no establece por sí solo un gusto declarado.', effective)
    positives = [p for p in declared if p.get('polarity') == 'LIKE']
    safe, used_restrictions, uncertain, scope_uncertain = [], [], False, False
    for preference in positives:
        blockers = [r for r in active_restrictions if _restricted(preference['food_item'], r)]
        used_restrictions.extend(blockers)
        if any(str(r.get('certainty', 'CONFIRMED')).upper() in
               ('UNCERTAIN', 'UNKNOWN', 'POSSIBLE', 'UNCONFIRMED') for r in blockers):
            uncertain = True
        if any(_status(r, context, when) for r in blockers):
            scope_uncertain = True
        if not blockers:
            safe.append(preference['food_item'])
    if uncertain or (scope_uncertain and not safe):
        return finish('VERIFY', 'Debe verificarse una restricción relevante antes de recomendar.',
                      effective + used_restrictions)
    if positives and not safe:
        return finish('ABSTAIN', 'Las opciones con evidencia favorable incumplen restricciones explícitas.',
                      effective + used_restrictions)
    return finish('ANSWER', 'Decisión fundada en las evidencias aplicables, sin descartar el historial.',
                  effective + used_restrictions, safe)


class FoodPreferenceEngine:
    """SQLite-backed adapter using the existing Store rather than replacing it."""
    def __init__(self, store):
        self.store = store

    def ingest(self, events):
        # Validate/extract the entire public batch before any persistent mutation.
        prepared = []
        for original in events:
            event = dict(original)
            assert_no_private_keys(event)
            def reject_hidden(value):
                if isinstance(value, dict):
                    if set(value) & {'true_state', 'uncommunicated', 'clarification_responses', 'expected_food_items', 'gold'}:
                        raise ValueError('Información privada dentro de evento público')
                    for child in value.values():
                        reject_hidden(child)
                elif isinstance(value, (list, tuple)):
                    for child in value:
                        reject_hidden(child)
            reject_hidden(event)
            allowed = {'message_id', 'event_id', 'at', 'session_id', 'role', 'channel', 'source',
                       'text', 'context', 'food_aliases', 'meal_aliases', 'valid_from', 'valid_to',
                       'supersedes', 'confidence', 'user_id'}
            if set(event) - allowed:
                raise ValueError('Evento contiene campos ajenos al contrato público')
            if event.get('user_id', self.store.person_id) != self.store.person_id:
                raise ValueError('Evento pertenece a otro usuario')
            if 'message_id' not in event and 'event_id' in event:
                event['message_id'] = event['event_id']
            required = {'message_id', 'at', 'session_id', 'role', 'channel', 'source', 'text'}
            if not required <= set(event) or any(not isinstance(event[k], str) for k in required):
                raise ValueError('Evento público incompleto o de tipos inválidos')
            _instant(event['at'])  # Validate timestamps before persisting any event.
            extracted = extract_preferences(event, self.store.person_id)
            prepared.append((event, extracted))
        for event, extracted in prepared:
            self.store.add(event)
            for preference in extracted.preferences:
                self.store.add_preference(preference)
            for restriction in extracted.restrictions:
                self.store.add_restriction(restriction)

    def decide(self, scenario_id, user_id, query, reference_time=None):
        if user_id != self.store.person_id:
            raise ValueError('Consulta pertenece a otro usuario')
        return resolve_preferences(self.store.preferences(), self.store.restrictions(), query,
                                   reference_time, scenario_id, user_id)
