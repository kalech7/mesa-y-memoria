"""Observable-only comparators for SPEC-RL-004.

Behavior cloning and Q-learning share precisely the same categorical state and
factored action catalog. They never call the rule policy at inference time.
The public temporal parser/reconciler is a fixed representation, not learned.
"""
from collections import Counter
from datetime import datetime, timezone
import json

from .observations import evidence_kind


MEMORY_OPERATIONS = ('ADD_MEMORY', 'UPDATE_WITH_HISTORY', 'PRESERVE_CONFLICT', 'NOOP')
QUESTION_SLOTS = ('preference', 'meal', 'day', 'source')


def _time(value):
    if not value:
        return None
    result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result


def _facts(observation):
    """Only the public projection. No access to scenarios or private env state."""
    result = []
    seen = set()
    for fact in list(observation.get('memory', [])) + list(observation.get('incoming_evidence', [])):
        identifier = fact.get('memory_id', json.dumps(fact, sort_keys=True))
        if identifier not in seen:
            seen.add(identifier)
            interpreted = dict(fact)
            interpreted['kind'] = evidence_kind(fact)
            result.append(interpreted)
    return sorted(result, key=lambda fact: (str(fact.get('observed_at', '')), str(fact.get('recorded_at', ''))))


def _scope_status(fact, context):
    missing = []
    for slot, value in (fact.get('scope') or fact.get('context') or {}).items():
        if value in (None, 'ANY'):
            continue
        if context.get(slot) in (None, 'ANY'):
            missing.append(slot)
        elif context[slot] != value:
            return False, []
    return True, missing


def public_analysis(observation):
    """Strong deterministic temporal interpretation of available evidence.

    Explicit replacement, validity intervals, scoped exceptions, restriction
    provenance, and already answered questions are all considered. Recency alone
    does not resolve disagreement. This is also the public feature extractor
    shared by the two learned models, so neither receives a hidden oracle.
    """
    facts = _facts(observation)
    at = _time(observation.get('query_at'))
    target = observation.get('target_food')
    context = observation.get('context', {})
    ends = {f.get('memory_id'): f.get('valid_to') for f in facts}
    for fact in facts:
        if fact.get('kind') in ('EXPLICIT_CHANGE', 'CLARIFICATION'):
            for previous in fact.get('supersedes', []):
                ends[previous] = fact.get('valid_from') or fact.get('observed_at')
    active, potential, missing = [], [], set()
    encoded = []
    incoming_ids = {f.get('memory_id') for f in observation.get('incoming_evidence', [])}
    for fact in facts:
        start = _time(fact.get('valid_from') or fact.get('observed_at'))
        end = _time(ends.get(fact.get('memory_id')))
        temporal = 'FUTURE' if at and start and start > at else 'ENDED' if at and end and end <= at else 'CURRENT'
        applicable, gaps = _scope_status(fact, context)
        relevant = fact.get('food') == target
        encoded.append((relevant, fact.get('polarity'), fact.get('kind'), fact.get('source'), temporal,
                        tuple(sorted((fact.get('scope') or {}).items())), applicable,
                        tuple(sorted(gaps)), fact.get('memory_id') in incoming_ids,
                        bool(fact.get('supersedes'))))
        if not relevant or not applicable or temporal != 'CURRENT' or fact.get('status') == 'INACTIVE':
            continue
        potential.append(fact)
        missing.update(gaps)
        if not gaps:
            active.append(fact)
    # Explicit communicated replacements remain authoritative even when a
    # producer has omitted an optional supersedes relation. They replace only
    # matching food/scope, never restrictions or contextual alternatives.
    replacements = [f for f in active if f.get('kind') in ('EXPLICIT_CHANGE', 'CLARIFICATION')]
    for replacement in replacements:
        active = [f for f in active if f is replacement or f.get('kind') == 'RESTRICTION'
                  or (f.get('scope') or {}) != (replacement.get('scope') or {})
                  or str(f.get('observed_at', '')) >= str(replacement.get('observed_at', ''))]
    exceptions = [f for f in active if f.get('kind') == 'TEMPORARY_EXCEPTION']
    if exceptions:
        active = [f for f in active if f in exceptions or f.get('kind') == 'RESTRICTION']
    blocked = any(f.get('kind') == 'RESTRICTION' or f.get('polarity') == 'FORBID' for f in potential)
    known = [f for f in active if f.get('kind') not in ('AMBIGUOUS', 'UNRESOLVED', 'OBSERVED_BEHAVIOR')
             and f.get('source') != 'ORDER_HISTORY' and f.get('uncertainty', 'KNOWN') == 'KNOWN'
             and f.get('polarity') in ('LIKE', 'DISLIKE')]
    polarities = {f['polarity'] for f in known}
    uncertain = any(f.get('kind') in ('AMBIGUOUS', 'UNRESOLVED', 'OBSERVED_BEHAVIOR')
                    or f.get('uncertainty', 'KNOWN') != 'KNOWN' for f in active)
    conflict = len(polarities) > 1 or uncertain
    return {'encoded_facts': encoded, 'active': active, 'potential': potential,
            'missing': sorted(missing), 'blocked': blocked, 'conflict': conflict,
            'polarity': next(iter(polarities)) if len(polarities) == 1 and not conflict and not missing else None}


def _action(conversation, operation='NOOP', polarity=None, slot=None):
    return {'conversation': conversation, 'memory_operation': operation, 'polarity': polarity, 'slot': slot}


def _memory_operation(observation, analysis):
    incoming = observation.get('incoming_evidence', [])
    if not incoming:
        return 'NOOP'
    kinds = {evidence_kind(f) for f in incoming}
    if kinds & {'EXPLICIT_CHANGE', 'CLARIFICATION'}:
        return 'UPDATE_WITH_HISTORY'
    if analysis['conflict'] or kinds & {'AMBIGUOUS', 'UNRESOLVED', 'OBSERVED_BEHAVIOR'}:
        return 'PRESERVE_CONFLICT'
    return 'ADD_MEMORY'


class LastWriteWins:
    name = 'LAST_WRITE_WINS'

    def select_action(self, observation):
        facts = [f for f in _facts(observation) if f.get('food') == observation.get('target_food')
                 and f.get('kind') != 'RESTRICTION' and f.get('polarity') in ('LIKE', 'DISLIKE')]
        if not facts:
            return _action('CONDITIONAL_ANSWER')
        latest = facts[-1]
        operation = 'UPDATE_WITH_HISTORY' if observation.get('incoming_evidence') else 'NOOP'
        return _action('ANSWER', operation, latest['polarity'])


class HistoricalClarificationRules:
    name = 'HISTORICAL_CLARIFICATION_RULES'

    def select_action(self, observation):
        analysis = public_analysis(observation)
        operation = _memory_operation(observation, analysis)
        if analysis['blocked']:
            return _action('ANSWER', operation, 'DISLIKE')
        if analysis['polarity']:
            return _action('ANSWER', operation, analysis['polarity'])
        history = observation.get('clarification_history', [])
        unanswered = observation.get('remaining_clarifications', 0) > 0 and not history
        if unanswered and (analysis['conflict'] or analysis['missing']):
            slot = analysis['missing'][0] if analysis['missing'] else 'preference'
            return _action('CLARIFY', 'NOOP', slot=slot)
        if analysis['conflict']:
            return _action('PRESERVE_CONFLICT', operation)
        return _action('CONDITIONAL_ANSWER', operation)


def feature_key(observation):
    """ID, absolute date, user, family, template and private-label-free key.

    Relative temporal applicability and communicated scopes are retained. The
    feature extractor is equally available to B2, B3 and B4 in each condition;
    it is an explicitly shared fixed symbolic capability, not attributed to RL.
    """
    analysis = public_analysis(observation)
    history = observation.get('clarification_history', [])
    questions = [(h.get('slot', h.get('question_key')), h.get('status'),
                  h.get('resolved')) for h in history]
    value = {'facts': analysis['encoded_facts'], 'context': observation.get('context', {}),
             'budget': observation.get('remaining_clarifications', 0), 'questions': questions,
             'missing': analysis['missing'], 'conflict': analysis['conflict'],
             'blocked': analysis['blocked'], 'polarity': analysis['polarity']}
    # O3 relations retain the semantic relation type, never evidence IDs. These
    # fields are derived from visible text/history in observations.py.
    enriched = observation.get('enriched', {})
    if enriched:
        value['enrichment_available'] = True
        value['known_contexts'] = enriched.get('known_context', {})
        value['missing_contexts'] = enriched.get('missing_context', [])
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def action_catalog(observation):
    actions = []
    for operation in MEMORY_OPERATIONS:
        actions.extend([_action('ANSWER', operation, 'LIKE'), _action('ANSWER', operation, 'DISLIKE'),
                        _action('CONDITIONAL_ANSWER', operation), _action('PRESERVE_CONFLICT', operation),
                        _action('NOOP', operation)])
        if observation.get('remaining_clarifications', 0) > 0:
            actions.extend(_action('CLARIFY', operation, slot=slot) for slot in QUESTION_SLOTS)
    return [json.dumps(action, sort_keys=True, separators=(',', ':')) for action in actions]


class SupervisedPolicy:
    name = 'SUPERVISED_POLICY'

    def __init__(self, counts=None, prior=None):
        self.counts = counts or {}
        self.prior = prior or {}
        self.seen_decisions = self.unseen_decisions = 0

    def select_action(self, observation):
        state = feature_key(observation)
        values = self.counts.get(state)
        if values is None:
            self.unseen_decisions += 1
            values = self.prior
        else:
            self.seen_decisions += 1
        actions = action_catalog(observation)
        return json.loads(max(actions, key=lambda a: (values.get(a, 0), -actions.index(a))))

    def to_dict(self):
        return {'algorithm': 'tabular categorical maximum-likelihood behavior cloning',
                'counts': self.counts, 'prior': self.prior, 'unseen_state': 'empirical_action_prior',
                'rule_fallback': False, 'feature_version': 'SPEC-RL-004-v1'}

    @classmethod
    def from_dict(cls, checkpoint):
        if checkpoint.get('feature_version') != 'SPEC-RL-004-v1' or 'counts' not in checkpoint:
            raise ValueError('Unsupported supervised pilot checkpoint')
        return cls(checkpoint['counts'], checkpoint.get('prior', {}))


class QLearningPolicy:
    name = 'RL_POLICY'

    def __init__(self, q_values=None):
        self.q_values = q_values or {}
        self.seen_decisions = self.unseen_decisions = 0

    def select_action(self, observation):
        state = feature_key(observation)
        values = self.q_values.get(state)
        if values is None:
            self.unseen_decisions += 1
            values = {}
        else:
            self.seen_decisions += 1
        actions = action_catalog(observation)
        return json.loads(max(actions, key=lambda a: (values.get(a, 0.0), -actions.index(a))))

    def to_dict(self):
        return {'algorithm': 'tabular Q-learning', 'q_values': self.q_values,
                'unseen_state': 'zero_initialization_fixed_catalog_tiebreak',
                'rule_fallback': False, 'feature_version': 'SPEC-RL-004-v1'}

    @classmethod
    def from_dict(cls, checkpoint):
        if checkpoint.get('feature_version') != 'SPEC-RL-004-v1' or 'q_values' not in checkpoint:
            raise ValueError('Unsupported Q-learning pilot checkpoint')
        return cls(checkpoint['q_values'])
