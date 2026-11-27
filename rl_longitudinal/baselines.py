"""Observable-only longitudinal policies and an equally expressive SFT baseline.

The planner enumerates hypothetical *public* replies and future queries under a
declared, uniform response model. It never reads an environment's private state.
The symbolic reconciliation supplied to every policy is shared infrastructure,
not a capability learned by either the supervised or reinforcement learner.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from dataclasses import asdict
import json
from pathlib import Path


FOODS = ('ensalada', 'pizza', 'sopa', 'arroz', 'pollo', 'sushi', 'ramen', 'maní')
MEMORY_OPERATIONS = ('ADD', 'UPDATE_WITH_HISTORY', 'NOOP', 'PRESERVE_CONFLICT')
SLOT_VALUES = {'day': ('WEEKDAY', 'WEEKEND'),
               'meal': ('BREAKFAST', 'LUNCH', 'DINNER', 'SNACK'),
               'company': ('alone', 'friends', 'family'), 'place': ('home', 'work')}


def _facts(observation):
    return list(observation.get('historical_memory', [])) + list(observation.get('incoming_evidence', []))


def _time(value):
    if not value:
        return None
    try:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def canonical_foods(observation):
    """Ordering depends on communicated chronology/content, never record order/ID."""
    facts = sorted(_facts(observation), key=lambda f: (
        str(f.get('observed_at', '')), str(f.get('food', f.get('food_item', ''))),
        str(f.get('polarity', '')), str(f.get('kind', ''))))
    result = []
    for fact in facts:
        food = fact.get('food', fact.get('food_item'))
        if food and food not in result:
            result.append(food)
    return result + [food for food in FOODS if food not in result]


def _reconciliation(observation):
    return observation.get('reconciliation', {})


def _required(observation):
    return observation.get('required_decision', {})


def _safe_foods(observation):
    blocked = set(_reconciliation(observation).get('forbidden_foods', []))
    return [food for food in canonical_foods(observation) if food not in blocked]


def _supported(observation):
    rec = _reconciliation(observation)
    result = rec.get('safe_foods', rec.get('supported_foods', []))
    return [food for food in canonical_foods(observation) if food in result and food in _safe_foods(observation)]


def _question(observation):
    decision = _required(observation)
    return decision.get('question_key', decision.get('clarification_key'))


def memory_operation(observation):
    """Ingest provenance; substitution requires explicit communicated change."""
    incoming = observation.get('incoming_evidence', [])
    if not incoming:
        return 'NOOP'
    kinds = {fact.get('kind') for fact in incoming}
    if kinds <= {'EXPLICIT_CHANGE', 'CLARIFICATION'}:
        return 'UPDATE_WITH_HISTORY'
    if kinds & {'AMBIGUOUS', 'OBSERVED_BEHAVIOR', 'AMBIGUOUS_CHANGE', 'UNRESOLVED', 'CONTRADICTION'}:
        return 'PRESERVE_CONFLICT'
    return 'ADD'


def _action(conversation, observation, *, food=None, question_key=None, operation=None):
    return {'conversation': conversation, 'food': food, 'question_key': question_key,
            'memory_operation': operation or memory_operation(observation)}


def _final(observation):
    supported = _supported(observation)
    if supported:
        return _action('ANSWER', observation, food=supported[0])
    rec = _reconciliation(observation)
    if rec.get('conflicts') or rec.get('uncertain_foods') or rec.get('unresolved_conflicts') or rec.get('conflict'):
        return _action('PRESERVE_CONFLICT', observation)
    return _action('ABSTAIN', observation)


def _question_available(observation, key):
    if not key or observation.get('remaining_clarifications', 0) <= 0:
        return False
    if observation.get('user_available') is False:
        return False
    previous = [row for row in observation.get('clarification_history', []) if row.get('question_key') == key]
    if any(row.get('status') == 'HYPOTHETICAL' for row in previous):
        return False  # The planner cannot ask the same question twice in its tree.
    # The shared core has already checked occasion-specific answers and whether
    # new evidence reopens an ambiguity. Do not impose an eternal ban by key.
    required = _required(observation)
    if required.get('action', required.get('conversation')) == 'CLARIFY' and required.get('question_key') == key:
        return True
    if key in SLOT_VALUES:
        query_at = _time(observation.get('query_at'))
        previous = [row for row in previous if
                    (_time(row['query_at']) == query_at if row.get('query_at') else
                     _time(row.get('at')).date() == query_at.date()
                     if _time(row.get('at')) and query_at else True)]
    elif previous:
        food = key.partition(':')[2]
        dates = [_time(f.get('recorded_at')) for f in _facts(observation) if f.get('food') == food]
        dates = [value for value in dates if value is not None]
        if dates:
            previous = [row for row in previous if _time(row.get('at')) is None or _time(row['at']) >= max(dates)]
    if previous:
        return False
    return True


class LastWriteWins:
    """Explicit MEM-02/history ablation; the environment retains safety shielding."""
    name = 'LAST_WRITE_WINS'
    ablations = ('MEM-02', 'MEM-05', 'MEM-06')

    def select_action(self, observation):
        facts = [f for f in _facts(observation) if f.get('kind') != 'RESTRICTION']
        if facts:
            latest_at = max(str(f.get('observed_at', '')) for f in facts)
            latest = [f for f in facts if str(f.get('observed_at', '')) == latest_at]
            # One utterance can contain "no sushi, now ramen" with identical
            # timestamps. Keep that simultaneous cohort rather than letting a
            # food-name tie break throw away its positive declaration.
            negatives = {f.get('food', f.get('food_item')) for f in latest if f.get('polarity') == 'DISLIKE'}
            positives = {f.get('food', f.get('food_item')) for f in latest if f.get('polarity') == 'LIKE'}
            candidates = positives - negatives
            for food in canonical_foods(observation):
                if food in candidates and food in _safe_foods(observation):
                    return _action('ANSWER', observation, food=food)
        return _action('ABSTAIN', observation)


class HistoricalRuleBased:
    """Full historical reconciliation with a clearly marked no-question ablation."""
    name = 'HISTORICAL_RULE_BASED'
    ablations = ('MEM-05', 'MEM-06')

    def select_action(self, observation):
        return _final(observation)


class ClarificationRuleBased:
    name = 'CLARIFICATION_RULE_BASED'
    ablations = ()

    def select_action(self, observation):
        required = _required(observation)
        key = _question(observation)
        conversation = required.get('conversation', required.get('selected_action', required.get('action')))
        if conversation == 'CLARIFY' and _question_available(observation, key):
            return _action('CLARIFY', observation, question_key=key)
        return _final(observation)


class PlanningBaseline:
    """Depth-two finite lookahead over observable beliefs, not a private oracle.

    Declared model: missing scopes are uniform over values already communicated
    in memory (otherwise the public slot domain); a preference clarification has
    equiprobable LIKE, DISLIKE, and unresolved replies. Future query dates are
    30/90 days ahead, with the currently communicated context and no invented
    future events. These are planning assumptions, not learned population facts.
    The contract may require asking; search then selects useful question order.
    """
    name = 'PLANNING_BASELINE'
    ablations = ()
    model_version = 'visible-uniform-response-depth2-v1'

    def __init__(self, depth=2, ask_cost=0.10):
        self.depth = max(1, min(2, int(depth)))
        self.ask_cost = float(ask_cost)
        self.last_plan = None

    def _refresh(self, observation):
        from rl_longitudinal.memory_reconciliation import reconcile
        value = deepcopy(observation)
        value['reconciliation'] = asdict(reconcile(_facts(value), context=value.get('context', {}),
                                                 query_at=value['query_at'], target_food=value.get('target_food')))
        return value

    def _terminal_value(self, observation):
        if _supported(observation):
            return 1.0
        rec = _reconciliation(observation)
        if rec.get('conflicts') or rec.get('uncertain_foods'):
            return 0.30
        return -0.35

    def _questions(self, observation):
        rec = _reconciliation(observation)
        values = list(rec.get('missing_slots', []))
        values += ['current:' + row['food'] for row in rec.get('conflicts', [])]
        values += ['current:' + food for food in rec.get('uncertain_foods', [])]
        if _question(observation):
            values.insert(0, _question(observation))
        return list(dict.fromkeys(key for key in values if _question_available(observation, key)))

    def _branches(self, observation, question):
        """Construct only hypotheses; none of these are actual simulated replies."""
        if question in SLOT_VALUES:
            values = sorted({f.get('scope', {}).get(question) for f in _facts(observation)
                             if f.get('scope', {}).get(question) not in (None, 'ANY')})
            values = values or SLOT_VALUES[question]
            branches = []
            for answer in values:
                branch = deepcopy(observation)
                branch['context'][question] = answer
                branches.append(branch)
        else:
            prefix, _, food = question.partition(':')
            if prefix == 'restriction':
                # An unverified report cannot revoke the explicit restriction.
                return [deepcopy(observation)]
            branches = [deepcopy(observation)]  # Honest uncertainty remains possible.
            related = [f for f in _facts(observation) if f.get('food') == food and f.get('kind') != 'RESTRICTION']
            if not related:
                return branches
            exemplar = related[-1]
            for polarity in ('LIKE', 'DISLIKE'):
                branch = deepcopy(observation)
                evidence = deepcopy(exemplar)
                evidence.update(memory_id='public-model-hypothesis', evidence_id='public-model-reply',
                                polarity=polarity, kind='CLARIFICATION', source='USER_STATEMENT',
                                status='ACTIVE', uncertainty='KNOWN', observed_at=observation['query_at'],
                                recorded_at=observation['query_at'], valid_from=observation['query_at'],
                                valid_to=None, supersedes=[f['memory_id'] for f in related])
                branch['incoming_evidence'].append(evidence)
                branches.append(branch)
        for branch in branches:
            branch['remaining_clarifications'] = max(0, branch.get('remaining_clarifications', 0) - 1)
            branch.setdefault('clarification_history', []).append({'question_key': question, 'status': 'HYPOTHETICAL'})
        return [self._refresh(branch) for branch in branches]

    def _value(self, observation, depth):
        terminal = self._terminal_value(observation)
        if depth <= 0 or _supported(observation):
            return terminal
        candidates = []
        for question in self._questions(observation):
            branches = self._branches(observation, question)
            candidates.append(-self.ask_cost + sum(self._value(b, depth - 1) for b in branches) / len(branches))
        return max([terminal] + candidates)

    def _future_memory_value(self, observation, operation):
        if not observation.get('remaining_sessions'):
            return 0.0
        value = deepcopy(observation)
        incoming = list(value.get('incoming_evidence', []))
        value['incoming_evidence'] = []
        if operation != 'NOOP':
            value['historical_memory'].extend(incoming)
        scores = []
        at = _time(value['query_at'])
        for gap in (30, 90):
            value['query_at'] = (at + timedelta(days=gap)).isoformat()
            scores.append(self._terminal_value(self._refresh(value)))
        # Evidence fidelity is explicit: discarding communicated facts is not an
        # innocuous strategy for making contradictions disappear from memory.
        loss = 0.25 * len(incoming) if operation == 'NOOP' else 0.0
        return sum(scores) / len(scores) - loss

    def select_action(self, observation):
        required = _required(observation)
        required_action = required.get('action', required.get('conversation', required.get('selected_action')))
        operation = memory_operation(observation)
        operations = [operation]
        if observation.get('incoming_evidence') and operation != 'NOOP':
            operations.append('NOOP')
        memory_scores = {op: self._future_memory_value(observation, op) for op in operations}
        # Preserve genuine unresolved evidence even if erasure improves the
        # planner's own terminal support score. Erasure violates the contract.
        chosen_operation = operation
        question_scores = {}
        if required_action == 'CLARIFY':
            for question in self._questions(observation):
                branches = self._branches(observation, question)
                question_scores[question] = -self.ask_cost + sum(
                    self._value(branch, self.depth - 1) + 0.25 * self._future_memory_value(branch, chosen_operation)
                    for branch in branches) / len(branches)
        self.last_plan = {'model': self.model_version, 'depth': self.depth,
                          'question_values': question_scores, 'memory_rollout_values': memory_scores,
                          'memory_choice_constrained_by_evidence_preservation': True,
                          'uses_actual_future_events_or_private_responses': False}
        if question_scores:
            best = max(question_scores, key=lambda key: (question_scores[key], key == _question(observation)))
            return _action('CLARIFY', observation, question_key=best, operation=chosen_operation)
        return _final(observation)


def feature_key(observation):
    """ID/seed/wording-free chronology abstraction, shared unchanged by SFT/Q.

    Exact dates and session indices are omitted, but applicability relative to
    query_at, memory kind, source, context, contradiction, and answered questions
    remain. This is not claimed to be a sufficient POMDP belief statistic.
    """
    foods = canonical_foods(observation)
    at = _time(observation.get('query_at'))
    def food_index(food):
        return foods.index(food) if food in foods else None
    encoded_facts = []
    for fact in _facts(observation):
        start, end = _time(fact.get('valid_from')), _time(fact.get('valid_to'))
        relative = ('FUTURE' if start and at and start > at else
                    'ENDED' if end and at and end <= at else 'CURRENT')
        encoded_facts.append((food_index(fact.get('food', fact.get('food_item'))),
                              fact.get('polarity'), fact.get('kind'), fact.get('source'),
                              tuple(sorted((fact.get('scope') or {}).items())), relative,
                              fact in observation.get('incoming_evidence', [])))
    rec = _reconciliation(observation)
    questions = []
    for item in observation.get('clarification_history', []):
        key = item.get('question_key', '')
        for food in foods:
            key = key.replace(':' + food, ':@' + str(food_index(food)))
        questions.append((key, item.get('status', item.get('resolved', None))))
    return json.dumps({'facts': sorted(encoded_facts, key=repr),
                       'supported': sorted(food_index(f) for f in _supported(observation)),
                       'blocked': sorted(food_index(f) for f in rec.get('forbidden_foods', []) if f in foods),
                       'missing': sorted(rec.get('missing_slots', [])),
                       'context': observation.get('context', {}),
                       'questions': sorted(questions, key=repr),
                       'budget': observation.get('remaining_clarifications', 0),
                       'future_sessions': min(2, observation.get('remaining_sessions', 0)),
                       'user_available': observation.get('user_available', True)},
                      ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _canonical_action(action, observation):
    value = dict(action)
    foods = canonical_foods(observation)
    if value.get('food') in foods:
        value['food'] = '@' + str(foods.index(value['food']))
    if value.get('question_key'):
        for food in foods:
            value['question_key'] = value['question_key'].replace(':' + food, ':@' + str(foods.index(food)))
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def _real_action(encoded, observation):
    value = json.loads(encoded) if isinstance(encoded, str) else dict(encoded)
    foods = canonical_foods(observation)
    if str(value.get('food', '')).startswith('@'):
        value['food'] = foods[int(value['food'][1:])]
    if value.get('question_key') and ':@' in value['question_key']:
        prefix, number = value['question_key'].rsplit(':@', 1)
        value['question_key'] = prefix + ':' + foods[int(number)]
    return value


def action_catalog(observation):
    """Same factored conversational/memory choices for SFT and RL."""
    actions = []
    keys = list(SLOT_VALUES)
    keys += [prefix + ':' + food for prefix in ('current', 'change', 'restriction')
             for food in canonical_foods(observation) if any(
                 f.get('food', f.get('food_item')) == food for f in _facts(observation))]
    for operation in MEMORY_OPERATIONS:
        for food in _safe_foods(observation):
            actions.append(_action('ANSWER', observation, food=food, operation=operation))
        for conversation in ('ABSTAIN', 'PRESERVE_CONFLICT', 'CONDITIONAL_ANSWER'):
            actions.append(_action(conversation, observation, operation=operation))
        for key in keys:
            if _question_available(observation, key):
                actions.append(_action('CLARIFY', observation, question_key=key, operation=operation))
    return list(dict.fromkeys(_canonical_action(action, observation) for action in actions))


class SupervisedLongitudinal:
    name = 'SUPERVISED_LONGITUDINAL'
    ablations = ()

    def __init__(self, counts=None):
        self.counts = counts or {}
        self.fallback = ClarificationRuleBased()
        self.reset_usage()

    def reset_usage(self):
        self.seen_decisions = 0
        self.fallback_decisions = 0

    def usage_summary(self):
        total = self.seen_decisions + self.fallback_decisions
        return {'decisions': total, 'learned_table_decisions': self.seen_decisions,
                'rule_fallback_decisions': self.fallback_decisions,
                'rule_fallback_rate': self.fallback_decisions / total if total else None}

    def select_action(self, observation):
        counts = self.counts.get(feature_key(observation), {})
        actions = action_catalog(observation)
        present = [a for a in actions if a in counts]
        if not present:
            self.fallback_decisions += 1
            return self.fallback.select_action(observation)
        self.seen_decisions += 1
        action = max(present, key=lambda a: (counts[a], -actions.index(a)))
        return _real_action(action, observation)

    def to_dict(self):
        return {'policy_type': 'longitudinal_behavior_cloning', 'feature_version': 1,
                'counts': self.counts, 'unseen_state_fallback': 'CLARIFICATION_RULE_BASED'}

    @classmethod
    def from_dict(cls, value):
        if value.get('policy_type') != 'longitudinal_behavior_cloning':
            raise ValueError('Unsupported supervised checkpoint')
        return cls(value['counts'])


def train_supervised(scenarios, *, env_factory=None, condition='ORACLE_STRUCTURED',
                     output_dir=None, teacher=None):
    if env_factory is None:
        from rl_longitudinal.longitudinal_environment import LongitudinalEnv
        env_factory = lambda scenario: LongitudinalEnv(scenario, condition=condition)
    teacher = teacher or PlanningBaseline()
    counts = defaultdict(Counter)
    examples = 0
    for scenario in scenarios:
        env = env_factory(scenario)
        try:
            obs, _ = env.reset()
            for _ in range(100):
                action = teacher.select_action(obs)
                counts[feature_key(obs)][_canonical_action(action, obs)] += 1
                examples += 1
                obs, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    break
            else:
                raise RuntimeError('Longitudinal environment did not terminate within 100 steps')
        finally:
            if hasattr(env, 'close'):
                env.close()
    policy = SupervisedLongitudinal({key: dict(value) for key, value in counts.items()})
    metrics = {'examples': examples, 'unique_observation_keys': len(counts), 'teacher': teacher.name,
               'uses_private_labels_or_rewards': False, 'fallback': policy.fallback.name,
               'model': 'tabular maximum-likelihood behavior cloning; same features/actions as Q'}
    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=False)
        (path / 'supervised_policy.json').write_text(json.dumps(policy.to_dict(), ensure_ascii=False, indent=2))
        (path / 'supervised_training.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    return {'policy': policy, **metrics}


def make_baselines(supervised=None):
    result = [LastWriteWins(), HistoricalRuleBased(), ClarificationRuleBased(), PlanningBaseline()]
    if supervised is not None:
        result.append(supervised)
    return {policy.name: policy for policy in result}
