"""Append-only temporal facts on the repository's real SQLite Store.

Intervals are UTC and half-open [valid_from, valid_to). Naive timestamps are
rejected; ISO date-only values mean 00:00 UTC. Historical query time and time at
which the agent knows evidence are deliberately independent.
"""
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import json

from memory.store import Store


def instant(value):
    if isinstance(value, datetime):
        result = value
    else:
        text = str(value)
        result = datetime.fromisoformat(text.replace('Z', '+00:00'))
        if len(text) == 10:
            result = result.replace(tzinfo=timezone.utc)
    if result.tzinfo is None:
        raise ValueError('Timestamp must include a timezone (or be an ISO date)')
    return result.astimezone(timezone.utc)


def utc(value):
    return instant(value).isoformat().replace('+00:00', 'Z')


@dataclass(frozen=True)
class TemporalFact:
    memory_id: str
    user_id: str
    food: str
    polarity: str
    scope: dict = field(default_factory=lambda: {'day': None, 'meal': None, 'company': None, 'place': None})
    kind: str = 'STABLE_PREFERENCE'
    observed_at: str | None = None
    recorded_at: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    source: str = 'USER_STATEMENT'
    evidence_id: str = ''
    status: str = 'ACTIVE'
    supersedes: tuple = ()
    uncertainty: str = 'KNOWN'
    text: str = ''

    def __post_init__(self):
        if not self.memory_id or not self.user_id or not self.evidence_id or not self.food:
            raise ValueError('Fact IDs, user, food and original evidence are required')
        if self.observed_at is None or self.recorded_at is None:
            raise ValueError('Observed and recorded timestamps must be explicitly supplied')
        if self.polarity not in ('LIKE', 'DISLIKE', 'NEUTRAL', 'FORBID'):
            raise ValueError('Unknown polarity')
        if self.kind not in ('STABLE_PREFERENCE', 'EXPLICIT_CHANGE', 'TEMPORARY_EXCEPTION', 'CONTEXTUAL_PREFERENCE',
                             'OBSERVED_BEHAVIOR', 'AMBIGUOUS', 'UNRESOLVED', 'RESTRICTION', 'CLARIFICATION'):
            raise ValueError('Unknown temporal kind')
        if self.status not in ('ACTIVE', 'UNRESOLVED', 'INACTIVE'):
            raise ValueError('Unknown status')
        if self.uncertainty not in ('KNOWN', 'UNKNOWN', 'UNRESOLVED'):
            raise ValueError('Unknown uncertainty')
        scope = {'day': None, 'meal': None, 'company': None, 'place': None}
        if set(self.scope) - set(scope):
            raise ValueError('Unknown scope dimension')
        scope.update(self.scope)
        if scope['day'] not in (None, 'ANY', 'WEEKDAY', 'WEEKEND'):
            raise ValueError('Unknown day partition')
        if scope['meal'] not in (None, 'ANY', 'BREAKFAST', 'LUNCH', 'DINNER', 'SNACK'):
            raise ValueError('Unknown meal')
        object.__setattr__(self, 'scope', scope)
        for key in ('observed_at', 'recorded_at', 'valid_from', 'valid_to'):
            value = getattr(self, key)
            if value is not None:
                object.__setattr__(self, key, utc(value))
        if instant(self.recorded_at) < instant(self.observed_at):
            raise ValueError('Cannot record evidence before it was observed')
        if self.valid_from and self.valid_to and instant(self.valid_from) >= instant(self.valid_to):
            raise ValueError('Invalid half-open validity interval')
        if self.kind == 'TEMPORARY_EXCEPTION' and not self.valid_to:
            raise ValueError('Temporary exceptions require an explicit end')
        object.__setattr__(self, 'supersedes', tuple(self.supersedes or ()))
        if self.kind in ('EXPLICIT_CHANGE', 'CLARIFICATION') and self.source == 'ORDER_HISTORY':
            raise ValueError('Observed orders cannot declare an explicit preference replacement')
        if self.supersedes and self.kind not in ('EXPLICIT_CHANGE', 'CLARIFICATION'):
            raise ValueError('Only explicit communicated changes can supersede facts')


def fact_from(value):
    if isinstance(value, TemporalFact):
        # Copy mutable scope to prevent caller mutation after insertion.
        return TemporalFact(**asdict(value))
    data = asdict(value) if is_dataclass(value) else dict(value)
    if 'food_item' in data:
        scope = {'day': data.get('day_scope'), 'meal': data.get('meal_scope'),
                 'company': data.get('context', {}).get('company'), 'place': data.get('context', {}).get('place')}
        kind = ('RESTRICTION' if 'restriction_type' in data else 'TEMPORARY_EXCEPTION' if data.get('temporary')
                else 'EXPLICIT_CHANGE' if data.get('explicit_update') else 'OBSERVED_BEHAVIOR'
                if data.get('source') == 'ORDER_HISTORY' else 'STABLE_PREFERENCE')
        return TemporalFact(memory_id=data.get('preference_id', data.get('restriction_id')), user_id=data['user_id'],
                            food=data['food_item'], polarity=data.get('polarity', 'FORBID'), scope=scope, kind=kind,
                            observed_at=data['timestamp'], recorded_at=data['timestamp'], valid_from=data.get('valid_from'),
                            valid_to=data.get('valid_to'), source=data.get('source', 'USER_STATEMENT'),
                            evidence_id=data['evidence_id'], supersedes=tuple(data.get('supersedes', ())))
    return TemporalFact(**data)


@dataclass(frozen=True)
class Retrieval:
    facts: tuple
    trace_id: int
    query_at: str
    known_at: str
    context: dict
    target_food: str | None = None


class TemporalMemoryStore(Store):
    """Extends, rather than replaces, legacy SQLite schema and owner protection."""
    def __init__(self, path, user_id):
        super().__init__(path, user_id)
        self.user_id = user_id
        self.conn.executescript('''
        CREATE TABLE IF NOT EXISTS temporal_facts(memory_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS temporal_operations(seq INTEGER PRIMARY KEY, operation TEXT NOT NULL,
            at TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS temporal_retrievals(seq INTEGER PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS temporal_clarifications(seq INTEGER PRIMARY KEY, question_key TEXT NOT NULL,
            at TEXT NOT NULL, response TEXT, status TEXT NOT NULL, memory_id TEXT);
        CREATE TABLE IF NOT EXISTS temporal_clarification_values(clarification_seq INTEGER PRIMARY KEY,
            value TEXT NOT NULL, query_at TEXT NOT NULL);
        ''')
        self.conn.commit()

    def facts(self):
        return [fact_from(json.loads(r[0])) for r in self.conn.execute('SELECT payload FROM temporal_facts ORDER BY memory_id')]

    def add_fact(self, fact, operation='ADD', reason='Communicated evidence', at=None):
        fact = fact_from(fact)
        if fact.user_id != self.user_id:
            raise ValueError('Fact belongs to a different user')
        if operation not in ('ADD', 'UPDATE_WITH_HISTORY', 'PRESERVE_CONFLICT'):
            raise ValueError('Unsupported append operation')
        if fact.supersedes and operation != 'UPDATE_WITH_HISTORY':
            raise ValueError('Supersession requires the explicit UPDATE_WITH_HISTORY operation')
        previous = {f.memory_id: f for f in self.facts()}
        for identifier in fact.supersedes:
            if identifier not in previous:
                raise ValueError('Superseded fact does not exist')
            old = previous[identifier]
            if old.kind == 'RESTRICTION' or old.polarity == 'FORBID':
                raise ValueError('A preference update cannot retire a safety restriction')
            if old.food != fact.food or old.scope != fact.scope or not fact.valid_from:
                raise ValueError('Explicit supersession requires same food/scope and known onset')
            if instant(fact.valid_from) < instant(old.valid_from or old.observed_at):
                raise ValueError('An update cannot close a fact before its known onset')
        serialized = json.dumps(asdict(fact), sort_keys=True, ensure_ascii=False)
        old = self.conn.execute('SELECT payload FROM temporal_facts WHERE memory_id=?', (fact.memory_id,)).fetchone()
        if old:
            if old[0] != serialized:
                raise ValueError('Non-idempotent immutable fact collision')
            return fact.memory_id
        with self.conn:
            self.conn.execute('INSERT INTO temporal_facts VALUES (?,?)', (fact.memory_id, serialized))
            self._operation(operation, at or fact.recorded_at, reason, before=[asdict(previous[i]) for i in fact.supersedes],
                            evidence_id=fact.evidence_id, linked_ids=list(fact.supersedes), after=asdict(fact), uncertainty=fact.uncertainty)
        return fact.memory_id

    def _operation(self, operation, at, reason, **details):
        payload = {'reason': reason, **details}
        self.conn.execute('INSERT INTO temporal_operations(operation,at,payload) VALUES (?,?,?)',
                          (operation, utc(at), json.dumps(payload, sort_keys=True, ensure_ascii=False)))

    def record_noop(self, at, reason, evidence_id='', linked_ids=()):
        with self.conn:
            self._operation('NOOP', at, reason, evidence_id=evidence_id, linked_ids=list(linked_ids), before=None, after=None)

    def retrieve(self, query_at, known_at=None, context=None, food=None):
        when, known = utc(query_at), utc(known_at or query_at)
        facts = tuple(f for f in self.facts() if instant(f.recorded_at) <= instant(known)
                      and instant(f.observed_at) <= instant(known)
                      and (food is None or f.food == food or f.kind == 'RESTRICTION' or f.polarity == 'FORBID'))
        payload = {'query_at': when, 'known_at': known, 'context': dict(context or {}), 'target_food': food,
                   'retrieved_ids': [f.memory_id for f in facts], 'evidence_ids': sorted({f.evidence_id for f in facts}),
                   'empty': not facts}
        with self.conn:
            cursor = self.conn.execute('INSERT INTO temporal_retrievals(payload) VALUES (?)', (json.dumps(payload, sort_keys=True),))
        return Retrieval(facts, cursor.lastrowid, when, known, dict(context or {}), food)

    def complete_trace(self, trace_id, **details):
        row = self.conn.execute('SELECT payload FROM temporal_retrievals WHERE seq=?', (trace_id,)).fetchone()
        if row is None:
            raise ValueError('Decision must refer to an actual historical retrieval')
        payload = json.loads(row[0]); payload.update(details)
        with self.conn:
            self.conn.execute('UPDATE temporal_retrievals SET payload=? WHERE seq=?', (json.dumps(payload, sort_keys=True), trace_id))

    def retrieval_traces(self):
        return [dict(trace_id=r[0], **json.loads(r[1])) for r in self.conn.execute('SELECT seq,payload FROM temporal_retrievals ORDER BY seq')]

    def operations(self):
        return [dict(seq=r[0], operation=r[1], at=r[2], **json.loads(r[3]))
                for r in self.conn.execute('SELECT seq,operation,at,payload FROM temporal_operations ORDER BY seq')]

    def record_clarification(self, question_key, at, response=None, status='UNRESOLVED', resolved_fact=None,
                             value=None, query_at=None):
        if status not in ('RESOLVED', 'UNRESOLVED', 'REFUSED'):
            raise ValueError('Invalid clarification status')
        if resolved_fact is not None and status != 'RESOLVED':
            raise ValueError('An unanswered or uncertain clarification cannot update stable memory')
        if status == 'RESOLVED' and not response and resolved_fact is None:
            raise ValueError('Resolution requires communicated evidence')
        if value is not None:
            if status != 'RESOLVED' or question_key not in ('day', 'meal', 'company', 'place'):
                raise ValueError('Only resolved context questions have reusable context values')
            if question_key == 'day' and value not in ('WEEKDAY', 'WEEKEND', 'ANY'):
                raise ValueError('Invalid day clarification')
            if question_key == 'meal' and value not in ('BREAKFAST', 'LUNCH', 'DINNER', 'SNACK', 'ANY'):
                raise ValueError('Invalid meal clarification')
            if not isinstance(value, str) or not value:
                raise ValueError('Context answer must be an explicit string')
        memory_id = self.add_fact(resolved_fact, operation='UPDATE_WITH_HISTORY', reason='Explicit clarification') if resolved_fact else None
        with self.conn:
            cursor = self.conn.execute('INSERT INTO temporal_clarifications(question_key,at,response,status,memory_id) VALUES (?,?,?,?,?)',
                              (question_key, utc(at), response, status, memory_id))
            if value is not None:
                if status != 'RESOLVED':
                    raise ValueError('An unresolved reply cannot supply a resolved context value')
                self.conn.execute('INSERT INTO temporal_clarification_values VALUES (?,?,?)',
                                  (cursor.lastrowid, json.dumps(value), utc(query_at or at)))
        return memory_id

    def clarification_history(self, known_at=None):
        rows = [dict(r) for r in self.conn.execute('SELECT * FROM temporal_clarifications ORDER BY at,seq')]
        extra = {r[0]: (json.loads(r[1]), r[2]) for r in self.conn.execute('SELECT * FROM temporal_clarification_values')}
        for row in rows:
            if row['seq'] in extra:
                row['value'], row['query_at'] = extra[row['seq']]
        return [r for r in rows if known_at is None or instant(r['at']) <= instant(known_at)]
