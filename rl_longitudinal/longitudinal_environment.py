"""SQLite-backed multi-session interaction with private evaluation boundaries.

The archive retains raw evidence. The managed fact index is what memory retrieval
uses. Index-loss experiments are explicitly invariant-breaking ablations, never
evidence that a conforming agent can ignore historical messages.
"""
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile

from simulator.preferences import extract_preferences, parse_context
from .temporal_memory import TemporalFact, TemporalMemoryStore, fact_from, instant
from .clarification_policy import decide
from .memory_actions import apply_memory_action
from .scenario_generator import generate_episodes, FOODS
from .reward import RewardConfig, score_decision

CONVERSATIONS = ('ANSWER', 'CLARIFY', 'CONDITIONAL_ANSWER', 'PRESERVE_CONFLICT', 'ABSTAIN')
MEMORY_OPERATIONS = ('ADD', 'UPDATE_WITH_HISTORY', 'NOOP', 'PRESERVE_CONFLICT')


class _DecisionView:
    """A real historical retrieval plus current communicated, not-yet-indexed facts."""
    def __init__(self, store, pending):
        self.store, self.pending = store, pending

    def retrieve(self, query_at, known_at=None, context=None, food=None):
        r = self.store.retrieve(query_at, known_at, context, food)
        facts = {f.memory_id: f for f in r.facts}
        for f in self.pending:
            if instant(f.recorded_at) <= instant(known_at or query_at) and (food is None or f.food == food or f.kind == 'RESTRICTION'):
                facts[f.memory_id] = f
        result = replace(r, facts=tuple(facts.values()))
        self.store.complete_trace(r.trace_id, indexed_retrieved_ids=[f.memory_id for f in r.facts],
                                  incoming_consulted_ids=[f.memory_id for f in self.pending],
                                  retrieved_ids=[f.memory_id for f in result.facts],
                                  consulted_archive_evidence_ids=[row[0] for row in self.store.conn.execute(
                                      'SELECT evidence_id FROM longitudinal_evidence_archive WHERE recorded_at<=? ORDER BY evidence_id',
                                      (known_at or query_at,))])
        return result

    def __getattr__(self, key):
        return getattr(self.store, key)


class LongitudinalEnv:
    def __init__(self, scenario=None, condition='ORACLE_STRUCTURED', db_path=None,
                 enforce_invariants=True, max_clarifications_per_session=2,
                 reward_config=None, no_persistence=False, no_clarification=False,
                 no_persist_clarifications=False):
        if condition not in ('ORACLE_STRUCTURED', 'PIPELINE'):
            raise ValueError('Unknown condition')
        self._initial = deepcopy(scenario)
        self.condition, self.db_path = condition, str(db_path) if db_path else None
        self.enforce_invariants = enforce_invariants
        self.max_clarifications = max_clarifications_per_session
        self.reward_config = reward_config or RewardConfig()
        self.no_persistence, self.no_clarification = no_persistence, no_clarification
        self.no_persist_clarifications = no_persist_clarifications
        self.store = None

    def reset(self, seed=42, scenario_id=None):
        self.close()
        self._scenario = deepcopy(self._initial or generate_episodes(1, seed=seed, split='test')[0])
        if scenario_id is not None and scenario_id != self._scenario['episode_id']:
            raise ValueError('Unknown scenario_id')
        self._validate_private_scenario()
        if self.db_path:
            path = Path(self.db_path)
            if path.exists():
                raise FileExistsError('Reset requires a fresh database; no historical file is overwritten')
            path.parent.mkdir(parents=True, exist_ok=True)
            self._path = str(path)
        else:
            self._temporary = tempfile.TemporaryDirectory(prefix='food-longitudinal-')
            self._path = str(Path(self._temporary.name) / 'memory.sqlite')
        self.store = TemporalMemoryStore(self._path, self._scenario['user_id'])
        self.store.conn.execute('CREATE TABLE IF NOT EXISTS longitudinal_evidence_archive(evidence_id TEXT PRIMARY KEY, recorded_at TEXT, payload TEXT)')
        self.store.conn.commit()
        self._index, self._return, self._steps, self._sessions = 0, 0., [], []
        self._terminated = False
        self._begin_session()
        return self._observe(), {}

    def _validate_private_scenario(self):
        sessions = self._scenario['sessions']
        if not 3 <= len(sessions) <= 10:
            raise ValueError('An episode requires 3–10 sessions')
        last = None
        for ses in sessions:
            at = instant(ses['at'])
            if last is not None and at <= last:
                raise ValueError('Sessions must advance in time')
            last = at
            for f in ses['incoming']:
                if instant(f['recorded_at']) > at or instant(f['observed_at']) > at:
                    raise ValueError('Future evidence in present session')
                fact_from(f)

    def _begin_session(self):
        self._session = self._scenario['sessions'][self._index]
        self._context = deepcopy(self._session['context'])
        self._pending = self._extract(self._session)
        self._asked, self._question_count = [], 0
        self._session_steps = []
        self._session_return = 0.
        self._memory_before = [asdict(f) for f in self.store.facts()]
        for event in self._session.get('events', []):
            self.store.conn.execute('INSERT OR IGNORE INTO longitudinal_evidence_archive VALUES (?,?,?)',
                (event['message_id'], self._session['at'], json.dumps(event, ensure_ascii=False, sort_keys=True)))
            self.store.add({**event, 'session_id': f'session_{self._index}', 'role': 'user',
                            'channel': 'conversation', 'source': event.get('source', 'USER_STATEMENT')})
        self.store.conn.commit()

    def _extract(self, session):
        if self.condition == 'ORACLE_STRUCTURED':
            return [fact_from(f) for f in session['incoming']]
        values = []
        for event in session.get('events', []):
            parsed = extract_preferences(event, self._scenario['user_id'])
            for f in (*parsed.preferences, *parsed.restrictions):
                d = asdict(f)
                if d.get('temporary') and d.get('valid_to') and instant(d['valid_to']) <= instant(d.get('valid_from') or d['timestamp']):
                    d['valid_to'] = (instant(d['valid_to']) + timedelta(days=1)).isoformat()
                # Extraction timestamps describe observation; recording is now.
                value = fact_from(d)
                scope = dict(value.scope)
                # Narrow adapter for an explicit universal phrase, not a default.
                if 'sin importar la compañía ni el lugar' in event.get('text','').lower():
                    scope.update(company='ANY',place='ANY')
                value = replace(value, recorded_at=session['at'],scope=scope)
                if value.kind == 'TEMPORARY_EXCEPTION' and value.valid_to and instant(value.valid_to) <= instant(value.valid_from or value.observed_at):
                    value = replace(value, valid_to=(instant(value.valid_to) + timedelta(days=1)).isoformat())
                values.append(value)
        return values

    def _required(self):
        decision = decide(_DecisionView(self.store, self._pending),
            self._session['query_at'], context=self._context,
            target_food=self._session.get('target_food'),
            user_available=self._session.get('user_available', True) and not self.no_clarification,
            known_at=self._session['at'])
        self._required_decision = decision
        return decision

    @staticmethod
    def _public_fact(f):
        d = asdict(f)
        d['user_id'] = 'user'
        return d

    def _observe(self):
        if self._terminated:
            return {'terminated': True}
        d = self._required()
        rec = asdict(d.reconciliation)
        rec.pop('history', None)  # Facts are separately whitelisted below.
        # No private scenario identifier, category, seed, expected label or reply.
        observation = dict(query=self._session['query'], query_at=self._session['query_at'],
            current_at=self._session['at'], context=deepcopy(self._context),
            target_food=self._session.get('target_food'),
            historical_memory=[self._public_fact(f) for f in self.store.facts()
                               if instant(f.recorded_at) <= instant(self._session['at'])],
            incoming_evidence=[self._public_fact(f) for f in self._pending],
            clarification_history=deepcopy(self.store.clarification_history(self._session['at'])),
            session_index=self._index, remaining_sessions=len(self._scenario['sessions']) - self._index,
            remaining_clarifications=max(0, self.max_clarifications-self._question_count),
            user_available=self._session.get('user_available', True),
            required_decision={k: deepcopy(v) for k, v in asdict(d).items() if k != 'reconciliation'},
            reconciliation=rec)
        return observation

    def _parse_action(self, action):
        if isinstance(action, str):
            conv, _, value = action.partition(':')
            action = {'conversation': conv, 'food': value if conv == 'ANSWER' else None,
                      'question_key': value if conv == 'CLARIFY' else None}
        if not isinstance(action, dict):
            raise ValueError('Action must be a mapping or semantic action string')
        result = deepcopy(action)
        result.setdefault('memory_operation', 'ADD' if self._pending else 'NOOP')
        result.setdefault('question_key', result.get('question'))
        if result.get('conversation') not in CONVERSATIONS or result['memory_operation'] not in MEMORY_OPERATIONS:
            raise ValueError('Unknown action')
        if result.get('food') is not None and result['food'] not in FOODS:
            raise ValueError('Food outside declared catalogue')
        return result

    def _memory_action(self, operation):
        violations = []
        actual_operations = []
        for f in self._pending:
            actual = operation
            if self.enforce_invariants and operation == 'NOOP':
                actual = 'PRESERVE_CONFLICT' if f.kind in ('AMBIGUOUS', 'OBSERVED_BEHAVIOR', 'UNRESOLVED') else 'ADD'
                violations.append('NOOP_WOULD_OMIT_NEW_EVIDENCE')
            if self.enforce_invariants and f.supersedes and actual != 'UPDATE_WITH_HISTORY':
                actual = 'UPDATE_WITH_HISTORY'
                violations.append('EXPLICIT_CHANGE_REQUIRES_HISTORICAL_UPDATE')
            if actual == 'UPDATE_WITH_HISTORY' and f.kind not in ('EXPLICIT_CHANGE', 'CLARIFICATION'):
                violations.append('UNJUSTIFIED_STABLE_UPDATE')
                actual = 'PRESERVE_CONFLICT' if self.enforce_invariants else 'NOOP'
            if self.no_persist_clarifications and f.memory_id.startswith('clar_'):
                actual = 'NOOP'
            if self.no_persistence:
                actual = 'NOOP'
            # A dropped predecessor in an explicit ablation must not crash;
            # retain the new evidence without claiming a linked replacement.
            known = {old.memory_id for old in self.store.facts()}
            if set(f.supersedes) - known:
                f = replace(f, supersedes=tuple(m for m in f.supersedes if m in known))
            if actual in ('ADD', 'PRESERVE_CONFLICT') and f.supersedes:
                f = replace(f, supersedes=())
            apply_memory_action(self.store, actual, f, at=self._session['at'],
                                reason='Policy-managed memory; raw evidence retained in archive')
            actual_operations.append(dict(memory_id=f.memory_id, evidence_id=f.evidence_id,
                                          proposed=operation, executed=actual))
        if not self._pending:
            self.store.record_noop(self._session['at'], 'No unindexed incoming evidence')
        self._pending = []
        return violations, actual_operations

    def step(self, action):
        if self._terminated:
            raise RuntimeError('Episode already terminated')
        proposed = self._parse_action(action)
        required = self._required()  # MEM-01: even an invalid proposal consults SQLite.
        executed = deepcopy(proposed)
        shield_reasons = []
        if self.enforce_invariants:
            if required.action == 'CLARIFY' and proposed['conversation'] != 'CLARIFY':
                executed.update(conversation='CLARIFY', food=None, question_key=required.question_key)
                shield_reasons.append('MATERIAL_UNCERTAINTY')
            elif proposed['conversation'] == 'CLARIFY':
                if required.action != 'CLARIFY':
                    executed.update(conversation=required.action, food=required.foods[0] if required.foods else None, question_key=None)
                    shield_reasons.append('IRRELEVANT_OR_REPEATED_QUESTION')
                elif proposed.get('question_key') != required.question_key:
                    executed['question_key'] = required.question_key
                    shield_reasons.append('IRRELEVANT_QUESTION_REPLACED')
            if executed['conversation'] == 'ANSWER' and executed.get('food') not in required.reconciliation.safe_foods:
                executed.update(conversation=required.action if required.action != 'ANSWER' else 'ABSTAIN', food=None)
                shield_reasons.append('UNSUPPORTED_COMMITMENT')
        # Safety is never disabled by recency/history/clarification ablations.
        if executed['conversation'] == 'ANSWER' and executed.get('food') in required.reconciliation.forbidden_foods:
            executed.update(conversation='PRESERVE_CONFLICT', food=None)
            shield_reasons.append('RESTRICTION_SHIELD')
        pending_before = [asdict(f) for f in self._pending]
        memory_violations, actual_memory_operations = self._memory_action(executed['memory_operation'])
        reward = -self.reward_config.unjustified_update_penalty * memory_violations.count('UNJUSTIFIED_STABLE_UPDATE')
        attempted_safety_penalty = 0.
        public_forbidden = required.reconciliation.forbidden_foods
        if (proposed['conversation'] == 'ANSWER' and proposed.get('food') in public_forbidden
                and not (executed['conversation'] == 'ANSWER' and executed.get('food') in public_forbidden)):
            # The shield prevents execution; the unsafe proposal still incurs
            # its declared cost, exactly once rather than receiving shield credit.
            attempted_safety_penalty = self.reward_config.unsafe_attempt_penalty
            reward -= attempted_safety_penalty
        trace = dict(session_index=self._index, at=self._session['at'], query_at=self._session['query_at'],
                     query=self._session['query'], proposed_action=proposed, executed_action=executed,
                     required_action=required.action, shield_reasons=shield_reasons,
                     memory_violations=memory_violations, trace_id=required.trace_id,
                     attempted_safety_penalty=attempted_safety_penalty,
                     actual_memory_operations=actual_memory_operations,
                     consulted_memory_ids=list(required.retrieved_ids),
                     applicable_memory_ids=list(required.applicable_ids), evidence_ids=list(required.evidence_ids),
                     pending_before=pending_before)
        conv = executed['conversation']
        if conv == 'CLARIFY' and self._question_count < self.max_clarifications and not self.no_clarification:
            key = executed.get('question_key')
            previous = [row for row in self.store.clarification_history(self._session['at']) if row['question_key'] == key]
            if key in ('day', 'meal', 'company', 'place'):
                previous = [row for row in previous if row['at'][:10] == self._session['at'][:10]]
            elif previous:
                last = max(row['at'] for row in previous)
                food = key.partition(':')[2] if key else None
                if any(f.food == food and f.observed_at > last for f in self.store.facts()):
                    previous = []
            repeated = bool(previous)
            response = deepcopy(self._session.get('responses', {}).get(key,
                dict(text='No puedo aclararlo en este momento.', status='UNRESOLVED', value=None, fact=None)))
            self._question_count += 1
            self._asked.append(key)
            observed_value = response.get('value') if key in ('day','meal','company','place') else None
            if self.condition == 'PIPELINE' and key in ('day','meal','company','place'):
                parsed = parse_context(response['text'], self._session['at'])
                observed_value = {'day':parsed.day_scope,'meal':parsed.meal_scope,**parsed.context}.get(key)
            self.store.record_clarification(key or 'unspecified', self._session['at'], response['text'], response['status'],
                                            value=observed_value, query_at=self._session['query_at'])
            if response.get('fact'):
                f = deepcopy(response['fact'])
                f['memory_id'] = 'clar_' + f['memory_id']
                f['evidence_id'] = 'clar_' + f['evidence_id']
                event = dict(message_id=f['evidence_id'], text=response['text'], at=self._session['at'],
                             session_id=f'session_{self._index}', role='user', channel='conversation', source='clarification')
                if self.condition == 'ORACLE_STRUCTURED':
                    self._pending.append(fact_from(f))
                else:
                    self._pending.extend(self._extract(dict(events=[event], at=self._session['at'])))
                self.store.add(event)
                self.store.conn.execute('INSERT OR IGNORE INTO longitudinal_evidence_archive VALUES (?,?,?)',
                    (event['message_id'], self._session['at'], json.dumps(event, ensure_ascii=False, sort_keys=True)))
                self.store.conn.commit()
            if key in ('day', 'meal', 'company', 'place') and observed_value is not None:
                self._context[key] = observed_value
            reward -= self.reward_config.clarification_cost
            trace.update(response=response['text'], response_status=response['status'], repeated_question=repeated,
                         necessary_question=bool(key in self._session['gold'].get('clarification_keys', ()) and not previous),
                         terminal=False, correct=None)
            self._record_step(trace, reward)
            return self._observe(), reward, False, False, {}
        if conv == 'CLARIFY':
            executed.update(conversation='PRESERVE_CONFLICT', food=None)
            trace['executed_action'] = executed
            trace['shield_reasons'].append('QUESTION_BUDGET_EXHAUSTED')
        visible = [asdict(f) for f in self.store.facts()]
        current_evidence = {f['memory_id']: f for f in visible}
        current_evidence.update({f['memory_id']: f for f in pending_before})
        quality = score_decision(executed, self._session['gold'], list(current_evidence.values()),
                                 self._session['query_at'], self._context, self.reward_config)
        reward += quality['reward']
        trace.update(quality, terminal=True, memory_after=visible)
        self._record_step(trace, reward)
        session_result = dict(session_index=self._index, date=self._session['at'], query_at=self._session['query_at'],
            query=self._session['query'], expected=deepcopy(self._session['gold']), action=executed,
            correct=quality['correct'], grounded=quality['grounded'], restriction_violation=quality['restriction_violation'],
            question_count=self._question_count, return_=self._session_return,
            memory_before=self._memory_before, memory_after=visible, steps=deepcopy(self._session_steps),
            future_error=self._index > 0 and not quality['correct'],
            history_retrieved_ids=list(required.retrieved_ids), consultation_trace_id=required.trace_id)
        session_result['return'] = session_result.pop('return_')
        steps = self._session_steps
        asked = [s for s in steps if s['executed_action']['conversation'] == 'CLARIFY']
        expected_ids = set(self._session['gold'].get('relevant_memory_ids', ()))
        retrieved_ids = set(required.retrieved_ids)
        source_or_conflict = self._scenario['family'] in ('SOURCE_CHANGE','AMBIGUOUS_BEHAVIOR','RESTRICTION','UNANSWERED') and self._index > 0
        operation_evaluations = []
        proposed_operation_evaluations = []
        for s in steps:
            for f in s['pending_before']:
                allowed = {'UPDATE_WITH_HISTORY'} if f.get('supersedes') else {'ADD','UPDATE_WITH_HISTORY'} if f['kind'] in ('EXPLICIT_CHANGE','CLARIFICATION') else {'ADD','PRESERVE_CONFLICT'}
                actual = next(op['executed'] for op in s['actual_memory_operations'] if op['memory_id']==f['memory_id'])
                operation_evaluations.append(actual in allowed)
                proposed_operation_evaluations.append(s['proposed_action']['memory_operation'] in allowed)
        session_result.update(asks=len(asked), unnecessary_asks=sum(not s.get('necessary_question') for s in asked),
            repeated_asks=sum(bool(s.get('repeated_question')) for s in asked),
            restriction_violations=int(quality['restriction_violation']),
            attempted_restriction_violations=sum(s['proposed_action'].get('food') in self._session['gold'].get('restrictions', ()) for s in steps),
            shield_interventions=sum(bool(s['shield_reasons'] or s['memory_violations']) for s in steps),
            actions_proposed=len(steps), historical_relevant=len(expected_ids), historical_retrieved=len(expected_ids & retrieved_ids),
            needed_asks=int(self._session['gold'].get('needs_clarification',False)),
            needed_asks_performed=int(any(s.get('necessary_question') for s in asked)),
            unjustified_updates=sum('UNJUSTIFIED_STABLE_UPDATE' in s['memory_violations'] for s in steps),
            memory_updates=sum(s['executed_action']['memory_operation']=='UPDATE_WITH_HISTORY' for s in steps),
            evaluated_memory_operations=len(operation_evaluations), correct_memory_operations=sum(operation_evaluations),
            proposed_correct_memory_operations=sum(proposed_operation_evaluations),
            future_sessions=int(self._index>0), propagated_future_errors=int(self._index>0 and not quality['correct'] and not expected_ids <= retrieved_ids),
            relevant_conflicts=int(source_or_conflict and self._session.get('target_food') is not None),
            conflicts_detected=int(source_or_conflict and bool(required.reconciliation.conflicts or required.reconciliation.uncertain_foods or required.reconciliation.forbidden_foods)))
        self._sessions.append(session_result)
        self.store.close()
        self.store = TemporalMemoryStore(self._path, self._scenario['user_id'])
        self._index += 1
        self._terminated = self._index >= len(self._scenario['sessions'])
        if not self._terminated:
            self._begin_session()
        return self._observe(), reward, self._terminated, False, {}

    def _record_step(self, trace, reward):
        trace['reward'] = reward
        self._session_return += reward
        self._return += reward
        self._steps.append(deepcopy(trace))
        self._session_steps.append(deepcopy(trace))

    def diagnostic_metrics(self):
        """Private evaluator endpoint; never pass this object to a policy."""
        result = dict(episode_id=self._scenario['episode_id'], user_id=self._scenario['user_id'],
            family=self._scenario['family'], temporal_family=self._scenario.get('temporal_family'),
            sessions=deepcopy(self._sessions), steps=deepcopy(self._steps),
            return_=self._return, terminated=self._terminated, condition=self.condition,
            operations=self.store.operations(), retrieval_traces=self.store.retrieval_traces(),
            archive_evidence_ids=[row[0] for row in self.store.conn.execute('SELECT evidence_id FROM longitudinal_evidence_archive ORDER BY evidence_id')],
            sqlite_path=self._path, invariant_enforcement=self.enforce_invariants,
            ablations=dict(no_persistence=self.no_persistence,no_clarification=self.no_clarification,
                           no_persist_clarifications=self.no_persist_clarifications))
        result['return'] = result.pop('return_')
        return result

    def close(self):
        if getattr(self, 'store', None) is not None:
            self.store.close()
            self.store = None
        if getattr(self, '_temporary', None) is not None:
            self._temporary.cleanup()
            self._temporary = None


Environment = LongitudinalEnv
