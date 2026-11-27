"""Minimal three-session SQLite environment with an explicit private boundary.

Session 1 is deterministic ingestion. The policy controls sessions 2 and 3.
Only explicit safety restrictions and the one-question budget can block execution;
no decision contract/reconciler replaces a proposed action with an expected label.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
import json
from pathlib import Path
from functools import lru_cache
import tempfile

from simulator.preferences import extract_preferences
from rl_longitudinal.temporal_memory import TemporalMemoryStore, fact_from, instant
from .observations import build_observation, evidence_kind, interpret_visible

CONVERSATIONS=('ANSWER','CLARIFY','PRESERVE_CONFLICT','CONDITIONAL_ANSWER','NOOP')
MEMORY_OPERATIONS=('ADD_MEMORY','UPDATE_WITH_HISTORY','PRESERVE_CONFLICT','NOOP')

@lru_cache(maxsize=32768)
def _cached_extract(event_json,user_id):
    # Cache a deterministic legacy parser call using public input only.
    return extract_preferences(json.loads(event_json),user_id)

@dataclass(frozen=True)
class RewardConfig:
    correct: float = 1.0
    incorrect: float = -1.0
    clarification_cost: float = 0.1
    unjustified_update_penalty: float = 0.5
    unsupported_answer_penalty: float = 0.6
    unsafe_penalty: float = 2.0
    justified_preservation: float = 0.7


class MinimalEnv:
    def __init__(self,scenario,representation='O3',extraction='VISIBLE_STRUCTURED',
                 db_path=None,reward_config=None):
        if representation not in ('O1','O2','O3'):
            raise ValueError('Unknown representation')
        if extraction not in ('VISIBLE_STRUCTURED','PIPELINE'):
            raise ValueError('Unknown extraction condition')
        self._scenario=deepcopy(scenario)
        self.representation,self.extraction=representation,extraction
        self.db_path=Path(db_path) if db_path is not None else None
        self.reward_config=reward_config or RewardConfig()
        self.store=None
        self._temporary=None

    def _open_store(self):
        store=TemporalMemoryStore(self._path,self._scenario['user_id'])
        if self.db_path is None:
            # Temporary, single-process experiment databases still use SQLite
            # and survive close/reopen across sessions; fsync is unnecessary for
            # disposable training trajectories. Persistent evidence keeps defaults.
            store.conn.execute('PRAGMA synchronous=OFF')
            store.conn.execute('PRAGMA journal_mode=MEMORY')
        return store

    def reset(self,seed=None):
        self.close()
        if self.db_path:
            if self.db_path.exists():
                raise FileExistsError('A historical SQLite database is never overwritten')
            self.db_path.parent.mkdir(parents=True,exist_ok=True)
            self._path=str(self.db_path)
        else:
            self._temporary=tempfile.TemporaryDirectory(prefix='food-observability-')
            self._path=str(Path(self._temporary.name)/'memory.sqlite')
        self.store=self._open_store()
        self.store.conn.execute('CREATE TABLE IF NOT EXISTS observability_evidence_archive(evidence_id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
        self._steps=[]; self._sessions=[]; self._return=0.; self._index=0
        self._extraction_records=[]; self._communicated=[]
        self._resolved_reply=False; self._unresolved_reply=False
        self._terminated=False
        initial=self._scenario['sessions'][0]
        self._archive(initial)
        for fact in self._extract(initial):
            self.store.add_fact(fact)
        self._index=1
        self._begin_session()
        return self._observe()

    def _archive(self,session):
        for event in session.get('events',[]):
            with self.store.conn:
                self.store.conn.execute('INSERT OR IGNORE INTO observability_evidence_archive VALUES (?,?)',
                    (event['message_id'],json.dumps(event,ensure_ascii=False,sort_keys=True)))
        self._communicated.extend(deepcopy(session.get('incoming',[])))

    def _extract(self,session):
        declared=session.get('incoming',[])
        parsed=[]
        for event in session.get('events',[]):
            expected=[f for f in declared if f['evidence_id']==event['message_id']]
            extraction=_cached_extract(json.dumps(event,sort_keys=True,ensure_ascii=False),self._scenario['user_id'])
            values=[]
            for original in (*extraction.preferences,*extraction.restrictions):
                data=asdict(original)
                # The legacy parser can emit an empty today interval at midnight.
                # This adapter changes only that invalid end, with an explicit log.
                if data.get('temporary') and data.get('valid_to') and instant(data['valid_to'])<=instant(data.get('valid_from') or data['timestamp']):
                    data['valid_to']=(instant(data['valid_to'])+timedelta(days=1)).isoformat()
                try:
                    fact=asdict(fact_from(data))
                except ValueError as error:
                    self._extraction_records.append(dict(evidence_id=event['message_id'],
                        extraction_error=True,temporal_error=True,error=str(error),fields=['invalid_fact']))
                    continue
                fact['recorded_at']=session['at']
                fact['text']=event['text']
                # Preserve visible event provenance lost by the legacy source
                # normalizer; a user reply remains an actual clarification.
                if event.get('source')=='CLARIFICATION':
                    fact['source']='CLARIFICATION'
                    if fact['kind']=='EXPLICIT_CHANGE':
                        fact['kind']='CLARIFICATION'
                # Match IDs, never meaning, to permit paired audit of field errors.
                reference=next((f for f in expected if f['food']==fact['food']),None)
                if reference:
                    fact['memory_id']=reference['memory_id']
                    fact['evidence_id']=reference['evidence_id']
                values.append(fact)
            for reference in expected:
                actual=next((f for f in values if f['food']==reference['food']),None)
                fields=[]
                if actual is None:
                    fields=['missing_fact']
                else:
                    for name in ('polarity','kind'):
                        if actual[name]!=reference[name] and not (name=='kind' and reference[name]=='AMBIGUOUS' and actual[name]=='STABLE_PREFERENCE'):
                            fields.append(name)
                    for name in ('valid_from','valid_to'):
                        left,right=actual.get(name),reference.get(name)
                        if name=='valid_from':
                            left=left or actual['observed_at']
                            right=right or reference['observed_at']
                        if (left is None)!=(right is None) or (left and right and instant(left)!=instant(right)):
                            fields.append(name)
                self._extraction_records.append(dict(evidence_id=reference['evidence_id'],
                    extraction_error=bool(fields),temporal_error=bool(set(fields)&{'kind','valid_from','valid_to'}),
                    fields=fields,declared=deepcopy(reference),parsed=deepcopy(actual)))
            parsed.extend(values)
        # Structured annotations describe visible literal evidence and dates only.
        # They are a controlled extraction condition, never latent preferences.
        return deepcopy(declared) if self.extraction=='VISIBLE_STRUCTURED' else parsed

    def _begin_session(self):
        self._session=deepcopy(self._scenario['sessions'][self._index])
        self._archive(self._session)
        self._pending=self._extract(self._session)
        self._question_count=0
        self._session_steps=[]
        self._session_return=0.
        self._memory_before=[asdict(f) for f in self.store.facts()]
        self._initial_observation=None
        self._initial_resolved_reply=self._resolved_reply

    def _observe(self):
        if self._terminated:
            return {'terminated':True}
        obs,self._trace_id=build_observation(self.store,self._pending,self._session,self._index,
                                           self.representation,self._question_count)
        self._observation=deepcopy(obs)
        if self._initial_observation is None:
            self._initial_observation=deepcopy(obs)
        return obs

    @staticmethod
    def _parse_action(action):
        if not isinstance(action,dict):
            raise ValueError('Use a composed action mapping')
        result=deepcopy(action)
        result.setdefault('conversation','NOOP')
        result.setdefault('memory_operation','NOOP')
        result.setdefault('polarity',None)
        result.setdefault('slot',None)
        if result['memory_operation']=='ADD':
            result['memory_operation']='ADD_MEMORY'
        if result['conversation'] not in CONVERSATIONS or result['memory_operation'] not in MEMORY_OPERATIONS:
            raise ValueError('Unknown action')
        if result['polarity'] not in (None,'LIKE','DISLIKE'):
            raise ValueError('Answer polarity must be LIKE or DISLIKE')
        return result

    def _evaluator_expected(self):
        """Independent authored labels, adjusted only for communicated replies.

        A private planned resolution cannot justify an answer before the user
        actually gives it. Preservation is then correct for either conflict script.
        """
        family=self._scenario['family']
        if family.startswith('RELEVANT_'):
            return 'LIKE' if self._resolved_reply else None
        return self._session['gold']['polarity']

    def _answer_grounded(self,observation,polarity):
        """Independent support audit of exposed evidence, never the rule policy.

        Original evidence annotations are used only in this evaluator. A policy's
        invented supersession or a lucky guess cannot manufacture backing. Facts
        omitted from the exposed observation cannot support its answer.
        """
        visible_ids={f['memory_id'] for f in observation.get('memory',[])}
        original={f['memory_id']:f for f in self._communicated}
        at=instant(observation['query_at'])
        context=observation.get('context',{})
        active=[]
        for identifier in visible_ids:
            fact=original.get(identifier)
            if not fact or fact['food']!=observation['target_food']:
                continue
            if instant(fact.get('valid_from') or fact['observed_at'])>at:
                continue
            if fact.get('valid_to') and instant(fact['valid_to'])<=at:
                continue
            if any(value not in (None,'ANY') and context.get(slot)!=value
                   for slot,value in fact['scope'].items()):
                continue
            active.append(fact)
        if any(f['polarity']=='FORBID' for f in active):
            return polarity=='DISLIKE'
        replacements=[f for f in active if f['kind'] in ('EXPLICIT_CHANGE','CLARIFICATION')]
        for replacement in sorted(replacements,key=lambda f:(f['observed_at'],f['memory_id'])):
            active=[f for f in active if f is replacement or f['scope']!=replacement['scope']
                    or f['observed_at']>replacement['observed_at']
                    or (f['observed_at']==replacement['observed_at'] and replacement['kind']!='CLARIFICATION')]
        exceptions=[f for f in active if f['kind']=='TEMPORARY_EXCEPTION']
        if exceptions:
            active=[f for f in active if f in exceptions or all(f['scope']!=e['scope'] for e in exceptions)]
        if any(f['kind'] in ('OBSERVED_BEHAVIOR','UNRESOLVED') for f in active):
            return False
        polarities={f['polarity'] for f in active if f['polarity'] in ('LIKE','DISLIKE')}
        return polarities=={polarity}

    def _memory_action(self,operation):
        updates=[]; incorrect=0
        has_explicit_change=any(f['kind'] in ('EXPLICIT_CHANGE','CLARIFICATION') for f in self._pending)
        for original in self._pending:
            fact=deepcopy(original)
            item_operation=operation
            # Composite UPDATE appends accompanying evidence and revises only
            # facts explicitly marked as changes by communicated evidence.
            if operation=='UPDATE_WITH_HISTORY' and has_explicit_change and fact['kind'] not in ('EXPLICIT_CHANGE','CLARIFICATION'):
                item_operation='ADD_MEMORY'
            known={f.memory_id:f for f in self.store.facts()}
            actual=item_operation
            if item_operation=='NOOP':
                self.store.record_noop(self._session['at'],'Policy omitted indexing; raw evidence retained',fact['evidence_id'])
                updates.append(dict(memory_id=fact['memory_id'],proposed=operation,executed=operation))
                continue
            justified=True
            if item_operation=='UPDATE_WITH_HISTORY':
                justified=fact['kind'] in ('EXPLICIT_CHANGE','CLARIFICATION')
                if not justified:
                    incorrect+=1
                # An unjustified update is allowed to corrupt the managed memory
                # interpretation and affect the future. Safety facts are immutable.
                candidates=[f for f in known.values() if f.food==fact['food'] and f.polarity!='FORBID'
                            and f.kind!='RESTRICTION' and f.scope==fact['scope']
                            and instant(f.valid_from or f.observed_at)<=instant(fact.get('valid_from') or fact['observed_at'])]
                fact['supersedes']=[f.memory_id for f in candidates]
                if not justified:
                    fact['kind']='EXPLICIT_CHANGE'
                    fact['valid_to']=None
                    fact['text']='[Interpretación de la política: cambio estable] '+fact['text']
                fact['valid_from']=fact.get('valid_from') or fact['observed_at']
                actual='UPDATE_WITH_HISTORY'
            else:
                fact['supersedes']=[]
                actual='PRESERVE_CONFLICT' if item_operation=='PRESERVE_CONFLICT' else 'ADD'
            # Constraints about ownership/history/safety are validated by SQLite's
            # append-only store. Ordinary bad decisions are not replaced by gold.
            self.store.add_fact(fact,operation=actual,at=self._session['at'],reason='Policy composed action')
            updates.append(dict(memory_id=fact['memory_id'],proposed=operation,
                                executed=item_operation,unjustified=item_operation=='UPDATE_WITH_HISTORY' and not justified))
        self._pending=[]
        return updates,incorrect

    def _safety_block(self,action):
        if action['conversation']!='ANSWER' or action['polarity']!='LIKE':
            return False
        return any(f['food']==self._session['target_food'] and f['polarity']=='FORBID'
                   for f in self._communicated)

    def _record(self,trace,reward):
        trace['reward']=reward
        self._return+=reward
        self._session_return+=reward
        self._steps.append(deepcopy(trace))
        self._session_steps.append(deepcopy(trace))

    def step(self,action):
        if self._terminated:
            raise RuntimeError('Episode already terminated')
        proposed=self._parse_action(action)
        executed=deepcopy(proposed)
        obs=deepcopy(self._observation)
        pending_before=deepcopy(self._pending)
        reasons=[]
        blocked=self._safety_block(proposed)
        if blocked:
            executed.update(conversation='PRESERVE_CONFLICT',polarity=None,slot=None)
            reasons.append('EXPLICIT_RESTRICTION')
        if executed['conversation']=='CLARIFY' and (self._index!=1 or self._question_count>=1):
            # Budget enforcement executes no replacement decision. NOOP is scored
            # as a failure to finish the interaction, not as an expected response.
            executed.update(conversation='NOOP',slot=None,polarity=None)
            reasons.append('CLARIFICATION_BUDGET')
        available_before=[asdict(f) for f in self.store.facts()]
        deferred=deepcopy(self._pending) if executed['conversation']=='CLARIFY' and executed['memory_operation']=='NOOP' else []
        operations,bad_updates=self._memory_action(executed['memory_operation'])
        if deferred:
            self._pending=deferred
        cfg=self.reward_config
        reward=-bad_updates*cfg.unjustified_update_penalty-int(blocked)*cfg.unsafe_penalty
        trace=dict(session_index=self._index,query=self._session['query'],at=self._session['at'],
            query_at=self._session['query_at'],original_input=deepcopy(self._session['events']),
            available_memories=available_before,pending_evidence=pending_before,
            retrieved_memories=deepcopy(obs['memory']),observation=obs,
            proposed_action=proposed,executed_action=executed,shield_reasons=reasons,
            memory_operations=operations,incorrect_updates=bad_updates,
            trace_id=self._trace_id,expected_polarity=self._evaluator_expected(),
            repeated_question=False,necessary_question=False,terminal=False)
        if executed['conversation']=='CLARIFY':
            self._question_count+=1
            key=executed.get('slot') or 'unspecified'
            private_response=self._session.get('responses',{}).get(key)
            response=deepcopy(private_response or dict(text='No tengo una aclaración para esa pregunta.',status='UNRESOLVED',fact=None))
            trace['necessary_question']=bool(key=='preference' and self._session['gold']['needs_clarification'])
            trace['repeated_question']=any(h['question_key']==key for h in self.store.clarification_history())
            self.store.record_clarification(key,self._session['at'],response['text'],response['status'])
            if response.get('fact'):
                response['fact']['source']='CLARIFICATION'
                response['fact']['kind']='CLARIFICATION'
                self._resolved_reply=True
                event=dict(message_id=response['fact']['evidence_id'],text=response['text'],at=self._session['at'],source='CLARIFICATION')
                reply_session=dict(at=self._session['at'],incoming=[response['fact']],events=[event])
                self._archive(reply_session)
                self._pending.extend(self._extract(reply_session))
            elif private_response:
                self._unresolved_reply=True
            reward-=cfg.clarification_cost
            trace.update(response=response['text'],response_status=response['status'],correct=None)
            self._record(trace,reward)
            return self._observe(),reward,False,False,{}
        expected=self._evaluator_expected()
        conv=executed['conversation']
        uncertainty_response=conv in ('PRESERVE_CONFLICT','CONDITIONAL_ANSWER')
        outcome_correct=(conv=='ANSWER' and expected is not None and executed['polarity']==expected) or (uncertainty_response and expected is None)
        if conv=='ANSWER':
            grounded=self._answer_grounded(obs,executed['polarity'])
        else:
            # Uncertainty must itself have observable backing. Agreement with a
            # private unresolved label cannot justify preservation when the
            # observation contains just one clear, current preference.
            unresolved_reply=any(h.get('status')=='UNRESOLVED' and h.get('question_key')=='preference'
                                 for h in obs.get('clarification_history',[]))
            no_unique_support=not any(self._answer_grounded(obs,p) for p in ('LIKE','DISLIKE'))
            grounded=bool(uncertainty_response and expected is None and
                          (unresolved_reply or no_unique_support))
        correct=outcome_correct and grounded
        reward+=cfg.justified_preservation if correct and uncertainty_response else cfg.correct if correct else cfg.incorrect
        unsupported=conv=='ANSWER' and not grounded
        reward-=cfg.unsupported_answer_penalty*int(unsupported)
        trace.update(terminal=True,correct=bool(correct),outcome_correct=bool(outcome_correct),grounded=bool(grounded),
            unsupported_answer=unsupported,expected_polarity=expected,
            memory_after=[asdict(f) for f in self.store.facts()],
            preserved_uncertainty=bool(correct and uncertainty_response))
        self._record(trace,reward)
        expected_ids=set(self._session['gold']['relevant_memory_ids'])
        if self._resolved_reply:
            expected_ids|={f['memory_id'] for f in self._communicated if f['source']=='CLARIFICATION'}
        retrieved_ids={f['memory_id'] for f in obs['memory']}
        interpreted=interpret_visible(self._initial_observation)
        declared_temporal={f['memory_id']:f for f in self._communicated if f['food']==self._session['target_food']}
        compared=[f for f in self._initial_observation['memory'] if f['memory_id'] in declared_temporal]
        # This measures an explicit public semantic interpreter against authored
        # visible annotations; it is not a proxy based on response correctness.
        temporal_mismatches=sum(evidence_kind(f)!=declared_temporal[f['memory_id']]['kind']
            and not (declared_temporal[f['memory_id']]['kind']=='AMBIGUOUS' and evidence_kind(f)=='STABLE_PREFERENCE')
            for f in compared)
        active_ids={f['memory_id'] for f in interpreted['active_facts']}
        at=instant(self._session['query_at'])
        interval_mismatches=0
        for f in compared:
            declared=declared_temporal[f['memory_id']]
            should_be_active=instant(declared.get('valid_from') or declared['observed_at'])<=at and (not declared.get('valid_to') or at<instant(declared['valid_to']))
            interval_mismatches+=int((f['memory_id'] in active_ids)!=should_be_active)
        questions=[s for s in self._session_steps if s['executed_action']['conversation']=='CLARIFY']
        result=dict(session_index=self._index,correct=bool(correct),grounded=bool(grounded),
            return_=self._session_return,expected_polarity=expected,steps=deepcopy(self._session_steps),
            questions=len(questions),necessary_asks=sum(s['necessary_question'] for s in questions),
            unnecessary_asks=sum(not s['necessary_question'] for s in questions),
            repeated_asks=sum(s['repeated_question'] for s in questions),
            incorrect_updates=sum(s['incorrect_updates'] for s in self._session_steps),
            shield_interventions=sum(bool(s['shield_reasons']) for s in self._session_steps),
            relevant_memory_count=len(expected_ids),retrieved_relevant_count=len(expected_ids&retrieved_ids),
            potential_conflict=self._session['gold']['conflict'] and not self._initial_resolved_reply,
            conflict_detected=interpreted['conflict'],
            temporal_comparisons=len(compared),temporal_interpretation_errors=temporal_mismatches,
            interval_applicability_errors=interval_mismatches,
            temporal_decision_error=bool(self._session['gold']['temporal'] and not correct),
            final_action=executed,memory_before=self._memory_before,memory_after=trace['memory_after'])
        result['return']=result.pop('return_')
        self._sessions.append(result)
        self._index+=1
        self._terminated=self._index>=3
        if not self._terminated:
            # Verify the next session actually reloads the same persistent SQLite.
            self.store.close()
            self.store=self._open_store()
            self._begin_session()
        return self._observe(),reward,self._terminated,False,{'correct':bool(correct),'decision_correct':bool(correct)}

    def diagnostic_metrics(self):
        """Private evaluator output. Never pass this mapping to a policy."""
        sessions=self._sessions
        terminal=[s for s in self._steps if s['terminal']]
        future=[s for s in sessions if s['session_index']==2]
        relevant=sum(s['relevant_memory_count'] for s in sessions)
        temporal_n=sum(s['temporal_comparisons'] for s in sessions)
        conflicts=[s for s in sessions if s['potential_conflict']]
        # Deduplicate only by evidence ID; one evidence event can have multiple
        # extracted records, but is one source unit in the extraction audit.
        extraction={r['evidence_id']:r for r in self._extraction_records}
        records=list(extraction.values())
        parser_errors=sum(r['extraction_error'] for r in records)
        result=dict(episode_id=self._scenario['episode_id'],user_id=self._scenario['user_id'],
            family=self._scenario['family'],representation=self.representation,extraction=self.extraction,
            temporary_sqlite_fast_mode=self.db_path is None,
            sessions=deepcopy(sessions),steps=deepcopy(self._steps),
            policy_observations=[deepcopy(s['observation']) for s in self._steps],
            extraction_records=deepcopy(records),operations=self.store.operations(),
            retrieval_traces=self.store.retrieval_traces(),sqlite_path=self._path,
            terminated=self._terminated,error_rate=sum(not s['correct'] for s in terminal)/max(1,len(terminal)),
            accuracy=sum(s['correct'] for s in terminal)/max(1,len(terminal)),
            future_accuracy=sum(s['correct'] for s in future)/max(1,len(future)),
            return_=self._return,mean_reward=self._return/max(1,len(self._steps)),
            extraction_error_rate=parser_errors/max(1,len(records)) if self.extraction=='PIPELINE' else 0.,
            original_parser_error_rate=parser_errors/max(1,len(records)),
            temporal_error_rate=sum(s['temporal_interpretation_errors'] for s in sessions)/max(1,temporal_n),
            temporal_type_error_rate=sum(s['temporal_interpretation_errors'] for s in sessions)/max(1,temporal_n),
            interval_applicability_error_rate=sum(s['interval_applicability_errors'] for s in sessions)/max(1,temporal_n),
            temporal_decision_errors=sum(s['temporal_decision_error'] for s in sessions),
            conflicts_detected=sum(s['conflict_detected'] for s in conflicts),
            conflict_detection_rate=sum(s['conflict_detected'] for s in conflicts)/max(1,len(conflicts)),
            questions=sum(s['questions'] for s in sessions),
            necessary_asks=sum(s['necessary_asks'] for s in sessions),
            unnecessary_asks=sum(s['unnecessary_asks'] for s in sessions),
            repeated_asks=sum(s['repeated_asks'] for s in sessions),
            incorrect_updates=sum(s['incorrect_updates'] for s in sessions),
            future_errors=sum(not s['correct'] for s in future),
            relevant_memory_recall=sum(s['retrieved_relevant_count'] for s in sessions)/max(1,relevant),
            relevant_memory_count=relevant,
            retrieved_relevant_count=sum(s['retrieved_relevant_count'] for s in sessions),
            potential_conflicts=len(conflicts),
            temporal_comparisons=temporal_n,
            temporal_interpretation_errors=sum(s['temporal_interpretation_errors'] for s in sessions),
            interval_applicability_errors=sum(s['interval_applicability_errors'] for s in sessions),
            extraction_total=len(records),
            extraction_errors=parser_errors if self.extraction=='PIPELINE' else 0,
            original_parser_errors=parser_errors,
            needed_questions=int(self._scenario['sessions'][1]['gold']['needs_clarification']),
            shield_interventions=sum(s['shield_interventions'] for s in sessions),
            unsupported_answers=sum(s.get('unsupported_answer',False) for s in terminal))
        result['return']=result['return_']
        return result

    def close(self):
        if self.store is not None:
            self.store.close()
            self.store=None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary=None

ObservabilityEnv=MinimalEnv
