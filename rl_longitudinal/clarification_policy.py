"""Memory-first decision contract shared by baselines and learned policies."""
from dataclasses import dataclass, asdict
from .memory_reconciliation import reconcile
from .temporal_memory import instant
import json


@dataclass(frozen=True)
class Decision:
    action: str
    foods: tuple = ()
    question: str | None = None
    question_key: str | None = None
    evidence_ids: tuple = ()
    retrieved_ids: tuple = ()
    applicable_ids: tuple = ()
    trace_id: int | None = None
    reason: str = ''
    reconciliation: object = None


def decide(store, query_at, context=None, target_food=None, user_available=True, known_at=None):
    context = dict(context or {})
    # Retrieve prior communicated answers; an occasion's meal/company is never
    # generalized to all future sessions. Fact clarifications use linked facts.
    previous_answers = store.clarification_history(known_at or query_at)
    answers = {}
    for row in previous_answers:
        key = row['question_key']
        if row['status'] == 'RESOLVED' and key in ('day', 'meal', 'company', 'place') and 'value' in row and row.get('query_at') and instant(row['query_at']) == instant(query_at):
            answers.setdefault(key, {})[json.dumps(row['value'], sort_keys=True)] = row['value']
    for key, values in answers.items():
        if len(values) == 1 and context.get(key) is None:
            context[key] = next(iter(values.values()))
    retrieval = store.retrieve(query_at, known_at=known_at, context=context, food=target_food)
    r = reconcile(retrieval)
    common = dict(evidence_ids=r.evidence_ids, retrieved_ids=r.retrieved_ids, applicable_ids=r.applicable_ids,
                  trace_id=retrieval.trace_id, reconciliation=r)
    if r.safe_foods:
        result = Decision('ANSWER', foods=r.safe_foods, reason='Safe applicable declaration, with local conflicts retained', **common)
    else:
        if target_food in r.forbidden_foods:
            key, question = 'restriction:' + target_food, 'Consta una restricción de seguridad para ' + target_food + '. ¿Puedes aclarar la restricción? No lo recomendaré mientras siga vigente.'
        elif r.missing_slots:
            slot = r.missing_slots[0]
            key, question = slot, {'day':'¿Es para un día laborable o fin de semana?', 'meal':'¿Para qué comida: desayuno, almuerzo, cena o merienda?',
                                   'company':'¿Con quién vas a comer?', 'place':'¿Dónde vas a comer?'}.get(slot, '¿Puedes precisar '+slot+'?')
        elif r.conflicts or r.uncertain_foods:
            food = target_food or (r.conflicts[0]['food'] if r.conflicts else r.uncertain_foods[0])
            key, question = 'current:' + food, 'Hay evidencias distintas sobre ' + food + '. ¿Cambió tu preferencia de forma estable, fue una excepción o depende del contexto?'
        else:
            key, question = None, None
        previous = [row for row in store.clarification_history(known_at or query_at) if row['question_key'] == key]
        if key in ('day', 'meal', 'company', 'place'):
            # Context answers describe an occasion, not an eternal user state.
            previous = [row for row in previous if (instant(row['query_at']) == instant(query_at) if row.get('query_at') else instant(row['at']).date() == instant(query_at).date())]
        elif key:
            # New communicated evidence can legitimately reopen an old ambiguity.
            food = key.split(':', 1)[-1]
            relevant_dates = [instant(f.recorded_at) for f in retrieval.facts if f.food == food]
            if relevant_dates:
                previous = [row for row in previous if instant(row['at']) >= max(relevant_dates)]
        if key and user_available and not previous:
            result = Decision('CLARIFY', question=question, question_key=key, reason='Material uncertainty; specific clarification before commitment', **common)
        elif key:
            result = Decision('PRESERVE_CONFLICT' if r.conflicts or r.uncertain_foods or target_food in r.forbidden_foods else 'CONDITIONAL_ANSWER',
                              reason='Uncertainty retained; unavailable user or already attempted clarification', **common)
        else:
            result = Decision('ABSTAIN', reason='Historical memory was consulted; no grounded applicable preference', **common)
    store.complete_trace(retrieval.trace_id, action=result.action, selected_foods=list(result.foods),
                         applicable_ids=list(result.applicable_ids), applicable_evidence_ids=list(result.evidence_ids),
                         question_key=result.question_key, reason=result.reason,
                         effective_intervals=r.effective_intervals, conflicts=list(r.conflicts))
    return result
