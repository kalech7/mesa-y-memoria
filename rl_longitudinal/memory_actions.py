"""Auditable append-only memory operations; no DELETE for conflict resolution."""
from .temporal_memory import fact_from
from .memory_reconciliation import reconcile

MEMORY_OPERATIONS = ('ADD', 'UPDATE_WITH_HISTORY', 'NOOP', 'PRESERVE_CONFLICT')


def apply_memory_action(store, operation, fact=None, *, at=None, reason='Policy operation', evidence_id='', linked_ids=(),
                        query_at=None, known_at=None, context=None):
    if operation not in MEMORY_OPERATIONS:
        raise ValueError('Unsupported or destructive memory operation')
    if at is None and fact is not None:
        at = fact_from(fact).recorded_at
    if at is None:
        raise ValueError('Memory operation needs a reproducible operation time')
    retrieval = store.retrieve(query_at or at, known_at=known_at or at, context=context)
    assessment = reconcile(retrieval)
    store.complete_trace(retrieval.trace_id, applicable_ids=list(assessment.applicable_ids),
                         conflicts=list(assessment.conflicts), temporal_intervals=assessment.effective_intervals)
    if operation == 'NOOP':
        if at is None:
            at = fact_from(fact).recorded_at if fact is not None else None
        if at is None:
            raise ValueError('NOOP needs a reproducible operation time')
        store.record_noop(at, reason, evidence_id, linked_ids)
        store.complete_trace(retrieval.trace_id, memory_operation=operation, reason=reason)
        return None
    if fact is None:
        raise ValueError('A memory operation requires communicated evidence')
    value = fact_from(fact)
    if operation == 'UPDATE_WITH_HISTORY' and value.kind not in ('EXPLICIT_CHANGE', 'CLARIFICATION'):
        raise ValueError('Ambiguous behavior/exception cannot permanently update stable memory')
    identifier = store.add_fact(value, operation=operation, reason=reason, at=at)
    store.complete_trace(retrieval.trace_id, memory_operation=operation, reason=reason, new_memory_id=identifier)
    return identifier
