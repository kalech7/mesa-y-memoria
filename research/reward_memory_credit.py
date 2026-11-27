"""Attribute future success to writes while holding the conversation fixed."""

from research.src.readers.symbolic import read
from research.src.scoring.evidence import Evaluator


def memory_credit(trace, future=None, unsupported=None):
    """Compare the observed future success with the same episode without writes.

    This runs the fixed symbolic reader on the initial store, using the exact
    future query and the original conversation (including any clarification).
    A valid write earns positive credit only for a new justified success and
    only when all writes are supported. Any loss of success counts as harm.
    A valid NOOP receives zero write credit.

    Optional `future` and `unsupported` reuse the evaluator's original future
    score and count of unsupported operations from the same trace.
    """
    evaluator = Evaluator()
    if future is None:
        future = evaluator.score_future(trace)
    if unsupported is None:
        unsupported = evaluator.score_memory(trace)["unsupported_operations"]

    memory = trace["initial_memory"]
    query = trace["episode"]["future_query"]
    without_write = {
        **trace,
        "final_memory": memory,
        "future_response": read(memory, query),
        "executed_operations": [],
        "write_result": None,
    }
    baseline = evaluator.score_future(without_write)["success"]
    actual = future["success"]
    valid_write = (trace["write_result"] or {}).get("valid") is True
    supported_write = valid_write and unsupported == 0
    delta = actual - baseline
    return {
        "baseline_success": baseline,
        "actual_success": actual,
        "benefit": delta if delta < 0 or supported_write else 0,
        "valid_write": valid_write,
        "supported_write": supported_write,
    }
