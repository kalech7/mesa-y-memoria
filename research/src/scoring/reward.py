"""Fixed utility; changing the horizon changes only future credit."""
from __future__ import annotations


def rewards(r2, r3, questions=0, unsupported_operations=0, invalid_outputs=0, question_cost=.10):
    if questions not in (0, 1) or not 0 <= unsupported_operations <= 2 or not 0 <= invalid_outputs <= 3:
        raise ValueError("Episode counters outside protocol bounds")
    if r2 not in (-1, 0, 1) or r3 not in (-1, 0, 1):
        raise ValueError("Session utilities must be -1, 0 or 1")
    components = {"current": r2, "future": r3, "question_penalty": -question_cost * questions,
                  "unsupported_operation_penalty": -.50 * unsupported_operations,
                  "invalid_output_penalty": -.20 * invalid_outputs}
    short = r2 + sum(components[k] for k in ("question_penalty", "unsupported_operation_penalty", "invalid_output_penalty"))
    return {"R_long": short + r3, "R_short": short, "components": components}
