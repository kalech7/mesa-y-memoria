"""Ground conditional premises in public evidence, independently of latent gold."""
from copy import deepcopy


PENDING_APPLIES = "si el cambio pendiente sigue siendo aplicable"
PENDING_EXPIRED = "si el cambio pendiente ya no es aplicable"
UNSPECIFIED = "según el alcance pendiente"


def _context(condition, records):
    if isinstance(condition, str):
        text = " ".join(condition.casefold().strip().rstrip(".").split())
        aliases = {"casa": "casa", "en casa": "casa", "si estás en casa": "casa",
                   "si está en casa": "casa", "oficina": "oficina", "en la oficina": "oficina",
                   "si estás en la oficina": "oficina", "si está en la oficina": "oficina",
                   "trabajo": "oficina", "en el trabajo": "oficina"}
        for record in records:
            public_context = record.get("scope", {}).get("context")
            if isinstance(public_context, str):
                aliases[public_context.casefold()] = public_context
                aliases["en " + public_context.casefold()] = public_context
        return aliases.get(text)
    if not isinstance(condition, dict) or not condition:
        return None
    if set(condition) == {"context"}:
        return _context(condition["context"], records)
    public_keys = {"place", "company", "day_type", "location"}
    public_keys.update(k for r in records if isinstance(r.get("scope", {}).get("context"), dict)
                       for k in r["scope"]["context"])
    if not set(condition) <= public_keys:
        return None
    if set(condition) == {"place"} and not any(isinstance(r.get("scope", {}).get("context"), dict) for r in records):
        return _context(condition["place"], records)
    return deepcopy(condition)


def condition_errors(response, observation, *, query=None, records=None):
    """Check every branch, not just the set of food values or citation IDs.

    Named contexts are interventions on the declared query, checked against all
    observable evidence. No particular hidden world, return or gold is read.
    The legacy noncommittal placeholder and matching compatible_value remain
    admissible, but their clarity is distinct from factual justification.
    """
    from ..policy.rule_controller import compatible_worlds, expected_memory
    records = records if records is not None else expected_memory(observation)
    effective_query = deepcopy(query or observation["query"])
    errors = []
    for index, branch in enumerate(response.get("branches", [])):
        condition, value = branch.get("condition"), branch.get("value")
        if isinstance(condition, dict) and set(condition) == {"compatible_value"}:
            valid = condition["compatible_value"] == value
        elif isinstance(condition, str) and condition.casefold().strip() == UNSPECIFIED:
            valid = True  # No specific premise is asserted by this placeholder.
        elif isinstance(condition, str) and condition in (PENDING_APPLIES, PENDING_EXPIRED):
            pending = [r for r in records if r.get("scope", {}).get("kind") == "unknown"]
            if condition == PENDING_APPLIES:
                valid = any(r["value"] == value for r in pending)
            else:
                valid = bool(pending) and any(r["value"] == value and r.get("scope", {}).get("kind") != "unknown" for r in records)
        else:
            context = _context(condition, records)
            if context is None:
                valid = False
            else:
                hypothetical = {**effective_query, "context": context}
                possible = {w["value"] for w in compatible_worlds(observation, query=hypothetical)}
                valid = possible == {value}
        if not valid:
            errors.append(f"branch_condition_not_entailed:{index}")
    return errors
