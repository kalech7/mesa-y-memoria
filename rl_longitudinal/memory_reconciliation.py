"""Temporal applicability and local conflict reconciliation, without recency wins."""
from dataclasses import dataclass, field
from collections import defaultdict
from .temporal_memory import Retrieval, fact_from, instant


def context_status(fact, context):
    missing = []
    for key, expected in fact.scope.items():
        actual = context.get(key)
        if expected == 'ANY':
            continue
        if expected is None:
            # Unknown qualification is not the universal scope.
            missing.append('scope:' + key)
        elif actual is None or actual == 'ANY':
            missing.append(key)
        elif expected != actual:
            return None
    return tuple(missing)


def scopes_overlap(a, b):
    return all(x is None or y is None or x == 'ANY' or y == 'ANY' or x == y
               for x, y in zip((a.scope[k] for k in ('day','meal','company','place')),
                               (b.scope[k] for k in ('day','meal','company','place'))))


@dataclass(frozen=True)
class Reconciliation:
    applicable_ids: tuple = ()
    retrieved_ids: tuple = ()
    evidence_ids: tuple = ()
    supported_foods: tuple = ()
    safe_foods: tuple = ()
    forbidden_foods: tuple = ()
    conflicts: tuple = ()
    missing_slots: tuple = ()
    uncertain_foods: tuple = ()
    history: tuple = ()
    effective_intervals: dict = field(default_factory=dict)
    conflict_types: tuple = ()

    @property
    def has_material_uncertainty(self):
        return not self.safe_foods and bool(self.conflicts or self.missing_slots or self.uncertain_foods)


def reconcile(retrieval, context=None, query_at=None, target_food=None):
    if isinstance(retrieval, Retrieval):
        facts = list(retrieval.facts)
        context = dict(retrieval.context if context is None else context)
        query_at = query_at or retrieval.query_at
        target_food = target_food or retrieval.target_food
    else:
        facts = [fact_from(f) for f in retrieval]
        context = dict(context or {})
    if query_at is None:
        raise ValueError('Reconciliation requires an explicit query date')
    when = instant(query_at)
    # Dates of questions determine the actual day partition unless caller asked
    # a recurring day context explicitly. An absent day stays absent: this avoids
    # interpreting arbitrary reference dates as the user's intended eating day.
    ends = {f.memory_id: f.valid_to for f in facts}
    by_id = {f.memory_id: f for f in facts}
    for new in facts:
        for old_id in new.supersedes:
            old = by_id.get(old_id)
            if old and new.kind in ('EXPLICIT_CHANGE', 'CLARIFICATION') and new.valid_from and old.food == new.food and old.scope == new.scope and old.kind != 'RESTRICTION':
                previous_end = ends[old_id]
                if previous_end is None or instant(new.valid_from) < instant(previous_end):
                    ends[old_id] = new.valid_from
    applicable, potential, unknown = [], [], set()
    for f in facts:
        if f.status == 'INACTIVE':
            continue
        if when < instant(f.valid_from or f.observed_at):
            continue
        if ends[f.memory_id] and when >= instant(ends[f.memory_id]):
            continue
        missing = context_status(f, context)
        if missing is None:
            continue
        potential.append(f)
        if missing:
            unknown.update(missing)
        else:
            applicable.append(f)
    forbidden = {f.food for f in potential if f.kind == 'RESTRICTION' or f.polarity == 'FORBID'}
    grouped = defaultdict(list)
    for f in applicable:
        if f.kind != 'RESTRICTION' and f.polarity != 'FORBID':
            grouped[f.food].append(f)
    conflicts, uncertain, support = [], {f.food for f in potential if any(k.startswith('scope:') for k in (context_status(f, context) or ()))}, set()
    types = set()
    for food, records in sorted(grouped.items()):
        positive = [f for f in records if f.polarity == 'LIKE']
        negative = [f for f in records if f.polarity == 'DISLIKE']
        declared = [f for f in positive if f.kind not in ('OBSERVED_BEHAVIOR', 'AMBIGUOUS', 'UNRESOLVED') and f.source != 'ORDER_HISTORY' and f.uncertainty == 'KNOWN' and f.status != 'UNRESOLVED']
        ambiguous = [f for f in records if f.kind in ('AMBIGUOUS', 'UNRESOLVED') or f.uncertainty != 'KNOWN' or f.status == 'UNRESOLVED']
        # Temporary declarations override only incompatible tastes of exactly
        # this occasion, not their future validity or stored content.
        temporary = [f for f in declared if f.kind == 'TEMPORARY_EXCEPTION']
        if temporary:
            negative = [n for n in negative if n.kind == 'TEMPORARY_EXCEPTION']
        disagreement = positive and negative
        if disagreement or ambiguous:
            uncertain.add(food)
            ids = tuple(sorted(f.memory_id for f in records))
            conflicts.append({'food': food, 'memory_ids': ids, 'reason': 'incompatible_evidence' if disagreement else 'uncertain_validity'})
            sources = {f.source for f in records}
            types.add('SCC' if len(sources) > 1 else 'BOC' if len({f.observed_at for f in records}) > 1 else 'SAME_SCOPE')
        elif declared:
            support.add(food)
        elif positive:
            # Consumption alone leaves taste unknown; asking can add a real
            # declaration, but never promotes behavior to stable preference.
            uncertain.add(food)
    # A possibly applicable counterevidence about the SAME chosen food makes
    # commitment unsafe. Other foods' unresolved conflicts do not block it.
    for food in list(support):
        if any(f.food == food and f not in applicable and f.polarity == 'DISLIKE' for f in potential):
            support.remove(food); uncertain.add(food)
    support -= uncertain
    safe = support - forbidden
    if target_food:
        safe &= {target_food}; support &= {target_food}
        conflicts = [c for c in conflicts if c['food'] == target_food]
        uncertain &= {target_food}
        related = [f for f in potential if f.food == target_food]
        unknown = {k for f in related for k in (context_status(f, context) or ())}
        if target_food in forbidden:
            types.add('RESTRICTION')
    if len({f.scope.get('day') for f in potential if f.scope.get('day') not in (None,'ANY')}) > 1 or len({f.scope.get('meal') for f in potential if f.scope.get('meal') not in (None,'ANY')}) > 1:
        types.add('CPC')
    return Reconciliation(tuple(sorted(f.memory_id for f in applicable)), tuple(sorted(f.memory_id for f in facts)),
                          tuple(sorted({f.evidence_id for f in applicable})), tuple(sorted(support)), tuple(sorted(safe)),
                          tuple(sorted(forbidden)), tuple(conflicts), tuple(sorted(k for k in unknown if not k.startswith('scope:'))), tuple(sorted(uncertain)),
                          tuple(sorted(f.memory_id for f in facts if f.memory_id not in {a.memory_id for a in applicable})),
                          {f.memory_id: {'valid_from': f.valid_from, 'valid_to': ends[f.memory_id]} for f in facts}, tuple(sorted(types)))
