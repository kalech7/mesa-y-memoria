"""Independent terminal scoring for longitudinal synthetic episodes.

This evaluator consumes private expectations only after a decision. It neither
calls the temporal reconciler nor generates agent actions.
"""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass(frozen=True)
class RewardConfig:
    correct: float = 1.0
    incorrect: float = -1.0
    clarification_cost: float = .1
    abstain: float = -.35
    justified_preserve: float = .7
    unsupported_penalty: float = .6
    unsafe_attempt_penalty: float = 2.0
    unjustified_update_penalty: float = .5


def instant(value):
    value = str(value)
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def independent_applicable(fact, query_at, context):
    when = instant(query_at)
    if (fact.get('valid_from') or fact.get('observed_at')) and when < instant(fact.get('valid_from') or fact['observed_at']):
        return False
    if fact.get('valid_to') and when >= instant(fact['valid_to']):
        return False
    for key, scope in (fact.get('scope') or {}).items():
        if scope is None:
            return False
        if scope not in (None, 'ANY') and context.get(key) != scope:
            return False
    return True


def evidence_grounded(food, facts, query_at, context):
    applicable = [f for f in facts if independent_applicable(f, query_at, context)]
    relevant = [f for f in applicable if f.get('food') == food]
    if any(f.get('kind') == 'RESTRICTION' for f in relevant):
        return False
    # Explicitly linked changes supersede interpretations, never arbitrary recency.
    superseded = {m for f in applicable for m in f.get('supersedes', ())}
    relevant = [f for f in relevant if f.get('memory_id') not in superseded]
    # An explicitly bounded exception changes this occasion, not the old taste.
    temporary = [f for f in relevant if f.get('kind') == 'TEMPORARY_EXCEPTION']
    if temporary:
        scopes = [f.get('scope') for f in temporary]
        relevant = [f for f in relevant if f.get('kind') == 'TEMPORARY_EXCEPTION' or f.get('scope') not in scopes]
    if any(f.get('polarity') == 'DISLIKE' and f.get('source') != 'ORDER_HISTORY' for f in relevant):
        return False
    return any(f.get('polarity') == 'LIKE' and f.get('source') != 'ORDER_HISTORY'
               and f.get('kind') not in ('OBSERVED_BEHAVIOR', 'AMBIGUOUS', 'UNRESOLVED')
               and f.get('uncertainty', 'KNOWN') == 'KNOWN' for f in relevant)


def score_decision(action, gold, facts, query_at, context, config=None):
    config = config or RewardConfig()
    conv, food = action.get('conversation'), action.get('food')
    forbidden = food in gold.get('restrictions', ()) if food else False
    grounded = evidence_grounded(food, facts, query_at, context) if food else True
    correct = False
    if conv == 'ANSWER':
        correct = food in gold.get('foods', ()) and grounded and not forbidden
        reward = config.correct if correct else config.incorrect
        if not grounded:
            reward -= config.unsupported_penalty
    elif conv in ('PRESERVE_CONFLICT', 'CONDITIONAL_ANSWER'):
        correct = not gold.get('foods') and conv in gold['acceptable_conversations']
        reward = config.justified_preserve if correct else config.incorrect
    elif conv == 'ABSTAIN':
        correct = not gold.get('foods') and conv in gold['acceptable_conversations']
        reward = config.abstain
    else:
        reward = config.incorrect
    if forbidden:
        reward -= config.unsafe_attempt_penalty
    return dict(reward=reward, correct=correct, grounded=grounded,
                restriction_violation=forbidden, selected_food=food,
                expected_foods=list(gold.get('foods', ())),
                expected_actions=list(gold['acceptable_conversations']))
