"""Preregistered terminal utility, independent of the existing heuristic.

There is no information-gain bonus in the main experiment: interactions are useful
only through a better supported terminal outcome. Costs and safety are explicit.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RewardConfig:
    correct: float = 1.0
    incorrect: float = -1.0
    unsupported_penalty: float = 0.6
    restriction_penalty: float = 2.0
    clarification_cost: float = 0.10
    preserve_correct: float = 0.7
    preserve_incorrect: float = -0.2
    abstain: float = -0.35
    truncation: float = -0.35
    information_bonus: float = 0.0

    def __post_init__(self):
        if self.clarification_cost < 0 or self.unsupported_penalty < 0 or self.restriction_penalty < 0:
            raise ValueError('Costs must be nonnegative')
        if self.information_bonus != 0:
            raise ValueError('Main protocol does not authorize information-gain shaping')

    def to_dict(self):
        return asdict(self)


def terminal_reward(action, outcome, config=None):
    """Outcome is evaluator-only and is never returned in a policy observation."""
    config = config or RewardConfig()
    if action.startswith('ANSWER:'):
        value = config.correct if outcome['latent_correct'] else config.incorrect
        if not outcome['evidence_supported']:
            value -= config.unsupported_penalty
        if outcome.get('attempted_restriction_violation', outcome['restriction_violation']):
            value -= config.restriction_penalty
        return value
    if action == 'PRESERVE_CONFLICT':
        return config.preserve_correct if outcome['correct_conflict_preservation'] else config.preserve_incorrect
    if action == 'ABSTAIN':
        return config.abstain
    return config.truncation
