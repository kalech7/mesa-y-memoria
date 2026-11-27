"""Exact TRAIN action values for the existing v6 two-visit, one-question task.

Only the policy margin at the public second visit after WAIT enters the
expectation. The reward on every terminal path is the longitudinal score
actually produced by the frozen SQLite simulator, including its question cost.
"""
from collections import defaultdict
from dataclasses import dataclass
from math import exp, fsum, isclose, isfinite
from statistics import mean, median

from research.tesis_timing_v6_bank import build_worlds, verify_worlds
from research.tesis_timing_v6_env import enumerate_world, prompt_text
from research.tesis_timing_v6_eval import ASK, WAIT, group_worlds


@dataclass(frozen=True)
class Record:
    root: str
    pair: str
    variant: str
    visit: int
    public: str
    weight: float = 1.


def _ask_probability(margin, temperature):
    if not isfinite(margin):
        raise ValueError("Expected a finite second-visit ASK minus WAIT margin")
    z = margin / temperature
    e = exp(-abs(z))
    return 1. / (1. + e) if z >= 0 else e / (1. + e)


class TrainRewards:
    def __init__(self):
        worlds = build_worlds("train")
        verify_worlds("train", worlds)
        trees = [enumerate_world(world, budget=1) for world in worlds]
        # This also checks that legal menus and both public decision histories
        # agree across all latent worlds of a root before computing any Q.
        roots = sorted(group_worlds(trees), key=lambda r: (r.pair, r.variant))
        self.records = tuple(
            Record(r.identity, r.pair, r.variant, visit, prompt_text(public))
            for r in roots for visit, public in enumerate((r.first, r.second_after_wait), start=1)
        )
        self.prompts = tuple(dict.fromkeys(r.public for r in self.records))
        self._later = {r.root: r.public for r in self.records if r.visit == 2}
        by_root = defaultdict(list)
        for tree in trees:
            by_root[tree.story["root_id"]].append(tree)
        self.path_returns = {
            root: {path: fsum(t.mass * t.outcomes[path]["longitudinal"] for t in alternatives)
                   for path in ((ASK, WAIT), (WAIT, ASK), (WAIT, WAIT))}
            for root, alternatives in by_root.items()
        }

    def exact_returns(self, scores=None, *, continuation="policy", temperature=4.):
        """Return (Q_WAIT, Q_ASK) per record, in the same order as records.

        policy expects raw ASK-minus-WAIT margins keyed by public prompt.
        optimal is an explicit Q* ablation: choose a single best second action
        after averaging over worlds consistent with the public information.
        """
        if continuation not in ("policy", "optimal"):
            raise ValueError("Expected 'policy' or 'optimal' continuation")
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be positive and finite")
        if continuation == "policy" and (scores is None or not set(self._later.values()).issubset(scores)):
            raise ValueError("Missing public second-visit scores")
        result = []
        for record in self.records:
            paths = self.path_returns[record.root]
            wait_twice, ask_later = paths[WAIT, WAIT], paths[WAIT, ASK]
            if record.visit == 2:
                result.append((wait_twice, ask_later))
                continue
            if continuation == "optimal":
                wait_now = max(wait_twice, ask_later)
            else:
                prob = _ask_probability(scores[self._later[record.root]], temperature)
                wait_now = (1. - prob) * wait_twice + prob * ask_later
            result.append((wait_now, paths[ASK, WAIT]))
        return result

    def audit(self, returns):
        """Summarise the signs and magnitudes of unnormalised TRAIN advantages."""
        gaps = defaultdict(list)
        first = defaultdict(dict)
        for record, (q_wait, q_ask) in zip(self.records, returns, strict=True):
            if not all(isfinite(q) for q in (q_wait, q_ask)):
                raise ValueError("Nonfinite action return")
            gap = q_ask - q_wait
            if isclose(q_ask, q_wait, rel_tol=0., abs_tol=1e-12):
                gap = 0.
            gaps[record.visit, record.variant].append(gap)
            if record.visit == 1:
                first[record.pair][record.variant] = gap
        return {
            "pairs": len(first), "roots": len(self.path_returns), "records": len(self.records),
            "pairs_aligned_v1": sum(g["ask_now"] > 0 > g["save"] for g in first.values()),
            "pairs_reversed_v1": sum(g["ask_now"] < 0 < g["save"] for g in first.values()),
            "pairs_with_a_tie_v1": sum(0. in g.values() for g in first.values()),
            "gap_ask_minus_wait": {
                f"v{visit}/{variant}": {
                    "min": min(values), "mean": mean(values), "median": median(values), "max": max(values),
                    "positive": sum(x > 0 for x in values), "negative": sum(x < 0 for x in values),
                    "ties": values.count(0.),
                } for (visit, variant), values in gaps.items()
            },
        }
