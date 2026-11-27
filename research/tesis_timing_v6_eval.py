"""Causal comparison with exactly ten questions shared by twenty concurrent stories."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from math import isclose, isfinite
from random import Random
from statistics import mean


ASK, WAIT = "ASK", "WAIT"
PACKET_SIZE, QUESTIONS = 20, 10
METRICS = ("future_supported", "current_supported", "future_unsupported_reuse", "questions")


@dataclass
class Root:
    identity: str
    pair: str
    packet: str
    variant: str
    first: dict
    second_after_wait: dict
    outcomes: dict


def group_worlds(trees):
    """Score paths in expectation over worlds *after* exposing public prompts only."""
    groups = defaultdict(list)
    for tree in trees:
        groups[tree.story["root_id"]].append(tree)
    roots = []
    for identity, worlds in sorted(groups.items()):
        first = worlds[0]
        obs = [first.decisions[p].observation for p in ((), (WAIT,))]
        weights = [t.mass for t in worlds]
        if not isclose(sum(weights), 1.0) or any(w <= 0 for w in weights):
            raise ValueError("Each root needs normalized positive world probabilities")
        for tree in worlds:
            for prefix, visible in zip(((), (WAIT,)), obs):
                node = tree.decisions[prefix]
                if set(node.legal) != {ASK, WAIT} or node.observation != visible:
                    raise ValueError("Unanswered histories must have a common public observation and menu")
        by_path = {}
        for path in ((ASK, WAIT), (WAIT, ASK), (WAIT, WAIT)):
            by_path[path] = {
                m: sum(w * t.outcomes[path][m] for w, t in zip(weights, worlds))
                for m in (
                    *METRICS,
                    "future_queries",
                    "future_positive_queries",
                    "future_positive_supported",
                    "future_correct_abstentions",
                )
            }
        story = first.story
        roots.append(Root(identity, story["pair_id"], story["packet_id"], story["variant"], *obs, by_path))
    return roots


def score_public(roots, scorer):
    # Batched computation is equivalent to the two sequential reads. First-stage
    # allocation only receives the first column and never the later user text.
    first = scorer.margins([r.first for r in roots])
    later = scorer.margins([r.second_after_wait for r in roots])
    if not all(isfinite(x) for x in [*first, *later]):
        raise ValueError("All two-action margins must be finite")
    return {r.identity: (a, b) for r, a, b in zip(roots, first, later, strict=True)}


def thresholds(margins):
    ordered = sorted(v[0] for v in margins.values())
    # 'never' and 'always' are explicit finite-JSON endpoints. No reserve
    # scores enter this grid; numerical cutoffs transfer unchanged from validation.
    cuts = {0.0, *(ordered[round((len(ordered) - 1) * f)] for f in (0.25, 0.5, 0.75, 0.90))}
    return ["always", *sorted(cuts), "never"]


def wants_question(margin, cutoff):
    if cutoff == "always":
        return True
    if cutoff == "never":
        return False
    return margin > cutoff


def select_slots(ids, scores, count, visit):
    def rank(identity):
        tie = sha256(f"timing-v6/{visit}/{identity}".encode()).digest()
        return -scores[identity], tie

    return set(sorted(ids, key=rank)[:count])


def allocate_packet(ids, first_scores, later_scorer, cutoff, questions=QUESTIONS):
    """Open today's opportunities before asking tomorrow's scorer for surviving IDs."""
    if len(ids) != PACKET_SIZE or len(set(ids)) != len(ids) or not 0 <= questions <= len(ids):
        raise ValueError("Expected a 20-story packet with at most one question per story")
    want = [i for i in ids if wants_question(first_scores[i], cutoff)]
    early = select_slots(want, first_scores, questions, 1)
    still_waiting = [i for i in ids if i not in early]
    late = set()
    if len(early) < questions:
        scores = later_scorer(still_waiting)
        late = select_slots(still_waiting, scores, questions - len(early), 2)
    return {i: (ASK, WAIT) if i in early else (WAIT, ASK) if i in late else (WAIT, WAIT) for i in ids}


def aggregate(rows):
    if not rows:
        raise ValueError("No observations to average")
    return {k: mean(r[k] for r in rows) for k in rows[0]}


def evaluate_pool(roots, margins, cutoff):
    packets = defaultdict(list)
    for root in roots:
        packets[root.packet].append(root)
    result = {}
    for packet_id, group in sorted(packets.items()):
        ids = [r.identity for r in group]
        first_scores = {i: margins[i][0] for i in ids}
        actions = allocate_packet(
            ids, first_scores, lambda remaining: {i: margins[i][1] for i in remaining}, cutoff
        )
        rows = []
        for root in group:
            path = actions[root.identity]
            rows.append(
                {
                    **root.outcomes[path],
                    "asked_first": float(path[0] == ASK),
                    "asked_second": float(path[1] == ASK),
                    "asked_second_nonpositive_margin": float(
                        path[1] == ASK and margins[root.identity][1] <= 0
                    ),
                }
            )
        if not isclose(sum(row["questions"] for row in rows), QUESTIONS, abs_tol=1e-8):
            raise ValueError("Unequal realized question spending")
        result[packet_id] = {
            "metrics": aggregate(rows),
            "actions": {i: list(path) for i, path in actions.items()},
        }
    return {"macro": aggregate([p["metrics"] for p in result.values()]), "packets": result}


def independent_timing(roots, margins):
    """Unconstrained greedy rollout and forced one-question timing, diagnostics only."""
    pairs = defaultdict(list)
    native, exact_one, correct_native = [], [], {}
    first_by_agenda = defaultdict(list)
    for root in roots:
        a, b = margins[root.identity]
        path = (ASK, WAIT) if a > 0 else (WAIT, ASK) if b > 0 else (WAIT, WAIT)
        native.append(root.outcomes[path])
        early = root.outcomes[ASK, WAIT]["future_supported"]
        late = root.outcomes[WAIT, ASK]["future_supported"]
        if isclose(early, late, abs_tol=1e-8):
            continue
        chosen_early = a > 0
        first_by_agenda[root.variant].append(int(chosen_early))
        optimal = (ASK, WAIT) if early > late else (WAIT, ASK)
        exact_one.append(root.outcomes[(ASK, WAIT) if chosen_early else (WAIT, ASK)])
        correct_native.setdefault(root.pair, []).append(path == optimal)
        pairs[root.pair].append((early > late, chosen_early))
    flips = [v for v in pairs.values() if len(v) == 2 and v[0][0] != v[1][0]]
    native_pairs = [correct_native[k] for k, v in pairs.items() if len(v) == 2 and v[0][0] != v[1][0]]
    return {
        "greedy_free_budget": aggregate(native),
        "one_question_per_root": aggregate(exact_one) if exact_one else None,
        "both_timings_correct_native": mean(all(v) for v in native_pairs) if native_pairs else None,
        "evaluable_flipped_pairs": len(flips),
        "both_timings_correct_at_one_question": mean(all(want == got for want, got in v) for v in flips)
        if flips
        else None,
        "first_question_rate_by_agenda": {k: mean(v) for k, v in first_by_agenda.items()},
    }


def paired_difference(left, right, metric="future_supported", draws=5000):
    seeds = sorted(left)
    if not seeds or set(left) != set(right):
        raise ValueError("Expect equal nonempty seed sets")
    packets = sorted(left[seeds[0]]["packets"])
    for method in [left, right]:
        if any(set(method[s]["packets"]) != set(packets) for s in seeds):
            raise ValueError("Mismatched packets")

    def delta(s, p):
        return left[s]["packets"][p]["metrics"][metric] - right[s]["packets"][p]["metrics"][metric]

    differences = [mean(delta(s, p) for s in seeds) for p in packets]
    rng = Random(60421)
    boot = sorted(mean(rng.choices(differences, k=len(packets))) for _ in range(draws))
    return {
        "estimate": mean(differences),
        "interval": [boot[int(0.025 * draws)], boot[int(0.975 * draws) - 1]],
        "per_seed": {s: mean(delta(s, p) for p in packets) for s in seeds},
        "packets": len(packets),
        "draws": draws,
    }
