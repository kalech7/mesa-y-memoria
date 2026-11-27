"""Development ablation adding reward-derived agenda ranking to the exact RL actor."""
import argparse
from pathlib import Path

from research import tesis_rl_v8 as experiment
from research.tesis_timing_v6_experiment import PROJECT

PAIR_WEIGHT = 1.
BASE_LOSS = experiment.relative_loss


def paired_actor_loss(margins, rewards, old_margins, anchor_margins):
    import torch.nn.functional as F

    loss, diagnostics = BASE_LOSS(margins, rewards, old_margins, anchor_margins)
    # Batches contain full pairs ordered V1-positive, V1-negative, then both V2.
    # The direction is derived from TRAIN counterfactual returns, never from
    # a label exposed in the model's observation.
    gaps = (rewards[:, 1] - rewards[:, 0]).reshape(-1, 4)
    direction = (gaps[:, 0] - gaps[:, 1]).sign().detach()
    first = margins.reshape(-1, 4)[:, :2]
    ranking = F.softplus(direction * (first[:, 1] - first[:, 0])).mean()
    return loss + PAIR_WEIGHT * ranking, {**diagnostics, 'pair_ranking_loss': float(ranking.detach())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=PROJECT / 'reports/tesis_rl_credit_pares_a100_v8b')
    parser.add_argument('--reserve', type=Path)
    args = parser.parse_args()
    experiment.ARMS = ('sft-pares-control', 'grpo-exacto-pares')
    experiment.SOURCES = (*experiment.SOURCES, 'research/tesis_rl_v8b.py')
    experiment.relative_loss = paired_actor_loss
    if args.reserve:
        experiment.score_reserve(args.run.resolve(), args.reserve.resolve())
    else:
        experiment.fit(args.run.resolve())


if __name__ == '__main__':
    main()
