"""Development-only comparison of RL credit and exploration after a common SFT warm start."""
import argparse
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from random import Random
from statistics import mean

from research.src.artifacts import digest, file_hash
from research.tesis_shared_budget_qwen import atomic_json, batches, padded_scoring
from research.tesis_timing_v6_bank import build_worlds
from research.tesis_timing_v6_eval import group_worlds, score_public
from research.tesis_timing_v6_experiment import BASE_FILES, PROJECT, SEEDS, WEIGHTS, calibrate, forest, signature
from research.tesis_timing_v6_train import MODEL, PARENT, MarginScorer
from research.tesis_timing_v7 import loss_for, measure, read
from research.tesis_rl_v8_rewards import TrainRewards

ARMS = ('sft-pares-control', 'grpo-exacto', 'grpo-bellman')
EPOCHS = (2, 4, 6, 8)
V7 = PROJECT / 'reports/tesis_timing_pares_a100_v7'
T, LR, KL, ENTROPY, CLIP = 4., 3e-5, .005, .01, .2
SOURCES = ('research/tesis_rl_v8.py', 'research/tesis_rl_v8_rewards.py')


def relative_loss(margins, rewards, old_margins, anchor_margins):
    import torch
    logits = torch.stack((torch.zeros_like(margins), margins / T), dim=-1)
    logp = logits.log_softmax(-1)
    old_logits = torch.stack((torch.zeros_like(old_margins), old_margins / T), dim=-1)
    old = old_logits.log_softmax(-1).detach()
    ref = torch.stack((torch.zeros_like(anchor_margins), anchor_margins / T), dim=-1).log_softmax(-1).detach()
    mass = old.exp()
    scale = rewards.std(-1, unbiased=False, keepdim=True).clamp_min(.05)
    advantage = ((rewards - (mass * rewards).sum(-1, keepdim=True)) / scale).detach()
    ratio = (logp - old).clamp(-30, 30).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1-CLIP, 1+CLIP) * advantage)
    kl = (logp.exp() * (logp - ref)).sum(-1)
    entropy = -(logp.exp() * logp).sum(-1)
    loss = -(mass * surrogate).sum(-1) + KL * kl - ENTROPY * entropy
    return loss.mean(), {'entropy': float(entropy.mean().detach()), 'kl': float(kl.mean().detach()),
                         'clipped_fraction': float(((ratio-1).abs() > CLIP).float().mean().detach())}


def initialize(run):
    import torch
    config = {'seeds': list(SEEDS), 'arms': list(ARMS), 'epochs': list(EPOCHS),
              'learning_rate': LR, 'temperature': T, 'kl': KL, 'entropy': ENTROPY, 'clip': CLIP,
              'advantage_scale': 'per-state uniform action std, minimum .05; policy-weighted baseline',
              'training': 'TRAIN only, budget1, exact two-action continuation expectation',
              'batch': 'two complete pairs, four observations per pair; one optimizer update per batch',
              'warm_start': 'v7 sft-pares selected in historical validation for corresponding seed',
              'selection': 'validation complete-pair accuracy, then matched future accuracy, then earlier epoch',
              'scope': 'development; historical reserves are previously observed; fresh reserve required after locking',
              'source_sha256': signature(PROJECT, SOURCES), 'qwen': signature(MODEL, BASE_FILES),
              'parent': signature(PARENT, WEIGHTS),
              'splits': {split: digest(build_worlds(split)) for split in ('train', 'validation')},
              'warm_weights': {str(s): signature(V7 / f'adaptadores/seed-{s}/sft-pares', WEIGHTS) for s in SEEDS},
              'environment': {'gpu': torch.cuda.get_device_name(), **{k: version(k) for k in ('torch', 'peft', 'transformers')}}}
    run.mkdir(parents=True, exist_ok=True)
    path = run / 'PLAN.json'
    if path.exists() and read(path) != config:
        raise ValueError('Another plan occupies this directory')
    atomic_json(path, config)


def fit(run):
    import torch
    initialize(run)
    bank = TrainRewards()
    records = bank.records
    indexed = defaultdict(dict)
    for i, row in enumerate(records):
        indexed[row.pair][row.variant, row.visit] = i
    groups = []
    for pair, members in sorted(indexed.items()):
        groups.append([members['ask_now', 1], members['save', 1], members['ask_now', 2], members['save', 2]])
    valid = group_worlds(forest(build_worlds('validation'), 1))
    scorer = MarginScorer(trainable=True)
    texts = [r.public for r in records]
    results = {}
    for seed in SEEDS:
        warm = V7 / f'adaptadores/seed-{seed}/sft-pares'
        scorer.load(warm)
        anchor = dict(zip(texts, scorer.score_texts(texts), strict=True))
        baseline = measure(valid, score_public(valid, scorer), calibrate(valid, score_public(valid, scorer))[0]['cutoff'])
        results[str(seed)] = {'warm_start': baseline, 'arms': {}}
        for arm in ARMS:
            directory = run / f'seed-{seed}' / arm
            finished = directory / 'FIT.json'
            if finished.exists():
                result = read(finished)
                if signature(directory / result['adapter'], WEIGHTS) != result['weights']:
                    raise ValueError('Selected checkpoint changed')
                results[str(seed)]['arms'][arm] = result
                continue
            directory.mkdir(parents=True, exist_ok=True)
            scorer.active = None
            scorer.load(warm)
            optimizer = torch.optim.AdamW((p for p in scorer.model.parameters() if p.requires_grad), lr=LR)
            torch.manual_seed(seed)
            candidates, training, counter = [], [], Counter()
            for epoch in range(1, EPOCHS[-1]+1):
                scorer.cache.clear()
                old = dict(zip(texts, scorer.score_texts(texts), strict=True))
                rewards = bank.exact_returns(old, continuation='optimal' if arm == 'grpo-bellman' else 'policy', temperature=T)
                audit = bank.audit(rewards)
                order = list(range(len(groups)))
                Random(seed+epoch).shuffle(order)
                scorer.model.train()
                scorer.active = None
                diagnostics, losses, grad_norms = [], [], []
                for selection in batches(order, 2):
                    indices = [index for group in selection for index in groups[group]]
                    encoded = [scorer.record(texts[i]) for i in indices]
                    logits = padded_scoring(scorer.model, encoded, scorer.labels, scorer.tokenizer.pad_token_id, 'cuda', counter)
                    margins = logits[:, 1] - logits[:, 0]
                    if arm == ARMS[0]:
                        loss = loss_for(margins, 'sft-pares')
                        stats = {}
                    else:
                        tensor = lambda values: torch.tensor(values, device='cuda', dtype=torch.float32)
                        loss, stats = relative_loss(margins, tensor([rewards[i] for i in indices]),
                                                    tensor([old[texts[i]] for i in indices]),
                                                    tensor([anchor[texts[i]] for i in indices]))
                    if not bool(loss.isfinite()):
                        raise ValueError('Nonfinite loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(scorer.model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    counter['optimizer_updates'] += 1
                    grad_norms.append(float(norm))
                    losses.append(float(loss.detach()))
                    diagnostics.append(stats)
                training.append({'epoch': epoch, 'loss': mean(losses), 'gradient_norm_before_clip': mean(grad_norms),
                                 'reward_audit': audit, 'old_abs_margin_mean': mean(abs(x) for x in old.values()),
                                 'old_ask_fraction': mean(x > 0 for x in old.values()),
                                 'policy': {k: mean(row[k] for row in diagnostics) for k in diagnostics[0]}})
                if epoch in EPOCHS:
                    scorer.cache.clear()
                    margins = score_public(valid, scorer)
                    cutoff = calibrate(valid, margins)[0]['cutoff']
                    metrics = measure(valid, margins, cutoff)
                    adapter = f'epoch-{epoch}/adapter'
                    scorer.model.save_pretrained(directory / adapter, safe_serialization=True)
                    candidates.append({'epoch': epoch, 'adapter': adapter, 'cutoff': cutoff, 'metrics': metrics})
                    print(seed, arm, epoch, 'validation pairs', metrics['individual']['both_timings_correct_native'],
                          'future', metrics['macro']['future_supported'], flush=True)
            selected = max(candidates, key=lambda r: (r['metrics']['individual']['both_timings_correct_native'],
                                                      r['metrics']['macro']['future_supported'], -r['epoch']))
            result = {**selected, 'weights': signature(directory / selected['adapter'], WEIGHTS),
                      'candidates': candidates, 'training': training, 'budget': dict(counter)}
            atomic_json(finished, result)
            results[str(seed)]['arms'][arm] = result
            del optimizer
    for data in results.values():
        budgets = [data['arms'][arm]['budget'] for arm in ARMS]
        if any(budget != budgets[0] for budget in budgets[1:]):
            raise ValueError('Unequal optimization budgets within a seed')
    keys = ('optimizer_updates', 'input_tokens', 'decisions', 'forward_batches')
    across_seeds = [tuple(data['arms'][ARMS[0]]['budget'][key] for key in keys) for data in results.values()]
    if len(set(across_seeds)) != 1:
        raise ValueError('Unequal unpadded training budgets across seeds')
    atomic_json(run / 'VALIDATION.json', {'plan_sha256': file_hash(run / 'PLAN.json'), 'seeds': results})
    lock = {'plan_sha256': file_hash(run / 'PLAN.json'), 'selected': {
        seed: {arm: {k: result[k] for k in ('epoch', 'adapter', 'weights', 'cutoff')}
               for arm, result in data['arms'].items()} for seed, data in results.items()}}
    atomic_json(run / 'LOCK.json', lock)


def score_reserve(run, reserve):
    from research.tesis_ecuador_worlds import audit, build_worlds as external_worlds
    from research.tesis_ecuador_eval import pair_scores, review
    from research.tesis_timing_v6_env import fit_public_planner
    from research.tesis_timing_v6_experiment import PublicScorer, trees
    import torch

    plan, lock = read(run / 'PLAN.json'), read(run / 'LOCK.json')
    if plan['source_sha256'] != signature(PROJECT, SOURCES) or lock['plan_sha256'] != file_hash(run / 'PLAN.json'):
        raise ValueError('The selected experimental inputs changed')
    if review(reserve)['matching_orientations'] != 60:
        raise ValueError('The new reserve needs a complete independent review')
    inputs = signature(reserve, ('PARES.json', 'CLAVE.json', 'REVISION_CIEGA.json', 'JUICIOS.json'))
    seal = {'lock_sha256': file_hash(run / 'LOCK.json'), 'reserve_sha256': inputs,
            'audit': audit(read(reserve / 'PARES.json'))}
    seal_file = run / 'RESERVE_LOCK.json'
    if seal_file.exists() and read(seal_file) != seal:
        raise ValueError('The reserve was already opened under another seal')
    atomic_json(seal_file, seal)
    roots = group_worlds(forest(external_worlds(read(reserve / 'PARES.json')), 1))
    scorer = MarginScorer()
    methods, raw = defaultdict(dict), defaultdict(dict)
    for seed in map(str, SEEDS):
        choices = dict(lock['selected'][seed])
        warm_fit = read(V7 / 'LOCK.json')['selected'][seed]['sft-pares']
        choices['sft-pares-v7'] = warm_fit
        for arm, selection in choices.items():
            if arm == 'sft-pares-v7':
                path = V7 / f'adaptadores/seed-{seed}/sft-pares'
                expected = read(V7 / 'ADAPTADORES.json')[f'adaptadores/seed-{seed}/sft-pares']
            else:
                path = run / f'seed-{seed}' / arm / selection['adapter']
                expected = selection['weights']
            if signature(path, expected) != expected:
                raise ValueError('Selected weights changed')
            scorer.load(path)
            margins = score_public(roots, scorer)
            raw[arm][seed] = margins
            metrics = measure(roots, margins, selection['cutoff'])
            methods[arm][seed] = {**metrics, 'hits': pair_scores(roots, margins)}
    planner = PublicScorer(fit_public_planner(trees('train', 1)))
    planner_margins = score_public(roots, planner)
    reference = methods[ARMS[0]]
    comparisons = {}
    for arm in (*ARMS[1:], 'sft-pares-v7'):
        deltas = [mean(methods[arm][str(seed)]['hits'][pair] - reference[str(seed)]['hits'][pair]
                       for seed in SEEDS) for pair in sorted(reference[str(SEEDS[0])]['hits'])]
        rng = Random(231127)
        samples = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(5000))
        comparisons[arm] = {'pair_difference': mean(deltas), 'pair_interval95': [samples[125], samples[4874]],
                            'matched_future_difference': mean(methods[arm][str(seed)]['macro']['future_supported'] -
                                reference[str(seed)]['macro']['future_supported'] for seed in SEEDS)}
    atomic_json(run / 'RESERVE_MARGINS.json', {'lock_sha256': file_hash(seal_file), 'gpu': torch.cuda.get_device_name(),
                                             'methods': raw, 'planner': planner_margins})
    atomic_json(run / 'RESERVE_RESULTS.json', {'lock_sha256': file_hash(seal_file), 'methods': methods,
        'comparisons_against_sft_control': comparisons, 'planner_hits': pair_scores(roots, planner_margins)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=PROJECT / 'reports/tesis_rl_credit_a100_v8')
    parser.add_argument('--reserve', type=Path, help='Score a newly reviewed reserve after training and selection')
    args = parser.parse_args()
    if args.reserve:
        score_reserve(args.run.resolve(), args.reserve.resolve())
    else:
        fit(args.run.resolve())


if __name__ == '__main__':
    main()
