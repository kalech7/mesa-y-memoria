"""Recompute the RL reserve tables on CPU from frozen prompts and GPU margins."""
import argparse
from pathlib import Path
from random import Random
from statistics import mean

from research.src.artifacts import file_hash
from research.tesis_ecuador_eval import pair_scores, review
from research.tesis_ecuador_worlds import audit, build_worlds
from research.tesis_timing_v6_eval import group_worlds
from research.tesis_timing_v6_experiment import PROJECT, SEEDS, forest, signature
from research.tesis_timing_v7 import measure, read


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(run, reserve, replay=False):
    plan, lock = read(run / 'PLAN.json'), read(run / 'LOCK.json')
    require(lock['plan_sha256'] == file_hash(run / 'PLAN.json'), 'Selection refers to another plan')
    require(signature(PROJECT, plan['source_sha256']) == plan['source_sha256'], 'Training sources changed')
    execution = read(run / 'EXECUTION_INPUTS.json')
    require(signature(PROJECT, execution['sources']) == execution['sources'], 'Evaluation sources changed')
    require(execution['base'] == plan['qwen'] and execution['parent'] == plan['parent'], 'Model signatures differ')
    seal = read(run / 'RESERVE_LOCK.json')
    require(seal['lock_sha256'] == file_hash(run / 'LOCK.json'), 'Reserve refers to another selection')
    require(signature(reserve, seal['reserve_sha256']) == seal['reserve_sha256'], 'Reserve inputs changed')
    require(audit(read(reserve / 'PARES.json')) == seal['audit'], 'SQLite audit differs')
    require(review(reserve)['matching_orientations'] == 60, 'Incomplete review')
    manifest = read(run / 'ADAPTADORES.json')
    for seed, arms in lock['selected'].items():
        for arm, choice in arms.items():
            relative = f'adaptadores/seed-{seed}/{arm}'
            portable = manifest[relative]
            require(signature(run / relative, portable) == portable, 'Published adapter changed')
            require(portable['adapter_model.safetensors'] == choice['weights']['adapter_model.safetensors'],
                    'Published tensors differ from selected tensors')
            fit = read(run / f'seed-{seed}' / arm / 'FIT.json')
            require(all(fit[k] == choice[k] for k in choice), 'Fit and selection disagree')
    roots = group_worlds(forest(build_worlds(read(reserve / 'PARES.json')), 1))
    raw, result = read(run / 'RESERVE_MARGINS.json'), read(run / 'RESERVE_RESULTS.json')
    require(raw['lock_sha256'] == result['lock_sha256'] == file_hash(run / 'RESERVE_LOCK.json'), 'Wrong reserve seal')
    require(set(raw['methods']) == set(result['methods']), 'Missing a method')
    scorer = None
    if replay:
        from research.tesis_timing_v6_train import MarginScorer, MODEL, PARENT
        from research.tesis_timing_v6_experiment import BASE_FILES, WEIGHTS
        from research.tesis_timing_v6_eval import score_public
        require(signature(MODEL, BASE_FILES) == plan['qwen'], 'The Qwen base differs')
        expected_parent = dict(plan['parent'])
        parent = signature(PARENT, WEIGHTS)
        if parent != expected_parent:
            conversion = read(PROJECT / 'reports/tesis_timing_qwen_a100_v6/PROCEDENCIA.json')
            expected_parent['adapter_config.json'] = conversion['configuraciones_sha256'][expected_parent['adapter_config.json']]
        require(parent == expected_parent, 'The common parent differs')
        scorer = MarginScorer()
    for arm, seeds in raw['methods'].items():
        require(set(seeds) == set(map(str, SEEDS)), 'Missing a seed')
        for seed, margins in seeds.items():
            if arm == 'sft-pares-v7':
                cutoff = read(PROJECT / 'reports/tesis_timing_pares_a100_v7/LOCK.json')['selected'][seed]['sft-pares']['cutoff']
            else:
                cutoff = lock['selected'][seed][arm]['cutoff']
            if replay:
                path = (PROJECT / f'reports/tesis_timing_pares_a100_v7/adaptadores/seed-{seed}/sft-pares'
                        if arm == 'sft-pares-v7' else run / f'adaptadores/seed-{seed}/{arm}')
                scorer.load(path)
                recomputed = {key: list(values) for key, values in score_public(roots, scorer).items()}
                require(recomputed == margins, 'CUDA replay differs from archived margins')
            expected = {**measure(roots, margins, cutoff), 'hits': pair_scores(roots, margins)}
            require(expected == result['methods'][arm][seed], 'Raw margins and metrics disagree')
    control = result['methods']['sft-pares-control']
    for arm, contrast in result['comparisons_against_sft_control'].items():
        observed = result['methods'][arm]
        deltas = [mean(observed[str(seed)]['hits'][pair] - control[str(seed)]['hits'][pair] for seed in SEEDS)
                  for pair in sorted(control[str(SEEDS[0])]['hits'])]
        rng = Random(231127)
        draws = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(5000))
        expected = {'pair_difference': mean(deltas), 'pair_interval95': [draws[125], draws[4874]],
                    'matched_future_difference': mean(observed[str(seed)]['macro']['future_supported'] -
                                                       control[str(seed)]['macro']['future_supported'] for seed in SEEDS)}
        require(contrast == expected, 'Recomputed contrast differs')
    require(pair_scores(roots, raw['planner']) == result['planner_hits'], 'Planner result differs')
    historical_path = run / 'HISTORICAL_BASELINES.json'
    if historical_path.exists():
        historical = read(historical_path)
        old_lock = read(PROJECT / 'reports/tesis_timing_qwen_a100_v6/LOCK.json')
        for arm, seeds in historical['margins'].items():
            for seed, margins in seeds.items():
                cutoff = old_lock['selected'][seed][arm]['cutoff']
                expected = {**measure(roots, margins, cutoff), 'hits': pair_scores(roots, margins)}
                require(expected == historical['methods'][arm][seed], 'Historical baseline differs')
    print(f'{run.name}: corpus, sources, adapters, six decisions per pair and contrasts verified.')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run', type=Path, required=True)
    cli.add_argument('--reserve', type=Path, required=True)
    cli.add_argument('--replay', action='store_true')
    args = cli.parse_args()
    verify(args.run, args.reserve, args.replay)
