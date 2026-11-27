#!/usr/bin/env python3
"""Execute the prospectively declared O1/O2 supplement after the primary pilot.

No training, checkpoint selection, remote inference, installation or downloads.
The original B0 pretraining ablation and primary B2 comparisons stay unchanged.
All B0 plus all three locked B1 seeds are mandatory, regardless of their results.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import fmean
import sys
import time


REPO = Path(__file__).resolve().parents[1]
FIELDS = ('return', 'error_rate', 'future_accuracy', 'format_errors', 'format_error_rate')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'Empty primary result: {path}')
    return rows


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (index - low) * (values[high] - values[low])


def interval(values, repetitions, seed):
    rng = random.Random(seed)
    estimates = [fmean(rng.choices(values, k=len(values))) for _ in range(repetitions)]
    return {'mean': fmean(values), 'ci95': [percentile(estimates, .025), percentile(estimates, .975)],
            'users': len(values)}


def grouped_values(rows, field):
    groups = defaultdict(list)
    for row in rows:
        value = (row.get('format_errors', 0) / max(1, len(row['steps'])) if field == 'format_error_rate'
                 else row[field])
        groups[row['user_id']].append(float(value))
    return {user: fmean(values) for user, values in groups.items()}


def summarize(rows, protocol):
    return {field: interval(list(grouped_values(rows, field).values()),
                            protocol['bootstrap_repetitions'], protocol['bootstrap_seed'])
            for field in FIELDS}


def paired(o2, o1, protocol):
    result = {}
    for field in FIELDS:
        a, b = grouped_values(o2, field), grouped_values(o1, field)
        if a.keys() != b.keys():
            raise ValueError('O1/O2 must contain exactly the same complete users')
        result[field] = interval([a[user] - b[user] for user in sorted(a)],
                                 protocol['bootstrap_repetitions'], protocol['bootstrap_seed'])
    return result


def verify_completed(root, protocol_path):
    """Gate before opening evaluator_private/test or test model result files."""
    status = load(root / 'run_status.json')
    if status.get('status') != 'COMPLETED_TECHNICAL_PILOT' or status.get('test_evaluated') is not True:
        raise ValueError('Primary pilot must have finished training, selection and final TEST first')
    freeze = load(root / 'final_freeze.json')
    if freeze.get('before_test_evaluation') is not True:
        raise ValueError('Primary checkpoint/configuration freeze is missing')
    protocol = load(protocol_path)
    config = protocol['configuration']
    if digest(config) != protocol['config_hash']:
        raise ValueError('Prospective supplemental protocol configuration hash mismatch')
    if (protocol.get('prespecified_before_main_test') is not True or protocol.get('test_read') is not False
            or protocol.get('test_evaluated') is not False):
        raise ValueError('Prospective protocol must explicitly predate TEST and declare no TEST use')
    bound = protocol['pilot_binding']
    implementation = protocol['implementation']
    if sha256(__file__) != implementation['source_sha256']:
        raise ValueError('Supplement implementation changed after prospective registration')
    if Path(bound['pilot_path']).resolve() != root:
        raise ValueError('Supplement is bound to a different primary pilot')
    freeze_at = datetime.fromisoformat(freeze['created_at'])
    if any(datetime.fromisoformat(protocol[key]) > freeze_at for key in ('created_at',)) or (
            datetime.fromisoformat(bound['bound_at']) > freeze_at) or (
            datetime.fromisoformat(implementation['prepared_at']) > freeze_at):
        raise ValueError('Protocol and exact pilot binding must predate the main final TEST freeze')
    record = load(root / 'config.json')
    if (record['provenance']['config_hash'] != bound['primary_config_hash'] or
            digest(record['configuration']) != bound['primary_config_hash'] or
            sha256(root / 'config.json') != bound['primary_config_file_sha256']):
        raise ValueError('Bound primary experiment configuration changed')
    if sorted(record['configuration']['seeds']) != sorted(config['training_seeds']):
        raise ValueError('All prespecified B1 seeds must be included')
    if len(config['training_seeds']) != 3 or config['methods'] != ['B0', 'B1']:
        raise ValueError('Protocol requires B0 plus all three B1 seeds')
    manifest = load(root / 'dataset_manifest.json')
    if not (freeze['test_hash'] == bound['test_frozen_hash'] == manifest['test_frozen_hash']):
        raise ValueError('Frozen TEST hash changed')
    if manifest['splits']['test']['users'] != config['test_users']:
        raise ValueError('Supplement must use all prespecified TEST users')
    sources = load(root / 'source_hashes.json')['files']
    for source, expected in sources.items():
        if sha256(REPO / source) != expected:
            raise ValueError(f'Primary frozen source changed: {source}')
    selected = freeze['observation']
    if selected not in ('O1', 'O2'):
        raise ValueError('Only the two prespecified observations are permitted')
    return protocol, record['configuration'], freeze, selected


def annotate(rows):
    """Persist actual exposed evidence counts beside the unmodified raw traces."""
    for row in rows:
        counts = []
        for generation in row.get('generations', []):
            prompt = generation.get('prompt', '')
            public = json.loads(prompt.split('OBSERVACION:\n', 1)[1])
            memory = public.get('memory', [])
            counts.append({'session_index': generation['session_index'],
                           'retrieved_evidence_count': len(memory),
                           'evidence_ids': [fact['evidence_id'] for fact in memory],
                           'incoming_evidence_count': len(public.get('incoming_evidence', []))})
        row['supplement_actual_exposure_counts'] = counts
    return rows


def execute(root, protocol_path, output=None):
    started = time.monotonic()
    root, protocol_path = Path(root).resolve(), Path(protocol_path).resolve()
    protocol, main_config, freeze, selected = verify_completed(root, protocol_path)
    config = protocol['configuration']
    destination = Path(output).resolve() if output else root
    destination.mkdir(parents=True, exist_ok=True)
    raw_dir = destination / 'supplementary_observability_raw'
    outputs = [destination / f'supplementary_observability.{extension}' for extension in ('json', 'csv')]
    if raw_dir.exists() or any(path.exists() for path in outputs):
        raise FileExistsError('Preserve prior/partial supplement; use a fresh --output directory')
    complement = 'O1' if selected == 'O2' else 'O2'
    # This is the first access to TEST contents, after the completed-run gate.
    scenarios_path = root / 'evaluator_private' / 'test.json'
    scenarios = load(scenarios_path)['scenarios']
    if digest(scenarios) != freeze['test_hash'] or len(scenarios) != config['test_users']:
        raise ValueError('Actual frozen TEST trajectories do not match the locked hash/count')
    expected_users = {scenario['user_id'] for scenario in scenarios}
    if len(expected_users) != len(scenarios):
        raise ValueError('Expected one complete trajectory per TEST user')
    checkpoints = load(root / 'checkpoint_manifest.json')
    selections = freeze['checkpoints']
    if checkpoints['checkpoints'] != selections:
        raise ValueError('Checkpoint manifest differs from the pre-TEST final freeze')
    gate = load(root / 'prepared_gate.json')
    model_path = gate['selected_model']['path']
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                      OMP_NUM_THREADS=str(config['cpu_threads']), MKL_NUM_THREADS=str(config['cpu_threads']))
    sys.path.insert(0, str(REPO))
    import torch
    torch.set_num_threads(config['cpu_threads'])
    torch.set_num_interop_threads(config['cpu_threads'])
    from llm_local_001.model import LocalPolicy
    from llm_local_001.experiment import evaluate, release

    provenance = {'created_at': datetime.now(timezone.utc).isoformat(), 'source': str(Path(__file__).resolve()),
                  'source_sha256': sha256(__file__), 'protocol_path': str(protocol_path),
                  'protocol_sha256': sha256(protocol_path), 'protocol_config_hash': protocol['config_hash'],
                  'primary_pilot': str(root), 'primary_config_hash': protocol['pilot_binding']['primary_config_hash'],
                  'primary_final_freeze_sha256': sha256(root / 'final_freeze.json'),
                  'frozen_test_hash': freeze['test_hash'], 'primary_observation': selected,
                  'newly_evaluated_observation': complement, 'no_training_or_selection': True,
                  'remote_inference_calls': 0}
    raw_dir.mkdir()
    write_json(raw_dir / 'supplement_started.json', {'provenance': provenance,
               'mandatory_configurations': ['B0'] + [f'B1_seed{seed}' for seed in config['training_seeds']],
               'all_configurations_required_regardless_of_outcome': True})
    primary = {method: read_rows(root / 'test' / f'{method}.jsonl') for method in config['methods']}
    experiments = [('B0', main_config['seed'], None)]
    experiments += [('B1', seed, selections[f'B1_seed{seed}']['selected']['path'])
                    for seed in config['training_seeds']]
    all_rows, summaries, differences, input_hashes = {}, {}, {}, {}
    combined = {'O1': [], 'O2': []}
    csv_rows = []
    for method, seed, checkpoint in experiments:
        key = f'{method}_seed{seed}'
        old = [row for row in primary[method] if method == 'B0' or row.get('seed') == seed]
        if len(old) != config['test_users'] or {row['user_id'] for row in old} != expected_users:
            raise ValueError(f'Missing or duplicated primary users for {key}')
        if any(row['representation'] != selected for row in old):
            raise ValueError(f'Primary rows have unexpected observation: {key}')
        print(f'Suplemento bloqueado previamente: {key}, {complement}, mismos {len(scenarios)} usuarios.', flush=True)
        if checkpoint is None:
            policy = LocalPolicy(model_path, seed=seed, device=config['execution_device'],
                                 lora=not main_config['full_finetune'], full_finetune=main_config['full_finetune'],
                                 max_new_tokens=main_config['max_new_tokens'])
        else:
            # Existing loader checks exact adapter/base artifact hashes before
            # generating. No optimizer or trainable adapter is enabled.
            policy = LocalPolicy.from_checkpoint(checkpoint, device=config['execution_device'])
            expected_manifest = selections[f'B1_seed{seed}']['selected']['manifest']
            if load(Path(checkpoint) / 'local_checkpoint.json') != expected_manifest:
                raise ValueError(f'Selected B1 manifest changed since main freeze: {key}')
        if policy.base_fingerprint() != checkpoints['base_fingerprint']:
            raise ValueError(f'Base model/tokenizer differs from primary comparison: {key}')
        new = evaluate(policy, scenarios, complement, method=method, seed=seed,
                       directory=raw_dir / 'sqlite' / key)
        release(policy)
        del policy
        torch.cuda.empty_cache()
        values = {selected: annotate(old), complement: annotate(new)}
        all_rows[key] = values
        summaries[key] = {obs: summarize(values[obs], config) for obs in ('O1', 'O2')}
        differences[key] = paired(values['O2'], values['O1'], config)
        for obs in ('O1', 'O2'):
            path = raw_dir / f'{key}_{obs}.jsonl'
            with path.open('w', encoding='utf-8') as handle:
                for row in values[obs]:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            input_hashes[str(path)] = sha256(path)
            if method == 'B1':
                combined[obs].extend(values[obs])
            for field in FIELDS:
                result = summaries[key][obs][field]
                csv_rows.append({'method': method, 'seed': seed, 'observation': obs, 'comparison': None,
                                 'metric': field, **result})
        for field, result in differences[key].items():
            csv_rows.append({'method': method, 'seed': seed, 'observation': None,
                             'comparison': 'O2-O1', 'metric': field, **result})
        write_json(raw_dir / f'{key}_completed.json', {'provenance': provenance,
                   'summaries': summaries[key], 'paired_O2_minus_O1': differences[key],
                   'primary_reused': selected, 'newly_generated': complement})
    differences['B1_ALL_SEEDS'] = paired(combined['O2'], combined['O1'], config)
    summaries['B1_ALL_SEEDS'] = {obs: summarize(combined[obs], config) for obs in ('O1', 'O2')}
    for field, result in differences['B1_ALL_SEEDS'].items():
        csv_rows.append({'method': 'B1', 'seed': 'ALL_SEEDS', 'observation': None,
                         'comparison': 'O2-O1', 'metric': field, **result})
    for path in (scenarios_path, root / 'test' / 'B0.jsonl', root / 'test' / 'B1.jsonl',
                 root / 'checkpoint_manifest.json', root / 'source_hashes.json'):
        input_hashes[str(path)] = sha256(path)
    result = {'provenance': provenance, 'status': 'COMPLETED_PRESPECIFIED_SUPPLEMENT',
              'all_locked_configurations_evaluated': True, 'elapsed_seconds': time.monotonic() - started,
              'summaries': summaries, 'paired_O2_minus_O1': differences,
              'input_and_raw_logs_sha256': input_hashes,
              'confidence_unit': 'complete user; B1 seeds averaged within user for aggregate estimate',
              'interpretation': ['B0 pretraining O1/O2 remains the original validation comparison; its format floor is not repaired retroactively.',
                                 'This held-checkpoint TEST supplement measures the information effect separately at base and supervised competence levels.',
                                 'All B1 selected seeds are reported; no favorable seed, checkpoint, family or metric was selected.',
                                 'Primary B2-B1/B3 RL comparisons, checkpoint selection, prompts and frozen rewards remain unchanged.',
                                 'Supplementary O1/O2 trajectories may diverge after policy actions; user/scenario/checkpoint are paired, not post-action states.',
                                 'A positive B1 historical-memory effect is not an RL benefit or general proof of thesis value.']}
    write_json(outputs[0], result)
    columns = ['created_at', 'source_sha256', 'protocol_config_hash', 'frozen_test_hash', 'method', 'seed',
               'observation', 'comparison', 'metric', 'mean', 'ci95', 'users']
    with outputs[1].open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({key: provenance[key] for key in columns[:4]} |
                            {**row, 'ci95': json.dumps(row['ci95'])})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot', type=Path)
    parser.add_argument('--protocol', type=Path,
                        default=REPO / 'reports/llm_local_001_preparation/locked_observability_protocol.json')
    parser.add_argument('--output', type=Path, help='Default: completed primary pilot directory')
    args = parser.parse_args(argv)
    try:
        result = execute(args.pilot, args.protocol, args.output)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'Supplement refused: {error}\n')
    print(f"{result['status']}: {result['elapsed_seconds']:.1f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
