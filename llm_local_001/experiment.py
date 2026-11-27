"""Verified local runtime stage. Normally launched by llm_local_001.run.

No network inference, package installs, symbolic policy heads or hidden labels.
This stage is unreachable until the resource, dataset and local-test gates pass.
"""
from dataclasses import asdict, replace
import gc
import json
import shutil
from pathlib import Path
import random
import sys
from time import monotonic

from .artifacts import digest, file_hash, now, write_csv, write_json, write_jsonl
from .environment import (LLMEnvironment, format_prompt, historical_rule_policy,
                          run_episode, run_historical_episode, serialize_rule_action)
from .metrics import error_rows, paired, summarize
from .model import LocalPolicy, artifact_hashes
from .report import BASE_FIELDS, CURVE_FIELDS, ERROR_FIELDS, OBS_FIELDS, csv_summary
from .training import EpisodeTrajectory, TrainingConfig, TrainingResult, train_rl, train_sft


def release(policy):
    torch = policy.torch
    policy.model.to('cpu')
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate(policy, scenarios, observation, *, method, seed, condition='ORACLE', directory=None, log_path=None):
    rows = []
    for i, scenario in enumerate(scenarios):
        path = directory / f'{i}.sqlite' if directory is not None else None
        generated = []
        def generate(prompt):
            item = policy.generate(prompt)
            generated.append(item)
            return item.text
        row = run_episode(scenario, generate, observation, condition, path)
        row.update(method=method, seed=seed)
        row.update(prompt_tokens=sum(len(g.prompt_ids) for g in generated),
                   completion_tokens=sum(len(g.completion_ids) for g in generated),
                   generation_truncations=sum(g.truncated for g in generated))
        rows.append(row)
        if log_path:
            write_jsonl(log_path, rows)
        if (i+1) % 10 == 0 or i+1 == len(scenarios):
            print(f'{method} seed={seed}: {i+1}/{len(scenarios)} usuarios, retorno parcial '
                  f'{sum(r["return"] for r in rows)/len(rows):.3f}', flush=True)
    return rows


def demonstrations(scenarios, observation):
    """TRAIN-only teacher actions from observable rules, not simulator gold."""
    from rl_observability.policies import HistoricalClarificationRules
    from copy import deepcopy
    rules = HistoricalClarificationRules()
    demos = []
    for scenario in scenarios:
        env = LLMEnvironment(scenario, observation, 'ORACLE')
        try:
            obs = env.reset()
            for _ in range(4):
                raw = serialize_rule_action(rules.select_action(deepcopy(env._observation)), env._observation)
                demos.append({'prompt': format_prompt(obs), 'completion': raw,
                              'user_id': scenario['user_id'], 'partition': 'train',
                              'teacher': 'native_historical_rules_no_gold'})
                obs, _, done, truncated, _ = env.step(raw)
                if done or truncated:
                    break
        finally:
            env.close()
    return demos


def verify_gate(output):
    gate = json.loads((output / 'prepared_gate.json').read_text())
    config_record = json.loads((output / 'config.json').read_text())
    config = config_record['configuration']
    if gate['status'] != 'ready' or digest(config) != config_record['provenance']['config_hash']:
        raise ValueError('Prepared resource/configuration gate failed')
    for source, sha in gate['source_hashes'].items():
        if file_hash(Path(source)) != sha:
            raise ValueError(f'Source changed after preflight: {source}; use a fresh run')
    splits = {name: json.loads((output / 'evaluator_private' / f'{name}.json').read_text())['scenarios']
              for name in ('train', 'validation', 'test')}
    if digest(splits['test']) != gate['test_hash']:
        raise ValueError('Frozen test changed before training')
    return config_record['provenance'], config, gate, splits


def reusable_stages(config, splits, output, metadata):
    """Reuse identical completed stages only; never choose a policy using TEST.

    Model-independent memory fixes can otherwise force expensive retraining of
    already audited B1. Original artifacts stay immutable with original dates.
    """
    if not config.get('reuse_completed_stages'):
        return None
    previous = Path(config['reuse_completed_stages']).resolve()
    old = json.loads((previous / 'config.json').read_text())['configuration']
    keys = ('model_path', 'device', 'full_finetune', 'gradient_checkpointing', 'cpu_threads',
            'reference_device', 'train_users', 'validation_users', 'test_users', 'seed', 'seeds',
            'sft_updates', 'continuation_updates', 'batch_size', 'learning_rate', 'kl_coefficient',
            'baseline_decay', 'validation_every', 'max_new_tokens', 'bootstrap')
    if any(config[key] != old[key] for key in keys):
        raise ValueError('Cannot reuse stages with different model/data/training hyperparameters')
    if (previous / 'final_freeze.json').exists() or any((previous / 'test').glob('*.jsonl')):
        raise ValueError('Stage reuse is restricted to runs which never evaluated TEST')
    for partition, scenarios in splits.items():
        old_rows = json.loads((previous / 'evaluator_private' / f'{partition}.json').read_text())['scenarios']
        if digest(old_rows) != digest(scenarios):
            raise ValueError('Frozen trajectories differ from the reuse source')
    for source in ('llm_local_001/environment.py', 'rl_observability/environment.py',
                   'rl_small_pilot/environment_adapter.py', 'llm_local_001/data.py'):
        if file_hash(Path(source)) != file_hash(previous / 'source_snapshot' / source):
            raise ValueError(f'Observation/action/data semantics changed: {source}; stages cannot be reused')
    record = {'provenance': metadata, 'source_run': str(previous), 'test_previously_used': False,
              'compatible_configuration_keys': list(keys), 'artifacts': {},
              'reason': 'Resume completed stages after interrupted local execution; no hyperparameter or reward change',
              'reused_stages': {}}
    for source in (previous / 'validation/observability').glob('*.jsonl'):
        target = output / 'validation/observability' / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        record['artifacts'][str(source)] = file_hash(source)
    for name in ('observation_selection.json', 'diagnostics_before_training.json'):
        if (previous / name).exists():
            record['artifacts'][str(previous / name)] = file_hash(previous / name)
    record['completed_B1'] = {}
    for seed in config['seeds']:
        source = previous / 'checkpoints' / str(seed) / 'B1/training_result.json'
        if source.exists():
            record['completed_B1'][str(seed)] = {'path': str(source), 'sha256': file_hash(source)}
    write_json(output / 'stage_reuse_manifest.json', record)
    return previous


def clone_completed_training(previous, output, seed, method, expected_config, *, b1_path=None,
                             target_tokens=None):
    """Clone a fully completed audited stage, preserving original provenance.

    Missing final results are partial runs and must be retrained from scratch.
    Result paths are relocated; checkpoint files and their origin metadata stay
    byte-identical. No model loading or evaluation is needed for this operation.
    """
    if previous is None:
        return None
    source = previous / 'checkpoints' / str(seed) / method
    result_file = source / 'training_result.json'
    if not result_file.exists():
        return None
    result = TrainingResult(**json.loads(result_file.read_text()))
    if result.config != asdict(expected_config):
        raise ValueError(f'Completed {method} training configuration differs from requested budget')
    expected_algorithm = 'REINFORCE_EMA_EXACT_CONDITIONAL_KL' if method == 'B2' else 'SFT'
    if result.algorithm != expected_algorithm:
        raise ValueError(f'Completed {method} algorithm differs')
    if [point['step'] for point in result.history] != list(range(expected_config.updates + 1)):
        raise ValueError(f'Completed {method} history is incomplete')
    if not {'initial', 'selected', 'last'} <= result.checkpoints.keys():
        raise ValueError(f'Completed {method} is missing required checkpoints')
    if target_tokens is not None and [point.get('rl_target_tokens_this_update')
                                     for point in result.history[1:]] != target_tokens:
        raise ValueError('Completed B3 token targets differ from the reused B2 budget')
    for label, checkpoint in result.checkpoints.items():
        path = source / label
        manifest = json.loads((path / 'local_checkpoint.json').read_text())
        if manifest != checkpoint['manifest'] or artifact_hashes(path) != manifest['artifacts_sha256']:
            raise ValueError(f'Completed {method}/{label} checkpoint artifact mismatch')
        if manifest['parameters'] != result.parameters:
            raise ValueError(f'Completed {method}/{label} trainable parameters differ')
    original_source = result.source_checkpoint
    if method in {'B2', 'B3'}:
        for name in ('model.py', 'training.py'):
            path = Path('llm_local_001') / name
            if file_hash(path) != file_hash(previous / 'source_snapshot' / path):
                raise ValueError(f'Cannot reuse {method}: likelihood/training implementation changed')
        old_b1 = previous / 'checkpoints' / str(seed) / 'B1/selected'
        if not original_source or Path(original_source).resolve() != old_b1.resolve():
            raise ValueError(f'Completed {method} did not originate from its recorded B1')
        old_manifest = json.loads((old_b1 / 'local_checkpoint.json').read_text())
        new_manifest = json.loads((Path(b1_path) / 'local_checkpoint.json').read_text())
        if artifact_hashes(Path(b1_path)) != new_manifest['artifacts_sha256']:
            raise ValueError('Current B1 checkpoint artifacts changed')
        for key in ('artifacts_sha256', 'base_artifacts_sha256', 'config', 'parameters'):
            if old_manifest[key] != new_manifest[key]:
                raise ValueError(f'Completed {method} B1 source differs from current B1')
        if result.checkpoints['initial']['manifest']['artifacts_sha256'] != old_manifest['artifacts_sha256']:
            raise ValueError(f'Completed {method} initial weights differ from B1')
        result.source_checkpoint = str(Path(b1_path).resolve())
    destination = output / 'checkpoints' / str(seed) / method
    shutil.copytree(source, destination)
    relocated = {}
    for label, checkpoint in result.checkpoints.items():
        relocated[checkpoint['path']] = str((destination / label).resolve())
        checkpoint['path'] = str((destination / label).resolve())
    write_json(destination / 'training_result.json', result.as_dict())
    validation_source = previous / 'validation' / f'{method}_seed{seed}'
    if validation_source.exists():
        shutil.copytree(validation_source, output / 'validation' / validation_source.name)
    rollout_source = previous / 'training' / f'{method}_seed{seed}_rollouts.jsonl'
    if rollout_source.exists():
        (output / 'training').mkdir(parents=True, exist_ok=True)
        shutil.copy2(rollout_source, output / 'training' / rollout_source.name)
    manifest_path = output / 'stage_reuse_manifest.json'
    record = json.loads(manifest_path.read_text())
    record.setdefault('reused_stages', {})[f'{method}_seed{seed}'] = {
        'source_result': str(result_file.resolve()), 'source_result_sha256': file_hash(result_file),
        'destination': str(destination.resolve()), 'original_source_checkpoint': original_source,
        'relocated_source_checkpoint': result.source_checkpoint, 'relocated_checkpoint_paths': relocated,
        'checkpoint_files_and_origin_metadata_unchanged': True,
        'configuration_and_full_update_history_verified': True}
    write_json(manifest_path, record)
    print(f'{method} semilla {seed}: etapa completa reutilizada; paso seleccionado {result.best_step}.', flush=True)
    return result


def execute(output):
    started = monotonic()
    metadata, config, gate, splits = verify_gate(output)
    import torch
    torch.set_num_threads(config.get('cpu_threads', 4))
    reuse = reusable_stages(config, splits, output, metadata)
    model_path = gate['selected_model']['path']
    options = dict(device=config['device'], lora=not config['full_finetune'],
                   full_finetune=config['full_finetune'], max_new_tokens=config['max_new_tokens'],
                   gradient_checkpointing=config.get('gradient_checkpointing', False))
    base = LocalPolicy(model_path, seed=config['seed'], **options)
    base_hashes = base.base_fingerprint()
    observations, observation_rows, obs_csv = {}, {}, []
    print('B0: comparación O1/O2 con los mismos usuarios VALIDATION.', flush=True)
    for observation in ('O1', 'O2'):
        if reuse:
            rows = [json.loads(line) for line in (output / 'validation/observability' / f'{observation}.jsonl').read_text().splitlines()]
            if len(rows) != len(splits['validation']):
                raise ValueError('Cannot reuse a partially completed observability condition')
            print(f'B0 {observation}: reutilizados {len(rows)} usuarios completos; sin cambios de prompt/pesos.', flush=True)
        else:
            rows = evaluate(base, splits['validation'], observation, method=f'B0_{observation}', seed=config['seed'],
                            log_path=output / 'validation/observability' / f'{observation}.jsonl')
        observations[observation] = summarize(rows, config['bootstrap'])
        observation_rows[observation] = rows
        obs_csv.append(csv_summary(observations[observation], 'B0', observation, config['seed'], 'validation'))
        write_jsonl(output / 'validation/observability' / f'{observation}.jsonl', rows)
    selected = max(('O2', 'O1'), key=lambda obs: observations[obs]['metrics']['return']['mean'])
    write_json(output / 'diagnostics_before_training.json', {'provenance': metadata,
               'comparison_O2_minus_O1': paired(observation_rows['O2'], observation_rows['O1'], config['bootstrap']),
               'errors': {obs: error_rows(rows) for obs, rows in observation_rows.items()},
               'O3_decision': 'Not automatically added: O2 contains original text, dates, validity, context and actual history links; '
                              'format/action errors do not establish a missing observable relation. '
                              'Residual causal attribution requires independent semantic review.',
               'test_used': False})
    write_json(output / 'observation_selection.json', {'provenance': metadata,
               'selected': selected, 'split': 'validation', 'same_base_weights': True,
               'O3': 'not enabled: requires separately identified residual relational need',
               'scores': observations, 'base_fingerprint': base_hashes})
    write_csv(output / 'observability_results.csv', obs_csv, OBS_FIELDS, metadata)
    release(base)
    del base
    demos = demonstrations(splits['train'], selected)
    write_jsonl(output / 'train_demonstrations.jsonl', ({**row, 'provenance': metadata} for row in demos))
    from rl_small_pilot.reward_audit import audit_rewards
    reward = audit_rewards(splits['validation'][:10], selected)
    write_json(output / 'reward_audit.json', {'provenance': metadata, **reward})
    if not reward['passed']:
        raise RuntimeError('Existing reward audit failed; no model training performed')
    checkpoints, training, curves = {}, {}, []
    for seed in config['seeds']:
        validation_index = 0
        active_method = 'B1'
        common = TrainingConfig(updates=config['sft_updates'], learning_rate=config['learning_rate'],
                                batch_size=config['batch_size'], validation_every=config['validation_every'],
                                gamma=1.0, baseline_decay=config['baseline_decay'],
                                kl_coefficient=config['kl_coefficient'], seed=seed)
        def validation(policy):
            nonlocal validation_index
            validation_index += 1
            rows = evaluate(policy, splits['validation'], selected, method=f'validation_{active_method}', seed=seed,
                            log_path=output / 'validation' / f'{active_method}_seed{seed}' / f'eval_{validation_index}.jsonl')
            return sum(row['return'] for row in rows) / len(rows)
        print(f'B1 SFT: semilla {seed}', flush=True)
        b1 = clone_completed_training(reuse, output, seed, 'B1', common)
        if b1 is None:
            policy = LocalPolicy(model_path, seed=seed, **options)
            if policy.base_fingerprint() != base_hashes:
                raise RuntimeError('Base model/tokenizer changed between branches')
            b1 = train_sft(policy, demos, output / 'checkpoints' / str(seed) / 'B1', common, validation)
            release(policy)
            del policy
        b1_path = b1.checkpoints['selected']['path']
        checkpoints[f'B1_seed{seed}'] = b1.checkpoints
        training[f'B1_seed{seed}'] = b1.as_dict()
        rng = random.Random(seed)
        order = list(splits['train'])
        rng.shuffle(order)
        counter = 0
        rollout_log = []
        def rollout(current_policy):
            nonlocal counter
            scenario = order[counter % len(order)]
            counter += 1
            env = LLMEnvironment(scenario, selected, 'ORACLE')
            generations, rewards = [], []
            try:
                obs = env.reset()
                for _ in range(4):
                    generation = current_policy.generate(format_prompt(obs), sample=True)
                    obs, reward, done, truncated, _ = env.step(generation.text)
                    generations.append(generation)
                    rewards.append(reward)
                    if done or truncated:
                        break
                row = env.diagnostic_metrics()
                rollout_log.append(row)
                write_jsonl(output / 'training' / f'B2_seed{seed}_rollouts.jsonl', rollout_log)
                return EpisodeTrajectory(generations, rewards,
                                         {'user_id': scenario['user_id'], 'partition': 'train'})
            finally:
                env.close()
        print(f'B2 REINFORCE: semilla {seed}; inicio exacto B1.', flush=True)
        active_method = 'B2'
        continuation = replace(common, updates=config['continuation_updates'])
        b2 = clone_completed_training(reuse, output, seed, 'B2', continuation, b1_path=b1_path)
        if b2 is None:
            policy = LocalPolicy.from_checkpoint(b1_path, trainable=True, device=config['device'])
            reference = LocalPolicy.from_checkpoint(b1_path, device=config.get('reference_device', 'cpu'))
            b2 = train_rl(policy, rollout, output / 'checkpoints' / str(seed) / 'B2', continuation, validation,
                          reference_policy=reference)
            release(reference)
            del reference
            release(policy)
            del policy
        training[f'B2_seed{seed}'] = b2.as_dict()
        checkpoints[f'B2_seed{seed}'] = b2.checkpoints
        print(f'B3 SFT adicional: semilla {seed}; inicio exacto B1.', flush=True)
        active_method = 'B3'
        cumulative = [point['tokens_processed'] for point in b2.history]
        target_tokens = [b-a for a, b in zip(cumulative, cumulative[1:])]
        b3 = clone_completed_training(reuse, output, seed, 'B3', continuation, b1_path=b1_path,
                                      target_tokens=target_tokens)
        if b3 is None:
            policy = LocalPolicy.from_checkpoint(b1_path, trainable=True, device=config['device'])
            b3 = train_sft(policy, demos, output / 'checkpoints' / str(seed) / 'B3', continuation,
                           validation, target_tokens_per_update=target_tokens)
            release(policy)
            del policy
        training[f'B3_seed{seed}'] = b3.as_dict()
        checkpoints[f'B3_seed{seed}'] = b3.checkpoints
        if b2.source_checkpoint != b1_path or b3.source_checkpoint != b1_path:
            raise RuntimeError('B2 and B3 did not start from the same B1')
        for method, result in (('B1', b1), ('B2', b2), ('B3', b3)):
            for point in result.history:
                curves.append({'status': 'EXECUTED', 'method': method, 'seed': seed,
                               'step': point['step'], 'loss': point.get('supervised_loss', point.get('reinforce_loss_sum')),
                               'validation_return': point.get('validation_score'),
                               'tokens': point['tokens_processed'],
                               'gradient_norm': point.get('gradient_norm_before_clip'),
                               'seconds': point.get('elapsed_seconds')})
        write_json(output / 'training_metrics.json', {'provenance': metadata, 'runs': training,
                   'budget_control': 'same updates; B3 matches B2 training forward tokens with <=one demo overshoot/update',
                   'remaining_difference': 'RL has additional generation/reference passes; no claim of equal wall time'})
        write_csv(output / 'training_curves.csv', curves, CURVE_FIELDS, metadata)
    write_json(output / 'checkpoint_manifest.json', {'provenance': metadata, 'checkpoints': checkpoints,
               'base_fingerprint': base_hashes, 'adapters_only': not config['full_finetune']})
    write_json(output / 'final_freeze.json', {'provenance': metadata, 'created_at': now(),
               'before_test_evaluation': True, 'observation': selected, 'test_hash': gate['test_hash'],
               'checkpoints': checkpoints, 'configuration': config,
               'reward': json.loads((output / 'reward_freeze.json').read_text())})
    all_rows, comparison_rows, errors, summaries = {}, [], [], {}
    base = LocalPolicy(model_path, seed=config['seed'], **options)
    all_rows['B0'] = evaluate(base, splits['test'], selected, method='B0', seed=config['seed'],
                              directory=output / 'sqlite/test/B0', log_path=output / 'test/B0.jsonl')
    release(base)
    del base
    for method in ('B1', 'B2', 'B3'):
        all_rows[method] = []
        for seed in config['seeds']:
            path = checkpoints[f'{method}_seed{seed}']['selected']['path']
            policy = LocalPolicy.from_checkpoint(path, device=config['device'])
            rows = evaluate(policy, splits['test'], selected, method=method, seed=seed,
                            directory=output / 'sqlite/test' / f'{method}_seed{seed}',
                            log_path=output / 'test' / f'{method}_seed{seed}.jsonl')
            all_rows[method].extend(rows)
            comparison_rows.append(csv_summary(summarize(rows, config['bootstrap']), method, selected, seed))
            release(policy)
            del policy
    all_rows['B4'] = [run_historical_episode(scenario, selected, 'ORACLE',
                                           output / 'sqlite/test/B4' / f'{i}.sqlite')
                       for i, scenario in enumerate(splits['test'])]
    for method, rows in all_rows.items():
        summaries[method] = summarize(rows, config['bootstrap'])
        comparison_rows.append(csv_summary(summaries[method], method, selected))
        write_jsonl(output / 'test' / f'{method}.jsonl', rows)
        errors.extend({**r, 'method': method, 'status': 'EXECUTED'} for r in error_rows(rows))
    comparisons = {}
    for a, b in (('B1','B0'), ('B2','B1'), ('B2','B3'), ('B2','B4')):
        key = f'{a}-{b}'
        comparisons[key] = paired(all_rows[a], all_rows[b], config['bootstrap'])
        comparison_rows.append({'status': 'EXECUTED', 'split': 'test', 'comparison': key,
                                'delta_return': comparisons[key]['return']['mean'],
                                'ci_delta_return': comparisons[key]['return']['ci95'],
                                'delta_error_rate': comparisons[key]['error_rate']['mean']})
    seed_comparisons = {}
    for seed in config['seeds']:
        seed_comparisons[str(seed)] = {}
        for a, b in (('B2', 'B1'), ('B2', 'B3')):
            score = paired([r for r in all_rows[a] if r['seed'] == seed],
                           [r for r in all_rows[b] if r['seed'] == seed], config['bootstrap'])
            seed_comparisons[str(seed)][f'{a}-{b}'] = score
            comparison_rows.append({'status': 'EXECUTED', 'split': 'test', 'seed': seed,
                                    'comparison': f'{a}-{b}', 'delta_return': score['return']['mean'],
                                    'ci_delta_return': score['return']['ci95'],
                                    'delta_error_rate': score['error_rate']['mean']})
    write_csv(output / 'baseline_comparison.csv', comparison_rows, BASE_FIELDS, metadata)
    write_csv(output / 'error_analysis.csv', errors, ERROR_FIELDS, metadata)
    write_json(output / 'final_metrics.json', {'provenance': metadata, 'summaries': summaries,
               'comparisons': comparisons, 'comparisons_by_seed': seed_comparisons, 'by_family': {
                   method: {family: summarize([r for r in rows if r['family'] == family], config['bootstrap'])
                            for family in sorted({r['family'] for r in rows})}
                   for method, rows in all_rows.items()}})
    from .report import completed_report
    completed_report(output, metadata, config, observations, selected, summaries, comparisons, training,
                     seed_comparisons=seed_comparisons)
    write_json(output / 'run_status.json', {'provenance': metadata, 'status': 'COMPLETED_TECHNICAL_PILOT',
               'elapsed_seconds': monotonic()-started, 'test_evaluated': True,
               'external_inference_calls': 0, 'human_review': 'pending'})


if __name__ == '__main__':
    execute(Path(sys.argv[1]).resolve())
