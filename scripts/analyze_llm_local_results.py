#!/usr/bin/env python3
"""Read a COMPLETED local pilot; add interpretation without training or inference.

Only Python's standard library is used. The completion/freeze gate is checked
before opening TEST traces. Primary metrics and selected checkpoints are copied
from the completed run, never changed. Additional behavior diagnostics address
the reward-asymmetry concern declared before TEST.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from statistics import fmean
import sys


METHODS = ('B0', 'B1', 'B2', 'B3', 'B4')
PAIRS = (('B1', 'B0'), ('B2', 'B1'), ('B2', 'B3'), ('B2', 'B4'))


def read_json(path):
    def reject(value):
        raise ValueError(f'Nonfinite JSON in {path}: {value}')
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle, parse_constant=reject)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f'Expected nonempty JSONL objects: {path}')
    for row in rows:
        for field in ('user_id', 'family', 'return', 'error_rate', 'future_accuracy', 'steps'):
            if field not in row:
                raise ValueError(f'Missing {field} in {path}')
    return rows


def mean_metric(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['user_id']].append(float(row[field]))
    return fmean(fmean(values) for values in grouped.values())


def parsed(step):
    return (step.get('generated_response') or {}).get('parsed')


def describe(rows):
    steps = [step for row in rows for step in row['steps']]
    terminal = [step for step in steps if step.get('terminal')]
    actions = Counter((parsed(step) or {}).get('conversation_action', 'INVALID') for step in steps)
    invalid = sum(bool(step.get('format_error') or (step.get('generated_response') or {}).get('format_error'))
                  for step in steps)
    # Some archived trace producers retain format counts only on the episode.
    invalid = max(invalid, sum(int(row.get('format_errors', 0)) for row in rows))
    uncertainty = actions['PRESERVE_CONFLICT'] + actions['CONDITIONAL_ANSWER']
    return {
        'users': len({row['user_id'] for row in rows}), 'episodes': len(rows),
        'generated_actions': len(steps), 'terminal_decisions': len(terminal),
        'terminal_errors': sum(step.get('correct') is False for step in terminal),
        'error_rate': mean_metric(rows, 'error_rate'),
        'mean_return': mean_metric(rows, 'return'),
        's3_accuracy': mean_metric(rows, 'future_accuracy'),
        'format_errors': invalid, 'format_error_rate_per_generated_action': invalid / max(1, len(steps)),
        'unsupported_answers': sum(int(row.get('unsupported_answers', 0)) for row in rows),
        'incorrect_updates': sum(int(row.get('incorrect_updates', 0)) for row in rows),
        'extraction_errors': sum(int(row.get('extraction_errors', 0)) for row in rows),
        'temporal_decision_errors': sum(int(row.get('temporal_decision_errors', 0)) for row in rows),
        'conflicts_not_detected': sum(int(row.get('conflicts_not_detected', 0)) for row in rows),
        'questions': sum(int(row.get('questions', 0)) for row in rows),
        'necessary_asks': sum(int(row.get('necessary_asks', 0)) for row in rows),
        'unnecessary_asks': sum(int(row.get('unnecessary_asks', 0)) for row in rows),
        'generation_truncations': sum(int(row.get('generation_truncations', 0)) for row in rows),
        'elapsed_seconds': sum(float(row.get('elapsed_seconds', 0)) for row in rows),
        'prompt_tokens': sum(int(row.get('prompt_tokens', 0)) for row in rows),
        'completion_tokens': sum(int(row.get('completion_tokens', 0)) for row in rows),
        'generated_conversation_actions': dict(actions),
        'uncertainty_action_frequency': uncertainty / max(1, len(steps)),
        'uncertainty_frequency_among_valid_actions': uncertainty / max(1, len(steps) - invalid),
        'free_text_semantics_independently_evaluated': False,
    }


def percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def paired(left, right, repetitions=1000):
    def index(rows):
        result = defaultdict(list)
        for row in rows:
            result[row['user_id']].append(row)
        return result
    a, b = index(left), index(right)
    if not a or a.keys() != b.keys():
        raise ValueError('Paired comparison requires identical complete user sets')
    result = {}
    for field in ('return', 'error_rate', 'future_accuracy'):
        differences = [fmean(float(row[field]) for row in a[user]) -
                       fmean(float(row[field]) for row in b[user]) for user in sorted(a)]
        rng = random.Random(819)
        estimates = [fmean(rng.choices(differences, k=len(differences))) for _ in range(repetitions)]
        result[field] = {'mean': fmean(differences),
                         'ci95': [percentile(estimates, .025), percentile(estimates, .975)],
                         'users': len(differences)}
    return result


def groups_by_seed(rows):
    result = defaultdict(list)
    for row in rows:
        result[str(row.get('seed', 'conventional'))].append(row)
    return dict(result)


def checkpoint_details(runs):
    output = {}
    for name, run in runs.items():
        if not name.startswith('B2_'):
            continue
        output[name] = {
            'best_step': run['best_step'], 'best_step_zero': run['best_step'] == 0,
            'selected_validation_score': run.get('selected_score'),
            'source_checkpoint': run.get('source_checkpoint'),
            'optimizer_updates': run.get('config', {}).get('updates'),
            'tokens_processed': run.get('tokens_processed'),
            'changed_parameters': run.get('changed_parameters'),
            'parameter_delta_l2': run.get('parameter_delta_l2'),
            'elapsed_seconds': run.get('elapsed_seconds'),
            'parameters': run.get('parameters'),
            'initial_selected_last': run.get('checkpoints', {}),
            'initial_validation_score': run.get('history', [{}])[0].get('validation_score'),
            'last_validation_score': run.get('history', [{}])[-1].get('validation_score'),
        }
    return output


def behavior_interpretation(summaries, comparisons):
    flags = {}
    for left, right in (('B2', 'B1'), ('B2', 'B3')):
        a, b = summaries[left], summaries[right]
        delta = comparisons[f'{left}-{right}']
        rises = delta['return']['mean'] > 0
        invalid_rises = a['format_error_rate_per_generated_action'] > b['format_error_rate_per_generated_action']
        no_quality_gain = delta['error_rate']['mean'] >= 0 and delta['future_accuracy']['mean'] <= 0
        flags[f'{left}-{right}'] = {
            'positive_return_difference': rises,
            'higher_invalid_output_rate': invalid_rises,
            'no_terminal_error_or_s3_improvement': no_quality_gain,
            'reward_shortcut_risk': rises and invalid_rises and no_quality_gain,
            'return_only_gain': rises and no_quality_gain,
            'fewer_unnecessary_questions': a['unnecessary_asks'] < b['unnecessary_asks'],
            'format_improves_without_s3_gain':
                a['format_error_rate_per_generated_action'] < b['format_error_rate_per_generated_action']
                and delta['future_accuracy']['mean'] <= 0,
            'interpretation_status': 'descriptive warning; not a changed primary metric or causal proof',
        }
    flags['B2_uncertainty_actions'] = {
        'frequency': summaries['B2']['uncertainty_action_frequency'],
        'frequency_among_valid_actions': summaries['B2']['uncertainty_frequency_among_valid_actions'],
        'questions': summaries['B2']['questions'],
        'interpretation': 'Compare the exact action counts by family. High preservation frequency alone '
                          'does not prove a shortcut; preserving clear-preference cases remains penalized.',
    }
    return flags


def examples(all_rows, limit=4):
    candidates = []
    for method in ('B0', 'B1', 'B2', 'B3'):
        for row in all_rows[method]:
            for position, step in enumerate(row['steps']):
                generated = step.get('generated_response') or {}
                if step.get('correct') is not False:
                    continue
                causes = []
                if generated.get('format_error') or step.get('format_error'):
                    causes.append('observed_response_format_failure')
                if step.get('unsupported_answer'):
                    causes.append('observed_unsupported_structured_answer')
                if step.get('incorrect_updates'):
                    causes.append('observed_unjustified_memory_update')
                if not causes:
                    causes.append('incorrect_decision_cause_requires_human_review')
                memory = (step.get('observation') or {}).get('memory', step.get('retrieved_memories', []))
                candidates.append({
                    'method': method, 'seed': row.get('seed'), 'user_id': row['user_id'],
                    'family': row['family'], 'session_index': step.get('session_index'),
                    'source_trace': f'test/{method}.jsonl', 'step_index_within_episode': position,
                    'query': step.get('query'), 'query_at': step.get('query_at'),
                    'visible_evidence': [{key: fact.get(key) for key in
                                          ('memory_id', 'evidence_id', 'text', 'observed_at', 'valid_from', 'valid_to', 'source')}
                                         for fact in memory],
                    'actual_generated_text': generated.get('raw'),
                    'actual_parsed_action': generated.get('parsed'),
                    'actual_executed_action': step.get('executed_action'),
                    'format_error': generated.get('format_error') or step.get('format_error'),
                    'terminal_correct': step.get('correct'), 'reward': step.get('reward'),
                    'observed_failure_categories': causes,
                    'attribution': 'Only recorded behavior is established; semantic/causal diagnosis requires independent review.',
                })
    # Fixed deterministic coverage: one actual failure per method, then differing
    # categories. Never fabricate an example when a method has no failures.
    chosen = []
    for method in ('B0', 'B1', 'B2', 'B3'):
        candidate = next((row for row in candidates if row['method'] == method), None)
        if candidate is not None and len(chosen) < limit:
            chosen.append(candidate)
    return chosen


def classify(comparisons, seeds, runs, flags):
    a = comparisons['B2-B1']['return']
    b = comparisons['B2-B3']['return']
    shortcuts = any(flags[key]['reward_shortcut_risk']
                    for key in ('B2-B1', 'B2-B3'))
    all_zero = bool(runs) and all(run['best_step_zero'] for run in runs.values())
    consistent = bool(seeds) and all(
        seed['B2-B1']['return']['mean'] > 0 and seed['B2-B3']['return']['mean'] > 0
        for seed in seeds.values())
    if shortcuts:
        return 'F', 'Ganancia de retorno compatible con evitar penalizaciones; beneficio decisional inconcluso.'
    if len(seeds) >= 3 and a['ci95'][0] > 0 and b['ci95'][0] > 0 and consistent and not any(
            run['best_step_zero'] for run in runs.values()):
        return 'A', 'Señal exploratoria favorable y consistente por semilla; falta réplica independiente.'
    if a['ci95'][0] > 0 and b['ci95'][0] <= 0:
        return 'B', 'Mejora frente a B1 sin ventaja específica demostrada frente a continuar SFT.'
    if a['mean'] <= 0 or all_zero:
        return 'C', 'No se demostró beneficio incremental de RL en el modelo, datos y presupuesto ensayados.'
    return 'F', 'Incertidumbre estadística, replicación insuficiente o inconsistencia entre semillas.'


def analyze(root, reward_audit=None, destination=None, observability_supplement=None):
    root = Path(root).resolve()
    destination = Path(destination).resolve() if destination else root
    # Do not open TEST or final metrics before confirming training/final selection
    # have finished. A live or failed experiment is deliberately rejected.
    status = read_json(root / 'run_status.json')
    if status.get('status') != 'COMPLETED_TECHNICAL_PILOT' or not status.get('test_evaluated'):
        raise ValueError('Only a completed frozen pilot with final TEST evaluation may be analyzed')
    freeze = read_json(root / 'final_freeze.json')
    if freeze.get('before_test_evaluation') is not True:
        raise ValueError('Missing pre-TEST final checkpoint/configuration freeze')
    supplement = None
    supplement_protocol_path = None
    if observability_supplement:
        observability_supplement = Path(observability_supplement).resolve()
        supplement = read_json(observability_supplement)
        if (supplement.get('status') != 'COMPLETED_PRESPECIFIED_SUPPLEMENT' or
                supplement.get('all_locked_configurations_evaluated') is not True):
            raise ValueError('Observability supplement must be complete for all locked configurations')
        info = supplement['provenance']
        if Path(info['primary_pilot']).resolve() != root:
            raise ValueError('Observability supplement belongs to another primary pilot')
        if info['primary_final_freeze_sha256'] != sha256(root / 'final_freeze.json'):
            raise ValueError('Supplement was not evaluated against this exact primary final freeze')
        supplement_protocol_path = Path(info['protocol_path']).resolve()
        if sha256(supplement_protocol_path) != info['protocol_sha256']:
            raise ValueError('Supplement prospective protocol file changed')
        registered = read_json(supplement_protocol_path)
        canonical = json.dumps(registered['configuration'], sort_keys=True, ensure_ascii=False,
                               allow_nan=False).encode()
        if (hashlib.sha256(canonical).hexdigest() != info['protocol_config_hash'] or
                registered.get('test_read') is not False or registered.get('test_evaluated') is not False or
                registered.get('prespecified_before_main_test') is not True):
            raise ValueError('Supplement prospective registration is inconsistent')
        if datetime.fromisoformat(registered['created_at']) > datetime.fromisoformat(freeze['created_at']):
            raise ValueError('Supplement protocol did not predate the primary TEST freeze')
        expected = {'B0_seed' + str(freeze['configuration']['seed'])}
        expected.update(f'B1_seed{seed}' for seed in registered['configuration']['training_seeds'])
        expected.add('B1_ALL_SEEDS')
        if not expected <= supplement['paired_O2_minus_O1'].keys():
            raise ValueError('A prospectively required B0/B1 configuration is missing')
    # When requested, both main and supplementary completion gates have passed
    # before opening any experimental TEST trace below.
    final = read_json(root / 'final_metrics.json')
    training = read_json(root / 'training_metrics.json')
    rows = {method: read_rows(root / 'test' / f'{method}.jsonl') for method in METHODS}
    observations = {obs: read_rows(root / 'validation' / 'observability' / f'{obs}.jsonl')
                    for obs in ('O1', 'O2')}
    summaries = {method: describe(values) for method, values in rows.items()}
    obs_summary = {obs: describe(values) for obs, values in observations.items()}
    comparisons = final['comparisons']
    # Recompute paired means as an integrity check, retaining the original
    # preregistered primary confidence intervals and metrics for the conclusion.
    for left, right in PAIRS:
        recomputed = paired(rows[left], rows[right])
        for field in ('return', 'error_rate', 'future_accuracy'):
            if abs(recomputed[field]['mean'] - comparisons[f'{left}-{right}'][field]['mean']) > 1e-9:
                raise ValueError(f'Final metrics disagree with raw traces: {left}-{right}/{field}')
    by_seed = {method: groups_by_seed(values) for method, values in rows.items()}
    per_seed = {}
    for seed in sorted(set(by_seed['B1']) & set(by_seed['B2']) & set(by_seed['B3'])):
        per_seed[seed] = {
            f'{left}-{right}': paired(by_seed[left][seed], by_seed[right].get(seed, rows[right]))
            for left, right in PAIRS
        }
    runs = checkpoint_details(training['runs'])
    flags = behavior_interpretation(summaries, comparisons)
    case, interpretation = classify(comparisons, per_seed, runs, flags)
    obs_difference = paired(observations['O2'], observations['O1'])
    obs_floor = all(value['format_errors'] == value['generated_actions'] for value in obs_summary.values())
    files = [root / name for name in ('run_status.json', 'final_freeze.json', 'final_metrics.json', 'training_metrics.json')]
    files += [root / 'test' / f'{method}.jsonl' for method in METHODS]
    files += [root / 'validation' / 'observability' / f'{obs}.jsonl' for obs in ('O1', 'O2')]
    if supplement is not None:
        files += [observability_supplement, supplement_protocol_path]
    audit = None
    if reward_audit:
        reward_audit = Path(reward_audit).resolve()
        audit = read_json(reward_audit)
        if audit.get('test_read') is not False or audit.get('test_evaluated') is not False:
            raise ValueError('Reward interpretation audit must explicitly declare no TEST access')
        freeze_time = freeze.get('created_at') or freeze.get('provenance', {}).get('created_at')
        if freeze_time and datetime.fromisoformat(audit['created_at']) > datetime.fromisoformat(freeze_time):
            raise ValueError('Reward audit was not recorded before final TEST freeze')
        files.append(reward_audit)
    provenance = {'created_at': datetime.now(timezone.utc).isoformat(),
                  'source': str(Path(__file__).resolve()), 'source_sha256': sha256(__file__),
                  'completed_pilot': str(root), 'inputs_sha256': {str(path): sha256(path) for path in files},
                  'read_only_postprocessing': True, 'model_calls': 0, 'training_updates': 0,
                  'hyperparameters_or_checkpoints_changed': False,
                  'confidence_unit': 'complete user; training seeds averaged within user for aggregate comparisons'}
    detail = {'provenance': provenance, 'case': case, 'interpretation': interpretation,
              'frozen_configuration': freeze.get('configuration', {}),
              'metrics': summaries, 'primary_comparisons_unchanged': comparisons,
              'paired_comparisons_by_training_seed': per_seed,
              'descriptive_metrics_by_seed': {method: {seed: describe(values) for seed, values in groups.items()}
                                               for method, groups in by_seed.items()},
              'observability': {'validation': obs_summary, 'O2_minus_O1': obs_difference,
                                'all_generations_invalid_in_both_conditions': obs_floor,
                                'interpretation': 'Format floor: information benefit is unidentifiable.' if obs_floor
                                else 'Observed validation difference; individual historical fields were not independently ablated.'},
              'rl_checkpoints': runs, 'behavior_interpretations': flags,
              'supplementary_observability': supplement,
              'training_run_resources': {name: {**{key: run.get(key) for key in
                                         ('elapsed_seconds', 'tokens_processed', 'completion_tokens',
                                          'parameters', 'best_step', 'selected_score', 'changed_parameters',
                                          'parameter_delta_l2', 'config')},
                                          'cuda_peak_allocated_bytes': max((point.get('cuda_peak_allocated_bytes', 0)
                                                                          for point in run.get('history', [])), default=0),
                                          'cuda_peak_reserved_bytes': max((point.get('cuda_peak_reserved_bytes', 0)
                                                                         for point in run.get('history', [])), default=0)}
                                         for name, run in training['runs'].items()},
              'pretest_reward_audit': {'path': str(reward_audit), 'created_at': audit['created_at']} if audit else None,
              'actual_trace_examples': examples(rows),
              'limitations': ['Synthetic shared mechanisms; no independent human-history sample.',
                              'Free response prose is not independently semantically scored.',
                              'User bootstrap intervals do not estimate uncertainty over all possible training seeds.',
                              'A weak supervised model limits conclusions about temporal reasoning versus schema learning.',
                              'An improvement in weights or ability to train locally is not evidence that RL is useful.']}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'behavior_diagnostics.json').write_text(json.dumps(detail, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    family_rows = []
    for method, values in rows.items():
        groupings = {'ALL_SEEDS': values, **by_seed[method]}
        for seed, group in groupings.items():
            for family in sorted({row['family'] for row in group}):
                description = describe([row for row in group if row['family'] == family])
                family_rows.append({'created_at': provenance['created_at'], 'source': provenance['source'],
                                    'source_sha256': provenance['source_sha256'],
                                    'pilot': str(root), 'raw_trace_sha256': provenance['inputs_sha256'][str(root / 'test' / f'{method}.jsonl')],
                                    'method': method, 'seed': seed, 'family': family,
                                    **{key: json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value
                                       for key, value in description.items()}})
    with (destination / 'per_family_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(family_rows[0]))
        writer.writeheader()
        writer.writerows(family_rows)
    write_report(destination, detail, training)
    return detail


def write_report(destination, detail, training):
    p = detail['provenance']
    configuration = detail['frozen_configuration']
    text = ['# Decisión sobre la tesis a partir del piloto local', '',
            f"Fecha UTC: {p['created_at']}. Corrida: `{p['completed_pilot']}`.",
            f"Procedencia: `{p['source']}`; SHA-256 `{p['source_sha256']}`.", '',
            f"**Caso {detail['case']}: {detail['interpretation']}**", '',
            'Este análisis usa los checkpoints y métricas congelados de la corrida terminada. '
            'No genera respuestas, no entrena y no modifica la selección ni la recompensa.', '',
            f"Modelo local: `{configuration.get('model_path', 'consultar model_manifest.json')}`. "
            f"Dispositivo: `{configuration.get('device', 'consultar hardware_audit.json')}`. "
            f"Semillas: `{configuration.get('seeds', [])}`.", '',
            '| Método | Retorno | Error | Exactitud S3 | Formato inválido / acciones | Sin fundamento | Actualizaciones incorrectas |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for method, value in detail['metrics'].items():
        text.append(f"| {method} | {value['mean_return']:.4f} | {value['error_rate']:.1%} | "
                    f"{value['s3_accuracy']:.1%} | {value['format_errors']}/{value['generated_actions']} | "
                    f"{value['unsupported_answers']} | {value['incorrect_updates']} |")
    text += ['', 'Las cuentas agregadas de B1/B2/B3 incluyen todas las semillas; las tasas y retornos '
             'promedian primero dentro de cada usuario. No se eliminan las generaciones inválidas.', '',
             'Comparaciones primarias originales, con IC95% agrupado por usuario:', '']
    for name, value in detail['primary_comparisons_unchanged'].items():
        text.append(f"- {name}: Δ retorno {value['return']['mean']:.4f}, IC95% {value['return']['ci95']}; "
                    f"Δ error {value['error_rate']['mean']:.4f}; Δ exactitud S3 {value['future_accuracy']['mean']:.4f}.")
    text += ['', 'Resultados por semilla de entrenamiento:', '',
             '| Semilla | B2−B1 retorno | B2−B3 retorno | B2−B1 error | B2−B1 S3 |',
             '|---|---:|---:|---:|---:|']
    for seed, comparisons in detail['paired_comparisons_by_training_seed'].items():
        a, b = comparisons['B2-B1'], comparisons['B2-B3']
        text.append(f"| {seed} | {a['return']['mean']:.4f} | {b['return']['mean']:.4f} | "
                    f"{a['error_rate']['mean']:.4f} | {a['future_accuracy']['mean']:.4f} |")
    text += ['', 'Los IC por semilla y todas las métricas por familia están en '
             '`behavior_diagnostics.json` y `per_family_metrics.csv`. Los intervalos de usuarios '
             'no describen toda la variabilidad posible de semillas de entrenamiento.', '']
    for name, run in detail['rl_checkpoints'].items():
        text.append(f"- {name}: mejor paso {run['best_step']}, validación seleccionada "
                    f"{run['selected_validation_score']}; {run['optimizer_updates']} actualizaciones, "
                    f"{run['tokens_processed']} tokens. Parámetros cambiados: {run['changed_parameters']}.")
        for label in ('initial', 'selected', 'last'):
            record = run['initial_selected_last'].get(label, {})
            text.append(f"  - {label}: `{record.get('path', 'no registrado')}`.")
    if any(run['best_step_zero'] for run in detail['rl_checkpoints'].values()):
        text += ['', 'En cada semilla cuyo mejor paso es cero, RL no produjo una mejora validada '
                 'y el checkpoint B2 seleccionado conserva B1. El último checkpoint se conserva '
                 'para diagnosticar el entrenamiento, sin seleccionarlo usando TEST.']
    obs = detail['observability']
    text += ['', '**P1. Información observable.**', '',
             f"O2−O1 en VALIDATION: Δ retorno {obs['O2_minus_O1']['return']['mean']:.4f}, "
             f"IC95% {obs['O2_minus_O1']['return']['ci95']}."]
    if obs['all_generations_invalid_in_both_conditions']:
        text += ['Ambas observaciones produjeron exclusivamente salidas inválidas. La igualdad '
                 'de puntuación refleja un efecto suelo de formato: **no permite concluir que '
                 'la historia temporal no ayude**. P1 permanece inconclusa con este modelo.']
    else:
        text += ['La diferencia corresponde al mismo B0 y usuarios de validación. No identifica '
                 'el efecto independiente de fecha, procedencia o contexto sin ablaciones propias. '
                 'No se convierte automáticamente en un caso D: las comparaciones de observabilidad '
                 'y entrenamiento proceden de particiones distintas.']
    supplement = detail.get('supplementary_observability')
    if supplement:
        text += ['', 'Comparación suplementaria O1/O2 en TEST, fijada prospectivamente antes de '
                 'evaluar TEST y ejecutada para B0 y los tres B1 seleccionados. Cada par mantiene '
                 'idénticos pesos, tokenizer y usuarios; solo cambia la exposición histórica. '
                 'Se reutiliza la condición primaria y se genera la complementaria. Los estados '
                 'posteriores pueden divergir como consecuencia de las propias acciones.', '',
                 '| Modelo bloqueado | Δ retorno O2−O1 (IC95%) | Δ error (IC95%) | Δ exactitud S3 (IC95%) |',
                 '|---|---|---|---|']
        for name, values in supplement['paired_O2_minus_O1'].items():
            text.append(f"| {name} | {values['return']['mean']:.4f} ({values['return']['ci95']}) | "
                        f"{values['error_rate']['mean']:.4f} ({values['error_rate']['ci95']}) | "
                        f"{values['future_accuracy']['mean']:.4f} ({values['future_accuracy']['ci95']}) |")
        text += ['', 'Los IC agrupan por usuario; el agregado B1 promedia sus semillas dentro de '
                 'cada usuario. Formato inválido y todos los valores por condición constan en '
                 '`supplementary_observability.json/csv`; las generaciones completas, evidencia '
                 'expuesta y SQLite se conservan en `supplementary_observability_raw/`. '
                 'No se seleccionó una semilla favorable ni se reajustó el modelo.', '',
                 'Este resultado estima por separado el efecto informativo en los checkpoints '
                 'base y supervisados. **No reinterpreta el B0 previo al entrenamiento, no '
                 'atribuye el efecto a RL y no modifica el caso ni las comparaciones primarias.**']
    text += ['', '**P2. Causas y posibles atajos.**', '',
             'Las respuestas inválidas reciben −1 por decisión; una respuesta sin evidencia recibe '
             '−1,6 y una actualización injustificada añade penalización. Por ello puede aumentar '
             'el retorno sustituyendo una mala respuesta por JSON inválido. Ese cambio conserva '
             'el fallo y no demuestra mejor razonamiento. La recompensa congelada se conserva.']
    for name in ('B2-B1', 'B2-B3'):
        flag = detail['behavior_interpretations'][name]
        text.append(f"- {name}: riesgo de evitar penalizaciones mediante más salidas inválidas "
                    f"= {flag['reward_shortcut_risk']}; ganancia de retorno sin mejora en error/S3 "
                    f"= {flag['return_only_gain']}.")
    uncertainty = detail['behavior_interpretations']['B2_uncertainty_actions']
    text += [f"B2 preserva o responde condicionalmente en {uncertainty['frequency']:.1%} de las acciones "
             f"y formula {uncertainty['questions']} preguntas. Las frecuencias por familia permiten "
             'examinar si esa conducta discrimina contradicciones reales; la frecuencia sola no '
             'demuestra un atajo ni una mejora.', '',
             'Una reducción de preguntas innecesarias puede mejorar legítimamente el retorno '
             'sin cambiar error o S3; por eso una ganancia de retorno aislada solo activa una '
             'advertencia descriptiva. El riesgo específico señalado requiere además que aumenten '
             'las salidas inválidas y no mejoren error ni S3.', '',
             'Ejemplos exactos tomados de las trazas; la atribución semántica sigue pendiente:', '']
    for example in detail['actual_trace_examples']:
        text.append(f"- {example['method']}, semilla {example['seed']}, usuario `{example['user_id']}`, "
                    f"{example['family']}, sesión {example['session_index']}, paso {example['step_index_within_episode']} "
                    f"en `{example['source_trace']}`: {', '.join(example['observed_failure_categories'])}; "
                    f"recompensa {example['reward']}.")
        if example['visible_evidence']:
            fact = example['visible_evidence'][0]
            text.append(f"  Evidencia real `{fact['evidence_id']}`: " + json.dumps(fact['text'], ensure_ascii=False) + '.')
        text.append('  Generación real: ' + json.dumps((example['actual_generated_text'] or '')[:500], ensure_ascii=False) + '.')
    text += ['', '**P3 y P4. Qué se comprobó y qué falta.**', '',
             'El banco reutiliza cinco familias y tres sesiones, con SQLite, generación local real, '
             'SFT, continuación RL y control SFT adicional. B4 conserva las reglas convencionales; '
             'su comparación no representa igualdad de capacidad con un LLM. El ajuste modifica '
             'los parámetros registrados en `training_metrics.json`: cuando el alcance es LoRA, '
             'son adaptadores integrados en el LLM, no todos los pesos base.', '',
             'Antes de escalar se necesitan historias y anotaciones independientes, revisión de '
             'preguntas y respuestas, y un SFT que domine el formato y demuestre competencia mínima '
             'en validación. Los mecanismos abstractos compartidos limitan la generalización. '
             'No hay evidencia aquí sobre un dominio conversacional abierto.', '',
             '**P5. Recomendación para la tesis.**', '',
             detail['interpretation'],
             'La viabilidad técnica de LoRA/REINFORCE no basta para elegir RL como método principal. '
             'Si SFT sigue fallando mayoritariamente o la comparación O1/O2 queda en el suelo de '
             'formato, este piloto distingue mal aprendizaje de estructura de decisiones temporales. '
             'Conviene resolver esa medición y evaluar observabilidad/memoria antes de aumentar '
             'el presupuesto de RL. Un resultado nulo con este modelo pequeño y presupuesto '
             'no significa que RL falle en general ni que la tesis carezca de valor.', '',
             'Los fallos, tiempos, tokens, acciones, semillas, checkpoints y hashes de entrada '
             'se conservan en los artefactos complementarios. La revisión independiente está pendiente.']
    if supplement:
        difference = supplement['paired_O2_minus_O1']['B1_ALL_SEEDS']['return']
        if difference['ci95'][0] > 0:
            statement = 'La comparación suplementaria detecta una señal favorable de historia observable '
            statement += 'en el mismo modelo supervisado; respalda estudiar observabilidad/memoria, sin demostrar una ventaja de RL.'
        elif difference['ci95'][1] < 0:
            statement = 'La historia adicional empeora el retorno del modelo supervisado en este suplemento; '
            statement += 'hace falta revisar el uso de evidencia y los fallos de formato antes de atribuir una causa.'
        else:
            statement = 'La comparación suplementaria del modelo supervisado no identifica una diferencia '
            statement += 'de retorno con precisión suficiente; no demuestra equivalencia entre observaciones.'
        text += ['', statement]
    (destination / 'DECISION_TESIS.md').write_text('\n'.join(text) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot', type=Path)
    parser.add_argument('--reward-audit', type=Path)
    parser.add_argument('--output', type=Path, help='Default: the completed pilot directory')
    parser.add_argument('--observability-supplement', type=Path,
                        help='Optional completed, prospectively registered O1/O2 supplement')
    args = parser.parse_args(argv)
    try:
        result = analyze(args.pilot, args.reward_audit, args.output, args.observability_supplement)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'Analysis refused: {error}\n')
    print(f"Caso {result['case']}: {result['interpretation']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
