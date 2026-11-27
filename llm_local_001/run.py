"""Offline entry point. Audit first, freeze data, and run only verified stages."""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

from .artifacts import digest, file_hash, now, write_json
from .data import freeze_data
from .preflight import run_preflight
from .report import blocked_report
from rl_observability.environment import RewardConfig


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('reports/llm_local_001'))
    p.add_argument('--model-path', help='Existing local causal HF checkpoint; never downloaded')
    p.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    p.add_argument('--full-finetune', action='store_true', help='Explicitly opt into all base weights instead of LoRA')
    p.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True,
                   help='Recompute activations to reduce training VRAM; identical observations')
    p.add_argument('--cpu-threads', type=int, default=4, help='CPU threads, including frozen B1 reference')
    p.add_argument('--reference-device', choices=('cuda', 'cpu'), default='cuda',
                   help='Device of exact frozen B1 KL reference; preflight and smoke must verify memory')
    p.add_argument('--preparation-manifest', type=Path, help='Provenance of explicitly authorized downloads/installations')
    p.add_argument('--reuse-completed-stages', type=Path,
                   help='Reuse complete pre-test B0/B1 stages after a documented technical failure; config/data/prompts must match')
    p.add_argument('--train-users', type=int, default=60)
    p.add_argument('--validation-users', type=int, default=20)
    p.add_argument('--test-users', type=int, default=40)
    p.add_argument('--seed', type=int, default=61001)
    p.add_argument('--seeds', type=int, nargs='+', default=[101, 202, 303])
    p.add_argument('--sft-updates', type=int, default=24)
    p.add_argument('--continuation-updates', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--learning-rate', type=float, default=0.0001)
    p.add_argument('--kl-coefficient', type=float, default=0.02)
    p.add_argument('--baseline-decay', type=float, default=0.9)
    p.add_argument('--validation-every', type=int, default=4)
    p.add_argument('--max-new-tokens', type=int, default=192)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--preflight-only', action='store_true', help='Audit and frozen-data report; no inference or training')
    return p


def execute(args):
    started = monotonic()
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError('Seeds must be nonempty and distinct')
    if min(args.sft_updates, args.continuation_updates, args.batch_size,
           args.max_new_tokens, args.validation_every, args.cpu_threads) < 1 or args.bootstrap < 100:
        raise ValueError('Positive training budgets and >=100 bootstrap samples required')
    if args.learning_rate <= 0 or args.kl_coefficient < 0 or not 0 <= args.baseline_decay < 1:
        raise ValueError('Invalid optimizer/stabilization coefficients')
    if any(n < 20 or n % 10 for n in (args.train_users, args.validation_users, args.test_users)):
        raise ValueError('Each partition needs >=20 users in multiples of ten')
    out = args.output.resolve()
    if out.exists() and any(p.name not in {'hardware_audit.json', 'model_manifest.json'} for p in out.iterdir()):
        raise FileExistsError(f'Use a new output directory; previous experiment is preserved: {out}')
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(spec='SPEC-LLM-LOCAL-001', created_at=now(),
                  primary_metric='paired B2-B1 and B2-B3 longitudinal episode return',
                  secondary_metric='terminal decision error rate incl invalid generations',
                  observation_selection='B0 O1/O2 validation return; ties O2; O3 not enabled without residual evidence',
                  checkpoint_selection='validation return, earliest tie including step zero',
                  test_used_for_selection=False, training_scope='all base weights' if args.full_finetune else 'LoRA adapters only',
                  demonstrations='TRAIN only, unchanged conventional historical rules teacher; synthetic, not human',
                  reference_date='explicit per trajectory; independent of system clock',
                  human_review='pending; technical synthetic pilot',
                  seed_limitation=None if len(args.seeds) >= 3 else 'Fewer than three seeds: exploratory only',
                  comparison_budget='B2/B3 same optimizer updates; match prompt+completion training tokens adaptively; record overshoot and reference/rollout overhead',
                  scientific_scope='lexical/compositional within five shared mechanisms; not unseen-mechanism generalization')
    config['interpretation_rules'] = {
        'case_A_requires': 'positive paired user CI versus B1/B3 and positive mean differences in every training seed',
        'ceiling_reference': 'attainable historical B4 on matched observations; do not use hidden resolvability upper envelope',
        'pilot_null_scope': 'A null small-budget synthetic pilot does not reject the thesis or RL in general',
        'case_D': 'describe observability contribution on validation alongside SFT/RL; no post-test metric selection',
    }
    metadata = {'created_at': config['created_at'], 'source': 'python -m llm_local_001.run',
                'config_hash': digest(config)}
    write_json(out / 'config.json', {'provenance': metadata, 'configuration': config})
    if args.preparation_manifest:
        import json
        write_json(out / 'preparation_manifest.json', {'provenance': metadata,
                   'source_sha256': file_hash(args.preparation_manifest),
                   'preparation': json.loads(args.preparation_manifest.read_text())})
    # Snapshot exact sources/hashes without copying historical model checkpoints.
    source_paths = [p for directory in ('llm_local_001', 'rl_small_pilot', 'rl_longitudinal',
                                       'rl_observability', 'memory', 'simulator')
                    for p in Path(directory).glob('*.py')]
    source_paths += [p for p in (Path('SPEC_LLM_LOCAL_001.md'), Path('SPEC_CODEX.md'), Path('METODOLOGIA.md')) if p.exists()]
    source_paths += list(Path('tests').glob('test_llm_local*.py'))
    source_paths.append(Path('llm_local_001/README.md'))
    source_paths.append(Path('scripts/download_llm_local.py'))
    source_paths.append(Path('scripts/launch_local_experiment.py'))
    source_hashes = {}
    for source in source_paths:
        destination = out / 'source_snapshot' / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        source_hashes[str(source)] = file_hash(source)
    write_json(out / 'source_hashes.json', {'provenance': metadata, 'files': source_hashes})
    copy_manifest = Path('MANIFIESTO_COPIA.json')
    if copy_manifest.is_file():
        import json
        previous = json.loads(copy_manifest.read_text()).get('files', [])
        historical = [record for record in previous if record['path'].startswith(
                      ('rl_longitudinal/', 'rl_small_pilot/', 'rl_observability/', 'reports/'))]
        mismatches = [record['path'] for record in historical
                      if not Path(record['path']).is_file() or file_hash(record['path']) != record['sha256']]
        write_json(out / 'historical_preservation.json', {'provenance': metadata,
                   'reference_manifest': str(copy_manifest), 'reference_sha256': file_hash(copy_manifest),
                   'verified_historical_files': len(historical), 'mismatches': mismatches})
        if mismatches:
            raise RuntimeError('Historical experiment files differ from the copy manifest; inspect historical_preservation.json')
    print('Preflight local: CPU, RAM, CUDA, entornos y modelos; sin descargas.', flush=True)
    audit = run_preflight(out, args.model_path,
                          training_mode='full' if args.full_finetune else 'lora', device=args.device)
    splits, dataset = freeze_data(out, (args.train_users, args.validation_users, args.test_users),
                                  args.seed, metadata)
    write_json(out / 'reward_freeze.json', {'provenance': metadata,
               'configuration': asdict(RewardConfig()),
               'frozen_before_training_and_test': True,
               'source': 'rl_observability.environment.RewardConfig'})
    diagnostic = None
    if not args.preflight_only:
        print('Verificando suite local y reglas históricas en VALIDATION.', flush=True)
        test_python = audit.get('selected_python') or sys.executable
        with (out / 'tests.log').open('w') as log:
            check = subprocess.run([test_python, '-B', '-m', 'unittest', 'discover', '-s', 'tests'],
                                   stdout=log, stderr=subprocess.STDOUT)
        write_json(out / 'test_status.json', {'provenance': metadata,
                   'command': [test_python, '-B', '-m', 'unittest', 'discover', '-s', 'tests'],
                   'returncode': check.returncode, 'log': 'tests.log'})
        if check.returncode:
            raise RuntimeError('Local test suite failed. See tests.log; no training permitted')
        from .environment import run_historical_episode, historical_rule_policy, LLMEnvironment
        from .artifacts import write_jsonl
        from .metrics import summarize
        native, text = [], []
        for scenario in splits['validation']:
            native.append(run_historical_episode(scenario, 'O2', 'ORACLE'))
            env = LLMEnvironment(scenario, 'O2', 'ORACLE')
            try:
                obs = env.reset()
                for _ in range(4):
                    obs, _, done, truncated, _ = env.step(historical_rule_policy(obs))
                    if done or truncated:
                        break
                text.append(env.diagnostic_metrics())
            finally:
                env.close()
        equivalent = all(a['return'] == b['return'] and a['future_accuracy'] == b['future_accuracy']
                         for a, b in zip(native, text))
        diagnostic = {'provenance': metadata, 'status': 'VALIDATION_ONLY_SOFTWARE_DIAGNOSTIC',
                      'not_a_final_llm_comparison': True, 'native': summarize(native, args.bootstrap),
                      'text_only': summarize(text, args.bootstrap), 'equivalent_on_validation': equivalent,
                      'b4_policy': 'native existing historical rules retained; no degradation',
                      'test_evaluated': False}
        write_json(out / 'validation/b4_diagnostic.json', diagnostic)
        write_jsonl(out / 'validation/b4_native.jsonl', native)
        if not equivalent:
            print('Advertencia: proyección textual difiere de B4 nativo; se conserva B4 nativo.', flush=True)
    if audit['status'] != 'ready' or args.preflight_only:
        if args.preflight_only and audit['status'] == 'ready':
            audit = {**audit, 'reasons': ['Preflight-only requested: no inference or training executed.']}
        blocked_report(out, audit, dataset, metadata, diagnostic)
        write_json(out / 'run_status.json', {'provenance': metadata, 'status': 'BLOCKED' if audit['status'] != 'ready' else 'AUDIT_ONLY',
                   'case': 'E' if audit['status'] != 'ready' else None, 'reasons': audit['reasons'],
                   'elapsed_seconds': monotonic()-started, 'model_inference_calls': 0,
                   'llm_optimizer_updates': 0, 'external_inference_calls': 0,
                   'test_evaluated': False, 'completed_artifacts': True})
        print(f"Resultado: {'CASO E' if audit['status'] != 'ready' else 'solo auditoría'}. Informes: {out}", flush=True)
        return 2 if audit['status'] != 'ready' else 0
    if not args.full_finetune:
        selected = next((e for e in audit['hardware']['python_environments']
                         if e.get('requested_executable') == audit['selected_python']), {})
        if not selected.get('packages', {}).get('peft'):
            audit['reasons'].append('Selected environment lacks PEFT; default LoRA training is unavailable. No packages installed.')
            audit['status'] = 'blocked'
            blocked_report(out, audit, dataset, metadata, diagnostic)
            write_json(out / 'run_status.json', {'provenance': metadata, 'status': 'BLOCKED', 'case': 'E',
                       'reasons': audit['reasons'], 'elapsed_seconds': monotonic()-started})
            return 2
    write_json(out / 'prepared_gate.json', {'provenance': metadata, 'status': 'ready',
               'selected_model': audit['selected_model'], 'selected_python': audit['selected_python'],
               'source_hashes': source_hashes, 'test_hash': dataset['test_frozen_hash']})
    # The current interpreter need not have torch/Transformers. Reuse an existing
    # compatible interpreter without changing the project interpreter or packages.
    command = [audit['selected_python'], '-B', '-m', 'llm_local_001.experiment', str(out)]
    code = subprocess.run(command).returncode
    if code:
        write_json(out / 'run_status.json', {'provenance': metadata, 'status': 'FAILED', 'case': 'F',
                   'stage': 'local_llm_experiment', 'returncode': code,
                   'elapsed_seconds': monotonic()-started,
                   'conclusion': 'Incomplete execution; no scientific advantage established'})
        return code
    return 0


def main():
    try:
        code = execute(parser().parse_args())
    except (ValueError, FileExistsError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        code = 1
    raise SystemExit(code)


if __name__ == '__main__':
    main()
