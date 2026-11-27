"""Finish the locked evaluations after a live primary pilot, without retraining.

Runs only already specified evaluators. File-backed launch survives client exits.
Existing partial evaluations are preserved and deliberately require inspection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot', type=Path)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--primary-pid', type=int, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = args.pilot.resolve()
    protocol = args.protocol.resolve()
    commands = []
    started = datetime.now(timezone.utc).isoformat()
    status_path = root / 'run_status.json'
    print(f'{started}: waiting for completed primary pilot {root}', flush=True)
    while True:
        if status_path.exists():
            status = read(status_path)
            if status.get('status') == 'COMPLETED_TECHNICAL_PILOT' and status.get('test_evaluated') is True:
                break
            raise RuntimeError(f'Primary pilot did not complete: {status}')
        process_status = Path(f'/proc/{args.primary_pid}/stat')
        if not process_status.exists() or process_status.read_text().split(') ')[1].startswith('Z'):
            raise RuntimeError('Primary process stopped without a completed result; preserve and inspect logs')
        time.sleep(15)

    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(repo), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       HF_HUB_DISABLE_TELEMETRY='1')

    def execute(command):
        print(f'{datetime.now(timezone.utc).isoformat()}: {command}', flush=True)
        commands.append(command)
        subprocess.run(command, cwd=repo, env=environment, check=True)

    supplement = root / 'supplementary_observability.json'
    if supplement.exists():
        if read(supplement).get('status') != 'COMPLETED_PRESPECIFIED_SUPPLEMENT':
            raise RuntimeError('Existing supplement is incomplete; preserve it')
    else:
        execute([sys.executable, '-B', str(repo / 'scripts/evaluate_locked_observability.py'),
                 str(root), '--protocol', str(protocol)])
    execute([sys.executable, '-B', str(repo / 'scripts/analyze_llm_local_results.py'), str(root),
             '--reward-audit', str(repo / 'reports/llm_local_001_preparation/reward_shortcut_audit.json'),
             '--observability-supplement', str(supplement)])
    figures = root / 'figures'
    if figures.exists():
        raise RuntimeError('Existing figures preserved; inspect before regenerating')
    execute([str(repo / '.venv-llm-plots/bin/python'), '-B',
             str(repo / 'scripts/plot_llm_local_results.py'), str(root)])
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'started_at': started,
                'source': str(Path(__file__).resolve()),
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'primary_pilot': str(root), 'protocol': str(protocol),
                'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
                'commands': commands, 'training_or_checkpoint_selection': False,
                'status': 'COMPLETED_PRIMARY_SUPPLEMENT_ANALYSIS_AND_PLOTS'}
    destination = root / 'completion_pipeline.json'
    with destination.open('x') as handle:
        json.dump(manifest, handle, indent=2)
        handle.write('\n')
    print(manifest['status'], flush=True)


if __name__ == '__main__':
    main()
