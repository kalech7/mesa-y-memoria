"""Launch the authorized offline experiment with a durable, exclusive logfile.

The child owns a separate session and file-backed stdout/stderr, so losing the
interactive client cannot raise BrokenPipeError inside training/validation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--process-manifest', type=Path, required=True)
    parser.add_argument('experiment_arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.experiment_arguments
    if argv and argv[0] == '--':
        argv = argv[1:]
    if not argv:
        parser.error('Pass the llm_local_001.run arguments after --')
    root = Path(__file__).resolve().parents[1]
    log = args.log.resolve()
    manifest = args.process_manifest.resolve()
    if log.exists() or manifest.exists():
        raise FileExistsError('Use new log and process-manifest paths; previous runs are preserved')
    log.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-B', '-m', 'llm_local_001.run', *argv]
    with log.open('xb') as stream:
        process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   close_fds=True, start_new_session=True)
    record = {'created_at': datetime.now(timezone.utc).isoformat(),
              'source': 'scripts/launch_local_experiment.py', 'pid': process.pid,
              'command': command, 'cwd': str(root), 'log': str(log),
              'stdout_stderr': 'exclusive regular file, not an interactive pipe',
              'new_process_session': True}
    manifest.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
