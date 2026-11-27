"""Explicitly authorized model acquisition, separate from offline experiments.

Downloads weights/tokenizer/license only; never performs remote inference.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def main():
    from huggingface_hub import HfApi, snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--destination', type=Path, default=Path('models/Qwen2.5-0.5B-Instruct'))
    args = parser.parse_args()
    api = HfApi()
    info = api.model_info(args.repo, files_metadata=True)
    revision = info.sha
    allowed = ['*.safetensors', '*.json', 'merges.txt', 'vocab.json', 'LICENSE', 'README.md']
    print(f'Descargando {args.repo} revision {revision} a {args.destination}', flush=True)
    snapshot_download(args.repo, revision=revision, local_dir=str(args.destination),
                      allow_patterns=allowed, max_workers=3)
    files = {}
    for path in sorted(args.destination.iterdir()):
        if path.is_file() and path.name != 'download_manifest.json':
            sha = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    sha.update(block)
            files[path.name] = {'bytes': path.stat().st_size, 'sha256': sha.hexdigest()}
    record = {'created_at': datetime.now(timezone.utc).isoformat(),
              'source': 'scripts/download_llm_local.py via huggingface_hub',
              'repo': args.repo, 'revision': revision, 'files': files,
              'authorization': 'User explicitly requested searching and downloading a model and executing local experiment',
              'remote_inference_calls': 0, 'local_directory': str(args.destination.resolve())}
    (args.destination/'download_manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'revision': revision, 'files': len(files),
                      'bytes': sum(item['bytes'] for item in files.values())}), flush=True)


if __name__ == '__main__':
    main()
