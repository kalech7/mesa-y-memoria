"""Traceable, strict JSON/CSV artifacts, with no training dependencies."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + '\n')


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + '\n')


def write_csv(path, rows, fields, metadata):
    """Even unavailable results carry explicit status and provenance, never zeroes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(metadata) + [field for field in fields if field not in metadata]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({**metadata, **{k: json.dumps(v, ensure_ascii=False)
                                           if isinstance(v, (dict, list)) else v
                                           for k, v in row.items()}})
