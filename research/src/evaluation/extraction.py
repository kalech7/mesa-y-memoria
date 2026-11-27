"""Shared, immutable cache for the frozen S1 extraction robustness track."""
from pathlib import Path
from copy import deepcopy
import json
from ..artifacts import digest, write_json
from ..memory.executor import validate_record


def extract_cached(messages, frozen_extractor, cache_dir, *, extractor_hash, prompt_hash):
    key = digest({"messages": messages, "extractor_hash": extractor_hash, "prompt_hash": prompt_hash})
    path = Path(cache_dir) / (key + ".json")
    if path.exists():
        return deepcopy(json.loads(path.read_text())["records"])
    records = frozen_extractor(deepcopy(messages))
    for record in records:
        validate_record(record, initial=True)
    write_json(path, {"key": key, "extractor_hash": extractor_hash, "prompt_hash": prompt_hash, "records": records})
    return deepcopy(records)
