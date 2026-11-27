import json
import math
import os
from data_generation.schema import digest
from agents.client import http_json, TechnicalError


def cosine(left, right):
    if len(left) != len(right) or not left or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in left + right):
        raise TechnicalError("invalid_embedding")
    norm = math.sqrt(sum(x*x for x in left) * sum(x*x for x in right))
    if not norm:
        raise TechnicalError("zero_embedding")
    return sum(a*b for a, b in zip(left, right)) / norm


class BGE:
    def __init__(self, base, model="bge-m3:latest", timeout=60, cache=None, recorder=None):
        self.base, self.model, self.timeout = base.rstrip("/"), model, timeout
        self.cache_path, self.recorder = cache, recorder
        self.cache = json.loads(cache.read_text()) if cache and cache.exists() else {}

    def vectors(self, texts):
        keys = [digest([self.model, text]) for text in texts]
        missing = {key: text for key, text in zip(keys, texts) if key not in self.cache}
        if missing:
            import time
            start = time.monotonic()
            raw = http_json(self.base + "/api/embed", {"model": self.model, "input": list(missing.values())}, self.timeout,
                            os.environ.get("OLLAMA_API_KEY", ""))
            vectors = raw.get("embeddings", [])
            if len(vectors) != len(missing):
                raise TechnicalError("missing_embeddings")
            for key, vector in zip(missing, vectors):
                cosine(vector, vector)
                self.cache[key] = vector
            if self.cache_path:
                temporary = self.cache_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(self.cache), encoding="utf8")
                temporary.replace(self.cache_path)
            if self.recorder:
                self.recorder({"client": "embedding", "model": self.model, "count": len(missing),
                               "latency_seconds": time.monotonic() - start, "usage": {}})
        return [self.cache[key] for key in keys]


def retrieve(records, query, mode="full", topk=8, embedder=None):
    if mode == "full":
        return records
    if mode != "bge" or embedder is None or topk < 1:
        raise ValueError("Configuración de recuperación inválida")
    if not records:
        return []
    vectors = embedder.vectors([m["text"] for m in records] + [query])
    ranking = sorted(range(len(records)), key=lambda i: (-cosine(vectors[i], vectors[-1]), records[i]["event_id"]))
    selected = set(ranking[:topk])
    return [record for i, record in enumerate(records) if i in selected]


def pair_recall(records, annotation):
    if not annotation["historical_conflict"]:
        return None
    origins = {m["origin_id"] for m in records}
    return all(e["message_id"] in origins for e in annotation["evidence"])

