"""Diagnóstico real BGE de evidencia inicial, sin agente ni juez y solo desarrollo."""
import argparse
import json
import os
from pathlib import Path
from agents.client import TechnicalError
from data_generation.schema import digest, validate_dataset
from evaluation.runner import load_jsonl, balanced_select, atomic_json, append_jsonl, source_hash, check_gates
from memory.retrieval import BGE, retrieve, pair_recall


def probe(case, annotation, embedder):
    records = [{"event_id": m["message_id"], "origin_id": m["message_id"], "text": m["text"]} for m in case["history"]]
    rows = []
    corrections = {m["message_id"] for m in case["history"] if "Corrijo lo anterior" in m["text"]}
    for k in (4, 8, 16):
        selected = retrieve(records, case["query"], "bge", k, embedder)
        origins = {m["origin_id"] for m in selected}
        rows.append({"case_id": case["case_id"], "person_id": case["person_id"], "topk": k,
            "retrieved_origin_ids": sorted(origins), "pair_recall": pair_recall(selected, annotation),
            "correction_retrieved": bool(corrections & origins) if corrections else None})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default=os.environ.get("OLLAMA_BASE_URL", ""))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)
    if not args.url or not 5 <= args.limit <= 10:
        raise SystemExit("Requiere endpoint privado y 5–10 casos de desarrollo")
    public = load_jsonl(args.dataset / "casos_agent.jsonl")
    gold = load_jsonl(args.dataset / "private" / "casos_gold.jsonl")
    validate_dataset(public, gold)
    selected = balanced_select(public, gold, "dev", args.limit, args.seed)
    reasons = check_gates(selected, gold)
    if reasons:
        print(json.dumps({"probe_blocked": True, "reasons": reasons}, ensure_ascii=False))
        return 2
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "manifest.json", {"public_hash": digest(public), "gold_hash": digest(gold),
        "source_hash": source_hash(), "seed": args.seed, "cases": len(selected), "split": "dev",
        "gold_provisional": True, "purpose": "initial_retrieval_only", "agent_inference": False})
    embedder = BGE(args.url, cache=args.output / "embedding_cache.json", recorder=lambda r: append_jsonl(args.output / "embedding_trace.jsonl", r))
    rows = []
    try:
        for case, annotation in selected:
            for row in probe(case, annotation, embedder):
                append_jsonl(args.output / "per_case.jsonl", row)
                rows.append(row)
    except TechnicalError as exc:
        atomic_json(args.output / "error.json", {"kind": exc.kind, "http_status": exc.status})
        return 3
    summary = {}
    for k in (4, 8, 16):
        sub = [r for r in rows if r["topk"] == k]
        pairs = [r for r in sub if r["pair_recall"] is not None]
        corrections = [r for r in sub if r["correction_retrieved"] is not None]
        summary[str(k)] = {"n": len(sub), "pair_hits": sum(r["pair_recall"] for r in pairs), "pair_denominator": len(pairs),
            "correction_hits": sum(r["correction_retrieved"] for r in corrections), "correction_denominator": len(corrections)}
    atomic_json(args.output / "metrics.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
