"""Challenge the WRITE verifier with public prefixes from reserved multi-turn traces."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from research.src.artifacts import file_hash, write_json
from research.tesis_interactive_env import REUSE
from research.tesis_memory_guard import NOOP, plan_write, write_effect


METHOD = "GRPO-longitudinal-terminal"


def audit(run):
    result_path = run / "RESULTS.json"
    result = json.loads(result_path.read_bytes())
    counts = Counter()
    trace_hashes = {}
    for seed in result["seeds"]:
        panel = result["panels"][str(seed)][METHOD]
        path = run / panel["traces"]
        trace_hashes[str(seed)] = file_hash(path)
        if trace_hashes[str(seed)] != panel["trace_sha256"]:
            raise ValueError(f"Changed input trace for seed {seed}")
        rows = list(map(json.loads, path.read_text(encoding="utf-8").splitlines()))
        identities = {(row["root_id"], row["world_id"]) for row in rows}
        if len(rows) != len(identities) or len(rows) != result["evaluation_episodes_per_seed"]:
            raise ValueError("Expected unique episodes from the entire reserved split")
        for row in rows:
            if row["split"] != "evaluation":
                raise ValueError("Expected a reserved trace")
            counts["episodes"] += 1
            for visit in row["visits"]:
                counts["visits"] += 1
                messages = visit["opening_messages"] + [
                    message for event in visit["events"] for message in (event["question"], event["reply"])
                ]
                observation = {"user_messages": messages, "query": visit["query"],
                               "memory": visit["memory_before"], "reuse_regime": REUSE}
                replay = []
                for applied in visit["executed_operations"]:
                    operation = deepcopy(applied)
                    if operation != NOOP:
                        operation["record"].pop("memory_id")
                        operation["record"].pop("recorded_at")
                    replay.append(operation)
                verified = plan_write(observation, {"operations": replay or [NOOP]}, mode="veto")
                confirmed = [op for op in replay if op != NOOP]
                counts["confirmed_operations"] += len(confirmed)
                counts["confirmed_operations_retained"] += len(verified["accepted_indices"])
                if confirmed:
                    repaired = plan_write(observation, {"operations": [NOOP]}, mode="repair")
                    recovered = {write_effect(op) for op in repaired["recovered_operations"]}
                    counts["confirmed_operations_recovered_from_noop"] += sum(
                        write_effect(op) in recovered for op in confirmed
                    )
                for operation in confirmed:
                    fabricated = deepcopy(operation)
                    fabricated["record"]["value"] = "caramelo de prueba sin mención"
                    variants = {"fabricated_value": fabricated}
                    if operation["record"]["scope"]["kind"] in {"session", "interval", "contextual"}:
                        expanded = deepcopy(operation)
                        expanded["record"]["scope"] = {**expanded["record"]["scope"], "kind": "ongoing",
                                                        "context": None, "session_id": None,
                                                        "valid_until_exclusive": None}
                        variants["expanded_scope"] = expanded
                    for variant, mutation in variants.items():
                        checked = plan_write(observation, {"operations": [mutation]}, mode="veto")
                        counts[variant] += 1
                        counts[f"{variant}_rejected"] += checked["operations"] == [NOOP]
    return {"scope": "Fixed reserved WRITE prefixes; fabricated values and enlarged scopes are injected controls",
            "method": METHOD, "seeds": result["seeds"], "results_sha256": file_hash(result_path),
            "trace_sha256": trace_hashes, "counts": dict(counts)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checked = audit(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, checked)
    print(json.dumps(checked["counts"], ensure_ascii=False))
