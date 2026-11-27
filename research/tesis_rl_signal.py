"""Audit delayed reward and the effect of verifying WRITE on sampled TRAIN branches."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from research.grpo_select_pilot import policy_return, scores
from research.src.artifacts import digest, file_hash, write_json
from research.tesis_memory_replay import replay_case


def signal(draws):
    def varies(values):
        return len({round(value, 10) for value in values}) > 1

    task, returns = zip(*(policy_return(metrics) for metrics in draws))
    immediate = [.5 * m["C2"] - .10 * m["questions"] - .50 * m["unsupported_write"]
                 - .20 * m["invalid_output"] for m in draws]

    def best(values):
        maximum = round(max(values), 10)
        return {i for i, value in enumerate(values) if round(value, 10) == maximum}

    future = varies([m["FJS"] for m in draws])
    current = varies([m["C2"] for m in draws])
    task_varies = varies(task)
    return_varies = varies(returns)
    penalties_only = return_varies and not task_varies
    return {"groups": 1, "task_varies": int(task_varies), "return_varies": int(return_varies),
            "penalties_only": int(penalties_only), "question_cost_only": int(penalties_only and not any(
                varies([m[k] for m in draws]) for k in ("unsupported_write", "invalid_output"))),
            "future_varies": int(future), "future_varies_with_current_fixed": int(future and not current),
            "preferred_draws_change_with_future_terms": int(best(immediate) != best(returns))}


def audit(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    plan_path = run / "PLAN.json"
    plan = json.loads(plan_path.read_bytes())
    sources = {str(plan_path): file_hash(plan_path), str(Path(__file__)): file_hash(Path(__file__))}
    for name in ("tesis_memory_guard.py", "tesis_memory_replay.py"):
        source = Path(__file__).with_name(name)
        sources[str(source)] = file_hash(source)
    groups = []
    counters = defaultdict(lambda: {arm: Counter() for arm in ("original", "repair")})
    with TemporaryDirectory(prefix="tesis-grpo-signal-") as scratch:
        for seed in plan["seeds"]:
            log = run / "training" / str(seed) / "GRPO-SELECT/training.jsonl"
            sources[str(log)] = file_hash(log)
            for line in log.read_text(encoding="utf-8").splitlines():
                group = json.loads(line)
                original, repaired, trace_hashes, family = [], [], {}, None
                for draw in group["draws"]:
                    path = Path(draw["trajectory"])
                    if str(path) not in trace_hashes:
                        branch = json.loads(path.read_bytes())
                        if branch["split"] != "train" or branch["training_seed"] != seed:
                            raise ValueError("Expected archived TRAIN branches for the recorded seed")
                        family = branch["scenario_family"]
                        replay = replay_case(branch, Path(scratch) / f"{seed}_{path.stem}")
                        trace_hashes[str(path)] = {"sha256": file_hash(path), "repair": scores(replay["repair"]),
                                                    "original": scores(replay["original"])}
                    saved = trace_hashes[str(path)]
                    if saved["original"] != draw["metrics"]:
                        raise ValueError("Archived training draw score changed")
                    original.append(saved["original"])
                    repaired.append(saved["repair"])
                arms = {"original": signal(original), "repair": signal(repaired)}
                for key in ("total", str(seed), family):
                    for arm, counts in arms.items():
                        counters[key][arm].update(counts)
                groups.append({"seed": seed, "group": group["group"], "episode": group["episode"], "family": family,
                               "actions": [draw["action"] for draw in group["draws"]], "signals": arms,
                               "original": original, "repair": repaired, "source_draws_sha256": digest(group["draws"]),
                               "branches": trace_hashes})
    result = {"scope": "retrospective_rescoring_of_sampled_train_actions_no_weight_updates",
              "comparison_sha256": file_hash(run / "RESULTS.json"), "sources": sources,
              "counts": counters, "groups": groups}
    if output.exists():
        raise FileExistsError(output)
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.output)
    print(json.dumps(result["counts"]["total"], ensure_ascii=False))


if __name__ == "__main__":
    main()
