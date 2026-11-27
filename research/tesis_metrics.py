"""Paired family-level diagnostics for saved conversational memory trajectories."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean

from research.src.artifacts import canonical, digest, file_hash
from research.src.env.user_simulator import response_availability
from research.src.policy.rule_controller import compatible_worlds, expected_memory
from research.src.readers.symbolic import applicability, read
from research.src.scoring.evidence import Evaluator, audit_operation, probe_queries


DEFINITIONS = {
    "current_supported": "Current correct, supported answers / completed episodes.",
    "future_supported": "Future correct answers with supported persisted citations / completed episodes.",
    "joint": "Episodes with both supported answers / completed episodes.",
    "strict_joint": "Episodes with both supported answers, valid output and supported writes / completed episodes.",
    "asked": "Episodes with a delivered question / completed episodes.",
    "informative_ask": "Questions resolving a current or future value that was not established before ASK / delivered questions.",
    "informative_now": "Questions resolving the current value / delivered questions.",
    "informative_later": "Questions resolving the future value / delivered questions.",
    "redundant_ask": "Questions in episodes annotated as initially sufficient / initially sufficient episodes.",
    "missed_ask": "Episodes with no question / episodes with an authored decisive slot and an available scripted reply.",
    "unresolved_acquisition": "Episodes without a question that resolved a decision / episodes with an authored decisive slot and an available scripted reply.",
    "unproductive_ask": "Questions that did not resolve a decision / delivered questions.",
    "permanent_intent_unmet": "Permanent changes missing from faithful, functioning memory / authored new permanent changes.",
    "missed_authorized_permanent_write": "Permanent changes missing from faithful, functioning memory / authored new permanent changes that the writer can infer from its actual observation.",
    "temporary_stored_without_expiry": "Episodes with a confirmed, nonexpiring write of the temporary preference / authored session or bounded preference changes.",
    "unaffected_memory_damage": "Episodes with at least one degraded previously supported answer / episodes with at least one protected probe.",
    "unaffected_probe_damage": "Degraded answers / protected finite calendar, session and context probes.",
    "unsupported_write": "Episodes with at least one unsupported executed operation / completed episodes.",
    "invalid_output": "Episodes with at least one invalid output / completed episodes.",
}


@dataclass(frozen=True)
class Rate:
    numerator: int
    denominator: int

    @property
    def value(self):
        return self.numerator / self.denominator if self.denominator else None


def rate(event, eligible=True):
    return Rate(int(bool(event) and bool(eligible)), int(bool(eligible)))


def _reply_is_available(reply):
    if isinstance(reply, str):
        reply = {"text": reply}
    return bool(reply) and response_availability(reply) == "available"


def _question_rates(trace):
    episode = trace["episode"]
    truth = episode["truth"]
    asked = trace.get("asked_slot") is not None
    decisive_slot = truth.get("decisive_slot")
    authored_opportunity = bool(decisive_slot and _reply_is_available(episode.get("replies", {}).get(decisive_slot)))
    sufficient = "decisive_slot" in truth and decisive_slot is None
    gains = []
    for query, value in ((None, truth["current_value"]), (episode["future_query"], truth["future_value"])):
        gain = False
        if asked and trace.get("user_reply") and trace.get("user_response_status") != "unavailable":
            observations = trace["observations"]
            before, after = ({world["value"] for world in compatible_worlds(observations[phase], query=query)}
                             for phase in ("select", "respond"))
            gain = value is not None and before != {value} and after == {value}
        gains.append(gain)
    informative_now, informative_later = gains
    informative = any(gains)
    return {
        "asked": rate(asked),
        "informative_ask": rate(informative, asked),
        "informative_now": rate(informative_now, asked),
        "informative_later": rate(informative_later, asked),
        "redundant_ask": rate(asked, sufficient),
        "missed_ask": rate(not asked, authored_opportunity),
        "unresolved_acquisition": rate(not informative, authored_opportunity),
        "unproductive_ask": rate(not informative, asked),
    }


def _record_fidelity(initial, final, observation):
    originals = {record["memory_id"]: record for record in initial}
    return {
        record["memory_id"]: (record == originals[record["memory_id"]] if record["memory_id"] in originals
                              else audit_operation({"op": "SUPERSEDE" if record.get("supersedes") else "ADD",
                                                    "record": record, "target_ids": record.get("supersedes", [])},
                                                   observation))
        for record in final
    }


def _is_durable(record):
    scope = record["scope"]
    return (record["status"] == "confirmed" and scope["kind"] in {"ongoing", "contextual"}
            and not scope.get("valid_until_exclusive"))


def _global_records(memory):
    return [record for record in memory if _is_durable(record) and record["scope"].get("context") is None]


def _answers(memory, query, value, fidelity=None):
    result = read(memory, query)
    return (result["mode"] == "ANSWER" and result["value"] == value
            and (fidelity is None or any(fidelity[mid] for mid in result["evidence_ids"])))


def _scope_rates(trace, fidelity, writer_reference):
    episode = trace["episode"]
    truth = episode["truth"]
    scope = truth.get("scope") or {}
    attribute, value = truth.get("scope_attribute"), truth.get("new_value")
    authored_change = bool(attribute and value and value != truth.get("initial_value"))
    initial, final = trace["initial_memory"], trace["final_memory"]
    permanent = authorized = stored = False
    if authored_change and scope.get("kind") == "ongoing" and not scope.get("context") and not scope.get("valid_until_exclusive"):
        start = scope.get("valid_from") or episode["current_query"]["date"]
        later = max(start, episode["future_query"]["date"])
        queries = [{**episode["future_query"], "attribute": attribute, "context": None, "date": day}
                   for day in sorted({start, later})]
        permanent = not all(_answers(_global_records(initial), query, value) for query in queries)
        authorized = permanent and all(_answers(_global_records(writer_reference), query, value) for query in queries)
        stored = all(_answers(_global_records(final), query, value, fidelity) for query in queries)
    temporary = authored_change and (scope.get("kind") == "session" or bool(scope.get("valid_until_exclusive")))
    original_ids = {record["memory_id"] for record in initial}
    unbounded = any(record["memory_id"] not in original_ids and record["attribute"] == attribute
                    and record["value"] == value and _is_durable(record) for record in final)
    return {
        "permanent_intent_unmet": rate(not stored, permanent),
        "missed_authorized_permanent_write": rate(not stored, authorized),
        "temporary_stored_without_expiry": rate(unbounded, temporary),
    }


def _damage_rates(trace, fidelity):
    initial, final = trace["initial_memory"], trace["final_memory"]
    episode = trace["episode"]
    prior = {"query": episode["current_query"], "memory": initial, "user_messages": episode["user_messages"]}
    truth = episode["truth"]
    scope = truth.get("scope")
    authored = ({"scope": scope, "attribute": truth["scope_attribute"], "value": truth["new_value"]}
                if scope and truth.get("scope_attribute") and truth.get("new_value") else None)
    initial_ids = {record["memory_id"] for record in initial}
    # Judge protection against the annotated scope of the target change. Use
    # the public reference for other observable changes in the same dialogue.
    reference = [record for record in expected_memory(prior)
                 if not (authored and record["memory_id"] not in initial_ids
                         and record["attribute"] == authored["attribute"] and record["value"] == authored["value"])]
    probes = {canonical(probe): probe for probe in probe_queries(prior, *([authored] if authored else []))}
    opportunities = damaged = 0
    for attribute in sorted({record["attribute"] for record in initial}):
        for probe in probes.values():
            query = {**probe, "attribute": attribute}
            previous = read(initial, query)
            # Conservative fixed opportunity: exclude a gold change's entire
            # applicable region even if the language parser failed to see it.
            if authored and previous["value"] != authored["value"] and applicability(authored, query) is not False:
                continue
            if previous["mode"] == "ANSWER" and _answers(reference, query, previous["value"]):
                opportunities += 1
                damaged += not _answers(final, query, previous["value"], fidelity)
    return {
        "unaffected_memory_damage": rate(damaged > 0, opportunities > 0),
        "unaffected_probe_damage": Rate(damaged, opportunities),
    }


def episode_rates(row):
    """Re-score a complete native trajectory with the current evaluator."""
    trace = row["trace"]
    scorer = Evaluator()
    current = scorer.score_current(trace)["success"]
    future = scorer.score_future(trace)["success"]
    unsupported = scorer.score_memory(trace)["unsupported_operations"]
    valid_write = (trace.get("write_result") or {}).get("valid") is True
    invalid = bool(trace["errors"]) or not valid_write
    write_observation = trace["observations"]["write"]
    fidelity = _record_fidelity(trace["initial_memory"], trace["final_memory"], write_observation)
    return {
        "current_supported": rate(current),
        "future_supported": rate(future),
        "joint": rate(current and future),
        "strict_joint": rate(current and future and not invalid and not unsupported),
        "unsupported_write": rate(unsupported > 0),
        "invalid_output": rate(invalid),
        **_question_rates(trace),
        **_scope_rates(trace, fidelity, expected_memory(write_observation)),
        **_damage_rates(trace, fidelity),
    }


def _episode(row):
    return row.get("trace", {}).get("episode", row.get("episode"))


def _index(rows):
    index = {}
    for row in rows:
        episode = _episode(row)
        if not episode:
            raise ValueError("Expected native trajectory with its complete episode")
        identity = tuple(episode[key] for key in ("root_id", "latent_world_id", "realization_id"))
        seed = row["training_seed"]
        key = (seed, *identity)
        if any(row.get(field, episode[field]) != episode[field]
               for field in ("root_id", "latent_world_id", "realization_id", "split", "scenario_family")):
            raise ValueError(f"Trajectory metadata conflicts with episode {key}")
        if key in index:
            raise ValueError(f"Duplicate seed/episode {key}")
        index[key] = row
    if not index:
        raise ValueError("Empty trajectory panel")
    return index


def _root_values(cells):
    roots = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (seed, root, world, _), value in cells.items():
        roots[root][seed][world].append(value)
    return {root: mean(mean(mean(realizations) for realizations in worlds.values())
                       for worlds in seeds.values()) for root, seeds in roots.items()}


def _summaries(cells, families):
    roots = _root_values({key: tally.value for key, tally in cells.items() if tally.denominator})
    grouped = {family: [value for root, value in roots.items() if families[root] == family]
               for family in sorted(set(families.values()))}
    result = {}
    for family, values in grouped.items():
        tallies = [tally for key, tally in cells.items() if families[key[1]] == family]
        result[family] = {"macro": mean(values) if values else None,
                          "eligible_roots": len(values), "eligible_episodes": sum(bool(t.denominator) for t in tallies),
                          "numerator": sum(t.numerator for t in tallies), "denominator": sum(t.denominator for t in tallies)}
    means = [entry["macro"] for entry in result.values() if entry["macro"] is not None]
    result["all"] = {"macro": mean(means) if means else None,
                     **{key: sum(entry[key] for entry in result.values())
                        for key in ("eligible_roots", "eligible_episodes", "numerator", "denominator")}}
    return result


def _contrasts(left, right, families):
    overlap = {key: tally.value - right[key].value for key, tally in left.items()
               if tally.denominator and right[key].denominator}
    root_values = _root_values(overlap)
    groups = {family: [value for root, value in root_values.items() if families[root] == family]
              for family in sorted(set(families.values()))}
    estimates = [mean(values) for values in groups.values() if values]
    return {"macro_difference": mean(estimates) if estimates else None,
            "paired_eligible_episodes": len(overlap), "paired_eligible_roots": len(root_values),
            "only_method_eligible_episodes": sum(bool(left[key].denominator) and not right[key].denominator for key in left),
            "only_reference_eligible_episodes": sum(bool(right[key].denominator) and not left[key].denominator for key in left),
            "root_directions": dict(Counter("higher" if value > 0 else "lower" if value < 0 else "tie"
                                            for value in root_values.values())),
            "by_family": {family: {"difference": mean(values) if values else None, "paired_eligible_roots": len(values)}
                          for family, values in groups.items()}}


def compare_trajectories(panels, *, reference):
    """Match native method panels exactly, then describe changes by family."""
    if len(panels) < 2 or reference not in panels:
        raise ValueError("Provide at least two panels and an explicit reference label")
    indexes = {method: _index(rows) for method, rows in panels.items()}
    base = indexes[reference]
    for method, index in indexes.items():
        if index.keys() != base.keys():
            raise ValueError(f"Unpaired episodes in {method}: {len(base.keys() - index.keys())} missing, "
                             f"{len(index.keys() - base.keys())} extra")
    families, exclusions, contexts = {}, Counter(), set()
    seed_episodes = defaultdict(dict)
    cells = {method: {metric: {} for metric in DEFINITIONS} for method in panels}
    cache = {}
    for key, baseline_row in base.items():
        seed, root, world, realization = key
        episode = _episode(baseline_row)
        scenario = (episode["split"], baseline_row.get("condition"), baseline_row.get("route"), baseline_row.get("reader_id"))
        contexts.add(scenario)
        family = episode["scenario_family"]
        if families.setdefault(root, family) != family:
            raise ValueError(f"Root {root} belongs to multiple families")
        seed_episodes[seed][root, world, realization] = digest(episode)
        complete = True
        for method, index in indexes.items():
            row = index[key]
            if _episode(row) != episode or (row.get("condition"), row.get("route"), row.get("reader_id")) != scenario[1:]:
                raise ValueError(f"Mismatched episode, condition, route or reader in {method}: {key}")
            status = row.get("technical_status", row.get("status"))
            if status not in {"completed", "technical_failure"}:
                raise ValueError(f"Unrecognized trajectory status in {method}: {key} ({status})")
            if status == "technical_failure":
                exclusions[method] += 1
                complete = False
        if not complete:
            exclusions["paired_cells_excluded"] += 1
            continue
        for method, index in indexes.items():
            row = index[key]
            if id(row) not in cache:
                cache[id(row)] = episode_rates(row)
            for metric, tally in cache[id(row)].items():
                cells[method][metric][key] = tally
    if len(contexts) != 1:
        raise ValueError("Compare one split, observation condition, route and reader at a time")
    expected_episodes = next(iter(seed_episodes.values()))
    if any(episodes != expected_episodes for episodes in seed_episodes.values()):
        raise ValueError("Training seeds must evaluate the same episodes")
    seeds = sorted(seed_episodes)
    split, condition, route, reader = next(iter(contexts))
    return {
        "scope": {"split": split, "condition": condition, "route": route, "reader": reader,
                  "training_seeds": seeds, "independent_roots": len(families), "scheduled_paired_cells": len(base),
                  "complete_paired_cells": len(base) - exclusions["paired_cells_excluded"],
                  "evaluator_version": Evaluator.version},
        "technical_exclusions": {"by_method": {method: exclusions[method] for method in panels},
                                 "paired_cells_excluded": exclusions["paired_cells_excluded"]},
        "definitions": DEFINITIONS,
        "interpretation": [
            "Explicit technical failures exclude the same seed/episode in all panels. Invalid outputs remain scored.",
            "Raw counts pool eligible episodes, or probes for unaffected_probe_damage. Macro means weight eligible realizations within worlds, worlds within seeds, seeds within roots, roots within families and families equally.",
            "Contrasts compare only cells eligible in both panels. An empty denominator yields null. Results are descriptive and conditional on the observed seeds.",
            "Questions use literal decisive-slot and reply annotations for opportunities and the public solver for resolved answers. The authored reply may become unavailable under a robustness intervention.",
            "Protected probes require a previously decisive answer outside the annotated region of a conflicting target change, unchanged by other observable changes in the shared initial dialogue. This finite test does not inventory global memory.",
        ],
        "reference": reference,
        "methods": {method: {metric: _summaries(values, families) for metric, values in metrics.items()}
                    for method, metrics in cells.items()},
        "by_seed": {seed: {method: {metric: _summaries({key: value for key, value in values.items() if key[0] == seed}, families)
                                   for metric, values in metrics.items()} for method, metrics in cells.items()}
                    for seed in seeds},
        "paired_vs_reference": {method: {metric: _contrasts(values, cells[reference][metric], families)
                                         for metric, values in metrics.items()}
                                for method, metrics in cells.items() if method != reference},
    }


def _comparison_inputs(result_path):
    with result_path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if result.get("status") != "completed_exploratory":
        raise ValueError("Expected completed comparison RESULTS.json")
    if set(result["panels"]) != {str(seed) for seed in result["seeds"]}:
        raise ValueError("Comparison panels do not match declared training seeds")
    entries = defaultdict(list)
    for seed, panels in result["panels"].items():
        for label, panel in panels.items():
            path = Path(panel["trajectories"])
            if not path.is_absolute():
                path = result_path.parent / path
            entries[label].append((path, panel["trajectories_sha256"], int(seed)))
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--panel", action="append", metavar="LABEL=PATH",
                        help="Repeat for each method and each seed's JSONL trace")
    source.add_argument("--comparison-results", type=Path, help="Completed thesis comparison RESULTS.json")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    entries = _comparison_inputs(args.comparison_results) if args.comparison_results else defaultdict(list)
    for panel in args.panel or ():
        label, separator, raw_path = panel.partition("=")
        if not separator or not label or not raw_path:
            parser.error("Use --panel LABEL=PATH")
        entries[label].append((Path(raw_path), None, None))
    records = {}
    hashes = {}
    sources = defaultdict(list)
    panels = defaultdict(list)
    for label, inputs in entries.items():
        for path, expected_hash, seed in inputs:
            path = path.resolve()
            if path not in records:
                hashes[path] = file_hash(path)
                with path.open(encoding="utf-8") as stream:
                    records[path] = [json.loads(line) for line in stream if line.strip()]
            if expected_hash and hashes[path] != expected_hash:
                raise ValueError(f"Trajectory hash mismatch: {path}")
            if seed is not None and {row["training_seed"] for row in records[path]} != {seed}:
                raise ValueError(f"Trajectory training seed mismatch: {path}")
            panels[label].extend(records[path])
            sources[label].append({"path": str(path), "sha256": hashes[path]})
    result = compare_trajectories(panels, reference=args.reference)
    result["sources"] = sources
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{result['scope']['complete_paired_cells']} matched seed/episode cells · {args.output}")


if __name__ == "__main__":
    main()
