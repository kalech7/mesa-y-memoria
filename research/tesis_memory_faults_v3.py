"""Paired fault injection into writes and retrieval with fixed clarification actions."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import date, timedelta
from itertools import product
from pathlib import Path
import random
from statistics import mean

from research.src.artifacts import digest, file_hash, write_json
from research.src.memory.executor import validate_record
from research.src.readers.symbolic import applicability, read
from research.src.scoring.evidence import audit_operation
from research.tesis_bank import FOOD, _remember
from research.tesis_extended_env import ExtendedEpisode
from research.tesis_interactive_env import _fact, _possible_values, _prior, _turn, _update_truth
from research.tesis_shared_budget_env import _expired_citations


SEEDS = (101, 202, 303)
FAMILIES = ("intervalo_explicito", "intervalo_aclarado", "intervalo_contextual", "solo_recuperacion")
EXTRA_DAYS = 30
ARM_FACTORS = tuple(product((False, True), (False, True), ("veto", "off")))


def _query(attribute, day, session, context=None):
    return {"attribute": attribute, "date": day.isoformat(), "session_id": session, "context": context}


def _messages(number, opening, statement, meal):
    return [
        _turn(number, "inicio", "user", opening),
        _turn(number, "asistente", "assistant", "Tengo la conversación anterior a mano."),
        _turn(number, "preferencia", "user", statement),
        _turn(number, "consulta", "user", f"¿Qué opción pongo para {meal}?"),
    ]


def fixture(seed, family, index):
    """Independent diagnostic worlds with a known date boundary and a stable control."""
    rng = random.Random(digest(["memory-faults-v3", seed, family, index]))
    primary, secondary, control = rng.sample(list(FOOD), 3)
    menu, options = FOOD[primary]
    other_menu, other_options = FOOD[secondary]
    control_menu, control_options = FOOD[control]
    baseline, changed = rng.sample(options, 2)
    other_default, obsolete = rng.sample(other_options, 2)
    # One case per seed and family deliberately has the same value in the
    # stale and active records, so correctness alone misses the bad citation.
    coincident = index == 0
    if coincident:
        obsolete = other_default
    stable = rng.choice(control_options)
    start = date(2044, 3, 5) + timedelta(days=rng.randint(0, 130))
    end = start + timedelta(days=rng.randint(3, 5))
    last_day = (end - timedelta(days=1)).isoformat()
    office = "oficina" if family == "intervalo_contextual" else None
    memory, histories = [], {}
    recent = (start - timedelta(days=45)).isoformat()
    for attribute, meal, value in ((primary, menu, baseline), (secondary, other_menu, other_default),
                                   (control, control_menu, stable)):
        _remember(memory, histories, attribute, value, recent,
                  f"Para {meal}, prefiero {value} de forma habitual.")
    archived_start = (start - timedelta(days=20)).isoformat()
    archived_end = (start - timedelta(days=17)).isoformat()
    old_last = (date.fromisoformat(archived_end) - timedelta(days=1)).isoformat()
    _remember(memory, histories, secondary, obsolete, archived_start,
              f"Para {other_menu}, prefiero {obsolete}. Hasta el {old_last} inclusive.",
              kind="interval", end=archived_end)

    written = family != "solo_recuperacion"
    date_clarified = family == "intervalo_aclarado"
    fact = (_fact(primary, changed, "contextual" if office else "interval",
                  start.isoformat(), context=office, expiry=end.isoformat()) if written else None)
    declaration = f"Para {menu}, prefiero {changed}" + (" en la oficina." if office else ".")
    if date_clarified:
        statement = declaration + " El cambio es por unos días."
    elif written:
        statement = declaration + f" Hasta el {last_day} inclusive."
    else:
        statement = "Hoy mantengo mi elección habitual."
    first_query = _query(primary, start, "S2", office)
    states = [_prior(record) for record in memory]
    truth = states + ([fact] if fact else [])
    alternative = None
    if date_clarified:
        later = _fact(primary, changed, "interval", start.isoformat(),
                      expiry=(end + timedelta(days=7)).isoformat())
        alternative = [fact, later]
    visits = [{
        "query": first_query,
        "messages": _messages(1, "Voy a dejar por escrito mi menú de esta semana.", statement, menu),
        "expected": _update_truth(truth, first_query),
        "slot": "VALID_UNTIL" if date_clarified else None, "fact": fact, "alternatives": alternative,
        "style": "direct", "answer": f"Hasta el {last_day} inclusive." if date_clarified else None,
    }]
    for number in (2, 3):
        query = _query(primary, start + timedelta(days=number - 1), f"S{number + 1}", office)
        visits.append({
            "query": query,
            "messages": _messages(number, "Volví al cuaderno del menú.", "Quiero consultar lo que quedó guardado.", menu),
            "expected": _update_truth(truth, query), "slot": None, "fact": None, "alternatives": None,
            "style": "direct", "answer": None,
        })
    probes = (
        ("ultimo_dia_vigente", _query(primary, end - timedelta(days=1), "S5", office)),
        ("dia_de_vencimiento", _query(primary, end, "S5", office)),
        ("fecha_posterior", _query(primary, end + timedelta(days=4), "S5", office)),
        ("fuera_de_contexto", _query(primary, end - timedelta(days=1), "S5", "casa")),
        ("vencido_preexistente", _query(secondary, end, "S5")),
        ("preferencia_estable", _query(control, end, "S5")),
    )
    root_id = "fault-" + digest([seed, family, index, memory, first_query])[:24]
    return {
        "root_id": root_id, "world_id": root_id + "-world", "realization_id": "v0",
        "split": "validation", "scenario_family": family, "second_visit_kind": "reuse",
        "impatient": False, "initial_memory": memory, "prior_sessions": list(histories.values()),
        "visits": visits, "question_budget": 1, "world_probability": 1.,
        "future_queries": [{"kind": kind, "query": query, "expected": _update_truth(truth, query)}
                           for kind, query in probes],
        "coincident_stale_value": coincident, "recipe_id": root_id,
    }


def broaden_expiry(operation):
    """Change only the end date of a well-formed finite write."""
    corrupted = deepcopy(operation)
    scope = corrupted["record"]["scope"]
    end = date.fromisoformat(scope["valid_until_exclusive"])
    scope["valid_until_exclusive"] = (end + timedelta(days=EXTRA_DAYS)).isoformat()
    validate_record(corrupted["record"])
    return corrupted


def read_ignoring_expiry(memory, query):
    """Simulate skipping the end-date check while preserving all other reader logic."""
    candidates = deepcopy(memory)
    for record in candidates:
        scope = record["scope"]
        expiry = scope["valid_until_exclusive"]
        if expiry is not None and expiry <= query["date"]:
            day = date.fromisoformat(query["date"])
            scope["valid_until_exclusive"] = (day + timedelta(days=1)).isoformat()
    return read(candidates, query)


def guard_expired_citations(memory, query, candidate):
    """At the response boundary, abstain if a candidate cites a stored expired record."""
    if not _expired_citations(memory, candidate, query):
        return candidate, False
    return {"mode": "DEFER", "value": None, "branches": [], "evidence_ids": []}, True


def _signature(operation):
    record = operation.get("record")
    if not record:
        return None
    return [operation["op"], record["attribute"], record["value"], record["scope"],
            record["source_turn_ids"]]


def _declare_future(memory, query, expected, knowledge, tainted, result):
    ids = set(result["evidence_ids"])
    source_by_id = {record["memory_id"]: record for record in memory}
    answer = result["mode"] == "ANSWER"
    correct = answer and result["value"] == expected
    expired = _expired_citations(memory, result, query)
    bad = sorted(ids & tainted)
    supported = (correct and bool(ids)
                 and all(mid in source_by_id and applicability(source_by_id[mid], query) is True for mid in ids)
                 and not expired and not bad and _possible_values(knowledge, query) == {expected})
    return {
        "response": result, "correct": int(correct), "supported": int(supported),
        "unsupported_commitment": int(answer and not supported),
        "expired_citation_ids": expired, "unsupported_write_citation_ids": bad,
    }


class FaultEpisode(ExtendedEpisode):
    def __init__(self, story, *, writer_fault, reader_fault, verifier):
        self.writer_fault = writer_fault
        self.reader_fault = reader_fault
        self.writer_attempts = []
        super().__init__(story, verifier=verifier)

    def _propose_writes(self, candidates, visible):
        proposed = []
        for operation, supported in candidates:
            record = operation.get("record")
            finite = bool(record and record["scope"]["valid_until_exclusive"])
            injected = self.writer_fault and supported and finite
            after = broaden_expiry(operation) if injected else operation
            outcome = audit_operation(after, visible) if injected else supported
            self.writer_attempts.append({
                "visit": len(self._visits) + 1, "original": deepcopy(operation),
                "proposal": deepcopy(after), "injected": bool(injected), "evidence_ok": bool(outcome),
                "vetoed": not outcome and self.verifier == "veto",
            })
            proposed.append((after, outcome))
        return tuple(proposed)

    def _read_future(self, query):
        return (read_ignoring_expiry if self.reader_fault else read)(self._final, query)

    def fault_summary(self):
        trace = self.summary()
        writes = []
        for entry in self.writer_attempts:
            executed = trace["visits"][entry["visit"] - 1]["executed_operations"]
            match = next((operation for operation in executed
                          if _signature(operation) == _signature(entry["proposal"])), None)
            writes.append({**entry, "applied": match is not None,
                           "persisted_id": match["record"]["memory_id"] if match else None})
        queries = []
        final, tainted, belief = trace["final_memory"], self._unsupported_memory_ids, self._knowledge
        for probe in trace["future"]:
            query, expected, candidate = probe["query"], probe["expected"], probe["response"]
            guarded, rejected = guard_expired_citations(final, query, candidate)
            raw = _declare_future(final, query, expected, belief, tainted, candidate)
            safe = _declare_future(final, query, expected, belief, tainted, guarded)
            queries.append({"kind": probe["kind"], "query": query, "expected": expected,
                            "candidate": raw, "guarded": safe, "reader_veto": rejected})
        return {**trace, "write_audit": writes, "query_audit": queries}


def fixed_path(story):
    """Give the same authorized clarification and closure to every intervention."""
    first = ("VALID_UNTIL",) if story["scenario_family"] == "intervalo_aclarado" else ()
    return (*first, "PROCEED", "PROCEED", "PROCEED")


def replay(story, *, writer_fault, reader_fault, verifier):
    path = fixed_path(story)
    env = FaultEpisode(story, writer_fault=writer_fault, reader_fault=reader_fault, verifier=verifier)
    try:
        for action in path:
            env.step(action)
        trace = env.fault_summary()
        if tuple(trace["actions"]) != path:
            raise ValueError("The paired clarification path changed during the replay")
        return trace
    finally:
        env.close()


def _arm_label(writer, reader, verifier):
    return f"W{int(writer)}-R{int(reader)}-{verifier}"


def _metrics(trace):
    audit = trace["query_audit"]
    writes = trace["write_audit"]
    reopened_ids = {record["memory_id"] for record in trace["final_memory"]}
    return {
        "queries": len(audit), "questions": trace["questions"],
        "proposals": len(writes), "fault_attempts": sum(w["injected"] for w in writes),
        "fault_vetoes": sum(w["injected"] and w["vetoed"] for w in writes),
        "fault_applied": sum(w["injected"] and w["applied"] for w in writes),
        "fault_persisted": sum(w["injected"] and w["persisted_id"] in reopened_ids for w in writes),
        "unsupported_writes": sum(w["applied"] and not w["evidence_ok"] for w in writes),
        "fault_applied_but_not_cited": sum(
            w["injected"] and w["applied"]
            and all(w["persisted_id"] not in probe["candidate"]["response"]["evidence_ids"] for probe in audit)
            for w in writes
        ),
        "current_supported": mean(visit["current_supported"] for visit in trace["visits"]),
        "current_correct": mean(visit["current_correct"] for visit in trace["visits"]),
        "correct": sum(p["candidate"]["correct"] for p in audit),
        "supported": sum(p["candidate"]["supported"] for p in audit),
        "unsupported_commitments": sum(p["candidate"]["unsupported_commitment"] for p in audit),
        "expired_citations": sum(bool(p["candidate"]["expired_citation_ids"]) for p in audit),
        "expired_correct": sum(bool(p["candidate"]["expired_citation_ids"]) and p["candidate"]["correct"] for p in audit),
        "tainted_citations": sum(bool(p["candidate"]["unsupported_write_citation_ids"]) for p in audit),
        "reader_vetoes": sum(p["reader_veto"] for p in audit),
        "guarded_expired_citations": sum(bool(p["guarded"]["expired_citation_ids"]) for p in audit),
        "guarded_unsupported_commitments": sum(p["guarded"]["unsupported_commitment"] for p in audit),
        "guarded_supported": sum(p["guarded"]["supported"] for p in audit),
    }


def _compose(rows):
    """Proportions refer to queries, counts to attempted operations and citations."""
    counts = Counter()
    for row in rows:
        counts.update(row["metrics"])
    queries = counts["queries"]
    return {
        "roots": len(rows), **dict(counts), "current_supported": counts["current_supported"] / len(rows),
        "current_correct": counts["current_correct"] / len(rows),
        "future_supported_rate": counts["supported"] / queries,
        "future_correct_rate": counts["correct"] / queries,
        "guarded_supported_rate": counts["guarded_supported"] / queries,
    }


def _paired(rows):
    """One matched effect per root and a mean for each generation seed."""
    arms = {}
    for row in rows:
        root = (row["generation_seed"], row["root_id"])
        arms.setdefault(row["arm"], {})[root] = row["metrics"]
    baseline = arms[_arm_label(False, False, "veto")]
    effect = {}
    for arm, roots in arms.items():
        if roots.keys() != baseline.keys():
            raise ValueError("Every intervention must replay exactly the same roots")
        by_seed = {}
        for seed in SEEDS:
            observations = [root for root in roots if root[0] == seed]
            by_seed[str(seed)] = {
                field: mean((roots[root][field] - baseline[root][field]) / baseline[root]["queries"]
                            for root in observations)
                for field in ("supported", "expired_citations", "tainted_citations", "unsupported_commitments")
            }
        effect[arm] = {
            "mean_difference": {field: mean(row[field] for row in by_seed.values())
                                for field in next(iter(by_seed.values()))},
            "difference_by_generation_seed": by_seed,
        }
    return effect


def _audit(rows):
    """Check treatment coverage and the paired no-fault and stable controls."""
    labels = {_arm_label(*args) for args in ARM_FACTORS}
    by_root = {}
    for row in rows:
        by_root.setdefault(row["root_id"], {})[row["arm"]] = row
    for arms in by_root.values():
        if arms.keys() != labels or len({tuple(r["actions"]) for r in arms.values()}) != 1:
            raise ValueError("Missing arm or changing actions under intervention")
        if len({digest([[visit["opening_messages"], visit["events"]] for visit in row["trace"]["visits"]])
                for row in arms.values()}) != 1:
            raise ValueError("A paired arm changed the original user messages or replies")
        no_fault = arms[_arm_label(False, False, "veto")]
        expected_attempt = int(no_fault["family"] != "solo_recuperacion")
        if no_fault["metrics"]["supported"] != no_fault["metrics"]["queries"]:
            raise ValueError("A no-fault control did not recover all reference values")
        if arms[_arm_label(False, False, "off")]["metrics"] != no_fault["metrics"]:
            raise ValueError("The writer veto changed a valid-write control")
        for writer, verifier in product((False, True), ("veto", "off")):
            clean = arms[_arm_label(writer, False, verifier)]["trace"]
            stale = arms[_arm_label(writer, True, verifier)]["trace"]
            if any(clean[key] != stale[key] for key in ("visits", "final_memory", "write_audit")):
                raise ValueError("The reader intervention altered the acquisition trajectory")
        for row in arms.values():
            if row["writer_fault"] and row["metrics"]["fault_attempts"] != expected_attempt:
                raise ValueError("A targeted writer mutation was missing or duplicated")
            if row["metrics"]["guarded_expired_citations"]:
                raise ValueError("An expired citation survived the reader guard")
            stable = next(probe for probe in row["trace"]["query_audit"]
                          if probe["kind"] == "preferencia_estable")
            if not stable["candidate"]["supported"]:
                raise ValueError("The negative-control attribute was affected")


def summarize(rows):
    arms = {_arm_label(*args): _compose([row for row in rows if row["arm"] == _arm_label(*args)])
            for args in ARM_FACTORS}
    by_query = {}
    counter_names = ("correct", "supported", "unsupported_commitment")
    for label in arms:
        groups = {}
        for row in rows:
            if row["arm"] != label:
                continue
            for probe in row["trace"]["query_audit"]:
                candidate, guarded = probe["candidate"], probe["guarded"]
                count = groups.setdefault(probe["kind"], Counter())
                count.update(
                    queries=1, **{name: candidate[name] for name in counter_names},
                    expired_citations=bool(candidate["expired_citation_ids"]),
                    tainted_citations=bool(candidate["unsupported_write_citation_ids"]),
                    guarded_supported=guarded["supported"], reader_vetoes=probe["reader_veto"],
                )
        by_query[label] = groups
    return {"arms": arms, "by_query": by_query, "paired_vs_no_fault_veto": _paired(rows)}


def experiment(output, *, roots_per_family=3):
    if roots_per_family < 1:
        raise ValueError("At least one root per family and seed is needed")
    trials = [(seed, fixture(seed, family, index)) for seed in SEEDS
              for family in FAMILIES for index in range(roots_per_family)]
    output.mkdir(parents=True, exist_ok=False)
    files = (
        "research/tesis_memory_faults_v3.py", "research/tesis_extended_env.py",
        "research/tesis_shared_budget_env.py", "research/tesis_interactive_env.py",
        "research/tesis_bank.py", "research/src/readers/symbolic.py", "research/src/scoring/evidence.py",
        "research/src/policy/rule_controller.py", "research/src/policy/dialogue.py", "research/src/scoring/conditions.py",
        "research/src/memory/executor.py", "research/src/memory/store.py", "memory/store.py",
        "research/src/env/episode.py", "research/src/env/user_simulator.py", "research/src/artifacts.py",
    )
    project = Path(__file__).resolve().parents[1]
    write_json(output / "PLAN.json", {
        "generation_seeds": SEEDS, "roots_per_family_per_seed": roots_per_family, "families": FAMILIES,
        "writer_expiry_extension_days": EXTRA_DAYS, "arms": [_arm_label(*args) for args in ARM_FACTORS],
        "fixed_action_paths": {family: fixed_path({"scenario_family": family}) for family in FAMILIES},
        "read_guard": "reject expired evidence IDs from candidate response; fail closed as DEFER",
        "reader_scope": "future retrieval after SQLite reopen",
        "source_hashes": {name: file_hash(project / name) for name in files},
        "fixture_hash": digest(trials),
    })
    rows = []
    for seed, story in trials:
        for writer, reader, verifier in ARM_FACTORS:
            trace = replay(story, writer_fault=writer, reader_fault=reader, verifier=verifier)
            rows.append({
                "root_id": story["root_id"], "world_id": story["world_id"], "generation_seed": seed,
                "family": story["scenario_family"], "coincident_stale_value": story["coincident_stale_value"],
                "arm": _arm_label(writer, reader, verifier), "writer_fault": writer,
                "reader_fault": reader, "writer_verifier": verifier,
                "actions": list(fixed_path(story)), "metrics": _metrics(trace), "trace": trace,
            })
    _audit(rows)
    result = {"roots": len(trials), "replays": len(rows), "questions_and_closures_fixed": True,
              "generation_seeds": SEEDS, **summarize(rows)}
    write_json(output / "RESULTS.json", result)
    write_json(output / "TRACES.json", rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roots-per-family", type=int, default=3)
    args = parser.parse_args()
    experiment(args.output, roots_per_family=args.roots_per_family)


if __name__ == "__main__":
    main()
