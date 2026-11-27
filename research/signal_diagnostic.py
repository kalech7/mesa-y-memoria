"""Exhaustive development action diagnosis with frozen B1; never trains a model.

python -m research.signal_diagnostic --output research/runs/diagnostico_senal_50_001
The historical closure, its code and all test banks remain untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import csv
import json
from pathlib import Path
import re
from statistics import mean, pstdev
import tempfile
import time

from .src.artifacts import ROOT, canonical, digest, file_hash, now, write_json
from .src.closure_protocol import source_hashes
from .src.closure_environment import run_environment_checks
from .src.env.episode import EpisodeEnvironment
from .src.env.user_simulator import QUESTIONS, UserSimulator
from .src.evaluation.runner import run_episode, select_valid
from .src.policy.llm_controller import LLMController, PHASE_LIMITS
from .src.policy.rule_controller import RuleController, evidence_model
from .src.readers.symbolic import SymbolicReader
from .src.scoring.evidence import Evaluator

DEFAULT_CLOSURE = ROOT / "runs/closure_v4_20261024_003"
ACTIONS = ("PROCEED", *("ASK:" + slot for slot in sorted(QUESTIONS)))
TOLERANCE = 1e-9


def options(action):
    if action == "PROCEED":
        return {"method": "C-NEVER"}
    if action not in ACTIONS:
        raise ValueError("Illegal action: " + action)
    return {"intervention": "I2", "forced_slot": action.split(":", 1)[1], "method": "FORCED-ASK"}


def action_id(action):
    if not select_valid(action):
        return "INVALID"
    return "PROCEED" if action["action"] == "PROCEED" else "ASK:" + action["slot"]


def load_development(closure):
    closure = Path(closure)
    manifest = json.loads((closure / "DEVELOPMENT_MANIFEST_V4.json").read_text())
    path = closure / "DEVELOPMENT_V4.jsonl"
    if file_hash(path) != manifest["development_bank_sha256"]:
        raise ValueError("Fixed development bank changed")
    episodes = [json.loads(line) for line in path.open() if line.strip()]
    if len({e["root_id"] for e in episodes}) != 50 or any(e["split"] not in {"train", "validation", "dev_smoke"} for e in episodes):
        raise ValueError("Exactly 50 fixed non-test development roots required")
    return episodes, path


def load_versioned_development(bank):
    """A new diagnostic bank must be frozen, development-only and isolated."""
    from .src.data.development_v2 import load_historical
    from .src.data.semantic_partition_v2 import assert_semantic_split_isolation_v2
    path = Path(bank).resolve()
    manifest = json.loads(path.with_name("manifest.json").read_text())
    if manifest.get("data_role") != "prospective_development_only" or file_hash(path) != manifest.get("bank_sha256"):
        raise ValueError("A frozen prospective development bank is required")
    episodes = [json.loads(line) for line in path.open() if line.strip()]
    if len({e["root_id"] for e in episodes}) != 50 or any(e["split"] != "validation" for e in episodes):
        raise ValueError("Exactly 50 validation-only development roots required")
    assert_semantic_split_isolation_v2({"validation": episodes}, historical_splits=load_historical(ROOT / "data/bank_1_1"))
    return episodes, path


def enumerate_observations(episodes):
    """Use the real tool transitions before any actor call, without consulting gold."""
    cases, preconditions = [], []
    builder = LLMController(object())
    with tempfile.TemporaryDirectory(prefix="signal_50_preflight_") as work:
        for index, episode in enumerate(episodes):
            case_id = digest([episode[k] for k in ("root_id", "latent_world_id", "realization_id")])[:24]
            prior = None
            observed = {}
            for number, action in enumerate(ACTIONS):
                env = EpisodeEnvironment(episode, Path(work) / f"{index}_{number}.sqlite")
                try:
                    select = env.observe("select")
                    prior = select
                    if action != "PROCEED":
                        env.ask(action.split(":", 1)[1])
                    observed[action] = {phase: env.observe(phase, "O2", "I2" if action != "PROCEED" else None)
                                        for phase in ("respond", "write")}
                finally:
                    env.dispose()
            select_prompt = builder.prompt("select", prior)
            state_id = digest({"select_prompt": select_prompt})
            cases.append({"episode": episode, "case_id": case_id, "state_id": state_id,
                          "select_observation": prior, "select_prompt": select_prompt,
                          "observations": observed})
            # Metadata, truth and future query never enter the actor's prefix.
            altered = deepcopy(episode)
            altered.update(root_id="PRIVATE_ROOT_SENTINEL", latent_world_id="PRIVATE_WORLD_SENTINEL", scenario_family="PRIVATE_FAMILY")
            altered["truth"] = {"secret": "PRIVATE_GOLD_SENTINEL"}
            altered["future_query"] = {"secret": "PRIVATE_FUTURE_SENTINEL"}
            env = EpisodeEnvironment(altered, Path(work) / f"metadata_{index}.sqlite")
            try:
                assert builder.prompt("select", env.observe("select")) == select_prompt
            finally:
                env.dispose()
            preconditions.append(case_id)
    return cases


def audit_bank(cases):
    """Check this entire bank, not just the generic minimal fixtures."""
    checks, errors = [], []
    state_groups = defaultdict(list)
    for case in cases:
        episode, prior = case["episode"], case["select_observation"]
        state_groups[case["state_id"]].append(case)
        before = evidence_model(prior)
        for slot in sorted(QUESTIONS):
            reply = UserSimulator(episode["replies"], root_id=episode["root_id"]).reply(slot)
            reply_text = reply["text"].casefold()
            actual_scope = episode["truth"].get("scope")
            temporal_kind = ("session" if actual_scope and actual_scope["kind"] == "session" else
                             "interval" if actual_scope and actual_scope.get("valid_until_exclusive") else
                             "ongoing" if actual_scope and actual_scope["kind"] in {"ongoing", "contextual"} else
                             actual_scope["kind"] if actual_scope else None)
            explicit_kind = "session" if re.search(r"solo (?:para|por) hoy|solo esta ocasión", reply_text) else "ongoing" if "habitual" in reply_text else None
            # Inspect the reply itself as well: the existing evidence parser's
            # precedence could otherwise mask a contradictory redundant reply.
            if explicit_kind and actual_scope and explicit_kind != temporal_kind:
                errors.append({"case_id": case["case_id"], "slot": slot, "error": "reply text contradicts latent duration", "reply": reply})
            explicit_date = re.search(r"hasta el (\d{4}-\d{2}-\d{2}) inclusive", reply_text)
            if explicit_date and actual_scope:
                from datetime import date, timedelta
                boundary = (date.fromisoformat(explicit_date.group(1)) + timedelta(days=1)).isoformat()
                if boundary != actual_scope.get("valid_until_exclusive"):
                    errors.append({"case_id": case["case_id"], "slot": slot, "error": "reply end boundary contradicts latent duration"})
            obs = case["observations"]["ASK:" + slot]["write"]
            delivered = obs["user_messages"][-1]
            if any(delivered.get(key) != reply.get(key) for key in ("turn_id", "text")):
                errors.append({"case_id": case["case_id"], "slot": slot, "error": "delivery mismatch"})
            after = evidence_model(obs)
            original, changed = before["new_record"], after["new_record"]
            if original and changed and original["scope"]["kind"] != "unknown" and original["scope"] != changed["scope"]:
                errors.append({"case_id": case["case_id"], "slot": slot, "error": "redundant reply changed known scope"})
            if changed and episode["truth"].get("scope") and changed["scope"]["kind"] != "unknown":
                if changed["scope"] != episode["truth"]["scope"]:
                    errors.append({"case_id": case["case_id"], "slot": slot, "error": "revealed scope contradicts synthetic truth"})
            if slot == "CURRENT_CONTEXT" and after["user_context"] is not None:
                truth_context = episode["truth"].get("current_context")
                if truth_context is not None and after["user_context"] != truth_context:
                    errors.append({"case_id": case["case_id"], "slot": slot, "error": "revealed context contradicts synthetic truth"})
            if set(obs) != {"memory", "user_messages", "query", "reuse_regime"}:
                errors.append({"case_id": case["case_id"], "error": "unexpected actor keys"})
        for action, phases in case["observations"].items():
            if canonical(phases["respond"]) != canonical(phases["write"]):
                errors.append({"case_id": case["case_id"], "action": action, "error": "I2 phase evidence differs"})
    for state_id, items in state_groups.items():
        if len({item["select_prompt"] for item in items}) != 1:
            errors.append({"state_id": state_id, "error": "compatible-world prefix mismatch"})
    checks.append({"id": "BANK-INFO-SLOTS", "status": "passed" if not errors else "failed",
                   "cases_checked": len(cases), "legal_actions": list(ACTIONS), "errors": errors})
    return {"status": "passed" if not errors else "failed", "checks": checks,
            "observable_states": len(state_groups), "interpretation": "Gold is used only for auditing responses, never for constructing actor inputs or legal slots"}


class GreedyCache:
    """Exact frozen-policy input cache. Every physical environment still executes."""
    def __init__(self, backend, namespace, directory, allowed):
        self.original, self.namespace = backend, namespace
        self.directory, self.allowed = Path(directory), allowed
        self.directory.mkdir()
        self.cache, self.calls, self.hits, self.requests = {}, 0, 0, 0

    def __getattr__(self, name):
        return getattr(self.original, name)

    def generate(self, prompt, sample=False):
        if sample:
            raise ValueError("This diagnosis uses greedy decoding; no sampling or training")
        request = digest({"prompt": prompt, "max_new_tokens": self.config.max_new_tokens})
        if request not in self.allowed:
            raise ValueError("Unplanned actor prefix")
        key = digest({"namespace": self.namespace, "request": request, "seed": 101, "sample": False})
        self.requests += 1
        hit = key in self.cache
        if hit:
            self.hits += 1
        else:
            self.calls += 1
            generated = self.original.generate(prompt, sample=False)
            self.cache[key] = deepcopy(generated)
            write_json(self.directory / (key + ".json"), {"key": key, "namespace": self.namespace,
                "prompt": prompt, "input_token_ids_sha256": digest(generated.prompt_ids),
                "generation": asdict(generated), "inference_batch_size": 1, "seed": 101})
        result = deepcopy(self.cache[key])
        result.metadata = {**(result.metadata or {}), "frozen_cache_key": key, "cache_hit": hit}
        return result


class ReferenceWriter(RuleController):
    def __init__(self, response):
        self.response = deepcopy(response)

    def respond(self, observation):
        return deepcopy(self.response)


def independent_read(memory, query):
    """Audit-only structured reference consumer; uses no latent world or labels."""
    definite, uncertain = [], []
    for record in memory:
        if record["attribute"] != query["attribute"]:
            continue
        scope = record["scope"]
        if scope.get("valid_from") and query["date"] < scope["valid_from"]:
            continue
        if scope.get("valid_until_exclusive") and query["date"] >= scope["valid_until_exclusive"]:
            continue
        if scope["kind"] == "session" and scope["session_id"] != query.get("session_id"):
            continue
        unknown = scope["kind"] == "unknown" or record.get("status") == "unresolved"
        if scope.get("context") is not None:
            if query.get("context") is None:
                unknown = True
            else:
                expected, actual = scope["context"], query["context"]
                matches = all(actual.get(k) == v for k, v in expected.items()) if isinstance(expected, dict) and isinstance(actual, dict) else expected == actual
                if not matches:
                    continue
        (uncertain if unknown else definite).append(record)
    removed = {mid for r in definite for mid in r.get("supersedes", [])}
    definite = [r for r in definite if r["memory_id"] not in removed]
    uncertain = [r for r in uncertain if r["memory_id"] not in removed]
    rank = {"ongoing": 0, "contextual": 1, "interval": 2, "session": 3}
    if definite:
        def specificity(record):
            scope = record["scope"]
            if scope["kind"] == "session":
                return 4
            if scope.get("context") is not None and scope.get("valid_until_exclusive"):
                return 3
            return rank[scope["kind"]]
        top = max(specificity(r) for r in definite)
        definite = [r for r in definite if specificity(r) == top]
    values = {r["value"] for r in definite}
    if len(values) == 1 and all(r["value"] in values for r in uncertain):
        return {"mode": "ANSWER", "value": next(iter(values)), "branches": [],
                "evidence_ids": [r["memory_id"] for r in definite]}
    return {"mode": "DEFER", "value": None, "branches": [], "evidence_ids": []}


def enrich(row, case, action, writer, natural_action):
    row.update(diagnostic_case_id=case["case_id"], observable_state_id=case["state_id"],
               action_id=action, writer=writer, natural_action_id=natural_action,
               diagnostic_version="signal_50_v1", forced_intervention=action != "NATURAL",
               legal_actions=list(ACTIONS), J_primary=row["metrics"]["C2"] * row["metrics"]["FJS"])
    actual_action = natural_action if action == "NATURAL" else action
    row["question_delivered"] = QUESTIONS.get(actual_action.split(":", 1)[1]) if actual_action.startswith("ASK:") else None
    trace = row["trace"]
    score = {"current": Evaluator().score_current(trace), "future": Evaluator().score_future(trace),
             "memory": Evaluator().score_memory(trace)}
    row["scores"] = score
    r2, r3 = score["current"]["utility"], score["future"]["utility"]
    penalty = -.1 * row["metrics"]["questions"] - .5 * row["metrics"]["unsupported_operations"] - .2 * row["metrics"]["invalid_outputs"]
    assert abs(row["rewards_by_component"]["R_long"] - (r2 + r3 + penalty)) < TOLERANCE
    assert abs(row["rewards_by_component"]["R_short"] - (r2 + penalty)) < TOLERANCE
    assert set(trace["reuse_observation"]) == {"memory", "query"}
    assert trace["reuse_observation"]["memory"] == trace["final_memory"]
    candidate = independent_read(trace["final_memory"], trace["episode"]["future_query"])
    independent_trace = {**trace, "future_response": candidate}
    row["reader_diagnostic"] = {"reference_response": candidate,
        "reference_score": Evaluator().score_future(independent_trace),
        "fixed_reader_response": trace["future_response"],
        "interpretation": "Independent structured consumer on identical persisted memory; no LLM-reader claim"}
    row["reference_slot_from_observable_evidence"] = evidence_model(case["select_observation"])["missing_slot"]
    return row


def compact(row):
    reward = row["rewards_by_component"]
    return {"case_id": row["diagnostic_case_id"], "root_id": row["root_id"],
            "latent_world_id": row["latent_world_id"], "realization_id": row["realization_id"],
            "observable_state_id": row["observable_state_id"], "family": row["scenario_family"],
            "action": row["action_id"], "writer": row["writer"], "B1_selected_action": row["natural_action_id"],
            "C2": row["metrics"]["C2"], "FJS": row["metrics"]["FJS"], "J_primary": row["J_primary"],
            "R_short": reward["R_short"], "R_long": reward["R_long"], **reward["components"],
            "invalid_output_count": row["metrics"]["invalid_outputs"],
            "unsupported_write_count": row["metrics"]["unsupported_operations"],
            "independent_reader_FJS": row["reader_diagnostic"]["reference_score"]["success"],
            "response_informative": row["metrics"].get("response_informative"),
            "trace_line": row["trace_line"], "trace_sha256": row["trace_sha256"]}


def hierarchical_mean(rows, metric):
    families = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for row in rows:
        families[row["family"]][row["root_id"]][row["latent_world_id"]][row["realization_id"]].append(row[metric])
    return mean(mean(mean(mean(mean(values) for values in realizations.values()) for realizations in worlds.values())
                     for worlds in roots.values()) for roots in families.values())


def analyze(rows):
    forced = [r for r in rows if r["writer"] == "B1" and r["action"] in ACTIONS]
    natural = {r["case_id"]: r for r in rows if r["action"] == "NATURAL"}
    references = {(r["case_id"], r["action"]): r for r in rows if r["writer"] == "reference"}
    by_case, by_state = defaultdict(list), defaultdict(list)
    for row in forced:
        by_case[row["case_id"]].append(row)
        by_state[row["observable_state_id"]].append(row)
    state_results = []
    for state, items in by_state.items():
        by_action = defaultdict(list)
        for row in items:
            by_action[row["action"]].append(row)
        if set(by_action) != set(ACTIONS):
            raise ValueError("Incomplete action enumeration")
        # Uniform synthetic compatible-world weights, with realizations averaged
        # inside each world. Never choose a different action after seeing z.
        values = {}
        for action, action_rows in by_action.items():
            worlds = defaultdict(list)
            for row in action_rows:
                worlds[row["latent_world_id"]].append(row)
            values[action] = {metric: mean(mean(r[metric] for r in rs) for rs in worlds.values())
                              for metric in ("R_long", "R_short", "C2", "FJS", "J_primary")}
        chosen = {natural[r["case_id"]]["B1_selected_action"] for r in items}
        if len(chosen) != 1:
            raise ValueError("Frozen deterministic selector changed across identical visible states")
        chosen = next(iter(chosen))
        result = {"observable_state_id": state, "root_id": items[0]["root_id"], "family": items[0]["family"],
                  "cases": len(by_action["PROCEED"]), "B1_action": chosen, "action_values": values,
                  "case_ids": [r["case_id"] for r in by_action["PROCEED"]],
                  "natural_trace_lines": [natural[r["case_id"]]["trace_line"] for r in by_action["PROCEED"]]}
        for metric in ("R_long", "R_short", "J_primary"):
            maximum = max(v[metric] for v in values.values())
            best = next(a for a in ACTIONS if values[a][metric] >= maximum - TOLERANCE)
            result["best_" + metric] = best
            result["range_" + metric] = maximum - min(v[metric] for v in values.values())
            result["regret_" + metric] = maximum - values[chosen][metric] if chosen in values else None
        task_values = {(round(v["C2"], 9), round(v["FJS"], 9), round(v["J_primary"], 9)) for v in values.values()}
        result["task_success_varies"] = len(task_values) > 1
        result["penalty_only_variation"] = not result["task_success_varies"] and result["range_R_long"] > TOLERANCE
        result["reward_collision"] = any(abs(values[a]["R_long"] - values[b]["R_long"]) <= TOLERANCE
                and any(abs(values[a][m] - values[b][m]) > TOLERANCE for m in ("C2", "FJS", "J_primary"))
                for i, a in enumerate(ACTIONS) for b in ACTIONS[i+1:])
        state_results.append(result)
    classes = defaultdict(list)
    for state in state_results:
        if state["regret_R_long"] is not None and state["regret_R_long"] > TOLERANCE:
            classes["selector_longitudinal_suboptimal"].append(state)
        if state["regret_R_short"] is not None and state["regret_R_short"] > TOLERANCE:
            classes["selector_immediate_suboptimal"].append(state)
        if state["regret_R_long"] is not None and state["regret_R_long"] <= TOLERANCE:
            classes["B1_saturated_for_longitudinal_selection"].append(state)
        if state["penalty_only_variation"]:
            classes["task_saturated_penalty_only_variation"].append(state)
        if state["reward_collision"]:
            classes["reward_collision_not_automatically_a_defect"].append(state)
    for row in forced:
        reference = references[(row["case_id"], row["action"])]
        evidence = {"case_id": row["case_id"], "action": row["action"], "trace_line": row["trace_line"],
                    "B1": row, "reference": reference}
        if reference["FJS"] > row["FJS"] or reference["unsupported_write_count"] < row["unsupported_write_count"]:
            classes["writing_loss_same_delivered_evidence"].append(evidence)
        if row["independent_reader_FJS"] > row["FJS"]:
            classes["reading_loss_same_persisted_memory"].append(row)
        if row["invalid_output_count"]:
            classes["format_error"].append(row)
    natural_rows = list(natural.values())
    natural_writer_losses = [r for r in natural_rows if (r["case_id"], r["B1_selected_action"]) in references
        and references[(r["case_id"], r["B1_selected_action"])]["FJS"] > r["FJS"]]
    natural_reader_losses = [r for r in natural_rows if r["independent_reader_FJS"] > r["FJS"]]
    primary = {metric: hierarchical_mean(natural_rows, metric) for metric in ("C2", "FJS", "J_primary", "R_long", "R_short")}
    return {"status": "executed", "root_count": len({r["root_id"] for r in forced}), "cases": len(by_case),
            "observable_states": len(state_results), "legal_actions": list(ACTIONS),
            "states_with_distinct_long_returns": sum(s["range_R_long"] > TOLERANCE for s in state_results),
            "states_with_distinct_short_returns": sum(s["range_R_short"] > TOLERANCE for s in state_results),
            "states_with_distinct_task_success": sum(s["task_success_varies"] for s in state_results),
            "class_counts": {name: len(items) for name, items in classes.items()},
            "natural_writer_loss_cases": natural_writer_losses, "natural_reader_loss_cases": natural_reader_losses,
            "states": state_results, "classes": dict(classes), "natural_B1_hierarchical": primary,
            "conditional_world_weighting": "uniform worlds compatible with identical actor prompt; average before choosing action",
            "inference_scope": "Descriptive on fixed synthetic development, conditional on its grammar and seed 101; no human or held-out-test claims"}


def pilot_audit(closure):
    result = {"groups": [], "sources": []}
    for path in sorted((Path(closure) / "pilot").glob("*/training/group_diagnostics.jsonl")):
        for number, line in enumerate(path.open(), 1):
            row = json.loads(line)
            result["groups"].append({"arm": path.parents[1].name, "line": number, **row})
        result["sources"].append({"path": str(path.resolve()), "sha256": file_hash(path)})
    result.update(group_count=len(result["groups"]),
        zero_task_variance_groups=sum(pstdev(g["task_returns"]) == 0 for g in result["groups"]),
        zero_total_variance_groups=sum(pstdev(g["returns"]) == 0 for g in result["groups"]),
        single_action_groups=sum(len({canonical(a) for a in g["actions"]}) == 1 for g in result["groups"]),
        unique_observable_token_states=len({v for g in result["groups"] for v in g["observable_prompt_token_sha256"]}),
        interpretation="Historical training trajectories only; no test read. New development regret cannot retroactively identify unvisited actions in different pilot roots.")
    return result


def write_reports(output, analysis, cases, calls):
    classes = analysis["classes"]
    case_map = {case["case_id"]: case for case in cases}
    lines = ["# Fallos y limitaciones clasificados", "", analysis["inference_scope"], "",
             "Las categorías pueden solaparse. Los slots se enumeraron desde la interfaz pública; no se eligieron a partir del mundo oculto.", ""]
    for name in ("selector_longitudinal_suboptimal", "selector_immediate_suboptimal", "B1_saturated_for_longitudinal_selection",
                 "task_saturated_penalty_only_variation", "reward_collision_not_automatically_a_defect",
                 "writing_loss_same_delivered_evidence", "reading_loss_same_persisted_memory", "format_error"):
        items = classes.get(name, [])
        lines += [f"## {name}: {len(items)}", ""]
        if not items:
            lines += ["No se observaron ejemplos en este diagnóstico; esto no acredita ausencia universal.", ""]
        else:
            for item in items[:3]:
                case_id = item.get("case_id", item.get("case_ids", [None])[0])
                if case_id in case_map:
                    episode = case_map[case_id]["episode"]
                    lines += [f"Caso `{case_id}`; raíz `{episode['root_id']}`; mensaje visible: "
                              + " / ".join(m["text"] for m in episode["user_messages"]), ""]
                lines += ["```json", json.dumps(item, ensure_ascii=False, indent=2), "```", ""]
    (output / "fallos_clasificados.md").write_text("\n".join(lines), encoding="utf-8")
    gaps = classes.get("selector_longitudinal_suboptimal", [])
    writer_losses = classes.get("writing_loss_same_delivered_evidence", [])
    reader_losses = classes.get("reading_loss_same_persisted_memory", [])
    if gaps:
        decision = "investigar cobertura de selección antes de continuar GRPO"
        next_experiment = "Medir sin actualizaciones la cobertura y probabilidad de las acciones mejores identificadas, en los mismos estados de desarrollo. Si el muestreo no las visita, comparar una política discreta o exploración declarada, manteniendo B1 continuador y lector congelados y un control SFT con los mismos pares. No tratar las acciones forzadas como muestras on-policy de GRPO."
    elif analysis["natural_writer_loss_cases"] or analysis["natural_reader_loss_cases"]:
        decision = "modificar o estudiar el componente que pierde información; no prolongar RL del selector"
        next_experiment = "Repetir solo los casos concretos con sustitución controlada del escritor o lector y la misma evidencia. Si el daño aparece únicamente tras preguntas innecesarias que B1 ya evita, conservar esa restricción de alcance."
    else:
        decision = "cerrar la rama de selección longitudinal en este banco de desarrollo"
        next_experiment = "No añadir pasos GRPO sobre estos estados saturados. Para estudiar aprendizaje, preparar otro banco de desarrollo con errores alcanzables del selector y validación semántica antes de bloquear un test nuevo; conservar este resultado sin cambiar recompensas ni elegir casos de test."
    pilot = analysis["pilot_audit"]
    text = ["# Decisión sobre RL", "", f"**Recomendación: {decision}.**", "",
        f"Se evaluaron {analysis['root_count']} raíces, {analysis['cases']} combinaciones raíz/mundo/realización y {analysis['observable_states']} estados observables con las cinco acciones legales.", "",
        f"Estados con distintos R_long: {analysis['states_with_distinct_long_returns']}; con distinto éxito de tarea: {analysis['states_with_distinct_task_success']}; B1 inferior a otra acción bajo R_long: {len(gaps)}. La elección de la mejor acción promedia primero los mundos compatibles, sin revelar el mundo al selector.", "",
        f"El piloto anterior tuvo {pilot['group_count']} grupos: {pilot['zero_total_variance_groups']} con retorno total idéntico, {pilot['zero_task_variance_groups']} con tarea idéntica y {pilot['single_action_groups']} con una sola acción/slot observado. Representaron {pilot['unique_observable_token_states']} estados de tokens distintos. Esto explica algebraicamente las ventajas cero, pero la falta de diversidad por sí sola no demuestra que exista una acción mejor.", "",
        f"B1 presenta {len(classes.get('selector_immediate_suboptimal', []))} estados subóptimos bajo R_short. El horizonte inmediato puede preferir ahorrar una pregunta que solo mejora el futuro; no se debe confundir ese objetivo con éxito longitudinal.", "",
        f"Contrastes de escritura: {len(writer_losses)} pérdidas bajo idéntica evidencia. Contrastes de lectura: {len(reader_losses)} pérdidas frente al consumidor estructurado independiente sobre la misma memoria. Son conteos por caso y acción, no usuarios independientes.", "",
        "## Siguiente experimento mínimo", "", next_experiment, "",
        "## Alcance", "", "No hubo entrenamiento, modificación de recompensas, selección de checkpoints ni apertura de test. B1-101, memoria inicial, mundo y consultas se mantuvieron fijos en cada par. El lector primario fue el simbólico histórico; este diagnóstico no acredita la competencia del lector LLM secundario. Los retornos obtenidos aquí son consecuencias greedy con componentes congelados; no son valores esperados integrados sobre toda la generación estocástica de GRPO.", "",
        f"Las pérdidas de escritura en la acción natural de B1 fueron {len(analysis['natural_writer_loss_cases'])}; las de lectura, {len(analysis['natural_reader_loss_cases'])}. Los errores exclusivos de preguntas forzadas que B1 evita no se atribuyen como limitación de su política natural.", "",
        "La condición de referencia solo escribe evidencia realmente recibida. No revela todas las variables latentes ni rescata una pregunta que no se entregó. Los gold privados se conservan únicamente en trazas de auditoría y scoring.", "",
        f"Generaciones nuevas: {calls['actual_generations']}; reutilizaciones exactas de generación greedy: {calls['cache_hits']}. No se reutilizaron rollouts históricos como muestras on-policy.", ""]
    (output / "decision_rl.md").write_text("\n".join(text), encoding="utf-8")


def run(closure, output, *, device="cuda", backend=None, bank=None, checkpoint=None):
    closure, output = Path(closure).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new directory; existing evidence is preserved")
    if output == closure or output.is_relative_to(closure):
        raise ValueError("Preserve the historical closure")
    output.mkdir(parents=True, exist_ok=True)
    episodes, bank_path = load_versioned_development(bank) if bank else load_development(closure)
    source = {**source_hashes(), str(Path(__file__).resolve()): file_hash(__file__)}
    started = time.monotonic()
    checks = run_environment_checks()
    write_json(output / "auditoria_contratos.json", checks)
    if checks["status"] != "passed":
        raise RuntimeError("Environment contracts failed before action comparison")
    cases = enumerate_observations(episodes)
    audit = audit_bank(cases)
    write_json(output / "auditoria_banco_50.json", audit)
    if audit["status"] != "passed":
        raise RuntimeError("Bank contracts failed before actor calls")
    builder = LLMController(object())
    allowed = {digest({"prompt": case["select_prompt"], "max_new_tokens": PHASE_LIMITS["select"]}) for case in cases}
    for case in cases:
        for phases in case["observations"].values():
            for phase, observation in phases.items():
                allowed.add(digest({"prompt": builder.prompt(phase, observation), "max_new_tokens": PHASE_LIMITS[phase]}))
    checkpoint = Path(checkpoint) if checkpoint else Path(json.loads((Path(json.loads((closure / "MANIFIESTO_V4.json").read_text())["parent_run"]) / "sft/101/B1/training.json").read_text())["selected_checkpoint"])
    from llm_local_001.model import artifact_hashes
    checkpoint_hashes = artifact_hashes(checkpoint)
    expected = json.loads((checkpoint / "local_checkpoint.json").read_text())["artifacts_sha256"]
    if checkpoint_hashes != expected:
        raise ValueError("B1 checkpoint artifacts changed")
    plan = {"created_at": now(), "training": False, "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint_hashes), "bank": str(bank_path), "bank_sha256": file_hash(bank_path),
        "roots": 50, "episodes": len(cases), "observable_states": audit["observable_states"],
        "actions": list(ACTIONS), "writers": ["B1", "reference"], "include_natural_B1": True,
        "maximum_actual_generations": len(allowed), "maximum_unshared_generation_requests": len(cases) * 14,
        "generation": {"sample": False, "seed": 101, "batch_size": 1, "phase_limits": PHASE_LIMITS},
        "reader": "fixed_symbolic", "reward_changed": False, "nonresponse_rate": 0,
        "world_weights": "uniform compatible worlds before selecting action", "tie_order": list(ACTIONS),
        "numeric_tolerance": TOLERANCE, "source_files_sha256": source,
        "scope": "explicit new exhaustive diagnostic, separate from the completed 600-call closure; no test data"}
    plan["bank_role"] = "prospective_development_only" if bank else "historical_development"
    write_json(output / "plan.json", plan)
    import shutil
    for name in source:
        source_path = Path(name) if Path(name).is_absolute() else ROOT.parent / name
        relative = source_path.relative_to(ROOT.parent)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
    print(canonical({"stage": "preflight_passed", "roots": 50, "episodes": len(cases), "unique_generation_upper_bound": len(allowed)}), flush=True)
    if backend is None:
        from .src.policy.llm_controller import SpecLocalPolicy
        backend = SpecLocalPolicy.from_checkpoint(checkpoint, trainable=False, device=device)
    if any(p.requires_grad for p in backend.model.parameters()):
        raise ValueError("B1 must be frozen")
    backend.model.eval()
    from .src.training.sft import parameter_hash
    weights_before = parameter_hash(backend)
    counted = GreedyCache(backend, digest({"checkpoint": checkpoint_hashes, "generation": plan["generation"]}), output / "generation_cache", allowed)
    (output / "states").mkdir()
    compact_rows, line_count = [], 0
    path = output / "diagnostico_50.jsonl"
    with path.open("x", encoding="utf-8") as stream:
        def emit(row, case, action, writer, natural_action):
            nonlocal line_count
            line_count += 1
            row = enrich(row, case, action, writer, natural_action)
            payload = canonical(row)
            stream.write(payload + "\n")
            stream.flush()
            row["trace_line"] = line_count
            import hashlib
            row["trace_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
            compact_rows.append(compact(row))
            return row

        for index, case in enumerate(cases):
            episode, case_id = case["episode"], case["case_id"]
            selector = LLMController(counted)
            try:
                selected = selector.select(deepcopy(case["select_observation"]))
            except ValueError:
                selected = {}
            natural_action = action_id(selected)
            primary_rows = {}
            for number, action in enumerate(ACTIONS):
                branch = run_episode(episode, LLMController(counted), rollout_seed=101,
                    db_path=output / "states" / f"{case_id}_{number}_b1.sqlite", **options(action))
                assert canonical(branch["trace"]["observations"]["write"]) == canonical(case["observations"][action]["write"])
                primary_rows[action] = emit(branch, case, action, "B1", natural_action)
                reference = run_episode(episode, ReferenceWriter(branch["trace"]["response"]), rollout_seed=101,
                    db_path=output / "states" / f"{case_id}_{number}_reference.sqlite", **options(action))
                assert canonical(branch["trace"]["observations"]["write"]) == canonical(reference["trace"]["observations"]["write"])
                assert branch["trace"]["response"] == reference["trace"]["response"]
                reference["B1_response_source"] = {"trace_line": primary_rows[action]["trace_line"],
                    "trace_sha256": primary_rows[action]["trace_sha256"]}
                emit(reference, case, action, "reference", natural_action)
            natural = run_episode(episode, LLMController(counted), rollout_seed=101, method="B1-frozen-natural",
                db_path=output / "states" / f"{case_id}_natural.sqlite")
            assert action_id(natural["parsed_actions"]["select"]) == natural_action
            if natural_action in primary_rows:
                assert natural["rewards_by_component"] == primary_rows[natural_action]["rewards_by_component"]
                assert natural["metrics"] == primary_rows[natural_action]["metrics"]
            emit(natural, case, "NATURAL", "B1", natural_action)
            write_json(output / "progress.json", {"status": "running", "cases_completed": index + 1,
                "cases_scheduled": len(cases), "trajectories": line_count, "new_generations": counted.calls,
                "cache_hits": counted.hits, "elapsed_seconds": time.monotonic() - started})
            print(canonical({"case": index+1, "total_cases": len(cases), "root_id": episode["root_id"],
                "family": episode["scenario_family"], "new_generations": counted.calls,
                "cache_hits": counted.hits, "elapsed_seconds": round(time.monotonic()-started,2)}), flush=True)
    weights_after = parameter_hash(backend)
    if weights_after != weights_before or any(p.requires_grad for p in backend.model.parameters()):
        raise RuntimeError("Frozen B1 weights changed")
    if source != {**source_hashes(), str(Path(__file__).resolve()): file_hash(__file__)}:
        raise RuntimeError("Diagnostic source changed during execution")
    calls = {"actual_generations": counted.calls, "cache_hits": counted.hits, "requests": counted.requests,
             "predeclared_upper_bound": len(allowed)}
    fields = list(compact_rows[0])
    with (output / "comparacion_acciones.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(compact_rows)
    analysis = analyze(compact_rows)
    analysis["pilot_audit"] = pilot_audit(closure)
    analysis["calls"] = calls
    analysis["weights"] = {"before": weights_before, "after": weights_after, "frozen": True, "optimizer_steps": 0}
    write_json(output / "analisis.json", analysis)
    write_reports(output, analysis, cases, calls)
    write_json(output / "completion.json", {"status": "executed", "completed_at": now(), "training": False,
        "trajectories": line_count, "calls": calls, "elapsed_seconds": time.monotonic()-started,
        "artifacts_sha256": {name: file_hash(output/name) for name in ("diagnostico_50.jsonl", "comparacion_acciones.csv", "fallos_clasificados.md", "decision_rl.md", "analisis.json", "plan.json")}})
    write_json(output / "progress.json", {"status": "executed", "cases_completed": len(cases), "trajectories": line_count, "calls": calls})
    return analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure", default=str(DEFAULT_CLOSURE))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bank", help="Frozen prospective development bank; never a test partition")
    parser.add_argument("--checkpoint", help="Explicit frozen B1 checkpoint")
    args = parser.parse_args(argv)
    try:
        result = run(args.closure, args.output, device=args.device, bank=args.bank, checkpoint=args.checkpoint)
        print(canonical({"status": result["status"], "output": args.output, "class_counts": result["class_counts"]}))
        return 0
    except Exception as error:
        output = Path(args.output)
        if output.is_dir() and (output / "plan.json").exists() and not (output / "completion.json").exists():
            write_json(output / "failure.json", {"status": "failed", "error": repr(error), "partial_outputs_preserved": True})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
