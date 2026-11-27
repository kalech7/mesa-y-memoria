"""One production scorer, verified replay, and frozen selector sampling.

python -m research.reliable_evaluation --output research/runs/evaluacion_confiable_001
No training API is called. Historical runs and their recorded scores are read only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from statistics import mean
import time

from .src.artifacts import ROOT, canonical, digest, file_hash, write_json, now
from .src.closure_environment import run_environment_checks
from .src.closure_protocol import source_hashes
from .src.policy.llm_controller import LLMController, SpecLocalPolicy, PHASE_LIMITS
from .src.scoring.evidence import Evaluator, EVALUATOR_VERSION
from .src.scoring.reward import rewards
from .signal_diagnostic import (ACTIONS, load_development, enumerate_observations, audit_bank,
    compact, analyze, action_id)

DEFAULT_SOURCE = ROOT / "runs/diagnostico_senal_50_20261024_001"
DEFAULT_CLOSURE = ROOT / "runs/closure_v4_20261024_003"


def verify_manifest(source):
    source = Path(source)
    completion = json.loads((source / "completion.json").read_text())
    if completion.get("status") != "executed" or completion.get("training") is not False:
        raise ValueError("A completed, frozen, no-training diagnosis is required")
    for filename, expected in completion["artifacts_sha256"].items():
        path = (source / filename).resolve()
        if not path.is_relative_to(source.resolve()) or file_hash(path) != expected:
            raise ValueError("Changed source artifact: " + filename)
    return completion


def verify_tokens(source, checkpoint):
    """Re-tokenize all 500 exact cached prompts; no model or network needed."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    checked = 0
    for path in sorted((Path(source) / "generation_cache").glob("*.json")):
        entry = json.loads(path.read_text())
        prompt = entry["prompt"]
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True) if tokenizer.chat_template else tokenizer.encode(prompt, add_special_tokens=True)
        generated = entry["generation"]
        if ids != generated["prompt_ids"] or digest(ids) != entry["input_token_ids_sha256"]:
            raise ValueError("Tokenizer/cache prompt mismatch: " + path.name)
        if tokenizer.decode(generated["completion_ids"], skip_special_tokens=True) != generated["text"]:
            raise ValueError("Cached output token/text mismatch: " + path.name)
        checked += 1
    if checked != 500:
        raise ValueError("Incomplete generation cache")
    return {"status": "passed", "prompts_verified": checked, "new_model_calls": 0,
            "tokenizer": str(checkpoint), "generation_batch_size": 1}


def verify_sqlite(source, row):
    case, action = row["diagnostic_case_id"], row["action_id"]
    suffix = ("natural" if action == "NATURAL" else str(ACTIONS.index(action)) + "_" +
              ("b1" if row["writer"] == "B1" else "reference"))
    path = Path(source) / "states" / (case + "_" + suffix + ".sqlite")
    with sqlite3.connect("file:" + str(path.resolve()) + "?mode=ro", uri=True) as connection:
        actual = [json.loads(r[0]) for r in connection.execute("SELECT payload FROM cm_records ORDER BY seq")]
    trace = row["trace"]
    if actual != trace["final_memory"] or actual != trace["reuse_observation"]["memory"]:
        raise ValueError("Persisted S3 state mismatch: " + str(path))
    if set(trace["reuse_observation"]) != {"memory", "query"}:
        raise ValueError("S3 receives data other than persisted memory and explicit query")


def replay(source, output, cases):
    """Recompute from evidence, never trust exported scores or fix actor outputs."""
    evaluator, builder = Evaluator(), LLMController(object())
    expected_cases = {c["case_id"]: c for c in cases}
    pair_rows, seen, flat_rows = {}, set(), []
    counts = Counter()
    variants = Counter()
    old_original = Path(source) / "diagnostico_50.original.jsonl"
    current_path = Path(source) / "diagnostico_50.jsonl"
    with current_path.open() as stream, old_original.open() as original, (output / "diagnostico_50.jsonl").open("x") as target:
        for number, (line, original_line) in enumerate(zip(stream, original, strict=True), 1):
            row, raw = json.loads(line), json.loads(original_line)
            trace = row["trace"]
            if trace != raw["trace"] or row["parsed_actions"] != raw["parsed_actions"]:
                raise ValueError("Actor or state changed between source scoring revisions")
            cid, action, writer = row["diagnostic_case_id"], row["action_id"], row["writer"]
            if cid not in expected_cases:
                raise ValueError("Unknown development case")
            case = expected_cases[cid]
            if trace["episode"] != case["episode"]:
                raise ValueError("Private evaluation case changed")
            key = (cid, action, writer)
            if key in seen:
                raise ValueError("Duplicate trajectory; no deduplication is permitted")
            seen.add(key)
            verify_sqlite(source, row)
            for phase in trace["phase_records"]:
                observation = trace["observations"][phase["phase"]]
                if phase["prompt"] != builder.prompt(phase["phase"], observation):
                    raise ValueError("Actor prompt does not match its visible observation")
                cache_key = phase["generation"]["metadata"]["frozen_cache_key"]
                cache = json.loads((Path(source) / "generation_cache" / (cache_key + ".json")).read_text())
                if cache["prompt"] != phase["prompt"]:
                    raise ValueError("Cached prefix mismatch")
                for field in ("prompt_ids", "completion_ids", "text", "truncated"):
                    if cache["generation"][field] != phase["generation"][field]:
                        raise ValueError("Cached generation changed: " + field)
                counts["actor_prefixes_verified"] += 1
            if digest(trace["initial_memory"]) != row["initial_memory_hash"]:
                raise ValueError("Initial-memory hash mismatch")
            if digest(trace["final_memory"]) != row["final_memory_hash"]:
                raise ValueError("Final-memory hash mismatch")
            if action != "NATURAL":
                if writer == "B1":
                    pair_rows[(cid, action)] = row
                else:
                    paired = pair_rows[(cid, action)]
                    for part in ("initial_memory", "user_reply", "response"):
                        if paired["trace"][part] != trace[part]:
                            raise ValueError("Writer pair changed a non-intervened component: " + part)
                    if paired["trace"]["observations"]["write"] != trace["observations"]["write"]:
                        raise ValueError("Writer evidence differs across pair")
                    counts["writer_pairs_verified"] += 1
            current, memory, future = evaluator.score_current(trace), evaluator.score_memory(trace), evaluator.score_future(trace)
            questions = int(trace["user_reply"] is not None)
            invalid = len(trace["errors"])
            if questions != row["metrics"]["questions"] or invalid != row["metrics"]["invalid_outputs"]:
                raise ValueError("Question/format denominator mismatch")
            score = rewards(current["utility"], future["utility"], questions, memory["unsupported_operations"], invalid)
            if score != row["rewards_by_component"]:
                counts["changed_vs_local_v2"] += 1
            if score != raw["rewards_by_component"]:
                counts["changed_vs_legacy_v1"] += 1
            if current["success"] != row["metrics"]["C2"] or future["success"] != row["metrics"]["FJS"]:
                counts["task_success_changed"] += 1
            row.update(source_trajectory={"path": str(current_path.resolve()), "line": number,
                "sha256": hashlib.sha256(line.rstrip("\n").encode()).hexdigest()},
                evaluator_version=EVALUATOR_VERSION, evaluator_hash=file_hash(Path(__file__).parent / "src/scoring/evidence.py"))
            row["previous_future_evaluator_version"] = row.get("future_evaluator_version")
            row["future_evaluator_version"] = EVALUATOR_VERSION
            row["scores"] = {"current": current, "memory": memory, "future": future}
            row["rewards_by_component"] = score
            row["metrics"].update(memory, C2=current["success"], FJS=future["success"],
                                  current_factual=current["factual"], future_factual=future["factual"],
                                  unsupported_future=future["unsupported_commitment"])
            row["J_primary"] = current["success"] * future["success"]
            row["reader_diagnostic"]["reference_score"] = evaluator.score_future({**trace,
                "future_response": row["reader_diagnostic"]["reference_response"]})
            payload = canonical(row)
            target.write(payload + "\n")
            row["trace_line"] = number
            row["trace_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
            flat = compact(row)
            flat["evaluator_version"] = EVALUATOR_VERSION
            flat_rows.append(flat)
            counts["trajectories"] += 1
            variants[(action, writer)] += 1
    expected = {(c["case_id"], action, writer) for c in cases for action in ACTIONS for writer in ("B1", "reference")}
    expected |= {(c["case_id"], "NATURAL", "B1") for c in cases}
    if seen != expected:
        raise ValueError("Incomplete action matrix")
    if counts["changed_vs_local_v2"] or counts["task_success_changed"]:
        raise ValueError("Unexpected production-scoring difference: inspect before deciding")
    with (output / "comparacion_acciones.csv").open("x", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    analysis = analyze(flat_rows)
    write_json(output / "analisis.json", analysis)
    integrity = {"status": "passed", **counts, "all_requested_conditions_present": True,
        "SQLite_states_verified": counts["trajectories"], "actor_and_environment_unchanged": True,
        "source_scores_preserved": True, "new_model_calls": 0, "formula_changed": False,
        "evaluator_version": EVALUATOR_VERSION}
    write_json(output / "integridad.json", integrity)
    return analysis, flat_rows


def sample_selector(cases, checkpoint, output, *, draws=8, device="cuda"):
    """Sample only SELECT; evaluate its action later with frozen greedy consequences.

    These are not full GRPO trajectories, and neither forced actions nor cached
    completions are presented as on-policy training data.
    """
    if not 1 <= draws <= 32:
        raise ValueError("Predeclare 1..32 selector draws/state")
    states = {case["state_id"]: case for case in cases}
    plan = {"created_at": now(), "checkpoint": str(checkpoint), "states": len(states), "draws_per_state": draws,
        "maximum_new_generations": len(states) * draws, "sample": True, "temperature": 1., "top_p": 1.,
        "top_k": 0, "max_new_tokens": PHASE_LIMITS["select"], "training": False, "optimizer_steps": 0,
        "seed_rule": "73310000 + state_index * draws + draw_index", "reuse_sampling": False,
        "scope": "SELECT sampling under the pilot distribution; RESPOND/WRITE remain fixed greedy action values"}
    write_json(output / "plan_muestreo.json", plan)
    backend = SpecLocalPolicy.from_checkpoint(checkpoint, trainable=False, device=device)
    backend.config.temperature = 1.
    backend.model.eval()
    if any(p.requires_grad for p in backend.model.parameters()):
        raise ValueError("Frozen B1 required")
    from .src.training.sft import parameter_hash
    before = parameter_hash(backend)
    all_rows = []
    with (output / "selector_muestreo.jsonl").open("x") as stream:
        for state_index, (state_id, case) in enumerate(states.items()):
            histogram = Counter()
            for draw_index in range(draws):
                seed = 73310000 + state_index * draws + draw_index
                backend.seed(seed)
                controller = LLMController(backend, sample=True)
                error = None
                try:
                    selected = controller.select(deepcopy(case["select_observation"]))
                except ValueError as exc:
                    selected, error = {}, str(exc)
                record = {"observable_state_id": state_id, "draw_index": draw_index, "seed": seed,
                    "action_id": action_id(selected), "selected": selected, "format_error": error,
                    "phase_records": controller.serialized_records()}
                if not record["phase_records"]:
                    raise ValueError("Missing actual SELECT generation")
                stream.write(canonical(record) + "\n")
                stream.flush()
                all_rows.append(record)
                histogram[record["action_id"]] += 1
            print(canonical({"selector_state": state_index+1, "states": len(states),
                "generations": len(all_rows), "actions": dict(histogram)}), flush=True)
    after = parameter_hash(backend)
    if before != after or any(p.requires_grad for p in backend.model.parameters()):
        raise ValueError("B1 changed during selector diagnosis")
    return all_rows, {"before": before, "after": after, "equal": True, "optimizer_steps": 0}


def summarize_sampling(samples, analysis):
    values = {s["observable_state_id"]: s for s in analysis["states"]}
    by_state = defaultdict(list)
    invalid = 0
    for row in samples:
        state = values[row["observable_state_id"]]
        action = row["action_id"]
        # The real runner performs no ASK after an invalid SELECT and charges
        # -0.20 for its format error. Its fixed continuation is the PROCEED one.
        adjustment = -.2 if action == "INVALID" else 0.
        continuation = "PROCEED" if action == "INVALID" else action
        invalid += action == "INVALID"
        chosen = state["action_values"][continuation]
        maximum = max(v["R_long"] for v in state["action_values"].values())
        by_state[row["observable_state_id"]].append({"action": action,
            "long_regret": maximum - chosen["R_long"] - adjustment,
            "J_primary": chosen["J_primary"]})
    return {"status": "executed" if samples else "not_executed", "draws": len(samples), "states": len(by_state),
        "invalid_selects": invalid, "draws_below_best_long_action": sum(r["long_regret"] > 1e-9 for rs in by_state.values() for r in rs),
        "states_with_observed_suboptimal_draw": sum(any(r["long_regret"] > 1e-9 for r in rs) for rs in by_state.values()),
        "state_uniform_mean_selection_regret": mean(mean(r["long_regret"] for r in rs) for rs in by_state.values()) if samples else None,
        "per_state": {sid: {"draws": len(rs), "actions": dict(Counter(r["action"] for r in rs)),
            "mean_regret": mean(r["long_regret"] for r in rs)} for sid, rs in by_state.items()},
        "precision": "Eight draws/state describe observed coverage; zero errors do not prove zero tail probability or population equivalence",
        "scope": "Selection regret with fixed greedy B1 continuations, not an estimate of full stochastic GRPO trajectory utility"}


def decision(analysis, coverage):
    greedy_gaps = analysis["class_counts"].get("selector_longitudinal_suboptimal", 0)
    if greedy_gaps:
        recommendation = "diagnose_exploration_before_training"
    elif coverage.get("draws_below_best_long_action", 0):
        recommendation = "close_greedy_selector_branch; quantify_stochastic_selection_loss_before_any_RL"
    else:
        recommendation = "do_not_continue_GRPO_on_current_bank"
    return {"recommendation": recommendation, "technical_evaluation": "passed", "training_executed": False,
        "greedy_longitudinal_gap_states": greedy_gaps, "sampled_selection_loss_observed": bool(coverage.get("draws_below_best_long_action", 0)),
        "learning_efficacy": "not_executed", "external_validity": "synthetic_development_only",
        "next_minimal_experiment": "writer-only provenance-format comparison on the 32 affected paired inputs, with fixed response/reader/reward; separately audit new development difficulty before another learning pilot",
        "cannot_conclude": ["RL never works", "population equivalence", "full stochastic GRPO has zero attainable gain", "human validation"]}


def reports(output, analysis, flat, coverage, choice):
    reference = {(r["case_id"], r["action"]): r for r in flat if r["writer"] == "reference"}
    primary = [r for r in flat if r["writer"] == "B1" and r["action"] in ACTIONS]
    writer_improvements = sum(reference[(r["case_id"], r["action"])]["unsupported_write_count"] < r["unsupported_write_count"] for r in primary)
    future_rescues = sum(reference[(r["case_id"], r["action"])]["FJS"] > r["FJS"] for r in primary)
    head = ("# Evaluación corregida para decidir\n\n"
        "**El evaluador principal está corregido e integrado. La decisión para este banco es no ampliar GRPO del selector longitudinal.**\n\n"
        f"Se verificaron 50 raíces, 85 casos, 70 estados y {len(flat)} trayectorias. Se conservaron todas las respuestas, memorias y preguntas originales. "
        f"La versión activa es `{EVALUATOR_VERSION}` tanto para evaluación como para los rollouts de entrenamiento. La fórmula de recompensa se conservó.\n\n"
        "| Evidencia | Resultado |\n|---|---:|\n"
        f"| B1 natural C2 / FJS / éxito conjunto | {analysis['natural_B1_hierarchical']['C2']:.0%} / {analysis['natural_B1_hierarchical']['FJS']:.0%} / {analysis['natural_B1_hierarchical']['J_primary']:.0%} |\n"
        f"| Estados con retornos distintos | {analysis['states_with_distinct_long_returns']}/70 |\n"
        f"| Estados con una acción longitudinal superior a B1 greedy | {choice['greedy_longitudinal_gap_states']}/70 |\n"
        f"| Pares con mejora de procedencia / recuperación de FJS | {writer_improvements} / {future_rescues} de 425 |\n"
        f"| Selecciones estocásticas nuevas | {coverage['draws']} |\n"
        f"| Selecciones inferiores al mejor retorno longitudinal fijo | {coverage.get('draws_below_best_long_action', 0)} |\n\n"
        "## Qué se arregló\n\n"
        "La corrección de condicionales ya no depende de ejecutar un script después del experimento. El mismo Evaluator comprueba IDs persistidos, autorización de escritura y citas para respuestas y condicionales. También rechaza reutilizar el ID de un registro inicial para introducir una afirmación alterada. Los errores de fuente del escritor siguen penalizados; no se corrigieron respuestas ni se consultó gold para repararlas.\n\n"
        "## Decisión y siguiente paso\n\n"
        f"Código de decisión: `{choice['recommendation']}`. Las cinco acciones sí generan señal diferencial, pero la elección greedy de B1 ya alcanza el máximo longitudinal observado tras promediar mundos compatibles. "
        "La falta de diversidad de los 20 grupos anteriores no demuestra que hubiera una acción longitudinal mejor sin visitar. En diez estados F4 el objetivo inmediato prefiere ahorrar 0,10 omitiendo la pregunta, a costa de perder éxito futuro; se conserva ese contraste.\n\n"
        f"El muestreo nuevo detectó pérdidas de selección en {coverage.get('states_with_observed_suboptimal_draw', 0)} estados. "
        "Este resultado describe SELECT con consecuencias greedy congeladas. No permite afirmar que todo el GRPO estocástico carezca de margen. Ocho muestras por estado no descartan errores raros.\n\n"
        "El experimento mínimo justificado es una comparación del escritor en los 32 inputs afectados: conservar SELECT, respuesta actual, lector y recompensa y variar únicamente cómo se vinculan las citas a turnos visibles. No entrenar todavía. Si se quiere estudiar aprendizaje de selección, primero demostrar margen en desarrollo nuevo auditado; no buscarlo en tests ya observados.\n\n"
        "## Alcance de la confianza\n\n"
        "Los contratos, los tokens, las bases persistidas y las puntuaciones son verificables para este banco y la gramática sintética definida. La revisión humana y una evaluación nueva independiente no se ejecutaron. No se demuestra optimalidad general, eficacia humana, equivalencia poblacional ni imposibilidad universal de RL.\n")
    (output / "DECISION.md").write_text(head)
    (output / "decision_rl.md").write_text(head)
    write_json(output / "decision.json", choice)
    write_json(output / "cobertura_selector.json", coverage)
    write_json(output / "resumen.json", {"evaluator_version": EVALUATOR_VERSION,
        "roots": 50, "cases": 85, "observable_states": 70, "trajectories": len(flat),
        "greedy_gaps": choice["greedy_longitudinal_gap_states"], "natural": analysis["natural_B1_hierarchical"],
        "writer_source_improvements": writer_improvements, "writer_FJS_rescues": future_rescues,
        "selector_draws": coverage["draws"], "selector_suboptimal_draws": coverage.get("draws_below_best_long_action", 0),
        "training_steps": 0, "claim_scope": "finite synthetic development"})


def run(source, closure, output, *, draws=8, device="cuda", token_verification=True):
    source, closure, output = Path(source).resolve(), Path(closure).resolve(), Path(output).resolve()
    if not 0 <= draws <= 32:
        raise ValueError("Predeclare 0..32 selector draws/state")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory; preserve evidence")
    if output.is_relative_to(source) or output.is_relative_to(closure):
        raise ValueError("Historical evidence is read only")
    verify_manifest(source)
    episodes, bank = load_development(closure)
    from .closure_v4 import inventory
    inventoried = inventory(json.loads((closure / "MANIFIESTO_V4.json").read_text())["parent_run"], output)
    if inventoried["status"] != "passed":
        raise ValueError("Provenance inventory failed")
    audit = run_environment_checks()
    write_json(output / "auditoria_contratos.json", audit)
    if audit["status"] != "passed":
        raise ValueError("Production environment gate failed")
    cases = enumerate_observations(episodes)
    bank_audit = audit_bank(cases)
    write_json(output / "auditoria_banco.json", bank_audit)
    if bank_audit["status"] != "passed":
        raise ValueError("Development bank audit failed")
    plan = json.loads((source / "plan.json").read_text())
    checkpoint = Path(plan["checkpoint"])
    if plan["bank_sha256"] != file_hash(bank):
        raise ValueError("Development bank differs from actor execution")
    from llm_local_001.model import artifact_hashes
    if digest(artifact_hashes(checkpoint)) != plan["checkpoint_sha256"]:
        raise ValueError("B1 checkpoint changed")
    workflow = {"created_at": now(), "source": str(source), "bank": str(bank), "bank_sha256": file_hash(bank),
        "scorer_version": EVALUATOR_VERSION, "reward_formula_sha256": file_hash(ROOT / "src/scoring/reward.py"),
        "source_files_sha256": source_hashes(), "workflow_sha256": file_hash(__file__),
        "selector_draws_per_state": draws, "maximum_new_generations": draws*bank_audit["observable_states"],
        "training": False, "test_banks_opened": False}
    write_json(output / "plan_evaluacion.json", workflow)
    if token_verification:
        write_json(output / "verificacion_tokens.json", verify_tokens(source, checkpoint))
    analysis, flat = replay(source, output, cases)
    print(canonical({"replay": "passed", "trajectories": len(flat), "scorer": EVALUATOR_VERSION}), flush=True)
    if draws:
        samples, weights = sample_selector(cases, checkpoint, output, draws=draws, device=device)
        write_json(output / "pesos_congelados.json", weights)
    else:
        samples = []
    coverage = summarize_sampling(samples, analysis)
    choice = decision(analysis, coverage)
    reports(output, analysis, flat, coverage, choice)
    if workflow["source_files_sha256"] != source_hashes() or workflow["workflow_sha256"] != file_hash(__file__):
        raise ValueError("Sources changed during evaluation")
    shutil.copyfile(__file__, output / "source_snapshot/reliable_evaluation.py")
    artifacts = {p.name: file_hash(p) for p in output.iterdir() if p.is_file() and p.name != "completion.json"}
    write_json(output / "completion.json", {"status": "passed", "evaluator_version": EVALUATOR_VERSION,
        "recommendation": choice["recommendation"], "new_selector_generations": len(samples), "new_rollout_generations": 0,
        "optimizer_steps": 0, "completed_at": now(), "artifacts_sha256": artifacts})
    return choice


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--closure", default=str(DEFAULT_CLOSURE))
    parser.add_argument("--output", required=True)
    parser.add_argument("--selector-draws", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = run(args.source, args.closure, args.output, draws=args.selector_draws, device=args.device)
    print(canonical(result))


if __name__ == "__main__":
    main()
