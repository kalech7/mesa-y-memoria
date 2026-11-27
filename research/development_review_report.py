"""Evidence-only report for the predeclared frozen B1 development review."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from statistics import mean

from .src.artifacts import ROOT, file_hash, now, write_json
from .signal_diagnostic import ACTIONS


def check_completion(folder):
    data = json.loads((folder / "completion.json").read_text())
    assert data["status"] == "executed" and data["training"] is False
    for name, expected in data["artifacts_sha256"].items():
        assert file_hash(folder / name) == expected, name
    return data


def irreversible_prefix_error(text):
    """Conservative proof that appending tokens cannot fix an interior closer."""
    try:
        json.loads(text)
    except json.JSONDecodeError as error:
        if error.msg == "Extra data":
            return True
        return error.pos < len(text.rstrip()) - 16 and text[error.pos] in "]},:"
    return False


def paired_effect(left, right, strata, *, repeats=20000):
    assert set(left["by_root"]) == set(right["by_root"])
    groups = defaultdict(list)
    for rid in left["by_root"]:
        groups[strata[rid]].append(rid)
    rng = random.Random(20261025)
    deltas = {m: {rid: left["by_root"][rid][m] - right["by_root"][rid][m] for rid in left["by_root"]}
              for m in ("C2", "FJS", "J_primary", "R_long")}
    samples = defaultdict(list)
    for _ in range(repeats):
        roots = [rid for group in groups.values() for rid in rng.choices(group, k=len(group))]
        for metric in deltas:
            samples[metric].append(mean(deltas[metric][rid] for rid in roots))
    return {m: {"difference": mean(d.values()), "bootstrap95": [sorted(samples[m])[int(repeats*.025)],
                sorted(samples[m])[int(repeats*.975)-1]], "discordant_roots": sum(abs(v) > 1e-9 for v in d.values())}
            for m, d in deltas.items()}


def run(directory):
    directory = Path(directory).resolve()
    base, contrasts = directory / "paired_b1", directory / "contrasts"
    baseline_completion, completion = check_completion(base), check_completion(contrasts)
    study = json.loads((directory / "study_plan.json").read_text())
    assert completion["combined_new_calls"] <= study["maximum_new_generations"]
    bank = ROOT.parent / study["bank"]
    assert file_hash(bank) == study["bank_sha256"]
    episodes = [json.loads(line) for line in bank.open()]
    strata = {e["root_id"]: e["authoring"]["stratum"] for e in episodes}
    analysis = json.loads((base / "analisis.json").read_text())
    results = json.loads((contrasts / "controlled_results.json").read_text())
    selector = json.loads((contrasts / "selector_results.json").read_text())
    groups = json.loads((contrasts / "group_results.json").read_text())
    group_details = json.loads((contrasts / "group_diagnostics.json").read_text())
    horizon_signal = {}
    for temperature in study["selector_temperatures"]:
        selected = [g for g in group_details if g["temperature"] == temperature]
        horizon_signal[str(temperature)] = {
            "groups": len(selected),
            "short_task_signal_groups": sum(g["short"]["std_task"] > 0 for g in selected),
            "long_task_signal_groups": sum(g["long"]["std_task"] > 0 for g in selected),
            "future_task_signal_with_current_fixed_groups": sum(g["long"]["std_task"] > 0 and g["short"]["std_task"] == 0 for g in selected),
            "long_total_signal_with_short_zero_groups": sum(g["long"]["std_total"] > 0 and g["short"]["std_total"] == 0 for g in selected),
            "return_order_changes_groups": sum(any((g["long"]["returns"][i] - g["long"]["returns"][j]) *
                                                     (g["short"]["returns"][i] - g["short"]["returns"][j]) < 0
                                                for i in range(4) for j in range(i)) for g in selected)}
    write_json(directory / "horizon_signal.json", horizon_signal)
    baseline_rows = [json.loads(line) for line in (base / "diagnostico_50.jsonl").open()]
    natural = [r for r in baseline_rows if r["action_id"] == "NATURAL"]
    from .development_review import summarize
    contrast_rows = [json.loads(line) for line in (contrasts / "trajectories.jsonl").open()]
    assisted_rows = [r for r in contrast_rows if r["review_arm"] == "B1_assisted_prompt"]
    format_diagnosis = {}
    for arm, rows in [("B1_O2", natural), *[(arm, [r for r in contrast_rows if r["review_arm"] == arm])
                      for arm in ("B1_O1", "B1_assisted_prompt", "B1_I1")]]:
        truncated = [p for r in rows for p in r["trace"]["phase_records"] if p.get("truncated")]
        format_diagnosis[arm] = {"truncated_generations": len(truncated),
            "truncated_with_irreparable_interior_JSON_error": sum(irreversible_prefix_error(p["raw_output"]) for p in truncated),
            "definition": "Interior unexpected closing punctuation at least 16 characters before the cutoff, or extra content after a full JSON value. Conservative append-only test; no actor output repaired."}
    write_json(directory / "FORMAT_DIAGNOSIS.json", format_diagnosis)
    forms = {arm: {variant: summarize([r for r in rows if r["realization_id"] == variant], episodes)
                   for variant in ("v0", "v1")}
             for arm, rows in (("B1", natural), ("B1_assisted_prompt", assisted_rows))}
    write_json(directory / "paraphrase_results.json", forms)
    error_codes = Counter()
    truncated_phases = Counter()
    for row in natural:
        for error in row["trace"]["errors"]:
            detail = error.get("error", error.get("kind", "unspecified"))
            if str(detail).startswith("InvalidOutput("):
                detail = "JSON_OR_SCHEMA_INVALID"
            error_codes[error["phase"] + ":" + str(detail)] += 1
        for record in row["trace"]["phase_records"]:
            if record.get("truncated"):
                truncated_phases[record["phase"]] += 1
    future_success_with_write_error = sum(bool(row["metrics"]["FJS"]) and
        any(e["phase"] == "write" for e in row["trace"]["errors"]) for row in natural)
    assert len(natural) == len(episodes) == 120
    assert baseline_completion["trajectories"] == 1320
    comparisons = {"PI1_O2_minus_O1": ("B1_O2", "B1_O1"), "PI2_I2_minus_I1": ("B1_I2_paired", "B1_I1"),
                   "assisted_prompt_minus_B1": ("B1_assisted_prompt", "B1_O2"),
                   "reference_writer_minus_B1": ("B1_reference_writer_natural_action", "B1_O2")}
    effects = {name: paired_effect(results[a], results[b], strata) for name, (a, b) in comparisons.items()}
    write_json(directory / "paired_effects.json", {"effects": effects, "bootstrap": "20000 paired root resamples, stratified by five construction strata, seed 20261025, percentile95; exploratory, no multiplicity-adjusted confirmatory claim. Zero-width intervals do not establish equivalence."})
    sampled = [json.loads(line) for line in (contrasts / "sampled_trajectories.jsonl").open()]
    sampled_groups = defaultdict(list)
    for row in sampled:
        sampled_groups[(row["temperature"], row["root_id"], row["latent_world_id"])].append(row)
    real_signal = {}
    for temperature in study["selector_temperatures"]:
        selected = [rows for (t, _, _), rows in sampled_groups.items() if t == temperature]
        real_signal[str(temperature)] = {"groups": len(selected),
            "groups_with_distinct_joint_success": sum(len({r["metrics"]["C2"] * r["metrics"]["FJS"] for r in rows}) > 1 for rows in selected),
            "groups_with_any_joint_success": sum(any(r["metrics"]["C2"] * r["metrics"]["FJS"] for r in rows) for rows in selected),
            "groups_with_identical_WRITE_prefix_and_different_FJS": sum(
                len({json.dumps(r["trace"]["observations"]["write"], sort_keys=True) for r in rows}) == 1
                and len({r["metrics"]["FJS"] for r in rows}) > 1 for rows in selected),
            "groups_with_joint_variation_and_valid_success": sum(
                len({r["metrics"]["C2"] * r["metrics"]["FJS"] for r in rows}) > 1
                and any(r["metrics"]["C2"] * r["metrics"]["FJS"] and not r["metrics"]["invalid_outputs"]
                        and not r["metrics"]["unsupported_operations"] for r in rows) for rows in selected),
            "invalid_outputs": sum(r["metrics"]["invalid_outputs"] for rows in selected for r in rows),
            "rollouts": sum(len(rows) for rows in selected),
            "joint_successes": sum(r["metrics"]["C2"] * r["metrics"]["FJS"] for rows in selected for r in rows)}
    classes = analysis["class_counts"]
    gaps = classes.get("selector_longitudinal_suboptimal", 0)
    joint_gaps = 0
    joint_reward_gaps = set()
    informative_joint_gaps = set()
    continuation_only_joint_gaps = set()
    action_rows = defaultdict(list)
    for row in baseline_rows:
        if row["writer"] == "B1" and row["action_id"] in ACTIONS:
            action_rows[(row["observable_state_id"], row["action_id"])].append(row)
    state_map = {s["observable_state_id"]: s for s in analysis["states"]}
    for state in analysis["states"]:
        chosen = state["action_values"].get(state["B1_action"])
        if chosen is None:
            continue
        joint_gaps += any(v["J_primary"] > chosen["J_primary"] + 1e-9 for v in state["action_values"].values())
        if any(v["J_primary"] > chosen["J_primary"] + 1e-9 and v["R_long"] > chosen["R_long"] + 1e-9
               for v in state["action_values"].values()):
            joint_reward_gaps.add(state["observable_state_id"])
            informative = False
            for action, candidate in state["action_values"].items():
                if candidate["J_primary"] > chosen["J_primary"] + 1e-9 and candidate["R_long"] > chosen["R_long"] + 1e-9:
                    informative |= any(r["metrics"].get("response_informative") for r in action_rows[(state["observable_state_id"], action)])
            (informative_joint_gaps if informative else continuation_only_joint_gaps).add(state["observable_state_id"])
    joint_sampling = defaultdict(set)
    informative_sampling = defaultdict(set)
    for line in (contrasts / "selector_samples.jsonl").open():
        sample = json.loads(line)
        state = state_map[sample["observable_state_id"]]
        candidate = state["action_values"].get(sample["action"])
        chosen = state["action_values"].get(state["B1_action"])
        if candidate and chosen and candidate["J_primary"] > chosen["J_primary"] + 1e-9 and candidate["R_long"] > chosen["R_long"] + 1e-9:
            joint_sampling[str(sample["temperature"])].add(sample["observable_state_id"])
            if any(r["metrics"].get("response_informative") for r in action_rows[(sample["observable_state_id"], sample["action"]) ]):
                informative_sampling[str(sample["temperature"])].add(sample["observable_state_id"])
    reference_values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in baseline_rows:
        if row["writer"] == "reference":
            reference_values[row["observable_state_id"]][row["action_id"]][row["latent_world_id"]].append(row)
    reference_joint_regret_states = 0
    for state in analysis["states"]:
        values = {action: mean(mean(r["J_primary"] for r in variants) for variants in worlds.values())
                  for action, worlds in reference_values[state["observable_state_id"]].items()}
        if state["B1_action"] in values:
            reference_joint_regret_states += max(values.values()) > values[state["B1_action"]] + 1e-9
    invalid_select_states = sum(s["B1_action"] == "INVALID" for s in analysis["states"])
    writer_losses = len(analysis["natural_writer_loss_cases"])
    reader_losses = len(analysis["natural_reader_loss_cases"])
    improved_sampling = max(s["states_visiting_improvement"] for s in selector.values())
    if joint_reward_gaps and not informative_joint_gaps:
        decision = "Hay acciones que mejoran éxito conjunto y recompensa sin aportar información nueva. Priorizar respuesta/escritura con la evidencia ya disponible; no atribuir ese margen a aprender aclaraciones informativas."
    elif informative_joint_gaps and any(informative_sampling.values()):
        decision = "Hay margen alcanzable de selección que mejora éxito conjunto y recompensa, y el muestreo lo visita en desarrollo. No se demuestra todavía aprendizaje: preparar entrenamiento independiente y un piloto con controles antes de una campaña RL."
    elif joint_reward_gaps:
        decision = "Hay margen de selección que mejora éxito conjunto y recompensa bajo consecuencias greedy, pero el muestreo probado no lo visita. Priorizar cobertura/exploración o supervisión de la decisión antes de aumentar pasos GRPO."
    elif writer_losses or reader_losses or invalid_select_states:
        decision = "Priorizar el componente que falla y cerrar RL del selector en este banco bajo consecuencias greedy. La variación debida a escritura/formato no demuestra que preguntar de otro modo mejore el resultado."
    elif gaps:
        decision = "La selección puede mejorar recompensa sin demostrar una mejora conjunta de respuesta y futuro; separar ahorro de preguntas y éxito real. No justificar una campaña RL para mejorar la tarea solo con ese margen de coste."
    else:
        decision = "Cerrar RL del selector en este banco: no se observó una mejora longitudinal alcanzable por otra acción legal con B1 continuador congelado."
    sampled_joint_signal = any(r["groups_with_joint_variation_and_valid_success"] for r in real_signal.values())
    sampled_writer_signal = any(r["groups_with_identical_WRITE_prefix_and_different_FJS"] for r in real_signal.values())
    if sampled_joint_signal:
        integrated_decision = "Se observó señal de tarea muestreada con éxitos válidos y fallos dentro de un mismo grupo. Justifica estudiar la continuación del componente que falla, sobre entrenamiento independiente y con controles; no demuestra que RL ya mejore."
    else:
        integrated_decision = "No se observó variación de éxito conjunto con éxitos válidos en los grupos probados. Priorizar competencia/supervisión del componente que falla antes de ampliar GRPO; el muestreo finito no prueba ausencia universal de señal."
    assisted_metrics = results["B1_assisted_prompt"]["root_weighted"]
    original_metrics = results["B1_O2"]["root_weighted"]
    if assisted_metrics["J_primary"] > original_metrics["J_primary"] + 1e-9 and assisted_metrics["invalid_outputs"] <= original_metrics["invalid_outputs"]:
        next_experiment = "Reproducir primero la ayuda por instrucciones ya medida sobre otra cohorte de desarrollo independiente, con el mismo checkpoint y controles de daño por estrato. Si persisten pérdidas de escritura, el piloto de aprendizaje debe usar un train nuevo con versiones, atributos y alcances combinados; estas 50 raíces quedan como desarrollo, sin actualizar pesos sobre ellas."
    elif writer_losses:
        next_experiment = "Preparar un train independiente que cubra sustitución de versiones, contexto con fecha y consulta de otro atributo. Ejecutar un piloto pequeño de competencia del escritor frente a más SFT; si la señal estocástica lo respalda, contrastar RL inmediato y longitudinal bajo el mismo presupuesto y parent B1. Una primera sonda puede congelar SELECT/RESPOND para aislar WRITE, identificándola como variante del método; después revalidar el controlador completo. No entrenar sobre estas 50 raíces ni usar los tests observados."
    else:
        next_experiment = "Atender el componente concreto identificado en CASOS_MODELO.md y reproducir el contraste en desarrollo independiente antes de aumentar entrenamiento. No ampliar GRPO por la mera existencia de varianza de penalizaciones."
    final = {"status": "completed_exploratory_development_review", "completed_at": now(), "roots": 50,
        "episodes": 120, "observable_states": analysis["observable_states"], "new_generations": completion["combined_new_calls"],
        "training_steps": 0, "checkpoint_changed": False, "new_confirmatory_test_opened": False,
        "selector_states_with_longitudinal_regret": gaps, "states_with_invalid_select": invalid_select_states,
        "selector_states_with_joint_success_regret": joint_gaps,
        "states_with_joint_and_reward_improvement": len(joint_reward_gaps),
        "joint_reward_improvement_states_with_informative_question": len(informative_joint_gaps),
        "joint_reward_improvement_states_without_information_gain": len(continuation_only_joint_gaps),
        "sampled_states_with_joint_and_reward_improvement": {str(t): len(joint_sampling[str(t)]) for t in study["selector_temperatures"]},
        "sampled_states_visiting_informative_joint_reward_improvement": {str(t): len(informative_sampling[str(t)]) for t in study["selector_temperatures"]},
        "joint_regret_states_after_reference_writer": reference_joint_regret_states,
        "natural_writer_loss_cases": writer_losses, "natural_reader_loss_cases": reader_losses,
        "sampling_states_visiting_improvement": improved_sampling,
        "natural_B1": results["B1_O2"], "assisted_B1": results["B1_assisted_prompt"],
        "selector_sampling": selector, "sampled_groups": groups, "sampled_real_success": real_signal,
        "immediate_versus_longitudinal_group_signal": horizon_signal,
        "integrated_policy_observed_joint_task_signal": sampled_joint_signal,
        "observed_writer_signal_at_identical_public_prefix": sampled_writer_signal,
        "integrated_RL_decision": integrated_decision,
        "next_minimal_experiment": next_experiment,
        "effects": effects, "decision": decision, "assistance_is_RL": False,
        "natural_error_codes": dict(error_codes), "natural_truncated_phases": dict(truncated_phases),
        "natural_future_success_despite_write_error": future_success_with_write_error,
        "paraphrase_results": {arm: {v: value["root_weighted"] for v, value in variants.items()} for arm, variants in forms.items()},
        "format_diagnosis": format_diagnosis,
        "limits": ["One checkpoint/seed, fixed synthetic food grammar, AI-authored development, not confirmatory or independent human validation.",
                   "No proof of RL superiority over B1, continued SFT or immediate RL without new matched training and locked tests.",
                   "Temperature changes are development probes, not a demonstrated optimal setting."]}
    write_json(directory / "DECISION_FINAL.json", final)
    table = ["| Condición | C2 | FJS | Éxito conjunto | Preguntas | Salidas inválidas |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("B1_O2", "B1_O1", "B1_I1", "B1_I2_paired", "B1_assisted_prompt", "B1_reference_writer_natural_action"):
        r = results[name]
        c = r["counts"]
        table.append(f"| {name} | {c['C2']}/120 | {c['FJS']}/120 | {c['J_primary']}/120 | {c['questions']} | {c['invalid_outputs']} |")
    weighted_table = ["| Condición | C2 por raíz | FJS por raíz | J por raíz | Errores medios por raíz |", "|---|---:|---:|---:|---:|"]
    for name in ("B1_O2", "B1_O1", "B1_I1", "B1_I2_paired", "B1_assisted_prompt", "B1_reference_writer_natural_action"):
        value = results[name]["root_weighted"]
        weighted_table.append(f"| {name} | {value['C2']:.3f} | {value['FJS']:.3f} | {value['J_primary']:.3f} | {value['invalid_outputs']:.3f} |")
    stratum_table = ["| Estrato B1 | C2 | FJS | J | Errores medios |", "|---|---:|---:|---:|---:|"]
    for name, value in results["B1_O2"]["strata"].items():
        stratum_table.append(f"| {name} | {value['C2']:.3f} | {value['FJS']:.3f} | {value['J_primary']:.3f} | {value['invalid_outputs']:.3f} |")
    effect_lines = []
    for name, metrics in effects.items():
        value = metrics["J_primary"]
        low, high = value["bootstrap95"]
        effect_lines.append(f"- {name}: ΔJ por raíz={value['difference']:+.4f}; intervalo exploratorio 95% [{low:+.4f}, {high:+.4f}], {value['discordant_roots']}/50 raíces discordantes.")
    director = ["# Respuesta a Vincent: evaluación nueva y revisión de RL", "", decision, "", integrated_decision, "",
        "Las tres preguntas provienen de la SPEC y de RESPUESTAS_DIRECTOR.md archivado; no se atribuyen a Vincent exigencias adicionales ni una cita literal no disponible.", "",
        "## Resultados ejecutados", "", *table, "",
        "Los conteos incluyen todos los episodios. Las 120 variantes no son 120 usuarios: las comparaciones promedian realizaciones, mundos y luego 50 raíces, con cinco estratos del mismo tamaño. Se conservaron 1.320 trayectorias pareadas y todas las salidas inválidas.", "",
        "Las tasas primarias ponderadas por raíz son las siguientes; los conteos anteriores sobreponderan las raíces que contienen dos mundos.", "", *weighted_table, "",
        "### Desglose de B1", "", *stratum_table, "",
        f"Hay {future_success_with_write_error} episodios con FJS=1 pese a un error de WRITE. Una consulta futura fuera del alcance de la nueva excepción puede acertarse usando memoria previa aunque haya fallado guardar el cambio. Por eso FJS se acompaña de validez de escritura, alcance y éxito conjunto; no se afirma que la memoria esté bien porque una consulta futura salga bien.", "",
        f"Errores naturales por tipo: {json.dumps(dict(error_codes), ensure_ascii=False)}. Truncamientos por fase: {json.dumps(dict(truncated_phases), ensure_ascii=False)}. Un intervalo vacío, una cita inexacta o un contexto ausente no se corrigen aumentando el contexto disponible.", "",
        f"Con B1, el éxito conjunto ponderado por raíz fue {forms['B1']['v0']['root_weighted']['J_primary']:.4f} con la redacción v0 y {forms['B1']['v1']['root_weighted']['J_primary']:.4f} con v1. Las parejas mantienen memoria, mundo y consultas; cambia la redacción. Es un contraste de dos formas sintéticas fijadas, no una estimación de robustez a todo el español. Detalle por estrato en paraphrase_results.json.", "",
        f"En O1 hubo {format_diagnosis['B1_O1']['truncated_generations']} generaciones truncadas; {format_diagnosis['B1_O1']['truncated_with_irreparable_interior_JSON_error']} ya contenían un error JSON interior que no se arregla añadiendo tokens. El resto no se clasifica causalmente solo con este test. El límite de 512 se mantuvo igual en todas las condiciones; no se amplió retrospectivamente para favorecer O1. Detalle en FORMAT_DIAGNOSIS.json.", "",
        "## PI1: evidencia disponible", "",
        "Se ejecutó B1 congelado O1/O2 en los mismos episodios. O1 reduce la memoria visible al registro pertinente más reciente y O2 muestra toda la memoria; el estado inicial persistido y las consultas permanecen iguales. Es una comparación de políticas naturales: también puede cambiar qué pregunta decide hacer el modelo. La diferencia incorpora fallos de formato; si O1 no logra escribir JSON válido, no se puede atribuir su pérdida únicamente al valor semántico del historial. Las tasas de error y la repetición/truncamiento se conservan como explicación del mecanismo y límite de PI1.", "",
        "## PI2: adquisición y persistencia", "",
        f"Se comprobaron {completion['PI2_pairs_verified']} pares I1/I2: mismo slot fijado desde evidencia pública, misma respuesta del simulador y misma respuesta actual de B1. La intervención cambia únicamente si WRITE recibe la aclaración. No se escoge el slot usando el mundo oculto.", "",
        f"En duración inicialmente ambigua, J ponderado por raíz pasa de {results['B1_I1']['strata']['ambiguous_temporal_boundary']['J_primary']:.4f} en I1 a {results['B1_I2_paired']['strata']['ambiguous_temporal_boundary']['J_primary']:.4f} en I2. Los otros cuatro estratos sirven para comprobar daño o cambios de conducta ante preguntas redundantes; no toda respuesta entregada aporta información nueva. Su desglose se conserva en controlled_results.json.", "",
        "## PI3: valor añadido de RL", "",
        f"No hubo actualizaciones de pesos. Hay {gaps}/{analysis['observable_states']} estados con una acción legal de mayor R_long que la seleccionada por B1, {joint_gaps} con margen de éxito conjunto, {len(joint_reward_gaps)} con mejora simultánea de J y R_long, y {invalid_select_states} estados con SELECT inválido. Se distingue ahorrar una pregunta de resolver mejor la tarea. Las acciones se comparan tras promediar mundos compatibles; no se da al actor el mundo verdadero. El diagnóstico del escritor rescata {writer_losses} casos naturales; el del lector, {reader_losses}.", "",
        f"Entre los márgenes simultáneos, {len(informative_joint_gaps)} estados tienen una pregunta mejor que aporta información y {len(continuation_only_joint_gaps)} mejoran sin ganancia de información detectada. Una mejora por evitar una pregunta o por cambiar el prefijo con una respuesta neutra no demuestra adquisición útil de conocimiento.", "",
        f"Tras sustituir solo WRITE en todas las acciones y conservar sus respuestas actuales, quedan {reference_joint_regret_states} estados con margen de éxito conjunto para cambiar la selección. Este es un diagnóstico condicionado al escritor de referencia, no otra puntuación del B1 original.", "",
        "Se ejecutaron 800 selecciones estocásticas y 96 trayectorias completas en 24 grupos nuevos, correspondientes a diez raíces y doce combinaciones raíz/mundo probadas en dos temperaturas. Tener varianza en la recompensa permite examinar gradientes potenciales; no demuestra que los cambios de peso mejoren la tarea. Estos grupos de otra cohorte no se suman al piloto histórico para satisfacer su ventana predeclarada de 40. Pertenecen a desarrollo, no son entrenamiento y no se reciclan como datos on-policy tras una actualización.", "",
        f"Se recalcularon ambos horizontes sobre las mismas trayectorias. Grupos donde R_long tiene señal y R_short es constante: {json.dumps({t: h['long_total_signal_with_short_zero_groups'] for t, h in horizon_signal.items()}, ensure_ascii=False)}. La señal potencial diferenciada del horizonte se separa del efecto de entrenar un actor con ella; este último no se ejecutó. Detalle en horizon_signal.json.", "",
        integrated_decision, "",
        f"Variación de FJS con prefijo de WRITE idéntico: {json.dumps({t: v['groups_with_identical_WRITE_prefix_and_different_FJS'] for t, v in real_signal.items()}, ensure_ascii=False)} grupos. El lector es fijo; esta comparación localiza variación estocástica en escritura bajo la misma evidencia, sin confundirla con cambiar de pregunta.", "",
        "## Ayuda de IA y efectos pareados", "",
        "La condición assisted_prompt usa el mismo B1 con instrucciones generales fijadas antes de inspeccionar las puntuaciones del banco nuevo. Cambia instrucciones de tres fases; por sí sola no identifica cuál de esas fases explica una diferencia. La sustitución del escritor sí conserva la respuesta actual y la evidencia recibida. Ninguna condición se etiqueta como RL ni como revisión humana.", "", *effect_lines, "",
        "Los intervalos son exploratorios por raíces; no autorizan declaraciones confirmatorias de superioridad, equivalencia o seguridad ni sustituyen las tres semillas y controles de la SPEC.", "",
        "## Alcance y siguiente paso", "", decision, "", integrated_decision, "", next_experiment, "",
        "El banco histórico contaminado y los tests observados no se usaron para seleccionar cambios nuevos. El banco nuevo sigue siendo desarrollo. Para una afirmación de mejora por aprendizaje hacen falta entrenamiento separado, validación para selección y un test nuevo bloqueado. No se entrenó sobre estas 50 raíces ni se modificó la recompensa para crear ventajas.", ""]
    (directory / "RESPUESTA_VINCENT.md").write_text("\n".join(director))
    parameter_table = ["| Parámetro/componente | Valor efectivo | Qué hacer y por qué |", "|---|---|---|"]
    parameter_table += [
        "| Temperatura de muestreo | 1,0 frente a 1,3, evaluadas aquí | Solo influye con sample=True; cambiarla en la evaluación greedy no cambia las respuestas. Comparar cobertura de mejoras, éxito real y errores; diversidad sola no basta. El futuro entrenamiento debe usar la misma temperatura para actor, logprobs y referencia. |",
        "| Learning rate de RL | 5e-6 | Mantenerlo como punto de partida. Solo con señal de tarea observada comparar 1e-6 frente a 5e-6 en entrenamiento/validación nuevos; este rango es una propuesta sin ensayo de optimización aquí. LR mayor no arregla ventajas cero. |",
        "| Tamaño de grupo | 4 | Mantener para el siguiente piloto comparable. Ocho trayectorias requieren una variante auditada: el entrenador actual exige exactamente cuatro. No introducir acciones forzadas dentro de grupos on-policy. |",
        "| Raíces por actualización | 2 en el piloto previo | Aumentar a 4 u 8 solo si hay señal en los datos de entrenamiento y cabe en el presupuesto. Más raíces no resuelven ausencia de variación dentro de cada grupo. |",
        "| KL | 0,02 contra el B1 con su adaptador | No bajarlo para fabricar cambios. Solo estudiar 0,01/0,02 con deriva medida y validación independiente si la regularización domina un gradiente de tarea no nulo. En el piloto previo ambos gradientes eran cero. |",
        "| Clipping | 0,20; una pasada de optimización por rollout | El piloto registró ratios=1 y clipping=0. Cambiar epsilon no recupera señal nula; con una pasada antes de cada step no hay reutilización de muestras que active el clipping en ese step. |",
        "| Epsilon de ventajas | 1e-6 | No tocar para resolver grupos con recompensas idénticas: el numerador sigue siendo cero. |",
        "| Crédito por fase | Un retorno de episodio a SELECT/RESPOND/WRITE; normalizado por tokens | En el piloto previo los pesos medios por longitud fueron 10,26% SELECT, 24,29% RESPOND y 65,45% WRITE; no son porcentajes medidos del gradiente. Si el contraste de escritura rescata pérdidas, estudiar primero escritura con el selector fijo. Una variante que entrene solo SELECT requiere congelar su continuador B1 y declararla como método distinto. |",
        "| LoRA | rango 16, alpha 32, dropout 0, q/k/v/o | Mantener para conservar el parent B1 y la comparabilidad. Cambiar rango/backbone no es un ajuste pequeño del mismo checkpoint. |",
        "| Entrada/salida | 2048; SELECT 64, RESPOND 256, WRITE 512 | El preflight de todas las condiciones dio un máximo de 1600 tokens de entrada. No hay evidencia para ampliar contexto ni recortar WRITE; conservar errores de truncamiento como fallos. |",
        "| SFT y splits | B1 proviene de dos épocas; separación nueva validada | Si faltan competencias de fechas/citas/contexto, ampliar supervisión en un train independiente y compararlo como más SFT, con igual presupuesto. No reducir B1 adrede para crear margen para RL. |",
        "| Recompensa | C2 + FJS con penalizaciones originales | No modificar coeficientes a partir de este resultado. Conservar éxito conjunto y componentes para detectar variación solo por penalizaciones. |",
    ]
    param = ["# Parámetros que conviene tocar y parámetros que no resuelven el fallo", "",
        "La prioridad depende del componente limitante medido; los valores propuestos para futuros entrenamientos no son óptimos demostrados.", "", *parameter_table, "",
        "## Ensayo ejecutado de temperatura", "", "```json", json.dumps({"selector": selector, "full_groups": groups, "real_success": real_signal}, ensure_ascii=False, indent=2), "```", "",
        "## Fundamento y límites", "",
        "Los logs locales del piloto anterior muestran grad_norm=0, policy_loss=0, KL=0, ratios=1 y 20 grupos con retornos idénticos. Es evidencia directa para no intentar arreglar ese piloto aumentando LR o cambiando clipping. La revisión del código conserva referencia congelada, máscaras causales, independencia SQLite y semillas; las pruebas del objetivo no se sustituyen por los resultados de B4.", "",
        "GRPO usa comparaciones dentro del grupo ([DeepSeekMath](https://arxiv.org/abs/2402.03300)). DAPO estudia, entre otros problemas, grupos sin señal y exploración ([artículo del proyecto](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf)). Estos trabajos motivan diagnosticar la varianza; sus resultados y tamaños de modelos no validan copiar hiperparámetros a este Qwen de 0,5B. No se implementó DAPO ni filtrado dinámico de ejemplos en esta ejecución.", "", decision, ""]
    (directory / "PARAMETROS_RECOMENDADOS.md").write_text("\n".join(param))
    failures = ["# Casos completos para revisar el modelo", "", "Todos los fallos naturales se incluyen; los golds de auditoría nunca entraron al prompt del actor.", ""]
    references = {(r["diagnostic_case_id"], r["action_id"]): r for r in baseline_rows if r["writer"] == "reference"}
    for row in natural:
        if row["metrics"]["C2"] and row["metrics"]["FJS"] and not row["metrics"]["invalid_outputs"]:
            continue
        episode = row["trace"]["episode"]
        ref = references.get((row["diagnostic_case_id"], row["natural_action_id"]))
        item = {"case_id": row["diagnostic_case_id"], "root": row["root_id"], "stratum": episode["authoring"]["stratum"],
                "visible_user": episode["user_messages"], "query": episode["current_query"],
                "model_actions": row["parsed_actions"], "reply": row["user_reply"], "errors": row["trace"]["errors"],
                "initial_memory": row["trace"]["initial_memory"], "final_memory": row["trace"]["final_memory"],
                "future_query": episode["future_query"], "future_response": row["trace"]["future_response"],
                "audit_truth_not_visible_to_actor": episode["truth"], "scores": row["scores"],
                "same_evidence_reference_writer_FJS": ref["metrics"]["FJS"] if ref else None}
        failures += [f"## {row['diagnostic_case_id']}", "", "```json", json.dumps(item, ensure_ascii=False, indent=2), "```", ""]
    (directory / "CASOS_MODELO.md").write_text("\n".join(failures))
    names = ("DECISION_FINAL.json", "RESPUESTA_VINCENT.md", "PARAMETROS_RECOMENDADOS.md", "CASOS_MODELO.md", "paired_effects.json",
             "study_plan.json", "execution_freeze.json", "assistance_predeclared.json", "TOKEN_PREFLIGHT.json",
             "paired_b1/completion.json", "contrasts/completion.json", "REVIEW_TESTS.xml", "ALL_TESTS.xml", "RL_PARAMETER_AUDIT.json", "paraphrase_results.json", "horizon_signal.json", "FORMAT_DIAGNOSIS.json")
    write_json(directory / "completion.json", {"status": "executed", "training": False, "completed_at": now(),
        "new_generations": completion["combined_new_calls"], "optimizer_steps": 0,
        "artifacts_sha256": {name: file_hash(directory / name) for name in names}, "report_source_sha256": file_hash(__file__)})
    print(json.dumps({"status": final["status"], "decision": decision, "B1": results["B1_O2"]["root_weighted"],
                      "assisted": results["B1_assisted_prompt"]["root_weighted"], "new_generations": completion["combined_new_calls"]}, ensure_ascii=False))
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    run(parser.parse_args(argv).directory)


if __name__ == "__main__":
    main()
