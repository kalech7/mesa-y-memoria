"""Audit whether later outcomes change the value of questions in TRAIN and VALIDATION."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from research.src.artifacts import canonical, digest, file_hash, write_json
from research.tesis_interactive_reward import DEFAULT_CONFIG, aggregate, paired_group


POLICIES = ("proceed", "observable_reference", "generic_then_reference")
QUESTION_COSTS = (.02, .08, .16, .32, .64)


def rollout(story, policy, path):
    from research.tesis_interactive_env import InteractiveEpisode, reference_action

    episode = InteractiveEpisode(story, path)
    try:
        observation = episode.observe()
        prefix = digest(observation)
        for _ in range(20):
            if policy == "proceed":
                action = "PROCEED"
            elif policy == "generic_then_reference" and "GENERIC" in episode.legal_actions and not observation["asked_slots"]:
                action = "GENERIC"
            else:
                action = reference_action(observation)
                if action not in episode.legal_actions:
                    action = "PROCEED"
            observation, done = episode.step(action)
            if done:
                return prefix, episode.summary()
        raise RuntimeError("Interactive episode exceeded the fixed audit action limit")
    finally:
        episode.close()


def render(result):
    lines = ["# Auditoría de la recompensa longitudinal", "",
             "La auditoría compara ramificaciones reales del simulador en entrenamiento y validación. "
             "Cada política empieza con la misma observación y abre su propia base SQLite. La regla observable "
             "sirve de control del mecanismo. El informe registra resultados de la simulación sin entrenamiento.", "",
             "| Partición | Episodios | Cambia el retorno de tarea | Cambia solo la penalización | "
             "Cambia el acierto futuro con acierto actual fijo | El horizonte futuro cambia la trayectoria preferida |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, panel in result["partitions"].items():
        item = panel["paired_episode_diagnostics"]["all"]
        lines.append(f"| {name} | {item['episodes']} | {item['task_varies']} | {item['only_penalties_vary']} | "
                     f"{item['future_varies_with_current_fixed']} | {item['horizon_changes_preference']} |")
    lines += ["", "| Partición | Política | Acierto actual | Acierto futuro | Preguntas por episodio | Retorno longitudinal |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, panel in result["partitions"].items():
        for policy, summary in panel["policies"].items():
            values = summary["macro"]
            lines.append(f"| {name} | {policy} | {values['current_supported']:.1%} | {values['future_supported']:.1%} | "
                         f"{values['questions']:.2f} | {values['longitudinal_return']:.3f} |")
    lines += ["", "La recompensa larga asigna 1 punto al promedio de aciertos actuales sustentados y 1,5 al "
              "promedio de aciertos sustentados en las consultas terminales. Cada pregunta cuesta 0,08 puntos. "
              "Las penalizaciones cuentan salidas inválidas, escrituras aplicadas sin respaldo y afirmaciones "
              "actuales sin evidencia. La variante inmediata elimina el término de consultas terminales. Las "
              "respuestas de las visitas intermedias pueden depender de lo que se guardó antes.", "",
              "Una pregunta informativa, vaga, redundante o sin respuesta paga el mismo coste. Sus consecuencias "
              "pueden cambiar los aciertos actuales o futuros. El detalle JSON publica sensibilidad al coste, "
              "resultados por familia y conteos con varios turnos. Las trazas se exportan con los resultados.", ""]
    return "\n".join(lines)


def audit(bank, output):
    output = Path(output)
    output.mkdir(parents=True)
    source_paths = [Path(__file__).with_name(name) for name in (
        "tesis_interactive_env.py", "tesis_interactive_reward.py", Path(__file__).name)]
    sources = {path.name: file_hash(path) for path in source_paths}
    split_reports = {}
    traces_path = output / "TRAZAS.jsonl"
    with TemporaryDirectory(prefix="tesis-reward-audit-") as workspace, traces_path.open("x", encoding="utf-8") as log:
        for split in ("train", "validation"):
            panel, comparisons = defaultdict(list), defaultdict(Counter)
            sensitivity = {cost: Counter() for cost in QUESTION_COSTS}
            roots = defaultdict(set)
            for index, story in enumerate(bank[split]):
                rollouts = [rollout(story, policy, Path(workspace) / f"{split}-{index}-{policy}.sqlite")
                            for policy in POLICIES]
                prefixes, rows = zip(*rollouts)
                if len(set(prefixes)) != 1 or any(row["split"] != split for row in rows):
                    raise ValueError("Audit requires the same public starting observation and partition")
                roots[rows[0]["root_id"]].add(prefixes[0])
                for policy, row in zip(POLICIES, rows):
                    panel[policy].append(row)
                comparison = paired_group(rows[:2])
                for cost, count in sensitivity.items():
                    contrast = paired_group(rows[:2], config=replace(DEFAULT_CONFIG, question_cost=cost))
                    preferred = contrast["preferred_trajectories"]
                    count["reference_strictly_preferred_longitudinal"] += int(preferred["longitudinal"] == [1])
                    count["reference_strictly_preferred_immediate"] += int(preferred["immediate"] == [1])
                    count["future_only_switch_from_proceed_to_reference"] += int(
                        preferred == {"longitudinal": [1], "immediate": [0]}
                        and contrast["future_varies_with_current_fixed"])
                family = rows[0].get("family", rows[0].get("scenario_family", "all"))
                visit_kind = story.get("second_visit_kind", "unknown")
                for name in ("all", family, f"{family}/{visit_kind}"):
                    counts = comparisons[name]
                    counts["episodes"] += 1
                    for field in ("return_varies", "task_varies", "only_penalties_vary", "future_varies",
                                  "future_varies_with_current_fixed", "horizon_changes_preference"):
                        counts[field] += int(comparison[field])
                    long = comparison["preferred_trajectories"]["longitudinal"]
                    short = comparison["preferred_trajectories"]["immediate"]
                    counts["future_only_switch_from_proceed_to_reference"] += int(
                        long == [1] and short == [0] and comparison["future_varies_with_current_fixed"])
                log.write(canonical({"split": split, "root_id": rows[0]["root_id"], "world_id": rows[0]["world_id"],
                                     "family": family, "second_visit_kind": visit_kind,
                                     "initial_observation_sha256": prefixes[0],
                                     "comparison_proceed_reference": comparison,
                                     "policies": dict(zip(POLICIES, rows))}) + "\n")
            split_reports[split] = {"policies": {name: aggregate(rows) for name, rows in panel.items()},
                                    "paired_episode_diagnostics": dict(comparisons),
                                    "question_cost_sensitivity": {str(cost): count for cost, count in sensitivity.items()}}
            if any(len(prefixes) != 1 for prefixes in roots.values()):
                raise ValueError("World variants of a story expose different initial observations")
    if sources != {path.name: file_hash(path) for path in source_paths}:
        raise ValueError("The environment or reward source changed during the branch audit")
    result = {"scope": "TRAIN/VALIDATION-only; deterministic observable branch controls; no optimizer updates",
              "policies": POLICIES, "reward": asdict(DEFAULT_CONFIG), "partitions": split_reports,
              "bank_sha256": {name: digest(bank[name]) for name in ("train", "validation")},
              "trace_sha256": file_hash(traces_path), "source_sha256": sources}
    write_json(output / "RESULT.json", result)
    (output / "INFORME.md").write_text(render(result), encoding="utf-8")
    return result


def main():
    from research.tesis_interactive_env import build_bank

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bank = build_bank() if args.seed is None else build_bank(seed=args.seed)
    result = audit(bank, args.output)
    print(json.dumps({name: panel["paired_episode_diagnostics"] for name, panel in result["partitions"].items()},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
