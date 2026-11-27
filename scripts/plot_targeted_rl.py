"""Export the phase-credit RL run, with frozen-baseline and strict evaluation."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

from plot_live_rl import (
    BLUE, COLORS, ORANGE, PURPLE, Evidence, pending, plot_validation, setup, validation_points,
)
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def make_figure(run, output):
    evidence = Evidence()
    plan = evidence.json(run / "PILOT_PLAN.json")
    if plan is None:
        raise FileNotFoundError(run / "PILOT_PLAN.json")
    result = evidence.json(run / "rl/optimizer/training.json")
    progress = evidence.jsonl(run / "rl/optimizer/progress.jsonl")
    passes = evidence.jsonl(run / "rl/optimizer/optimizer_passes.jsonl")
    groups = evidence.jsonl(run / "rl/optimizer/group_diagnostics.jsonl")
    rollout_metrics = evidence.jsonl(run / "rl/optimizer/training_metrics.jsonl")
    baseline = evidence.json(run / "baseline_validation/evaluation_00/strict_metrics.json")
    comparison = evidence.json(run / "COMPARISON.json")
    validation = dict(validation_points(progress, result))
    if baseline is not None:
        validation.setdefault(0, baseline["J_strict"])
    by_pass = defaultdict(list)
    seen = set()
    for row in passes:
        identity = (row["outer_update"], row["optimization_pass"])
        if identity in seen:
            raise ValueError(f"Repeated outer-update/pass observation {identity}")
        seen.add(identity)
        by_pass[row["optimization_pass"]].append(row)
    observed_outer = max((row["step"] for row in progress if row.get("stage") == "training"), default=0)
    observed_adam = max((row["optimizer_step"] for row in passes), default=0)
    planned_outer = plan["rl"]["updates"]
    timestamp = datetime.now(ZoneInfo("America/Guayaquil")).strftime("%d %b %Y %H:%M %Z")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10,
                         "legend.fontsize": 8.5, "svg.fonttype": "none", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.subplots_adjust(left=.065, right=.98, bottom=.23, top=.84, hspace=.72, wspace=.32)
    fig.suptitle("RL con crédito por fase y memoria persistente", x=.065, y=.972, ha="left",
                 fontsize=21, fontweight="bold", color="#172B42")
    status = "evaluación retenida completa" if comparison else "evaluación retenida pendiente"
    fig.text(.065, .926, f"Qwen2.5-0.5B · semilla {plan['seed']} · SFT adicional 0 · "
             f"grupos {observed_outer}/{planned_outer} · pasos Adam {observed_adam}/{plan['rl']['optimizer_steps']} · {status}",
             fontsize=11, color="#47586B")
    fig.text(.065, .895, f"{run.name} · corte {timestamp}", fontsize=9, color="#607185")
    pass_colors = {1: BLUE, 2: ORANGE}

    axis = axes[0, 0]
    setup(axis, "A  Pérdida GRPO por pasada", "Grupo RL / actualización externa", "Loss total media")
    for index, rows in sorted(by_pass.items()):
        axis.plot([row["outer_update"] for row in rows], [row["loss"] for row in rows],
                  color=pass_colors.get(index), marker="o", markersize=3.5, linewidth=1.2,
                  label=f"TRAIN · pasada {index}")
    if passes:
        axis.axhline(0, color="#798797", linewidth=.7)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        axis.legend()
    else:
        pending(axis, "Sin pasadas completas todavía")
    axis.set_xlim(0, planned_outer + .5)

    axis = axes[0, 1]
    setup(axis, "B  Cambio de probabilidades", "Grupo RL / actualización externa", "Ratio frente a política de muestreo")
    for index, rows in sorted(by_pass.items()):
        x = [row["outer_update"] for row in rows]
        color = pass_colors.get(index)
        axis.fill_between(x, [row["ratio_min"] for row in rows], [row["ratio_max"] for row in rows],
                          alpha=.12, color=color)
        axis.plot(x, [row["ratio_mean"] for row in rows], color=color, marker="o", markersize=3.5,
                  linewidth=1.1, label=f"Pasada {index} · media y rango")
    axis.axhline(1, color="#798797", linewidth=.7)
    for value in (.8, 1.2):
        axis.axhline(value, color="#909CA8", linewidth=.8, linestyle="--")
    if passes:
        axis.legend()
    else:
        pending(axis, "Sin ratios medidos todavía")
    axis.set_xlim(0, planned_outer + .5)

    axis = axes[0, 2]
    setup(axis, "C  Retorno asignado a cada fase", "Grupo RL de cuatro trayectorias", "Retorno medio de fase")
    for phase, color in COLORS.items():
        rows = [row for row in groups if phase in row.get("phase_returns", {})]
        if rows:
            axis.plot([row["optimizer_update"] for row in rows],
                      [mean(row["phase_returns"][phase]) for row in rows], color=color,
                      marker="o", markersize=3.5, linewidth=1.1, label=phase.upper())
    if groups:
        axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, -.24), frameon=False)
    else:
        pending(axis, "Sin crédito de fase observado todavía")
    axis.set_xlim(0, planned_outer + .5)

    axis = axes[1, 0]
    setup(axis, "D  Éxito conjunto estricto", "Grupo TRAIN / checkpoint RL", "J estricto · proporción")
    train_strict = [row for row in rollout_metrics if "rollout/J_strict" in row["values"]]
    if train_strict:
        axis.plot([row["step"] for row in train_strict], [row["values"]["rollout/J_strict"] for row in train_strict],
                  color=ORANGE, marker="o", markersize=3.5, linewidth=1.1, label="TRAIN · 4 rollouts muestreados")
    plot_validation(axis, sorted(validation.items()))
    if baseline is not None:
        axis.axhline(baseline["J_strict"], color="#788795", linewidth=1, linestyle="--",
                     label=f"Baseline VALIDATION · J {baseline['J_strict']:.1%}")
    if train_strict or validation:
        axis.legend(loc="upper center", bbox_to_anchor=(.5, -.24), frameon=False, ncol=1)
    else:
        pending(axis, "Validación inicial en curso")
    axis.set_ylim(-.04, 1.14)
    axis.set_xlim(-.5, planned_outer + .5)
    axis.yaxis.set_major_formatter(PercentFormatter(1))

    axis = axes[1, 1]
    setup(axis, "E  Seguridad y beneficio de WRITE", "Grupo TRAIN de cuatro trayectorias", "Fracción de trayectorias")
    rate_labels = {"rollout/helpful_write_rate": ("WRITE con beneficio causal", "#009E73"),
                   "rollout/unsafe_rate": ("Trayectoria insegura", ORANGE),
                   "rollout/unsupported_write_rate": ("Escritura sin sustento", PURPLE)}
    for tag, (label, color) in rate_labels.items():
        rows = [row for row in rollout_metrics if tag in row["values"]]
        if rows:
            axis.plot([row["step"] for row in rows], [row["values"][tag] for row in rows],
                      color=color, marker="o", markersize=3.5, linewidth=1.1, label=label)
    if rollout_metrics:
        axis.legend(loc="upper center", bbox_to_anchor=(.5, -.24), frameon=False, ncol=1)
    else:
        pending(axis, "Sin métricas TRAIN todavía")
    axis.set_ylim(-.04, 1.14)
    axis.set_xlim(0, planned_outer + .5)
    axis.yaxis.set_major_formatter(PercentFormatter(1))

    axis = axes[1, 2]
    setup(axis, "F  Comparación final retenida", "", "J estricto · proporción")
    selected_step = result.get("best_step") if result else None
    labels = {"SFT-frozen": "Baseline\ncongelado",
              "RL-credit": f"RL seleccionado\npaso {selected_step}" if selected_step is not None else "RL\nseleccionado",
              "RL-credit-terminal": f"RL terminal\npaso {planned_outer}"}
    final_values = {}
    final_unnecessary_questions = {}
    friction_only = []
    if comparison:
        for method in labels:
            panel = comparison["panels"][method]
            if "strict" in panel:
                final_values[method] = panel["strict"]["metrics"]["J_strict"]
                if "unnecessary_questions" in panel["strict"]["metrics"]:
                    final_unnecessary_questions[method] = panel["strict"]["metrics"]["unnecessary_questions"]
            else:
                final_values[method] = panel["metrics"]["J_faithful"]
        bars = axis.bar(range(3), list(final_values.values()), color=["#788795", BLUE, ORANGE], width=.62, zorder=3)
        axis.set_xticks(range(3), list(labels.values()))
        axis.bar_label(bars, labels=[f"{value:.1%}" for value in final_values.values()], padding=5, fontsize=10)
        base_joint = final_values["SFT-frozen"]
        base_asks = final_unnecessary_questions.get("SFT-frozen")
        if base_asks is not None:
            friction_only = [method for method in ("RL-credit", "RL-credit-terminal")
                             if abs(final_values[method] - base_joint) < 1e-12
                             and method in final_unnecessary_questions
                             and final_unnecessary_questions[method] < base_asks - 1e-12]
        axis.text(.5, -.26, f"{comparison['evaluation_roots']} historias · "
                  f"{comparison['evaluation_episodes']} episodios por política", ha="center",
                  transform=axis.transAxes, fontsize=9, color="#566579")
        if friction_only:
            axis.text(.5, -.38, "J empatado y menos ASK innecesarias · menor fricción",
                      ha="center", transform=axis.transAxes, fontsize=8.5, color=PURPLE)
    else:
        pending(axis, "Evaluación retenida pendiente")
        axis.set_xticks([])
    axis.set_ylim(0, 1.14)
    axis.yaxis.set_major_formatter(PercentFormatter(1))

    fig.text(.065, .075, "J estricto exige C2 y FJS correctos, WRITE válido y cero escrituras sin sustento. "
             "Baseline y RL comparten prompts, decoder y episodios de evaluación.", fontsize=9, color="#405267")
    fig.text(.065, .045, "Las dos pasadas optimizan el mismo grupo y conservan sus log-probabilidades antiguas. "
             "Los rangos de ratio son observados, sin intervalo inferencial.", fontsize=9, color="#405267")
    closing_note = ("J empatado y menos ASK innecesarias significa menor fricción. "
                    "VALIDATION solo en checkpoints medidos; sin suavizado ni métricas imputadas.")
    if result and selected_step is not None:
        rejected = [str(row["step"]) for row in result.get("history", [])
                    if "validation" in row and row.get("selection_guard_passed") is False]
        closing_note = f"Selección en VALIDATION: paso {selected_step}. "
        if rejected:
            closing_note += f"Fuera de guardas: paso {', '.join(rejected)}. "
        if comparison:
            delta = 100 * (final_values["RL-credit"] - final_values["SFT-frozen"])
            closing_note += f"Reservado: ΔJ {delta:+.0f} pp"
            if all(method in final_unnecessary_questions for method in ("RL-credit", "SFT-frozen")):
                delta_ask = 100 * (final_unnecessary_questions["RL-credit"] - final_unnecessary_questions["SFT-frozen"])
                closing_note += f" · ΔASK innecesarias {delta_ask:+.0f} pp"
            closing_note += ". "
        closing_note += "Sin suavizado ni métricas imputadas."
    fig.text(.065, .015, closing_note, fontsize=9, color="#405267")
    output.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "svg"):
        path = output / f"targeted_rl_curves.{suffix}"
        fig.savefig(path, dpi=180)
        outputs.append(str(path.resolve()))
    plt.close(fig)
    metadata = dict(run=str(run.resolve()), snapshot_local=timestamp, observed_outer_updates=observed_outer,
                    observed_adam_steps=observed_adam, observed_groups=len(groups), observed_passes=len(passes),
                    sources=evidence.files, ignored_incomplete_final_lines=evidence.partial_lines,
                    validation=sorted(validation.items()), final_strict_values=final_values,
                    final_unnecessary_questions=final_unnecessary_questions, friction_only_methods=friction_only,
                    final_comparison_available=comparison is not None, smoothing=False, missing_metrics_imputed=False,
                    pass_identity=["outer_update", "optimization_pass"], outputs=outputs)
    (output / "targeted_rl_curves_sources.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: metadata[key] for key in ("observed_outer_updates", "observed_adam_steps", "final_comparison_available", "outputs")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_figure(args.run.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
