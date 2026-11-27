"""Export observed SFT/GRPO telemetry without smoothing or metric imputation."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

COLORS = {"select": "#0072B2", "respond": "#D55E00", "write": "#009E73"}
BLUE = "#0072B2"
ORANGE = "#D55E00"
PURPLE = "#7A3E9D"


class Evidence:
    def __init__(self):
        self.files = {}
        self.partial_lines = []

    def read_bytes(self, path):
        path = Path(path)
        if not path.is_file():
            return None
        data = path.read_bytes()
        self.files[str(path.resolve())] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        return data

    def json(self, path):
        data = self.read_bytes(path)
        return json.loads(data) if data is not None else None

    def jsonl(self, path):
        data = self.read_bytes(path)
        if data is None:
            return []
        rows = []
        lines = data.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1 or line.endswith(b"\n"):
                    raise
                self.partial_lines.append(str(Path(path).resolve()))
        return rows


def validation_points(progress, training):
    completed = {row["step"]: row["metrics"] for row in progress
                 if row.get("stage") == "validation_completed"}
    for entry in (training or {}).get("history", []):
        if "validation" in entry:
            completed[entry["step"]] = entry["validation"]
    return sorted((step, metrics["J_strict"]) for step, metrics in completed.items() if "J_strict" in metrics)


def setup(axis, title, xlabel, ylabel):
    axis.set_title(title, loc="left", fontweight="bold", fontsize=12, pad=12)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#DDE2E8", linewidth=.7, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))


def pending(axis, text):
    axis.text(.5, .5, text, ha="center", va="center", transform=axis.transAxes,
              color="#606B78", fontsize=12, linespacing=1.5)


def plot_validation(axis, values, color=PURPLE, label="VALIDATION · greedy"):
    if not values:
        return
    x, y = zip(*values)
    axis.scatter(x, y, color=color, marker="D", s=54, edgecolor="white", linewidth=.8,
                 zorder=4, label=label)
    for step, value in values:
        axis.annotate(f"{value:.1%}", (step, value), xytext=(0, 9), textcoords="offset points",
                      ha="center", fontsize=9, color=color)


def make_figure(run, output):
    evidence = Evidence()
    plan = evidence.json(run / "PILOT_PLAN.json")
    if plan is None:
        raise FileNotFoundError(run / "PILOT_PLAN.json")
    sft_updates = evidence.jsonl(run / "sft/optimizer/updates.jsonl")
    sft_progress = evidence.jsonl(run / "sft/optimizer/progress.jsonl")
    sft_result = evidence.json(run / "sft/optimizer/training.json")
    rl_progress = evidence.jsonl(run / "rl/optimizer/progress.jsonl")
    rl_result = evidence.json(run / "rl/optimizer/training.json")
    groups = evidence.jsonl(run / "rl/optimizer/group_diagnostics.jsonl")
    rollout_metrics = evidence.jsonl(run / "rl/optimizer/training_metrics.jsonl")
    comparison = evidence.json(run / "COMPARISON.json")
    sft_validation = validation_points(sft_progress, sft_result)
    rl_validation = validation_points(rl_progress, rl_result)
    rl_updates = {row["step"]: row for row in rl_progress if row.get("stage") == "training"}
    for entry in (rl_result or {}).get("history", []):
        if "loss" in entry:
            rl_updates[entry["step"]] = entry
    observed_sft = max((row["step"] for row in sft_updates), default=0)
    observed_rl = max(rl_updates, default=0)
    timestamp = datetime.now(ZoneInfo("America/Guayaquil")).strftime("%d %b %Y %H:%M %Z")
    final_status = "evaluación retenida completa" if comparison else "evaluación retenida pendiente"

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10,
                         "legend.fontsize": 8.5, "svg.fonttype": "none", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.subplots_adjust(left=.065, right=.98, bottom=.23, top=.84, hspace=.72, wspace=.3)
    fig.suptitle("SFT y GRPO con evaluación estricta de memoria", x=.065, y=.972,
                 ha="left", fontsize=21, fontweight="bold", color="#172B42")
    fig.text(.065, .926,
             f"Qwen2.5-0.5B · semilla {plan['seed']} · SFT {observed_sft}/{plan['sft']['updates']} pasos · "
             f"RL {observed_rl}/{plan['rl']['updates']} pasos · {final_status}", fontsize=11, color="#47586B")
    fig.text(.065, .895, f"{run.name} · corte {timestamp}", fontsize=9, color="#607185")

    axis = axes[0, 0]
    setup(axis, "A  Pérdida supervisada", "Paso de optimizador SFT", "NLL media por token · escala log")
    if sft_updates:
        for phase, color in COLORS.items():
            rows = [row for row in sft_updates if row["phase"] == phase]
            axis.scatter([row["step"] for row in rows], [row["loss"] for row in rows],
                         s=22, color=color, alpha=.85, label=f"TRAIN · {phase.upper()}", zorder=3)
        axis.set_yscale("log")
        axis.legend(loc="upper center", bbox_to_anchor=(.5, -.24), ncol=3, frameon=False)
    else:
        pending(axis, "SFT pendiente")
    axis.set_xlim(-1, plan["sft"]["updates"] + 1)

    axis = axes[0, 1]
    setup(axis, "B  Objetivo GRPO", "Paso de optimizador RL", "Loss total media")
    if rl_updates:
        steps = sorted(rl_updates)
        axis.plot(steps, [rl_updates[step]["loss"] for step in steps], color=BLUE,
                  marker="o", markersize=4, linewidth=1.2, label="TRAIN · objetivo con KL")
        axis.axhline(0, linewidth=.7, color="#788592")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        axis.legend()
    else:
        pending(axis, "GRPO pendiente")
    axis.set_xlim(0, plan["rl"]["updates"] + .5)

    axis = axes[0, 2]
    setup(axis, "C  Recompensa observada por grupo", "Grupo RL de cuatro trayectorias", "Retorno con feedback y penalizaciones")
    if groups:
        x = [group["optimizer_update"] for group in groups]
        axis.fill_between(x, [min(group["returns"]) for group in groups],
                          [max(group["returns"]) for group in groups],
                          color=BLUE, alpha=.13, label="Mínimo–máximo observado")
        axis.plot(x, [mean(group["returns"]) for group in groups], color=BLUE,
                  marker="o", markersize=4, linewidth=1.2, label="TRAIN · media del grupo")
        axis.legend()
    else:
        pending(axis, "Sin grupos completos todavía")
    axis.set_xlim(0, plan["rl"]["updates"] + .5)

    axis = axes[1, 0]
    setup(axis, "D  Éxito estricto durante SFT", "Checkpoint SFT evaluado", "J estricto · proporción")
    if sft_validation:
        plot_validation(axis, sft_validation)
        axis.legend(loc="lower left")
    else:
        pending(axis, "Sin validación completa todavía")
    axis.set_ylim(-.04, 1.14)
    axis.set_xlim(-1, plan["sft"]["updates"] + 1)
    axis.yaxis.set_major_formatter(PercentFormatter(1))

    axis = axes[1, 1]
    setup(axis, "E  Éxito estricto durante RL", "Grupo TRAIN / checkpoint RL", "J estricto · proporción")
    actual_metrics = [row for row in rollout_metrics if "rollout/J_strict" in row["values"]]
    if actual_metrics:
        axis.plot([row["step"] for row in actual_metrics],
                  [row["values"]["rollout/J_strict"] for row in actual_metrics], color=ORANGE,
                  marker="o", markersize=4, alpha=.85, linewidth=1.1, label="TRAIN · 4 rollouts muestreados")
    plot_validation(axis, rl_validation)
    if actual_metrics or rl_validation:
        axis.legend(loc="upper center", bbox_to_anchor=(.5, -.24), ncol=2, frameon=False)
    else:
        pending(axis, "Sin métricas RL completas todavía")
    axis.set_xlim(-.5, plan["rl"]["updates"] + .5)
    axis.set_ylim(-.04, 1.14)
    axis.yaxis.set_major_formatter(PercentFormatter(1))

    axis = axes[1, 2]
    setup(axis, "F  Comparación final retenida", "", "J estricto · proporción")
    labels = {"SFT-start": "SFT\ninicial", "SFT-balanced": "SFT\nseleccionado",
              "RL-live": "RL\nseleccionado", "RL-live-terminal": "RL\nterminal"}
    if comparison:
        panels = comparison["panels"]
        training = comparison["training"]
        labels["SFT-balanced"] += f"\npaso {training['sft_selected_step']}"
        labels["RL-live"] += f"\npaso {training['rl_selected_step']}"
        labels["RL-live-terminal"] += f"\npaso {training['rl_updates']}"
        methods = [method for method in labels if method in panels]
        values = [panels[method]["metrics"]["J_faithful"] for method in methods]
        bars = axis.bar(range(len(methods)), values, color=["#75879A", BLUE, PURPLE, ORANGE][:len(methods)],
                        width=.62, zorder=3)
        axis.set_xticks(range(len(methods)), [labels[method] for method in methods])
        axis.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=5, fontsize=10)
        axis.text(.5, -.37, f"{comparison['evaluation_roots']} historias · "
                  f"{comparison['evaluation_episodes']} episodios por política",
                  ha="center", transform=axis.transAxes, fontsize=9, color="#566579")
    else:
        pending(axis, "Evaluación retenida pendiente\nLas barras aparecerán al completarse")
        axis.set_xticks([])
    axis.set_ylim(0, 1.14)
    axis.yaxis.set_major_formatter(PercentFormatter(1))

    fig.text(.065, .085,
             "J estricto exige C2 y FJS correctos, WRITE válido y cero escrituras sin sustento. "
             "La evaluación final lo registra como J_faithful.", fontsize=9, color="#405267")
    fig.text(.065, .055,
             "TRAIN y VALIDATION usan cohortes y decodificación distintas. "
             "Los puntos de validación se muestran solo donde se midieron. Sin suavizado.", fontsize=9, color="#405267")
    fig.text(.065, .025,
             "El rango de recompensa describe cuatro trayectorias, sin intervalo inferencial. "
             "Las pérdidas de SFT y RL optimizan objetivos distintos.", fontsize=9, color="#405267")
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for suffix in ("png", "svg"):
        path = output / f"training_curves.{suffix}"
        fig.savefig(path, dpi=180)
        files.append(str(path.resolve()))
    plt.close(fig)
    manifest = {"run": str(run.resolve()), "snapshot_local": timestamp,
                "sft_steps": observed_sft, "rl_steps": observed_rl,
                "sft_validation": sft_validation, "rl_validation": rl_validation,
                "reward_groups": len(groups), "training_metric_groups": len(actual_metrics),
                "final_comparison_available": comparison is not None,
                "source_files": evidence.files, "ignored_incomplete_final_lines": evidence.partial_lines,
                "outputs": files, "smoothing": False, "missing_metrics_imputed": False,
                "strict_metric": "C2 and FJS and valid WRITE and zero unsupported writes",
                "final_metric_key": "J_faithful"}
    (output / "training_curves_sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("sft_steps", "rl_steps", "reward_groups", "final_comparison_available", "outputs")},
                     ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_figure(args.run.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
