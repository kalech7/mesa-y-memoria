"""Training curves and budget tables from actual immutable trainer summaries."""
import csv
import json
from pathlib import Path
import subprocess
import sys


def _load(path):
    return json.loads(Path(path).read_text())


def plot_curves(source, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(Path(source).open()))
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, method in zip(axes.flat, ("B1", "B2-L", "B3", "B2-I")):
        for seed in (101, 202, 303):
            points = [r for r in rows if r["method"] == method and int(r["seed"]) == seed and r["validation_utility"] != ""]
            if points:
                axis.plot([int(r["step"]) for r in points], [float(r["validation_utility"]) for r in points], marker="o", label=str(seed))
        axis.set_title(method)
        axis.set_xlabel("Actualizaciones del método")
        axis.set_ylabel("Utilidad longitudinal en validación")
        axis.grid(alpha=.25)
        axis.legend(title="Semilla")
    figure.suptitle("Ejecución exploratoria: checkpoints reales, incluido paso cero")
    figure.savefig(Path(destination) / "CURVAS_VALIDACION.svg")
    figure.savefig(Path(destination) / "CURVAS_VALIDACION.png", dpi=180)
    plt.close(figure)


def build_training_artifacts(metadata, destination):
    destination = Path(destination)
    if not metadata.get("SFT_campaign") or not metadata.get("continuations_campaign"):
        return {"status": "not_executed", "reason": "No executed training campaigns supplied"}
    sft, continuation = _load(metadata["SFT_campaign"]), _load(metadata["continuations_campaign"])
    runs = []
    for seed in (101, 202, 303):
        runs.append(("B1", seed, sft[str(seed)]))
        runs.extend((method, seed, continuation[str(seed)][method]) for method in ("B2-L", "B3", "B2-I"))
    curve_path = destination / "CURVAS_ENTRENAMIENTO.csv"
    with curve_path.open("w", newline="") as stream:
        fields = ["method", "seed", "step", "loss", "grad_norm", "validation_utility", "validation_FJS", "selected"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method, seed, run in runs:
            for entry in run.get("history", []):
                validation = entry.get("validation", {})
                writer.writerow({"method": method, "seed": seed, "step": entry["step"], "loss": entry.get("loss"),
                                 "grad_norm": entry.get("grad_norm"), "validation_utility": validation.get("utility"),
                                 "validation_FJS": validation.get("future_success"), "selected": int(entry["step"] == run.get("best_step"))})
    budget_rows = []
    for method, seed, run in runs:
        counter = run.get("resource_counters", {})
        budget_rows.append({"method": method, "seed": seed, "status": run.get("status"),
            "selected_step": run.get("best_step"), "final_updates": counter.get("updates"),
            "elapsed_seconds": counter.get("elapsed_seconds"), "allocated_seconds": counter.get("allocated_seconds"),
            "budget_overrun_seconds": counter.get("budget_overrun_seconds"), "input_tokens": counter.get("input_tokens"),
            "trained_tokens": counter.get("trained_tokens"), "generated_tokens": counter.get("generated_tokens"),
            "episodes": counter.get("episodes"), "questions": counter.get("questions"),
            "validation_evaluations": counter.get("validation_evaluations"),
            "stop_reason": run.get("stop_reason"), "initial_checkpoint_selected": run.get("best_step") == 0,
            "backbone_unchanged": run.get("frozen_before") == run.get("frozen_after"),
            "reference_unchanged": run.get("reference_before") == run.get("reference_after") if method in {"B2-L", "B2-I"} else None})
    with (destination / "PRESUPUESTOS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(budget_rows[0]))
        writer.writeheader()
        writer.writerows(budget_rows)
    with (destination / "INFORME_SFT_RL.md").open("a") as stream:
        stream.write("\n## Entrenamiento realmente ejecutado\n\n| Método | Semilla | Paso seleccionado | Actualizaciones finales | Segundos |\n|---|---:|---:|---:|---:|\n")
        for row in budget_rows:
            elapsed = f"{row['elapsed_seconds']:.1f}" if row["elapsed_seconds"] is not None else "N/A"
            stream.write(f"| {row['method']} | {row['seed']} | {row['selected_step']} | {row['final_updates']} | {elapsed} |\n")
        stream.write("\nLas curvas contienen las evaluaciones reales, incluidos el inicial, el seleccionado y el final. "
                     "Las unidades de actualización difieren entre SFT y RL; no se interpretan como FLOPs iguales. "
                     "Seleccionar el paso cero no demuestra mejora por las actualizaciones. PRESUPUESTOS.csv conserva consumo y exceso del límite.\n")
    plotter = Path(__file__).resolve().parents[3] / ".venv-llm-plots/bin/python"
    plotted = {"status": "not_executed", "reason": "Local plotting runtime missing"}
    if plotter.is_file():
        result = subprocess.run([str(plotter), "-m", "research.src.analysis.training_artifacts", "--plot-only", str(curve_path), str(destination)],
                                cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, timeout=120)
        plotted = {"status": "executed" if result.returncode == 0 else "failed", "returncode": result.returncode,
                   "error": result.stderr[-2000:] if result.returncode else None}
    return {"status": "executed", "runs": budget_rows, "curves_csv": str(curve_path), "plot": plotted}


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--plot-only":
    plot_curves(sys.argv[2], sys.argv[3])
