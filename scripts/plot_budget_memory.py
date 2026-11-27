"""Export six evidence-only panels for one bounded-memory RL run, including pending stages."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import fmean
from zoneinfo import ZoneInfo

try:
    from scripts.plot_live_rl import BLUE, ORANGE, PURPLE, Evidence, pending, setup
except ModuleNotFoundError:
    from plot_live_rl import BLUE, ORANGE, PURPLE, Evidence, pending, setup
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

GREEN = '#009E73'
GREY = '#788795'
METHODS = ('Instruct-frozen', 'RL-selected', 'FIFO', 'Random', 'Public-rules')
LABELS = ('Instruct\ncongelado', 'RL\nseleccionado', 'FIFO', 'Azar', 'Reglas\npúblicas')


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def metrics(value):
    if not isinstance(value, dict):
        return {}
    nested = value.get('metrics')
    return nested if isinstance(nested, dict) else value


def series(rows, key, *, nested=None):
    values = {}
    for row in rows:
        payload = row.get(nested, {}) if nested else row
        if number(row.get('step')) and isinstance(payload, dict) and number(payload.get(key)):
            values[row['step']] = payload[key]
    return sorted(values.items())


def line(axis, points, *, label, color, linestyle='-', marker='o'):
    if points:
        x, y = zip(*points)
        axis.plot(x, y, label=label, color=color, linestyle=linestyle, marker=marker,
                  markersize=3.8, linewidth=1.2)


def legend(axis, *other_axes):
    handles, labels = [], []
    for obj in (axis, *other_axes):
        hs, ls = obj.get_legend_handles_labels()
        handles.extend(hs); labels.extend(ls)
    if handles:
        axis.legend(handles, labels, fontsize=8, frameon=False,
                    loc='upper center', bbox_to_anchor=(.5, -.23))


def read_optimizer(evidence, run):
    choices = (run / 'optimizer', run / 'rl/optimizer')
    directory = next((path for path in choices if path.is_dir()), choices[0])
    return directory, evidence.jsonl(directory / 'progress.jsonl'), evidence.jsonl(directory / 'training_metrics.jsonl'), evidence.jsonl(directory / 'group_diagnostics.jsonl'), evidence.json(directory / 'training.json')


def make_figure(run, output, *, ask_cost=.10, write_cost=.02):
    run, output = Path(run).resolve(), Path(output).resolve()
    if not all(number(value) and value >= 0 for value in (ask_cost, write_cost)):
        raise ValueError('Action costs must be finite nonnegative numbers')
    evidence = Evidence()
    plans = [evidence.json(run / name) for name in ('PILOT_PLAN.json', 'FIT_PLAN.json', 'RL_PLAN.json', 'TRAIN_PLAN.json')]
    plan = next((item for item in reversed(plans) if item), {})
    pilot = evidence.json(run / 'PILOT_RESULT.json')
    directory, progress, rollouts, groups, training = read_optimizer(evidence, run)
    comparison = evidence.json(run / 'COMPARISON.json')
    training = training or (comparison or {}).get('training') or {}
    if not isinstance(training, dict):
        training = {}
    updates = [row for row in progress if row.get('stage') == 'training']
    updates += [row for row in training.get('history', []) if isinstance(row, dict) and 'loss' in row]
    validation = {}
    for row in progress:
        if row.get('stage') == 'validation_completed' and number(row.get('step')):
            validation[row['step']] = metrics(row.get('metrics', row.get('validation', {})))
    for row in training.get('history', []):
        if isinstance(row, dict) and number(row.get('step')) and isinstance(row.get('validation'), dict):
            validation[row['step']] = metrics(row['validation'])
    observed = max((step for step, _ in series(updates, 'loss')), default=0)
    planned = next((value for value in (plan.get('updates'), plan.get('max_updates'),
                    plan.get('rl', {}).get('updates') if isinstance(plan.get('rl'), dict) else None,
                    training.get('updates'), training.get('plan', {}).get('updates') if isinstance(training.get('plan'), dict) else None)
                    if number(value)), None)
    final = {}
    panels = (comparison or {}).get('panels', {})
    for method in METHODS:
        value = metrics(panels.get(method, {})).get('net_return')
        if number(value):
            final[method] = value
    final_complete = (len(final) == len(METHODS)
                      and (comparison or {}).get('status') in {'completed', 'executed', 'complete', 'passed'})
    local_time = datetime.now(ZoneInfo('America/Guayaquil')).strftime('%Y-%m-%d %H:%M %Z')
    current_stage = progress[-1].get('stage', 'desconocida') if progress else 'sin diario de optimización'
    status = ('comparación final registrada' if final_complete else
              'comparación final parcial' if final else 'comparación final pendiente')
    step_label = f'{observed}/{int(planned)}' if planned is not None else str(observed)
    seed = plan.get('seed', training.get('seed', 'pendiente'))

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'legend.fontsize': 8,
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.subplots_adjust(left=.065, right=.96, bottom=.22, top=.84, wspace=.39, hspace=.79)
    fig.suptitle('RL con memoria limitada y recompensa demorada', x=.065, y=.972,
                 ha='left', fontsize=21, fontweight='bold', color='#172B42')
    fig.text(.065, .927, f'Instruct del proveedor · sin SFT local adicional · semilla {seed} · '
             f'actualizaciones observadas {step_label} · {status}', fontsize=10.5, color='#405267')
    fig.text(.065, .896, f'{run.name} · {current_stage} · corte {local_time}', fontsize=9, color='#607185')

    axis = axes[0, 0]
    setup(axis, 'A  Objetivo de optimización y KL', 'Actualización de optimizador', 'Pérdida media por decisión')
    loss, policy_loss, kl = (series(updates, key) for key in ('loss', 'policy_loss', 'kl'))
    line(axis, loss, label='Loss total', color=BLUE)
    line(axis, policy_loss, label='Pérdida de política', color=ORANGE, linestyle='--')
    twin = None
    if kl:
        twin = axis.twinx()
        line(twin, kl, label='KL categórica exacta · eje derecho', color=GREEN, marker='.')
        twin.set_ylabel('KL frente al Instruct inicial', color=GREEN, fontsize=9)
        twin.spines['top'].set_visible(False)
        twin.tick_params(axis='y', labelsize=8)
    if not loss and not policy_loss and not kl:
        pending(axis, 'Sin actualizaciones completas')
    legend(axis, *([twin] if twin else []))
    axis.axhline(0, color=GREY, linewidth=.6)

    axis = axes[0, 1]
    setup(axis, 'B  Retornos observados en TRAIN', 'Grupo de cuatro trayectorias', 'Retorno neto medio por episodio')
    environment_returns = series(rollouts, 'rollout/net_return', nested='values')
    returns = series(rollouts, 'rollout/AI_return', nested='values')
    ranges = []
    for row in groups:
        step = row.get('step', row.get('optimizer_update', row.get('update')))
        values = row.get('returns')
        if number(step) and isinstance(values, list) and values and all(number(x) for x in values):
            ranges.append((step, min(values), max(values), fmean(values)))
    if ranges:
        ranges.sort()
        axis.errorbar([row[0] for row in ranges], [row[3] for row in ranges],
                      yerr=([row[3]-row[1] for row in ranges], [row[2]-row[3] for row in ranges]),
                      fmt='none', color=BLUE, alpha=.25, capsize=3, linewidth=1.5,
                      label='Rango observado del grupo · IA')
    if returns:
        line(axis, returns, label='TRAIN · recompensa con juez IA', color=BLUE)
    elif ranges:
        line(axis, [(row[0], row[3]) for row in ranges], label='TRAIN · recompensa IA del grupo', color=BLUE)
    if environment_returns:
        line(axis, environment_returns, label='TRAIN · verificador del entorno', color=GREY, linestyle='--', marker='.')
    if not returns and not ranges and not environment_returns:
        pending(axis, 'Sin grupos TRAIN completos')
    legend(axis)

    axis = axes[0, 2]
    setup(axis, 'C  Utilidad en VALIDATION', 'Checkpoint evaluado', 'Retorno neto · greedy')
    valid_return = sorted((step, row['net_return']) for step, row in validation.items() if number(row.get('net_return')))
    if valid_return:
        axis.scatter(*zip(*valid_return), marker='D', s=45, color=PURPLE, zorder=4, label='VALIDATION medida')
        baseline = dict(valid_return).get(0)
        if baseline is not None:
            axis.axhline(baseline, color=GREY, linewidth=1, linestyle='--', label='Instruct inicial · checkpoint 0')
        legend(axis)
    else:
        pending(axis, 'Validación inicial pendiente')

    axis = axes[1, 0]
    setup(axis, 'D  Consultas correctas y sustentadas', 'Grupo TRAIN / checkpoint', 'Proporción de consultas')
    train_query = series(rollouts, 'rollout/query_success', nested='values')
    train_joint = series(rollouts, 'rollout/joint_success', nested='values')
    valid_query = sorted((step, row['query_success']) for step, row in validation.items() if number(row.get('query_success')))
    line(axis, train_query, label='TRAIN · éxito por consulta', color=BLUE)
    line(axis, train_joint, label='TRAIN · ambas consultas correctas', color=ORANGE, linestyle='--', marker='.')
    if valid_query:
        axis.scatter(*zip(*valid_query), marker='D', s=43, color=PURPLE, zorder=4, label='VALIDATION · éxito por consulta')
    if not train_query and not train_joint and not valid_query:
        pending(axis, 'Sin consultas evaluadas todavía')
    legend(axis)
    axis.set_ylim(-.04, 1.08)
    axis.yaxis.set_major_formatter(PercentFormatter(1))

    axis = axes[1, 1]
    setup(axis, 'E  Costes de preguntas y memoria', 'Grupo TRAIN', 'Penalización media por episodio')
    asks = series(rollouts, 'rollout/ask_count', nested='values')
    writes = series(rollouts, 'rollout/write_count', nested='values')
    ask_points = [(step, value * ask_cost) for step, value in asks]
    write_points = [(step, value * write_cost) for step, value in writes]
    line(axis, ask_points, label=f'ASK × {ask_cost:g}', color=ORANGE)
    line(axis, write_points, label=f'STORE × {write_cost:g}', color=GREEN)
    if not asks and not writes:
        pending(axis, 'Sin costes observados en TRAIN')
    legend(axis)
    axis.set_ylim(bottom=0)

    axis = axes[1, 2]
    setup(axis, 'F  Comparación final con igual presupuesto', '', 'Retorno neto · evaluación reservada')
    if final:
        positions = [index for index, name in enumerate(METHODS) if name in final]
        colors = [GREY, BLUE, GREEN, ORANGE, PURPLE]
        bars = axis.bar(positions, [final[METHODS[index]] for index in positions],
                        color=[colors[index] for index in positions], width=.6, zorder=3)
        axis.bar_label(bars, labels=[f'{final[METHODS[index]]:+.3f}' for index in positions], padding=5, fontsize=9)
        for index, method in enumerate(METHODS):
            if method not in final:
                axis.text(index, .5, 'Pendiente', ha='center', va='center', fontsize=8, color=GREY,
                          transform=axis.get_xaxis_transform(), rotation=90)
        method_labels = list(LABELS)
        selected_step = (comparison or {}).get('selected_step', training.get('best_step'))
        if selected_step is not None:
            method_labels[1] = f'RL elegido\npaso {selected_step}'
        axis.set_xticks(range(len(METHODS)), method_labels, fontsize=8.5)
        axis.axhline(0, color=GREY, linewidth=.7)
        axis.margins(y=.22)
        if not final_complete:
            axis.text(.5, -.29, 'Comparación parcial: faltan etapas o paneles.', ha='center',
                      transform=axis.transAxes, fontsize=8, color=PURPLE)
    else:
        pending(axis, 'Evaluación reservada pendiente')
        axis.set_xticks([])

    # A single checkpoint at step zero must not produce fractional step ticks.
    latest_step = max([observed, *validation.keys(),
                       *(row['step'] for row in rollouts if number(row.get('step')))], default=0)
    for axis in list(axes.flat)[:5]:
        axis.set_xlim(-.4, max(1, latest_step) + .4)

    last_std = series(rollouts, 'rollout/task_std', nested='values')
    std_note = f'Última σ de utilidad TRAIN: {last_std[-1][1]:.3g}. ' if last_std else ''
    fig.text(.065, .095, 'TRAIN separa recompensa con juez IA y verificador del entorno; VALIDATION usa inferencia greedy en checkpoints medidos.',
             fontsize=9, color='#405267')
    fig.text(.065, .066, 'Memoria de dos slots, lector congelado y misma interfaz entre brazos. Loss menor no demuestra mayor utilidad.',
             fontsize=9, color='#405267')
    fig.text(.065, .037, std_note + (f"Selección VALIDATION: paso {(comparison or {}).get('selected_step', training.get('best_step'))}. " if (comparison or {}).get('selected_step', training.get('best_step')) is not None else '') + 'Sin suavizado ni métricas imputadas.', fontsize=9, color='#405267')
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for suffix in ('png', 'svg'):
        path = output / f'budget_memory_curves.{suffix}'
        fig.savefig(path, dpi=180)
        artifacts.append(str(path))
    plt.close(fig)
    summary = {
        'run': str(run), 'run_exists': run.exists(), 'optimizer_directory': str(directory),
        'snapshot_local': local_time, 'stage': current_stage, 'pilot_status': (pilot or {}).get('status'),
        'observed_optimizer_updates': observed, 'planned_updates': planned, 'seed': seed,
        'training_ai_points': len(returns), 'training_environment_points': len(environment_returns),
        'selected_step': (comparison or {}).get('selected_step', training.get('best_step')), 'validation': sorted(validation.items()),
        'final_status': (comparison or {}).get('status'), 'final_comparison_complete': final_complete,
        'final_net_returns': final, 'sources': evidence.files,
        'ignored_incomplete_final_lines': evidence.partial_lines,
        'cost_transformation': {'ask_count_multiplier': ask_cost, 'write_count_multiplier': write_cost},
        'smoothing': False, 'missing_metrics_imputed': False,
        'one_seed_descriptive_comparison': True, 'outputs': artifacts,
    }
    path = output / 'budget_memory_curves_sources.json'
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ask-cost', type=float, default=.10)
    parser.add_argument('--write-cost', type=float, default=.02)
    args = parser.parse_args()
    result = make_figure(args.run, args.output, ask_cost=args.ask_cost, write_cost=args.write_cost)
    print(json.dumps({key: result[key] for key in ('observed_optimizer_updates', 'final_comparison_complete', 'outputs')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
