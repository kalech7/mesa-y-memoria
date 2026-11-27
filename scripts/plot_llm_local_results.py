"""Export all prespecified primary metrics and validation curves after completion."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.pilot.resolve()
    status = json.loads((root / 'run_status.json').read_text())
    if status.get('status') != 'COMPLETED_TECHNICAL_PILOT':
        raise SystemExit('Plot only a complete pilot; incomplete results are not presented as final')
    inputs = {name: sha(root / name) for name in ('final_metrics.json', 'training_metrics.json', 'config.json')}
    final = json.loads((root / 'final_metrics.json').read_text())
    training = json.loads((root / 'training_metrics.json').read_text())['runs']
    out = args.output or root / 'figures'
    out.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False})
    colors = {'B0': '#6b7280', 'B1': '#2563eb', 'B2': '#d97706', 'B3': '#059669', 'B4': '#7c3aed'}
    created = datetime.now(timezone.utc).isoformat()
    description = 'Local synthetic pilot. User-clustered 95% bootstrap intervals; training seeds averaged within user. ' + json.dumps(inputs)
    def save(fig, name):
        fig.savefig(out / f'{name}.png', dpi=180, bbox_inches='tight',
                    metadata={'Software': 'scripts/plot_llm_local_results.py',
                              'Description': description, 'Creation Time': created})
        fig.savefig(out / f'{name}.svg', bbox_inches='tight',
                    metadata={'Creator': 'scripts/plot_llm_local_results.py', 'Date': created,
                              'Description': description})
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7))
    methods = ['B0', 'B1', 'B2', 'B3', 'B4']
    for ax, field, title in zip(axes, ('return', 'error_rate', 'future_accuracy'),
                                ('Retorno longitudinal', 'Tasa de error S2/S3', 'Exactitud S3')):
        for i, method in enumerate(methods):
            metric = final['summaries'][method]['metrics'][field]
            mean, (low, high) = metric['mean'], metric['ci95']
            ax.vlines(i, low, high, color=colors[method], linewidth=2)
            ax.hlines([low, high], i-.09, i+.09, color=colors[method], linewidth=1.5)
            ax.scatter(i, mean, color=colors[method], s=45, zorder=3)
        ax.set(xticks=range(5), xticklabels=methods, title=title)
        ax.grid(axis='y', alpha=.2)
        if field != 'return':
            ax.set_ylim(-.03, 1.03)
        else:
            ax.axhline(0, color='#9ca3af', linewidth=.6)
    fig.suptitle('SPEC-LLM-LOCAL-001 · mismo LLM base, observación fijada y test congelado')
    fig.text(.5, .01, 'B0 base · B1 SFT · B2 SFT + RL · B3 SFT adicional · B4 reglas históricas\n'
             'Piloto sintético; los intervalos agrupan por usuario, no por fila ni por semilla.',
             ha='center', va='bottom', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .93))
    save(fig, 'primary_comparison')
    seeds = sorted({int(name.split('seed')[-1]) for name in training})
    fig, axes = plt.subplots(len(seeds), 2, figsize=(12, 3.6*len(seeds)), squeeze=False)
    for row, seed in enumerate(seeds):
        for method in ('B1', 'B2', 'B3'):
            run = training[f'{method}_seed{seed}']
            points = run['history']
            val = [p for p in points if 'validation_score' in p]
            axes[row, 0].plot([p['tokens_processed'] for p in val], [p['validation_score'] for p in val],
                              marker='o', color=colors[method], label=f"{method}; paso elegido {run['best_step']}")
            chosen = next(p for p in val if p['step'] == run['best_step'])
            axes[row, 0].scatter(chosen['tokens_processed'], chosen['validation_score'], marker='*',
                                 s=130, color=colors[method], edgecolor='black', linewidth=.5, zorder=5)
            loss = [p for p in points if 'supervised_loss' in p]
            if loss:
                axes[row, 1].plot([p['tokens_processed'] for p in loss], [p['supervised_loss'] for p in loss],
                                  color=colors[method], label=method, linewidth=1)
        axes[row, 0].set(title=f'Semilla {seed}: retorno de validación', ylabel='Retorno medio')
        axes[row, 1].set(title=f'Semilla {seed}: pérdida supervisada', ylabel='Entropía cruzada por token')
        for ax in axes[row]:
            ax.set_xlabel('Tokens de entrenamiento en cada etapa (prompt + respuesta)')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    fig.suptitle('La pérdida de entrenamiento y la calidad de las decisiones se evalúan por separado')
    fig.tight_layout(rect=(0, 0, 1, .97))
    save(fig, 'training_and_validation')
    manifest = {'created_at': created, 'source': str(Path(__file__).resolve()), 'source_sha256': sha(Path(__file__)),
                'input_sha256': inputs, 'primary_pilot': str(root),
                'packages': {name: importlib.metadata.version(name) for name in ('matplotlib', 'numpy')},
                'metric_selection': 'All prespecified return/error/S3 metrics; no favorable method omitted',
                'outputs': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / 'plot_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    main()
