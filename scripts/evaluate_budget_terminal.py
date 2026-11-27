"""Prospective descriptive amendment: report the fixed terminal RL checkpoint.

This never changes the original five-arm analysis or validation selection. The
amendment is registered after validation16, before training32 finishes and before
reserved inference. It is an extra descriptive comparison, not confirmatory.
"""
from pathlib import Path
import argparse

from research.budget_memory_experiment import (
    read, rows, fresh_reader, evaluate_policy, release, paired, check_sources)
from research.budget_memory_policy import BudgetMemoryPolicy
from research.src.artifacts import write_json, file_hash, now


def register(run):
    plan_path = run / 'TERMINAL_DIAGNOSTIC_PLAN.json'
    if plan_path.exists() or (run / 'SELECTION.json').exists() or (run / 'COMPARISON.json').exists():
        raise ValueError('Register the amendment once, before reserved selection/inference')
    training = read(run / 'TRAIN_PLAN.json')
    journal = rows(run / 'optimizer/progress.jsonl')
    completed = [row['step'] for row in journal if row['stage'] == 'training']
    if max(completed) >= training['max_updates'] or training['max_updates'] != 32:
        raise ValueError('The terminal32 checkpoint must still be prospective')
    validation = read(run / 'validation/step_000016/summary.json')
    plan = dict(created_at=now(), scope='Late descriptive amendment, after VAL16 and before terminal32/test',
                primary_analysis_unchanged=True, use_for_selection_or_tuning=False,
                fixed_terminal_step=32, seed=101, new_optimizer_steps=0, sft_updates=0,
                registration_after_training_step=max(completed),
                motivation='Show the actually trained terminal policy even if validation selects step0',
                validation16_sha256=file_hash(run / 'validation/step_000016/summary.json'),
                validation16_metrics=validation['metrics'],
                data_sha256=training['data_sha256'], source_sha256=training['source_sha256'],
                base_artifacts_sha256=training['base_artifacts_sha256'],
                reader_template_profile=training['reader_template_profile'],
                metrics='Same frozen score_trace, macro families/roots, paired against Instruct-frozen',
                reuse_main_panel_if_terminal_selected=True,
                extra_comparison_on_same_reserved_bank=True,
                script_sha256=file_hash(__file__))
    write_json(plan_path, plan)
    (run / 'source_snapshot/scripts').mkdir(parents=True, exist_ok=True)
    (run / 'source_snapshot/scripts/evaluate_budget_terminal.py').write_bytes(Path(__file__).read_bytes())
    print('registered_terminal_diagnostic', file_hash(plan_path), flush=True)


def evaluate(run):
    plan_path = run / 'TERMINAL_DIAGNOSTIC_PLAN.json'
    plan = read(plan_path)
    if file_hash(__file__) != plan['script_sha256'] or (run / 'TERMINAL_DIAGNOSTIC.json').exists():
        raise ValueError('Amendment source changed or diagnostic already completed')
    check_sources(plan)
    comparison = read(run / 'COMPARISON.json')
    selection = read(run / 'SELECTION.json')
    selection_sha = file_hash(run / 'SELECTION.json')
    if selection['selected_step'] == 32:
        panel = comparison['panels']['RL-selected']
        reused = True
    else:
        manifest_path = next(Path(path) for path in plan['data_sha256'] if path.endswith('MANIFEST.json'))
        bank_path = read(manifest_path)['files']['evaluation']['path']
        reader = fresh_reader()
        actor = BudgetMemoryPolicy.from_checkpoint(run / 'optimizer/step_000032', trainable=False, device='cuda')
        try:
            panel = evaluate_policy(actor, reader, rows(bank_path), run / 'evaluation/terminal_diagnostic',
                                    method='RL-terminal32-descriptive')
        finally:
            release(actor, reader)
        reused = False
    if selection_sha != file_hash(run / 'SELECTION.json'):
        raise ValueError('Original selection changed during diagnostic')
    result = dict(status='completed', analysis_role='late_descriptive_only',
                  plan_sha256=file_hash(plan_path), selection_sha256=selection_sha,
                  primary_selection_unchanged=True, reused_main_panel=reused, panel=panel,
                  paired_vs_Instruct=paired(comparison['panels']['Instruct-frozen'], panel))
    write_json(run / 'TERMINAL_DIAGNOSTIC.json', result)
    print('terminal_diagnostic_completed', panel['metrics'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    (register if args.register else evaluate)(args.run.resolve())


if __name__ == '__main__':
    main()
