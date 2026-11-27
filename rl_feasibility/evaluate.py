"""Reproducible offline SPEC-RL-002 runner. Training is conditional on Gates 1/2.

Run with ``python -m rl_feasibility.evaluate --output NEW_DIRECTORY``.
Policy selection uses validation only; test is evaluated after freeze.json exists.
"""
import argparse
import csv
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
import hashlib
from html import escape
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

from rl_feasibility.statistics import bootstrap_mean, mean, paired_difference

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf8')


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows), encoding='utf8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rollout(scenario, policy, condition='ORACLE_STRUCTURED', reward_config=None, first_action=None):
    from rl_feasibility.environment import FoodPreferenceEnv, diagnostic_metrics
    env = FoodPreferenceEnv(scenario, condition=condition, reward_config=reward_config)
    obs, info = env.reset(seed=719)
    total, trace = 0., []
    started = perf_counter()
    for step in range(4):
        # The only argument passed into a policy is a detached public observation.
        action = first_action if step == 0 and first_action is not None else policy.select_action(deepcopy(obs))
        following, reward, terminated, truncated, info = env.step(action)
        trace.append({'observation': obs, 'action': action, 'reward': reward,
                      'next_observation': following, 'terminated': terminated, 'truncated': truncated})
        total += reward
        obs = following
        if terminated or truncated:
            break
    metrics = diagnostic_metrics(env)
    return {'scenario_id': scenario['scenario_id'], 'user_id': scenario['user_id'],
            'category': scenario['category'], 'condition': condition, 'policy': policy.name,
            'return': total, 'elapsed_ms': (perf_counter() - started) * 1000,
            'actions': [step['action'] for step in trace], 'trace': trace, **metrics}


def evaluate_policy(scenarios, policy, condition='ORACLE_STRUCTURED', reward_config=None):
    # Errors deliberately propagate: no silent relabeling or denominator reduction.
    return [rollout(scenario, policy, condition, reward_config) for scenario in scenarios]


def summarize(rows, bootstrap=1000):
    numeric = sorted({key for row in rows for key, value in row.items()
                      if isinstance(value, (bool, float, int))})
    metrics = {'episodes': len(rows), 'return': bootstrap_mean([r['return'] for r in rows], bootstrap),
               'means': {key: mean([float(r[key]) for r in rows if isinstance(r.get(key), (bool, float, int))])
                         for key in numeric},
               'action_counts': dict(Counter(a for row in rows for a in row['actions']))}
    for numerator, denominator, name in (
        ('useful_clarifications', 'clarifications', 'clarification_quality'),
        ('unnecessary_clarifications', 'clarifications', 'unnecessary_clarification_rate'),
        ('correct_conflict_preservation', 'decision_unresolvable', 'unresolvable_preservation_recall'),
        ('restriction_violation', 'has_restriction', 'restriction_violation_rate_exposed')):
        num = sum(float(r.get(numerator, 0)) for r in rows if r.get(denominator, 0))
        den = sum(float(r.get(denominator, 0)) for r in rows)
        metrics[name] = {'numerator': num, 'denominator': den, 'value': num / den if den else None}
    num = sum(bool(r.get('necessary_clarification_omitted')) for r in rows)
    den = sum(bool(r.get('necessary_clarification_required')) for r in rows)
    metrics['necessary_clarifications_omitted'] = {'numerator': num, 'denominator': den,
                                                 'value': num / den if den else None}
    preserve_count = sum(r.get('selected_action') == 'PRESERVE_CONFLICT' for r in rows)
    preserve_correct = sum(bool(r.get('correct_conflict_preservation')) for r in rows)
    metrics['preserve_action_precision'] = {'numerator': preserve_correct, 'denominator': preserve_count,
                                           'value': preserve_correct / preserve_count if preserve_count else None}
    groups = defaultdict(list)
    for row in rows:
        groups[row['category']].append(row)
    metrics['by_category'] = {
        category: {'episodes': len(group), 'return': bootstrap_mean([r['return'] for r in group], bootstrap),
                   'means': {key: mean([float(r[key]) for r in group if isinstance(r.get(key), (bool, float, int))])
                             for key in numeric}}
        for category, group in sorted(groups.items())}
    return metrics


def compare_headroom(results, selected=None, bootstrap=1000):
    names = [name for name in results if name != 'STRONG_REFERENCE']
    best = selected or max(names, key=lambda name: mean(row['return'] for row in results[name]))
    reference = results['STRONG_REFERENCE']
    contrasts = {}
    for name in names:
        keys = ('return', 'decision_correct', 'clarifications', 'evidence_supported',
                'unsupported_answer', 'correct_conflict_preservation')
        contrasts[name] = {key: paired_difference(reference, results[name], key, bootstrap)
                           for key in keys if key in reference[0] and key in results[name][0]}
    gap = contrasts[best]['return']
    gate = 'PASS' if gap['mean'] >= .10 and gap['ci95'][0] > 0 else (
        'NO_MEANINGFUL_MARGIN' if gap['ci95'][1] < .10 else 'INCONCLUSIVE')
    return {'best_baseline_selected_on_validation': best, 'minimum_practical_difference': .10,
            'reference': 'STRONG_REFERENCE', 'headroom': gap, 'gate_2': gate, 'contrasts': contrasts,
            'definition': 'reference minus best conventional baseline, including supervised',
            'interpretation': 'Finite synthetic benchmark; reference is a visible-only rule, not a ceiling.'}


def sequentiality_analysis(scenarios, count=120):
    from rl_feasibility.baselines import StrongReference
    from rl_feasibility.environment import public_support
    reference = StrongReference()
    questions = ['CLARIFY_DAY', 'CLARIFY_MEAL', 'CLARIFY_COMPANY', 'CLARIFY_PLACE', 'CLARIFY_CURRENT_PREFERENCE']
    rows, examples = [], []
    for scenario in scenarios[:count]:
        branches = {a: rollout(scenario, reference, first_action=a) for a in questions}
        # Terminal alternatives provide an actual counterfactual, not just a cost.
        from rl_feasibility.baselines import NeverAsk
        immediate = rollout(scenario, NeverAsk())
        initial = immediate['trace'][0]['observation']
        initial_support = public_support(initial)
        changes = []
        for action, row in branches.items():
            after = row['trace'][0]['next_observation']
            changes.append({'action': action, 'history_changed': after.get('messages') != initial.get('messages'),
                            'support_changed': public_support(after) != initial_support,
                            'return': row['return'], 'decision_correct': row.get('decision_correct'),
                            'actions': row['actions']})
        different_quality = len({(v.get('decision_correct'), v.get('evidence_supported')) for v in branches.values()}) > 1
        correct_after = any(v.get('decision_correct') and not immediate.get('decision_correct') for v in branches.values())
        rows.append({'scenario_id': scenario['scenario_id'], 'category': scenario['category'],
                     'immediate_return': immediate['return'], 'immediate_correct': immediate.get('decision_correct'),
                     'different_terminal_quality_by_question': different_quality,
                     'clarification_enables_correct_decision': correct_after,
                     'asking_can_cost_more_than_immediate': any(v['return'] < immediate['return'] for v in branches.values()),
                     'branches': changes})
        if len(examples) < 12 and (different_quality or correct_after or scenario['category'].endswith('UNRESOLVABLE')):
            examples.append({'scenario_id': scenario['scenario_id'], 'category': scenario['category'],
                             'immediate': immediate['trace'], 'branches': {a: r['trace'] for a, r in branches.items()}})
    return {'states': len(rows), 'initial_randomness_controlled': True,
            'counts': {key: sum(bool(r[key]) for r in rows) for key in (
                'different_terminal_quality_by_question', 'clarification_enables_correct_decision',
                'asking_can_cost_more_than_immediate')}, 'per_state': rows, 'examples': examples,
            'fixed_penalty_alone_not_evidence': True}


def save_policy_results(output, prefix, scenarios, policies, condition, bootstrap=1000, reward_config=None):
    all_rows, metrics = {}, {}
    for name, policy in policies.items():
        rows = evaluate_policy(scenarios, policy, condition, reward_config)
        all_rows[name] = rows
        metrics[name] = summarize(rows, bootstrap)
        write_jsonl(output / f'{prefix}_{name.lower()}_episodes.jsonl', rows)
    return all_rows, metrics


def data_audit(partitions, policies=None):
    """Measure overlap rather than equating new user IDs with new latent worlds."""
    def signature(scenario):
        records = [{key: record.get(key) for key in ('food_item', 'polarity', 'day_scope', 'meal_scope',
                   'context', 'source', 'stability', 'valid_from', 'valid_to')}
                   for record in scenario['oracle_preferences']]
        restrictions = [{key: record.get(key) for key in ('food_item', 'restriction_type', 'certainty')}
                        for record in scenario['oracle_restrictions']]
        return json.dumps([records, restrictions, scenario['latent']], sort_keys=True)
    keys = {name: {signature(s) for s in cases} for name, cases in partitions.items()}
    people = {name: {s['user_id'] for s in cases} for name, cases in partitions.items()}
    result = {'partitions': {name: {'episodes': len(cases), 'unique_latent_signatures': len(keys[name]),
                'unique_users': len(people[name]), 'category_counts': dict(Counter(s['category'] for s in cases))}
                for name, cases in partitions.items()}, 'overlap': {}}
    for left in partitions:
        for right in partitions:
            if left >= right:
                continue
            result['overlap'][left + '/' + right] = {'users': len(people[left] & people[right]),
                'latent_signatures': len(keys[left] & keys[right])}
    result['limitations'] = 'Four-food finite world; exact latent/profile repetitions are reported, not hidden. IDs do not establish semantic independence.'
    if policies:
        from rl_feasibility.environment import FoodPreferenceEnv
        from rl_feasibility.baselines import feature_key
        train_keys = {feature_key(FoodPreferenceEnv(s).reset()[0]) for s in partitions['train']}
        result['initial_feature_key_overlap_with_train'] = {
            name: {'shared': sum(feature_key(FoodPreferenceEnv(s).reset()[0]) in train_keys for s in cases), 'n': len(cases)}
            for name, cases in partitions.items() if name != 'train'}
        category_actions = defaultdict(Counter)
        for scenario in partitions['train']:
            observation = FoodPreferenceEnv(scenario).reset()[0]
            category_actions[scenario['category']][policies['STRONG_REFERENCE'].select_action(observation)] += 1
        result['category_initial_reference_action_contingency'] = {k: dict(v) for k, v in category_actions.items()}
        result['correlation_control'] = 'Category is never a policy input. Strong action-category associations remain in this authored finite benchmark; they limit task diversity.'
    return result


def write_charts(output, metrics, training):
    names = list(metrics)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="460" viewBox="0 0 1000 460">',
             '<rect width="1000" height="460" fill="white"/>',
             '<g font-family="sans-serif" fill="#17243a"><text x="28" y="32" font-size="21">Retorno en test: episodios idénticos, ORACLE-STRUCTURED</text>',
             '<text x="28" y="58" font-size="13">Media e IC95 bootstrap por episodio; reglas y política supervisada. Sin validación humana.</text>']
    # Axis maps return [-1,1] to horizontal coordinates [320,900].
    x = lambda value: 320 + (value + 1) * 290
    for tick in (-1, -.5, 0, .5, 1):
        parts.append(f'<path d="M{x(tick)} 80 V400" stroke="#dddddd"/><text x="{x(tick)-10}" y="425" font-size="12">{tick:g}</text>')
    for i, name in enumerate(names):
        y = 100 + i * 47
        value = metrics[name]['return']['mean']
        lo, hi = metrics[name]['return']['ci95']
        parts.append(f'<text x="28" y="{y+5}" font-size="14">{escape(name)}</text><path d="M{x(lo)} {y} H{x(hi)}" stroke="#315c99" stroke-width="3"/><circle cx="{x(value)}" cy="{y}" r="5" fill="#213e70"/><text x="915" y="{y+5}" font-size="13">{value:.3f}</text>')
    parts.append('</g></svg>')
    (output / 'baseline_returns.svg').write_text(''.join(parts))
    if not training['executed']:
        (output / 'learning_status.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" width="960" height="180"><rect width="960" height="180" fill="#f4f6fa"/><g font-family="sans-serif" fill="#17243a"><text x="28" y="48" font-size="24">No se ejecutó entrenamiento RL</text><text x="28" y="88" font-size="17">Gate 2 no acreditó margen frente al mejor baseline convencional.</text><text x="28" y="124" font-size="15">No hay una curva de aprendizaje empírica que presentar.</text></g></svg>')
    else:
        lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="480"><rect width="960" height="480" fill="white"/><text x="30" y="30" font-family="sans-serif" font-size="22">Curvas de entrenamiento Q-learning (bloques de 100 episodios)</text>']
        for i, (seed, row) in enumerate(training['seeds'].items()):
            blocks = row['selected']['blocks']
            points = ' '.join(f"{60 + b['episodes']/max(1,blocks[-1]['episodes'])*840},{390-(b['mean_training_return']+2)*95}" for b in blocks)
            color = ['#2456a2', '#c25435', '#228555'][i % 3]
            lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/><text x="{80+i*260}" y="450" fill="{color}" font-family="sans-serif">Semilla {seed}</text>')
        lines.append('</svg>')
        (output / 'learning_curves.svg').write_text(''.join(lines))


def run_experiment(output, train_n=2000, validation_n=400, test_n=600, bootstrap=1000, run_tests=True, data_seed_offset=0):
    from rl_feasibility.scenario_generator import generate_scenarios
    from rl_feasibility.baselines import make_baselines, train_supervised
    from rl_feasibility.reward import RewardConfig
    started = perf_counter()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = ROOT / 'docs/rl_feasibility_protocol.md'
    history = ROOT / 'tests/fixtures/food_preferences/challenge.json'
    manifest = {'spec': 'SPEC-RL-002', 'python': platform.python_version(),
                'train_episodes': train_n, 'validation_episodes': validation_n, 'test_episodes': test_n,
                'training_seeds': [101, 202, 303], 'data_seeds': {name: base + data_seed_offset for name, base in
                    [('train', 142), ('validation', 243), ('test', 344), ('shift', 445)]},
                'reward': RewardConfig().to_dict(), 'bootstrap_repetitions': bootstrap,
                'minimum_practical_difference': .10, 'protocol_sha256': digest(protocol),
                'historical_challenge_sha256_before': digest(history), 'external_api_calls': 0,
                'new_installations': 0, 'test_used_for_selection': False,
                'hyperparameter_grid': {'alpha': [.1, .3], 'gamma': [1.], 'epsilon': [[1., .05]]},
                'per_run_max_seconds': 120, 'holdout_human_adjudicated': False}
    manifest['metrics_version'] = 2
    manifest['manual_fixture_sha256'] = digest(ROOT / 'tests/fixtures/rl_feasibility/manual_review_candidates.json')
    amendment = ROOT / 'docs/rl_feasibility_amendment_01.md'
    if amendment.exists():
        manifest['protocol_amendment_sha256'] = digest(amendment)
        (output / 'protocol_amendment_01.md').write_bytes(amendment.read_bytes())
    second_amendment = ROOT / 'docs/rl_feasibility_amendment_02.md'
    if second_amendment.exists():
        manifest['protocol_amendment_02_sha256'] = digest(second_amendment)
        (output / 'protocol_amendment_02.md').write_bytes(second_amendment.read_bytes())
    third_amendment = ROOT / 'docs/rl_feasibility_amendment_03.md'
    if third_amendment.exists():
        manifest['protocol_amendment_03_sha256'] = digest(third_amendment)
        (output / 'protocol_amendment_03.md').write_bytes(third_amendment.read_bytes())
    fourth_amendment = ROOT / 'docs/rl_feasibility_amendment_04.md'
    if fourth_amendment.exists():
        manifest['protocol_amendment_04_sha256'] = digest(fourth_amendment)
        (output / 'protocol_amendment_04.md').write_bytes(fourth_amendment.read_bytes())
    manifest['task'] = 'open recommendation with set-valued acceptable foods and local evidence conflicts'
    manifest['pre_evaluation_instrument_correction'] = 'Manual M06 normalized office to native work in memory/context/reply before any decision evaluation; semantic labels unchanged.'
    write_json(output / 'manifest.json', manifest)
    (output / 'protocol.md').write_bytes(protocol.read_bytes())
    sources = list((ROOT / 'rl_feasibility').glob('*.py')) + list((ROOT / 'tests').glob('test_rl_*.py')) + [ROOT / 'simulator/preferences.py', ROOT / 'simulator/resolution.py', ROOT / 'data_generation/schema.py']
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    for path in sources:
        dest = output / 'source_snapshot' / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    write_json(output / 'source_sha256.json', source_hashes)
    seeds = manifest['data_seeds']
    train = generate_scenarios(train_n, seeds['train'], 'train')
    validation = generate_scenarios(validation_n, seeds['validation'], 'validation')
    early_audit = data_audit({'train': train, 'validation': validation})
    write_json(output / 'train_validation_data_audit.json', early_audit)
    if any(row['latent_signatures'] or row['users'] for row in early_audit['overlap'].values()):
        write_json(output / 'verdict.json', {'dictamen': 'INVALIDATED', 'reason': 'Train/validation semantic or user overlap'})
        raise ValueError('Train/validation overlap: do not train or select models')
    private = output / 'evaluator_private'
    private.mkdir(mode=0o700)
    write_json(private / 'train.json', train)
    write_json(private / 'validation.json', validation)
    tests_pass = None
    if run_tests:
        completed = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                                   cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output / 'unittest.log').write_text(completed.stdout)
        tests_pass = completed.returncode == 0
    sequential = sequentiality_analysis(validation, min(120, len(validation)))
    write_json(output / 'sequentiality_analysis.json', sequential)
    gate1 = bool(tests_pass and sequential['states'] >= 100
                 and sequential['counts']['clarification_enables_correct_decision'] > 0
                 and sequential['counts']['different_terminal_quality_by_question'] > 0)
    write_json(output / 'environment_validation.json', {'gate_1': gate1, 'tests_passed': tests_pass,
               'counterfactual_states': sequential['states'], 'historical_sources_unchanged': digest(history) == manifest['historical_challenge_sha256_before'],
               'scope': 'Automated synthetic contract; no human validation or formal security proof.'})
    if not gate1:
        write_json(output / 'training_metrics.json', {'executed': False, 'reason': 'GATE_1_NOT_PASSED'})
        return {'dictamen': 'EVIDENCIA INSUFICIENTE', 'output': str(output)}
    print('Gate 1 passed. Training supervised baseline and measuring validation headroom.', flush=True)
    supervised = train_supervised(train, output_dir=output / 'supervised')
    policies = make_baselines(supervised['policy'])
    validation_rows, validation_metrics = save_policy_results(output, 'validation', validation, policies, 'ORACLE_STRUCTURED', bootstrap)
    headroom = compare_headroom(validation_rows, bootstrap=bootstrap)
    write_json(output / 'headroom_analysis.json', {'validation': headroom})
    write_json(output / 'baseline_metrics.json', {'validation': validation_metrics})
    trained = {}
    training = {'executed': False, 'reason': 'GATE_2_NOT_PASSED', 'gate_2': headroom['gate_2'],
                'supervised_training': {k: v for k, v in supervised.items() if k != 'policy'}}
    if headroom['gate_2'] == 'PASS':
        from rl_feasibility.train_pilot import train_q_learning
        training = {'executed': True, 'seeds': {}}
        for seed in manifest['training_seeds']:
            attempts = []
            for alpha in manifest['hyperparameter_grid']['alpha']:
                result = train_q_learning(train, validation, seed=seed, episodes=train_n, alpha=alpha,
                                          output_dir=output / f'q_seed{seed}_alpha{alpha}', max_seconds=120)
                attempts.append(result)
            best = max(attempts, key=lambda item: item['training_metrics']['validation_return'])
            trained[f'Q_LEARNING_{seed}'] = best['policy']
            training['seeds'][str(seed)] = {'selected': best['training_metrics'], 'checkpoint': best['checkpoint'],
                                          'checkpoint_sha256': best['checkpoint_sha256'],
                                          'candidates': [a['training_metrics'] for a in attempts]}
        policies.update(trained)
    write_json(output / 'training_metrics.json', training)
    # Freeze before generating or executing test. All choices are now immutable.
    freeze = {'selected_best_baseline': headroom['best_baseline_selected_on_validation'],
              'training_executed': training['executed'], 'policies': list(policies),
              'supervised_sha256': digest(output / 'supervised/supervised_policy.json'),
              'selected_q_checkpoints': training.get('seeds', {}), 'test_examined': False}
    write_json(output / 'freeze.json', freeze)
    print('Policies frozen. Evaluating common test episodes, pipeline and distribution shift.', flush=True)
    test = generate_scenarios(test_n, seeds['test'], 'test')
    shift = generate_scenarios(test_n, seeds['shift'], 'shift', 'shifted')
    write_json(private / 'test.json', test)
    write_json(private / 'shift.json', shift)
    audit = data_audit({'train': train, 'validation': validation, 'test': test, 'shift': shift}, policies)
    write_json(output / 'data_audit.json', audit)
    forbidden_overlap = {pair: row for pair, row in audit['overlap'].items() if pair != 'shift/test' and (row['latent_signatures'] or row['users'])}
    if forbidden_overlap:
        write_json(output / 'verdict.json', {'dictamen': 'INVALIDATED', 'reason': 'Train/validation/test semantic or user overlap',
                                            'overlap': forbidden_overlap})
        raise ValueError('Contaminated held-out states: do not evaluate test')
    test_rows, test_metrics = save_policy_results(output, 'test_oracle', test, policies, 'ORACLE_STRUCTURED', bootstrap)
    pipeline_rows, pipeline_metrics = save_policy_results(output, 'test_pipeline', test, policies, 'PIPELINE', bootstrap)
    shift_rows, shift_metrics = save_policy_results(output, 'shift_oracle', shift, policies, 'ORACLE_STRUCTURED', bootstrap)
    test_headroom = compare_headroom({name: rows for name, rows in test_rows.items() if name in validation_rows},
                                    headroom['best_baseline_selected_on_validation'], bootstrap)
    write_json(output / 'headroom_analysis.json', {'validation': headroom, 'test_confirmation': test_headroom})
    write_json(output / 'test_metrics.json', {'ORACLE_STRUCTURED': test_metrics, 'PIPELINE': pipeline_metrics,
                                            'SHIFT_ORACLE_STRUCTURED': shift_metrics})
    # A deliberately privileged relaxed bound: perfect free information and no
    # interaction costs. It is not an achievable policy or a training target.
    relaxed_bound = mean(1. if s['latent']['acceptable_foods'] else .7 if s['latent']['unresolvable'] else -.35 for s in test)
    write_json(output / 'privileged_relaxed_bound.json', {
        'label': 'COTA PRIVILEGIADA NO ALCANZABLE: información perfecta y gratuita',
        'mean_terminal_utility_bound': relaxed_bound,
        'minus_validation_selected_baseline': relaxed_bound - test_metrics[headroom['best_baseline_selected_on_validation']]['return']['mean'],
        'not_used_for_training_or_hyperparameter_selection': True})
    manual_path = ROOT / 'tests/fixtures/rl_feasibility/manual_review_candidates.json'
    if manual_path.exists():
        manual = json.loads(manual_path.read_text())
        manual_rows, manual_metrics = save_policy_results(output, 'manual', manual['scenarios'], policies, 'ORACLE_STRUCTURED', bootstrap)
        write_json(output / 'manual_test_metrics.json', {'provenance': manual['provenance'], 'metrics': manual_metrics,
                                                       'fixture_sha256': digest(manual_path), 'human_adjudicated': False})
        (output / 'manual_review_candidates.json').write_bytes(manual_path.read_bytes())
        with (output / 'human_review_blind.csv').open('w', newline='', encoding='utf8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['scenario_id', 'query', 'visible_events', 'acceptable_terminal_actions',
                                                        'useful_questions', 'rationale', 'reviewer', 'adjudicator'])
            writer.writeheader()
            for scenario in manual['scenarios']:
                writer.writerow({'scenario_id': scenario['scenario_id'], 'query': scenario['query'],
                                 'visible_events': json.dumps(scenario['events'], ensure_ascii=False)})
    sensitivity = {}
    for cost in (.05, .10, .20):
        sensitivity[str(cost)] = {name: {'mean_return': mean(r['return'] - (cost - .10) * r.get('clarifications', 0) for r in rows),
                                               'retrained': False}
                                 for name, rows in test_rows.items()}
    write_json(output / 'reward_sensitivity.json', {'clarification_cost': sensitivity,
               'scope': 'Frozen trajectories reweighted; not evidence about policies retrained at other weights.'})
    failures = []
    for condition, rows_by_name in [('ORACLE_STRUCTURED', test_rows), ('PIPELINE', pipeline_rows), ('SHIFT', shift_rows)]:
        for name, rows in rows_by_name.items():
            failures.extend({'condition': condition, 'policy': name, **r} for r in rows if not r.get('decision_correct'))
    write_jsonl(output / 'failure_analysis.jsonl', failures)
    write_charts(output, test_metrics, training)
    learning_contrasts = {}
    if trained:
        for name in trained:
            learning_contrasts[name] = {baseline: {key: paired_difference(test_rows[name], test_rows[baseline], key, bootstrap)
                for key in ('return', 'decision_correct', 'evidence_fidelity', 'clarifications')}
                for baseline in ('CURRENT_HEURISTIC', 'SUPERVISED_POLICY', 'STRONG_REFERENCE')}
        write_json(output / 'learning_comparisons.json', learning_contrasts)
    learning_pass = bool(trained) and all(learning_contrasts[name]['CURRENT_HEURISTIC']['return']['ci95'][0] > 0
                                        and learning_contrasts[name]['CURRENT_HEURISTIC']['decision_correct']['mean'] >= 0
                                        for name in trained)
    final = {'gate_1': gate1, 'gate_2': headroom['gate_2'], 'gate_3': 'NOT_EXECUTED' if not trained else ('PASS' if learning_pass else 'INCONCLUSIVE'),
             'gate_4': 'NO_RL_EFFECT_ESTIMATED' if not trained else 'REQUIRES_ANALYSIS',
             'dictamen': 'NO-GO PARA ESTA FORMULACIÓN' if headroom['gate_2'] == 'NO_MEANINGFUL_MARGIN' else 'GO-RL CONDICIONAL',
             'elapsed_seconds': perf_counter() - started, 'training_executed': bool(trained),
             'historical_challenge_unchanged': digest(history) == manifest['historical_challenge_sha256_before']}
    write_json(output / 'verdict.json', final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-episodes', type=int, default=2000)
    parser.add_argument('--validation-episodes', type=int, default=400)
    parser.add_argument('--test-episodes', type=int, default=600)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--skip-tests', action='store_true', help='Gate 1 cannot pass without running tests')
    parser.add_argument('--data-seed-offset', type=int, default=0)
    args = parser.parse_args()
    result = run_experiment(args.output, args.train_episodes, args.validation_episodes,
                            args.test_episodes, args.bootstrap, not args.skip_tests, args.data_seed_offset)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
