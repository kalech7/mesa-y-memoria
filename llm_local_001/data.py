"""Freeze reused trajectories and explicitly audit abstract support overlap.

The authored labels are private evaluator/demonstration data, not LLM inputs.
No claim of independent human authorship or abstract-mechanism generalization.
"""
from collections import Counter
from itertools import combinations

from rl_small_pilot.environment_adapter import generate_splits
from rl_observability.scenarios import FAMILIES

from .artifacts import digest, write_json


def abstract_signature(scenario, include_context=False):
    """Ignore identities, food lexemes, absolute dates, and split template IDs.

    Retain mechanism, opposite direction, event structure and relative intervals.
    Report both with actual context and with context values abstracted.
    """
    sessions = []
    target = scenario['sessions'][0]['target_food']
    for session in scenario['sessions']:
        sessions.append([(f['food'] == target, f['polarity'], f['kind'], f['source'],
                          bool(f.get('valid_to')), bool(f.get('supersedes')))
                         for f in session['incoming']])
    value = {'mechanism': scenario['family'], 'initial_direction': scenario['initial_direction'],
             'sessions': sessions}
    if include_context:
        value['context'] = scenario['context_combination']
    return digest(value)


def freeze_data(output, sizes=(60, 20, 40), seed=61001, metadata=None):
    if any(n < 20 or n % 10 for n in sizes):
        raise ValueError('Each split needs >=20 users in multiples of 10 for five families and both directions')
    splits = generate_splits(*sizes, seed=seed)
    audit = {'provenance': metadata or {}, 'source': 'rl_small_pilot.environment_adapter.generate_splits',
             'synthetic': True, 'independent_human_sample': False,
             'human_review': 'pending; technical pilot, not confirmatory evidence',
             'frozen_before_training': True, 'test_used_for_selection': False,
             'families_shared_by_design': list(FAMILIES), 'splits': {}, 'overlap': {}}
    for name, scenarios in splits.items():
        write_json(output / 'evaluator_private' / f'{name}.json',
                   {'provenance': metadata or {}, 'scenarios': scenarios})
        audit['splits'][name] = {
            'users': len(scenarios), 'trajectories': len(scenarios),
            'sessions': 3 * len(scenarios), 'sha256_canonical': digest(scenarios),
            'families': dict(Counter(s['family'] for s in scenarios)),
            'directions': dict(Counter(s['initial_direction'] for s in scenarios)),
            'contexts': sorted({tuple(s['context_combination']) for s in scenarios}),
            'surface_templates': sorted({s['template_id'] for s in scenarios}),
            'abstract_signature_counts': dict(Counter(abstract_signature(s) for s in scenarios)),
            'context_signature_counts': dict(Counter(abstract_signature(s, True) for s in scenarios)),
        }
    for left, right in combinations(splits, 2):
        a, b = splits[left], splits[right]
        ids_a, ids_b = {s['user_id'] for s in a}, {s['user_id'] for s in b}
        contexts_a = {tuple(s['context_combination']) for s in a}
        contexts_b = {tuple(s['context_combination']) for s in b}
        # Raw messages (ignoring safety phrase shared by design) must differ.
        def texts(rows):
            return {f['text'] for s in rows for session in s['sessions']
                    for f in session['incoming'] if f['polarity'] != 'FORBID'}
        signatures_a = {abstract_signature(s) for s in a}
        signatures_b = {abstract_signature(s) for s in b}
        overlap = {
            'user_overlap': sorted(ids_a & ids_b),
            'context_overlap': sorted(contexts_a & contexts_b),
            'preference_text_overlap': sorted(texts(a) & texts(b)),
            'abstract_overlap_count': len(signatures_a & signatures_b),
            'right_trajectories_with_seen_abstract_signature':
                sum(abstract_signature(s) in signatures_a for s in b),
            'right_trajectories': len(b),
            'interpretation': 'Novel wording and context combinations; mechanisms intentionally shared. '
                              'New users alone are not evidence of structural generalization.',
        }
        if overlap['user_overlap'] or overlap['context_overlap'] or overlap['preference_text_overlap']:
            raise ValueError(f'Dataset split leakage: {left}/{right}: {overlap}')
        audit['overlap'][f'{left}:{right}'] = overlap
    audit['test_frozen_hash'] = digest(splits['test'])
    write_json(output / 'dataset_manifest.json', audit)
    return splits, audit
