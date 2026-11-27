"""TRAIN target coverage and semantically stable distractor controls."""
from copy import deepcopy

import pytest

from research import competence_data as base
from research import targeted_curriculum_data as targeted
from research.src.policy.dialogue import declarations


def test_cross_attribute_tail_retains_queried_truth_and_grounded_teacher():
    source = next(row for row in base._fresh_batch(202611055, 0)
                  if row['scenario_family'] == 'F1')
    original = deepcopy(source)
    episode = targeted.enrich(source, seed=202611055, namespace='test_targeted')
    assert source == original
    assert episode['truth'] == source['truth']
    assert episode['current_query'] == source['current_query']
    assert episode['future_query'] == source['future_query']
    public = {'user_messages': episode['user_messages'], 'query': episode['current_query']}
    assert declarations(public)[-1][1] != episode['current_query']['attribute']
    episode['split'] = 'train'
    examples = base.teacher_for_episode(episode)
    for example in examples:
        targeted._check_target(example)
    assert 'query_differs_from_last_declaration' in targeted.curriculum_tags(examples[1])
    assert len(examples[2]['action']['operations']) <= 2


def test_stable_no_new_fact_case_has_noop_target():
    source = next(row for row in base._fresh_batch(202611055, 0)
                  if row['scenario_family'] == 'F0')
    episode = targeted.enrich(source, seed=202611055, namespace='test_targeted')
    targeted._no_new_preference(episode)
    assert episode['truth'] == source['truth']
    episode['split'] = 'train'
    examples = base.teacher_for_episode(episode)
    assert examples[2]['action'] == {'operations': [{'op': 'NOOP'}]}
    targeted._check_target(examples[2])


def test_target_rejects_invented_evidence_quote():
    source = next(row for row in base._fresh_batch(202611055, 0)
                  if row['scenario_family'] == 'F1')
    episode = targeted.enrich(source, seed=202611055, namespace='test_targeted')
    episode['split'] = 'train'
    write = base.teacher_for_episode(episode)[2]
    record = write['action']['operations'][0]['record']
    record['evidence_quotes'][0]['text'] = 'Invented preference without user evidence.'
    import json
    write['completion'] = json.dumps(write['action'])
    with pytest.raises(ValueError, match='unsupported'):
        targeted._check_target(write)


def test_context_question_acquires_location_without_widening_preference_scope():
    source = next(row for row in base._fresh_batch(202611055, 0)
                  if row['scenario_family'] == 'F3')
    episode = targeted.enrich(source, seed=202611055, namespace='test_targeted')
    episode['split'] = 'train'
    assert episode['current_query']['context'] is None
    assert episode['truth']['current_value'] == source['truth']['current_value']
    assert episode['truth']['future_value'] == source['truth']['new_value']
    assert episode['future_query']['context'] == 'casa'
    assert episode['future_query']['date'] > episode['current_query']['date']
    examples = base.teacher_for_episode(episode)
    assert examples[0]['action'] == {'action': 'ASK', 'slot': 'CURRENT_CONTEXT'}
    assert 'select_ASK_CURRENT_CONTEXT' in targeted.curriculum_tags(examples[0])
    for example in examples:
        targeted._check_target(example)
    queried_writes = [op for op in examples[2]['action']['operations']
                      if op.get('record', {}).get('attribute') == episode['current_query']['attribute']]
    assert queried_writes
    assert all(op['record']['scope']['context'] == 'casa' for op in queried_writes)


@pytest.mark.parametrize("inside", [True, False])
def test_interval_reuse_is_authored_inside_or_after_exclusive_end(inside):
    source = next(row for row in base._fresh_batch(202611057, 0)
                  if row['scenario_family'] == 'F2')
    episode = targeted.enrich(source, seed=202611057, namespace='test_targeted', interval_inside=inside)
    end = episode['truth']['scope']['valid_until_exclusive']
    assert episode['future_query']['date'] > episode['current_query']['date']
    assert (episode['future_query']['date'] < end) == inside
    assert episode['truth']['future_value'] == source['truth']['new_value' if inside else 'initial_value']


def test_ambiguous_worlds_preserve_same_public_prefix():
    rows = [row for row in base._fresh_batch(202611057, 0) if row['scenario_family'] == 'F4']
    root = rows[0]['root_id']
    variants = [targeted.enrich(row, seed=202611057, namespace='test_targeted') for row in rows if row['root_id'] == root]
    assert len(variants) == 2
    assert variants[0]['user_messages'] == variants[1]['user_messages']
    assert variants[0]['initial_memory'] == variants[1]['initial_memory']
    assert variants[0]['current_query'] == variants[1]['current_query']
    assert variants[0]['root_id'] == variants[1]['root_id']
    assert variants[0]['truth']['scope_kind'] != variants[1]['truth']['scope_kind']
