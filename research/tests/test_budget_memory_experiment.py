import pytest

from research.budget_memory_experiment import aggregate, better_selection, eligible, paired


def test_worlds_do_not_overweight_a_root_or_family_and_lists_are_not_metrics():
    rows = [
        dict(family='a', root_id='r1', metrics=dict(query_success=1, query_utilities=[1, 1])),
        dict(family='a', root_id='r1', metrics=dict(query_success=0, query_utilities=[0, 0])),
        dict(family='b', root_id='r2', metrics=dict(query_success=1, query_utilities=[1, 1])),
    ]
    result = aggregate(rows)
    assert result['metrics'] == {'query_success': .75}
    assert result['by_root']['r1']['query_success'] == .5
    assert result['roots'] == 2 and result['episodes'] == 3


def test_selection_requires_correctness_guard_and_ignores_roundoff_ties():
    baseline = dict(net_return=.2, query_success=.5, ask_count=.5)
    assert not eligible(dict(query_success=.49), baseline)
    assert eligible(dict(query_success=.5), baseline)
    assert not better_selection({**baseline, 'net_return': .2 + 1e-16}, baseline)
    assert better_selection({**baseline, 'net_return': .21}, baseline)
    assert better_selection({**baseline, 'ask_count': .4}, baseline)


def test_paired_counts_use_matching_roots():
    original = dict(by_root={'r1': {key: 0. for key in
                    ('net_return', 'query_success', 'joint_success', 'ask_count', 'write_count')}})
    updated = dict(by_root={'r1': {key: 1. for key in original['by_root']['r1']}})
    result = paired(original, updated)
    assert result['query_success'] == dict(mean_difference=1., improved=1, worse=0, tied=0)
    with pytest.raises(ValueError):
        paired(original, dict(by_root={'r2': updated['by_root']['r1']}))
