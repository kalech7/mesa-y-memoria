from copy import deepcopy
from types import SimpleNamespace

import pytest

from llm_local_001.model import Generation
from research.signal_diagnostic import (ACTIONS, GreedyCache, analyze, audit_bank,
    enumerate_observations, independent_read, load_versioned_development, options)
from research.src.artifacts import digest
from research.src.closure_environment import _episode, _record, _query, _scope, infrastructure_fixtures


def test_enumeration_has_same_legal_slots_across_hidden_worlds_and_no_private_prefix():
    first = _episode()
    second = deepcopy(first)
    second['latent_world_id'] = 'temporary'
    second['truth'].update(future_value='sopa', scope=_scope('session', start='2026-01-12', session='S2'))
    second['replies']['TEMPORAL_SCOPE']['text'] = 'Solo en esta conversación.'
    cases = enumerate_observations([first, second])
    assert cases[0]['state_id'] == cases[1]['state_id']
    assert set(cases[0]['observations']) == set(ACTIONS)
    assert 'world_ongoing' not in cases[0]['select_prompt']
    assert audit_bank(cases)['status'] == 'passed'


def test_bank_audit_finds_contradictory_redundant_reply_even_if_parser_masks_it():
    episode = _episode()
    episode['user_messages'][0]['text'] = 'Solo en esta conversación prefiero pasta. ¿Qué ceno hoy?'
    episode['truth']['scope'] = _scope('session', start='2026-01-12', session='S2')
    # Parser prioritizes the already-visible session statement. The independent
    # reply-text check must still reject this new ongoing assertion.
    assert audit_bank(enumerate_observations([episode]))['status'] == 'failed'


def test_independent_consumer_literal_fixtures():
    for fixture in infrastructure_fixtures():
        response = independent_read(fixture['memory'], fixture['query'])
        if fixture['expected_mode'] == 'ANSWER':
            assert response['mode'] == 'ANSWER', fixture['name']
            assert response['value'] == fixture['expected_value'], fixture['name']
        else:
            assert response['mode'] == 'DEFER', fixture['name']


def test_cache_never_keys_on_hidden_world_and_never_reuses_sampling(tmp_path):
    class Backend:
        config = SimpleNamespace(max_new_tokens=64)
        calls = 0
        def generate(self, prompt, sample=False):
            self.calls += 1
            return Generation([1, 2], [3], 'answer', metadata={})
    backend = Backend()
    allowed = {digest({'prompt':'public', 'max_new_tokens':64})}
    cached = GreedyCache(backend, 'actual_weights_namespace', tmp_path/'cache', allowed)
    assert cached.generate('public').metadata['cache_hit'] is False
    assert cached.generate('public').metadata['cache_hit'] is True
    assert backend.calls == 1
    with pytest.raises(ValueError, match='no sampling'):
        cached.generate('public', sample=True)
    with pytest.raises(ValueError, match='Unplanned'):
        cached.generate('other')


def records():
    rows=[]
    for world in ['a','b']:
        for action in ACTIONS:
            long = (1 if world=='a' else -1) if action=='PROCEED' else .9 if action=='ASK:TEMPORAL_SCOPE' else 0
            for writer in ['B1','reference']:
                rows.append(dict(case_id=world,root_id='root',latent_world_id=world,realization_id='v0',
                    observable_state_id='same_public_input',family='F4',action=action,writer=writer,
                    B1_selected_action='ASK:TEMPORAL_SCOPE',C2=1,FJS=1 if action=='ASK:TEMPORAL_SCOPE' else 0,
                    J_primary=1 if action=='ASK:TEMPORAL_SCOPE' else 0,R_long=long,R_short=.9,
                    unsupported_write_count=0,invalid_output_count=0,independent_reader_FJS=1 if action=='ASK:TEMPORAL_SCOPE' else 0,trace_line=len(rows)+1))
        selected=next(deepcopy(r) for r in rows if r['case_id']==world and r['action']=='ASK:TEMPORAL_SCOPE' and r['writer']=='B1')
        selected['action']='NATURAL';rows.append(selected)
    return rows


def test_expected_action_value_averages_worlds_before_argmax():
    result=analyze(records())
    state=result['states'][0]
    assert state['action_values']['PROCEED']['R_long']==0
    assert state['best_R_long']=='ASK:TEMPORAL_SCOPE'
    assert state['regret_R_long']==0
    assert result['class_counts'].get('selector_longitudinal_suboptimal',0)==0


def test_invalid_slot_not_repaired_using_gold():
    with pytest.raises(ValueError):options('ASK:HIDDEN_GOLD_SLOT')


def test_incomplete_action_matrix_is_rejected():
    rows=[r for r in records() if r['action']!='ASK:GENERIC']
    with pytest.raises(ValueError,match='Incomplete'):analyze(rows)


def test_new_diagnostic_requires_frozen_development_only(tmp_path):
    import json
    bank = tmp_path / 'validation.jsonl'
    bank.write_text('{}\n')
    (tmp_path / 'manifest.json').write_text(json.dumps({'data_role': 'test'}))
    with pytest.raises(ValueError, match='prospective development'):
        load_versioned_development(bank)


def test_independent_reader_preserves_context_and_bounded_specificity():
    memory = [_record('general', 'arroz', scope=_scope('ongoing', start='2026-01-01')),
              _record('local', 'pasta', scope=_scope('contextual', start='2026-01-01',
                      end='2026-01-20', context='casa'))]
    for context, at, expected in [('casa', '2026-01-19', 'pasta'),
                                  ('oficina', '2026-01-19', 'arroz'),
                                  ('casa', '2026-01-20', 'arroz')]:
        result = independent_read(memory, _query(date=at, context=context))
        assert result['value'] == expected
