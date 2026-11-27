"""CPU checks for the frozen TRAIN-selected reader presentation, no SGD."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from llm_local_001.model import LocalPolicy, artifact_hashes
from research.budget_memory_policy import BudgetMemoryPolicy, MAX_PROMPT_TOKENS
from research.budget_memory_reader_v2 import (
    FrozenReadableBudgetReader, ReadableBudgetPolicy, reader_template_profile,
)
from research.tests.test_budget_memory_policy import factory, observation


def reader_obs():
    return {'slots': [{'slot_id': 2, 'stored_at': '2026-01-01', 'source_session': 0,
                       'exchange': [{'role': 'user', 'turn_id': 'u1', 'text': 'Prefiero agua.'},
                                    {'role': 'assistant', 'turn_id': 'a1', 'text': '¿Quizá te?'}]}],
            'query': {'attribute': 'drink_preference', 'date': '2026-01-02', 'context': 'casa', 'session_id': 'S5'},
            'answer_menu': ['agua', 'te', 'DEFER']}


def test_reader_render_preserves_literal_turns_roles_dates_and_entire_menu(factory):
    policy = ReadableBudgetPolicy('/unused', lora=False)
    obs = reader_obs()
    text = policy.prompt(obs, obs['answer_menu'], kind='answer')
    assert 'USUARIO [u1]: Prefiero agua.' in text
    assert 'ASISTENTE [a1]: ¿Quizá te?' in text
    assert 'fecha 2026-01-01' in text and 'Fecha: 2026-01-02' in text
    assert 'Tema: bebida' in text and 'Lugar: casa' in text
    assert '{"answer":"valor"}' in text and '["agua","te","DEFER"]' in text
    assert 'Ejemplo' not in text
    assert obs == reader_obs()


def test_manager_prompt_distribution_and_canonical_tokens_are_unchanged(factory):
    base, readable = factory(), ReadableBudgetPolicy('/unused')
    candidates = ['NOOP', 'STORE_SLOT_1', 'STORE_SLOT_2']
    before = base.prompt(observation(), candidates)
    assert readable.prompt(observation(), candidates) == before
    assert readable.candidate_token_ids(candidates) == base.candidate_token_ids(candidates)
    ids = base.encode_prompt(before)
    base.torch.testing.assert_close(base.distribution(ids,candidates), readable.distribution(ids,candidates),rtol=0,atol=0)
    values = ['agua','te','DEFER']
    assert readable.candidate_token_ids(values, 'answer') == base.candidate_token_ids(values, 'answer')


def test_reader_profile_binds_instruction_render_and_rejects_old_checkpoint(factory, monkeypatch, tmp_path):
    base, readable = factory(), ReadableBudgetPolicy('/unused')
    profile = readable.policy_profile()
    assert profile['reader_interface_version']=='budget_memory_readable_reader_v1'
    for key in ('reader_instruction_sha256','reader_render_sha256','reader_template_sha256'):
        assert len(profile[key])==64
    assert profile['profile_sha256'] != base.policy_profile()['profile_sha256']
    def save(self, destination):
        destination=Path(destination);destination.mkdir()
        (destination/'weights.safetensors').write_bytes(b'CPU fixture')
        result={'config':{'model_path':str(destination),'temperature':1},'parameters':{'training_scope':'base_weights'},
                'base_artifacts_sha256':{},'artifacts_sha256':artifact_hashes(destination)}
        (destination/'local_checkpoint.json').write_text(json.dumps(result));return result
    monkeypatch.setattr(LocalPolicy,'save_checkpoint',save)
    readable.save_checkpoint(tmp_path/'new')
    assert ReadableBudgetPolicy.from_checkpoint(tmp_path/'new',device='cpu').policy_profile()==profile
    base.save_checkpoint(tmp_path/'old')
    with pytest.raises(ValueError,match='profile mismatch'):
        ReadableBudgetPolicy.from_checkpoint(tmp_path/'old',device='cpu')


def test_reader_still_frozen_and_requires_full_menu(factory):
    reader=FrozenReadableBudgetReader(ReadableBudgetPolicy('/unused',lora=False),{'drink_preference':('agua','te')})
    obs=reader_obs()
    result=reader.answer(obs,obs['answer_menu'],seed=101,decision_id='new-reader/q1')
    assert result['answer'] in obs['answer_menu'] and not result['metadata']['sampled']
    with pytest.raises(ValueError,match='full public menu'):
        reader.answer({**obs,'answer_menu':['agua','DEFER']},['agua','DEFER'],seed=101,decision_id='bad/q1')


def test_production_prompt_exactly_matches_calibrated_ten_original_train_cases():
    from scripts.calibrate_budget_reader import observation_from_logged_case
    source=Path('research/runs/reward_review_budget_20261105_001/FULL_HISTORY_READER.json')
    inference=Path('reports/reward_review_budget_20261105_001/reader_calibration_001/INFERENCE.jsonl')
    if not source.exists() or not inference.exists():
        pytest.skip('Local TRAIN calibration not present')
    rows=json.loads(source.read_text())
    recorded={r['case_index']:r for r in map(json.loads,inference.read_text().splitlines()) if r['variant']=='readable_json'}
    assert len(recorded)==len(rows)==10
    policy=ReadableBudgetPolicy.__new__(ReadableBudgetPolicy)
    from transformers import AutoTokenizer
    policy.tokenizer=AutoTokenizer.from_pretrained('models/Qwen2.5-0.5B-Instruct',local_files_only=True)
    policy._candidate_cache={}
    for i,row in enumerate(rows):
        obs=observation_from_logged_case(row)
        text=policy.prompt(obs,obs['answer_menu'],kind='answer')
        assert text==recorded[i]['prompt']
        assert policy.encode_prompt(text)==recorded[i]['prompt_ids']
        assert len(policy.encode_prompt(text))<=MAX_PROMPT_TOKENS


def test_two_slot_prompt_budget_on_original_train_subsets_and_synthetic_capacity():
    from itertools import combinations
    from transformers import AutoTokenizer
    from research.budget_memory_env import VALUE_MENUS
    from scripts.calibrate_budget_reader import observation_from_logged_case
    policy=ReadableBudgetPolicy.__new__(ReadableBudgetPolicy)
    policy.tokenizer=AutoTokenizer.from_pretrained('models/Qwen2.5-0.5B-Instruct',local_files_only=True)
    policy._candidate_cache={}
    observations=[]
    source=Path('research/runs/reward_review_budget_20261105_001/FULL_HISTORY_READER.json')
    if source.exists():
        for row in json.loads(source.read_text()):
            obs=observation_from_logged_case(row)
            for slots in combinations(obs['slots'],2):
                observations.append({**obs,'slots':deepcopy(list(slots))})
    token=policy.tokenizer.encode(' hola',add_special_tokens=False)[0]
    chunk=policy.tokenizer.decode([token]*32,skip_special_tokens=True)
    slots=[{'slot_id':i+1,'stored_at':'2030-01-01','source_session':i,
            'exchange':[{'role':role,'turn_id':f's{i}t{j}','text':chunk}
                        for j,role in enumerate(('user','assistant','user'))]} for i in range(2)]
    for attr,values in VALUE_MENUS.items():
        observations.append({'slots':deepcopy(slots),
           'query':{'attribute':attr,'context':'oficina','date':'2030-01-09','session_id':'future_1'},
           'answer_menu':list(values)+['DEFER']})
    for obs in observations:
        assert len(policy.encode_prompt(policy.prompt(obs,obs['answer_menu'],kind='answer')))<=MAX_PROMPT_TOKENS
