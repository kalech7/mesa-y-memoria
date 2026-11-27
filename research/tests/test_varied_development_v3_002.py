from copy import deepcopy
from collections import Counter
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import pytest

from research.varied_development import build_varied_development, diversity_plan, MEMORY_COUNTS
from research.varied_development_v3_002 import revise_for_observability, prepare_revision
from research.src.data.semantic_partition_v3 import (
    assert_semantic_split_isolation_v3, SemanticSplitLeakage, semantic_episode_id_v3,
)


def parent():
    rows=build_varied_development()
    return rows,{"data_namespace":"development_varied_v3_20261025_001",**diversity_plan(rows),
                 "limits":[],"data_role":"prospective_development_only","split":"validation"}


def test_revision_preserves_truth_current_dialogue_lineage_and_all_other_memories():
    old,manifest=parent();original=deepcopy(old)
    new,changes=revise_for_observability(old,manifest)
    assert old==original and len(changes)==9
    assert len(new)==60 and len({e['root_id'] for e in new})==50
    for before,after in zip(old,new):
        for key in ('root_id','latent_world_id','realization_id','truth','user_messages','current_query','future_query','replies'):
            assert before[key]==after[key]
        assert before['initial_memory'][:-1]==after['initial_memory'][:-1]
        assert len(before['initial_memory'])==len(after['initial_memory'])
    assert Counter(len(e['initial_memory']) for e in {e['root_id']:e for e in new}.values())==Counter(MEMORY_COUNTS)


def test_support_loss_controls_are_recent_expired_and_cited_to_actual_prior_users():
    old,manifest=parent();new,changes=revise_for_observability(old,manifest)
    positives=negatives=0
    for episode in new:
        control=episode['authoring'].get('observability_control')
        if not control:continue
        if control['kind']=='single_memory_negative_control':
            assert len(episode['initial_memory'])==1
            negatives+=1;continue
        positives+=1
        base,expired=episode['initial_memory'][0],episode['initial_memory'][-1]
        assert base['attribute']==expired['attribute']==episode['current_query']['attribute']
        assert base['value']==episode['truth']['current_value']==episode['truth']['future_value']
        assert expired['value']!=base['value'] and expired['supersedes']==[]
        assert expired['recorded_at']>base['recorded_at']
        assert expired['scope']['valid_until_exclusive']==episode['current_query']['date']
        assert expired['scope']['valid_from']==(date.fromisoformat(episode['current_query']['date'])-timedelta(days=1)).isoformat()
        latest=max(enumerate([r for r in episode['initial_memory'] if r['attribute']==base['attribute']]),
                   key=lambda item:(item[1]['recorded_at'],item[0]))[1]
        assert latest['memory_id']==expired['memory_id']
        assert 1<=len(episode['prior_sessions'])<=3
        sources={m['turn_id']:(s['date'],m) for s in episode['prior_sessions'] for m in s['messages']}
        for record in episode['initial_memory']:
            for quote in record['evidence_quotes']:
                at,source=sources[quote['turn_id']]
                assert source['role']=='user' and quote['text']==source['text']
                assert at==record['recorded_at']
    assert positives==9 and negatives==1


def test_parent_versions_are_explicitly_dependent_not_new_independent_users():
    old,manifest=parent();new,_=revise_for_observability(old,manifest)
    signatures={semantic_episode_id_v3(e) for e in old}
    assert sum(semantic_episode_id_v3(e) in signatures for e in new)==51
    with pytest.raises(SemanticSplitLeakage):
        assert_semantic_split_isolation_v3({'validation':new},historical_splits={'parent001':old})


def test_preparation_protects_parent_and_generator_snapshots(tmp_path):
    old,manifest=parent();parent_dir=tmp_path/'parent';parent_dir.mkdir()
    parent_bank=parent_dir/'validation.jsonl'
    parent_bank.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in old))
    generator=Path('research/varied_development.py')
    manifest.update(bank_sha256=hashlib.sha256(parent_bank.read_bytes()).hexdigest(),producer_sha256=hashlib.sha256(generator.read_bytes()).hexdigest())
    (parent_dir/'manifest.json').write_text(json.dumps(manifest))
    history=tmp_path/'history';history.mkdir()
    unrelated=deepcopy(old[0]);unrelated['root_id']='older_unrelated_root';unrelated['user_messages']=[{'role':'user','turn_id':'u1','text':'Otra historia antigua distinta.'}]
    for split in ('dev_smoke','train','validation','test_id','test_ood','test_external'):
        (history/f'{split}.jsonl').write_text(json.dumps(unrelated)+'\n')
    previous=tmp_path/'v2.jsonl';previous.write_text(json.dumps(unrelated)+'\n')
    data,report=tmp_path/'new',tmp_path/'report'
    result=prepare_revision(parent_dir,data,report,historical_directory=history,previous_development=previous,parent_generator=generator)
    assert result['parent_protection_before']==result['parent_protection_after']
    assert result['independent_of_parent'] is False and result['expected_parent_semantic_overlap_episodes']==51
    assert (report/'source_snapshot/parent_generator.py').read_bytes()==generator.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_revision(parent_dir,data,report,historical_directory=history,previous_development=previous,parent_generator=generator)
