"""Runner dispatch/provenance tests; no provider models or heldout data."""
import json
from pathlib import Path
import pytest
from research import stable_controller_experiment as runner


def test_freeze_covers_historical_sources_and_all_stable_modules():
    old=runner.legacy_sources();current=runner.sources()
    assert all(current[path]==sha for path,sha in old.items())
    for filename in ('stable_controller_policy.py','stable_controller_training.py','stable_controller_experiment.py'):
        assert str(Path(runner.__file__).with_name(filename)) in current


@pytest.mark.parametrize('kwargs',[{'lr':.003},{'groups':64},{'budget_seconds':3500}])
def test_schedule_cannot_silently_change_before_loading_models(tmp_path,kwargs):
    with pytest.raises(ValueError,match='Prospective schedule'):
        runner.train(tmp_path/'absent',tmp_path/'absent_data',**kwargs)
    assert not (tmp_path/'absent').exists()


def test_failed_inherited_review_blocks_all_model_loading(tmp_path,monkeypatch):
    inherited=tmp_path/'prior';inherited.mkdir()
    for name,payload in (
        ('PILOT_PLAN.json',{'version':'decision_memory_session_experiment_v1'}),
        ('PILOT_RESULT.json',{'status':'passed','optimizer_steps':0,'sft_updates':0}),
        ('PILOT_REVIEW.json',{'status':'failed'})):
        (inherited/name).write_text(json.dumps(payload))
    def forbidden(*args,**kwargs):raise AssertionError('Model loading before review verification')
    monkeypatch.setattr(runner,'controller',forbidden)
    monkeypatch.setattr(runner,'fresh_reader',forbidden)
    with pytest.raises(ValueError,match='Reviewed passing session pilot'):
        runner.pilot(tmp_path/'new',tmp_path/'missing_data',inherited)
    assert not (tmp_path/'new').exists()


def test_all_head_evaluation_loads_stable_class_and_preserves_alias(monkeypatch,tmp_path):
    selection={'initial_checkpoint':'zero','selected_checkpoint':'zero','terminal_checkpoint':'thirtytwo','selected_step':0}
    monkeypatch.setattr(runner,'check',lambda plan:None)
    monkeypatch.setattr(runner,'load_bank',lambda data:({'evaluation':['synthetic fixture']},{}))
    monkeypatch.setattr(runner,'fresh_reader',lambda:object())
    monkeypatch.setattr(runner,'release',lambda *args,**kwargs:None)
    monkeypatch.setattr(runner,'fresh_manager',lambda:'lm_fixture')
    monkeypatch.setattr(runner,'read',lambda path:selection if Path(path).name=='SELECTION.json' else {'resource_counters':{}})
    monkeypatch.setattr(runner,'file_hash',lambda path:'fixed')
    loaded=[]
    def load(path,**kwargs):loaded.append(path);return 'stable_fixture'
    monkeypatch.setattr(runner.StableControllerPolicy,'from_checkpoint',load)
    methods=[]
    def evaluate(actor,reader,episodes,output,method):
        methods.append(method);return {'metrics':{'query_success':0.},'by_root':{}}
    monkeypatch.setattr(runner,'evaluate_policy',evaluate)
    monkeypatch.setattr(runner,'paired',lambda a,b:{})
    outputs=[];monkeypatch.setattr(runner,'write_json',lambda path,value:outputs.append(value))
    runner.evaluate(tmp_path,tmp_path/'data_fixture')
    assert loaded==['zero','thirtytwo']
    result=outputs[-1]
    assert set(result['panels'])=={'Head0','RL-selected','RL-terminal','LM-frozen','FIFO','HierRandom','Public-rules'}
    assert result['panels']['RL-selected']['identical_to']=='Head0'
    assert 'RL-selected' not in methods


def test_telemetry_uses_cached_head_only_and_detects_hidden_normalization():
    import torch
    from types import SimpleNamespace
    head=torch.nn.Sequential(torch.nn.Linear(4,64),torch.nn.LayerNorm(64,elementwise_affine=False),
                             torch.nn.Tanh(),torch.nn.Linear(64,8))
    actor=SimpleNamespace(head=head,device=torch.device('cpu'))
    target={'metadata':{'feature_vector':[1e6,-2e6,3e6,-4e6]}}
    group={'ask_target':target,'write_targets':[target,target,target]}
    before={k:v.clone() for k,v in head.state_dict().items()}
    diagnostics=runner._head_activation_diagnostics(actor,group)
    assert diagnostics['cached_target_count']==4
    assert diagnostics['encoder_calls']==diagnostics['reader_calls']==0
    assert diagnostics['firstlinear_preactivation_absmax']>1e5
    assert diagnostics['hidden_saturation_fraction_max_state']<=.125
    assert diagnostics['post_layernorm_variance_max']<=1.00001
    assert all(torch.equal(before[k],v) for k,v in head.state_dict().items())
    assert all(p.grad is None for p in head.parameters())
