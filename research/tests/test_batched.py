from copy import deepcopy
import json
from dataclasses import asdict
from llm_local_001.model import Generation
from research.src.artifacts import canonical
from research.src.data.build_bank import build_bank
from research.src.policy.rule_controller import RuleController
from research.src.evaluation import batched


def fake_generate(backend, phase, observations, **options):
    rule=RuleController()
    rows=[]
    for obs in observations:
        action=rule.select_forced_ask(obs) if phase=='select' and options.get('force_ask') else getattr(rule,phase)(obs)
        text=canonical(action)
        rows.append({'phase':phase,'observation':deepcopy(obs),'prompt':canonical(obs),'raw_output':text,'parsed_action':action,'technical_status':'completed','retry_count':0,'generation':asdict(Generation([1],[2],text))})
    return rows


def test_batched_actions_replay_the_same_observed_prefixes(monkeypatch):
    monkeypatch.setattr(batched,'generate_batch',fake_generate)
    episodes=build_bank(roots=5)
    rows=list(batched.evaluate_batched(episodes,None,batch_size=4))
    assert len(rows)==len(episodes)
    assert all(row['metrics']['C2']==1 and row['metrics']['FJS']==1 for row in rows)
    assert all(len(row['trace']['observations'])==3 for row in rows)


def test_batched_intervention_writer_and_responder_information_bytes(monkeypatch):
    monkeypatch.setattr(batched,'generate_batch',fake_generate)
    episode=next(e for e in build_bank(roots=5) if e.get('opportunity',{}).get('eligible'))
    arms={arm:next(batched.evaluate_batched([episode],None,intervention=arm,batch_size=4,forced_slot='TEMPORAL_SCOPE')) for arm in ('I0','I1','I2')}
    assert canonical(arms['I0']['trace']['observations']['write'])==canonical(arms['I1']['trace']['observations']['write'])
    assert canonical(arms['I1']['trace']['observations']['respond'])==canonical(arms['I2']['trace']['observations']['respond'])
    assert arms['I2']['metrics']['FJS']==1
    assert arms['I1']['metrics']['FJS']==0
