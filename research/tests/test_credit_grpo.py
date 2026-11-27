"""Exercise complete two-pass credit optimization without SFT or a real model."""
from contextlib import nullcontext
import json
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace

import pytest

from llm_local_001.model import Generation
from research.src.training.credit_grpo import (
    EpisodeRollout, with_credit, phase_advantages, train_credit_grpo,
)
from research.src.training.episodic_grpo import PhaseSample


def test_phase_credit_does_not_collapse_to_episode_mean():
    group=[]
    for index in range(4):
        sign=1 if index<2 else -1
        group.append(with_credit(EpisodeRollout("r","w",index,[],0,0),
                                 dict(select=sign,respond=-sign,write=0)))
    assert all(fmean(r.phase_rewards.values())==0 for r in group)
    advantage=phase_advantages(group)
    assert advantage["select"][0]>0>advantage["respond"][0]
    assert advantage["write"]==[0]*4
    with pytest.raises(ValueError):
        with_credit(group[0],dict(select=float("nan"),respond=0,write=0))


def test_two_passes_keep_old_fixed_and_equalize_phase_lengths(tmp_path):
    torch=pytest.importorskip("torch")
    torch.set_num_threads(1)
    class ToyModel(torch.nn.Module):
        def __init__(self,trainable):
            super().__init__()
            self.lora_weight=torch.nn.Parameter(torch.zeros(3,2),requires_grad=trainable)
            self.base=torch.nn.Parameter(torch.ones(1),requires_grad=False)
            self.peft_config={}
    class ToyPolicy:
        def __init__(self,trainable):
            self.torch=torch;self.model=ToyModel(trainable);self.device="cpu"
            self.config=SimpleNamespace(seed=101,temperature=1.)
            self.origin_checkpoint="nonprotocol_CPU_unit_fixture"
        def seed(self,seed):
            torch.manual_seed(seed)
        def enable_training(self):
            pass
        def likelihood_mode(self):
            return nullcontext()
        def sequence_logprobs(self,generation,sampling_distribution=True):
            phase=generation.prompt_ids[0]-1
            action=generation.completion_ids[0]-1
            return self.model.lora_weight[phase].log_softmax(0)[action].expand(len(generation.completion_ids))
        def save_checkpoint(self,path):
            Path(path).mkdir(parents=True,exist_ok=True)
            torch.save(self.model.state_dict(),Path(path)/"toy.pt")
    actor,reference=ToyPolicy(True),ToyPolicy(False)
    def collect(policy,root,seed):
        draw=seed%4;sign=1 if draw<2 else -1;action=1 if sign==1 else 2
        phases=[PhaseSample(p,Generation([i+1,9],[action]*n,"{}"))
                for i,(p,n) in enumerate(zip(("select","respond","write"),(1,7,31)))]
        return with_credit(EpisodeRollout("r","w",seed,phases,0,0,
            memory_path=str(tmp_path/f"{seed}.sqlite"),split="dev_smoke"),
            dict(select=sign,respond=-sign,write=0.))
    output=tmp_path/"optimizer"
    result=train_credit_grpo(actor,reference,["r"],collect,output,lambda _:dict(utility=0.),
        updates=1,learning_rate=.4,kl_coefficient=0.,clip_epsilon=.1,checkpoint_every=1,
        optimization_passes=2,smoke=True,budget_seconds=30)
    passes=[json.loads(l) for l in (output/"optimizer_passes.jsonl").read_text().splitlines()]
    assert result["resource_counters"]["optimizer_steps"]==2
    assert result["resource_counters"]["generated_tokens"]==4*(1+7+31)
    assert result["resource_counters"]["trained_tokens"]==2*4*(1+7+31)
    assert passes[0]["ratio_min"]==pytest.approx(1.)
    assert passes[0]["ratio_max"]==pytest.approx(1.)
    assert passes[1]["ratio_max"]>1.1 and passes[1]["ratio_min"]<.9
    assert passes[1]["clip_fraction_phase_mean"]>0
    weight=actor.model.lora_weight.detach()
    assert float(weight[0,0])>float(weight[0,1])
    assert float(weight[1,0])<float(weight[1,1])
    assert bool(torch.equal(weight[0],-weight[1]))
    assert bool((weight[2]==0).all())
    assert result["reference_before"]==result["reference_after"]
    assert result["frozen_before"]==result["frozen_after"]
    assert result["trainable_before"]!=result["trainable_after"]
