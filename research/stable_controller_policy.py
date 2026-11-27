"""Normalized controller head with FP64 categorical likelihood arithmetic.

Only the small head trains. The provider encoder, its public feature extraction
and the independent reader remain unchanged. No fit or SFT occurs on loading.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from research.budget_memory_policy import digest
from research.decision_memory_policy import ASK_ACTIONS, WRITE_ACTIONS, DecisionMemoryPolicy, joint_distribution

VERSION = 'stable_controller_v1'
LAYER_NORM_EPS = 1e-5
GREEDY_TIE_TOLERANCE = 1e-8


class StableControllerPolicy(DecisionMemoryPolicy):
    """FP32 Linear→fixed LayerNorm→Tanh→Linear; FP64 action distributions."""

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        torch=self.torch
        target_device=self.device
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self.head=torch.nn.Sequential(
                torch.nn.Linear(self.feature_size,64),
                torch.nn.LayerNorm(64,eps=LAYER_NORM_EPS,elementwise_affine=False),
                torch.nn.Tanh(),
                torch.nn.Linear(64,8))
            torch.nn.init.zeros_(self.head[-1].weight)
            torch.nn.init.zeros_(self.head[-1].bias)
        self.head=self.head.float().to(target_device)
        self.head.requires_grad_(self.training_enabled)
        self.model=self.head

    def hierarchy(self,features,candidates,*,head=None):
        # Casting is differentiable back into FP32 head parameters. Both actor
        # and frozen reference use the identical FP64 categorical arithmetic.
        return joint_distribution(self._head_logits(features,head).double(),candidates)

    def decide(self,observation,candidates,*,seed,sample,decision_id):
        result=super().decide(observation,candidates,seed=seed,sample=sample,decision_id=decision_id)
        metadata=result['metadata']
        if not sample and not metadata['forced_action']:
            # Numerical ties affect only deterministic evaluation. Sampling and
            # every logged likelihood retain the original exact distribution.
            actions=ASK_ACTIONS if observation['decision_type']=='ask' else WRITE_ACTIONS
            gate=metadata['gate_logprobs'];conditional=metadata['conditional_logprobs']
            positive=int(gate[1]-gate[0]>GREEDY_TIE_TOLERANCE)
            branch=int(conditional[1]-conditional[0]>GREEDY_TIE_TOLERANCE)
            chosen=actions[1+branch] if positive else actions[0]
            result['action']=chosen
            metadata.update(action_index=candidates.index(chosen),chosen_gate_logprob=gate[positive],
                            chosen_conditional_logprob=conditional[branch] if positive else 0.)
        result['metadata'].update(policy_version=VERSION,distribution_dtype='float64',head_dtype='float32')
        return result

    def policy_profile(self):
        profile=super().policy_profile()
        profile.pop('profile_sha256')
        profile.update(version=VERSION,controller_class='StableControllerPolicy',
                       distribution_dtype='float64_logsoftmax_and_joint_action_likelihood',
                       greedy='hierarchical gate then conditional; negative/first ties within fixed logprob tolerance',
                       greedy_tie_logprob_tolerance=GREEDY_TIE_TOLERANCE,
                       head={**profile['head'],
                             'normalization':{'kind':'LayerNorm','normalized_shape':64,
                                              'elementwise_affine':False,'eps':LAYER_NORM_EPS},
                             'layer_order':['Linear','LayerNorm','Tanh','Linear'],
                             'parameter_count':sum(value.numel() for value in self.head.parameters())},
                       hierarchy_implementation_sha256=hashlib.sha256(inspect.getsource(self.hierarchy).encode()).hexdigest(),
                       head_initialization_sha256=hashlib.sha256(inspect.getsource(self.__init__).encode()).hexdigest())
        return {**profile,'profile_sha256':digest(profile)}

    def save_checkpoint(self,destination):
        # The inherited v1 transport hashes head.safetensors and binds its full
        # profile. The extra policy identity is distinct from the format version.
        manifest=super().save_checkpoint(destination)
        manifest['controller_policy_version']=VERSION
        manifest['training_scope']='contextual_controller_head_only; frozen_provider_encoder'
        (Path(destination)/'decision_memory_checkpoint.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        return manifest

    @classmethod
    def from_checkpoint(cls,checkpoint,**kwargs):
        manifest=json.loads((Path(checkpoint)/'decision_memory_checkpoint.json').read_text())
        if (manifest.get('controller_policy_version')!=VERSION
                or manifest.get('profile',{}).get('version')!=VERSION):
            raise ValueError('Stable-controller policy profile/version required; historical head is incompatible')
        return super().from_checkpoint(checkpoint,**kwargs)
