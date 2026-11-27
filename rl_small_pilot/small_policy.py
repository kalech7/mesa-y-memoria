"""A genuinely trainable CPU policy, independent of any language model.

Only observable features and the fixed action catalogue enter this network.
The only output mask enforces the public clarification budget.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import random
import struct

import torch
from torch import nn

from .observations import ACTIONS, FEATURE_DIM, action_mask, feature_vector


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


class SmallPolicy(nn.Module):
    name = 'SMALL_NEURAL_POLICY'

    def __init__(self, seed=101, hidden=64):
        super().__init__()
        seed_everything(seed)
        self.seed, self.hidden = seed, hidden
        self.network = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh(),
                                     nn.Linear(hidden, len(ACTIONS)))

    def logits(self, observation):
        features = torch.tensor(feature_vector(observation), dtype=torch.float32)
        logits = self.network(features)
        mask = torch.tensor(action_mask(observation), dtype=torch.bool)
        if not bool(mask.any()):
            raise ValueError('An observation must permit at least one action')
        return logits.masked_fill(~mask, float('-inf'))

    def distribution(self, observation):
        return torch.distributions.Categorical(logits=self.logits(observation))

    def sample_action(self, observation):
        distribution = self.distribution(observation)
        index = distribution.sample()
        return deepcopy(ACTIONS[index.item()]), distribution.log_prob(index)

    def select_action(self, observation, deterministic=True):
        with torch.no_grad():
            if deterministic:
                index = int(torch.argmax(self.logits(observation)).item())
            else:
                index = int(self.distribution(observation).sample().item())
        return deepcopy(ACTIONS[index])

    def clone(self):
        return deepcopy(self)

    def parameter_hash(self):
        digest = hashlib.sha256()
        for key, value in sorted(self.state_dict().items()):
            digest.update(key.encode())
            for scalar in value.detach().cpu().flatten().tolist():
                digest.update(struct.pack('<f', scalar))
        return digest.hexdigest()

    def parameter_delta(self, other):
        return sum(float(torch.sum((a.detach() - b.detach()) ** 2).item())
                   for a, b in zip(self.parameters(), other.parameters())) ** .5

    def to_dict(self):
        return {'format': 1, 'seed': self.seed, 'hidden': self.hidden,
                'feature_dim': FEATURE_DIM, 'actions': deepcopy(list(ACTIONS)),
                'state_dict': {key: value.detach().cpu().tolist()
                               for key, value in self.state_dict().items()},
                'parameter_hash': self.parameter_hash()}

    @classmethod
    def from_dict(cls, payload):
        if payload['feature_dim'] != FEATURE_DIM or payload['actions'] != list(ACTIONS):
            raise ValueError('Checkpoint feature/action schema differs from this experiment')
        policy = cls(seed=payload['seed'], hidden=payload['hidden'])
        policy.load_state_dict({key: torch.tensor(value, dtype=torch.float32)
                                for key, value in payload['state_dict'].items()})
        if payload.get('parameter_hash') and policy.parameter_hash() != payload['parameter_hash']:
            raise ValueError('Checkpoint parameter checksum mismatch')
        return policy

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path):
        return cls.from_dict(torch.load(path, map_location='cpu', weights_only=True))

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
