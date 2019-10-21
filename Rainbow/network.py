import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

from Rainbow.NoiseLayer import NoisyDense

import typing

import numpy as np


class DDDQN(nn.Module):
    # PyCharm problems
    # https://youtrack.jetbrains.com/issue/PY-37601
    def __call__(self, *inp, **kwargs) -> typing.Any:
        return super().__call__(*inp, **kwargs)

    def __init__(self, inp_dim: int, out_dim: int, lr: float = 1e-3):
        """Input: (1, inp_dim)"""
        super().__init__()

        self.features = nn.Linear(inp_dim, 128)

        self.adv1 = NoisyDense(128, 128)
        self.adv2 = NoisyDense(128, out_dim)

        self.val1 = NoisyDense(128, 128)
        self.val2 = NoisyDense(128, 1)

        self.lr = lr
        self.opt = optim.Adam(self.parameters(), lr=self.lr)

    def predict(self, state: np.ndarray, single=True):
        if single:
            state = np.expand_dims(state, axis=0)
        state = torch.from_numpy(state).float()
        state = f.relu(self.features(state))
        advantage = self.adv2(f.relu(self.adv1(state)))
        value = self.val2(f.relu(self.val1(state)))
        return advantage + value - advantage.mean()

    def reset_noise(self):
        self.adv1.reset_noise()
        self.adv2.reset_noise()
        self.val1.reset_noise()
        self.val2.reset_noise()

    def fit(self, state, y_true, weights=None):
        y_pred = self.predict(state, single=False)
        y_true = torch.from_numpy(y_true).float()
        loss = (y_pred - y_true).pow(2).sum(dim=1)
        loss *= torch.from_numpy(weights).float()

        self.opt.zero_grad()
        loss.mean().backward()
        self.opt.step()

        self.reset_noise()

        return loss.detach().numpy()
