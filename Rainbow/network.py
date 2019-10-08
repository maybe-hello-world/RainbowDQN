import torch
import torch.nn as nn
import torch.optim as optim

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

        self.features = nn.Sequential(
            nn.Linear(inp_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.lr = lr
        self.opt = optim.Adam(self.parameters(), lr=self.lr)

    def predict(self, state: np.ndarray, single=True):
        if single:
            state = np.expand_dims(state, axis=0)
        state = torch.from_numpy(state).float()
        state = self.features(state)
        value = self.value(state)
        advantage = self.advantage(state)
        return advantage + value - advantage.mean()

    def fit(self, state, y_true, weights=None):
        y_pred = self.predict(state, single=False)
        y_true = torch.from_numpy(y_true).float()
        loss = (y_pred - y_true).pow(2).sum(dim=1)
        loss *= torch.from_numpy(weights).float()

        self.opt.zero_grad()
        loss.mean().backward()
        self.opt.step()
        return loss.detach().numpy()
