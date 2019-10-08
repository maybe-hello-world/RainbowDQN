import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np


class DQN:
    def __init__(self, inp_dim: int, out_dim: int, lr: float = 1e-3):
        """Input: (1, inp_dim)"""

        self.model = nn.Sequential(
            nn.Linear(inp_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )

        self.lr = lr
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)

    def predict(self, state: np.ndarray, single=True):
        if single:
            state = np.expand_dims(state, axis=0)
        return self.model(torch.from_numpy(state).float())

    def fit(self, state, y_true, weights=None):
        y_pred = self.model(torch.from_numpy(state).float())
        y_true = torch.from_numpy(y_true).float()
        loss = (y_pred - y_true).pow(2).sum(dim=1)
        loss *= torch.from_numpy(np.array(weights)).float()

        self.opt.zero_grad()
        loss.mean().backward()
        self.opt.step()
        return loss.detach().numpy()

    def get_state(self):
        return self.model.state_dict()

    def set_state(self, state_dict):
        self.model.load_state_dict(state_dict)
