import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

from Rainbow.NoiseLayer import NoisyDense

import typing


class DDDQN(nn.Module):
    # PyCharm problems
    # https://youtrack.jetbrains.com/issue/PY-37601
    def __call__(self, *inp, **kwargs) -> typing.Any:
        return super().__call__(*inp, **kwargs)

    def __init__(
            self, inp_dim: int,
            out_dim: int,
            V_min: float = -10.0,
            V_max: float = 10.0,
            lr: float = 1e-3,
            num_atoms: int = 51,
            noisy: bool = True):
        """Input: (1, inp_dim)"""
        super().__init__()

        self.V_min = V_min
        self.V_max = V_max

        self.num_atoms = num_atoms
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        self.noisy = noisy
        if noisy:
            wL = NoisyDense
        else:
            wL = nn.Linear

        self.fea1 = nn.Linear(inp_dim, 256)
        self.fea2 = nn.Linear(256, 128)

        self.adv1 = wL(128, 128)
        self.adv2 = wL(128, out_dim * num_atoms)

        self.val1 = wL(128, 128)
        self.val2 = wL(128, num_atoms)

        self.lr = lr
        self.opt = optim.Adam(self.parameters(), lr=self.lr)

    def predict(self, state: torch.Tensor):
        state = f.relu(self.fea2(f.relu(self.fea1(state))))
        advantage = self.adv2(f.relu(self.adv1(state)))
        value = self.val2(f.relu(self.val1(state)))

        value = value.view(-1, 1, self.num_atoms)
        advantage = advantage.view(-1, self.out_dim, self.num_atoms)
        q = advantage + value - advantage.mean(dim=1, keepdim=True)
        q = f.softmax(q.view(-1, self.num_atoms)).view(-1, self.out_dim, self.num_atoms)

        return q

    def reset_noise(self):
        if self.noisy:
            self.adv1.reset_noise()
            self.adv2.reset_noise()
            self.val1.reset_noise()
            self.val2.reset_noise()
