import torch
import torch.nn as nn
import torch.nn.functional as f

import typing


class NoisyDense(nn.Module):
    # PyCharm problems
    # https://youtrack.jetbrains.com/issue/PY-37601
    def __call__(self, *inp, **kwargs) -> typing.Any:
        return super().__call__(*inp, **kwargs)

    def __init__(self, inp_dim, out_dim, std_init=0.5):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.std_init = std_init

        self.weights_mu = nn.Parameter(torch.empty(out_dim, inp_dim).float())
        self.weights_sigma = nn.Parameter(torch.empty(out_dim, inp_dim).float())
        self.register_buffer('weights_epsilon', torch.empty(out_dim, inp_dim))

        self.biases_mu = nn.Parameter(torch.empty(out_dim).float())
        self.biases_sigma = nn.Parameter(torch.empty(out_dim).float())
        self.register_buffer('biases_epsilon', torch.empty(out_dim))

        self.fill_params()
        self.reset_noise()

    def fill_params(self):
        with torch.no_grad():
            mu_range = 1 / (self.inp_dim ** (1/2))
            self.weights_mu.detach().uniform_(-mu_range, mu_range)
            self.weights_sigma.fill_(self.std_init / (self.inp_dim ** (1/2)))

            self.biases_mu.uniform_(-mu_range, mu_range)
            self.biases_sigma.fill_(self.std_init / (self.out_dim ** (1/2)))

    def reset_noise(self):
        def _scale_noise(size):
            x = torch.randn(size)
            return x.sign() * (x.abs().sqrt_())

        e_in = _scale_noise(self.inp_dim)
        e_out = _scale_noise(self.out_dim)
        self.weights_epsilon.copy_(e_out.ger(e_in))
        self.biases_epsilon.copy_(e_out)

    def forward(self, inp: torch.Tensor):
        weights = self.weights_mu
        biases = self.biases_mu
        if self.training:
            weights = weights + self.weights_sigma * self.weights_epsilon
            biases = biases + self.biases_sigma * self.biases_epsilon
        return f.linear(inp, weights, biases)
