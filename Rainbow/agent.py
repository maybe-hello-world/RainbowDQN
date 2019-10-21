import gym
import numpy as np
import time

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from Rainbow.network import DDDQN
from Rainbow.ExpReplay import PER


class Agent:
    def __init__(
            self,
            gym_env: str,
            V_min: float = -10.0,
            V_max: float = 10.0,
            play_before_learn: int = 3000,
            num_atoms: int = 51,
            gamma: float = 0.99,
            batch_size: int = 100,
            copy_every: int = 500,
            n_step: int = 2,
            lr: float = 1e-3
    ):
        self.env_name = gym_env
        self.env = gym.make(gym_env)

        self.V_min = V_min
        self.V_max = V_max
        self.num_atoms = num_atoms
        self.support = torch.linspace(V_min, V_max, num_atoms)
        self.dz = (self.V_max - self.V_min) / (self.num_atoms - 1)

        self.replay_buffer = PER()
        self.pbl = play_before_learn

        self.obs_n = sum(self.env.observation_space.shape)
        self.act_n = self.env.action_space.n
        self.n_step = n_step

        self.gamma = gamma
        self.copy_every = copy_every
        self.batch_size = batch_size

        self.writer = SummaryWriter()

        self.q_model = DDDQN(inp_dim=self.obs_n, out_dim=self.act_n, lr=lr, V_min=V_min, V_max=V_max, num_atoms=num_atoms)
        self.target_model = DDDQN(inp_dim=self.obs_n, out_dim=self.act_n, V_min=V_min, V_max=V_max, num_atoms=num_atoms)
        self.__copy2target()

    def __del__(self):
        self.writer.close()

    def __copy2target(self):
        self.target_model.load_state_dict(self.q_model.state_dict())

    def act(self, obs: torch.Tensor):
        with torch.no_grad():
            dist = self.q_model.predict(obs.float())
            dist = dist * self.support
            action = dist.sum(dim=2).argmax(dim=1).item()
            return action

    def replay(self):
        indices, samples, weights = self.replay_buffer.sample(size=self.batch_size)

        states = torch.from_numpy(np.array([i[0] for i in samples])).float()
        actions = torch.from_numpy(np.array([i[1] for i in samples])).long()
        rewards = torch.from_numpy(np.array([i[2] for i in samples])).float()
        next_states = torch.from_numpy(np.array([i[3] for i in samples])).float()
        dones = torch.from_numpy(np.array([i[4] for i in samples])).float()

        probs = self.q_model.predict(states)
        probs_a = probs[range(self.batch_size), actions]

        with torch.no_grad():
            next_probs = self.q_model.predict(next_states)
            dist = self.support.expand_as(next_probs) * next_probs
            ns_a = dist.sum(dim=2).argmax(dim=1)
            self.target_model.reset_noise()
            next_probs = self.target_model.predict(next_states)  # double q learning
            next_probs_a = next_probs[range(self.batch_size), ns_a]

            # compute Tz
            Tz = rewards.view(-1, 1) + (torch.tensor(1) - dones).view(-1, 1) * (self.gamma ** self.n_step) * self.support.view(1, -1)
            Tz = Tz.clamp(min=self.V_min, max=self.V_max)

            b = (Tz - self.V_min) / self.dz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # some magic here
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.num_atoms - 1)) * (l == u)] += 1

            m = states.new_zeros([self.batch_size, self.num_atoms])
            offset = torch.linspace(0, ((self.batch_size - 1) * self.num_atoms), self.batch_size).unsqueeze(1).expand(
                self.batch_size, self.num_atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1),
                                  (next_probs_a * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                  (next_probs_a * (b - l.float())).view(-1))

        loss = - torch.sum(m * probs_a.log(), dim=1)
        self.q_model.zero_grad()
        weights = torch.from_numpy(np.array(weights)).float()
        (weights * loss).mean().backward()
        self.q_model.opt.step()

        loss = loss.detach().numpy()
        self.replay_buffer.update(indices=indices, tderrors=loss)
        return np.mean(loss)

    def train(self, steps: int):
        obs = self.env.reset()
        steps += self.pbl

        losses = []
        ep_results = []
        ep_sum = .0

        for i in tqdm(range(steps)):

            # n-step
            action = self.act(torch.from_numpy(obs).unsqueeze(0))
            next_obs, rew, done, _ = self.env.step(action)
            x = 1
            cur_rew = rew
            while x < self.n_step and not done:
                cur_obs = next_obs
                action = self.act(torch.from_numpy(cur_obs).unsqueeze(0))
                next_obs, rew, done, _ = self.env.step(action)
                cur_rew += rew
                x += 1
            ep_sum += rew
            self.replay_buffer.append((obs, action, rew, next_obs, done))
            obs = next_obs

            if done:
                obs = self.env.reset()
                ep_results.append(ep_sum)
                self.writer.add_scalar("Rainbow/reward", ep_sum, i)
                ep_sum = .0
            if i > self.pbl:
                loss = self.replay()
                self.writer.add_scalar("Rainbow/loss", loss, i)
                losses.append(loss)
            if i % self.copy_every == 0:
                self.__copy2target()

        return ep_results, losses

    def show(self):
        self.q_model.train(False)
        obs = self.env.reset()
        while True:
            act = self.act(torch.from_numpy(obs).unsqueeze(0))
            obs, _, done, _ = self.env.step(act)
            obs = obs if not done else self.env.reset()
            time.sleep(0.04)
            self.env.render()
