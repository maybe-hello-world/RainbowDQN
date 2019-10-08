import itertools
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
            gamma: float = 0.99,
            batch_size: int = 100,
            copy_every: int = 500,
            n_step: int = 2
    ):
        self.env_name = gym_env
        self.env = gym.make(gym_env)

        self.replay_buffer = PER()

        self.obs_n = sum(self.env.observation_space.shape)
        self.act_n = self.env.action_space.n
        self.n_step = n_step

        self.gamma = gamma
        self.copy_every = copy_every
        self.batch_size = batch_size

        self.writer = SummaryWriter()

        self.epsilon_f = lambda x: (max(1 - i / x, 0.01) for i in itertools.count())
        self.epsilon_gen = None

        self.q_model = DDDQN(self.obs_n, self.act_n, lr=1e-2)
        self.target_model = DDDQN(self.obs_n, self.act_n)
        self.__copy2target()

    def __del__(self):
        self.writer.close()

    def __copy2target(self):
        self.target_model.load_state_dict(self.q_model.state_dict())

    def act(self, obs):
        if np.random.sample() < next(self.epsilon_gen):
            return np.random.randint(0, self.act_n)
        with torch.no_grad():
            q_values = self.q_model.predict(obs)
            return np.argmax(q_values.numpy())

    def __count_rew(self, data):
        s, a, r, s_next, d = data
        with torch.no_grad():
            if not d:
                qv = self.q_model.predict(s_next).numpy()
                action = np.argmax(qv)
                q_value = self.target_model.predict(s_next).squeeze().numpy()[action]
                opt_q = (self.gamma ** self.n_step) * q_value
                r += opt_q

            qv = np.squeeze(self.q_model.predict(s).numpy())
            qv[a] = r
        return np.concatenate([s, qv])

    def replay(self):
        indices, samples, weights = self.replay_buffer.sample(size=self.batch_size)
        samples = np.array(list(map(self.__count_rew, samples)))
        x = samples[:, :self.obs_n]
        y = samples[:, self.obs_n:]
        weights = np.array(weights)
        tderrors = self.q_model.fit(x, y, weights=weights)
        self.replay_buffer.update(indices=indices, tderrors=tderrors)
        return tderrors.mean() / weights.mean()

    def train(self, steps: int):
        self.epsilon_gen = self.epsilon_f(steps // 2)

        obs = self.env.reset()
        steps += self.batch_size * 3

        losses = []
        ep_results = []
        ep_sum = .0

        for i in tqdm(range(steps)):

            # n-step
            action = self.act(obs)
            next_obs, rew, done, _ = self.env.step(action)
            x = 1
            cur_rew = rew
            while x < self.n_step and not done:
                obs = next_obs
                action = self.act(obs)
                next_obs, rew, done, _ = self.env.step(action)
                cur_rew += rew
                x += 1
            ep_sum += rew
            self.replay_buffer.append((obs, action, rew, next_obs, done))
            obs = next_obs

            if done:
                obs = self.env.reset()
                ep_results.append(ep_sum)
                self.writer.add_scalar("Dueling/reward", ep_sum, i)
                ep_sum = .0
            if i > self.batch_size * 3:
                loss = self.replay()
                self.writer.add_scalar("Dueling/loss", loss, i)
                losses.append(loss)
            if i % self.copy_every == 0:
                self.__copy2target()

        return ep_results, losses

    def show(self):
        obs = self.env.reset()
        while True:
            act = self.act(obs)
            obs, _, done, _ = self.env.step(act)
            obs = obs if not done else self.env.reset()
            time.sleep(0.04)
            self.env.render()
