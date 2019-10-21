from typing import Tuple

import numpy as np


class ER:
    def __init__(self, state_size: tuple, max_size: int = 10000):
        self.states = np.empty((max_size, *state_size), dtype=np.float)
        self.actions = np.empty(max_size, dtype=np.int)
        self.rewards = np.empty(max_size, dtype=np.float)
        self.next_states = np.empty((max_size, *state_size), dtype=np.float)
        self.dones = np.empty(max_size, dtype=np.bool)
        self.cur_i = 0
        self.max_i = 0
        self.max_size = max_size

    def append(self, data: Tuple[np.ndarray, int, float, np.ndarray, bool]):
        """(s, a, r, s', done)"""
        self.states[self.cur_i, :] = data[0]
        self.actions[self.cur_i] = data[1]
        self.rewards[self.cur_i] = data[2]
        self.next_states[self.cur_i] = data[3]
        self.dones[self.cur_i] = data[4]
        self.cur_i = (self.cur_i + 1) % self.max_size
        self.max_i = min(self.max_i + 1, self.max_size - 1)

    def clear(self):
        self.cur_i = 0
        self.max_i = 0

    def sample(self, batch_size: int = 100):
        indices = np.random.randint(0, self.max_i, size=batch_size)
        return \
            self.states.take(indices, axis=0),\
            self.actions.take(indices),\
            self.rewards.take(indices),\
            self.next_states.take(indices, axis=0),\
            self.dones.take(indices)


class PER(ER):
    def __init__(self, state_size: tuple, max_size: int = 10000):
        super().__init__(state_size, max_size)
        self.weights = np.empty(max_size, dtype=np.float)
        self.alpha = 0.5
        self.beta = 0.5

    def append(self, data: Tuple[np.ndarray, int, float, np.ndarray, bool]):
        """(s, a, r, s', done)"""
        self.weights[self.cur_i] = self.max_priority() ** self.alpha
        super().append(data)

    def max_priority(self):
        if self.max_i == 0:
            return 1.0
        return np.max(self.weights[:self.max_i])

    def sample(self, batch_size: int = 100):
        sum_weights = np.sum(self.weights[:self.max_i])
        probs = self.weights[:self.max_i] / sum_weights
        indices = np.random.choice(np.arange(self.max_i), size=batch_size, p=probs)
        weights = np.abs((1 / self.max_i) * (1/probs[indices])) ** self.beta

        return indices,\
            self.states.take(indices, axis=0),\
            self.actions.take(indices),\
            self.rewards.take(indices),\
            self.next_states.take(indices, axis=0),\
            self.dones.take(indices),\
            weights

    def update(self, indices, tderrors):
        self.weights[indices] = np.abs(tderrors) ** self.alpha + 0.01
