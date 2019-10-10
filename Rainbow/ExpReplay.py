from typing import Tuple
from collections import deque
import numpy as np


class ER:
    def __init__(self, size: int = 10000):
        self.buffer = deque(maxlen=size)

    def append(self, data: Tuple[np.ndarray, int, float, np.ndarray, bool]):
        """(s, a, r, s', done)"""
        self.buffer.append(data)

    def clear(self):
        self.buffer.clear()

    def sample(self, size: int = 100):
        indices = np.random.randint(0, len(self.buffer), size=size)
        return (self.buffer[i] for i in indices)


class PER(ER):
    def __init__(self, size: int = 10000):
        super().__init__(size)
        self.alpha = 0.5
        self.beta = 0.5

    def append(self, data: Tuple[np.ndarray, int, float, np.ndarray, bool]):
        """(s, a, r, s', done)"""
        data = (*data, self.max_priority() ** self.alpha)
        self.buffer.append(data)

    def max_priority(self):
        return max((i[-1] for i in self.buffer)) if len(self.buffer) > 0 else 1.0

    def sample(self, size: int = 100):
        sum_weights = sum(i[-1] for i in self.buffer)
        probs = [i[-1] / sum_weights for i in self.buffer]
        indices = np.random.choice(np.arange(len(self.buffer)), size=size, p=probs)
        weights = [(1/len(self.buffer)) * (1/probs[i]) ** self.beta for i in indices]
        batch = [self.buffer[i] for i in indices]
        return indices, [i[:-1] for i in batch], weights

    def update(self, indices, tderrors):
        for idx, error in zip(indices, tderrors):
            self.buffer[idx] = (*self.buffer[idx][:-1], error ** self.alpha + 0.0001)
