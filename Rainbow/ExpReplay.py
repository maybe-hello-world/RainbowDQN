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