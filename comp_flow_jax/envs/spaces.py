"""Minimal Box space, so the port does not need gym."""

import numpy as np


class Box:
    def __init__(self, low, high, shape=None, dtype=np.float32):
        if shape is None:
            low, high = np.asarray(low), np.asarray(high)
            shape = low.shape if low.ndim else high.shape
        self.shape = tuple(shape)
        self.dtype = dtype
        self.low = np.broadcast_to(np.asarray(low, dtype=dtype), self.shape).copy()
        self.high = np.broadcast_to(np.asarray(high, dtype=dtype), self.shape).copy()
        self._rng = np.random.RandomState()

    def seed(self, seed=None):
        self._rng = np.random.RandomState(seed)
        return [seed]

    def sample(self):
        # Matches gym's Box.sample for the bounded-on-both-sides case, which is
        # the only one these action spaces hit.
        return self._rng.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x):
        x = np.asarray(x)
        return x.shape == self.shape and bool(np.all(x >= self.low) and np.all(x <= self.high))

    def __repr__(self):
        return f"Box({self.low.min()}, {self.high.max()}, {self.shape}, {self.dtype.__name__})"
