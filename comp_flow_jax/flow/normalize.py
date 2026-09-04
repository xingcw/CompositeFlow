"""Per-dimension standardisation, as in FlowMatching.normalize/inverse."""

import jax.numpy as jnp
from flax import struct

STD_EPS = 1e-6


@struct.dataclass
class Normalizer:
    mean: jnp.ndarray
    std: jnp.ndarray

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


def fit(x):
    """Match the reference: sample std (ddof=1), plus a 1e-6 floor.

    torch's ``Tensor.std`` defaults to the unbiased estimator while
    ``jnp.std`` defaults to the biased one, so ``ddof=1`` is required for the
    normalisers to agree.
    """
    return Normalizer(mean=jnp.mean(x, axis=0),
                      std=jnp.std(x, axis=0, ddof=1) + STD_EPS)


def fit_state_action(state, action):
    return fit(jnp.concatenate([state, action], axis=-1))


def normalize_state_action(norm, state, action):
    """Normalise ``(s, a)`` jointly and split them back apart."""
    joint = norm(jnp.concatenate([state, action], axis=-1))
    return joint[:, : state.shape[1]], joint[:, state.shape[1]:]
