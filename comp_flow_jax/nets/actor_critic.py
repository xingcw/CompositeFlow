"""SAC actor and double-Q critic.

Port of the MLPNetwork / Policy / DoubleQFunc classes in vflow.py. The actor is
a tanh-squashed diagonal Gaussian; its log-density is written out directly
rather than through a TransformedDistribution, which is the same algebra
without the atanh round-trip.
"""

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from .layers import TorchDense

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class MLPNetwork(nn.Module):
    """Linear -> act -> Linear -> act -> Linear, as in the reference."""

    output_dim: int
    hidden_size: int = 256
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = getattr(nn, self.activation)
        x = act(TorchDense(self.hidden_size)(x))
        x = act(TorchDense(self.hidden_size)(x))
        return TorchDense(self.output_dim)(x)


class TanhGaussianPolicy(nn.Module):
    action_dim: int
    max_action: float
    hidden_size: int = 256
    activation: str = "relu"

    @nn.compact
    def __call__(self, state):
        """Return ``(mu, log_std)`` of the pre-squash Gaussian."""
        out = MLPNetwork(2 * self.action_dim, self.hidden_size, self.activation)(state)
        mu, log_std = jnp.split(out, 2, axis=-1)
        return mu, jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)


def tanh_gaussian_sample(key, mu, log_std, max_action):
    """Sample an action and its log-density under the tanh-squashed Gaussian."""
    std = jnp.exp(log_std)
    eps = jax.random.normal(key, mu.shape, dtype=mu.dtype)
    x = mu + std * eps
    raw_action = jnp.tanh(x)

    # log N(x; mu, std)
    log_normal = -0.5 * ((x - mu) / std) ** 2 - log_std - 0.5 * jnp.log(2.0 * jnp.pi)
    # TanhTransform.log_abs_det_jacobian, softplus form for stability
    log_det = 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))
    log_prob = jnp.sum(log_normal - log_det, axis=-1)

    return raw_action * max_action, log_prob


def deterministic_action(mu, max_action):
    """The evaluation action: tanh of the Gaussian mean."""
    return jnp.tanh(mu) * max_action


class DoubleQCritic(nn.Module):
    """Two independent Q heads; ``__call__`` returns both."""

    hidden_size: int = 256
    activation: str = "relu"

    @nn.compact
    def __call__(self, state, action):
        # (B, state_dim) with (B, N, action_dim) -> broadcast the state, as the
        # reference does for its CQL-style multi-action queries.
        if action.ndim == 3 and state.ndim == 2:
            n = action.shape[1]
            batch = state.shape[0]
            state = jnp.repeat(state[:, None, :], n, axis=1).reshape(-1, state.shape[-1])
            action = action.reshape(-1, action.shape[-1])
            reshape_to = (batch, n)
        else:
            reshape_to = None

        x = jnp.concatenate([state, action], axis=-1)
        q1 = jnp.squeeze(MLPNetwork(1, self.hidden_size, self.activation)(x), -1)
        q2 = jnp.squeeze(MLPNetwork(1, self.hidden_size, self.activation)(x), -1)

        if reshape_to is not None:
            q1 = q1.reshape(reshape_to)
            q2 = q2.reshape(reshape_to)
        return q1, q2
