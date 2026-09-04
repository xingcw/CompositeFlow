"""Flow-matching velocity network.

Port of guided_flow/backbone/mlp.py: a residual MLP over the flow state, a
sinusoidal time embedding and a (state, action) conditioning vector. The time
embedding is added to the projection of concat(x, cond), not concatenated.
"""

import math
from typing import Optional

import flax.linen as nn
import jax.numpy as jnp

from .layers import TorchDense

# torch LayerNorm eps; flax defaults to 1e-6.
LN_EPS = 1e-5


class SinusoidalPosEmb(nn.Module):
    """Fixed sinusoidal embedding, ``cat(sin, cos)`` in that order."""

    dim: int
    max_period: int = 10000

    @nn.compact
    def __call__(self, t):
        assert self.dim >= 2, "dim must be >= 2 for SinusoidalPosEmb"
        half_dim = self.dim // 2
        scale = math.log(self.max_period) / (half_dim - 1)
        freqs = jnp.exp(-scale * jnp.arange(half_dim, dtype=jnp.float32))

        t = jnp.atleast_1d(t)
        if t.ndim == 1:
            t = t[:, None]
        t = t.astype(jnp.float32)

        args = t * freqs[None, :]                      # (B, half_dim)
        emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
        if self.dim % 2 == 1:
            emb = jnp.concatenate([emb, jnp.zeros_like(emb[:, :1])], axis=-1)
        return emb


class ResidualBlock(nn.Module):
    """``x + Linear(act(LayerNorm(x)))``."""

    features: int
    activation: str = "relu"
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm(epsilon=LN_EPS)(x) if self.layer_norm else x
        h = getattr(nn, self.activation)(h)
        return x + TorchDense(self.features)(h)


class ResidualMLP(nn.Module):
    input_width: int
    depth: int
    output_dim: int
    activation: str = "relu"
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        h = TorchDense(self.input_width)(x)
        for _ in range(self.depth):
            h = ResidualBlock(self.input_width, self.activation, self.layer_norm)(h)
        if self.layer_norm:
            h = nn.LayerNorm(epsilon=LN_EPS)(h)
        h = getattr(nn, self.activation)(h)
        return TorchDense(self.output_dim)(h)


class ResidualMLPGuidance(nn.Module):
    """Conditional velocity field ``v(x, t | cond)``."""

    d_in: int
    cond_dim: Optional[int] = None
    dim_t: int = 128
    mlp_width: int = 1024
    num_layers: int = 6
    activation: str = "relu"
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x, timesteps, cond=None):
        if self.cond_dim is not None:
            assert cond is not None, "conditional model called without cond"
            x = jnp.concatenate([x, cond], axis=-1)

        # time_mlp: SinusoidalPosEmb -> Linear -> SiLU -> Linear
        te = SinusoidalPosEmb(self.dim_t)(timesteps)
        te = TorchDense(self.dim_t)(te)
        te = nn.silu(te)
        te = TorchDense(self.dim_t)(te)

        h = TorchDense(self.dim_t)(x) + te
        return ResidualMLP(
            input_width=self.mlp_width,
            depth=self.num_layers,
            output_dim=self.d_in,
            activation=self.activation,
            layer_norm=self.layer_norm,
        )(h)
