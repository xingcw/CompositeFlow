"""PyTorch-compatible parameter initialisers.

Flax defaults to lecun_normal kernels and zero biases; torch.nn.Linear uses
U(-1/sqrt(fan_in), +1/sqrt(fan_in)) for both. This port uses torch's so the
networks start from the same distribution as the reference.
"""

import jax
import jax.numpy as jnp


def torch_kernel_init(key, shape, dtype=jnp.float32):
    """Kernel of shape (fan_in, fan_out), drawn as torch.nn.Linear would."""
    fan_in = shape[0]
    bound = 1.0 / jnp.sqrt(jnp.asarray(fan_in, dtype))
    return jax.random.uniform(key, shape, dtype, minval=-1.0, maxval=1.0) * bound


def torch_bias_init(fan_in):
    """Bias initialiser matching torch.nn.Linear for a given fan_in."""

    def init(key, shape, dtype=jnp.float32):
        bound = 1.0 / jnp.sqrt(jnp.asarray(fan_in, dtype))
        return jax.random.uniform(key, shape, dtype, minval=-1.0, maxval=1.0) * bound

    return init
