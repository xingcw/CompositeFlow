"""Dense layer with torch-compatible init, used by every network here."""

import flax.linen as nn
import jax.numpy as jnp

from .init import torch_bias_init, torch_kernel_init


class TorchDense(nn.Module):
    """``nn.Dense`` with ``torch.nn.Linear``'s default initialisation."""

    features: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1]
        return nn.Dense(
            features=self.features,
            use_bias=self.use_bias,
            kernel_init=torch_kernel_init,
            bias_init=torch_bias_init(fan_in),
        )(x)
