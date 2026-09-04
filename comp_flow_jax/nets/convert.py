"""Load reference PyTorch weights into the flax modules.

torch stores Linear weights as (out, in) and LayerNorm scale as "weight"; flax
uses (in, out) and "scale".
"""

import jax.numpy as jnp


def _dense(state, prefix):
    return {"kernel": jnp.asarray(state[f"{prefix}.weight"].T),
            "bias": jnp.asarray(state[f"{prefix}.bias"])}


def _layernorm(state, prefix):
    return {"scale": jnp.asarray(state[f"{prefix}.weight"]),
            "bias": jnp.asarray(state[f"{prefix}.bias"])}


def residual_mlp_guidance_from_torch(state, num_layers):
    """Map a ``ResidualMLPGuidance`` state_dict onto this module's param tree.

    Module creation order in ``ResidualMLPGuidance.__call__`` fixes the flax
    names: the two time_mlp Denses come first (TorchDense_0/_1), then the input
    projection (TorchDense_2), then the ResidualMLP.
    """
    params = {
        "TorchDense_0": {"Dense_0": _dense(state, "time_mlp.1")},
        "TorchDense_1": {"Dense_0": _dense(state, "time_mlp.3")},
        "TorchDense_2": {"Dense_0": _dense(state, "proj")},
        "ResidualMLP_0": {
            # network.0 is the input Linear; network.1..num_layers are the
            # residual blocks; network.{num_layers+1} is the trailing LayerNorm.
            "TorchDense_0": {"Dense_0": _dense(state, "residual_mlp.network.0")},
            "TorchDense_1": {"Dense_0": _dense(state, "residual_mlp.final_linear")},
            "LayerNorm_0": _layernorm(state, f"residual_mlp.network.{num_layers + 1}"),
        },
    }
    for i in range(num_layers):
        params["ResidualMLP_0"][f"ResidualBlock_{i}"] = {
            "LayerNorm_0": _layernorm(state, f"residual_mlp.network.{i + 1}.ln"),
            "TorchDense_0": {"Dense_0": _dense(state, f"residual_mlp.network.{i + 1}.linear")},
        }
    return {"params": params}


def mlp_network_from_torch(state, prefix, flax_prefix=""):
    """Map an ``MLPNetwork`` state_dict slice (``network.{0,2,4}``)."""
    return {
        f"{flax_prefix}TorchDense_0": {"Dense_0": _dense(state, f"{prefix}.network.0")},
        f"{flax_prefix}TorchDense_1": {"Dense_0": _dense(state, f"{prefix}.network.2")},
        f"{flax_prefix}TorchDense_2": {"Dense_0": _dense(state, f"{prefix}.network.4")},
    }


def policy_from_torch(state):
    """``vflow.Policy`` -> ``TanhGaussianPolicy`` params."""
    return {"params": {"MLPNetwork_0": mlp_network_from_torch(state, "network")}}


def double_q_from_torch(state):
    """``vflow.DoubleQFunc`` -> ``DoubleQCritic`` params."""
    return {"params": {
        "MLPNetwork_0": mlp_network_from_torch(state, "network1"),
        "MLPNetwork_1": mlp_network_from_torch(state, "network2"),
    }}
