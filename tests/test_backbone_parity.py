"""Check the flax backbone against the PyTorch reference, layer for layer.

Regenerate the fixture with::

    <torch venv>/bin/python tests/parity_dump_torch.py

Requires float32 matmul precision, which ``comp_flow_jax/__init__.py`` sets on
import; under the TPU default (bfloat16) the forward drifts by 4e-3 relative.
"""

import comp_flow_jax  # noqa: F401  (sets jax_default_matmul_precision)

import pathlib

import jax.numpy as jnp
import numpy as np

from comp_flow_jax.nets.backbone import ResidualMLPGuidance, SinusoidalPosEmb
from comp_flow_jax.nets.convert import residual_mlp_guidance_from_torch

FIXTURE = pathlib.Path(__file__).resolve().parent / "_parity_backbone.npz"
D_IN, COND_DIM, WIDTH, LAYERS, DIM_T = 11, 14, 512, 6, 128


def _load():
    blob = np.load(FIXTURE)
    state = {k[len("w/"):]: blob[k] for k in blob.files if k.startswith("w/")}
    return blob, state


def test_time_embedding_matches():
    blob, _ = _load()
    ours = SinusoidalPosEmb(DIM_T).apply({}, jnp.asarray(blob["t"]))
    err = float(jnp.abs(ours - blob["t_emb"]).max())
    # ~1 ULP at magnitude 1; the embedding is elementwise, so this is just
    # float32 rounding in exp/sin/cos, not a structural difference.
    assert err < 1e-5, f"sinusoidal embedding differs by {err}"
    return err


def test_forward_matches():
    blob, state = _load()
    model = ResidualMLPGuidance(d_in=D_IN, cond_dim=COND_DIM, mlp_width=WIDTH,
                                num_layers=LAYERS, activation="relu")
    params = residual_mlp_guidance_from_torch(state, LAYERS)
    ours = model.apply(params, jnp.asarray(blob["x"]), jnp.asarray(blob["t"]),
                       jnp.asarray(blob["cond"]))
    ref = blob["y"]
    abs_err = float(jnp.abs(ours - ref).max())
    rel_err = abs_err / float(np.abs(ref).max())
    assert abs_err < 1e-4, f"forward differs by {abs_err} (rel {rel_err})"
    return abs_err, rel_err


if __name__ == "__main__":
    e = test_time_embedding_matches()
    print(f"SinusoidalPosEmb  max|diff| = {e:.3e}   OK")
    a, r = test_forward_matches()
    print(f"ResidualMLPGuidance max|diff| = {a:.3e}  (rel {r:.3e})   OK")
