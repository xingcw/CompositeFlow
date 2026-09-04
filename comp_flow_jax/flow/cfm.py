"""Conditional flow matching.

Port of guided_flow/flow/conditional_flow_matching_old.py, restricted to the
two matchers vflow builds: cfm for the source model, ot_cfm for the adaptation
model. Both reduce to t ~ U(0,1), xt = t*x1 + (1-t)*x0 + sigma*eps, ut = x1-x0.

Two reference bugs are fixed here. See docs/COUPLING.md.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from flax import struct

from . import ot as ot_lib


@struct.dataclass
class CFMConfig:
    """Static description of a conditional flow matcher."""

    sigma: float = 0.0
    use_ot: bool = False
    ot_solver: str = "exact"
    ot_reg: float = ot_lib.DEFAULT_SINKHORN_REG
    ot_iters: int = ot_lib.DEFAULT_SINKHORN_ITERS
    eta: float = 0.05


def get_cfm(name, sigma, **kwargs):
    """Mirror of ``conditional_flow_matching_old.get_cfm``."""
    if name == "cfm":
        return CFMConfig(sigma=sigma, use_ot=False, **kwargs)
    if name == "ot_cfm":
        return CFMConfig(sigma=sigma, use_ot=True, **kwargs)
    raise NotImplementedError(
        f"cfm {name!r} is not used by this algorithm; the reference also defined "
        "'vp_cfm' and 'sb_cfm', neither of which vflow constructs")


def _pad_t(t, x):
    """``pad_t_like_x``: (B,) -> (B, 1, ..., 1) matching x's rank."""
    return t.reshape(-1, *([1] * (x.ndim - 1)))


def sample_location_and_conditional_flow(key, cfg, x0, x1, t=None,
                                         s0=None, s1=None, a0=None, a1=None):
    """Return (t, xt, ut, coupling) for one minibatch.

    coupling is (i, j) when an OT plan was applied, else None. The caller needs
    it: xt is built from x0[i], so per-row conditioning must be gathered with
    the same i. See docs/COUPLING.md.
    """
    key_ot, key_t, key_eps = jax.random.split(key, 3)

    coupling = None
    if cfg.use_ot:
        coupling = ot_lib.sample_plan_indices(
            key_ot, x0, x1, s0, s1, a0, a1, eta=cfg.eta,
            solver=cfg.ot_solver, reg=cfg.ot_reg, n_iter=cfg.ot_iters)
        x0, x1 = x0[coupling[0]], x1[coupling[1]]

    if t is None:
        t = jax.random.uniform(key_t, (x0.shape[0],), dtype=x0.dtype)
    assert t.shape[0] == x0.shape[0], "t must have a batch dimension"

    eps = jax.random.normal(key_eps, x0.shape, dtype=x0.dtype)
    mu_t = _pad_t(t, x0) * x1 + (1.0 - _pad_t(t, x0)) * x0
    xt = mu_t + cfg.sigma * eps
    ut = x1 - x0
    return t, xt, ut, coupling
