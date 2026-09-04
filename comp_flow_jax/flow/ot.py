"""Minibatch optimal-transport couplings for OT-CFM.

Port of guided_flow/flow/optimal_transport_old.py. Solver timings, and why the
exact path uses scipy rather than pot.emd, are in docs/BACKEND.md.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_SINKHORN_REG = 0.05
DEFAULT_SINKHORN_ITERS = 200


def squared_cost(x0, x1, s0=None, s1=None, a0=None, a1=None, eta=0.0):
    """``||x0-x1||^2 + eta * (||s0-s1||^2 + ||a0-a1||^2)``.

    With ``eta=0`` this is the plain squared-Euclidean cost of ``get_map``;
    the extra terms are ``get_map_condition_version``, which makes the coupling
    prefer pairs whose conditioning (state, action) also matches.
    """

    def sqdist(u, v):
        u = u.reshape(u.shape[0], -1)
        v = v.reshape(v.shape[0], -1)
        return jnp.sum((u[:, None, :] - v[None, :, :]) ** 2, axis=-1)

    m = sqdist(x0, x1)
    if eta != 0.0 and s0 is not None:
        m = m + eta * (sqdist(s0, s1) + sqdist(a0, a1))
    return m


@functools.partial(jax.jit, static_argnames=("n_iter",))
def sinkhorn_plan(cost, reg=DEFAULT_SINKHORN_REG, n_iter=DEFAULT_SINKHORN_ITERS):
    """Entropic OT plan for uniform marginals, in the log domain.

    Log-domain throughout so that small ``reg`` does not underflow.
    """
    n, m = cost.shape
    log_a = -jnp.log(n)
    log_b = -jnp.log(m)
    k = -cost / reg

    def body(carry, _):
        f, g = carry
        f = reg * (log_a - jax.scipy.special.logsumexp(k + g[None, :] / reg, axis=1))
        g = reg * (log_b - jax.scipy.special.logsumexp(k + f[:, None] / reg, axis=0))
        return (f, g), None

    (f, g), _ = jax.lax.scan(body, (jnp.zeros(n), jnp.zeros(m)), None, length=n_iter)
    plan = jnp.exp(k + f[:, None] / reg + g[None, :] / reg)
    return plan / jnp.sum(plan)


def _hungarian_plan_host(cost):
    """Exact plan as a (n, n) permutation matrix scaled to sum to 1."""
    from scipy.optimize import linear_sum_assignment

    cost = np.asarray(cost, dtype=np.float64)
    n = cost.shape[0]
    rows, cols = linear_sum_assignment(cost)
    plan = np.zeros_like(cost)
    plan[rows, cols] = 1.0 / n
    return plan.astype(np.float32)


def exact_plan(cost):
    """Exact OT plan, computed on the host via ``jax.pure_callback``."""
    out = jax.ShapeDtypeStruct(cost.shape, jnp.float32)
    return jax.pure_callback(_hungarian_plan_host, out, cost)


def ot_plan(cost, solver="exact", reg=DEFAULT_SINKHORN_REG,
            n_iter=DEFAULT_SINKHORN_ITERS):
    if solver == "sinkhorn":
        return sinkhorn_plan(cost, reg, n_iter)
    if solver == "exact":
        return exact_plan(cost)
    raise ValueError(f"unknown OT solver {solver!r}; expected 'sinkhorn' or 'exact'")


def sample_map(key, plan, batch_size):
    """Draw ``batch_size`` index pairs from the plan, proportionally to it.

    Port of ``OTPlanSampler.sample_map`` (which samples *with* replacement, so
    the coupled minibatch is a bootstrap of the matched pairs, not the matching
    itself).
    """
    n_cols = plan.shape[1]
    flat = plan.reshape(-1)
    flat = flat / jnp.sum(flat)
    choices = jax.random.choice(key, flat.shape[0], shape=(batch_size,),
                                replace=True, p=flat)
    return choices // n_cols, choices % n_cols


def sample_plan_indices(key, x0, x1, s0=None, s1=None, a0=None, a1=None, eta=0.0,
                        solver="exact", reg=DEFAULT_SINKHORN_REG,
                        n_iter=DEFAULT_SINKHORN_ITERS, batch_size=None):
    """Index pairs ``(i, j)`` drawn from the OT coupling of ``x0`` and ``x1``.

    Returned as indices rather than gathered rows so the caller can carry other
    per-row tensors (the conditioning vector, above all) through the same
    permutation.
    """
    cost = squared_cost(x0, x1, s0, s1, a0, a1, eta)
    plan = ot_plan(cost, solver, reg, n_iter)
    return sample_map(key, plan, batch_size or x0.shape[0])


def sample_plan(key, x0, x1, s0=None, s1=None, a0=None, a1=None, eta=0.0,
                solver="exact", reg=DEFAULT_SINKHORN_REG,
                n_iter=DEFAULT_SINKHORN_ITERS, batch_size=None):
    """Reorder ``(x0, x1)`` according to an OT coupling."""
    i, j = sample_plan_indices(key, x0, x1, s0, s1, a0, a1, eta, solver, reg,
                               n_iter, batch_size)
    return x0[i], x1[j]
