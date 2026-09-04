"""Unit tests for the flow machinery: Euler ODE, normalisers, CFM, OT."""

import comp_flow_jax  # noqa: F401  (sets jax_default_matmul_precision)
import jax
import jax.numpy as jnp
import numpy as np

from comp_flow_jax.flow import normalize as norm_lib
from comp_flow_jax.flow import ot as ot_lib
from comp_flow_jax.flow.cfm import get_cfm, sample_location_and_conditional_flow
from comp_flow_jax.flow.flow_matching import bucketed_batch_count
from comp_flow_jax.flow.ode import euler_odeint


def test_euler_matches_torchdiffeq_semantics():
    """n grid points == n-1 steps, matching torchdiffeq on a linspace."""
    # dx/dt = 1 integrates exactly.
    x = euler_odeint(lambda t, x: jnp.ones_like(x), jnp.zeros((3, 2)), 11)
    assert np.allclose(np.asarray(x), 1.0, atol=1e-6)

    # dx/dt = x with 10 steps of 0.1 gives exactly (1.1)^10, not e.
    x = euler_odeint(lambda t, x: x, jnp.ones((1, 1)), 11)
    assert np.isclose(float(x[0, 0]), 1.1 ** 10, rtol=1e-5), float(x[0, 0])

    # Step count scales as documented.
    x2 = euler_odeint(lambda t, x: x, jnp.ones((1, 1)), 3)
    assert np.isclose(float(x2[0, 0]), 1.5 ** 2, rtol=1e-5)


def test_normalizer_uses_sample_std():
    """torch's Tensor.std is unbiased; jnp.std is not. fit() must use ddof=1."""
    x = jnp.asarray(np.random.RandomState(0).randn(50, 3).astype(np.float32))
    n = norm_lib.fit(x)
    expected = np.std(np.asarray(x), axis=0, ddof=1) + norm_lib.STD_EPS
    assert np.allclose(np.asarray(n.std), expected, rtol=1e-5)
    assert np.abs(np.asarray(n.inverse(n(x))) - np.asarray(x)).max() < 1e-4


def test_cfm_sigma_zero_is_exact_interpolation():
    key = jax.random.PRNGKey(0)
    x0 = jax.random.normal(key, (64, 5))
    x1 = jax.random.normal(jax.random.fold_in(key, 1), (64, 5))
    t, xt, ut, coupling = sample_location_and_conditional_flow(
        key, get_cfm("cfm", 0.0), x0, x1)
    assert coupling is None
    assert np.allclose(np.asarray(xt), np.asarray(t[:, None] * x1 + (1 - t[:, None]) * x0),
                       atol=1e-6)
    assert np.allclose(np.asarray(ut), np.asarray(x1 - x0), atol=1e-6)


def test_ot_cfm_returns_coupling_indices():
    """The caller must be able to gather the conditioning with the same indices."""
    key = jax.random.PRNGKey(0)
    x0 = jax.random.normal(key, (32, 5))
    x1 = jax.random.normal(jax.random.fold_in(key, 1), (32, 5))
    _, _, _, coupling = sample_location_and_conditional_flow(
        key, get_cfm("ot_cfm", 0.0), x0, x1)
    assert coupling is not None
    i, j = coupling
    assert i.shape == (32,) and j.shape == (32,)
    assert int(i.max()) < 32 and int(j.max()) < 32


def test_sinkhorn_close_to_exact():
    rng = np.random.RandomState(0)
    x0 = jnp.asarray(rng.randn(256, 8).astype(np.float32))
    x1 = jnp.asarray(rng.randn(256, 8).astype(np.float32))
    cost = ot_lib.squared_cost(x0, x1)
    c = np.asarray(cost)
    exact = float((np.asarray(ot_lib.exact_plan(cost)) * c).sum())
    approx = float((np.asarray(ot_lib.sinkhorn_plan(cost)) * c).sum())
    rel = abs(approx - exact) / exact
    assert rel < 0.01, f"sinkhorn transport cost is {rel:.3%} off the optimum"
    return exact, approx, rel


def test_eta_changes_the_cost():
    """The reference's eta never reached the cost matrix; this one must."""
    rng = np.random.RandomState(0)
    a = lambda n, d: jnp.asarray(rng.randn(n, d).astype(np.float32))
    x0, x1, s0, s1, a0, a1 = a(64, 5), a(64, 5), a(64, 5), a(64, 5), a(64, 2), a(64, 2)
    c0 = np.asarray(ot_lib.squared_cost(x0, x1, s0, s1, a0, a1, eta=0.0))
    c1 = np.asarray(ot_lib.squared_cost(x0, x1, s0, s1, a0, a1, eta=0.5))
    assert not np.allclose(c0, c1), "eta had no effect on the transport cost"


def test_batch_count_bucketing():
    """Bucketing must round up (never drop rows) and cut the shape count."""
    for n_rows in range(1, 40000, 137):
        nb = bucketed_batch_count(n_rows, 1024)
        assert nb * 1024 >= n_rows, f"{n_rows}: {nb} batches would drop rows"
        assert nb % 4 == 0 or nb == 4
    sizes = range(10000, 40001, 500)
    exact = {max(1, -(-int(n * 0.9) // 1024)) for n in sizes}
    bucketed = {bucketed_batch_count(int(n * 0.9), 1024) for n in sizes}
    assert len(bucketed) < len(exact) / 3
    return len(exact), len(bucketed)


if __name__ == "__main__":
    test_euler_matches_torchdiffeq_semantics()
    print("euler == torchdiffeq semantics    OK")
    test_normalizer_uses_sample_std()
    print("normaliser uses sample std        OK")
    test_cfm_sigma_zero_is_exact_interpolation()
    print("cfm sigma=0 interpolation         OK")
    test_ot_cfm_returns_coupling_indices()
    print("ot-cfm exposes coupling indices   OK")
    e, ap, rel = test_sinkhorn_close_to_exact()
    print(f"sinkhorn vs exact                 OK  ({ap:.4f} vs {e:.4f}, {rel:.3%} off)")
    test_eta_changes_the_cost()
    print("eta reaches the transport cost    OK")
    a, b = test_batch_count_bucketing()
    print(f"batch-count bucketing             OK  ({a} shapes -> {b})")
