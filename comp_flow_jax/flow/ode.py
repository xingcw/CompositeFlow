"""Fixed-step Euler integration of the flow ODE.

Matches torchdiffeq.odeint(method="euler") on a linspace: n grid points means
n-1 steps. Written as a lax.scan so the rollout stays in one XLA program.
"""

import jax
import jax.numpy as jnp


def euler_odeint(velocity_fn, x0, n_points, t0=0.0, t1=1.0):
    """Integrate ``dx/dt = velocity_fn(t, x)`` from ``t0`` to ``t1``.

    ``velocity_fn`` takes ``(t_scalar, x)`` and returns ``dx/dt``. ``n_points``
    matches ``torch.linspace(t0, t1, n_points)``, i.e. ``n_points - 1`` steps.
    """
    if n_points < 2:
        raise ValueError(f"need at least 2 grid points, got {n_points}")
    ts = jnp.linspace(t0, t1, n_points, dtype=x0.dtype)
    dts = ts[1:] - ts[:-1]

    def step(x, td):
        t, dt = td
        return x + dt * velocity_fn(t, x), None

    x1, _ = jax.lax.scan(step, x0, (ts[:-1], dts))
    return x1


def make_velocity_fn(apply_fn, params, cond):
    """Wrap a flax module into ``f(t, x)`` with a fixed conditioning batch."""

    def velocity(t, x):
        t_col = jnp.full((x.shape[0], 1), t, dtype=x.dtype)
        return apply_fn(params, x, t_col, cond)

    return velocity


def flow_forward(apply_fn, params, x0, cond, n_points):
    """Run the conditional flow from ``x0`` to ``t=1``."""
    return euler_odeint(make_velocity_fn(apply_fn, params, cond), x0, n_points)
