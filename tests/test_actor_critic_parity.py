"""Check the SAC actor/critic against the PyTorch reference in vflow.py.

Regenerate the fixture with ``<torch venv>/bin/python tests/parity_dump_torch.py``.
"""

import pathlib

import comp_flow_jax  # noqa: F401  (sets jax_default_matmul_precision)
import jax
import jax.numpy as jnp
import numpy as np

from comp_flow_jax.nets.actor_critic import (
    DoubleQCritic, TanhGaussianPolicy, deterministic_action)
from comp_flow_jax.nets.convert import double_q_from_torch, policy_from_torch

FIXTURE = pathlib.Path(__file__).resolve().parent / "_parity_actor_critic.npz"

# Linear/matmul paths reach float32 parity with the reference.
TOL = 1e-5
# tanh and softplus do not: this TPU evaluates them with a hardware
# approximation good to ~4.4e-5 (tanh) and ~1.0e-4 (softplus) against numpy,
# where JAX-on-CPU reaches 1.8e-7. Anything squashed therefore carries ~1e-4 of
# absolute error that no amount of care in the port removes. It is far below
# the noise floor of SAC's gradient signal, but it is why the squashed
# quantities get their own tolerance.
TOL_SQUASHED = 5e-4


def _load():
    blob = np.load(FIXTURE)
    pw = {k[len("pw/"):]: blob[k] for k in blob.files if k.startswith("pw/")}
    cw = {k[len("cw/"):]: blob[k] for k in blob.files if k.startswith("cw/")}
    return blob, pw, cw


def test_policy_head():
    blob, pw, _ = _load()
    actor = TanhGaussianPolicy(action_dim=3, max_action=1.0, hidden_size=256)
    mu, log_std = actor.apply(policy_from_torch(pw), jnp.asarray(blob["s"]))
    e_mu = float(jnp.abs(mu - blob["mu"]).max())
    e_ls = float(jnp.abs(log_std - blob["log_std"]).max())
    e_det = float(jnp.abs(deterministic_action(mu, 1.0) - blob["mean_action"]).max())
    assert max(e_mu, e_ls) < TOL
    assert e_det < TOL_SQUASHED, f"tanh(mu) differs by {e_det}"
    return e_mu, e_ls, e_det


def test_log_prob():
    """The tanh log-density, evaluated at the reference's own pre-squash sample."""
    blob, _, _ = _load()
    mu = jnp.asarray(blob["mu"]); log_std = jnp.asarray(blob["log_std"])
    x = jnp.asarray(blob["x"])
    log_normal = -0.5 * ((x - mu) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)
    log_det = 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))
    logp = jnp.sum(log_normal - log_det, axis=-1)
    err = float(jnp.abs(logp - blob["logp"]).max())
    assert err < TOL_SQUASHED, f"log-prob differs by {err}"
    return err


def test_critic():
    blob, _, cw = _load()
    critic = DoubleQCritic(hidden_size=256)
    q1, q2 = critic.apply(double_q_from_torch(cw), jnp.asarray(blob["s"]),
                          jnp.asarray(blob["a"]))
    e1 = float(jnp.abs(q1 - blob["q1"]).max())
    e2 = float(jnp.abs(q2 - blob["q2"]).max())
    assert max(e1, e2) < TOL
    return e1, e2


def test_critic_multi_action_broadcast():
    """(B, state) x (B, N, action) must fold to (B, N) Q-values."""
    blob, _, cw = _load()
    critic = DoubleQCritic(hidden_size=256)
    params = double_q_from_torch(cw)
    s = jnp.asarray(blob["s"])
    a = jnp.asarray(blob["a"])
    a_rep = jnp.repeat(a[:, None, :], 4, axis=1)
    q1m, _ = critic.apply(params, s, a_rep)
    q1, _ = critic.apply(params, s, a)
    assert q1m.shape == (s.shape[0], 4)
    err = float(jnp.abs(q1m - q1[:, None]).max())
    assert err < 1e-6, f"broadcast path disagrees with the flat path by {err}"
    return err


if __name__ == "__main__":
    print("policy   mu/log_std/mean_action max|diff| = %.3e / %.3e / %.3e   OK" % test_policy_head())
    print("log_prob                        max|diff| = %.3e   OK" % test_log_prob())
    print("critic   q1/q2                  max|diff| = %.3e / %.3e   OK" % test_critic())
    print("critic   multi-action broadcast max|diff| = %.3e   OK" % test_critic_multi_action_broadcast())
