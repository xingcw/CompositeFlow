"""Does the split-half reliability diagnostic itself work?

Before trusting it on the real benchmark, check it on the synthetic probe where
the answer is known: source dynamics s' = s + 0.5a, target s' = s + 1.5a. A
trained adaptation model should give a high split-half correlation; a randomly
initialised one should give ~0.
"""

import comp_flow_jax  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np

from comp_flow_jax.data.buffer import Batch
from comp_flow_jax.flow import flow_matching as fm

SD, AD, N = 4, 2, 20000


def make(rng, shift):
    s = rng.randn(N, SD).astype(np.float32)
    a = rng.uniform(-1, 1, (N, AD)).astype(np.float32)
    ns = s.copy()
    ns[:, :AD] += shift * a
    ns += 0.02 * rng.randn(N, SD).astype(np.float32)
    return Batch(jnp.asarray(s), jnp.asarray(a), jnp.asarray(ns),
                 jnp.zeros((N, 1)), jnp.ones((N, 1)))


def main():
    rng = np.random.RandomState(0)
    src, tar = make(rng, 0.5), make(rng, 1.5)
    key = jax.random.PRNGKey(0)

    base = dict(state_dim=SD, action_dim=AD, hidden_dim=256, source_num_layers=3,
                adaptation_num_layers=3, batch_size=512, source_epochs=40,
                adaptation_epochs=40, validation_start_epoch_source=3,
                validation_start_epoch_adaptation=3, max_epochs_since_update=8)

    probe_s = jnp.asarray(rng.randn(256, SD).astype(np.float32))
    probe_a = jnp.asarray(rng.uniform(-1, 1, (256, AD)).astype(np.float32))
    diag_key = jax.random.PRNGKey(12345)

    cfg0 = fm.FlowConfig(**base)
    st = fm.create_state(cfg0, key)
    st, _ = fm.train_source_flow(cfg0, st, jax.random.fold_in(key, 1), src, verbose=False)

    # Untrained adaptation model: the gap should be noise.
    r, m, sd = fm.gap_split_half_reliability(cfg0, st, diag_key, probe_s, probe_a, 30)
    print(f"adaptation model = random init : r = {float(r):+.4f}  "
          f"gap mean {float(m):.4f} std {float(sd):.4f}")

    for coupling, eta in [("ot_uncoupled", 0.05), ("ot", 0.05), ("ot", 1.0), ("identity", 0.0)]:
        cfg = fm.FlowConfig(**base, coupling=coupling, eta=eta, ot_solver="exact")
        st_a, val = fm.train_adaptation_flow(cfg, st, jax.random.fold_in(key, 2), tar,
                                             verbose=False)
        r, m, sd = fm.gap_split_half_reliability(cfg, st_a, diag_key, probe_s, probe_a, 30)
        print(f"coupling={coupling:14s} eta={eta:4.2f} : r = {float(r):+.4f}  "
              f"gap mean {float(m):.4f} std {float(sd):.4f}  (adapt val MSE {val:.4f})",
              flush=True)


if __name__ == "__main__":
    main()
