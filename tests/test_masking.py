"""The masked reductions must equal the reference's boolean indexing.

`algos/vflow.py` replaces the reference's ``src_state = src_state[mask]`` with a
fixed-shape batch plus a 0/1 mask, because a data-dependent shape cannot be
jitted. That substitution is only legitimate if every reduction gives the same
number. This checks the three that matter.
"""

import comp_flow_jax  # noqa: F401
import jax.numpy as jnp
import numpy as np

from comp_flow_jax.algos.vflow import _masked_mean_rows

RNG = np.random.RandomState(0)


def _setup(n_src=200, n_tar=128, keep_frac=0.8):
    """A source batch with a filter mask, plus an all-kept target batch."""
    gap = RNG.rand(n_src)
    threshold = np.quantile(gap, keep_frac)
    src_keep = gap < threshold
    mask = np.concatenate([src_keep.astype(np.float32), np.ones(n_tar, np.float32)])
    return src_keep, mask, n_src, n_tar


def test_critic_loss_reduction():
    """(q - target)^2 averaged over the concatenated kept batch."""
    src_keep, mask, n_src, n_tar = _setup()
    per_sample = RNG.randn(n_src + n_tar) ** 2

    # Reference: concatenate kept source with all target, then .mean()
    reference = np.concatenate([per_sample[:n_src][src_keep], per_sample[n_src:]]).mean()
    ours = float(_masked_mean_rows(jnp.asarray(per_sample), jnp.asarray(mask)))
    assert np.isclose(ours, reference, rtol=1e-6), f"{ours} vs {reference}"
    return ours, reference


def test_weighted_critic_loss_reduction():
    """(w * loss).mean() over the same batch -- the use_weight path."""
    src_keep, mask, n_src, n_tar = _setup()
    per_sample = RNG.randn(n_src + n_tar) ** 2
    weights = np.concatenate([RNG.rand(n_src), np.ones(n_tar)])

    reference = np.concatenate([(weights * per_sample)[:n_src][src_keep],
                                (weights * per_sample)[n_src:]]).mean()
    ours = float(_masked_mean_rows(jnp.asarray(weights * per_sample), jnp.asarray(mask)))
    assert np.isclose(ours, reference, rtol=1e-6), f"{ours} vs {reference}"


def test_bc_loss_reduction():
    """mse_loss over the kept source rows only, averaged over elements too."""
    src_keep, mask, n_src, _ = _setup()
    action_dim = 6
    diff = RNG.randn(n_src, action_dim)

    reference = (diff[src_keep] ** 2).mean()   # F.mse_loss default reduction
    ours = float(_masked_mean_rows(jnp.asarray(diff ** 2),
                                   jnp.asarray(mask[:n_src])))
    assert np.isclose(ours, reference, rtol=1e-6), f"{ours} vs {reference}"


def test_all_filtered_out_is_finite():
    """If the mask is empty the reduction must not blow up."""
    x = jnp.asarray(RNG.randn(64) ** 2)
    out = float(_masked_mean_rows(x, jnp.zeros(64)))
    assert np.isfinite(out) and out == 0.0, out


if __name__ == "__main__":
    a, b = test_critic_loss_reduction()
    print(f"critic loss reduction    OK  ({a:.10f} == {b:.10f})")
    test_weighted_critic_loss_reduction()
    print("weighted critic loss     OK")
    test_bc_loss_reduction()
    print("BC loss reduction        OK")
    test_all_filtered_out_is_finite()
    print("empty mask stays finite  OK")
