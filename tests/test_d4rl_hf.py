"""The vectorised ``qlearning_dataset`` must match the transcribed d4rl loop.

Downloads hopper-medium-v2 from the HF mirror on first run (~120 MB, cached).
"""

import time

import numpy as np

from comp_flow_jax.data.d4rl_hf import (
    _qlearning_dataset_loop, get_dataset, hf_filename, qlearning_dataset)


def test_name_mapping():
    assert hf_filename("hopper-medium-v2") == "hopper_medium-v2.hdf5"
    assert hf_filename("hopper-medium-replay-v2") == "hopper_medium_replay-v2.hdf5"
    assert hf_filename("halfcheetah-random-v2") == "halfcheetah_random-v2.hdf5"


def test_fast_path_matches_loop():
    raw = get_dataset("hopper-medium-v2")

    t0 = time.time()
    fast = qlearning_dataset(raw)
    t_fast = time.time() - t0

    t0 = time.time()
    slow = _qlearning_dataset_loop(raw, False, 1000)
    t_slow = time.time() - t0

    assert set(fast) == set(slow)
    for k in fast:
        assert fast[k].shape == slow[k].shape, f"{k}: {fast[k].shape} vs {slow[k].shape}"
        assert np.array_equal(fast[k], slow[k]), f"{k} differs"
    return t_fast, t_slow, fast["observations"].shape[0]


def test_shapes_and_dtypes():
    ds = qlearning_dataset(get_dataset("hopper-medium-v2"))
    n = ds["observations"].shape[0]
    assert ds["observations"].shape == (n, 11)
    assert ds["actions"].shape == (n, 3)
    assert ds["next_observations"].shape == (n, 11)
    assert ds["rewards"].shape == (n,)
    assert ds["terminals"].shape == (n,)
    assert ds["observations"].dtype == np.float32
    assert ds["terminals"].dtype == np.bool_
    return n


if __name__ == "__main__":
    test_name_mapping()
    print("name mapping   OK")
    n = test_shapes_and_dtypes()
    print(f"shapes/dtypes  OK  ({n:,} transitions)")
    tf, ts, n = test_fast_path_matches_loop()
    print(f"fast == loop   OK  ({tf * 1000:.0f} ms vs {ts * 1000:.0f} ms, "
          f"{ts / tf:.0f}x, bit-identical over {n:,} rows)")
