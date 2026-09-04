"""Device-resident replay buffers.

Both buffers live in TPU memory so that sampling happens inside the jitted
update. Departures from algo/utils.py::ReplayBuffer are in docs/DEVIATIONS.md.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class Batch:
    state: jnp.ndarray
    action: jnp.ndarray
    next_state: jnp.ndarray
    reward: jnp.ndarray     # (B, 1)
    not_done: jnp.ndarray   # (B, 1)

    def __len__(self):
        return self.state.shape[0]


@struct.dataclass
class ReplayBuffer:
    state: jnp.ndarray
    action: jnp.ndarray
    next_state: jnp.ndarray
    reward: jnp.ndarray
    not_done: jnp.ndarray
    ptr: jnp.ndarray      # int32 scalar, next write index
    size: jnp.ndarray     # int32 scalar, number of valid rows

    @property
    def max_size(self):
        return self.state.shape[0]

    @property
    def capacity(self):
        return self.state.shape[0]


def empty_buffer(state_dim, action_dim, max_size=int(1e6), dtype=jnp.float32):
    """Allocate a zero-filled ring buffer on the default device."""
    return ReplayBuffer(
        state=jnp.zeros((max_size, state_dim), dtype),
        action=jnp.zeros((max_size, action_dim), dtype),
        next_state=jnp.zeros((max_size, state_dim), dtype),
        reward=jnp.zeros((max_size, 1), dtype),
        not_done=jnp.zeros((max_size, 1), dtype),
        ptr=jnp.int32(0),
        size=jnp.int32(0),
    )


def from_d4rl(dataset, dtype=jnp.float32):
    """Freeze a ``qlearning_dataset`` dict into a full buffer.

    Sized exactly to the dataset, matching ``ReplayBuffer.convert_D4RL``, which
    replaced the preallocated arrays outright rather than filling the ring.
    """
    n = dataset["observations"].shape[0]
    return ReplayBuffer(
        state=jnp.asarray(dataset["observations"], dtype),
        action=jnp.asarray(dataset["actions"], dtype),
        next_state=jnp.asarray(dataset["next_observations"], dtype),
        reward=jnp.asarray(np.asarray(dataset["rewards"]).reshape(-1, 1), dtype),
        not_done=jnp.asarray(1.0 - np.asarray(dataset["terminals"]).reshape(-1, 1), dtype),
        ptr=jnp.int32(n % n if n else 0),
        size=jnp.int32(n),
    )


# donate_argnums=0: without donation each add copies the whole buffer (108 MB
# for 1M rows). The caller must not reuse the buffer it passed in.
@partial(jax.jit, donate_argnums=0)
def add(buf, state, action, next_state, reward, done):
    """Write one transition at ``ptr`` and advance the ring.

    Donates ``buf``: the argument is consumed and must not be used again.
    """
    i = buf.ptr

    def put(arr, row):
        return jax.lax.dynamic_update_slice(
            arr, jnp.asarray(row, arr.dtype).reshape(1, -1), (i, 0))

    return buf.replace(
        state=put(buf.state, state),
        action=put(buf.action, action),
        next_state=put(buf.next_state, next_state),
        reward=put(buf.reward, jnp.reshape(reward, (1,))),
        not_done=put(buf.not_done, jnp.reshape(1.0 - done, (1,))),
        ptr=(i + 1) % buf.max_size,
        size=jnp.minimum(buf.size + 1, buf.max_size),
    )


def sample(buf, key, batch_size):
    """Uniform sample with replacement from the valid prefix.

    Traceable: ``buf.size`` may be a tracer, so this works inside jit.
    """
    idx = jax.random.randint(key, (batch_size,), 0, buf.size)
    return take(buf, idx)


def take(buf, idx):
    return Batch(
        state=buf.state[idx],
        action=buf.action[idx],
        next_state=buf.next_state[idx],
        reward=buf.reward[idx],
        not_done=buf.not_done[idx],
    )


def as_batch(buf):
    """The valid prefix as a Batch, without gathering.

    ``size`` must be concrete. Use this instead of ``take(buf, arange(size))``
    when the whole buffer is wanted: for a full 1M-row source buffer the gather
    would copy another 108 MB for no reason.
    """
    n = int(buf.size)
    return Batch(state=buf.state[:n], action=buf.action[:n],
                 next_state=buf.next_state[:n], reward=buf.reward[:n],
                 not_done=buf.not_done[:n])


def shuffle_batch(batch, key, bootstrap=False):
    """Reorder a Batch so a prefix split is not a time-ordered split.

    bootstrap=True reproduces the reference's resample-with-replacement, and
    the train/validation overlap that came with it.
    """
    n = batch.state.shape[0]
    idx = (jax.random.randint(key, (n,), 0, n) if bootstrap
           else jax.random.permutation(key, n))
    return Batch(state=batch.state[idx], action=batch.action[idx],
                 next_state=batch.next_state[idx], reward=batch.reward[idx],
                 not_done=batch.not_done[idx])


def sample_all(buf, key, bootstrap=False):
    """Every stored transition, in random order.

    ``size`` must be concrete here (this runs outside jit, when preparing
    flow-matching epochs).
    """
    n = int(buf.size)
    if bootstrap:
        idx = jax.random.randint(key, (n,), 0, n)
    else:
        idx = jax.random.permutation(key, n)
    return take(buf, idx)


def downsample(buf, ratio, key):
    """Keep a random ``ratio`` fraction of the buffer, dropping the rest."""
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"downsample ratio must be in (0, 1], got {ratio}")
    n = int(buf.size)
    keep = max(1, int(round(n * ratio)))
    idx = jax.random.permutation(key, n)[:keep]
    b = take(buf, idx)
    return ReplayBuffer(
        state=b.state, action=b.action, next_state=b.next_state,
        reward=b.reward, not_done=b.not_done,
        ptr=jnp.int32(0), size=jnp.int32(keep),
    )
