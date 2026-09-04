"""Composite flow matching: source model, adaptation model, dynamics gap.

Port of guided_flow/flow_matching.py. The source model learns p_src(s'|s,a) as
a flow from noise; the adaptation model learns a flow carrying a source
prediction onto the target next state; the dynamics gap is the distance between
where the two land.

Structural changes for JAX and the behavioural fixes are in docs/DEVIATIONS.md.
"""

import dataclasses
import functools
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

from ..data import buffer as buf_lib
from ..nets.backbone import ResidualMLPGuidance
from . import cfm as cfm_lib
from . import normalize as norm_lib
from .ode import flow_forward

# n_batches is a static jit argument, so each distinct value costs a compile.
# Rounding up to a multiple of this cuts 28 shapes to 7 over a run.
BATCH_COUNT_BUCKET = 4

_MESH = None


def _batch_mesh():
    """A 1-D mesh over every local device, for sharding the gap estimate."""
    global _MESH
    if _MESH is None:
        _MESH = jax.sharding.Mesh(np.asarray(jax.devices()), ("batch",))
    return _MESH


def _shard_rows(x):
    """Split x across devices along its leading axis; see docs/BACKEND.md.

    Rows that do not divide evenly are left unsharded.
    """
    devices = jax.devices()
    if len(devices) < 2 or x.shape[0] % len(devices) != 0:
        return x
    spec = jax.sharding.NamedSharding(_batch_mesh(),
                                      jax.sharding.PartitionSpec("batch"))
    return jax.lax.with_sharding_constraint(x, spec)


def bucketed_batch_count(n_rows, batch_size):
    """Batches per epoch, rounded up to a multiple of ``BATCH_COUNT_BUCKET``."""
    exact = max(1, -(-n_rows // batch_size))  # ceil
    return -(-exact // BATCH_COUNT_BUCKET) * BATCH_COUNT_BUCKET


def _wrapped_perm(key, n_rows, n_batches, batch_size):
    """Shuffled row indices, wrapped to fill exactly ``n_batches`` batches."""
    perm = jax.random.permutation(key, n_rows)
    total = n_batches * batch_size
    return perm[jnp.arange(total) % n_rows].reshape(n_batches, batch_size)


@dataclasses.dataclass(frozen=True)
class FlowConfig:
    """Static hyper-parameters; hashable so it can be a jit static argument."""

    state_dim: int
    action_dim: int

    hidden_dim: int = 512
    source_num_layers: int = 6
    adaptation_num_layers: int = 6
    activation: str = "relu"

    source_flow_steps: int = 11
    adaptation_flow_steps: int = 11

    sigma: float = 0.0
    adaptation_sigma: float = 0.0

    lr: float = 3e-4
    batch_size: int = 1024
    holdout_ratio: float = 0.1

    source_epochs: int = 500
    adaptation_epochs: int = 200
    max_epochs_since_update: int = 20
    validation_start_epoch_source: int = 20
    validation_start_epoch_adaptation: int = 20
    improvement_threshold: float = 0.01

    use_ema: bool = False
    ema_decay: float = 0.995
    ema_update_steps: int = 20

    # "exact" is the reference's solver (pot.emd; scipy's Hungarian returns the
    # identical coupling and is faster). "sinkhorn" is an entropic relaxation
    # that runs on-device and is ~40x faster -- see docs/BACKEND.md for what
    # each costs.
    ot_solver: str = "exact"
    ot_reg: float = 0.05
    ot_iters: int = 200
    # The reference's argparse default.
    eta: float = 0.05

    # How the adaptation minibatch pairs source predictions with target next
    # states. The default is the paper's method: minibatch OT-CFM.
    #   "ot"           OT coupling, conditioning gathered with the same indices
    #                  (the paper's method; fixes the reference's wiring so the
    #                  velocity field sees the condition that produced its path)
    #   "ot_uncoupled" OT coupling, conditioning left in the original order --
    #                  literally what the reference wrote, had it ever run
    #   "identity"     pair each source prediction with its own target next
    #                  state. NOT the paper's method; a deviation that measures
    #                  better on a synthetic probe. See docs/COUPLING.md.
    coupling: str = "ot"

    # Spread the per-step gap estimate over every TPU chip. See _shard_rows.
    shard_gap: bool = True
    # Same for the ODE rollouts inside an adaptation refit, which dominate a
    # run once the target buffer grows.
    shard_refit: bool = True

    reproduce_original: bool = False

    @property
    def cond_dim(self):
        return self.state_dim + self.action_dim


@struct.dataclass
class FlowState:
    source: TrainState
    adaptation: TrainState
    source_ema: dict          # empty dict when use_ema is off
    adaptation_ema: dict
    sa_src: norm_lib.Normalizer
    ns_src: norm_lib.Normalizer
    sa_adapt: norm_lib.Normalizer
    ns_adapt: norm_lib.Normalizer
    source_fitted: bool = struct.field(pytree_node=False, default=False)
    adaptation_fitted: bool = struct.field(pytree_node=False, default=False)


def make_model(cfg, num_layers):
    return ResidualMLPGuidance(
        d_in=cfg.state_dim,
        cond_dim=cfg.cond_dim,
        mlp_width=cfg.hidden_dim,
        num_layers=num_layers,
        activation=cfg.activation,
    )


def _identity_normalizer(dim):
    return norm_lib.Normalizer(mean=jnp.zeros(dim), std=jnp.ones(dim))


def create_state(cfg, key):
    """Fresh models, optimisers and identity normalisers."""
    k_src, k_adapt = jax.random.split(key)
    src_model = make_model(cfg, cfg.source_num_layers)
    adapt_model = make_model(cfg, cfg.adaptation_num_layers)

    dummy_x = jnp.zeros((1, cfg.state_dim))
    dummy_t = jnp.zeros((1, 1))
    dummy_c = jnp.zeros((1, cfg.cond_dim))

    src_params = src_model.init(k_src, dummy_x, dummy_t, dummy_c)
    adapt_params = adapt_model.init(k_adapt, dummy_x, dummy_t, dummy_c)

    return FlowState(
        source=TrainState.create(apply_fn=src_model.apply, params=src_params,
                                 tx=optax.adam(cfg.lr)),
        adaptation=TrainState.create(apply_fn=adapt_model.apply, params=adapt_params,
                                     tx=optax.adam(cfg.lr)),
        # The EMA shadow starts as the initial weights. No copy: JAX arrays are
        # immutable and _ema_update rebuilds the tree.
        source_ema=src_params if cfg.use_ema else {},
        adaptation_ema=adapt_params if cfg.use_ema else {},
        sa_src=_identity_normalizer(cfg.cond_dim),
        ns_src=_identity_normalizer(cfg.state_dim),
        sa_adapt=_identity_normalizer(cfg.cond_dim),
        ns_adapt=_identity_normalizer(cfg.state_dim),
    )


def _ema_update(shadow, params, decay):
    return jax.tree.map(lambda s, p: decay * s + (1.0 - decay) * p, shadow, params)


def effective_params(train_state, ema, use_ema):
    """The weights to evaluate with: EMA shadow when enabled, else live."""
    return ema if (use_ema and ema) else train_state.params


def _sharded_flow_forward(apply_fn, params, x0, cond, n_points, enabled):
    """flow_forward, optionally split across devices along the batch axis."""
    if enabled:
        x0 = _shard_rows(x0)
        cond = _shard_rows(cond)
    return flow_forward(apply_fn, params, x0, cond, n_points)


# --------------------------------------------------------------------------
# source flow
# --------------------------------------------------------------------------

def _source_loss(cfg, apply_fn, params, key, s_norm, a_norm, ns_norm):
    cfg_cfm = cfm_lib.get_cfm("cfm", cfg.sigma)
    k_flow, k_noise = jax.random.split(key)
    ns_0 = jax.random.normal(k_noise, ns_norm.shape, ns_norm.dtype)
    t, ns_t, ut, _ = cfm_lib.sample_location_and_conditional_flow(
        k_flow, cfg_cfm, ns_0, ns_norm)
    cond = jnp.concatenate([s_norm, a_norm], axis=-1)
    vt = apply_fn(params, ns_t, t[:, None], cond)
    return jnp.mean((vt - ut) ** 2)


@functools.partial(jax.jit, static_argnames=("cfg", "n_batches"))
def _source_train_epoch(cfg, state, key, s, a, ns, n_batches):
    """One epoch of source training, all minibatches fused into a scan."""
    n = s.shape[0]
    k_perm, k_steps = jax.random.split(key)
    perm = _wrapped_perm(k_perm, n, n_batches, cfg.batch_size)

    def body(carry, idx):
        train_state, step_key, ema = carry
        step_key, k = jax.random.split(step_key)
        s_b, a_b, ns_b = s[idx], a[idx], ns[idx]

        s_n, a_n = norm_lib.normalize_state_action(state.sa_src, s_b, a_b)
        ns_n = state.ns_src(ns_b)

        loss, grads = jax.value_and_grad(
            lambda p: _source_loss(cfg, train_state.apply_fn, p, k, s_n, a_n, ns_n)
        )(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        if cfg.use_ema:
            ema = _ema_update(ema, train_state.params, cfg.ema_decay)
        return (train_state, step_key, ema), loss

    (new_state, _, new_ema), losses = jax.lax.scan(
        body, (state.source, k_steps, state.source_ema), perm)
    return state.replace(source=new_state, source_ema=new_ema), jnp.mean(losses)


@functools.partial(jax.jit, static_argnames=("cfg", "n_batches"))
def _source_val_epoch(cfg, state, key, s, a, ns, n_batches):
    """Validation loss: full Euler rollout, MSE against the raw next state."""
    params = effective_params(state.source, state.source_ema, cfg.use_ema)
    idx = (jnp.arange(n_batches * cfg.batch_size) % s.shape[0]
           ).reshape(n_batches, cfg.batch_size)

    def body(carry, batch_idx):
        k = jax.random.fold_in(key, batch_idx[0])
        s_b, a_b, ns_b = s[batch_idx], a[batch_idx], ns[batch_idx]
        s_n, a_n = norm_lib.normalize_state_action(state.sa_src, s_b, a_b)
        cond = jnp.concatenate([s_n, a_n], axis=-1)
        ns_0 = jax.random.normal(k, ns_b.shape, ns_b.dtype)
        pred_norm = _sharded_flow_forward(state.source.apply_fn, params, ns_0, cond,
                                          cfg.source_flow_steps, cfg.shard_refit)
        pred = state.ns_src.inverse(pred_norm)
        return carry, jnp.mean((pred - ns_b) ** 2)

    _, losses = jax.lax.scan(body, None, idx)
    return jnp.mean(losses)


# --------------------------------------------------------------------------
# adaptation flow
# --------------------------------------------------------------------------

def _source_prediction(cfg, state, key, s_b, a_b, shard=False):
    """Where the (frozen) source flow lands for this batch, in raw units."""
    src_params = effective_params(state.source, state.source_ema, cfg.use_ema)
    s_n, a_n = norm_lib.normalize_state_action(state.sa_src, s_b, a_b)
    cond_src = jnp.concatenate([s_n, a_n], axis=-1)
    ns_0 = jax.random.normal(key, (s_b.shape[0], cfg.state_dim), s_b.dtype)
    pred_norm = _sharded_flow_forward(state.source.apply_fn, src_params, ns_0,
                                      cond_src, cfg.source_flow_steps, shard)
    return state.ns_src.inverse(pred_norm), cond_src


def _adaptation_loss(cfg, state, params, key, s_b, a_b, ns_b):
    k_src, k_flow = jax.random.split(key)

    ns_src_pred, cond_src = _source_prediction(cfg, state, k_src, s_b, a_b)

    # x0 and x1 live in the *adaptation* normalisation; the conditioning stays
    # in the source normalisation, exactly as the reference had it.
    x0 = state.ns_adapt(ns_src_pred)
    x1 = state.ns_adapt(ns_b)
    s_adapt, a_adapt = norm_lib.normalize_state_action(state.sa_adapt, s_b, a_b)

    if cfg.coupling == "identity":
        cfg_cfm = cfm_lib.get_cfm("cfm", cfg.adaptation_sigma)
        t, xt, ut, coupling = cfm_lib.sample_location_and_conditional_flow(
            k_flow, cfg_cfm, x0, x1)
    else:
        cfg_cfm = cfm_lib.get_cfm(
            "ot_cfm", cfg.adaptation_sigma, ot_solver=cfg.ot_solver,
            ot_reg=cfg.ot_reg, ot_iters=cfg.ot_iters, eta=cfg.eta)
        t, xt, ut, coupling = cfm_lib.sample_location_and_conditional_flow(
            k_flow, cfg_cfm, x0, x1,
            s0=s_adapt, s1=s_adapt, a0=a_adapt, a1=a_adapt)

    if coupling is not None and cfg.coupling == "ot":
        # The path starts at x0[i], so it must be conditioned on the (s, a)
        # that produced x0[i]. The reference left cond in the original order,
        # which trains the field on mismatched (path, condition) pairs.
        cond_src = cond_src[coupling[0]]

    vt = state.adaptation.apply_fn(params, xt, t[:, None], cond_src)
    return jnp.mean((vt - ut) ** 2)


@functools.partial(jax.jit, static_argnames=("cfg", "n_batches"))
def _adaptation_train_epoch(cfg, state, key, s, a, ns, n_batches):
    n = s.shape[0]
    k_perm, k_steps = jax.random.split(key)
    perm = _wrapped_perm(k_perm, n, n_batches, cfg.batch_size)

    def body(carry, idx):
        train_state, step_key, ema = carry
        step_key, k = jax.random.split(step_key)
        s_b, a_b, ns_b = s[idx], a[idx], ns[idx]
        loss, grads = jax.value_and_grad(
            lambda p: _adaptation_loss(cfg, state, p, k, s_b, a_b, ns_b)
        )(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        if cfg.use_ema:
            ema = _ema_update(ema, train_state.params, cfg.ema_decay)
        return (train_state, step_key, ema), loss

    (new_state, _, new_ema), losses = jax.lax.scan(
        body, (state.adaptation, k_steps, state.adaptation_ema), perm)
    return state.replace(adaptation=new_state, adaptation_ema=new_ema), jnp.mean(losses)


@functools.partial(jax.jit, static_argnames=("cfg", "n_batches"))
def _adaptation_val_epoch(cfg, state, key, s, a, ns, n_batches):
    """Source flow, then adaptation flow, then MSE against the raw next state."""
    adapt_params = effective_params(state.adaptation, state.adaptation_ema, cfg.use_ema)
    idx = (jnp.arange(n_batches * cfg.batch_size) % s.shape[0]
           ).reshape(n_batches, cfg.batch_size)

    def body(carry, batch_idx):
        k = jax.random.fold_in(key, batch_idx[0])
        s_b, a_b, ns_b = s[batch_idx], a[batch_idx], ns[batch_idx]
        ns_src_pred, cond_src = _source_prediction(cfg, state, k, s_b, a_b,
                                                   shard=cfg.shard_refit)
        x0 = state.ns_adapt(ns_src_pred)
        pred_norm = _sharded_flow_forward(state.adaptation.apply_fn, adapt_params,
                                          x0, cond_src, cfg.adaptation_flow_steps,
                                          cfg.shard_refit)
        pred = state.ns_adapt.inverse(pred_norm)
        return carry, jnp.mean((pred - ns_b) ** 2)

    _, losses = jax.lax.scan(body, None, idx)
    return jnp.mean(losses)


# --------------------------------------------------------------------------
# dynamics gap
# --------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("cfg", "n_samples"))
def estimate_dynamics_gap(cfg, state, key, s, a, n_samples):
    """Region-level gap: mean over ``n_samples`` draws of ``||adapt - src||``.

    Each ``(s, a)`` is expanded ``n_samples`` times, the source flow is run from
    fresh noise, and the adaptation flow is run from the source landing point.
    Returns one scalar per input row.
    """
    src_params = effective_params(state.source, state.source_ema, cfg.use_ema)
    adapt_params = effective_params(state.adaptation, state.adaptation_ema, cfg.use_ema)

    batch = s.shape[0]
    s_n, a_n = norm_lib.normalize_state_action(state.sa_src, s, a)
    cond = jnp.repeat(jnp.concatenate([s_n, a_n], axis=-1), n_samples, axis=0)

    ns_0 = jax.random.normal(key, (batch * n_samples, cfg.state_dim), s.dtype)

    if cfg.shard_gap:
        cond = _shard_rows(cond)
        ns_0 = _shard_rows(ns_0)

    src_norm = flow_forward(state.source.apply_fn, src_params, ns_0, cond,
                            cfg.source_flow_steps)
    ns_source = state.ns_src.inverse(src_norm)

    adapt_norm = flow_forward(state.adaptation.apply_fn, adapt_params,
                              state.ns_adapt(ns_source), cond,
                              cfg.adaptation_flow_steps)
    ns_adaptation = state.ns_adapt.inverse(adapt_norm)

    per_sample = jnp.linalg.norm(
        ns_adaptation.reshape(batch, n_samples, -1)
        - ns_source.reshape(batch, n_samples, -1), axis=-1)
    return jnp.mean(per_sample, axis=-1)


@functools.partial(jax.jit, static_argnames=("cfg",))
def estimate_dynamics_gap_sample_level(cfg, state, s, a, ns):
    """Per-transition gap: how far the adaptation flow moves the observed ``ns``."""
    adapt_params = effective_params(state.adaptation, state.adaptation_ema, cfg.use_ema)
    s_n, a_n = norm_lib.normalize_state_action(state.sa_src, s, a)
    cond = jnp.concatenate([s_n, a_n], axis=-1)

    moved_norm = flow_forward(state.adaptation.apply_fn, adapt_params,
                              state.ns_adapt(ns), cond, cfg.adaptation_flow_steps)
    moved = state.ns_adapt.inverse(moved_norm)
    return jnp.linalg.norm(moved - ns, axis=-1)


# --------------------------------------------------------------------------
# host-side training drivers
# --------------------------------------------------------------------------

def _split(n, holdout_ratio):
    """Reference split: the first ``holdout_ratio`` rows validate, rest train."""
    n_holdout = int(n * holdout_ratio)
    return n_holdout, n - n_holdout


def _run_epochs(label, state, key, data, cfg, n_epochs, val_start, train_fn, val_fn,
                verbose=True):
    """Shared epoch loop: fused epochs, 1%-improvement early stopping."""
    s, a, ns = data
    n = s.shape[0]
    n_val, n_train = _split(n, cfg.holdout_ratio)
    if n_train < cfg.batch_size:
        raise ValueError(
            f"{label}: {n_train} training rows is fewer than one batch "
            f"({cfg.batch_size}); nothing to train on")

    s_val, a_val, ns_val = s[:n_val], a[:n_val], ns[:n_val]
    s_tr, a_tr, ns_tr = s[n_val:], a[n_val:], ns[n_val:]

    n_train_batches = bucketed_batch_count(n_train, cfg.batch_size)
    n_val_batches = bucketed_batch_count(max(n_val, 1), cfg.batch_size)

    best_loss = math.inf
    best_state = None
    best_ema = None
    epochs_since_update = 0

    for epoch in range(n_epochs):
        t0 = time.time()
        key, k_train, k_val = jax.random.split(key, 3)
        state, train_loss = train_fn(cfg, state, k_train, s_tr, a_tr, ns_tr,
                                     n_train_batches)

        if epoch < val_start:
            if verbose:
                print(f"  [{label}] epoch {epoch}: train {float(train_loss):.4f} "
                      f"({time.time() - t0:.1f}s, validation starts at {val_start})")
            continue

        val_loss = float(val_fn(cfg, state, k_val, s_val, a_val, ns_val, n_val_batches))
        improvement = (best_loss - val_loss) / best_loss if math.isfinite(best_loss) else math.inf

        if val_loss < best_loss and improvement > cfg.improvement_threshold:
            best_loss = val_loss
            # JAX arrays are immutable and every epoch rebuilds the parameter
            # tree, so holding the reference is enough -- no copy needed.
            best_state = (state.source.params if label == "source"
                          else state.adaptation.params)
            best_ema = (state.source_ema if label == "source" else state.adaptation_ema)
            epochs_since_update = 0
            marker = " *"
        else:
            epochs_since_update += 1
            marker = ""

        if verbose:
            print(f"  [{label}] epoch {epoch}: train {float(train_loss):.4f} "
                  f"val {val_loss:.4f} ({time.time() - t0:.1f}s){marker}")

        if epochs_since_update > cfg.max_epochs_since_update:
            if verbose:
                print(f"  [{label}] early stop at epoch {epoch} "
                      f"(best val {best_loss:.4f})")
            break

    if best_state is not None:
        if label == "source":
            state = state.replace(source=state.source.replace(params=best_state),
                                  source_ema=best_ema)
        else:
            state = state.replace(
                adaptation=state.adaptation.replace(params=best_state),
                adaptation_ema=best_ema)
    elif verbose:
        print(f"  [{label}] no epoch improved by >{cfg.improvement_threshold:.0%}; "
              "keeping the final weights")

    return state, best_loss


def train_source_flow(cfg, state, key, batch, verbose=True):
    """Fit the source dynamics flow on the offline source dataset."""
    k_shuffle, k_epochs = jax.random.split(key)
    # Shuffle before splitting: D4RL rows are in trajectory order, so the
    # leading `holdout_ratio` would otherwise hold out the earliest episodes.
    batch = buf_lib.shuffle_batch(batch, k_shuffle, bootstrap=cfg.reproduce_original)
    n_val, _ = _split(batch.state.shape[0], cfg.holdout_ratio)

    # Normalisers come from the training split only, as in the reference.
    s_tr, a_tr, ns_tr = batch.state[n_val:], batch.action[n_val:], batch.next_state[n_val:]
    state = state.replace(
        sa_src=norm_lib.fit_state_action(s_tr, a_tr),
        ns_src=norm_lib.fit(ns_tr),
        source_fitted=True,
    )
    return _run_epochs("source", state, k_epochs,
                       (batch.state, batch.action, batch.next_state), cfg,
                       cfg.source_epochs, cfg.validation_start_epoch_source,
                       _source_train_epoch, _source_val_epoch, verbose)


def train_adaptation_flow(cfg, state, key, batch, verbose=True):
    """(Re)fit the adaptation flow on the online target data.

    The reference re-initialises the adaptation model on every call; that is
    kept, since the target buffer grows and the previous fit was made against a
    different data distribution.
    """
    k_init, k_shuffle, k_epochs = jax.random.split(key, 3)
    # Same reason as the source split: the target buffer is in time order.
    batch = buf_lib.shuffle_batch(batch, k_shuffle, bootstrap=cfg.reproduce_original)
    n_val, _ = _split(batch.state.shape[0], cfg.holdout_ratio)
    s_tr, a_tr, ns_tr = batch.state[n_val:], batch.action[n_val:], batch.next_state[n_val:]

    fresh = create_state(cfg, k_init)
    state = state.replace(
        adaptation=fresh.adaptation,
        adaptation_ema=fresh.adaptation_ema,
        sa_adapt=norm_lib.fit_state_action(s_tr, a_tr),
        ns_adapt=norm_lib.fit(ns_tr),
        adaptation_fitted=True,
    )
    return _run_epochs("adaptation", state, k_epochs,
                       (batch.state, batch.action, batch.next_state), cfg,
                       cfg.adaptation_epochs, cfg.validation_start_epoch_adaptation,
                       _adaptation_train_epoch, _adaptation_val_epoch, verbose)
