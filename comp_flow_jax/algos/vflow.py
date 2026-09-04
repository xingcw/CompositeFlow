"""VFlow: SAC on a source+target mixture, gated by a learned dynamics gap.

Port of algo/offline_online/vflow.py. Each step draws a source and a target
minibatch, estimates the dynamics gap at the source samples, drops the worst,
optionally reweights and shapes the reward, then runs one SAC update plus a BC
term on the surviving source actions.

Filtering keeps a fixed batch shape and carries a 0/1 mask rather than boolean
indexing; the reductions are algebraically identical, checked in
tests/test_masking.py. See docs/DEVIATIONS.md.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

from ..data import buffer as buf_lib
from ..flow import flow_matching as fm
from ..nets.actor_critic import (
    DoubleQCritic, TanhGaussianPolicy, deterministic_action, tanh_gaussian_sample)

EPS = 1e-6


@dataclasses.dataclass(frozen=True)
class VFlowConfig:
    state_dim: int
    action_dim: int
    max_action: float = 1.0

    gamma: float = 0.99
    tau: float = 0.005
    hidden_sizes: int = 256
    batch_size: int = 128

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha: float = 0.2
    temperature_opt: bool = False
    backup_entropy: bool = True
    target_entropy: float = None

    # BC weight on surviving source actions ('weight' in the reference config).
    policy_bc_weight: float = 2.5

    # Gap machinery.
    start_gate_src_sample: int = 100_000
    dynamics_train_freq: int = 5_000
    filter_percent: float = 0.7
    use_weight: bool = False
    beta: float = 1.0
    use_sample_level: bool = False
    dynamics_gap_reward_scale: float = 0.0
    n_samples: int = 100
    upsample_src: bool = False
    downsample_src: float = 1.0

    reproduce_original: bool = False

    def __post_init__(self):
        if self.target_entropy is None:
            object.__setattr__(self, "target_entropy", -float(self.action_dim))

    @property
    def src_sample_size(self):
        """Source minibatch size after the gate opens.

        The reference sizes this so that ``batch_size`` samples survive
        filtering. ``keep_ratio`` is ``filter_percent`` -- the fraction the mask
        keeps -- not ``1 - filter_percent``; see the module docstring.
        """
        if not self.upsample_src:
            return self.batch_size
        keep = (max(1.0 - self.filter_percent, 0.01) if self.reproduce_original
                else max(self.filter_percent, 0.01))
        size = int(self.batch_size / keep / max(self.downsample_src, 1e-6))
        return max(size, self.batch_size)


@struct.dataclass
class VFlowState:
    actor: TrainState
    critic: TrainState
    target_critic_params: dict
    log_alpha: jnp.ndarray
    alpha_opt_state: optax.OptState
    flow: fm.FlowState
    total_it: jnp.ndarray

    @property
    def alpha(self):
        return jnp.exp(self.log_alpha)


def create_state(cfg, flow_cfg, key):
    k_actor, k_critic, k_flow = jax.random.split(key, 3)

    actor = TanhGaussianPolicy(action_dim=cfg.action_dim, max_action=cfg.max_action,
                               hidden_size=cfg.hidden_sizes)
    critic = DoubleQCritic(hidden_size=cfg.hidden_sizes)

    dummy_s = jnp.zeros((1, cfg.state_dim))
    dummy_a = jnp.zeros((1, cfg.action_dim))
    actor_params = actor.init(k_actor, dummy_s)
    critic_params = critic.init(k_critic, dummy_s, dummy_a)

    log_alpha = jnp.asarray(np.log(cfg.alpha), jnp.float32)
    alpha_tx = optax.adam(cfg.actor_lr)
    alpha_opt_state = alpha_tx.init(log_alpha) if cfg.temperature_opt else None

    return VFlowState(
        actor=TrainState.create(apply_fn=actor.apply, params=actor_params,
                                tx=optax.adam(cfg.actor_lr)),
        critic=TrainState.create(apply_fn=critic.apply, params=critic_params,
                                 tx=optax.adam(cfg.critic_lr)),
        target_critic_params=critic_params,  # immutable; polyak rebuilds the tree
        log_alpha=log_alpha,
        alpha_opt_state=alpha_opt_state,
        flow=fm.create_state(flow_cfg, k_flow),
        total_it=jnp.int32(0),
    )


def alpha_tx(cfg):
    return optax.adam(cfg.actor_lr)


# --------------------------------------------------------------------------
# masked reductions
# --------------------------------------------------------------------------

def _masked_mean_rows(x, mask):
    """Mean over rows, where ``mask`` selects rows and ``x`` is (B,) or (B, D)."""
    if x.ndim == 1:
        return jnp.sum(mask * x) / jnp.maximum(jnp.sum(mask), EPS)
    return jnp.sum(mask[:, None] * x) / jnp.maximum(jnp.sum(mask) * x.shape[1], EPS)


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------

def _critic_loss(cfg, state, critic_params, key, batch, mask, weights, gap):
    """SAC critic loss with the optional gap reward shaping and weighting."""
    reward = batch.reward
    if cfg.dynamics_gap_reward_scale != 0.0 and gap is not None:
        reward = reward + cfg.dynamics_gap_reward_scale * gap[:, None]

    mu, log_std = state.actor.apply_fn(state.actor.params, batch.next_state)
    next_action, next_logp = tanh_gaussian_sample(key, mu, log_std, cfg.max_action)
    tq1, tq2 = state.critic.apply_fn(state.target_critic_params,
                                     batch.next_state, next_action)
    target_q = jnp.minimum(tq1, tq2)

    bootstrap = target_q - state.alpha * next_logp if cfg.backup_entropy else target_q
    value_target = jax.lax.stop_gradient(
        reward[:, 0] + batch.not_done[:, 0] * cfg.gamma * bootstrap)

    q1, q2 = state.critic.apply_fn(critic_params, batch.state, batch.action)
    per_sample = (q1 - value_target) ** 2 + (q2 - value_target) ** 2
    if cfg.use_weight:
        per_sample = weights * per_sample
    loss = _masked_mean_rows(per_sample, mask)

    aux = {"q1": _masked_mean_rows(q1, mask), "q2": _masked_mean_rows(q2, mask),
           "value_target": _masked_mean_rows(value_target, mask),
           "batch_reward": _masked_mean_rows(reward[:, 0], mask)}
    return loss, aux


def _actor_loss(cfg, state, actor_params, key, batch, mask, src_slice):
    """SAC actor loss scaled by ``p_w``, plus BC on the surviving source actions."""
    mu, log_std = state.actor.apply_fn(actor_params, batch.state)
    action, logp = tanh_gaussian_sample(key, mu, log_std, cfg.max_action)
    q1, q2 = state.critic.apply_fn(state.critic.params, batch.state, action)
    qval = jnp.minimum(q1, q2)

    sac_term = _masked_mean_rows(state.alpha * logp - qval, mask)

    # BC on the surviving source rows. The reference calls ``self.policy(...)``
    # and takes its first return value, i.e. the reparameterised stochastic
    # sample, not the mean action -- so this uses ``action``, not ``mu``.
    n_src = src_slice
    src_mask = mask[:n_src]
    bc_loss = _masked_mean_rows((action[:n_src] - batch.action[:n_src]) ** 2, src_mask)

    qval_abs_mean = jax.lax.stop_gradient(_masked_mean_rows(jnp.abs(qval), mask))
    p_w = cfg.policy_bc_weight / (qval_abs_mean + EPS)

    loss = p_w * sac_term + bc_loss
    aux = {"sac_term": sac_term, "bc_loss": bc_loss, "p_w": p_w,
           "qval_abs_mean": qval_abs_mean, "logp": _masked_mean_rows(logp, mask)}
    return loss, (aux, jax.lax.stop_gradient(logp))


def _alpha_loss(cfg, log_alpha, logp, mask):
    return -_masked_mean_rows(jnp.exp(log_alpha) * (logp + cfg.target_entropy), mask)


# --------------------------------------------------------------------------
# one gradient step
# --------------------------------------------------------------------------

def _gap_and_mask(cfg, flow_cfg, state, key, src, tar):
    """Filter mask, per-sample weights and (optionally) the shaping gap."""
    n_src = src.state.shape[0]
    n_tar = tar.state.shape[0]

    if cfg.use_sample_level:
        gap_src = fm.estimate_dynamics_gap_sample_level(
            flow_cfg, state.flow, src.state, src.action, src.next_state)
        gap_tar = fm.estimate_dynamics_gap_sample_level(
            flow_cfg, state.flow, tar.state, tar.action, tar.next_state)
    elif cfg.dynamics_gap_reward_scale != 0.0:
        combined_s = jnp.concatenate([src.state, tar.state], 0)
        combined_a = jnp.concatenate([src.action, tar.action], 0)
        gap_all = fm.estimate_dynamics_gap(flow_cfg, state.flow, key,
                                           combined_s, combined_a, cfg.n_samples)
        gap_src, gap_tar = gap_all[:n_src], gap_all[n_src:]
    else:
        gap_src = fm.estimate_dynamics_gap(flow_cfg, state.flow, key,
                                           src.state, src.action, cfg.n_samples)
        gap_tar = jnp.zeros(n_tar)

    threshold = jnp.quantile(gap_src.astype(jnp.float32), cfg.filter_percent)
    src_mask = (gap_src < threshold).astype(jnp.float32)

    if cfg.use_weight:
        span = jnp.max(gap_src) - jnp.min(gap_src)
        normalized = (gap_src - jnp.max(gap_src)) / (span + 1e-8)
        src_weights = jnp.exp(cfg.beta * normalized)
    else:
        src_weights = jnp.ones_like(gap_src)

    mask = jnp.concatenate([src_mask, jnp.ones(n_tar)], 0)
    weights = jnp.concatenate([src_weights, jnp.ones(n_tar)], 0)
    gap = jnp.concatenate([gap_src, gap_tar], 0)
    return mask, weights, gap


def _concat(src, tar):
    return buf_lib.Batch(
        state=jnp.concatenate([src.state, tar.state], 0),
        action=jnp.concatenate([src.action, tar.action], 0),
        next_state=jnp.concatenate([src.next_state, tar.next_state], 0),
        reward=jnp.concatenate([src.reward, tar.reward], 0),
        not_done=jnp.concatenate([src.not_done, tar.not_done], 0),
    )


def _apply_updates(cfg, state, key, batch, mask, weights, gap, n_src):
    k_critic, k_actor = jax.random.split(key)

    (critic_loss, critic_aux), critic_grads = jax.value_and_grad(
        _critic_loss, argnums=2, has_aux=True)(
            cfg, state, state.critic.params, k_critic, batch, mask, weights, gap)
    state = state.replace(critic=state.critic.apply_gradients(grads=critic_grads))

    (actor_loss, (actor_aux, logp)), actor_grads = jax.value_and_grad(
        _actor_loss, argnums=2, has_aux=True)(
            cfg, state, state.actor.params, k_actor, batch, mask, n_src)
    state = state.replace(actor=state.actor.apply_gradients(grads=actor_grads))

    if cfg.temperature_opt:
        alpha_loss, alpha_grad = jax.value_and_grad(_alpha_loss, argnums=1)(
            cfg, state.log_alpha, logp, mask)
        updates, new_opt = alpha_tx(cfg).update(alpha_grad, state.alpha_opt_state,
                                                state.log_alpha)
        state = state.replace(log_alpha=optax.apply_updates(state.log_alpha, updates),
                              alpha_opt_state=new_opt)
        actor_aux = dict(actor_aux, alpha_loss=alpha_loss)

    state = state.replace(
        target_critic_params=jax.tree.map(
            lambda t, q: cfg.tau * q + (1.0 - cfg.tau) * t,
            state.target_critic_params, state.critic.params),
        total_it=state.total_it + 1,
    )

    metrics = {"critic_loss": critic_loss, "actor_loss": actor_loss,
               "alpha": state.alpha, "kept_fraction": jnp.mean(mask),
               **critic_aux, **actor_aux}
    if gap is not None:
        metrics["dynamics_gap"] = jnp.mean(gap)
    return state, metrics


@functools.partial(jax.jit, static_argnames=("cfg", "flow_cfg"))
def train_step_pregate(cfg, flow_cfg, state, key, src_buffer, tar_buffer):
    """Plain SAC + BC on the unfiltered source/target mixture."""
    k_src, k_tar, k_upd = jax.random.split(key, 3)
    src = buf_lib.sample(src_buffer, k_src, cfg.batch_size)
    tar = buf_lib.sample(tar_buffer, k_tar, cfg.batch_size)
    batch = _concat(src, tar)
    mask = jnp.ones(2 * cfg.batch_size)
    weights = jnp.ones(2 * cfg.batch_size)
    return _apply_updates(cfg, state, k_upd, batch, mask, weights, None, cfg.batch_size)


@functools.partial(jax.jit, static_argnames=("cfg", "flow_cfg"))
def train_step_postgate(cfg, flow_cfg, state, key, src_buffer, tar_buffer):
    """As above, with gap-based filtering, weighting and reward shaping."""
    k_src, k_tar, k_gap, k_upd = jax.random.split(key, 4)
    src = buf_lib.sample(src_buffer, k_src, cfg.src_sample_size)
    tar = buf_lib.sample(tar_buffer, k_tar, cfg.batch_size)
    mask, weights, gap = _gap_and_mask(cfg, flow_cfg, state, k_gap, src, tar)
    batch = _concat(src, tar)
    return _apply_updates(cfg, state, k_upd, batch, mask, weights, gap,
                          cfg.src_sample_size)


# --------------------------------------------------------------------------
# acting
# --------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("cfg", "deterministic"))
def select_action(cfg, state, key, obs, deterministic=True):
    """Action for a single observation or a batch of them."""
    single = obs.ndim == 1
    x = obs[None, :] if single else obs
    mu, log_std = state.actor.apply_fn(state.actor.params, x)
    if deterministic:
        action = deterministic_action(mu, cfg.max_action)
    else:
        action, _ = tanh_gaussian_sample(key, mu, log_std, cfg.max_action)
    return action[0] if single else action
