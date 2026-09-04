"""Training entry point.

Mirrors the reference train.py CLI, so an existing launch command works
unchanged. Loop-level differences are in docs/DEVIATIONS.md.
"""

import argparse
import functools
import json
import os
import pathlib
import signal
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import build, config as config_lib
from .algos import vflow
from .data import buffer as buf_lib
from .data.d4rl_hf import load_source_dataset
from .envs.infos import get_normalized_score
from .envs import ground_truth as gt
from .envs.registry import call_mujoco_env
from .envs.vec import evaluate
from .flow import cache as flow_cache
from .flow import flow_matching as fm

_EXIT_REQUESTED = False


def _on_sigterm(signum, frame):
    global _EXIT_REQUESTED
    print(f"\n[train] received {signal.strsignal(signum)}; finishing the current "
          "block and exiting\n", flush=True)
    _EXIT_REQUESTED = True


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="./logs", help="base directory for logs")
    p.add_argument("--policy", default="vflow", help="only 'vflow' is ported")
    p.add_argument("--env", default="halfcheetah-friction")
    p.add_argument("--srctype", default="medium", help="D4RL source dataset quality")
    p.add_argument("--shift_level", default=0.5)
    p.add_argument("--mode", default=1, type=int,
                   help="only the offline-online loop (1) exists")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--tar_env_interact_interval", default=10, type=int)
    p.add_argument("--max_step", default=int(4e5), type=int)
    p.add_argument("--eval_freq", default=int(5e3), type=int)
    p.add_argument("--eval_episode", default=10, type=int)
    p.add_argument("--params", default=None, help="JSON overriding config keys")
    p.add_argument("--extreme_shift", action="store_true")

    # Flow / gating.
    p.add_argument("--use_ema", action="store_true", default=None)
    p.add_argument("--use_weight", action="store_true", default=None)
    p.add_argument("--filter_percent", default=None, type=float)
    p.add_argument("--start_gate_src_sample", default=None, type=float)
    p.add_argument("--beta", default=None, type=float)
    p.add_argument("--dynamics_train_freq", default=None, type=int)
    p.add_argument("--dynamics_gap_reward_scale", default=None, type=float)
    p.add_argument("--n_samples", default=None, type=int)
    p.add_argument("--downsample_src", default=None, type=float)
    p.add_argument("--upsample_src", action="store_true", default=None)
    p.add_argument("--use_sample_level", action="store_true", default=None)

    # SAC.
    p.add_argument("--temperature_opt", action="store_true", default=None)
    p.add_argument("--actor_lr", default=None, type=float)
    p.add_argument("--critic_lr", default=None, type=float)
    p.add_argument("--alpha", default=None, type=float)
    p.add_argument("--weight", default=None, type=float,
                   help="BC weight in the policy loss")

    # OT / coupling (this port's additions; see docs/COUPLING.md).
    p.add_argument("--eta", default=None, type=float,
                   help="weight of the (s, a) term in the OT transport cost")
    p.add_argument("--coupling", default=None,
                   choices=["identity", "ot", "ot_uncoupled"])
    p.add_argument("--ot_solver", default=None, choices=["sinkhorn", "exact"])
    p.add_argument("--ot_reg", default=None, type=float)
    p.add_argument("--ot_iters", default=None, type=int)
    p.add_argument("--reproduce_original", action="store_true", default=None,
                   help="restore the reference implementation's buggy behaviour")

    p.add_argument("--source_cache_dir", default="source_flow_cache",
                   help="reuse a pretrained source flow across runs, as the "
                        "reference's {task}_source_model_state.pth did; "
                        "set empty to disable")
    p.add_argument("--gap_diagnostics", action="store_true",
                   help="at each evaluation, score the estimated dynamics gap "
                        "against the true one, computed by stepping both the "
                        "source and target simulators from the same state")
    p.add_argument("--gap_probe_size", default=256, type=int)
    p.add_argument("--wandb_project", default=None,
                   help="enable Weights & Biases logging under this project")
    return p


@functools.partial(jax.jit, static_argnames=("cfg", "flow_cfg", "n_steps", "gated"))
def _run_updates(cfg, flow_cfg, state, key, src_buffer, tar_buffer, n_steps, gated):
    """Fuse ``n_steps`` gradient steps into a single dispatch."""
    step_fn = vflow.train_step_postgate if gated else vflow.train_step_pregate

    def body(carry, step_key):
        carry, metrics = step_fn(cfg, flow_cfg, carry, step_key, src_buffer, tar_buffer)
        return carry, metrics

    state, metrics = jax.lax.scan(body, state, jax.random.split(key, n_steps))
    return state, jax.tree.map(lambda x: x[-1], metrics)


def main(argv=None):
    args = build_parser().parse_args(argv)
    signal.signal(signal.SIGTERM, _on_sigterm)

    if args.policy.lower() != "vflow":
        raise SystemExit(
            f"only 'vflow' is ported to JAX; got --policy {args.policy!r}. The "
            "PyTorch baselines (bc_sac, bc_par, bc_vgdf, h2o) are still in algo/.")
    if args.mode != 1:
        raise SystemExit(f"--mode {args.mode} does not exist; only 1 (offline-online)")

    env_name = args.env.replace("_", "-")
    robot = config_lib.robot_of(env_name)
    ref_env_name = f"{env_name}-{args.shift_level}"
    src_d4rl_name = f"{robot}-{args.srctype}-v2"

    print(f"[train] env={env_name} shift={args.shift_level} "
          f"extreme={args.extreme_shift} source={src_d4rl_name} seed={args.seed}")

    # --- environments -----------------------------------------------------
    env_config = {"env_name": env_name, "shift_level": args.shift_level,
                  "extreme_shift": args.extreme_shift}
    tar_env = call_mujoco_env(env_config)
    tar_env.seed(args.seed)
    tar_env.action_space.seed(args.seed)

    state_dim = tar_env.observation_space.shape[0]
    action_dim = tar_env.action_space.shape[0]
    max_action = float(tar_env.action_space.high[0])
    print(f"[train] state_dim={state_dim} action_dim={action_dim} "
          f"max_action={max_action} devices={jax.devices()}")

    # --- config -----------------------------------------------------------
    config = config_lib.load(args.policy, env_name, args, args.params)
    cfg = build.vflow_config(config, state_dim, action_dim, max_action)
    flow_cfg = build.flow_config(config, state_dim, action_dim)

    run_name = "-".join([
        args.policy, env_name, f"src_{args.srctype}", f"sl_{args.shift_level}",
        *(["extreme"] if args.extreme_shift else []), f"s{args.seed}"])
    outdir = pathlib.Path(args.dir) / run_name
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(
        {"config": config, "args": vars(args)}, indent=2, sort_keys=True, default=str))
    print(f"[train] output -> {outdir}")

    # The gate and the refit schedule are independent counters. With the
    # defaults (gate 100000, refit every 5000) the first refit lands exactly on
    # the gate; with other values the filter can spend its first steps deciding
    # on a randomly-initialised adaptation model, i.e. on noise.
    if cfg.start_gate_src_sample % cfg.dynamics_train_freq != 0:
        first_refit = (cfg.start_gate_src_sample // cfg.dynamics_train_freq + 1) \
            * cfg.dynamics_train_freq
        print(f"[train] warning: the gate opens at {cfg.start_gate_src_sample} but the "
              f"first adaptation refit is at {first_refit}. Source filtering will run "
              f"on an untrained adaptation model for {first_refit - cfg.start_gate_src_sample} "
              f"steps. Make start_gate_src_sample a multiple of dynamics_train_freq "
              f"to avoid this.")

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=run_name,
                               config=config, dir=str(outdir))

    # --- data -------------------------------------------------------------
    key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    print(f"[train] loading source dataset {src_d4rl_name}")
    t0 = time.time()
    src_dataset = load_source_dataset(src_d4rl_name)
    src_buffer = buf_lib.from_d4rl(src_dataset)
    print(f"[train] source buffer: {int(src_buffer.size):,} transitions "
          f"({time.time() - t0:.1f}s)")

    if cfg.downsample_src < 1.0:
        key, k = jax.random.split(key)
        src_buffer = buf_lib.downsample(src_buffer, cfg.downsample_src, k)
        print(f"[train] downsampled source buffer to {int(src_buffer.size):,}")

    tar_buffer = buf_lib.empty_buffer(state_dim, action_dim,
                                      max_size=int(config.get("buffer_size", 1e6)))

    # --- agent ------------------------------------------------------------
    key, k_init = jax.random.split(key)
    state = vflow.create_state(cfg, flow_cfg, k_init)

    key, k_flow = jax.random.split(key)
    cache_hit = False
    ckey = None
    if args.source_cache_dir:
        # The source flow depends on the dataset and the architecture, not on
        # the seed or the coupling, so downsample_src goes into the key but
        # nothing else about this run does.
        ckey = flow_cache.cache_key(src_d4rl_name, flow_cfg,
                                    extra={"downsample_src": cfg.downsample_src})
        loaded_flow, cache_hit = flow_cache.load(args.source_cache_dir, ckey,
                                                 state.flow)
        if cache_hit:
            state = state.replace(flow=loaded_flow)

    if cache_hit:
        print(f"[train] reusing cached source flow: "
              f"{flow_cache.paths(args.source_cache_dir, ckey)[0]}")
    else:
        print("[train] pretraining the source flow")
        flow_state, best_val = fm.train_source_flow(
            flow_cfg, state.flow, k_flow, buf_lib.as_batch(src_buffer))
        state = state.replace(flow=flow_state)
        if args.source_cache_dir:
            saved = flow_cache.save(args.source_cache_dir, ckey, state.flow,
                                    {"task": src_d4rl_name, "best_val_mse": best_val,
                                     "downsample_src": cfg.downsample_src,
                                     "seed": args.seed})
            print(f"[train] cached source flow -> {saved}")

    # --- diagnostics ------------------------------------------------------
    # A fixed probe of source (s, a) pairs -- that is where the filter acts --
    # with the true dynamics gap computed once by stepping both simulators.
    gap_probe = None
    true_gap = None
    if args.gap_diagnostics:
        key, k_probe = jax.random.split(key)
        # Oversample: rows whose velocities are clipped in the observation
        # cannot be turned back into a state and get dropped.
        raw = buf_lib.sample(src_buffer, k_probe, args.gap_probe_size * 2)
        probe_obs = np.asarray(raw.state)
        probe_act = np.asarray(raw.action)
        keep = gt.reconstructable(robot, probe_obs)[:]
        idx = np.flatnonzero(keep)[: args.gap_probe_size]

        unshifted = call_mujoco_env({"env_name": robot, "shift_level": 0.5})
        true_gap = gt.true_dynamics_gap(unshifted, tar_env, robot,
                                        probe_obs[idx], probe_act[idx])
        gap_probe = buf_lib.Batch(
            state=jnp.asarray(probe_obs[idx]), action=jnp.asarray(probe_act[idx]),
            next_state=jnp.asarray(probe_obs[idx]),
            reward=jnp.zeros((len(idx), 1)), not_done=jnp.ones((len(idx), 1)))
        print(f"[train] gap diagnostics: {len(idx)} probe rows "
              f"({int(keep.sum())}/{len(keep)} reconstructable); true gap "
              f"mean {true_gap.mean():.4f} std {true_gap.std():.4f}")

    metrics_log = []

    # --- evaluation helpers ----------------------------------------------
    eval_env_fns = [
        functools.partial(call_mujoco_env, env_config) for _ in range(args.eval_episode)]

    def run_eval(step):
        eval_key = jax.random.PRNGKey(args.seed + 100)
        policy_fn = functools.partial(_eval_policy, cfg, state, eval_key)
        ret, length = evaluate(policy_fn, eval_env_fns, args.eval_episode,
                               seeds=[args.seed + 100 + i for i in range(args.eval_episode)])
        score = get_normalized_score(ret, ref_env_name)
        row = {"step": step, "return": ret, "normalized_score": score,
               "episode_length": length}

        if gap_probe is not None and step >= cfg.start_gate_src_sample:
            est = np.asarray(fm.estimate_dynamics_gap(
                flow_cfg, state.flow, jax.random.PRNGKey(12345),
                gap_probe.state, gap_probe.action, cfg.n_samples))
            # How often does the filter drop what it should? The filter keeps
            # the lowest `filter_percent`; precision is the overlap between the
            # rows it drops and the rows that truly have the largest gap.
            n_drop = max(1, int(round(len(est) * (1.0 - cfg.filter_percent))))
            dropped = set(np.argsort(est)[-n_drop:].tolist())
            should_drop = set(np.argsort(true_gap)[-n_drop:].tolist())
            row.update(
                gap_pearson=gt.pearson(est, true_gap),
                gap_spearman=gt.rank_correlation(est, true_gap),
                gap_filter_precision=len(dropped & should_drop) / n_drop,
                gap_filter_chance=n_drop / len(est),
                gap_mean=float(est.mean()), gap_std=float(est.std()))

        metrics_log.append(row)
        (outdir / "metrics.json").write_text(json.dumps(metrics_log, indent=2))

        extra = ""
        if "gap_spearman" in row:
            extra = (f"  | gap rho={row['gap_spearman']:+.3f} "
                     f"r={row['gap_pearson']:+.3f} "
                     f"filter prec={row['gap_filter_precision']:.3f} "
                     f"(chance {row['gap_filter_chance']:.2f})")
        print(f"[eval @ {step}] return {ret:.1f}  normalised {score:.2f}  "
              f"mean length {length:.0f}{extra}", flush=True)
        if wandb_run:
            wandb_run.log({f"test/{k}": v for k, v in row.items() if k != "step"},
                          step=step)
        return ret, score

    # --- training loop ----------------------------------------------------
    interval = int(config.get("tar_env_interact_interval", args.tar_env_interact_interval))
    max_steps = args.max_step
    eval_freq = int(config.get("eval_freq", args.eval_freq))

    obs = tar_env.reset()
    episode_return, episode_len, episode_num = 0.0, 0, 0

    # Track the target buffer's fill level on the host. Reading tar_buffer.size
    # forces a device sync, and the loop consults it three times per iteration
    # (40k iterations over a default run) for a quantity the host already knows.
    tar_size = 0
    tar_capacity = tar_buffer.max_size

    run_eval(0)
    print(f"[train] starting {max_steps} gradient steps "
          f"(1 target interaction per {interval} steps)")

    t = 0
    loop_start = time.time()
    refit_seconds = 0.0
    while t < max_steps:
        if _EXIT_REQUESTED:
            print(f"[train] exiting at step {t}")
            break

        # One target interaction, then `interval` fused gradient steps.
        key, k_act = jax.random.split(key)
        action = np.asarray(vflow.select_action(cfg, state, k_act,
                                                jnp.asarray(obs, jnp.float32), False))
        next_obs, reward, done, info = tar_env.step(action)
        real_done = done and not info.get("TimeLimit.truncated", False)
        tar_buffer = buf_lib.add(tar_buffer, obs, action, next_obs, reward,
                                 float(real_done))
        obs = next_obs
        tar_size = min(tar_size + 1, tar_capacity)
        episode_return += reward
        episode_len += 1

        if done:
            score = get_normalized_score(episode_return, ref_env_name)
            print(f"[train @ {t}] episode {episode_num} len {episode_len} "
                  f"return {episode_return:.1f} normalised {score:.2f}", flush=True)
            if wandb_run:
                wandb_run.log({"train/target_return": episode_return,
                               "train/target_normalized_score": score}, step=t)
            obs = tar_env.reset()
            episode_return, episode_len = 0.0, 0
            episode_num += 1

        # Refit the adaptation flow on schedule, before the steps that use it.
        gated = t >= cfg.start_gate_src_sample
        # The flow driver needs at least one full training batch after the
        # holdout split, not just one buffer-worth of rows.
        enough_target = (tar_size * (1.0 - flow_cfg.holdout_ratio)
                         >= flow_cfg.batch_size)
        if gated and t % cfg.dynamics_train_freq == 0 and enough_target:
            key, k_adapt = jax.random.split(key)
            t_refit = time.time()
            state = state.replace(flow=fm.train_adaptation_flow(
                flow_cfg, state.flow, k_adapt, buf_lib.as_batch(tar_buffer),
                verbose=False)[0])
            refit_seconds += time.time() - t_refit
            print(f"[train @ {t}] refit adaptation flow on "
                  f"{tar_size:,} target transitions "
                  f"({time.time() - t_refit:.1f}s)", flush=True)

        n_steps = min(interval, max_steps - t)
        if tar_size >= cfg.batch_size:
            key, k_upd = jax.random.split(key)
            state, metrics = _run_updates(cfg, flow_cfg, state, k_upd, src_buffer,
                                          tar_buffer, n_steps, gated)
        t += n_steps

        if t % eval_freq < interval:
            run_eval(t)
            elapsed = time.time() - loop_start
            step_seconds = elapsed - refit_seconds
            print(f"[train] {t}/{max_steps} steps in {elapsed / 60:.1f} min "
                  f"({t / max(elapsed, 1e-9):.0f} steps/s) | "
                  f"refits {refit_seconds / 60:.1f} min, "
                  f"steps {step_seconds / 60:.1f} min "
                  f"({1000 * step_seconds / max(t, 1):.2f} ms/step)", flush=True)

    run_eval(t)
    if wandb_run:
        wandb_run.finish()
    print("[train] done")


def _eval_policy(cfg, state, key, obs):
    return vflow.select_action(cfg, state, key, jnp.asarray(obs, jnp.float32), True)


if __name__ == "__main__":
    main()
