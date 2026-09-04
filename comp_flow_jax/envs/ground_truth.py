"""Ground truth for the dynamics gap.

MuJoCo is deterministic and the two domains are the same robot with a modified
XML, so ||f_target(s,a) - f_source(s,a)|| is directly computable: reconstruct
(qpos, qvel) from an observation and step both simulators from it.

Diagnostics only; training never calls this. See docs/COUPLING.md.
"""

import numpy as np

# obs = qpos[SKIP:] ++ (clipped) qvel, per gym's *_v3 _get_obs. VEL_CLIP is the
# clip gym applies to qvel before it reaches the observation -- where it bites,
# the state cannot be reconstructed and the sample has to be dropped.
LAYOUT = {
    # robot:      (qpos skip, nq, nv, qvel clip or None)
    "hopper":      (1, 6, 6, 10.0),
    "walker2d":    (1, 9, 9, 10.0),
    "halfcheetah": (1, 9, 9, None),
    "ant":         (2, 15, 14, None),
}


def obs_to_state(robot, obs):
    """(qpos, qvel) for one observation. The dropped root coordinates are set
    to zero: the ground is flat, so they do not affect the dynamics, and they
    are excluded from the next observation anyway."""
    skip, nq, nv, _ = LAYOUT[robot]
    n_pos = nq - skip
    qpos = np.zeros(nq)
    qpos[skip:] = obs[:n_pos]
    qvel = np.asarray(obs[n_pos:n_pos + nv], dtype=np.float64)
    return qpos, qvel


def reconstructable(robot, obs_batch, margin=0.1):
    """Mask of rows whose velocities are not sitting on the clip boundary."""
    skip, nq, nv, clip = LAYOUT[robot]
    if clip is None:
        return np.ones(len(obs_batch), dtype=bool)
    n_pos = nq - skip
    vel = np.asarray(obs_batch)[:, n_pos:n_pos + nv]
    return np.all(np.abs(vel) < clip - margin, axis=1)


def _step_from(env, robot, obs, action):
    qpos, qvel = obs_to_state(robot, obs)
    env.set_state(qpos, qvel)
    # Bypass TimeLimit.step: this is a probe, not an episode.
    env.do_simulation(action, env.frame_skip)
    return env._get_obs()


def true_dynamics_gap(source_env, target_env, robot, obs_batch, action_batch):
    """||f_target(s,a) - f_source(s,a)|| for each row.

    Both envs must be the same robot, so that one observation reconstructs a
    valid state in each.
    """
    if (source_env.model.nq, source_env.model.nv) != \
            (target_env.model.nq, target_env.model.nv):
        raise ValueError(
            "source and target models disagree on nq/nv "
            f"({source_env.model.nq},{source_env.model.nv}) vs "
            f"({target_env.model.nq},{target_env.model.nv}); a shared "
            "observation cannot reconstruct a state in both")

    gaps = np.empty(len(obs_batch))
    for k, (obs, action) in enumerate(zip(obs_batch, action_batch)):
        ns_src = _step_from(source_env, robot, obs, action)
        ns_tar = _step_from(target_env, robot, obs, action)
        gaps[k] = np.linalg.norm(ns_tar - ns_src)
    return gaps


def rank_correlation(x, y):
    """Spearman rho without pulling in scipy."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
