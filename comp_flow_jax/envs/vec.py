"""Batched evaluation over several CPU MuJoCo environments.

Stepping the eval episodes in lockstep collapses one policy dispatch per
episode-step into one per timestep. See docs/BACKEND.md for the dispatch cost.
"""

import numpy as np


class SyncVectorEnv:
    """Step a list of envs in lockstep, exposing stacked observations."""

    def __init__(self, env_fns, seeds=None):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        if seeds is not None:
            for env, seed in zip(self.envs, seeds):
                env.seed(int(seed))
                env.action_space.seed(int(seed))
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        return np.stack([env.reset() for env in self.envs])

    def step(self, actions, active):
        """Step only the envs flagged active; return stacked results.

        ``active`` is a boolean array over envs. Inactive envs are skipped
        entirely and report zero reward and ``done=True``.
        """
        obs, rewards, dones, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            if not active[i]:
                obs.append(self._last_obs[i])
                rewards.append(0.0)
                dones.append(True)
                infos.append({})
                continue
            o, r, d, info = env.step(actions[i])
            obs.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        self._last_obs = np.stack(obs)
        return self._last_obs, np.asarray(rewards), np.asarray(dones), infos

    def close(self):
        for env in self.envs:
            env.close()


def evaluate(policy_fn, env_fns, n_episodes, seeds=None, max_steps=1000):
    """Run ``n_episodes`` episodes in lockstep and return the mean return.

    ``policy_fn`` maps a stacked observation array to a stacked action array.
    Returns ``(mean_return, mean_length)``.
    """
    vec = SyncVectorEnv(env_fns[:n_episodes], seeds)
    obs = vec.reset()
    vec._last_obs = obs

    returns = np.zeros(vec.num_envs)
    lengths = np.zeros(vec.num_envs, dtype=int)
    active = np.ones(vec.num_envs, dtype=bool)

    for _ in range(max_steps):
        if not active.any():
            break
        actions = np.asarray(policy_fn(obs))
        obs, rewards, dones, _ = vec.step(actions, active)
        returns += rewards * active
        lengths += active
        active &= ~np.asarray(dones)

    vec.close()
    return float(returns.mean()), float(lengths.mean())
