"""Hopper / Walker2d / HalfCheetah / Ant on the modern mujoco bindings.

Ported from gym 0.18.3's envs/mujoco/*_v3.py, the versions the reference
instantiates. gym reaches MuJoCo through mujoco-py, which cannot be installed
here. Deviations are in docs/DEVIATIONS.md.
"""

import numpy as np

import mujoco

from .spaces import Box


class MujocoEnv:
    """Port of gym 0.18.3 ``mujoco_env.MujocoEnv`` onto the mujoco bindings."""

    def __init__(self, xml_file, frame_skip):
        self.frame_skip = frame_skip
        self.model = mujoco.MjModel.from_xml_path(str(xml_file))
        self.data = mujoco.MjData(self.model)

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = Box(low=low, high=high, dtype=np.float32)

        self.np_random = np.random.RandomState()

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=obs.shape,
                                     dtype=np.float64)

        self.metadata = {"render.modes": [], "video.frames_per_second": int(np.round(1.0 / self.dt))}

    # --- gym.Env surface -------------------------------------------------

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self.reset_model()

    def close(self):
        pass

    def render(self, *args, **kwargs):
        raise NotImplementedError("rendering is not wired up in the JAX port")

    # --- simulation helpers ----------------------------------------------

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def do_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=n_frames)
        # mj_step leaves derived quantities (xpos, cfrc_ext, ...) one step
        # stale relative to mujoco-py's MjSim.step; refresh them so that
        # get_body_com / contact_forces read what gym read.
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def get_body_com(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"no body named {body_name!r} in this model")
        return self.data.xpos[body_id]

    def state_vector(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    # --- subclass hooks ---------------------------------------------------

    def reset_model(self):
        raise NotImplementedError

    def _get_obs(self):
        raise NotImplementedError


class HopperEnv(MujocoEnv):
    def __init__(self, xml_file, forward_reward_weight=1.0, ctrl_cost_weight=1e-3,
                 healthy_reward=1.0, terminate_when_unhealthy=True,
                 healthy_state_range=(-100.0, 100.0), healthy_z_range=(0.7, float("inf")),
                 healthy_angle_range=(-0.2, 0.2), reset_noise_scale=5e-3,
                 exclude_current_positions_from_observation=True):
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        super().__init__(xml_file, 4)

    @property
    def healthy_reward(self):
        return float(self.is_healthy or self._terminate_when_unhealthy) * self._healthy_reward

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        return all((healthy_state, healthy_z, healthy_angle))

    @property
    def done(self):
        return (not self.is_healthy) if self._terminate_when_unhealthy else False

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)
        if self._exclude_current_positions_from_observation:
            position = position[1:]
        return np.concatenate((position, velocity)).ravel()

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        observation = self._get_obs()
        reward = forward_reward + healthy_reward - ctrl_cost
        done = self.done
        info = {"x_position": x_position_after, "x_velocity": x_velocity}
        return observation, reward, done, info

    def reset_model(self):
        n = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(low=-n, high=n, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-n, high=n, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()


class Walker2dEnv(MujocoEnv):
    def __init__(self, xml_file, forward_reward_weight=1.0, ctrl_cost_weight=1e-3,
                 healthy_reward=1.0, terminate_when_unhealthy=True,
                 healthy_z_range=(0.8, 2.0), healthy_angle_range=(-1.0, 1.0),
                 reset_noise_scale=5e-3,
                 exclude_current_positions_from_observation=True):
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        super().__init__(xml_file, 4)

    @property
    def healthy_reward(self):
        return float(self.is_healthy or self._terminate_when_unhealthy) * self._healthy_reward

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range
        return (min_z < z < max_z) and (min_angle < angle < max_angle)

    @property
    def done(self):
        return (not self.is_healthy) if self._terminate_when_unhealthy else False

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)
        if self._exclude_current_positions_from_observation:
            position = position[1:]
        return np.concatenate((position, velocity)).ravel()

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        observation = self._get_obs()
        reward = forward_reward + healthy_reward - ctrl_cost
        done = self.done
        info = {"x_position": x_position_after, "x_velocity": x_velocity}
        return observation, reward, done, info

    def reset_model(self):
        n = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(low=-n, high=n, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-n, high=n, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()


class HalfCheetahEnv(MujocoEnv):
    def __init__(self, xml_file, forward_reward_weight=1.0, ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True):
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        super().__init__(xml_file, 5)

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        if self._exclude_current_positions_from_observation:
            position = position[1:]
        return np.concatenate((position, velocity)).ravel()

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)
        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        info = {"x_position": x_position_after, "x_velocity": x_velocity,
                "reward_run": forward_reward, "reward_ctrl": -ctrl_cost}
        return observation, reward, False, info

    def reset_model(self):
        n = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(low=-n, high=n, size=self.model.nq)
        qvel = self.init_qvel + n * self.np_random.randn(self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()


class AntEnv(MujocoEnv):
    def __init__(self, xml_file, ctrl_cost_weight=0.5, contact_cost_weight=5e-4,
                 healthy_reward=1.0, terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 1.0), contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True):
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        super().__init__(xml_file, 5)

    @property
    def healthy_reward(self):
        return float(self.is_healthy or self._terminate_when_unhealthy) * self._healthy_reward

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    @property
    def contact_forces(self):
        min_value, max_value = self._contact_force_range
        return np.clip(self.data.cfrc_ext, min_value, max_value)

    @property
    def contact_cost(self):
        return self._contact_cost_weight * np.sum(np.square(self.contact_forces))

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        return bool(np.isfinite(state).all() and min_z <= state[2] <= max_z)

    @property
    def done(self):
        return (not self.is_healthy) if self._terminate_when_unhealthy else False

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()
        if self._exclude_current_positions_from_observation:
            position = position[2:]
        return np.concatenate((position, velocity, contact_force))

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        x_velocity, y_velocity = (xy_position_after - xy_position_before) / self.dt

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
        done = self.done
        observation = self._get_obs()
        info = {"reward_forward": forward_reward, "reward_ctrl": -ctrl_cost,
                "reward_contact": -contact_cost, "reward_survive": healthy_reward,
                "x_position": xy_position_after[0], "y_position": xy_position_after[1],
                "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
                "x_velocity": x_velocity, "y_velocity": y_velocity,
                "forward_reward": forward_reward}
        return observation, reward, done, info

    def reset_model(self):
        n = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(low=-n, high=n, size=self.model.nq)
        qvel = self.init_qvel + n * self.np_random.randn(self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()


class TimeLimit:
    """Port of ``gym.wrappers.TimeLimit``, including the truncation info key."""

    def __init__(self, env, max_episode_steps):
        self.env = env
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def __getattr__(self, name):
        # Only reached for attributes TimeLimit itself does not define.
        return getattr(self.env, name)

    def step(self, action):
        assert self._elapsed_steps is not None, "step() called before reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
