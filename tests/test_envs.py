"""Environment smoke tests: every asset loads, dims match D4RL, shifts differ."""

import numpy as np

import mujoco

from comp_flow_jax.envs.registry import ASSET_DIR, call_mujoco_env

EXPECTED_DIMS = {"hopper": (11, 3), "walker2d": (17, 6),
                 "halfcheetah": (17, 6), "ant": (111, 8)}

CASES = [
    ("hopper-friction", 5.0), ("hopper-gravity", 0.1),
    ("walker2d-friction", 2.0), ("walker2d-kinematic-thighjnt", "medium"),
    ("halfcheetah-gravity", 5.0), ("halfcheetah-morph-thigh", "hard"),
    ("ant-friction", 0.5), ("ant-morph-alllegs", "easy"),
]


def test_all_assets_load():
    """All 89 converted XMLs must load under the runtime MuJoCo."""
    paths = sorted(ASSET_DIR.glob("*.xml"))
    assert len(paths) == 89, f"expected 89 assets, found {len(paths)}"
    for path in paths:
        mujoco.MjModel.from_xml_path(str(path))
    return len(paths)


def test_dims_match_d4rl():
    """Observation/action dims must match the D4RL v2 datasets."""
    for name, shift in CASES:
        env = call_mujoco_env({"env_name": name, "shift_level": shift})
        robot = name.split("-")[0]
        exp_obs, exp_act = EXPECTED_DIMS[robot]
        env.seed(0)
        obs = env.reset()
        assert obs.shape == (exp_obs,), f"{name}: obs {obs.shape} != ({exp_obs},)"
        assert env.action_space.shape == (exp_act,), f"{name}: action shape"


def test_episode_runs_and_time_limit():
    env = call_mujoco_env({"env_name": "halfcheetah-friction", "shift_level": 5.0})
    env.seed(0)
    env.action_space.seed(0)
    env.reset()
    steps = 0
    done = False
    while not done:
        _, _, done, info = env.step(env.action_space.sample())
        steps += 1
    # HalfCheetah never terminates, so this must be the 1000-step limit.
    assert steps == 1000, f"expected the time limit at 1000 steps, got {steps}"
    assert info.get("TimeLimit.truncated") is True


def test_shift_parameters_take_effect():
    """A friction/gravity shift must actually change the compiled model."""
    base = call_mujoco_env({"env_name": "hopper", "shift_level": 0.5})
    fric = call_mujoco_env({"env_name": "hopper-friction", "shift_level": 5.0})
    grav = call_mujoco_env({"env_name": "hopper-gravity", "shift_level": 5.0})

    def foot_friction(env):
        gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_geom")
        return float(env.model.geom_friction[gid][0])

    assert np.isclose(foot_friction(fric) / foot_friction(base), 5.0), "friction 5x"
    assert np.isclose(grav.model.opt.gravity[2] / base.model.opt.gravity[2], 5.0), "gravity 5x"
    return foot_friction(base), foot_friction(fric), float(grav.model.opt.gravity[2])


def test_ant_contact_cost_is_live():
    """cfrc_ext must be refreshed after mj_step, or contact_cost is silently 0.

    mj_step leaves cfrc_ext one step stale unless mj_rnePostConstraint is called
    explicitly; gymnasium's Ant-v4 shipped with exactly this bug.
    """
    env = call_mujoco_env({"env_name": "ant-friction", "shift_level": 0.5})
    env.seed(0)
    env.reset()
    saw_contact = False
    for _ in range(200):
        _, _, done, info = env.step(np.zeros(env.action_space.shape[0]))
        if info["reward_contact"] != 0.0:
            saw_contact = True
            break
        if done:
            env.reset()
    assert saw_contact, "contact cost stayed exactly zero; cfrc_ext is not being refreshed"


if __name__ == "__main__":
    n = test_all_assets_load()
    print(f"all {n} assets load                OK")
    test_dims_match_d4rl()
    print("obs/action dims match D4RL        OK")
    test_episode_runs_and_time_limit()
    print("time limit + truncation flag      OK")
    b, f, g = test_shift_parameters_take_effect()
    print(f"shift parameters applied          OK  (foot friction {b} -> {f}, gravity z {g})")
    test_ant_contact_cost_is_live()
    print("ant contact cost is live          OK")
