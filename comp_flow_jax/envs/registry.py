"""Name -> environment construction, mirroring envs/mujoco/call_mujoco_env.py.

Same naming scheme, resolved against the converted assets in assets_v3.
"""

import pathlib

from .mujoco_envs import AntEnv, HalfCheetahEnv, HopperEnv, TimeLimit, Walker2dEnv

ASSET_DIR = pathlib.Path(__file__).resolve().parent / "assets_v3"

MAX_EPISODE_STEPS = 1000

_ROBOTS = {
    "hopper": HopperEnv,
    "halfcheetah": HalfCheetahEnv,
    "walker2d": Walker2dEnv,
    "ant": AntEnv,
}

# Shift families whose level is a float scale vs. an easy/medium/hard label.
_SCALE_SHIFTS = ("friction", "gravity")
_LABEL_SHIFTS = ("morph", "kinematic")

VALID_SCALES = (0.1, 0.5, 2.0, 5.0)
VALID_LABELS = ("easy", "medium", "hard")


def _robot_of(env_name):
    for robot in ("halfcheetah", "walker2d", "hopper"):
        if robot in env_name:
            return robot
    if env_name.split("_")[0] == "ant":
        return "ant"
    raise NotImplementedError(f"no robot recognised in env name {env_name!r}")


def asset_path(env_name, shift_level, extreme_shift=False):
    """Resolve an env name + shift level to a converted XML path."""
    env_name = env_name.lower().replace("-", "_")

    if any(k in env_name for k in _LABEL_SHIFTS):
        if shift_level not in VALID_LABELS:
            raise ValueError(
                f"{env_name} is a morph/kinematic shift and needs a level in "
                f"{VALID_LABELS}, got {shift_level!r}")
        suffix = f"_{shift_level}"
    elif any(k in env_name for k in _SCALE_SHIFTS):
        if float(shift_level) not in VALID_SCALES:
            raise ValueError(
                f"{env_name} is a friction/gravity shift and needs a level in "
                f"{VALID_SCALES}, got {shift_level!r}")
        suffix = f"_{float(shift_level)}"
    elif env_name in _ROBOTS:
        suffix = ""  # unshifted reference robot
    else:
        raise NotImplementedError(f"unsupported env name {env_name!r}")

    if extreme_shift:
        suffix += "_extreme"

    path = ASSET_DIR / f"{env_name}{suffix}.xml"
    if not path.exists():
        raise FileNotFoundError(
            f"no asset for env={env_name!r} shift_level={shift_level!r} "
            f"extreme_shift={extreme_shift}: looked for {path}")
    return path


def call_mujoco_env(env_config):
    """Build a TimeLimit-wrapped env from the reference's config dict."""
    env_name = env_config["env_name"].lower().replace("-", "_")
    shift_level = env_config["shift_level"]
    extreme = env_config.get("extreme_shift", False)

    if "noise" in env_name:
        raise NotImplementedError(
            "the 'noise' shift family was never implemented upstream either")

    robot = _robot_of(env_name)
    path = asset_path(env_name, shift_level, extreme)
    return TimeLimit(_ROBOTS[robot](xml_file=path), max_episode_steps=MAX_EPISODE_STEPS)
