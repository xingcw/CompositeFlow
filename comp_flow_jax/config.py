"""Config assembly: YAML defaults, then CLI overrides.

Reads the same config/{domain}/{policy}/{robot}.yaml files as the reference.
"""

import json
import pathlib

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]

# Keys train.py copies from argparse over the YAML, verbatim from its
# ``override_keys`` list plus the flow/OT knobs this port adds.
OVERRIDE_KEYS = (
    "use_ema", "use_weight", "filter_percent", "start_gate_src_sample", "beta",
    "dynamics_train_freq", "temperature_opt", "actor_lr", "critic_lr", "alpha",
    "dynamics_gap_reward_scale", "n_samples", "downsample_src", "upsample_src",
    "use_sample_level", "weight", "eval_freq", "eta",
    "coupling", "ot_solver", "ot_reg", "ot_iters", "reproduce_original",
)

ROBOTS = ("halfcheetah", "hopper", "walker2d", "ant")


def detect_domain(env_name):
    if any(r in env_name for r in ("halfcheetah", "hopper", "walker2d")) \
            or env_name.split("-")[0] == "ant":
        return "mujoco"
    raise NotImplementedError(
        f"only the mujoco domain is ported; got {env_name!r}. The reference also "
        "named adroit and antmaze but never wired an env factory for either.")


def robot_of(env_name):
    return env_name.split("-")[0]


def load(policy, env_name, args, params_json=None):
    """YAML defaults for (policy, robot), then --params, then CLI overrides."""
    domain = detect_domain(env_name)
    robot = robot_of(env_name)
    path = REPO / "config" / domain / policy.lower() / f"{robot}.yaml"

    if path.exists():
        config = yaml.safe_load(path.read_text()) or {}
    else:
        print(f"[config] {path} not found; using defaults and CLI arguments only")
        config = {}

    if params_json:
        config.update(json.loads(params_json))

    for key in OVERRIDE_KEYS:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    config.setdefault("gamma", 0.99)
    config.setdefault("tau", 0.005)
    config.setdefault("hidden_sizes", 256)
    config.setdefault("batch_size", 256)
    config["domain"] = domain
    config["robot"] = robot
    return config
