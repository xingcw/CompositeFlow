"""D4RL source datasets, without the d4rl package.

d4rl needs mujoco-py, gym 0.18 and Python 3.8. The v2 MuJoCo hdf5 files are
mirrored on the HuggingFace dataset imone/D4RL, and qlearning_dataset below is
a transcription of d4rl's. See docs/DEVIATIONS.md.
"""

import os
import pathlib

import h5py
import numpy as np

HF_REPO = "imone/D4RL"
DEFAULT_CACHE = pathlib.Path(
    os.environ.get("COMP_FLOW_DATA_DIR",
                   pathlib.Path.home() / ".cache" / "comp_flow_jax" / "d4rl"))

# gym MuJoCo locomotion tasks all use a 1000-step limit.
MAX_EPISODE_STEPS = 1000

ROBOTS = ("hopper", "walker2d", "halfcheetah", "ant")
QUALITIES = ("random", "medium", "expert", "medium-replay", "medium-expert", "full-replay")


def hf_filename(d4rl_name):
    """``hopper-medium-replay-v2`` -> ``hopper_medium_replay-v2.hdf5``."""
    if not d4rl_name.endswith("-v2"):
        raise ValueError(f"only the v2 MuJoCo datasets are mirrored, got {d4rl_name!r}")
    stem = d4rl_name[: -len("-v2")]
    return f"{stem.replace('-', '_')}-v2.hdf5"


def download(d4rl_name, cache_dir=None):
    """Fetch one D4RL v2 hdf5 from the HF mirror, returning the local path."""
    from huggingface_hub import hf_hub_download

    cache_dir = pathlib.Path(cache_dir or DEFAULT_CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename=hf_filename(d4rl_name),
        local_dir=str(cache_dir),
    )
    return pathlib.Path(path)


def _read_h5(path):
    out = {}

    def visit(name, item):
        if isinstance(item, h5py.Dataset):
            out[name] = item[()]

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return out


def get_dataset(d4rl_name, cache_dir=None):
    """Raw D4RL dataset dict, as ``env.get_dataset()`` would return it."""
    return _read_h5(download(d4rl_name, cache_dir))


def qlearning_dataset(dataset, terminate_on_end=False,
                      max_episode_steps=MAX_EPISODE_STEPS):
    """Transcription of d4rl.qlearning_dataset.

    Pairs consecutive observations; the transition straddling a timeout is
    dropped. The timeouts branch is vectorised and checked bit-identical
    against the loop in tests/test_d4rl_hf.py.
    """
    n = dataset["rewards"].shape[0]

    if not terminate_on_end and "timeouts" in dataset:
        keep = ~dataset["timeouts"][: n - 1].astype(bool)
        return {
            "observations": dataset["observations"][: n - 1][keep].astype(np.float32),
            "actions": dataset["actions"][: n - 1][keep].astype(np.float32),
            "next_observations": dataset["observations"][1:n][keep].astype(np.float32),
            "rewards": dataset["rewards"][: n - 1][keep].astype(np.float32),
            "terminals": dataset["terminals"][: n - 1][keep].astype(bool),
        }
    return _qlearning_dataset_loop(dataset, terminate_on_end, max_episode_steps)


def _qlearning_dataset_loop(dataset, terminate_on_end, max_episode_steps):
    """Literal transcription of the d4rl loop; the fast path is checked against it."""
    n = dataset["rewards"].shape[0]
    obs_, next_obs_, action_, reward_, done_ = [], [], [], [], []

    # Newer dataset revisions carry an explicit timeouts field; older ones are
    # segmented by counting steps.
    use_timeouts = "timeouts" in dataset

    episode_step = 0
    for i in range(n - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i])

        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == max_episode_steps - 1

        if (not terminate_on_end) and final_timestep:
            # Last step of an episode: the pairing across the reset boundary is
            # meaningless, so drop it.
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
    }


def load_source_dataset(d4rl_name, cache_dir=None):
    """Download + convert in one call, as ``train.py`` needs it."""
    return qlearning_dataset(get_dataset(d4rl_name, cache_dir))
