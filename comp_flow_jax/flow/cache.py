"""Source-flow checkpoint cache.

Mirrors the reference's {task_name}_source_model_state.pth. The source flow
depends only on the dataset and the architecture, so every arm of a sweep
shares one instead of training its own.
"""

import hashlib
import json
import pathlib

import jax.numpy as jnp
from flax import serialization

DEFAULT_CACHE_DIR = pathlib.Path("source_flow_cache")

# Fields that change what the source model *is*. A cache entry is only valid
# for an identical tuple of these.
_KEY_FIELDS = ("state_dim", "action_dim", "hidden_dim", "source_num_layers",
               "activation", "source_flow_steps", "sigma", "lr", "batch_size",
               "holdout_ratio", "source_epochs", "max_epochs_since_update",
               "validation_start_epoch_source", "use_ema", "ema_decay")


def cache_key(task_name, cfg, extra=None):
    """Task name plus a short hash of everything that defines the model."""
    payload = {f: getattr(cfg, f) for f in _KEY_FIELDS}
    payload.update(extra or {})
    digest = hashlib.md5(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return f"{task_name}_source_flow_{digest}"


def paths(cache_dir, key):
    cache_dir = pathlib.Path(cache_dir)
    return cache_dir / f"{key}.msgpack", cache_dir / f"{key}.json"


def save(cache_dir, key, flow_state, meta):
    """Persist the source params and the source normalisers."""
    blob_path, meta_path = paths(cache_dir, key)
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": flow_state.source.params,
        "ema": flow_state.source_ema,
        "sa_src_mean": flow_state.sa_src.mean, "sa_src_std": flow_state.sa_src.std,
        "ns_src_mean": flow_state.ns_src.mean, "ns_src_std": flow_state.ns_src.std,
    }
    blob_path.write_bytes(serialization.to_bytes(payload))
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return blob_path


def load(cache_dir, key, flow_state):
    """Restore into ``flow_state``; returns ``(state, True)`` on a hit."""
    blob_path, _ = paths(cache_dir, key)
    if not blob_path.exists():
        return flow_state, False

    template = {
        "params": flow_state.source.params,
        "ema": flow_state.source_ema,
        "sa_src_mean": flow_state.sa_src.mean, "sa_src_std": flow_state.sa_src.std,
        "ns_src_mean": flow_state.ns_src.mean, "ns_src_std": flow_state.ns_src.std,
    }
    payload = serialization.from_bytes(template, blob_path.read_bytes())

    return flow_state.replace(
        source=flow_state.source.replace(params=payload["params"]),
        source_ema=payload["ema"],
        sa_src=flow_state.sa_src.replace(mean=jnp.asarray(payload["sa_src_mean"]),
                                         std=jnp.asarray(payload["sa_src_std"])),
        ns_src=flow_state.ns_src.replace(mean=jnp.asarray(payload["ns_src_mean"]),
                                         std=jnp.asarray(payload["ns_src_std"])),
        source_fitted=True,
    ), True
