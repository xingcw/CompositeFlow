"""Turn a flat config dict into the typed configs the algorithm wants."""

from .algos.vflow import VFlowConfig
from .flow.flow_matching import FlowConfig


def flow_config(config, state_dim, action_dim):
    reproduce = bool(config.get("reproduce_original", False))
    # --reproduce_original selects the reference's coupling unless the user
    # named one explicitly. See docs/COUPLING.md.
    coupling = config.get("coupling") or ("ot_uncoupled" if reproduce else "ot")

    return FlowConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=int(config.get("flow_matching_hidden_dim", 512)),
        source_num_layers=int(config.get("flow_matching_source_num_layers", 6)),
        adaptation_num_layers=int(config.get("flow_matching_adaptation_num_layers", 6)),
        activation=config.get("flow_matching_activation", "relu"),
        source_flow_steps=int(config.get("source_flow_steps", 11)),
        adaptation_flow_steps=int(config.get("adaptation_flow_steps", 11)),
        sigma=float(config.get("flow_matching_sigma", 0.0)),
        adaptation_sigma=float(config.get("flow_matching_adaptation_sigma", 0.0)),
        lr=float(config.get("flow_matching_lr", 3e-4)),
        batch_size=int(config.get("flow_matching_batch_size", 1024)),
        holdout_ratio=float(config.get("flow_matching_holdout_ratio", 0.1)),
        source_epochs=int(config.get("flow_matching_training_max_epochs_source", 500)),
        adaptation_epochs=int(config.get("flow_matching_training_max_epochs_adaptation", 200)),
        max_epochs_since_update=int(config.get("max_epochs_since_update", 20)),
        validation_start_epoch_source=int(config.get("validation_start_epoch_source", 20)),
        validation_start_epoch_adaptation=int(config.get("validation_start_epoch_adaptation", 20)),
        use_ema=bool(config.get("use_ema", False)),
        ema_decay=float(config.get("ema_decay", 0.995)),
        ema_update_steps=int(config.get("ema_update_steps", 20)),
        coupling=coupling,
        shard_gap=bool(config.get("shard_gap", True)),
        shard_refit=bool(config.get("shard_refit", True)),
        ot_solver=config.get("ot_solver", "exact"),
        ot_reg=float(config.get("ot_reg", 0.05)),
        ot_iters=int(config.get("ot_iters", 200)),
        eta=float(config.get("eta", 0.05)),
        reproduce_original=reproduce,
    )


def vflow_config(config, state_dim, action_dim, max_action):
    return VFlowConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=float(max_action),
        gamma=float(config.get("gamma", 0.99)),
        tau=float(config.get("tau", 0.005)),
        hidden_sizes=int(config.get("hidden_sizes", 256)),
        batch_size=int(config.get("batch_size", 128)),
        actor_lr=float(config.get("actor_lr", 3e-4)),
        critic_lr=float(config.get("critic_lr", 3e-4)),
        alpha=float(config.get("alpha", 0.2)),
        temperature_opt=bool(config.get("temperature_opt", False)),
        backup_entropy=bool(config.get("backup_entropy", True)),
        policy_bc_weight=float(config.get("weight", 2.5)),
        start_gate_src_sample=int(float(config.get("start_gate_src_sample", 100_000))),
        dynamics_train_freq=int(config.get("dynamics_train_freq", 5_000)),
        filter_percent=float(config.get("filter_percent", 0.7)),
        use_weight=bool(config.get("use_weight", False)),
        beta=float(config.get("beta", 1.0)),
        use_sample_level=bool(config.get("use_sample_level", False)),
        dynamics_gap_reward_scale=float(config.get("dynamics_gap_reward_scale", 0.0)),
        n_samples=int(config.get("n_samples", 100)),
        upsample_src=bool(config.get("upsample_src", False)),
        downsample_src=float(config.get("downsample_src", 1.0)),
        reproduce_original=bool(config.get("reproduce_original", False)),
    )
