"""Run the reference PyTorch backbone and dump weights + activations.

Run under a venv that has torch (see tests/README.md); the JAX side is
tests/test_backbone_parity.py, which reloads the npz and compares.
"""

import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "algo" / "offline_online" / "guided_flow"))

from backbone.mlp import ResidualMLPGuidance, SinusoidalPosEmb  # noqa: E402

OUT = REPO / "tests" / "_parity_backbone.npz"

D_IN, COND_DIM, WIDTH, LAYERS, DIM_T, B = 11, 14, 512, 6, 128, 8


def main():
    torch.manual_seed(0)
    model = ResidualMLPGuidance(d_in=D_IN, cond_dim=COND_DIM, mlp_width=WIDTH,
                                num_layers=LAYERS, activation="relu").eval()

    rng = np.random.RandomState(0)
    x = rng.randn(B, D_IN).astype(np.float32)
    t = rng.rand(B, 1).astype(np.float32)
    cond = rng.randn(B, COND_DIM).astype(np.float32)

    with torch.no_grad():
        y = model(torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(cond)).numpy()
        t_emb = SinusoidalPosEmb(DIM_T)(torch.from_numpy(t)).numpy()

    blob = {f"w/{k}": v.numpy() for k, v in model.state_dict().items()}
    blob.update({"x": x, "t": t, "cond": cond, "y": y, "t_emb": t_emb})
    np.savez(OUT, **blob)
    print(f"wrote {OUT}  ({len(blob)} arrays)  out={y.shape}  |y|mean={np.abs(y).mean():.6f}")
    print("state_dict keys:")
    for k in model.state_dict():
        print("   ", k, tuple(model.state_dict()[k].shape))




def dump_actor_critic():
    """Second fixture: the SAC actor/critic from vflow.py."""
    sys.path.insert(0, str(REPO / "algo" / "offline_online"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vflow_ref", REPO / "algo" / "offline_online" / "vflow.py")
    # vflow.py imports the flow package relatively; load only the classes we
    # need by exec'ing the file with a stub for the relative import.
    src = (REPO / "algo" / "offline_online" / "vflow.py").read_text()
    src = src.replace("from .guided_flow.flow_matching import FlowMatching",
                      "FlowMatching = None")
    src = src.replace("import wandb", "wandb = None")
    ns = {"__name__": "vflow_ref"}
    exec(compile(src, "vflow.py", "exec"), ns)

    torch.manual_seed(1)
    S, A, H, B = 11, 3, 256, 8
    policy = ns["Policy"](S, A, 1.0, hidden_size=H).eval()
    critic = ns["DoubleQFunc"](S, A, hidden_size=H).eval()

    rng = np.random.RandomState(1)
    s = rng.randn(B, S).astype(np.float32)
    a = rng.randn(B, A).astype(np.float32).clip(-0.99, 0.99)

    with torch.no_grad():
        mu_logstd = policy.network(torch.from_numpy(s))
        mu, logstd = mu_logstd.chunk(2, dim=-1)
        logstd_c = torch.clamp(logstd, -20, 2)
        _, _, mean_action = policy(torch.from_numpy(s))
        q1, q2 = critic(torch.from_numpy(s), torch.from_numpy(a))

        # log-prob of a fixed pre-squash sample, to check the tanh correction
        x = mu + logstd_c.exp() * torch.from_numpy(rng.randn(B, A).astype(np.float32))
        from torch.distributions import Normal, TransformedDistribution
        dist = TransformedDistribution(Normal(mu, logstd_c.exp()),
                                       [ns["TanhTransform"](cache_size=1)])
        y = torch.tanh(x)
        logp = (Normal(mu, logstd_c.exp()).log_prob(x)
                - dist.transforms[0].log_abs_det_jacobian(x, y)).sum(-1)

    blob = {f"pw/{k}": v.numpy() for k, v in policy.state_dict().items()}
    blob.update({f"cw/{k}": v.numpy() for k, v in critic.state_dict().items()})
    blob.update({"s": s, "a": a, "mu": mu.numpy(), "log_std": logstd_c.numpy(),
                 "mean_action": mean_action.numpy(), "q1": q1.numpy(), "q2": q2.numpy(),
                 "x": x.numpy(), "logp": logp.numpy()})
    out = REPO / "tests" / "_parity_actor_critic.npz"
    np.savez(out, **blob)
    print(f"wrote {out}  mu{mu.shape} q1{q1.shape} logp{logp.shape}")
    print("policy keys:", list(policy.state_dict()))
    print("critic keys:", list(critic.state_dict()))


if __name__ == "__main__":
    main()
    dump_actor_critic()
