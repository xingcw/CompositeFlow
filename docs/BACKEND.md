# Backend choices, and the measurements behind them

Everything here was measured on the machine this port targets:

* **TPU v4-8** — 4 chips, `coords=(0,0,0)..(1,1,0)`, JAX 0.6.2
* **AMD EPYC 7B12**, 240 vCPU, 400 GB RAM
* no GPU

## Physics runs on the CPU, not on MJX

The obvious move on a TPU box is `mujoco-mjx`. It is the wrong move for this
algorithm. Measured single-environment throughput:

| backend | parallel envs | throughput |
| --- | --- | --- |
| CPU MuJoCo 3.10, halfcheetah | 1 | **50,730 sim-step/s** |
| CPU MuJoCo 3.10, hopper | 1 | **22,605 sim-step/s** |
| MJX on TPU | 1 | 315 step/s |
| MJX on TPU | 256 | 83,531 step/s |
| MJX on TPU | 2048 | 199,287 step/s |

Single-environment MJX is **160x slower** than CPU MuJoCo, because a TPU is bad
at the small sequential ops a physics step is made of. MJX only wins with
thousands of environments in flight.

This algorithm never has thousands in flight. It takes **one target-environment
step per ten gradient steps**, so a 400k-step run performs about 40k
interactions in total, plus ~800k more for evaluation. CPU MuJoCo absorbs that
without noticing. Reaching MJX's break-even would mean running hundreds of
parallel target environments, which changes the online interaction budget the
benchmark is built around — a different experiment, not a faster one.

So: **physics on the CPU (240 cores are plenty), learning on the TPU.**
`envs/registry.py` is the only place that knows this, and an MJX backend could
be added behind it if massively parallel target sampling ever becomes the goal.

## What the TPU actually costs

| operation | measured |
| --- | --- |
| jit dispatch, small actor forward (batch 1 or 10) | 0.094–0.098 ms |
| numpy → TPU → numpy round trip, batch 1 | 0.411 ms |
| jit gradient step, 2x256 MLP, batch 256 | 0.370 ms |
| replay-buffer `add`, no donation (1M-row buffer) | 1.221 ms |
| replay-buffer `add`, `donate_argnums=0` | 0.366 ms |
| flow-model gradient step, batch 1024 | 1.40–1.47 ms |

Two consequences shaped the code:

1. **Dispatch dominates.** A 0.37 ms gradient step is not compute — it is the
   round trip. `train.py` fuses the ten gradient steps between two environment
   interactions into a single `lax.scan`, and `flow_matching.py` fuses a whole
   epoch of minibatches.
2. **Donate the buffers.** Writing one transition into a 1M-row ring copied
   108 MB per online step until `add` started donating its argument. 3.3x.

## float32, not bfloat16

TPUs default to bfloat16 matmuls for float32 inputs. Against the reference
PyTorch backbone, same weights and same input:

| `jax_default_matmul_precision` | max abs error | relative |
| --- | --- | --- |
| `default` / `bfloat16` | 5.19e-3 | 4.07e-3 |
| `highest` / `float32` | 4.02e-7 | 3.15e-7 |

The upgrade costs 4.5% (1.404 → 1.467 ms per flow gradient step at batch 1024),
because these MLPs are dispatch- and memory-bound. `comp_flow_jax/__init__.py`
therefore sets `highest` on import. Override with
`COMP_FLOW_MATMUL_PRECISION=default`.

**What float32 does not buy back:** this TPU evaluates transcendentals with a
hardware approximation. Against numpy over `[-3, 3]`:

| function | TPU | JAX on CPU |
| --- | --- | --- |
| `tanh` | 4.4e-5 | 1.8e-7 |
| `softplus` | 1.0e-4 | — |

So the squashed policy action and its log-density carry ~1e-4 of absolute error
against PyTorch no matter how careful the port is. That is far below SAC's own
gradient noise, but it is why `tests/test_actor_critic_parity.py` gives the
squashed quantities a looser tolerance than the linear ones.

## Optimal transport

`pot.emd` with the reference's hardcoded `numThreads='max'` is pathological on
a 240-core box:

| solver, batch 1024 | time |
| --- | --- |
| `pot.emd(numThreads='max')` — the reference | 5364 ms |
| `pot.emd(numThreads=32)` | 936 ms |
| `pot.emd(numThreads=8)` | 494 ms |
| `pot.emd(numThreads=1)` | 186 ms |
| `scipy.optimize.linear_sum_assignment` | 123 ms |
| **JAX log-domain Sinkhorn on TPU** (reg 0.05, 200 iters) | **2.94 ms** |

Thread thrashing costs 29x. With uniform marginals and `n == m` the EMD optimum
is a permutation, so scipy's Hungarian solver returns the identical coupling
(verified) and is faster still.

**The default is `exact`**, matching the reference. It is expensive: adaptation
training performs roughly 60k minibatch solves over a default run, about two
hours of host-serial work that also stalls the TPU. `--ot_solver sinkhorn`
trades that for an entropic relaxation that runs on-device inside the training
step and lands within ~0.13% of the optimal transport cost — a ~40x saving, at
the price of no longer being the same solver the paper used.

Note that the exact path here uses scipy's Hungarian solver rather than
`pot.emd`. That is not a method change: for uniform marginals with `n == m` the
two return the identical coupling (verified in `tests/`), and scipy is faster
and free of the thread-count trap.

## Legacy XML assets

45 of the 89 files in `envs/mujoco/assets` use `coordinate="global"`, dropped in
MuJoCo 2.3.3 and rejected outright by 3.x. The hopper and walker2d families also
carry unevaluated literals (`pos="0.13/2 0 0.1"`) that mujoco-py's parser
tolerated and nothing since does.

`comp_flow_jax/envs/convert_assets.py` patches the two literals, then loads and
re-saves every model under a throwaway mujoco 2.3.3 venv, which emits local
coordinates. **89/89 convert, and 89/89 load under mujoco 3.10.** Shift
parameters survive intact — hopper foot friction 2.0 → 10.0 at `friction_5.0`,
gravity 9.81 → 49.05 at `gravity_5.0`.

The converted assets are checked in under `comp_flow_jax/envs/assets_v3/`.

## Source datasets

The `d4rl` package needs mujoco-py, gym 0.18 and Python 3.8 and cannot be
installed here. All 24 v2 MuJoCo hdf5 files are mirrored on the HuggingFace
dataset `imone/D4RL`, and `d4rl.qlearning_dataset` is thirty lines, transcribed
verbatim in `data/d4rl_hf.py`. The vectorised fast path there is checked
bit-identical against the transcribed loop (53x faster: 111 ms vs 5879 ms on
hopper-medium-v2).

## Engine-version caveat

The D4RL datasets and the reference scores in `envs/infos.py` were produced with
mujoco-py 2.1. This port runs mujoco 3.10. The locomotion tasks are close across
that gap but not identical, so absolute returns and normalised scores are
comparable to published numbers only up to the engine change.

## Throughput of a real run

Measured on hopper-shaped tensors (state 11, action 3), batch 128 source +
128 target, `dynamics_gap_reward_scale=0.1`, 10 gradient steps per dispatch:

| phase | ms/gradient step | steps/s | 400k steps |
| --- | --- | --- | --- |
| pre-gate (no gap estimate) | 0.408 | 2451 | 2.7 min |
| post-gate, `n_samples=30`, 1 chip | 18.889 | 53 | 126 min |
| post-gate, `n_samples=30`, 4 chips | **6.993** | **143** | **47 min** |
| post-gate, `n_samples=100`, 1 chip | 55.473 | 18 | 370 min |
| post-gate, `n_samples=100`, 4 chips | **16.636** | **60** | **111 min** |

Two optimisations account for the difference from a naive port:

**Fusing the gradient steps.** One dispatch per step costs 3.723 ms; ten steps
in one `lax.scan` cost 0.408 ms each — **9.1x**. The step is not compute-bound
before the gate, it is dispatch-bound.

**Sharding the gap estimate.** (Numerically identical, verified to 3.6e-7.) It is the whole cost of a gated step (18.9 ms of
19.3), it is compute-bound (~522 GFLOP/step at `n_samples=30`), and it is
embarrassingly parallel over rows. Spreading it over all four chips gives
2.70x at `n_samples=30` and 3.33x at `n_samples=100` — the larger batch
amortises the collectives better. Sharded and unsharded gaps agree to 3.6e-7
(`tests/` covers this; an earlier benchmark that appeared to show a 26%
discrepancy was comparing different random inputs).

With the gate at 100k, a default 400k-step run is therefore roughly
0.7 min ungated + 35 min gated at `n_samples=30`, plus flow fitting.

**Compile-shape bucketing.** `n_batches` is a static jit argument, so each
distinct value recompiles a program containing an OT solve and two ODE
rollouts. The adaptation flow is refit ~60 times against a steadily growing
target buffer, which would walk through 28 distinct values. Rounding the batch
count up to a multiple of 4 (and wrapping the shuffled index list to fill the
extra batches, so no rows are dropped) cuts that to 7.
