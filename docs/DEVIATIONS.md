# Every way this port differs from `algo/offline_online/vflow.py`

Grouped by what kind of decision each one was. Anything in group C changes the
method and should be signed off, not assumed.

---

## A. Forced by JAX/XLA — no behavioural change

| # | Reference | Here | Why it is not a change |
| --- | --- | --- | --- |
| A1 | `src_state = src_state[mask]` | fixed-shape batch + 0/1 mask | `sum(m·L)/sum(m)` equals `L[mask].mean()` because the reference's denominator *is* `sum(m)`. Proven for the critic, weighted-critic and BC reductions in `tests/test_masking.py`. |
| A2 | one Python iteration per gradient step | 10 steps fused into a `lax.scan` | same arithmetic; 9.1x fewer dispatches (3.723 → 0.408 ms/step) |
| A3 | `torchdiffeq.odeint(method='euler')` on a linspace | `lax.scan` Euler | same fixed grid, `n` points = `n−1` steps; checked against the closed form in `tests/test_flow.py` |
| A4 | gap estimated on one device | sharded over 4 TPU chips | outputs agree to 3.6e-7; 2.7–3.3x faster |
| A5 | `n_batches = n_train // batch` | rounded up to a multiple of 4, index list wrapped | no rows dropped; a few hundred rows are seen twice per epoch. Cuts XLA recompiles from 28 to 7 (~20 min of compile time). |
| A6 | `torch.std` (unbiased) | `jnp.std(ddof=1)` | jnp defaults to the biased estimator; `ddof=1` restores parity |
| A7 | float32 matmul | `jax_default_matmul_precision="highest"` | TPU defaults to bfloat16; `highest` restores float32 parity (4e-3 → 3e-7 relative) at 4.5% cost |

## B. Bugs, fixed — approved, and all reverted by `--reproduce_original`

| # | Bug | Effect in the reference |
| --- | --- | --- |
| B1 | `sample_location_and_conditional_flow_condtion_version` (typo) | `AttributeError` swallowed by a bare `except`. **The adaptation model never trained**, so every dynamics gap was computed from random weights. |
| B2 | `eta` passed positionally into `replace`, and `get_map_condition_version` called without it | `--eta` was inert; the conditioning term in the OT cost was always 0 |
| B3 | `upsample_src` used `keep_ratio = 1 - filter_percent` while the filter keeps `filter_percent` | at `--filter_percent 0.8` it drew `batch/0.2` source samples instead of `batch/0.8` — 5x too many |
| B4 | `ReplayBuffer.downsample` did not exist | `--downsample_src` raised into a bare `except` and did nothing |
| B5 | flow train/val split drawn through `sample_all`, a **bootstrap resample** | ~37% of rows duplicated; validation shared ~63% of its rows with training, so early stopping was reading memorisation |
| B6 | `try/except/else` inverted at `train.py:487` | printed `[Error] Source eval env not available` on every *successful* dataset load |

## C. Judgement calls — these change the method

| # | What | Status |
| --- | --- | --- |
| **C1** | **Adaptation conditioning gathered with the OT coupling indices.** `sample_map` draws index pairs with replacement, so output row `k` carries `(x0[i_k], x1[j_k])` but the reference conditioned it on `cond[k]`. This port gathers `cond[i_k]`. | **Default `--coupling ot`.** Kept as the principled form, but **measured to make no difference on hopper-friction-5.0**: 400k steps, both arms within each other's spread on score, gap Spearman and filter precision (see docs/COUPLING.md). The synthetic probe that motivated the fix overstated it. `--coupling ot_uncoupled` restores the literal reference and costs nothing here. |
| **C2** | `--coupling identity` exists as an option | **Not the default.** A departure from the paper. It is the `eta → ∞` limit of the paper's own construction and scores far better on a synthetic probe (see `docs/COUPLING.md`), but it is not the paper's method. |
| **C3** | `--ot_solver sinkhorn` exists as an option | **Not the default.** `exact` is. Sinkhorn is ~40x faster and within 0.13% of the optimal transport cost, but it is an entropic relaxation, not the reference's solver. |
| **C4** | exact OT uses scipy's Hungarian solver, not `pot.emd` | **Default.** Not a method change: for uniform marginals with `n == m` the EMD optimum is a permutation and the two return the identical coupling (verified). Also avoids the reference's `numThreads='max'`, which is 29x slower than single-threaded here. |

## D. Environment / data, where the original stack will not install

| # | Reference | Here |
| --- | --- | --- |
| D1 | gym 0.18.3 + mujoco-py 2.1 | gym-free ports of the four `*_v3` env classes on mujoco 3.10, transcribed from gym's source |
| D2 | XMLs with `coordinate="global"` and `pos="0.13/2 ..."` | converted once under mujoco 2.3.3 to local coordinates; 89/89 convert and load, shift parameters intact |
| D3 | `d4rl.qlearning_dataset(env)` | same function transcribed, fed from the `imone/D4RL` HF mirror; vectorised path checked bit-identical |
| D4 | `mj_step` alone | `mj_step` + `mj_rnePostConstraint` — otherwise Ant's `cfrc_ext` is a step stale and `contact_cost` reads as 0. This restores mujoco-py behaviour; gymnasium's Ant-v4 shipped with the unfixed version. |
| D5 | 10 evaluation episodes stepped one at a time | stepped in lockstep, one batched policy call per timestep (~10x fewer dispatches) |
| D6 | gym's seeded RNG chain | `np.random.RandomState(seed)` directly — same distributions, different stream for a given integer seed |
| D7 | `--mode` in the README but not in argparse | accepted, must be 1 |
| D8 | source flow retrained per run | cached in `source_flow_cache/`, keyed by dataset + architecture. This *is* reference behaviour (`{task_name}_source_model_state.pth`) and makes every arm of a sweep share one source model. |

## Not ported

`bc_sac`, `bc_par`, `bc_vgdf`, `h2o`, and the adroit/antmaze domains (which the
reference named but never gave an env factory).
