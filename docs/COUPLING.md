# The adaptation coupling

**Default: `ot` — the paper's method.** `identity` measures better on the probe
below but is a departure from the paper, so it is opt-in, not the default.

## What the reference did

`train_adaptation_flow_matching` builds each minibatch like this (reference
line numbering, `guided_flow/flow_matching.py:747`):

```python
t, ns_t, ut = self.CFM_adaptation.sample_location_and_conditional_flow_condtion_version(
    ns_src_pred_norm_adapt, ns_b_norm_adapt,
    s_b_norm_adapt, s_b_norm_adapt, a_b_norm_adapt, a_b_norm_adapt, t)
cond_adapt = cond_src
vt = self.adaptation_model(ns_t, t, cond=cond_adapt)
```

Three problems compound here.

**1. It never ran.** The method is spelled `..._condtion_version`, missing an
`i`. `ExactOptimalTransportConditionalFlowMatcher` only defines
`..._condition_version`. Every adaptation minibatch therefore raised
`AttributeError`, which `vflow.py::train_adaptation_flow` caught in a bare
`except Exception` and reduced to one `[Error]` line. **The adaptation model
stayed at its random initialisation for the whole run**, and so did every
dynamics gap computed from it — the paper's central mechanism.

**2. `eta` never reached the cost.** One level down,
`sample_location_and_conditional_flow_condition_version` passes `eta`
positionally into `OTPlanSampler.sample_plan_condition_version`, whose seventh
parameter is `replace`. That method then calls `get_map_condition_version`
without an `eta` at all, so the conditioning term in the transport cost was
always zero and `--eta` was inert.

**3. The conditioning does not follow the coupling.** `sample_map` draws index
pairs `(i_k, j_k)` from the OT plan *with replacement*, so output row `k`
carries `(x0[i_k], x1[j_k])` — but `cond_adapt` is left in its original order,
so row `k` is conditioned on `cond[k]`. The velocity field is trained on
mismatched (path, condition) pairs. Note that raising `eta` does **not** fix
this: even a perfectly diagonal plan still emits rows in resampled order.

## What that costs, measured

A controlled setup where the answer is known: 4-dimensional state, 2-dimensional
action, source dynamics `s' = s + 0.5a`, target `s' = s + 1.5a`, observation
noise 0.02. The dynamics gap should grow with `|a|`, and be near zero when the
action is near zero. Source flow trained once (val MSE 0.00079, against a 0.0004
noise floor), then each coupling trained its own adaptation flow from it.

| coupling | eta | adaptation val MSE | gap(\|a\|≈0.05) | gap(\|a\|≈0.58) | ratio |
| --- | --- | --- | --- | --- | --- |
| `identity` | 0.0 | **0.0008** | 0.074 | 0.783 | **10.58x** |
| `ot` | 0.0 | 0.0714 | 0.290 | 0.530 | 1.82x |
| `ot` | 1.0 | **0.0008** | 0.074 | 0.783 | **10.59x** |
| `ot_uncoupled` | 0.0 | 0.1585 | 0.369 | 0.372 | 1.01x |
| `ot_uncoupled` | 1.0 | 0.1582 | 0.360 | 0.364 | 1.01x |

`ot_uncoupled` is the reference's formulation. On this probe its gap ratio is
**1.01x**.

**That result does not transfer to the benchmark.** See the next section: on
hopper-friction-5.0 the two couplings are indistinguishable. The probe is
misleading because its shift lives in the *action* coefficient
(0.5a -> 1.5a), and the adaptation flow's starting point, x0 = s + 0.5a, barely
carries the action when s dominates. A conditioning-blind model therefore
cannot recover the signal there. On the MuJoCo task the shift correlates with
the state region, x0 is the predicted next state, and the model reads the
signal straight off its own input -- the conditioning vector is not needed.

## What the benchmark says

400k steps on hopper-friction-5.0, seed 0, both arms sharing one cached source
model and differing only in `--coupling`. 62 post-gate measurements each,
scored against the simulator ground truth.

| | score (post-gate mean) | gap Spearman | filter precision |
| --- | --- | --- | --- |
| untrained adaptation model (t = 100k control) | -- | +0.06 | 0.18 |
| `ot` (conditioning gathered) | 7.99 +/- 0.36 | +0.125 +/- 0.076 | 0.359 +/- 0.045 |
| `ot_uncoupled` (reference) | 7.86 +/- 0.10 | +0.152 +/- 0.042 | 0.386 +/- 0.041 |
| chance | -- | 0 | 0.199 |

Three things follow.

1. **The gap mechanism works, modestly.** Both arms lift filter precision from
   the 0.18 of an untrained adaptation model to ~0.37, against a 0.199 chance
   level. Roughly 1.9x chance. Spearman ~0.14 with Pearson ~0.39 says the
   signal sits in the tail: the filter reliably finds the worst samples and is
   near-random about the middle of the ranking.
2. **C1 makes no measurable difference here.** The arms are within each other's
   spread on every metric, and `ot_uncoupled` is marginally ahead on both gap
   metrics. One seed cannot resolve small score differences, but it does not
   take seed variance to see that these two are not far apart.
3. **The gap machinery buys no visible return.** Both arms plateau near 7.7 by
   50k, before the gate opens at 100k, and end at 7.9. Whatever the filtering,
   weighting and reward shaping are doing, it does not show up in the score on
   this task.

## Why `identity` scores so well on the probe

OT-CFM couples two *marginals* to straighten the flow. Here the two
distributions are *conditional* on `(s, a)`: `x0` is the source model's
predictive distribution at that `(s, a)`, and `x1` is the target's next state at
that same `(s, a)`. Coupling across a minibatch mixes different conditions,
which is simply the wrong object for a conditional flow.

The correct object is conditional OT — couple only within a condition. With one
sample per condition, that **is** the identity pairing. The `eta` term is a soft
approximation of the same idea: because `s0 == s1` and `a0 == a1` (the reference
passes the same batch twice), the conditioning cost vanishes on the diagonal, so
`eta → ∞` drives the plan to the identity. The table confirms it: `ot` at
`eta=1.0` reproduces `identity` to three decimal places, at four times the wall
time.

So `identity` is the `eta → ∞` limit of the paper's own construction, and it is
also the fastest (13 s vs 53 s for the adaptation fit above). That is an
argument for it, not a licence to substitute it: it changes the method, so it
stays behind a flag.

**What this does mean for the default:** at the reference's `--eta 0.05` the
conditioning term is weak, and the probe says `ot` at low `eta` recovers only
1.82x of the 10.6x available. If you run the paper's method, `--eta` is the knob
that matters most.

## The flags

```
--coupling ot              # DEFAULT. The paper's method: OT plan, with the
                           # conditioning gathered by the same indices so the
                           # velocity field sees the (s, a) that produced its
                           # path. Sensitive to --eta.
--coupling ot_uncoupled    # the reference's formulation, conditioning left in
                           # place. Selected by --reproduce_original. Present
                           # for comparison; the probe says its gap is noise.
--coupling identity        # a deviation from the paper: pair each source
                           # prediction with its own target next state.
--ot_solver exact|sinkhorn # DEFAULT exact (the reference's solver). See
                           # docs/BACKEND.md for what each costs.
--eta FLOAT                # weight of the (s, a) term in the transport cost;
                           # default 0.05, the reference's argparse default.
                           # Unlike the reference, it now reaches the cost.
```

`--reproduce_original` sets `coupling=ot_uncoupled` along with the other
reference-behaviour switches.

## Measuring it on the real benchmark

The synthetic probe above is suggestive, not conclusive: it is a linear system
with a hand-made shift. On the MuJoCo tasks the true dynamics gap is directly
computable, because MuJoCo is deterministic and the two domains are the same
robot with a modified XML:

    true_gap(s, a) = || f_target(s, a) - f_source(s, a) ||

Reconstruct ``(qpos, qvel)`` from an observation (``envs/ground_truth.py``),
step both simulators from it with the same action, subtract. Simulating the
source rather than reading D4RL's ``next_observations`` matters: the D4RL files
were made with mujoco-py 2.1, so their next states carry the 2.1-vs-3.10 engine
difference on top of the dynamics shift. The reconstruction is exact (obs
round-trip error 0.0) and source-vs-source gives identically zero.

`--gap_diagnostics` scores the estimated gap against that ground truth on a
fixed probe of source ``(s, a)`` rows at every evaluation:

* **`gap_spearman`** — rank correlation between estimated and true gap. Rank is
  what matters: the filter takes a quantile.
* **`gap_filter_precision`** — of the rows the filter actually drops, the
  fraction that genuinely belong to the largest-gap tail. Chance level is
  printed alongside (`gap_filter_chance` = `1 - filter_percent`).
* `gap_pearson`, `gap_mean`, `gap_std`.

Note on what was *not* used: an earlier version of this diagnostic measured
split-half reliability — estimate the gap twice with independent noise and
correlate. That is worthless here. It came back at r = +1.0000 for a *randomly
initialised* adaptation model, because a random network is still a
deterministic function of ``(s, a)`` and the Monte-Carlo noise averages out.
Reliability is not validity; only the simulator ground truth separates them.

## Open question

Whether `ot` should gather the conditioning with the coupling indices at all is
a judgement call about what the paper intends, not a transcription question.
This port treats the ungathered version as a wiring bug of the same class as
the method-name typo, and fixes it. If that reading is wrong, `--coupling
ot_uncoupled` is the literal reference behaviour.
