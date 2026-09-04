# Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data

<p align="center">
  <br />
  <a href="./LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-red.svg" /></a>
  <a href="Python 3.8"><img alt="Python 3.8" src="https://img.shields.io/badge/python-3.8-blue.svg" /></a>
</p>

<!-- ![A brief overview of the Composite Flow Framework.](./imgs/compositeflow.png) -->

🔥 [Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data](https://arxiv.org/abs/2505.23062)

CompositeFlow adapts RL agents to dynamics shift by composing flow matching models for source and target MDPs and guiding exploration with learned dynamics gaps.

## 🛠️ Installation (CompositeFlow)

Clone project and create environment with conda:

```bash
# Create and activate conda environment
conda create -n compositeflow python=3.8
conda activate compositeflow

# Install dependencies
conda env update --file environment.yml --prune
```

## 🚀 Run Training (CompositeFlow)

After setting up the environment, you can launch a single training run with:

```bash
python train.py \
    --policy vflow \
    --env hopper-friction \
    --mode 1 \
    --srctype medium-replay \
    --shift_level 5.0 \
    --seed 0 \
    --n_samples 30 \
    --dynamics_gap_reward_scale 0.1 \
    --filter_percent 0.8 \
    --dir training_output/seed_0
```

## 🧩 Training (Baselines)

To train a single baseline agent (e.g., BC-SAC) on Hopper-Friction:

```bash
python train.py \
    --policy bc_sac \
    --env hopper-friction \
    --mode 1 \
    --srctype medium-replay \
    --shift_level 5.0 \
    --seed 0 \
    --dir training_output/bc_sac_seed_0
```

## Licences

Our repository is licensed under the MIT licence. The adopted ODRL codeabse, Gym environments, and mujoco-py are also licensed under the MIT License. For the D4RL library (including the Antmaze domain and the Adroit domain, and offline datasets), all datasets are licensed under the Creative Commons Attribution 4.0 License (CC BY), and code is licensed under the Apache 2.0 License.

## References
This codebase has been adapted from [ORDL](https://arxiv.org/html/2410.20750v1). 

## 📄Citing CompositeFlow
Please consider citing us if you find our work useful!

```
@inproceedings{kong2025composite,
  title     = {Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data},
  author    = {Kong, Lingkai and Wang, Haichuan and Wang, Tonghan and Xiong, Guojun and Tambe, Milind},
  booktitle = {Proceedings of the 39th Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```

## Reproduction notes (Hopper Morphology, RTX 5090, Sep 2026)

### Environment
The README pin (python 3.8 / torch 2.2) cannot run on Blackwell GPUs (sm_120), so the env used here is
python 3.10 + torch 2.8.0+cu128 with gym 0.18.3, mujoco-py 2.1.2.14 (mujoco210) and D4RL. `source env.sh`
activates it. Notes:
- gym 0.18.3 and mujoco-py need `pip install --no-build-isolation` (gym also needs its `opencv-python>=3.` specifier patched).
- The D4RL server (rail.eecs.berkeley.edu) was unreachable; datasets were fetched from the HuggingFace mirror `imone/D4RL`.
- `COMPFLOW_TF32=1` enables TF32 matmuls in `train.py` (~1.5x faster dynamics-gap estimation).

### Code fixes needed to run the released code
- `train.py`: added the `--mode` flag used in the README commands (argparse rejected it).
- `optimal_transport_old.py`: `OTPlanSampler.get_map` body was over-indented (SyntaxError on import).
- `flow_matching.py`: the adaptation flow called `sample_location_and_conditional_flow_condtion_version` (typo);
  the AttributeError was swallowed by a broad handler in `vflow.py`, so the online flow was never trained.
  Fixed the name and passed `eta` through. `vflow.py` now prints the traceback in those handlers.

### Protocol
Target env `hopper-morph-foot`, shift `hard` (foot 0.4x, matches paper Appendix K.1.3). 400K gradient steps,
target interaction every 10 steps (40K interactions), 100K warmup gate, batch 128, eval every 10K steps;
return reported at 400K. CompFlow: beta (`--dynamics_gap_reward_scale`) = 0.1, xi (`--filter_percent`) = 0.5,
`--n_samples 30`. Launcher: `experiments/run_hopper_morph.sh`; results: `experiments/collect_results.py`.

### Results (return at 400K steps)
| Setting | Reproduced | Paper Table 1 |
|---|---|---|
| BC-SAC, medium-replay (3 seeds) | 343.8 +/- 5.8 (352 / 339 / 341) | 346 +/- 4 |
| CompFlow, medium (1 seed) | 343 (last 12 evals: 282-366, mean ~323) | 604 +/- 173 |

Runs stopped early to free the GPU (last eval, not 400K): CompFlow medium-replay 340 at 150K (paper 355 +/- 6);
CompFlow medium-expert 481 at 130K (paper 462 +/- 89).

BC-SAC matches the paper. CompFlow on medium did not: it converged to the same ~340 plateau as BC-SAC.
Caveats: single seed on a row whose paper std is 173; the paper selects beta in {0.01, 0.1, 0.2} and xi in
{30%, 50%} per task without reporting the chosen pair; the released config uses flow width 512 while paper
Table 4 lists 256; `--eta` is not read by the released flow code (OT cost has no condition term).

### Cost
After the 100K gate every step runs the gap estimate (256 pairs x 30 samples through two 10-step Euler flows,
~34 ms with TF32 on this GPU), which saturates the GPU; one CompFlow run alone takes ~4 h, and concurrent runs
do not add throughput. BC-SAC takes ~2 h per run.
