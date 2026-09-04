# Source this before running: `source env.sh`
# conda env: python 3.10 + torch 2.8.0/cu128 (RTX 5090 needs sm_120 support, so the README's py3.8/torch 2.2 pin cannot be used)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate compflow
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export D4RL_SUPPRESS_IMPORT_ERROR=1
export WANDB_MODE=${WANDB_MODE:-disabled}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
