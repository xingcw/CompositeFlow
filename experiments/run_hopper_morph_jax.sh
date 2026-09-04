#!/usr/bin/env bash
# JAX/TPU counterpart of experiments/run_hopper_morph.sh, same protocol:
#   hopper-morph-foot, shift_level=hard, 400K steps, interact every 10, eval every 10K
#   beta=--dynamics_gap_reward_scale, xi=--filter_percent, n_samples=30
# Runs serially: the TPU is single-tenant.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$REPO/.venv/bin/python}
OUT=${OUT:-$REPO/runs/hopper_morph_jax}
SEEDS=${SEEDS:-"0"}
SRCTYPES=${SRCTYPES:-"medium-replay"}
BETA=${BETA:-0.1}
XI=${XI:-0.5}
SOLVER=${SOLVER:-sinkhorn}
mkdir -p "$OUT"

for src in $SRCTYPES; do for seed in $SEEDS; do
  tag="vflow_${src}_beta${BETA}_xi${XI}_s${seed}"
  [ -f "$OUT/$tag/DONE" ] && { echo "skip $tag"; continue; }
  mkdir -p "$OUT/$tag"
  echo "[$(date +%T)] start $tag"
  $PY -m comp_flow_jax.train --policy vflow --env hopper-morph-foot --mode 1 \
      --srctype "$src" --shift_level hard --seed "$seed" \
      --max_step 400000 --eval_freq 10000 --tar_env_interact_interval 10 \
      --n_samples 30 --dynamics_gap_reward_scale "$BETA" --filter_percent "$XI" \
      --ot_solver "$SOLVER" --gap_diagnostics \
      --dir "$OUT/$tag" 2>&1 | tee "$OUT/$tag/train.log"
  [ "${PIPESTATUS[0]}" = "0" ] && touch "$OUT/$tag/DONE"
  echo "[$(date +%T)] end   $tag"
done; done
echo "[jax] all runs complete"
