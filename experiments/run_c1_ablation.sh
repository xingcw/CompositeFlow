#!/usr/bin/env bash
# C1 ablation: does gathering the adaptation conditioning with the OT coupling
# indices matter on a real task?
#
#   arm A  --coupling ot            conditioning gathered with the coupling
#   arm B  --coupling ot_uncoupled  conditioning left in place (the reference)
#
# Everything else is the README's headline configuration. The OT solver is held
# fixed across arms so the only difference is the conditioning.
#
# The TPU is single-tenant, so runs are serial by construction.
#
#   SEEDS="0 1 2" MAX_STEP=400000 SOLVER=sinkhorn ./experiments/run_c1_ablation.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-./.venv/bin/python}
OUT=${OUT:-experiments/c1_ablation}
SEEDS=${SEEDS:-"0 1 2"}
ARMS=${ARMS:-"ot ot_uncoupled"}
MAX_STEP=${MAX_STEP:-400000}
SOLVER=${SOLVER:-sinkhorn}
ENV_NAME=${ENV_NAME:-hopper-friction}
SHIFT=${SHIFT:-5.0}
SRCTYPE=${SRCTYPE:-medium-replay}

mkdir -p "$OUT"
echo "[c1] arms='$ARMS' seeds='$SEEDS' steps=$MAX_STEP solver=$SOLVER -> $OUT"

# The TPU is single-tenant: a stray process holding it would kill the whole
# sweep on the next launch. Wait for it instead.
wait_for_tpu() {
  local waited=0
  # Ask the device who holds it. Matching on command lines is not safe here:
  # any shell whose text merely mentions the module -- a log tail, a monitor
  # loop, this script -- would look like a TPU holder.
  while fuser /dev/accel0 >/dev/null 2>&1 && [ "$waited" -lt 1800 ]; do
    echo "[c1] TPU busy, waiting... (${waited}s)"
    sleep 30
    waited=$((waited + 30))
  done
}

for seed in $SEEDS; do
  for arm in $ARMS; do
    tag="${arm}-s${seed}"
    log="$OUT/${tag}.log"
    if [ -f "$OUT/${tag}.done" ]; then
      echo "[c1] $tag already done, skipping"
      continue
    fi
    echo "[c1] === $tag === $(date -Is)"
    wait_for_tpu
    # A failed run must not abort the sweep: record it and carry on.
    if $PY -m comp_flow_jax.train \
      --policy vflow --env "$ENV_NAME" --mode 1 \
      --srctype "$SRCTYPE" --shift_level "$SHIFT" --seed "$seed" \
      --n_samples 30 --dynamics_gap_reward_scale 0.1 --filter_percent 0.8 \
      --max_step "$MAX_STEP" \
      --coupling "$arm" --ot_solver "$SOLVER" \
      --gap_diagnostics \
      --dir "$OUT/$tag" 2>&1 | tee "$log"
    then
      touch "$OUT/${tag}.done"
      echo "[c1] $tag finished $(date -Is)"
    else
      touch "$OUT/${tag}.failed"
      echo "[c1] $tag FAILED $(date -Is)" >&2
    fi
    tail -3 "$log"
  done
done
echo "[c1] all runs complete"
