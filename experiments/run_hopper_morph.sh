#!/usr/bin/env bash
# Reproduce the paper's Hopper (Morphology) rows of Table 1:
#   target env  : hopper-morph-foot, shift_level=hard (foot 0.4x, matches Appendix K.1.3)
#   protocol    : 400K gradient steps, target interaction every 10 steps (=40K env interactions), eval every 10K
#   hyperparams : beta = --dynamics_gap_reward_scale (paper sweep {0.01,0.1,0.2}),
#                 xi   = --filter_percent (fraction of offline batch kept; paper sweep {0.3,0.5}),
#                 n_samples=30 (Appendix M), warmup/gate=100K (Table 3), batch 128 (config/mujoco/vflow/hopper.yaml)
# Each run gets its own working directory so the cached source-flow checkpoint
# (<task>_source_model_state.pth, written to CWD by algo/.../flow_matching.py) is per-run.
#
# Usage (run inside tmux so the live output is visible):
#   tmux new -s <name> "experiments/run_hopper_morph.sh <jobs_in_parallel> [policy] [seeds] [betas] [xis] [srctypes]"
set -u
JOBS=${1:-6}
POLICY=${2:-vflow}
SEEDS=${3:-"0 1 2"}
BETAS=${4:-"0.1"}
XIS=${5:-"0.5"}
SRCTYPES=${6:-"medium-replay medium medium-expert"}
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$REPO/runs/hopper_morph}
mkdir -p "$OUT"

source "$REPO/env.sh"
export PYTHONUNBUFFERED=1  # keep tee/tmux output live

jobfile=$(mktemp)
for src in $SRCTYPES; do for beta in $BETAS; do for xi in $XIS; do for seed in $SEEDS; do
  if [ "$POLICY" = "vflow" ]; then
    tag="vflow_${src}_beta${beta}_xi${xi}_s${seed}"
    extra="--n_samples 30 --dynamics_gap_reward_scale $beta --filter_percent $xi"
  else
    tag="${POLICY}_${src}_s${seed}"
    extra=""
  fi
  d="$OUT/$tag"
  if [ -f "$d/DONE" ]; then echo "skip $tag (done)"; continue; fi
  mkdir -p "$d"
  echo "$d|$src|$seed|$extra" >> "$jobfile"
done; done; done; done

run_one() {
  IFS='|' read -r d src seed extra <<< "$1"
  cd "$d" || exit 1
  echo "[$(date +%T)] start $(basename "$d")"
  python "$REPO/train.py" --policy "$POLICY" --env hopper-morph-foot --mode 1 --srctype "$src" \
      --shift_level hard --seed "$seed" --max_step 400000 --eval_freq 10000 \
      --tar_env_interact_interval 10 $extra --dir . 2>&1 | tee train.log \
    && [ "${PIPESTATUS[0]}" = "0" ] && touch DONE
  echo "[$(date +%T)] end   $(basename "$d") exit=$?"
}
export -f run_one; export REPO POLICY
xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {} < "$jobfile"
rm -f "$jobfile"
