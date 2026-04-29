#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

OUT="${OUT:-runs/tabular}"
mkdir -p "$OUT"

python -m experiments_v2.run_coupled_reward_chain \
  --nS "${CHAIN_STATES:-25}" \
  --gamma "${GAMMA:-0.99}" \
  --iters "${ITERS:-250}" \
  --outdir "$OUT"

python -m experiments_v2.run_windy_gridworld \
  --width "${WIDTH:-10}" \
  --height "${HEIGHT:-7}" \
  --p_wind "${P_WIND:-0.2}" \
  --gamma "${GAMMA:-0.99}" \
  --iters "${ITERS:-250}" \
  --outdir "$OUT"

python -m experiments_v2.run_gap_windy_gridworld \
  --width "${WIDTH:-10}" \
  --height "${HEIGHT:-7}" \
  --p_wind "${P_WIND:-0.2}" \
  --gamma "${GAMMA:-0.99}" \
  --jipe_iters "${GAP_ITERS:-250}" \
  --n_mc "${N_MC:-500}" \
  --mc_horizon "${HORIZON:-300}" \
  --outdir "$OUT"
