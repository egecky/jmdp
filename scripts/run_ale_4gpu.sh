#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

OUT_ROOT="${OUT_ROOT:-runs/ale}"
STEPS="${STEPS:-50000}"
N_STATES="${N_STATES:-40}"
N_MC="${N_MC:-32}"
HORIZON="${HORIZON:-150}"
SIGMA_MODE="${SIGMA_MODE:-covgram}"
PAIR_MODE="${PAIR_MODE:-random}"
COUPLING_MODE="${COUPLING_MODE:-reward}"
P_SHARED_STICKY="${P_SHARED_STICKY:-0}"
REWARD_BONUS="${REWARD_BONUS:-1.0}"
GAP_WEIGHT="${GAP_WEIGHT:-0.1}"
SEED="${SEED:-0}"

mkdir -p logs

run_ale() {
  local gpu="$1"
  local env_id="$2"
  local safe_env="${env_id//\//_}"
  safe_env="${safe_env//-/_}"
  local outdir="${OUT_ROOT}/${safe_env}/seed_${SEED}/${SIGMA_MODE}"

  mkdir -p "$outdir"

  CUDA_VISIBLE_DEVICES="$gpu" python -u -m deep_learning.joint_eval.train_td_jipe2_atari \
    --env_id "$env_id" \
    --seed "$SEED" \
    --mode "$COUPLING_MODE" \
    --p_shared_sticky "$P_SHARED_STICKY" \
    --reward_bonus "$REWARD_BONUS" \
    --gap_weight "$GAP_WEIGHT" \
    --policy random \
    --gamma 0.99 \
    --sigma_mode "$SIGMA_MODE" \
    --steps "$STEPS" \
    --outdir "$outdir" \
    --log_every 1000

  CUDA_VISIBLE_DEVICES="$gpu" python -u -m deep_learning.joint_eval.eval_gap_atari \
    --env_id "$env_id" \
    --seed "$SEED" \
    --model_path "$outdir/atari_jipe2/weights.pt" \
    --sigma_mode "$SIGMA_MODE" \
    --outdir "$outdir/gap_eval" \
    --n_states "$N_STATES" \
    --n_mc "$N_MC" \
    --horizon "$HORIZON" \
    --pair_mode "$PAIR_MODE" \
    --coupling_mode "$COUPLING_MODE" \
    --mc_semantics recursive_pair
}

run_ale 0 "ALE/Pong-v5" > logs/ale_pong.log 2>&1 &
run_ale 1 "ALE/BattleZone-v5" > logs/ale_battlezone.log 2>&1 &
run_ale 2 "ALE/Boxing-v5" > logs/ale_boxing.log 2>&1 &
run_ale 3 "ALE/Atlantis-v5" > logs/ale_atlantis.log 2>&1 &

wait
