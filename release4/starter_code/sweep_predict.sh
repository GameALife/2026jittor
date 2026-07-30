#!/usr/bin/env bash
# Sweep blend_alpha / step_scale for CD–P2S tradeoff. Packs result_aXX_sYY.zip each.
#
# Usage:
#   cd release4/starter_code
#   ./sweep_predict.sh
#   BLEND_ALPHAS="0.5 0.65 0.8" STEP_SCALES="0.5 0.6" ./sweep_predict.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source ./setup_cuda_env.sh

BLEND_ALPHAS="${BLEND_ALPHAS:-0.45 0.55 0.65 0.75 0.85}"
STEP_SCALES="${STEP_SCALES:-0.6}"
BEST_CKPT="${BEST_CKPT_PATH:-experiments/vm_v2/checkpoint_best.pkl}"

for a in ${BLEND_ALPHAS}; do
  for s in ${STEP_SCALES}; do
    tag="a${a}_s${s}"
    echo "======== sweep ${tag} ========"
    PREDICT_BLEND_ALPHA="${a}" \
    PREDICT_STEP_SCALE="${s}" \
    PREDICT_OUTER_STEPS="${PREDICT_OUTER_STEPS:-1}" \
    PREDICT_INNER_STEPS="${PREDICT_INNER_STEPS:-1}" \
    SKIP_TRAIN=1 USE_ENSEMBLE=0 \
    RESULT_DIR="results/sweep_${tag}" \
    OUT_ZIP="result_${tag}.zip" \
    BEST_CKPT_PATH="${BEST_CKPT}" \
    ./run_all.sh
  done
done

echo "Done. Zips:"
ls -lh result_a*.zip 2>/dev/null || true
