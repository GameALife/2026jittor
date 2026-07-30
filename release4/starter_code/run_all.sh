#!/usr/bin/env bash
# One-shot: train → predict → pack result.zip
#
# Usage:
#   cd release4/starter_code
#   chmod +x run_all.sh setup_cuda_env.sh pick_ensemble.py
#   ./run_all.sh
#
# Optional env:
#   SKIP_TRAIN=1              # only predict + pack (needs existing best ckpt)
#   SKIP_PREDICT=1            # only train
#   SKIP_PACK=1               # skip zip
#   USE_ENSEMBLE=1            # average best±radius epoch ckpts (slower, higher score)
#   ENSEMBLE_RADIUS=2         # best±2 epochs
#   TTA_RUNS=1                # stochastic seed TTA averages
#   PREDICT_BLEND_ALPHA=0.65  # out=noisy+α(pred-noisy); lower → better CD
#   PREDICT_STEP_SCALE=0.6
#   PREDICT_OUTER_STEPS=1
#   PREDICT_INNER_STEPS=1
#   TRAIN_TASK=configs/task/train_vm_v2.yaml
#   PREDICT_TASK=configs/task/predict_vm_v2.yaml
#   BEST_CKPT_PATH=experiments/vm_v2/checkpoint_best.pkl
#   RESULT_DIR=results/dataset_test_noisy
#   OUT_ZIP=result.zip
#   CUDA_VISIBLE_DEVICES=0

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

TRAIN_TASK="${TRAIN_TASK:-configs/task/train_vm_v2.yaml}"
PREDICT_TASK="${PREDICT_TASK:-configs/task/predict_vm_v2.yaml}"
RESULT_DIR="${RESULT_DIR:-results/dataset_test_noisy}"
OUT_ZIP="${OUT_ZIP:-result.zip}"
LOG_DIR="${LOG_DIR:-logs}"
USE_ENSEMBLE="${USE_ENSEMBLE:-0}"
ENSEMBLE_RADIUS="${ENSEMBLE_RADIUS:-2}"
TTA_RUNS="${TTA_RUNS:-1}"
# Export so VelocityModule picks them up (optional overrides)
export TTA_RUNS
[[ -n "${PREDICT_BLEND_ALPHA:-}" ]] && export PREDICT_BLEND_ALPHA
[[ -n "${PREDICT_STEP_SCALE:-}" ]] && export PREDICT_STEP_SCALE
[[ -n "${PREDICT_OUTER_STEPS:-}" ]] && export PREDICT_OUTER_STEPS
[[ -n "${PREDICT_INNER_STEPS:-}" ]] && export PREDICT_INNER_STEPS
[[ -n "${PREDICT_SEED_K:-}" ]] && export PREDICT_SEED_K
mkdir -p "$LOG_DIR"

ts() { date +%Y%m%d_%H%M%S; }

echo "============================================================"
echo "Point-cloud denoising pipeline (deliverable)"
echo "  TRAIN_TASK     = ${TRAIN_TASK}"
echo "  PREDICT_TASK   = ${PREDICT_TASK}"
echo "  RESULT_DIR     = ${RESULT_DIR}"
echo "  OUT_ZIP        = ${OUT_ZIP}"
echo "  USE_ENSEMBLE   = ${USE_ENSEMBLE}"
echo "  BLEND_ALPHA    = ${PREDICT_BLEND_ALPHA:-"(yaml default)"}"
echo "  STEP_SCALE     = ${PREDICT_STEP_SCALE:-"(yaml default)"}"
echo "  OUTER/INNER    = ${PREDICT_OUTER_STEPS:-yaml}/${PREDICT_INNER_STEPS:-yaml}"
echo "============================================================"

# ---- CUDA / CURAND fix ----
# shellcheck disable=SC1091
source ./setup_cuda_env.sh

# ---- data checks ----
TRAIN_ROOT="${TRAIN_ROOT:-/root/autodl-tmp/dataset_train}"
TEST_ROOT="${TEST_ROOT:-/root/workspace/release4/dataset_test_noisy}"
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  if [[ ! -d "${TRAIN_ROOT}/shapenet" ]]; then
    echo "[error] train data missing: ${TRAIN_ROOT}/shapenet"
    echo "  Extract first:"
    echo "    tar xzf /root/workspace/release4/dataset_train.tar.gz -C /root/autodl-tmp"
    exit 1
  fi
fi
if [[ "${SKIP_PREDICT:-0}" != "1" ]]; then
  if [[ ! -d "${TEST_ROOT}/shapenet" ]]; then
    echo "[error] test data missing: ${TEST_ROOT}/shapenet"
    exit 1
  fi
fi

# ---- resolve best ckpt ----
if [[ -n "${BEST_CKPT_PATH:-}" ]]; then
  BEST_CKPT="${BEST_CKPT_PATH}"
elif [[ "${TRAIN_TASK}" == *train_vm_v2* ]]; then
  BEST_CKPT="experiments/vm_v2/checkpoint_best.pkl"
else
  BEST_CKPT="experiments/vm/checkpoint_best.pkl"
fi

CKPT_DIR="$(dirname "${BEST_CKPT}")"

# ---- train ----
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  TRAIN_LOG="${LOG_DIR}/train_$(ts).log"
  echo
  echo "[1/3] Training → ${TRAIN_LOG}"
  python run.py --task "${TRAIN_TASK}" 2>&1 | tee "${TRAIN_LOG}"
  echo "[1/3] Training done."
else
  echo
  echo "[1/3] SKIP_TRAIN=1, skip training."
fi

# ---- predict ----
if [[ "${SKIP_PREDICT:-0}" != "1" ]]; then
  if [[ ! -f "${BEST_CKPT}" ]]; then
    echo "[error] best checkpoint not found: ${BEST_CKPT}"
    echo "  Train first, or set BEST_CKPT_PATH=..."
    exit 1
  fi
  echo "[info] using checkpoint: ${BEST_CKPT}"

  PREDICT_LOG="${LOG_DIR}/predict_$(ts).log"
  echo
  echo "[2/3] Predicting → ${PREDICT_LOG}"
  rm -rf "${RESULT_DIR}"

  TMP_TASK="${LOG_DIR}/predict_task_$(ts).yaml"
  # Pin ckpt + writer save_dir to RESULT_DIR (needed for sweep packs)
  sed \
    -e "s|^load_ckpt:.*|load_ckpt: ${BEST_CKPT}|" \
    -e "s|^[[:space:]]*save_dir:.*|  save_dir: ${RESULT_DIR}|" \
    "${PREDICT_TASK}" > "${TMP_TASK}"
  echo "[info] predict task: ${TMP_TASK}"
  echo "[info] writer save_dir → ${RESULT_DIR}"

  export TTA_RUNS
  if [[ "${USE_ENSEMBLE}" == "1" ]]; then
    ENSEMBLE_CKPTS="$(python pick_ensemble.py --ckpt-dir "${CKPT_DIR}" --radius "${ENSEMBLE_RADIUS}")"
    export ENSEMBLE_CKPTS
    N_ENS=$(echo "${ENSEMBLE_CKPTS}" | tr ',' '\n' | grep -c . || true)
    echo "[info] ensemble (${N_ENS}): ${ENSEMBLE_CKPTS}"
  else
    unset ENSEMBLE_CKPTS || true
    echo "[info] ensemble disabled"
  fi

  python run.py --task "${TMP_TASK}" 2>&1 | tee "${PREDICT_LOG}"
  N_OUT=$(find "${RESULT_DIR}" -name 'denoised.npy' 2>/dev/null | wc -l | tr -d ' ')
  echo "[2/3] Predicting done. denoised.npy count=${N_OUT}"
  if [[ "${N_OUT}" -le 0 ]]; then
    echo "[error] no denoised.npy written under ${RESULT_DIR}"
    exit 1
  fi
else
  echo
  echo "[2/3] SKIP_PREDICT=1, skip predict."
fi

# ---- pack ----
if [[ "${SKIP_PACK:-0}" != "1" ]]; then
  echo
  echo "[3/3] Packing ${OUT_ZIP}"
  if [[ ! -d "${RESULT_DIR}/shapenet" ]]; then
    echo "[error] missing ${RESULT_DIR}/shapenet — cannot pack"
    exit 1
  fi
  ZIP_ABS="${ROOT}/${OUT_ZIP}"
  rm -f "${ZIP_ABS}"
  ( cd "${RESULT_DIR}" && zip -qr "${ZIP_ABS}" shapenet/ )
  ls -lh "${ZIP_ABS}"
  echo "[3/3] Pack done → ${ZIP_ABS}"
else
  echo
  echo "[3/3] SKIP_PACK=1, skip zip."
fi

echo
echo "============================================================"
echo "ALL DONE"
echo "  zip : ${ROOT}/${OUT_ZIP}"
echo "  ckpt: ${BEST_CKPT}"
echo "============================================================"
