#!/usr/bin/env bash
# Sequentially train dataset1 then dataset2 with the optimized CRAFT pipeline.
set -euo pipefail

cd "$(dirname "$0")"

EPOCHS="${EPOCHS:-100}"
EARLY_STOP="${EARLY_STOP:-10}"
DATA_DIR="${DATA_DIR:-./data}"
SAVE_DIR="${SAVE_DIR:-./saved_models}"
OUTPUT_DIR="${OUTPUT_DIR:-./data}"

# Optional: disable tqdm spam in logs (set TQDM_DISABLE=0 to keep progress bars)
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
# Force line-buffered / unbuffered Python stdout so epoch logs show under `tee`
export PYTHONUNBUFFERED=1
# Async GPU by default (override with JT_SYNC=1 for debug)
export JT_SYNC="${JT_SYNC:-0}"

# ---- CUDA / AutoDL fixes (CURAND_STATUS_INITIALIZATION_FAILED) ----
# 1) Make sure a /dev/nvidia0 node exists (some containers only expose nvidiaN)
if [[ ! -e /dev/nvidia0 ]]; then
  for d in /dev/nvidia{7,1,2,3,4,5,6}; do
    if [[ -e "$d" ]]; then
      ln -sf "$d" /dev/nvidia0
      echo "[cuda-fix] linked /dev/nvidia0 -> $d"
      break
    fi
  done
fi
# 2) Force CUDA 11.8 runtime/curand ahead of PyTorch nvidia-cu13 wheels
#    (cu13 libcurand.so.10 conflicts with Jittor's CUDA 11.8 and breaks curandCreateGenerator)
CUDA11_LIB="${CUDA11_LIB:-/usr/local/cuda/lib64}"
if [[ -e "${CUDA11_LIB}/libcurand.so" && -e "${CUDA11_LIB}/libcudart.so" ]]; then
  export LD_PRELOAD="${CUDA11_LIB}/libcudart.so:${CUDA11_LIB}/libcurand.so${LD_PRELOAD:+:$LD_PRELOAD}"
  echo "[cuda-fix] LD_PRELOAD=${LD_PRELOAD}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$SAVE_DIR" logs

echo "============================================================"
echo "CRAFT sequential training"
echo "  epochs=${EPOCHS}  early_stop=${EARLY_STOP}  PACK=${PACK:-1}"
echo "  data_dir=${DATA_DIR}  save_dir=${SAVE_DIR}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  JT_SYNC=${JT_SYNC:-0}"
echo "============================================================"

# Quick GPU sanity check before long training
python - <<'PY'
import jittor as jt
jt.flags.use_cuda = 1
x = jt.randn(4)
x.sync()
print('[cuda-fix] jittor GPU OK, randn=', x.numpy())
PY

for ds in dataset1 dataset2; do
  log="logs/${ds}_train_$(date +%Y%m%d_%H%M%S).log"
  echo ""
  echo ">>> START ${ds}  (log: ${log})"
  # python -u: unbuffered stdout/stderr (epoch lines appear immediately in tee log)
  python -u main.py \
    --dataset "$ds" \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --early_stop "$EARLY_STOP" \
    2>&1 | tee "$log"
  echo ">>> DONE ${ds}"
done

echo ""
echo "============================================================"
echo "All datasets finished."
echo "Results:"
echo "  ${OUTPUT_DIR}/dataset1/dataset1_result.csv"
echo "  ${OUTPUT_DIR}/dataset2/dataset2_result.csv"
echo "Models:"
echo "  ${SAVE_DIR}/dataset1_CRAFT_best.pkl"
echo "  ${SAVE_DIR}/dataset2_CRAFT_best.pkl"
echo "============================================================"

# Pack submission zip by default (set PACK=0 to skip)
if [[ "${PACK:-1}" == "1" ]]; then
  cp -f "${OUTPUT_DIR}/dataset1/dataset1_result.csv" ./dataset1.csv
  cp -f "${OUTPUT_DIR}/dataset2/dataset2_result.csv" ./dataset2.csv
  zip -q -j result.zip dataset1.csv dataset2.csv
  echo "Packed ./result.zip  (dataset1.csv + dataset2.csv)"
  ls -lh result.zip
fi
