#!/usr/bin/env bash
# Train + predict dataset2 only, then pack result.zip (keeps existing dataset1.csv).
set -euo pipefail
cd "$(dirname "$0")"

EPOCHS="${EPOCHS:-100}"
EARLY_STOP="${EARLY_STOP:-10}"
DATA_DIR="${DATA_DIR:-./data}"
SAVE_DIR="${SAVE_DIR:-./saved_models}"
OUTPUT_DIR="${OUTPUT_DIR:-./data}"

export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export PYTHONUNBUFFERED=1
export JT_SYNC="${JT_SYNC:-0}"

source ./setup_cuda_env.sh

mkdir -p "$SAVE_DIR" logs
log="logs/dataset2_retrain_$(date +%Y%m%d_%H%M%S).log"
echo ">>> START dataset2-only retrain  (log: ${log})"
python -u main.py \
  --dataset dataset2 \
  --data_dir "$DATA_DIR" \
  --save_dir "$SAVE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --early_stop "$EARLY_STOP" \
  2>&1 | tee "$log"

# Prefer ensemble predict if predict.py exists with tuned infer
if [[ -f predict.py ]]; then
  echo ">>> predict.py dataset2 (ensemble / tuned infer)"
  python -u predict.py --dataset dataset2 2>&1 | tee -a "$log" || true
fi

# Pack: keep dataset1 from previous good run if present
if [[ -f "${OUTPUT_DIR}/dataset1/dataset1_result.csv" ]]; then
  cp -f "${OUTPUT_DIR}/dataset1/dataset1_result.csv" ./dataset1.csv
fi
cp -f "${OUTPUT_DIR}/dataset2/dataset2_result.csv" ./dataset2.csv
zip -q -j result.zip dataset1.csv dataset2.csv
echo "Packed ./result.zip"
ls -lh result.zip dataset1.csv dataset2.csv 2>/dev/null || true
echo ">>> DONE dataset2-only"
