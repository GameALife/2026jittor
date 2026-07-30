#!/usr/bin/env bash
# Source this before running run.py manually:
#   source ./setup_cuda_env.sh
#   python run.py --task configs/task/train_vm_v2.yaml

# Fix missing /dev/nvidia0 on some AutoDL nodes
if [[ ! -e /dev/nvidia0 ]]; then
  for d in /dev/nvidia{7,1,2,3,4,5,6}; do
    if [[ -e "$d" ]]; then
      ln -sf "$d" /dev/nvidia0
      echo "[cuda-fix] linked /dev/nvidia0 -> $d"
      break
    fi
  done
fi

CUDA11_LIB="${CUDA11_LIB:-/usr/local/cuda/lib64}"
if [[ -e "${CUDA11_LIB}/libcurand.so" && -e "${CUDA11_LIB}/libcudart.so" ]]; then
  export LD_PRELOAD="${CUDA11_LIB}/libcudart.so:${CUDA11_LIB}/libcurand.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JT_SYNC="${JT_SYNC:-0}"

echo "[cuda-fix] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[cuda-fix] LD_PRELOAD=${LD_PRELOAD:-}"
echo "[cuda-fix] JT_SYNC=${JT_SYNC}"
