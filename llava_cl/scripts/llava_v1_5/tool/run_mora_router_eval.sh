#!/bin/bash
set -o pipefail

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=False
export PYTHONPATH=/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM:$PYTHONPATH
export $(cat .env | xargs -d '\r\n')

# ===== 可选参数 =====
# 第一个参数：最大训练样本数/任务（可选）
# 第二个参数：最大测试样本数/任务（可选）
max_train_per_task=${1:-0}   # 0 表示不限制
max_test_per_task=${2:-0}    # 0 表示不限制

OUTPUT_ROOT="/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/without_reasoning"
TRAIN_DATA_DIR="/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/train"
TASKS="numglue,art,math,fomc,medical,OCR,science"

mkdir -p "${OUTPUT_ROOT}"

echo "[INFO] Running MoRA-style router evaluation..."
echo "[INFO] OUTPUT_ROOT = ${OUTPUT_ROOT}"

python -u tools/run_mora_router_eval.py \
  --train_data_dir "${TRAIN_DATA_DIR}" \
  --output_root "${OUTPUT_ROOT}" \
  --tasks "${TASKS}" \
  $( [[ "${max_train_per_task}" -gt 0 ]] && echo "--max_train_per_task ${max_train_per_task}" ) \
  $( [[ "${max_test_per_task}" -gt 0 ]] && echo "--max_test_per_task ${max_test_per_task}" ) \
  --num_ranks 32 \
  --epochs 30 \
  --lr 1e-3 \
  --batch_size 128 \
  --seed 42 \
  --device cuda \
  2>&1 | tee "${OUTPUT_ROOT}/mora_router_latest.log"
