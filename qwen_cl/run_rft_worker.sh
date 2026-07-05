#!/usr/bin/env bash
# run_rft_worker.sh — 在 8 卡 worker 上跑纯 RFT（GRPO-only）顺序 CL，记录耗时。
#
# 7 个任务顺序：numglue → art → math → fomc → medical → OCR → science
# 每个任务：GRPO LoRA finetune → merge LoRA 到基座 → 下一任务
# 对应论文 Table VIII 的「RFT」行（参考 MLLM-CTBench paper）。
#
# 用法（作为 volc 启动命令直接挂上来）：
#   volc ml_devinstance ml_launch \
#     --resource_queue_id q-20260129141046-tmbpd \
#     --flavor_id ml.pni2l.28xlarge \
#     bash /mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/qwen_cl/run_rft_worker.sh
#
# 关键产出：
#   - /mnt/cxzx/.../TMM/qwen_cl/timing_rft.json        ← 总耗时 + 启停时间
#   - /mnt/cxzx/.../TMM/qwen_cl/grpo_cl_output/seqft/Qwen2.5-VL-3B_grpo_r64/
#     debug_Qwen2.5-VL-3B_grpo_r64.txt                 ← 详细日志 + 每任务时间戳
############################################
# DEBUG: 立刻写一份诊断日志到共享盘，看 worker 启动早期挂在哪里
############################################
DIAG=/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/qwen_cl/_worker_diag_rft.log
{
  echo "=== diag begin @ $(date) ==="
  echo "hostname=$(hostname)  whoami=$(whoami)  pwd=$(pwd)"
  echo "--- /mnt/cxzx visible ---"
  ls /mnt/cxzx/ 2>&1 | head
  echo "--- nvidia-smi ---"
  nvidia-smi --query-gpu=index,name --format=csv,noheader 2>&1 | head
  echo "--- conda binary ---"
  ls /mnt/cxzx/workspace/data_transfer/houzhiyan/miniconda3/envs/qwen_cl/bin/python 2>&1
} > "$DIAG" 2>&1

# 关键步骤跟踪也写到 DIAG，stderr 也合流
exec >>"$DIAG" 2>&1
set -x

set -uo pipefail

############################################
# 路径常量（已替换为本机真实位置）
############################################
# ---- 路径统一从仓库根 configs/paths.env 读取 ----
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/configs/paths.env"
QWEN_ROOT="${CTBENCH_ROOT}/qwen_cl"
BASE_MODEL="${QWEN_BASE}"
DATA_DIR="${TRAIN_DIR}"
IMG_ROOT="${IMAGE_ROOT}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${ENV_QWEN:-qwen_cl}"

############################################
# 启动参数（脚本里写死、不接命令行参数，方便 volc 直挂）
############################################
METHOD=grpo
HYPER=r64
ORDER=seqft        # 论文 Order-A 对应 seqft
RUN_NAME="Qwen2.5-VL-3B_${METHOD}_${HYPER}"
ROOT="${QWEN_ROOT}/grpo_cl_output/${ORDER}/${RUN_NAME}"
LOG="${ROOT}/debug_${RUN_NAME}.txt"
TIMING_FILE="${QWEN_ROOT}/timing_rft.json"
STOP_KEEPALIVE=/tmp/STOP_KEEPALIVE_MATMUL_RFT
KEEPALIVE_LOG=/tmp/keepalive_rft.log

mkdir -p "${ROOT}"
: > "${LOG}"

############################################
# 1. 启 keepalive 矩阵乘法（变化的低利用率，防 worker 闲置回收）
############################################
echo "[$(date '+%F %T')] launch keepalive matmul (varied duty cycle, small ~50MB/GPU)" | tee -a "${LOG}"
rm -f "${STOP_KEEPALIVE}" 2>/dev/null || true
"${CONDA_BASE}/envs/${CONDA_ENV}/bin/python" "${QWEN_ROOT}/keepalive_matmul.py" \
    --stop-file "${STOP_KEEPALIVE}" --mat-size 2048 \
    > "${KEEPALIVE_LOG}" 2>&1 &
KEEPALIVE_PID=$!
echo "[$(date '+%F %T')] keepalive pid=${KEEPALIVE_PID}" | tee -a "${LOG}"

cleanup() {
  echo "[$(date '+%F %T')] stopping keepalive..." | tee -a "${LOG}"
  touch "${STOP_KEEPALIVE}" 2>/dev/null || true
  sleep 3
  kill "${KEEPALIVE_PID}" 2>/dev/null || true
  pkill -P "${KEEPALIVE_PID}" 2>/dev/null || true
}
trap cleanup EXIT

############################################
# 2. 激活 conda + 环境变量
############################################
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${QWEN_ROOT}"
# 与 jointft 保持一致；RFT 流程虽然不直接调 eval，但 grpo_rec.py 可能间接依赖
export QWEN_TEST_ROOT="/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/qwen data/test"
export QWEN_IMAGE_ROOT="${IMG_ROOT}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

cd "${QWEN_ROOT}"

############################################
# 3. 任务序列 + 数据路径表
############################################
TASKS=(numglue art math fomc medical OCR science)
declare -A DS_PATH
for T in "${TASKS[@]}"; do
  DS_PATH[$T]="${DATA_DIR}/${T}.json"
done

############################################
# 4. 总耗时计时
############################################
TOTAL_START=$(date +%s)
TOTAL_START_HUMAN=$(date '+%F %T')
echo "[${TOTAL_START_HUMAN}] ====== RFT (GRPO-only) CL pipeline START ======" | tee -a "${LOG}"

PER_TASK_FILE="${ROOT}/_per_task_seconds.tsv"
: > "${PER_TASK_FILE}"

CUR_BASE="${BASE_MODEL}"
STEP=1

for TASK in "${TASKS[@]}"; do
  OUT_LORA="${ROOT}/${STEP}-lora"
  OUT_MERGE="${ROOT}/${STEP}"

  TASK_START=$(date +%s)
  echo -e "\n===== Step ${STEP}: ${TASK}  (start $(date '+%F %T')) =====" | tee -a "${LOG}"

  ############### GRPO LoRA 训练 ###############
  if [[ -f "${OUT_LORA}/preprocessor_config.json" ]]; then
    echo "[skip] GRPO LoRA already done: ${OUT_LORA}" | tee -a "${LOG}"
  else
    torchrun --nproc_per_node=8 \
        "${QWEN_ROOT}/src/src/open_r1/grpo_rec.py" \
        --deepspeed ./src/open-r1-multimodal/local_scripts/zero3.json \
        --output_dir "${OUT_LORA}" \
        --model_name_or_path "${CUR_BASE}" \
        --dataset_name "${DS_PATH[$TASK]}" \
        --image_root "${IMG_ROOT}" \
        --max_prompt_length 4096 \
        --num_generations 4 \
        --per_device_train_batch_size 16 \
        --num_iterations 8 \
        --epsilon 0.15 \
        --gradient_accumulation_steps 1 \
        --logging_steps 1 \
        --bf16 \
        --data_seed 42 \
        --report_to wandb \
        --gradient_checkpointing true \
        --attn_implementation flash_attention_2 \
        --num_train_epochs 1 \
        --run_name "${RUN_NAME}" \
        --save_only_model true \
        --save_steps 500 \
        --learning_rate 1e-5 \
        --use_peft true \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.05 \
        --lora_task_type CAUSAL_LM \
        --freeze_vision_modules true \
        >> "${LOG}" 2>&1 || {
          echo "[FAIL] GRPO step ${STEP}:${TASK}" | tee -a "${LOG}"
          exit 11
        }
  fi

  ############### 合并 LoRA ###############
  if [[ -f "${OUT_MERGE}/chat_template.json" ]]; then
    echo "[skip] merge already done: ${OUT_MERGE}" | tee -a "${LOG}"
  else
    python "${QWEN_ROOT}/src/src/open_r1/lora_merge.py" \
        --base_model_path "${CUR_BASE}" \
        --lora_path "${OUT_LORA}" \
        --save_path "${OUT_MERGE}" \
        >> "${LOG}" 2>&1 || {
          echo "[FAIL] merge step ${STEP}:${TASK}" | tee -a "${LOG}"
          exit 12
        }
  fi

  TASK_END=$(date +%s)
  TASK_SEC=$((TASK_END - TASK_START))
  echo "===== Step ${STEP}: ${TASK} DONE in ${TASK_SEC}s  ($(date '+%F %T')) =====" | tee -a "${LOG}"
  echo -e "${TASK}\t${TASK_SEC}" >> "${PER_TASK_FILE}"

  CUR_BASE="${OUT_MERGE}"
  STEP=$((STEP + 1))
done

############################################
# 5. 总耗时写 JSON
############################################
TOTAL_END=$(date +%s)
TOTAL_SEC=$((TOTAL_END - TOTAL_START))
TOTAL_END_HUMAN=$(date '+%F %T')
TOTAL_HOURS=$(awk "BEGIN{printf \"%.3f\", ${TOTAL_SEC}/3600}")

# 用 python 拼 JSON，避免 shell 转义坑
"${CONDA_BASE}/envs/${CONDA_ENV}/bin/python" - <<PY > "${TIMING_FILE}"
import json
rows = open("${PER_TASK_FILE}").read().strip().split("\n") if open("${PER_TASK_FILE}").read().strip() else []
per_task = {}
for r in rows:
    k, v = r.split("\t"); per_task[k] = int(v)
print(json.dumps({
    "mode": "RFT",
    "method": "${METHOD}",
    "order": "${ORDER}",
    "run_name": "${RUN_NAME}",
    "start_ts": "${TOTAL_START_HUMAN}",
    "end_ts": "${TOTAL_END_HUMAN}",
    "total_seconds": ${TOTAL_SEC},
    "total_hours": ${TOTAL_HOURS},
    "per_task_seconds": per_task,
    "output_root": "${ROOT}",
}, indent=2, ensure_ascii=False))
PY

echo -e "\n========================================" | tee -a "${LOG}"
echo "[${TOTAL_END_HUMAN}] RFT pipeline DONE, total ${TOTAL_SEC}s (~$(awk "BEGIN{printf \"%.2f\", ${TOTAL_SEC}/3600}") h)" | tee -a "${LOG}"
echo "Timing JSON: ${TIMING_FILE}" | tee -a "${LOG}"
cat "${TIMING_FILE}" | tee -a "${LOG}"
