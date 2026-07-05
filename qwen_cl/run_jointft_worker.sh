#!/usr/bin/env bash
# run_jointft_worker.sh — 在 8 卡 worker 上跑 Joint SFT-RFT 顺序 CL，记录耗时。
#
# 论文 Table VIII「Joint」行：每个任务先 1 epoch SFT 再 GRPO，然后合并、评测、下一任务。
# 7 个任务顺序：numglue → art → math → fomc → medical → OCR → science
#
# 用法（作为 volc 启动命令直接挂上来）：
#   volc ml_devinstance ml_launch \
#     --resource_queue_id q-20260129141046-tmbpd \
#     --flavor_id ml.pni2l.28xlarge \
#     bash /mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/qwen_cl/run_jointft_worker.sh
#
# 关键产出：
#   - /mnt/cxzx/.../TMM/qwen_cl/timing_jointft.json
#   - /mnt/cxzx/.../TMM/qwen_cl/sft+grpo_cl_output/seqft/Qwen2.5-VL-3B_grpo_r64/
#     debug_Qwen2.5-VL-3B_grpo_r64.txt
############################################
# DEBUG: 启动早期诊断
############################################
DIAG=/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/qwen_cl/_worker_diag_jointft.log
{
  echo "=== diag begin @ $(date) ==="
  echo "hostname=$(hostname)  whoami=$(whoami)"
  ls /mnt/cxzx/ 2>&1 | head
  nvidia-smi --query-gpu=index,name --format=csv,noheader 2>&1 | head
  ls /mnt/cxzx/workspace/data_transfer/houzhiyan/miniconda3/envs/qwen_cl/bin/python 2>&1
} > "$DIAG" 2>&1
exec >>"$DIAG" 2>&1
set -x

set -uo pipefail

############################################
# 路径常量
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
# 启动参数
############################################
METHOD=grpo
HYPER=r64
ORDER=seqft
RUN_NAME="Qwen2.5-VL-3B_${METHOD}_${HYPER}"
ROOT="${QWEN_ROOT}/sft+grpo_cl_output/${ORDER}/${RUN_NAME}"
LOG="${ROOT}/debug_${RUN_NAME}.txt"
TIMING_FILE="${QWEN_ROOT}/timing_jointft.json"
STOP_KEEPALIVE=/tmp/STOP_KEEPALIVE_MATMUL_JOINT
KEEPALIVE_LOG=/tmp/keepalive_jointft.log

mkdir -p "${ROOT}"
: > "${LOG}"

TASKS=(numglue art math fomc medical OCR science)
TASKS_STR="numglue,art,math,fomc,medical,OCR,science"
declare -A DS_PATH
for T in "${TASKS[@]}"; do
  DS_PATH[$T]="${DATA_DIR}/${T}.json"
done

############################################
# 1. keepalive 矩阵乘
############################################
echo "[$(date '+%F %T')] launch keepalive matmul" | tee -a "${LOG}"
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
# 2. conda + env
############################################
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export WANDB_MODE=offline
export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="${QWEN_ROOT}"
# sft.py / evaluate_cl.py 通过这两个 env 寻找 test json / 图像（原默认指向不存在的 /home 或 /mnt/cxzx/share 路径）
export QWEN_TEST_ROOT="/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/qwen data/test"
export QWEN_IMAGE_ROOT="${IMG_ROOT}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

cd "${QWEN_ROOT}"

############################################
# 3. timing 起点
############################################
TOTAL_START=$(date +%s)
TOTAL_START_HUMAN=$(date '+%F %T')
echo "[${TOTAL_START_HUMAN}] ====== Joint SFT-RFT CL pipeline START ======" | tee -a "${LOG}"

PER_TASK_FILE="${ROOT}/_per_task_seconds.tsv"
: > "${PER_TASK_FILE}"

CUR_BASE="${BASE_MODEL}"
STEP=1

############################################
# 4. 评测函数（中途 last-eval 报错容错）
############################################
run_eval() {
  local ROOT_DIR=$1
  echo -e "\n[$(date '+%F %T')] >>> 评测开始：${ROOT_DIR}" | tee -a "${LOG}"
  set +e
  python "${QWEN_ROOT}/src/src/open_r1/evaluate_cl.py" \
      --model_paths "${ROOT_DIR}" \
      --bs 30 \
      --tasks "${TASKS_STR}" \
      --sample_num 1500 \
      --max_new_tokens 4096 \
      | tee -a "${LOG}"
  STATUS=$?
  set -e
  if [[ $STATUS -ne 0 ]]; then
    echo "[WARN] evaluate_cl.py 非零退出（多半是 last-eval 未到最后），已忽略" | tee -a "${LOG}"
  fi
  echo -e "[$(date '+%F %T')] <<< 评测完成：${ROOT_DIR}\n" | tee -a "${LOG}"
}

############################################
# 5. 单任务流水：SFT → merge → GRPO → merge → eval
############################################
for TASK in "${TASKS[@]}"; do
  DATA="${DS_PATH[$TASK]}"

  OUT_SFT_LORA="${ROOT}/${STEP}-sft-lora"
  OUT_SFT_MERGE="${ROOT}/${STEP}-sft"
  OUT_GRPO_LORA="${ROOT}/${STEP}-grpo-lora"
  OUT_GRPO_MERGE="${ROOT}/${STEP}"

  TASK_START=$(date +%s)
  echo -e "\n========== Step ${STEP}: ${TASK} (start $(date '+%F %T')) ==========" | tee -a "${LOG}"

  ############## 1) SFT LoRA ##############
  if [[ -f "${OUT_SFT_LORA}/preprocessor_config.json" ]]; then
    echo "[skip] SFT LoRA exists: ${OUT_SFT_LORA}" | tee -a "${LOG}"
  else
    accelerate launch \
      --config_file=src/open-r1-multimodal/configs/zero3.yaml \
      "${QWEN_ROOT}/src/src/open_r1/sft.py" \
      --model_name_or_path "${CUR_BASE}" \
      --dataset_name "${DATA}" \
      --image_root "${IMG_ROOT}" \
      --learning_rate 1e-5 \
      --num_train_epochs 1 \
      --output_dir "${OUT_SFT_LORA}" \
      --packing \
      --max_seq_length 4096 \
      --per_device_train_batch_size 16 \
      --gradient_accumulation_steps 1 \
      --gradient_checkpointing true \
      --logging_steps 5 \
      --bf16 \
      --data_seed 42 \
      --report_to wandb \
      --attn_implementation flash_attention_2 \
      --save_only_model true \
      --save_steps 500 \
      --lora_target_modules q_proj v_proj visual \
      --use_peft true \
      --lora_r 64 \
      --lora_alpha 128 \
      --lora_dropout 0.05 \
      --lora_task_type CAUSAL_LM \
      --cur_task "${TASK}" \
      --torch_dtype bfloat16 \
      | tee -a "${LOG}" || { echo "[FAIL] SFT ${STEP}:${TASK}" | tee -a "${LOG}"; exit 21; }
  fi

  ############## 2) merge SFT LoRA ##############
  if [[ -f "${OUT_SFT_MERGE}/chat_template.json" ]]; then
    echo "[skip] SFT merge exists: ${OUT_SFT_MERGE}" | tee -a "${LOG}"
  else
    python "${QWEN_ROOT}/src/src/open_r1/lora_merge.py" \
      --base_model_path "${CUR_BASE}" \
      --lora_path "${OUT_SFT_LORA}" \
      --save_path "${OUT_SFT_MERGE}" \
      | tee -a "${LOG}" || { echo "[FAIL] SFT merge ${STEP}:${TASK}" | tee -a "${LOG}"; exit 22; }
  fi

  ############## 3) GRPO LoRA ##############
  if [[ -f "${OUT_GRPO_LORA}/adapter_config.json" || -f "${OUT_GRPO_LORA}/adapter_model.safetensors" ]]; then
    echo "[skip] GRPO LoRA exists: ${OUT_GRPO_LORA}" | tee -a "${LOG}"
  else
    export LOG_DIR="${OUT_GRPO_LORA}/logs"
    torchrun --nproc_per_node=8 \
      "${QWEN_ROOT}/src/src/open_r1/grpo_rec.py" \
      --deepspeed ./src/open-r1-multimodal/local_scripts/zero3.json \
      --output_dir "${OUT_GRPO_LORA}" \
      --model_name_or_path "${OUT_SFT_MERGE}" \
      --dataset_name "${DATA}" \
      --image_root "${IMG_ROOT}" \
      --max_prompt_length 4096 \
      --num_generations 4 \
      --per_device_train_batch_size 16 \
      --num_iterations 2 \
      --epsilon 0.15 \
      --gradient_accumulation_steps 1 \
      --logging_steps 1 \
      --bf16 \
      --data_seed 42 \
      --report_to wandb \
      --gradient_checkpointing true \
      --attn_implementation flash_attention_2 \
      --num_train_epochs 2 \
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
      | tee -a "${LOG}" || { echo "[FAIL] GRPO ${STEP}:${TASK}" | tee -a "${LOG}"; exit 23; }
  fi

  ############## 4) merge GRPO LoRA ##############
  if [[ -f "${OUT_GRPO_MERGE}/chat_template.json" ]]; then
    echo "[skip] GRPO merge exists: ${OUT_GRPO_MERGE}" | tee -a "${LOG}"
  else
    python "${QWEN_ROOT}/src/src/open_r1/lora_merge.py" \
      --base_model_path "${OUT_SFT_MERGE}" \
      --lora_path "${OUT_GRPO_LORA}" \
      --save_path "${OUT_GRPO_MERGE}" \
      | tee -a "${LOG}" || { echo "[FAIL] GRPO merge ${STEP}:${TASK}" | tee -a "${LOG}"; exit 24; }
  fi

  ############## 5) eval ##############
  run_eval "${ROOT}"

  TASK_END=$(date +%s)
  TASK_SEC=$((TASK_END - TASK_START))
  echo "========== Step ${STEP}: ${TASK} DONE in ${TASK_SEC}s  ($(date '+%F %T')) ==========" | tee -a "${LOG}"
  echo -e "${TASK}\t${TASK_SEC}" >> "${PER_TASK_FILE}"

  CUR_BASE="${OUT_GRPO_MERGE}"
  STEP=$((STEP + 1))
done

############################################
# 6. 总耗时 JSON
############################################
TOTAL_END=$(date +%s)
TOTAL_SEC=$((TOTAL_END - TOTAL_START))
TOTAL_END_HUMAN=$(date '+%F %T')
TOTAL_HOURS=$(awk "BEGIN{printf \"%.3f\", ${TOTAL_SEC}/3600}")

"${CONDA_BASE}/envs/${CONDA_ENV}/bin/python" - <<PY > "${TIMING_FILE}"
import json
rows = open("${PER_TASK_FILE}").read().strip().split("\n") if open("${PER_TASK_FILE}").read().strip() else []
per_task = {}
for r in rows:
    k, v = r.split("\t"); per_task[k] = int(v)
print(json.dumps({
    "mode": "JointFT (SFT+GRPO)",
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
echo "[${TOTAL_END_HUMAN}] JointFT pipeline DONE, total ${TOTAL_SEC}s (~$(awk "BEGIN{printf \"%.2f\", ${TOTAL_SEC}/3600}") h)" | tee -a "${LOG}"
echo "Timing JSON: ${TIMING_FILE}" | tee -a "${LOG}"
cat "${TIMING_FILE}" | tee -a "${LOG}"
