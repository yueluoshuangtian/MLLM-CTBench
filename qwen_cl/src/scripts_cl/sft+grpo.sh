#!/usr/bin/env bash
set -euo pipefail

########################################
# 运行入参
########################################
ORDER=${1:-"seqft"}     # seqft | seqftback
HYPER=${2:-"r64"}       # 仅用于命名标签，如 r64
METHOD="grpo"           # 保持与你的GRPO脚本命名一致

########################################
# 环境设置
########################################

export WANDB_MODE=offline
export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH=/home/houzhiyan/qwen
########################################
# 基座与公共路径
########################################
BASE_MODEL=/home/houzhiyan/Qwen2.5-VL-3B-Instruct
IMG_ROOT=/home/houzhiyan/dataset/images
RUN_NAME="Qwen2.5-VL-3B_${METHOD}_${HYPER}"
ROOT=sft+grpo_cl_output/${ORDER}/${RUN_NAME}
LOG=${ROOT}/debug_${RUN_NAME}.txt
mkdir -p "${ROOT}" && touch "${LOG}"

########################################
# 任务列表与数据映射
########################################
SEQFT=(numglue art math fomc medical OCR science)
SEQFTBACK=(science OCR medical fomc math art numglue)
declare -A DS_PATH=(
  [numglue]=/home/houzhiyan/dataset/qwen/train/numglue.json
  [art]=/home/houzhiyan/dataset/qwen/train/art.json
  [math]=/home/houzhiyan/dataset/qwen/train/math.json
  [fomc]=/home/houzhiyan/dataset/qwen/train/fomc.json
  [medical]=/home/houzhiyan/dataset/qwen/train/medical.json
  [OCR]=/home/houzhiyan/dataset/qwen/train/OCR.json
  [science]=/home/houzhiyan/dataset/qwen/train/science.json
)

TASKS=("${SEQFT[@]}")
[[ "${ORDER}" == "seqftback" ]] && TASKS=("${SEQFTBACK[@]}")

# evaluate_cl.py 的 --tasks 字符串
SEQFT_TASKS="numglue,art,math,fomc,medical,OCR,science"
SEQFTBACK_TASKS="science,OCR,medical,fomc,math,art,numglue"
TASKS_STR=$SEQFT_TASKS
[[ "${ORDER}" == "seqftback" ]] && TASKS_STR=$SEQFTBACK_TASKS

########################################
# 当前基座（会被循环更新）
########################################
CUR_BASE="${BASE_MODEL}"
STEP=1

########################################
# 小工具
########################################
timestamp() { date "+%F %T"; }

run_eval() {
  # 评测需传“实验根目录”，脚本会自动发现 1/,2/,... 子目录
  local ROOT_DIR=$1
  echo -e "\n[$(timestamp)] >>> 评测开始：${ROOT_DIR}" | tee -a "${LOG}"

  # 中途尚未到最后一步时，evaluate_cl.py 的 last-eval 会抛 RuntimeError；
  # 这里容错，不让主流程退出。
  set +e
  python src/src/open_r1/evaluate_cl.py \
      --model_paths "${ROOT_DIR}" \
      --bs 30 \
      --tasks "${TASKS_STR}" \
      --sample_num 1500 \
      --max_new_tokens 4096 \
      | tee -a "${LOG}"
  STATUS=$?
  set -e
  if [[ $STATUS -ne 0 ]]; then
    echo "[WARN] evaluate_cl.py 非零退出（多半因 last-eval 未到最后一步），已忽略。" | tee -a "${LOG}"
  fi
  echo -e "[$(timestamp)] <<< 评测完成：${ROOT_DIR}\n" | tee -a "${LOG}"
}

########################################
# 单任务流水：SFT → 合并 → GRPO → 合并 → 评测 → 更新基座
########################################
train_one_task() {
  local TASK=$1
  local DATA=${DS_PATH[$TASK]}

  local OUT_SFT_LORA=${ROOT}/${STEP}-sft-lora
  local OUT_SFT_MERGE=${ROOT}/${STEP}-sft
  local OUT_GRPO_LORA=${ROOT}/${STEP}-grpo-lora
  local OUT_GRPO_MERGE=${ROOT}/${STEP}

  echo -e "\n========== Step ${STEP}: ${TASK} ==========" | tee -a "${LOG}"
  echo "[SFT 基座] ${CUR_BASE}" | tee -a "${LOG}"

  ############################
  # 1) SFT（LoRA）
  ############################
  if [[ -f "${OUT_SFT_LORA}/preprocessor_config.json" ]]; then
    echo "[$(timestamp)] SFT 已存在，跳过：${OUT_SFT_LORA}" | tee -a "${LOG}"
  else
    accelerate launch --config_file=src/open-r1-multimodal/configs/zero3.yaml src/src/open_r1/sft.py \
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
      | tee -a "${LOG}"
  fi

  ############################
  # 2) 合并 SFT LoRA → SFT 基座
  ############################
  if [[ -f "${OUT_SFT_MERGE}/chat_template.json" ]]; then
    echo "[$(timestamp)] SFT 合并已存在，跳过：${OUT_SFT_MERGE}" | tee -a "${LOG}"
  else
    python /home/houzhiyan/qwen/src/src/open_r1/lora_merge.py \
      --base_model_path "${CUR_BASE}" \
      --lora_path "${OUT_SFT_LORA}" \
      --save_path "${OUT_SFT_MERGE}" \
      | tee -a "${LOG}"
  fi

  ############################
  # 3) GRPO（在 SFT 基座上做 LoRA 强化）
  ############################
  echo "[GRPO 基座] ${OUT_SFT_MERGE}" | tee -a "${LOG}"
    if [[ -f "${OUT_GRPO_LORA}/adapter_config.json" || -f "${OUT_GRPO_LORA}/adapter_model.safetensors" || -f "${OUT_GRPO_LORA}/pytorch_model.bin" ]]; then
    echo "[$(timestamp)] GRPO 已存在，跳过：${OUT_GRPO_LORA}" | tee -a "${LOG}"
  else
    export LOG_DIR="${OUT_GRPO_LORA}/logs"

    torchrun --nproc_per_node=8 \
      /home/houzhiyan/qwen/src/src/open_r1/grpo_rec.py \
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
      | tee -a "${LOG}"
  fi

  ############################
  # 4) 合并 GRPO LoRA → 得到该步最终模型
  ############################
  if [[ -f "${OUT_GRPO_MERGE}/chat_template.json" ]]; then
    echo "[$(timestamp)] GRPO 合并已存在，跳过：${OUT_GRPO_MERGE}" | tee -a "${LOG}"
  else
    python /home/houzhiyan/qwen/src/src/open_r1/lora_merge.py \
      --base_model_path "${OUT_SFT_MERGE}" \
      --lora_path "${OUT_GRPO_LORA}" \
      --save_path "${OUT_GRPO_MERGE}" \
      | tee -a "${LOG}"
  fi

  ############################
  # 5) 评测（根目录；post 总会跑，last 只有最后一步才会跑）
  ############################
  run_eval "${ROOT}"

  ############################
  # 6) 更新基座 → 下一任务
  ############################
  CUR_BASE="${OUT_GRPO_MERGE}"
  ((STEP++))
}

########################################
# 主循环
########################################
echo "==== 开始持续训练（先 SFT 再 GRPO），顺序：${ORDER}，RUN=${RUN_NAME} ====" | tee -a "${LOG}"
for TASK in "${TASKS[@]}"; do
  train_one_task "${TASK}"
done
echo "==== 全部任务完成。最终模型：${CUR_BASE} ====" | tee -a "${LOG}"
