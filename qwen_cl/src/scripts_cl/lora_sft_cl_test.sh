#!/usr/bin/env bash
set -e  # 出错即退出
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

######### 基本参数 #########
METHOD=$1            # 如 "grpo"
HYPER=$2             # 如 "r64"
ORDER=$3             # seqft | seqftback

BASE_MODEL=/public/home/houzhiyan/Qwen2.5-VL-3B-Instruct
RUN_NAME="Qwen2.5-VL-3B_${METHOD}_${HYPER}"
ROOT=cl_output/${ORDER}/${RUN_NAME}
LOG=${ROOT}/debug_${RUN_NAME}.txt
mkdir -p "${ROOT}" && touch "${LOG}"

######### 任务列表 #########
SEQFT=(numglue art math fomc medical OCR science)
SEQFTBACK=(science OCR medical fomc math art numglue)
declare -A DS_PATH=(
  [numglue]=/public/home/houzhiyan/VLM-R1-main/RLCL_datasets/train_for_test/numglue.json
  [art]=/public/home/houzhiyan/VLM-R1-main/RLCL_datasets/train_for_test/art.json
  [math]=/public/home/houzhiyan/VLM-R1-main/RLCL_datasets/train_for_test/math.json
  [fomc]=/public/home/houzhiyan/VLM-R1-main/RLCL_datasets/train_for_test/fomc.json
  [medical]=/public/home/houzhiyan/VLM-R1-main/RLCL_datasets/train_for_test/medical.json
  [OCR]=/public/home/houzhiyan/VLM-R1-main/RLCL_datasets/train_for_test/OCR.json
  [science]=/public/home/houzhiyan/VLM-R1-main/RLCL_datasets/train_for_test/science.json
)

TASKS=("${SEQFT[@]}")
[[ $ORDER == "seqftback" ]] && TASKS=("${SEQFTBACK[@]}")

######### 循环训练 + 合并 #########
CUR_BASE=$BASE_MODEL
STEP=1

train_and_merge () {
  local TASK=$1    # 当前任务名
  local OUT_LORA=${ROOT}/${STEP}-lora
  local OUT_MERGE=${ROOT}/${STEP}

  echo -e "\n===== Step ${STEP}: ${TASK} =====" | tee -a "${LOG}"

  # —— 1. LoRA Finetune ——
  echo "训练的模型是：${CUR_BASE}"
  echo "模型保存的位置是：${OUT_LORA}"
  echo "训练数据是：${DS_PATH[$TASK]}"

  if [[ -f "${OUT_LORA}/preprocessor_config.json" ]]; then
    echo "Preprocessor config exists at ${OUT_LORA}/preprocessor_config.json, skipping LoRA training." | tee -a "${LOG}"
  else
    accelerate launch --config_file=src/open-r1-multimodal/configs/zero3_test.yaml src/src/open_r1/sft.py \
        --model_name_or_path "${CUR_BASE}" \
        --dataset_name "${DS_PATH[$TASK]}" \
        --image_root /home/houzhiyan/dataset/images \
        --learning_rate 1e-5 \
        --num_train_epochs 2 \
        --output_dir "${OUT_LORA}" \
        --packing \
        --max_seq_length 4096 \
        --per_device_train_batch_size 12 \
        --gradient_accumulation_steps 1 \
        --gradient_checkpointing true \
        --logging_steps 5 \
        --bf16 \
        --data_seed 42 \
        --report_to wandb \
        --attn_implementation flash_attention_2 \
        --save_only_model true \
        --save_steps 500 \
        --lora_target_modules visual \
        --freeze_vision_modules true \
        --use_peft true \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.05 \
        --lora_task_type CAUSAL_LM \
        --cur_task $TASK \
        # >> $LOG 2>&1
  fi
  echo -e "\n===== Step ${STEP}: ${TASK}训练完成，开始合并模型 =====" | tee -a "${LOG}"
  # —— 2. 合并 LoRA 到基座 ——
  if [[ -f "${OUT_MERGE}/chat_template.json" ]]; then
    echo "Chat template exists at ${OUT_MERGE}/chat_template.json, skipping merge." | tee -a "${LOG}"
  else
    python /public/home/houzhiyan/VLM-R1-main/src/src/open_r1/lora_merge.py \
        --base_model_path "${CUR_BASE}" \
        --lora_path "${OUT_LORA}" \
        --save_path "${OUT_MERGE}" \
        # >> $LOG 2>&1
  fi
  # —— 3. 更新基座路径，进入下一轮 ——
  CUR_BASE="${OUT_MERGE}"
  ((STEP++))
}

for TASK in "${TASKS[@]}"; do
  train_and_merge "${TASK}"
done