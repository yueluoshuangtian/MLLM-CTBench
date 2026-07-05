#!/usr/bin/env bash
set -e
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ORDER=$1
METHOD=$2
expert_num=$3
gate_rank=$4
orth_lambda=$5
router=$6  #"a","mlp"
train_ratio=${7:-1.0}

if [[ -z "$METHOD" || -z "$ORDER" || -z "$expert_num" || -z "$router" ]]; then
  echo "Usage: bash run.sh <ORDER> <METHOD> <expert_num> <gate_rank> <orth_lambda> <router> [train_ratio]"
  echo "ORDER: order1|order2|order3|order4|order5|order6"
  exit 1
fi

BASE_MODEL=/home/houzhiyan/Qwen2.5-VL-3B-Instruct
RUN_NAME="Qwen2.5-VL-3B_${METHOD}_expert_num${expert_num}_gate_rank${gate_rank}_${orth_lambda}"

ROOT=/home/houzhiyan/qwen/cl_output/${ORDER}/${router}/${RUN_NAME}
LOG=${ROOT}/debug_${RUN_NAME}.txt
mkdir -p "${ROOT}" && touch "${LOG}"

# peft_method: lora | moelora
peft_method=${METHOD}

declare -A DS_PATH=(
  [numglue]=/home/houzhiyan/dataset/qwen/train/numglue.json
  [art]=/home/houzhiyan/dataset/qwen/train/art.json
  [math]=/home/houzhiyan/dataset/qwen/train/math.json
  [fomc]=/home/houzhiyan/dataset/qwen/train/fomc.json
  [medical]=/home/houzhiyan/dataset/qwen/train/medical.json
  [OCR]=/home/houzhiyan/dataset/qwen/train/OCR.json
  [science]=/home/houzhiyan/dataset/qwen/train/science.json

  # NOTE: these must be .json/.jsonl/.json for your Python dataset loader
  [GQA]=/home/houzhiyan/dataset/qwen/train/GQA/train.json
  [Grounding]=/home/houzhiyan/dataset/qwen/train/Grounding/train.json
  [ImageNet]=/home/houzhiyan/dataset/qwen/train/ImageNet/train.json
  [OCRVQA]=/home/houzhiyan/dataset/qwen/train/OCRVQA/train.json
  [ScienceQA]=/home/houzhiyan/dataset/qwen/train/ScienceQA/train.json
  [TextVQA]=/home/houzhiyan/dataset/qwen/train/TextVQA/train.json
  [VizWiz]=/home/houzhiyan/dataset/qwen/train/VizWiz/train.json
  [VQAv2]=/home/houzhiyan/dataset/qwen/train/VQAv2/train.json
)
declare -A IMAGE_ROOT=(
  [numglue]=""
  [fomc]=""
  [art]="/home/houzhiyan/dataset/images"
  [math]="/home/houzhiyan/dataset/images"
  [medical]="/home/houzhiyan/dataset/images"
  [OCR]="/home/houzhiyan/dataset/images"
  [science]="/home/houzhiyan/dataset/images"

  [ScienceQA]="/home/houzhiyan/dataset/CoIN"
  [TextVQA]="/home/houzhiyan/dataset/CoIN"
  [ImageNet]="/home/houzhiyan/dataset/CoIN"
  [GQA]="/home/houzhiyan/dataset/CoIN"
  [VizWiz]="/home/houzhiyan/dataset/CoIN"
  [Grounding]="/home/houzhiyan/dataset/CoIN"
  [VQAv2]="/home/houzhiyan/dataset/CoIN"
  [OCRVQA]="/home/houzhiyan/dataset/CoIN"
)

TASKS=()
if [[ "$ORDER" == "order1" ]]; then
  TASKS=(numglue art math fomc medical OCR science)
elif [[ "$ORDER" == "order2" ]]; then
  TASKS=(science OCR medical fomc math art numglue)
elif [[ "$ORDER" == "order3" ]]; then
  TASKS=(ScienceQA TextVQA ImageNet GQA VizWiz Grounding VQAv2 OCRVQA)
elif [[ "$ORDER" == "order4" ]]; then
  TASKS=(ScienceQA ImageNet VizWiz Grounding TextVQA GQA VQAv2 OCRVQA)
elif [[ "$ORDER" == "order5" ]]; then
  TASKS=(fomc art math science numglue OCR medical)
elif [[ "$ORDER" == "order6" ]]; then
  TASKS=(science medical fomc art math numglue OCR)
else
  echo "[ERROR] Unknown ORDER: $ORDER"
  exit 1
fi

build_image_root_args () {
  local TASK=$1
  local ROOT_PATH="${IMAGE_ROOT[$TASK]}"
  if [[ -z "$ROOT_PATH" ]]; then
    echo ""
  else
    echo "--image_root ${ROOT_PATH}"
  fi
}

STEP=1
for TASK in "${TASKS[@]}"; do
  OUT_DIR="${ROOT}/${STEP}-sft"
  mkdir -p "${OUT_DIR}"

  DATA_PATH="${DS_PATH[$TASK]}"
  if [[ -z "$DATA_PATH" ]]; then
    echo "[ERROR] DS_PATH not set for task: ${TASK}" | tee -a "${LOG}"
    exit 1
  fi

  IMG_ARGS="$(build_image_root_args "$TASK")"

  CONT_ARGS=""
  if [[ $STEP -gt 1 ]]; then
    PREV_ADAPTER_DIR="${ROOT}/$((STEP-1))-sft/task_$((STEP-1))"
    CONT_ARGS="--continue_lora_path ${PREV_ADAPTER_DIR}"
  fi

  echo -e "\n===== Step ${STEP}: ${TASK} =====" | tee -a "${LOG}"
  echo "BASE_MODEL  : ${BASE_MODEL}" | tee -a "${LOG}"
  echo "OUT_DIR     : ${OUT_DIR}" | tee -a "${LOG}"
  echo "DATA        : ${DATA_PATH}" | tee -a "${LOG}"
  echo "IMG_ARGS    : ${IMG_ARGS:-<none>}" | tee -a "${LOG}"
  echo "CONT_ARGS   : ${CONT_ARGS:-<none>}" | tee -a "${LOG}"
  echo "peft_method : ${peft_method}" | tee -a "${LOG}"
  echo "expert_num=${expert_num}, gate_rank=${gate_rank}, orth_lambda=${orth_lambda}, router=${router}" | tee -a "${LOG}"
  echo "train_ratio=${train_ratio}" | tee -a "${LOG}"

  # shellcheck disable=SC2086
  accelerate launch --num_processes 8 --config_file=src/open-r1-multimodal/configs/zero2.yaml src/src/open_r1/sft.py \
    --model_name_or_path "${BASE_MODEL}" \
    --dataset_name "${DATA_PATH}" \
    ${IMG_ARGS} \
    --output_dir "${OUT_DIR}" \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --max_length 4096 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --logging_steps 5 \
    --bf16 true \
    --data_seed 42 \
    --report_to wandb \
    --attn_implementation flash_attention_2 \
    --save_only_model true \
    --save_steps 500 \
    --freeze_vision_modules true \
    --use_peft true \
    --cur_task "$TASK" \
    --task_k ${STEP} \
    --order "${ORDER}" \
    --train_ratio ${train_ratio} \
    --train_ratio_strategy random \
    ${CONT_ARGS} \
    \
    --peft_method "${peft_method}" \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --expert_num ${expert_num} \
    --gate_rank ${gate_rank} \
    --moelora_orth_lambda ${orth_lambda} \
    --gate_router ${router} \
    --moelora_reg_enable True \
    --moelora_importance_top_p 0.4 \
    --moelora_B_reg_scale 0.5 \
    2>&1 | tee -a "$LOG"

  ((STEP++))
done