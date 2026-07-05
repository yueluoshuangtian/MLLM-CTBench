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
train_ratio=${7:-0.4}

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

# ----------------------------
# 自动检测断点：找到最后一个“已完成”的 step
# 规则：存在目录 ROOT/<k>-sft/task_<k>，且里面至少有一个：
#   adapter_model.safetensors | adapter_model.bin | adapter_config.json
# ----------------------------
detect_last_completed_step () {
  local last=0
  shopt -s nullglob

  local d
  for d in "${ROOT}"/*-sft/task_*; do
    [[ -d "$d" ]] || continue

    # 期望路径形如: .../<k>-sft/task_<k>
    # 从父目录名 "<k>-sft" 抽取 k
    local parent
    parent="$(basename "$(dirname "$d")")"  # e.g., "3-sft"
    if [[ "$parent" =~ ^([0-9]+)-sft$ ]]; then
      local k="${BASH_REMATCH[1]}"

      # 判定“保存完整”：至少有一个关键文件
      if [[ -f "$d/added_tokens.json" || -f "$d/non_lora_trainables.bin" || -f "$d/adapter_config.json" ]]; then
        if (( k > last )); then
          last="$k"
        fi
      fi
    fi
  done

  shopt -u nullglob
  echo "$last"
}

LAST_DONE="$(detect_last_completed_step)"
START_STEP=$((LAST_DONE + 1))
TOTAL_TASKS=${#TASKS[@]}

echo "============================================================" | tee -a "${LOG}"
echo "[Resume Check] ROOT=${ROOT}" | tee -a "${LOG}"
echo "[Resume Check] LAST_DONE=${LAST_DONE}, START_STEP=${START_STEP}, TOTAL_TASKS=${TOTAL_TASKS}" | tee -a "${LOG}"
echo "============================================================" | tee -a "${LOG}"

if (( START_STEP > TOTAL_TASKS )); then
  echo "[INFO] All tasks are already finished. Nothing to run." | tee -a "${LOG}"
  exit 0
fi

# ----------------------------
# 从 START_STEP 开始跑（跳过已完成的）
# ----------------------------
for i in "${!TASKS[@]}"; do
  STEP=$((i + 1))
  TASK="${TASKS[$i]}"

  if (( STEP < START_STEP )); then
    echo "[SKIP] Step ${STEP} (${TASK}) already done. Skip." | tee -a "${LOG}"
    continue
  fi

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
    --num_train_epochs 1 \
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
    --save_steps 2000 \
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
done