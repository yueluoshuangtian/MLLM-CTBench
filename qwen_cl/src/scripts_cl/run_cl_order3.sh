#!/bin/bash
# run_cl_order3.sh — sequential CL training on order3 for Qwen2.5-VL-3B-Instruct.
#
# Usage:
#   bash run_cl_order3.sh <CL_METHOD> [TRAIN_RATIO] [NUM_EPOCHS]
#
# CL_METHOD ∈ {none, ewc, mas, lwf, freeze_init, freeze_last, freeze, replay, der, l2p, max_merge}
# TRAIN_RATIO: 1.0 = full data; 0.005 = 0.5% smoke test
# NUM_EPOCHS: per-task epochs (paper Qwen LoRA: 3)
#
# Order-3 (paper difficulty curriculum):
#   art → OCR → fomc → science → numglue → math → medical
#
# Per-task LoRA r=64 α=128 dropout=0.05  lr=2e-5  (paper Table for Qwen2.5-VL-3B)

set -eu

# ===== config =====
METHOD=${1:?"need cl_method, e.g. none/ewc/mas/lwf/freeze/replay/der/l2p/max_merge"}
TRAIN_RATIO=${2:-1.0}
NUM_EPOCHS=${3:-3}

# ---- 路径统一从仓库根的 configs/paths.env 读取 ----
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(cd "${_HERE}/../../.." && pwd)/configs/paths.env"
ROOT_REPO="${CTBENCH_ROOT}/qwen_cl"
BASE_MODEL="${QWEN_BASE}"
DATA_ROOT="${TRAIN_DIR}"
TEST_ROOT="${TEST_DIR}"
IMG_ROOT="${IMAGE_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT}/qwen_cl/order3/${METHOD}_tr${TRAIN_RATIO}"

# CL hyper-params per method (paper defaults / sensible LoRA-scale)
CL_LBD=1.0
REPLAY_RATIO=0.0
case "${METHOD}" in
    ewc)         CL_LBD=1e3 ;;
    mas)         CL_LBD=1e2 ;;
    lwf)         CL_LBD=1.0 ;;
    der)         CL_LBD=0.5 ; REPLAY_RATIO=0.1 ;;
    replay)      REPLAY_RATIO=0.2 ;;
esac

export QWEN_TEST_ROOT="${TEST_ROOT}"
export QWEN_IMAGE_ROOT="${IMG_ROOT}"
export WANDB_MODE=offline
export DS_SKIP_CUDA_CHECK=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Mitigate OOM during in-process inference (especially OCR with large images).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ORDER=order3
TASKS=(art OCR fomc science numglue math medical)
NUM_TASKS=${#TASKS[@]}

mkdir -p "${OUTPUT_ROOT}"
LOG="${OUTPUT_ROOT}/run.log"
touch "${LOG}"

echo "=========================================" | tee -a "${LOG}"
echo "[run_cl_order3] method=${METHOD} train_ratio=${TRAIN_RATIO} epochs=${NUM_EPOCHS}" | tee -a "${LOG}"
echo "[run_cl_order3] cl_lbd=${CL_LBD} replay_ratio=${REPLAY_RATIO}" | tee -a "${LOG}"
echo "[run_cl_order3] OUTPUT_ROOT=${OUTPUT_ROOT}" | tee -a "${LOG}"
echo "[run_cl_order3] tasks=${TASKS[*]}" | tee -a "${LOG}"
echo "=========================================" | tee -a "${LOG}"

STEP=1
for TASK in "${TASKS[@]}"; do
    OUT_DIR="${OUTPUT_ROOT}/${STEP}-sft"
    mkdir -p "${OUT_DIR}"

    DATA_PATH="${DATA_ROOT}/${TASK}.json"

    # Always pass --image_root; text-only tasks (fomc/numglue) won't use it,
    # but replay paths may mix in image-bearing samples that need it.
    IMG_ARGS=(--image_root "${IMG_ROOT}")

    # continue from previous adapter for sequential methods.
    # max_merge is the exception: each task trains from base (independent FT)
    # so MagMaX gets diverse per-task LoRAs to max-merge.
    CONT_ARGS=()
    if [[ ${STEP} -gt 1 && "${METHOD}" != "max_merge" ]]; then
        # adapter saved at <step-1>-sft/task_<step-1>/ subdir (PEFT save_adapter_snapshot layout)
        PREV_ADAPTER_DIR="${OUTPUT_ROOT}/$((STEP-1))-sft/task_$((STEP-1))"
        CONT_ARGS=(--continue_lora_path "${PREV_ADAPTER_DIR}")
    fi

    # ER / DER need replay paths = prior tasks' train json (colon-separated)
    REPLAY_ARGS=()
    if [[ "${METHOD}" == "replay" || "${METHOD}" == "der" ]]; then
        if [[ ${STEP} -gt 1 ]]; then
            PRIOR_PATHS=""
            for (( j=0; j<STEP-1; j++ )); do
                PRIOR_PATHS+="${DATA_ROOT}/${TASKS[$j]}.json:"
            done
            PRIOR_PATHS="${PRIOR_PATHS%:}"
            REPLAY_ARGS=(--replay_paths "${PRIOR_PATHS}" --replay_ratio "${REPLAY_RATIO}")
        fi
    fi

    # last task: tell main() to run finalize() — only matters for max_merge
    IS_LAST=False
    if [[ ${STEP} -eq ${NUM_TASKS} ]]; then
        IS_LAST=True
    fi

    echo "" | tee -a "${LOG}"
    echo "===== Step ${STEP}/${NUM_TASKS}: ${TASK} =====" | tee -a "${LOG}"
    echo "  data=${DATA_PATH}"     | tee -a "${LOG}"
    echo "  out_dir=${OUT_DIR}"    | tee -a "${LOG}"
    echo "  continue=${CONT_ARGS[*]:-<none>}" | tee -a "${LOG}"
    echo "  replay=${REPLAY_ARGS[*]:-<none>}" | tee -a "${LOG}"
    echo "  is_last=${IS_LAST}"    | tee -a "${LOG}"

    accelerate launch \
        --num_processes 8 \
        --config_file "${ROOT_REPO}/src/open-r1-multimodal/configs/zero1_real.yaml" \
        "${ROOT_REPO}/src/src/open_r1/sft.py" \
        --model_name_or_path "${BASE_MODEL}" \
        --dataset_name "${DATA_PATH}" \
        "${IMG_ARGS[@]}" \
        --output_dir "${OUT_DIR}" \
        --learning_rate 2e-5 \
        --num_train_epochs "${NUM_EPOCHS}" \
        --max_length 4096 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --gradient_checkpointing true \
        --logging_steps 5 \
        --bf16 true \
        --data_seed 42 \
        --report_to none \
        --attn_implementation flash_attention_2 \
        --save_only_model true \
        --save_strategy "no" \
        --freeze_vision_modules true \
        --use_peft true \
        --cur_task "${TASK}" \
        --task_k ${STEP} \
        --order "${ORDER}" \
        --train_ratio "${TRAIN_RATIO}" \
        --train_ratio_strategy random \
        --peft_method lora \
        --lora_enable True \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_dropout 0.05 \
        --cl_method "${METHOD}" \
        --cl_lbd "${CL_LBD}" \
        --is_last_task "${IS_LAST}" \
        "${CONT_ARGS[@]}" \
        "${REPLAY_ARGS[@]}" \
        --do_infer true \
        --infer_bs 8 \
        --infer_max_new_tokens 128 \
        2>&1 | tee -a "${LOG}"

    STEP=$((STEP+1))
done

echo "" | tee -a "${LOG}"
echo "[run_cl_order3] all ${NUM_TASKS} tasks done for method=${METHOD}" | tee -a "${LOG}"
echo "  predictions:      ${OUTPUT_ROOT}/predictions/" | tee -a "${LOG}"
echo "  last_predictions: ${OUTPUT_ROOT}/last_predictions/" | tee -a "${LOG}"
echo "  next: python eval_matrix.py ${OUTPUT_ROOT}" | tee -a "${LOG}"
