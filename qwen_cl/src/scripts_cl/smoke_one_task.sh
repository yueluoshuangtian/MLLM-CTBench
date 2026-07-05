#!/bin/bash
# smoke_one_task.sh — fastest possible end-to-end smoke for one CL method.
# Runs ONLY task 1 (art) with 0.5% data + 1 epoch + 8 GPUs.
#
# Usage:  bash smoke_one_task.sh <CL_METHOD>
#         CL_METHOD ∈ {none, ewc, mas, lwf, freeze, replay, der, l2p, max_merge}

set -eu
METHOD=${1:?"need cl_method, e.g. none"}
ROOT_REPO=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/TMM/qwen_cl
BASE_MODEL=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/models/Qwen2.5-VL-3B-Instruct
TRAIN_DATA="/mnt/cxzx/share/workspace/data_transfer/houzhiyan/TMM/qwen data/train/art.json"
TEST_ROOT="/mnt/cxzx/share/workspace/data_transfer/houzhiyan/TMM/qwen data/test"
IMG_ROOT=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/clmm-benchmark
OUTPUT_ROOT=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/output/qwen_cl_smoke/${METHOD}

mkdir -p "${OUTPUT_ROOT}/1-sft"

export QWEN_TEST_ROOT="${TEST_ROOT}"
export QWEN_IMAGE_ROOT="${IMG_ROOT}"
export WANDB_MODE=offline
export DS_SKIP_CUDA_CHECK=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# CL hyper-params per method
CL_LBD=1.0
case "${METHOD}" in
    ewc) CL_LBD=1e3 ;;
    mas) CL_LBD=1e2 ;;
    lwf) CL_LBD=1.0 ;;
    der) CL_LBD=0.5 ;;
esac

echo "=== SMOKE method=${METHOD} ==="
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "TRAIN_DATA=${TRAIN_DATA}"

accelerate launch \
    --num_processes 8 \
    --config_file "${ROOT_REPO}/src/open-r1-multimodal/configs/zero2.yaml" \
    "${ROOT_REPO}/src/src/open_r1/sft.py" \
    --model_name_or_path "${BASE_MODEL}" \
    --dataset_name "${TRAIN_DATA}" \
    --image_root "${IMG_ROOT}" \
    --output_dir "${OUTPUT_ROOT}/1-sft" \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --bf16 true \
    --data_seed 42 \
    --report_to none \
    --attn_implementation flash_attention_2 \
    --save_only_model true \
    --save_strategy "no" \
    --freeze_vision_modules true \
    --use_peft true \
    --cur_task art \
    --task_k 1 \
    --order order3 \
    --train_ratio 0.02 \
    --train_ratio_strategy random \
    --peft_method lora \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --cl_method "${METHOD}" \
    --cl_lbd "${CL_LBD}" \
    --is_last_task False \
    --do_infer true \
    --infer_bs 16 \
    --infer_max_new_tokens 64 \
    --infer_sample_num 20 \
    2>&1 | tee "${OUTPUT_ROOT}/smoke.log"

echo ""
echo "=== SMOKE result ==="
echo "  predictions: ${OUTPUT_ROOT}/predictions/"
ls -la "${OUTPUT_ROOT}/predictions/" 2>&1 || echo "  (no predictions dir)"
