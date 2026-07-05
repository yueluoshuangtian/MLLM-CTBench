#!/bin/bash
# smoke_all_methods.sh — 9 个 CL 方法 × 2 任务的完整冒烟测试。
#
# 目标：在 ~5 min/方法的时间内验证每个方法的核心信号都正常：
#   - 训练可启动（model 载入 + LoRA 注入 + 反向传播）
#   - 跨任务串联可用（task 2 用 --continue_lora_path 接 task 1 的 adapter）
#   - 方法专属 hook 在 task 2 起效（EWC: reg loss；LwF: distill；DER: distill_mse；
#     Replay: 数据混入；Freeze: 冻部分 block；MagMaX: snapshot；L2P: prompt）
#   - 推理可保存到 predictions/<k>-<task>.json
#
# 参数：bash smoke_all_methods.sh [SAMPLE_LIMIT]
#       SAMPLE_LIMIT = 推理样本数（默认 50；为加速）

set -uo pipefail

SAMPLE_LIMIT=${1:-50}
TRAIN_RATIO=0.01
NUM_EPOCHS=1

ROOT_REPO=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/TMM/qwen_cl
BASE_MODEL=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/models/Qwen2.5-VL-3B-Instruct
DATA_ROOT=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/TMM/qwen\ data/train
TEST_ROOT=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/TMM/qwen\ data/test
IMG_ROOT=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/clmm-benchmark
SMOKE_ROOT=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/output/qwen_cl_smoke_full

# 2 任务序列（art 是 task 1，OCR 是 task 2 — 与 order3 头两个一致）
TASKS=(art OCR)

export QWEN_TEST_ROOT="${TEST_ROOT}"
export QWEN_IMAGE_ROOT="${IMG_ROOT}"
export WANDB_MODE=offline
export DS_SKIP_CUDA_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

mkdir -p "${SMOKE_ROOT}"
touch /mnt/cxzx/share/workspace/data_transfer/houzhiyan/gpu_occupy/DISABLE_WATCHDOG
echo "[smoke] DISABLE_WATCHDOG touched (no occupy during smoke)"
bash /mnt/cxzx/share/workspace/data_transfer/houzhiyan/gpu_occupy/stop_occupy.sh 2>&1 | tail -2 || true

METHODS=(none ewc mas lwf freeze replay der l2p max_merge)

REPORT="${SMOKE_ROOT}/smoke_report.txt"
echo "smoke_all_methods.sh report — $(date)" > "${REPORT}"
echo "TRAIN_RATIO=${TRAIN_RATIO}  NUM_EPOCHS=${NUM_EPOCHS}  SAMPLE_LIMIT=${SAMPLE_LIMIT}" >> "${REPORT}"
echo "tasks=${TASKS[*]}" >> "${REPORT}"
echo "" >> "${REPORT}"

run_method () {
    local METHOD=$1
    local OUT_DIR="${SMOKE_ROOT}/${METHOD}"
    rm -rf "${OUT_DIR}"
    mkdir -p "${OUT_DIR}"

    local CL_LBD=1.0
    local REPLAY_RATIO=0.0
    case "${METHOD}" in
        ewc) CL_LBD=1e3 ;;
        mas) CL_LBD=1e2 ;;
        lwf) CL_LBD=1.0 ;;
        der) CL_LBD=0.5; REPLAY_RATIO=0.1 ;;
        replay) REPLAY_RATIO=0.2 ;;
    esac

    local NUM_TASKS=${#TASKS[@]}
    local STEP=1
    local METHOD_OK=true
    for TASK in "${TASKS[@]}"; do
        local STEP_DIR="${OUT_DIR}/${STEP}-sft"
        mkdir -p "${STEP_DIR}"
        local DATA_PATH="${DATA_ROOT}/${TASK}.json"

        local IMG_ARGS=()
        if [[ "${TASK}" != "numglue" && "${TASK}" != "fomc" ]]; then
            IMG_ARGS=(--image_root "${IMG_ROOT}")
        fi

        local CONT_ARGS=()
        if [[ ${STEP} -gt 1 && "${METHOD}" != "max_merge" ]]; then
            CONT_ARGS=(--continue_lora_path "${OUT_DIR}/$((STEP-1))-sft/task_$((STEP-1))")
        fi

        local REPLAY_ARGS=()
        if [[ "${METHOD}" == "replay" || "${METHOD}" == "der" ]]; then
            if [[ ${STEP} -gt 1 ]]; then
                local PRIOR=""
                for (( j=0; j<STEP-1; j++ )); do
                    PRIOR+="${DATA_ROOT}/${TASKS[$j]}.json:"
                done
                REPLAY_ARGS=(--replay_paths "${PRIOR%:}" --replay_ratio "${REPLAY_RATIO}")
            fi
        fi

        local IS_LAST=False
        [[ ${STEP} -eq ${NUM_TASKS} ]] && IS_LAST=True

        local SMOKE_LOG="${OUT_DIR}/${STEP}-${TASK}.log"
        echo "  → smoke step ${STEP}/${NUM_TASKS} ${METHOD}/${TASK} → ${SMOKE_LOG}"

        accelerate launch \
            --num_processes 8 \
            --config_file "${ROOT_REPO}/src/open-r1-multimodal/configs/zero1_real.yaml" \
            "${ROOT_REPO}/src/src/open_r1/sft.py" \
            --model_name_or_path "${BASE_MODEL}" \
            --dataset_name "${DATA_PATH}" \
            "${IMG_ARGS[@]}" \
            --output_dir "${STEP_DIR}" \
            --learning_rate 2e-5 \
            --num_train_epochs "${NUM_EPOCHS}" \
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
            --cur_task "${TASK}" \
            --task_k ${STEP} \
            --order order3 \
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
            --infer_max_new_tokens 64 \
            --infer_sample_num "${SAMPLE_LIMIT}" \
            > "${SMOKE_LOG}" 2>&1

        local RC=$?
        if [[ ${RC} -ne 0 ]]; then
            METHOD_OK=false
            echo "    ✗ exit=${RC}" | tee -a "${REPORT}"
            tail -8 "${SMOKE_LOG}" | sed 's/^/      /' | tee -a "${REPORT}"
            break
        fi

        STEP=$((STEP+1))
    done

    # ===== verify method-specific signatures =====
    local SIG_OK=true
    local SIG_NOTE=""

    # all methods: prediction file must exist for at least task 1
    if [[ ! -s "${OUT_DIR}/predictions/1-art.json" ]]; then
        SIG_OK=false
        SIG_NOTE="${SIG_NOTE} no_predictions_1_art"
    fi

    case "${METHOD}" in
        ewc|mas)
            # at task 2 must see "[EWC] sup=" or "[MAS] sup=" reg log
            if ! grep -q "\[EWC\] sup=" "${OUT_DIR}/2-OCR.log" 2>/dev/null; then
                SIG_OK=false
                SIG_NOTE="${SIG_NOTE} no_reg_loss_task2"
            fi
            ;;
        lwf)
            if ! grep -q "\[LwF\] sup=" "${OUT_DIR}/2-OCR.log" 2>/dev/null; then
                SIG_OK=false
                SIG_NOTE="${SIG_NOTE} no_distill_task2"
            fi
            ;;
        der)
            if ! grep -q "\[DER\] sup=" "${OUT_DIR}/2-OCR.log" 2>/dev/null; then
                SIG_OK=false
                SIG_NOTE="${SIG_NOTE} no_distill_mse_task2"
            fi
            ;;
        replay)
            if ! grep -q "\[Replay\]" "${OUT_DIR}/2-OCR.log" 2>/dev/null; then
                SIG_OK=false
                SIG_NOTE="${SIG_NOTE} no_replay_log_task2"
            fi
            ;;
        freeze)
            # trainable_params should be < full LoRA (148M); look for half
            if ! grep -q "Freeze:" "${OUT_DIR}/1-art.log" 2>/dev/null; then
                SIG_OK=false
                SIG_NOTE="${SIG_NOTE} no_freeze_log"
            fi
            ;;
        l2p)
            if ! grep -q "\[L2P\] task" "${OUT_DIR}/1-art.log" 2>/dev/null; then
                SIG_OK=false
                SIG_NOTE="${SIG_NOTE} no_l2p_log"
            fi
            ;;
        max_merge)
            if ! grep -q "\[MaxMerge\] saved snapshot" "${OUT_DIR}/1-art.log" 2>/dev/null; then
                SIG_OK=false
                SIG_NOTE="${SIG_NOTE} no_snapshot_log"
            fi
            ;;
        none)
            : # baseline — no special signature
            ;;
    esac

    if ${METHOD_OK} && ${SIG_OK}; then
        echo "  ✅ ${METHOD}: PASS" | tee -a "${REPORT}"
    else
        echo "  ❌ ${METHOD}: FAIL (sig=${SIG_OK} train=${METHOD_OK}) ${SIG_NOTE}" | tee -a "${REPORT}"
    fi
}

# =========== RUN ALL METHODS ===========
for M in "${METHODS[@]}"; do
    echo "" | tee -a "${REPORT}"
    echo "===== smoke method: ${M} =====" | tee -a "${REPORT}"
    SECONDS=0
    run_method "${M}"
    echo "  duration=${SECONDS}s" | tee -a "${REPORT}"
done

echo "" | tee -a "${REPORT}"
echo "============ SUMMARY ============" | tee -a "${REPORT}"
grep -E "^  ✅|^  ❌" "${REPORT}" | tee -a "${REPORT}"
echo "" | tee -a "${REPORT}"
echo "Full report: ${REPORT}"
