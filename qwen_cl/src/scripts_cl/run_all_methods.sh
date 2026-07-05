#!/bin/bash
# run_all_methods.sh — sequential launch of all 9 CL methods on order3 full data.
# Each method ~5-6h, total ~50-60h.
#
# Usage:
#   bash run_all_methods.sh [TRAIN_RATIO] [NUM_EPOCHS]
#   default: 1.0 3
#
# Skips a method if its OUTPUT_ROOT exists and contains `last_predictions/`
# (already completed). Re-launching is safe; it picks up after completed methods.

set -u
TRAIN_RATIO=${1:-1.0}
NUM_EPOCHS=${2:-3}

ROOT_REPO=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/TMM/qwen_cl
OUTPUT_BASE=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/output/qwen_cl/order3

METHODS=(none ewc mas lwf freeze replay der l2p max_merge)

# Disable occupy while training; restore at end.
touch /mnt/cxzx/share/workspace/data_transfer/houzhiyan/gpu_occupy/DISABLE_WATCHDOG
bash /mnt/cxzx/share/workspace/data_transfer/houzhiyan/gpu_occupy/stop_occupy.sh 2>&1 | tail -2 || true

TOP_LOG=/tmp/run_all_methods_$(date +%Y%m%d_%H%M%S).log
echo "=== run_all_methods.sh launched $(date) ===" | tee -a "${TOP_LOG}"
echo "TRAIN_RATIO=${TRAIN_RATIO} NUM_EPOCHS=${NUM_EPOCHS}" | tee -a "${TOP_LOG}"

for M in "${METHODS[@]}"; do
    OUT_DIR="${OUTPUT_BASE}/${M}_tr${TRAIN_RATIO}"
    LAST_PRED="${OUT_DIR}/last_predictions"
    if [[ -d "${LAST_PRED}" ]] && [[ "$(ls ${LAST_PRED} 2>/dev/null | wc -l)" -gt 0 ]]; then
        echo "[SKIP] ${M}: last_predictions exists in ${LAST_PRED}" | tee -a "${TOP_LOG}"
        continue
    fi

    # Clean any partial state for retry
    rm -rf "${OUT_DIR}"

    echo "=== [START] ${M} @ $(date) ===" | tee -a "${TOP_LOG}"
    bash "${ROOT_REPO}/src/scripts_cl/run_cl_order3.sh" "${M}" "${TRAIN_RATIO}" "${NUM_EPOCHS}" 2>&1 | tee -a "${TOP_LOG}"
    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        echo "[FAIL] ${M} exited with ${rc}" | tee -a "${TOP_LOG}"
    else
        echo "=== [DONE] ${M} @ $(date) ===" | tee -a "${TOP_LOG}"
    fi
done

# Restore occupy at end
rm -f /mnt/cxzx/share/workspace/data_transfer/houzhiyan/gpu_occupy/DISABLE_WATCHDOG
echo "[run_all_methods] DISABLE_WATCHDOG removed; occupy will resume after 10 min idle" | tee -a "${TOP_LOG}"

echo "=== ALL METHODS DONE $(date) ===" | tee -a "${TOP_LOG}"
echo ""
echo "Next: bash eval all"
for M in "${METHODS[@]}"; do
    OUT_DIR="${OUTPUT_BASE}/${M}_tr${TRAIN_RATIO}"
    echo "  python ${ROOT_REPO}/src/src/open_r1/eval_matrix.py ${OUT_DIR} --order order3"
done | tee -a "${TOP_LOG}"
