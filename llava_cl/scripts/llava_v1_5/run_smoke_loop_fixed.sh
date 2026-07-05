#!/bin/bash
# 重跑 5 个已修方法的 smoke (none ewc mas der l2p), 之前 4 个已通过的不重跑.
set -u
REPO=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/TMM/CLMM
OCCUPY=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/start_occupy.sh
SUMMARY=/tmp/smoke_summary_fixed.txt
: > "$SUMMARY"

cd "$REPO"

methods=(none ewc mas der l2p)
for m in "${methods[@]}"; do
    echo "================ $m start $(date -Iseconds) ================" | tee -a "$SUMMARY"
    rm -rf "$REPO/checkpoints/smoke_test/${m}_"*
    LOG=/tmp/smoke_${m}.log
    bash "$REPO/scripts/llava_v1_5/smoke_test.sh" "$m" > "$LOG" 2>&1
    rc=$?
    tb_count=$(grep -c "^Traceback (most recent call last):" "$LOG" || true)
    has_acc=$(grep -cE "^\[\[" "$LOG" || true)
    if [ $rc -eq 0 ] && [ "$tb_count" -eq 0 ] && [ "$has_acc" -ge 1 ]; then
        echo "[OK] $m" | tee -a "$SUMMARY"
    else
        echo "[FAIL] $m (rc=$rc, traceback=$tb_count, acc_matrix=$has_acc)" | tee -a "$SUMMARY"
    fi
    echo "--- tail $LOG (skip tqdm) ---" | tee -a "$SUMMARY"
    grep -avE "it/s\]|it/s,|^ *[0-9]+%" "$LOG" | tail -20 | tee -a "$SUMMARY"
    echo "" | tee -a "$SUMMARY"
done

echo "================ smoke loop done $(date -Iseconds) ================" | tee -a "$SUMMARY"
echo "starting occupy_gpu ..." | tee -a "$SUMMARY"
bash "$OCCUPY" >> "$SUMMARY" 2>&1 || echo "[WARN] occupy failed" | tee -a "$SUMMARY"
