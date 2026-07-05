#!/bin/bash
# 重跑 der + l2p 验证 patch.
set -u
REPO=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/TMM/CLMM
OCCUPY=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/start_occupy.sh
SUMMARY=/tmp/smoke_summary_der_l2p.txt
: > "$SUMMARY"

cd "$REPO"

for m in der l2p; do
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
done

echo "================ smoke loop done $(date -Iseconds) ================" | tee -a "$SUMMARY"
echo "starting occupy_gpu ..." | tee -a "$SUMMARY"
bash "$OCCUPY" >> "$SUMMARY" 2>&1 || echo "[WARN] occupy failed" | tee -a "$SUMMARY"
