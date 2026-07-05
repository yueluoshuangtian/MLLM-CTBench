#!/bin/bash
# 串行跑 9 个方法的 smoke test, 完成后立即重启 occupy_gpu.
# 输出汇总到 /tmp/smoke_summary.txt.
set -u
REPO=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/TMM/CLMM
OCCUPY=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/start_occupy.sh
SUMMARY=/tmp/smoke_summary.txt
: > "$SUMMARY"

cd "$REPO"

methods=(none ewc mas lwf der freeze replay l2p max_merge)
for m in "${methods[@]}"; do
    echo "================ $m start $(date -Iseconds) ================" | tee -a "$SUMMARY"
    rm -rf "$REPO/checkpoints/smoke_test/${m}_"*
    LOG=/tmp/smoke_${m}.log
    bash "$REPO/scripts/llava_v1_5/smoke_test.sh" "$m" > "$LOG" 2>&1
    rc=$?
    # 真实成功判据: 退出码 0 + log 含 acc_matrix numpy 块 ('[[ ' 行) + 0 Traceback
    tb_count=$(grep -c "^Traceback (most recent call last):" "$LOG" || true)
    has_acc=$(grep -cE "^\[\[" "$LOG" || true)
    if [ $rc -eq 0 ] && [ "$tb_count" -eq 0 ] && [ "$has_acc" -ge 1 ]; then
        echo "[OK] $m" | tee -a "$SUMMARY"
    else
        echo "[FAIL] $m (rc=$rc, traceback=$tb_count, acc_matrix=$has_acc)" | tee -a "$SUMMARY"
    fi
    echo "--- tail $LOG ---" | tee -a "$SUMMARY"
    tail -25 "$LOG" | tee -a "$SUMMARY"
    echo "" | tee -a "$SUMMARY"
done

echo "================ smoke loop done $(date -Iseconds) ================" | tee -a "$SUMMARY"

# smoke 跑完立刻起 occupy
echo "starting occupy_gpu ..." | tee -a "$SUMMARY"
bash "$OCCUPY" >> "$SUMMARY" 2>&1 || echo "[WARN] occupy failed" | tee -a "$SUMMARY"
