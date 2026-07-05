#!/bin/bash
# 同时跑 der (GPU 0-3) + l2p (GPU 4-7) 验证 patch.
set -u
REPO=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/TMM/CLMM
OCCUPY=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/start_occupy.sh
SUMMARY=/tmp/smoke_summary_parallel.txt
: > "$SUMMARY"

cd "$REPO"

# 清旧 checkpoint
rm -rf "$REPO/checkpoints/smoke_test/der_"* "$REPO/checkpoints/smoke_test/l2p_"*

echo "================ parallel start $(date -Iseconds) ================" | tee -a "$SUMMARY"

GPUS=0,1,2,3 bash "$REPO/scripts/llava_v1_5/smoke_test.sh" der > /tmp/smoke_der.log 2>&1 &
PID_DER=$!
echo "der launched on GPU 0-3, PID=$PID_DER" | tee -a "$SUMMARY"

GPUS=4,5,6,7 bash "$REPO/scripts/llava_v1_5/smoke_test.sh" l2p > /tmp/smoke_l2p.log 2>&1 &
PID_L2P=$!
echo "l2p launched on GPU 4-7, PID=$PID_L2P" | tee -a "$SUMMARY"

# 等两个都结束
wait $PID_DER
rc_der=$?
echo "der finished rc=$rc_der at $(date -Iseconds)" | tee -a "$SUMMARY"

wait $PID_L2P
rc_l2p=$?
echo "l2p finished rc=$rc_l2p at $(date -Iseconds)" | tee -a "$SUMMARY"

# 判定
for m in der l2p; do
    LOG=/tmp/smoke_${m}.log
    rc_var="rc_${m}"
    rc=${!rc_var}
    tb=$(grep -c "^Traceback (most recent call last):" "$LOG" || true)
    has_acc=$(grep -cE "^\[\[" "$LOG" || true)
    if [ $rc -eq 0 ] && [ "$tb" -eq 0 ] && [ "$has_acc" -ge 1 ]; then
        echo "[OK] $m" | tee -a "$SUMMARY"
    else
        echo "[FAIL] $m (rc=$rc, traceback=$tb, acc_matrix=$has_acc)" | tee -a "$SUMMARY"
    fi
done

echo "================ parallel done $(date -Iseconds) ================" | tee -a "$SUMMARY"
bash "$OCCUPY" >> "$SUMMARY" 2>&1 || echo "[WARN] occupy failed" | tee -a "$SUMMARY"
