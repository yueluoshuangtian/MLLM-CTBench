#!/bin/bash
# 等 GPU 空出来 → 抢占启动 Machine A 全部 4 方法 (跑到全完成) → 跑完重启智能占卡.
# 解决: 另一容器占卡霸卡, 一旦释放本脚本立刻抢占训练, 避免被对方重新占走.
set -u
REPO=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/TMM/CLMM
GPU_OCCUPY=/mnt/cxzx/share/workspace/data_transfer/houzhiyan/gpu_occupy
cd "$REPO"; mkdir -p logs
LOG=logs/machine_a_subproc.log

echo "[$(date -Iseconds)] auto_run: 等 GPU 释放 (需要 gpu0 空闲 >40GB)..." | tee -a "$LOG"
# 持续放 STOP_OCCUPY + DISABLE_WATCHDOG 抑制本容器占卡, 并等任意时刻 GPU 空出
while true; do
    touch "$GPU_OCCUPY/STOP_OCCUPY" "$GPU_OCCUPY/DISABLE_WATCHDOG" 2>/dev/null
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    if [ "${free:-0}" -gt 40000 ]; then
        echo "[$(date -Iseconds)] GPU 空出 (min free=${free}MiB), 抢占启动训练!" | tee -a "$LOG"
        break
    fi
    sleep 10
done

# 跑 4 个方法 (freeze 会 resume 跳过 task1-4)
for m in freeze max_merge lwf der; do
    echo "############ METHOD $m start $(date -Iseconds) ############" | tee -a "$LOG"
    bash "$REPO/scripts/llava_v1_5/run_method_subproc.sh" "$m" default order3 2>&1 | tee -a "$LOG"
    echo "############ METHOD $m end rc=${PIPESTATUS[0]} $(date -Iseconds) ############" | tee -a "$LOG"
done
echo "############ MACHINE-A 4 METHODS ALL DONE $(date -Iseconds) ############" | tee -a "$LOG"

# 跑完后重启智能占卡 (全功能 watchdog: 自动占 + 自动让路)
rm -f "$GPU_OCCUPY/DISABLE_WATCHDOG" "$GPU_OCCUPY/STOP_OCCUPY"
bash "$GPU_OCCUPY/start.sh" >> "$LOG" 2>&1 || echo "[WARN] 智能占卡重启失败" | tee -a "$LOG"
echo "[$(date -Iseconds)] auto_run 结束, 智能占卡已重启" | tee -a "$LOG"
