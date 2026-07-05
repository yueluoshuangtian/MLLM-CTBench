#!/usr/bin/env bash
# check_runs.sh — RFT / JointFT 两个 worker 运行进度 + 最终耗时汇总
#
# 用法：
#   bash /mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/qwen_cl/check_runs.sh
#
# 看什么：
#   - 每个 mode 当前在哪个 task / 哪个阶段
#   - 已完成的 task 各用了多久
#   - 训练完后的 timing_{rft,jointft}.json
QWEN_ROOT=/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/qwen_cl

show_one() {
  local MODE=$1
  local ROOT=$2
  local TIMING=$3
  echo "========================================"
  echo "  ${MODE}"
  echo "========================================"

  if [[ -f "${TIMING}" ]]; then
    echo "[DONE] timing_${MODE,,}.json:"
    cat "${TIMING}"
    echo
    return
  fi

  if [[ ! -d "${ROOT}" ]]; then
    echo "[NOT STARTED] output dir absent: ${ROOT}"
    echo
    return
  fi

  echo "[output dir]   ${ROOT}"
  echo "[mtime]        $(stat -c '%y' "${ROOT}" 2>/dev/null | cut -d. -f1)"

  local LOG="${ROOT}/debug_Qwen2.5-VL-3B_grpo_r64.txt"
  if [[ -f "${LOG}" ]]; then
    echo "[log size]     $(du -h "${LOG}" | cut -f1)"
    echo "--- last 3 step markers ---"
    grep -E "^===== Step|^========== Step" "${LOG}" | tail -3
    echo "--- last 3 train_runtime ---"
    grep "'train_runtime'" "${LOG}" 2>/dev/null | tail -3
    echo "--- last 5 lines ---"
    tail -5 "${LOG}"
  else
    echo "[log absent: training maybe still in early setup]"
  fi

  local PER="${ROOT}/_per_task_seconds.tsv"
  if [[ -s "${PER}" ]]; then
    echo "--- finished tasks ---"
    awk -F'\t' '{printf "  %-10s %5ds  (%.2fh)\n", $1, $2, $2/3600}' "${PER}"
    local TOTAL=$(awk -F'\t' '{s+=$2} END{print s}' "${PER}")
    echo "  partial total: ${TOTAL}s ($(awk "BEGIN{printf \"%.2f\", ${TOTAL}/3600}")h)"
  fi
  echo
}

show_one "RFT"      "${QWEN_ROOT}/grpo_cl_output/seqft/Qwen2.5-VL-3B_grpo_r64"      "${QWEN_ROOT}/timing_rft.json"
show_one "JointFT"  "${QWEN_ROOT}/sft+grpo_cl_output/seqft/Qwen2.5-VL-3B_grpo_r64"  "${QWEN_ROOT}/timing_jointft.json"

# 如果两个都完了，出一个对比
if [[ -f "${QWEN_ROOT}/timing_rft.json" && -f "${QWEN_ROOT}/timing_jointft.json" ]]; then
  echo "========================================"
  echo "  对比汇总"
  echo "========================================"
  /mnt/cxzx/workspace/data_transfer/houzhiyan/miniconda3/envs/qwen_cl/bin/python - <<'PY'
import json
r = json.load(open("/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/qwen_cl/timing_rft.json"))
j = json.load(open("/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/qwen_cl/timing_jointft.json"))
print(f"{'mode':<12} {'total_h':>8} {'start':>20} {'end':>20}")
for x in (r, j):
    print(f"{x['mode']:<12} {x['total_hours']:>8.3f} {x['start_ts']:>20} {x['end_ts']:>20}")
print()
print("per-task seconds:")
tasks = list(r["per_task_seconds"].keys())
print(f"  {'task':<10} {'RFT_s':>8} {'Joint_s':>8} {'ratio':>8}")
for t in tasks:
    rs = r["per_task_seconds"].get(t, 0)
    js = j["per_task_seconds"].get(t, 0)
    ratio = (js / rs) if rs else float("nan")
    print(f"  {t:<10} {rs:>8} {js:>8} {ratio:>8.2f}x")
PY
fi
