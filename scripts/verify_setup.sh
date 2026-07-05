#!/usr/bin/env bash
# 环境/仓库自检：路径存在性 + 入口脚本语法 + 关键 python 可导入。
# 用法: bash scripts/verify_setup.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/configs/paths.env"
ok=0; bad=0
chk(){ if eval "$2" >/dev/null 2>&1; then echo "  ✓ $1"; ok=$((ok+1)); else echo "  ✗ $1"; bad=$((bad+1)); fi; }

echo "== 路径 =="
chk "LLAVA_BASE=$LLAVA_BASE"   "[ -e '$LLAVA_BASE' ]"
chk "QWEN_BASE=$QWEN_BASE"     "[ -e '$QWEN_BASE' ]"
chk "TRAIN_DIR=$TRAIN_DIR"     "[ -d '$TRAIN_DIR' ]"
chk "TEST_DIR=$TEST_DIR"       "[ -d '$TEST_DIR' ]"
chk "IMAGE_ROOT=$IMAGE_ROOT"   "[ -d '$IMAGE_ROOT' ]"

echo "== 入口脚本语法 (bash -n) =="
for s in qwen_cl/src/scripts_cl/run_cl_order3.sh qwen_cl/run_rft_worker.sh \
         qwen_cl/run_jointft_worker.sh data/download_data.sh eval/cot_quality/cl_eval/serve_vllm.sh; do
  chk "$s" "bash -n '$ROOT/$s'"
done

echo "== 关键 python 可导入 =="
chk "eval cl_config/cl_data" "cd '$ROOT/eval/cot_quality/cl_eval' && python -c 'import cl_config, cl_data'"

echo "== 汇总: 通过 $ok / 失败 $bad =="
[ "$bad" -eq 0 ] && echo "全部通过 ✅" || echo "有 $bad 项待修（多为未填模型/未下载数据，属预期）"
