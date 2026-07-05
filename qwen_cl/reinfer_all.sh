#!/usr/bin/env bash
# 其余 8 个方法依次 8 卡重推理(等 GPU 空闲再跑下一个)。none 已在外面单独跑。
set -uo pipefail
QWEN=/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/qwen_cl
OUTROOT=/mnt/cxzx/workspace/data_transfer/houzhiyan/output/qwen_cl/order3
source /mnt/cxzx/workspace/data_transfer/houzhiyan/miniconda3/etc/profile.d/conda.sh
conda activate qwen_cl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH="$QWEN"
export QWEN_TEST_ROOT="/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/qwen data/test"
export QWEN_IMAGE_ROOT="/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true
cd "$QWEN"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$QWEN/calib/reinfer_all.log"; }

for m in ewc mas lwf freeze replay der l2p max_merge; do
  # 等当前没有 reinfer 在跑(GPU 空)
  while pgrep -f 'reinfer.py' >/dev/null 2>&1; do sleep 30; done
  M="$OUTROOT/${m}_tr1.0"
  if [ ! -d "$M" ]; then log "SKIP $m (no dir)"; continue; fi
  rm -rf "$M/predictions_v2" "$M/last_predictions_v2"
  LG="$QWEN/calib/reinfer_${m}.log"; : > "$LG"
  log "==== 重推理 $m 开始 ===="
  torchrun --nproc_per_node=8 reinfer.py --method_root "$M" --max_new_tokens 1024 --bs 16 >> "$LG" 2>&1
  log "==== 重推理 $m 完成 (exit=$?) ===="
done
log "==== ALL_REINFER_DONE ===="
