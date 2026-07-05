#!/bin/bash
set -o pipefail

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=False
export PYTHONPATH=/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2

if [[ -f .env ]]; then
  export $(cat .env | xargs -d '\r\n')
fi

order="$1"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <order{order1|order2|order3}>"
  exit 1
fi

# ==== 任务顺序（保持和你原脚本一致） ====
if [[ "$order" == order1 ]]; then
  tasks="numglue,art,math,fomc,medical,OCR,science"
elif [[ "$order" == order2 ]]; then
  tasks="science,OCR,medical,fomc,math,art,numglue"
elif [[ "$order" == order3 ]]; then
  tasks="fomc,art,math,science,numglue,OCR,medical"
else
  echo "[ERROR] Unknown order: $order"
  exit 1
fi

# ==== 缓存 & 日志根目录（按你要求） ====
cache_root="/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/without_reasoning/hide_router_eval"
mkdir -p "$cache_root"

# 单独给这个 order 一个 log 目录
order_name=$(echo "$tasks" | tr ',' '_')
log_dir="${cache_root}/${order_name}"
mkdir -p "$log_dir"

# ==== CLIP 模型路径（用你本地的 clip-vit-large-patch14-336） ====
CLIP_PATH="/mnt/cxzx/workspace/data_transfer/houzhiyan/models/clip-vit-large-patch14-336"

echo "[INFO] Evaluate HiDe-LLaVA router on tasks: $tasks"
echo "[INFO] cache_root = $cache_root"
echo "[INFO] Logs will be saved to: $log_dir/log.txt"

python /mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/llava/tool/hide_router_eval.py \
  --tasks "$tasks" \
  --train_data_dir "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/train" \
  --train_image_root "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark" \
  --clip_model_path "$CLIP_PATH" \
  --alpha 0.5 \
  --beta 0.5 \
  --temperature 0.07 \
  --batch_size 32 \
  --cache_root "$cache_root" \
  2>&1 | tee "$log_dir/log.txt"


# 只画热力图
python /mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/llava/tool/analyze_hide_router.py \
  --exp_dir /mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/without_reasoning/hide_router_eval/numglue_art_math_fomc_medical_OCR_science \
  --mode heatmap

# 只看最容易混淆的 top-10
python /mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/llava/tool/analyze_hide_router.py \
  --exp_dir /mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/without_reasoning/hide_router_eval/numglue_art_math_fomc_medical_OCR_science \
  --mode pairs --topk 10



