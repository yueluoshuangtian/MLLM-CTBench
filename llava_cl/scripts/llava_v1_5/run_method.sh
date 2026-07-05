#!/bin/bash
# 统一 launcher：跑论文里的 8 个 CL 方法 (LLaVA-1.5-7B full-FT 路径)
# Usage:
#   bash scripts/llava_v1_5/run_method.sh <method> [<hyperparam>] [<order>]
# method ∈ { none ewc mas lwf der freeze replay max_merge l2p }
# order  ∈ { order1, order2 } (默认 order1)
# 必备路径:
#   - $MODEL_DIR/llava-v1.5-7b      (需手动下载: hf.co/liuhaotian/llava-v1.5-7b)
#   - $MODEL_DIR/clip-vit-large-patch14-336 (需手动下载: hf.co/openai/clip-vit-large-patch14-336)

set -e
set -o pipefail
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=False
# CUDA 12.4 vs torch cu121, deepspeed JIT 编译时要这个
export DS_SKIP_CUDA_CHECK=1
# 离线
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY

# 激活 conda env (deepspeed/torch 都在 llava env 里), 让脚本自包含
source /mnt/cxzx/workspace/data_transfer/houzhiyan/miniconda3/etc/profile.d/conda.sh
conda activate llava

REPO=/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM
MODEL_DIR=/mnt/cxzx/workspace/data_transfer/houzhiyan/models
IMG_DIR=/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark
DATA_DIR=/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/llava_train

export PYTHONPATH=$REPO:$PYTHONPATH

method=${1:?missing method}
hyperparam=${2:-default}
order=${3:-order1}

case $order in
  order1) tasks="numglue,art,math,fomc,medical,OCR,science" ;;
  order2) tasks="science,OCR,medical,fomc,math,art,numglue" ;;
  # order3: difficulty-curriculum (易→难), 测 stability-plasticity tradeoff.
  # 详见 PROGRESS.md §order3.
  order3) tasks="art,OCR,fomc,science,numglue,math,medical" ;;
  *)      echo "unknown order: $order"; exit 1 ;;
esac
initial_tasks="numglue,art,math,fomc,medical,OCR,science"

# 各方法默认超参 (来自论文/原代码)
ewc_lbd=1e8
lwf_lbd=0.2
replay_ratio=0
extra_args=""

# hyperparam=${2:-default} 在第二个 arg 没传时落到字面 "default", 下游 ${hyperparam:-X}
# 不会走 fallback (非空就用原值, "default" 会原样传给 python 撞 argparse float 解析).
# 这里显式判 "default" 字面值, 才能在 `bash run_method.sh mas default order3` 时拿到真实默认值.
case $method in
  none|seqft)         method=none ;;
  ewc)                [ "$hyperparam" = "default" ] && ewc_lbd=1e8 || ewc_lbd=$hyperparam ;;
  mas)                [ "$hyperparam" = "default" ] && ewc_lbd=1e8 || ewc_lbd=$hyperparam ;;
  lwf)                [ "$hyperparam" = "default" ] && lwf_lbd=0.2 || lwf_lbd=$hyperparam ;;
  der)                if [ "$hyperparam" = "default" ]; then lwf_lbd=0.5; replay_ratio=0.05; else lwf_lbd=$hyperparam; replay_ratio=0.05; fi ;;
  replay)             [ "$hyperparam" = "default" ] && replay_ratio=0.2 || replay_ratio=$hyperparam ;;
  replay_fixed)       { [ "$hyperparam" = "default" ] && replay_ratio=0.2 || replay_ratio=$hyperparam; }; extra_args="--non_duplicates False" ;;
  freeze)             ;;
  max_merge)          ;;
  l2p)                ;;
  *)                  echo "unknown method: $method"; exit 1 ;;
esac

output_dir="$REPO/checkpoints/llava-v1.5-7b/${order}/${method}_${hyperparam}"
mkdir -p "$output_dir/predictions"

LR=2e-5
if [[ $method == eproj ]]; then LR=1e-3; fi

port=$(shuf -i25000-30000 -n1)
echo "[$(date)] launching method=$method order=$order hyperparam=$hyperparam"

deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port $port $REPO/llava/train/train_cl.py \
    --deepspeed $REPO/scripts/zero2.json \
    --model_name_or_path $MODEL_DIR/llava-v1.5-7b \
    --vision_tower $MODEL_DIR/clip-vit-large-patch14-336 \
    --version v1 \
    --tasks $tasks \
    --initial_tasks $initial_tasks \
    --data_dir $DATA_DIR \
    --image_folder $IMG_DIR \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy no \
    --save_strategy steps --save_steps 50000 --save_total_limit 1 \
    --learning_rate $LR \
    --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine \
    --logging_steps 5 --tf32 True \
    --model_max_length 2048 \
    --max_new_tokens 2048 \
    --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True \
    --report_to none \
    --cl $method --ewc_lbd $ewc_lbd --lwf_lbd $lwf_lbd \
    --replay_ratio $replay_ratio \
    --optim adamw_bnb_8bit \
    --seed 42 \
    $extra_args \
    2>&1 | tee -a $output_dir/log.txt
