#!/bin/bash
# 单卡 + 极小子集 (train_ratio=0.001) 跑 2 个任务, 验证训练 + 评估链路.
# Usage: bash scripts/llava_v1_5/smoke_test.sh <method>
set -e
set -o pipefail
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=False

# 激活 llava conda env (deepspeed/torch 都在这里面)
source /mnt/cxzx/workspace/data_transfer/houzhiyan/miniconda3/etc/profile.d/conda.sh
conda activate llava

# 离线模式 (本地权重 + 数据, 无需 HF)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 让 DeepSpeed 接受 CUDA 12.4 / torch cu121 的小版本差异
export DS_SKIP_CUDA_CHECK=1

REPO=/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM
MODEL_DIR=/mnt/cxzx/workspace/data_transfer/houzhiyan/models
IMG_DIR=/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark
DATA_DIR=/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/llava_train

export PYTHONPATH=$REPO:$PYTHONPATH
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY

method=${1:-none}
tasks="numglue,art"
output_dir="$REPO/checkpoints/smoke_test/${method}_$(date +%s)"
mkdir -p "$output_dir/predictions"
port=$(shuf -i25000-30000 -n1)
echo "[$(date)] smoke test method=$method → $output_dir"

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
deepspeed --include=localhost:$GPUS --master_port $port $REPO/llava/train/train_cl.py \
    --deepspeed $REPO/scripts/zero2.json \
    --model_name_or_path $MODEL_DIR/llava-v1.5-7b \
    --vision_tower $MODEL_DIR/clip-vit-large-patch14-336 \
    --version v1 \
    --tasks $tasks \
    --initial_tasks $tasks \
    --data_dir $DATA_DIR \
    --image_folder $IMG_DIR \
    --train_ratio 0.001 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy no --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine \
    --logging_steps 1 --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 --lazy_preprocess True \
    --report_to none \
    --cl $method --ewc_lbd 1e6 --lwf_lbd 0.2 --replay_ratio 0.0 \
    --max_new_tokens 32 \
    --eval_max_samples 30 \
    --optim adamw_bnb_8bit \
    --seed 42 \
    2>&1 | tee $output_dir/log.txt
