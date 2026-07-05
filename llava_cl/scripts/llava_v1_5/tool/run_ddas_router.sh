#!/bin/bash
set -e

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=False
export PYTHONPATH=/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

CLIP_PATH="/mnt/cxzx/workspace/data_transfer/houzhiyan/models/clip-vit-large-patch14-336"
TASKS="numglue,art,math,fomc,medical,OCR,science"
OUT_ROOT="/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/without_reasoning"

# 例如：只跑图像模式
python llava/tool/ddas_train_eval.py \
  --clip_model_path "$CLIP_PATH" \
  --tasks "$TASKS" \
  --mode image \
  --output_root "$OUT_ROOT" \
  --ae_hidden_dim 128 \
  --ae_epochs 3 \
  --ae_lr 1e-3 \
  --batch_size_feat 64 \
  --batch_size_ae 256 \
  --num_workers 8 \
  --device cuda

# 2) 只看文本
python llava/tool/ddas_train_eval.py \
  --clip_model_path "$CLIP_PATH" \
  --tasks "$TASKS" \
  --mode text \
  --ae_hidden_dim 128 \
  --ae_epochs 3 \
  --ae_lr 1e-3 \
  --output_root "$OUT_ROOT" \
  --batch_size_feat 64 \
  --batch_size_ae 256 \
  --num_workers 8 \
  --device cuda

# 3) 图像 + 文本联合
python llava/tool/ddas_train_eval.py \
  --clip_model_path "$CLIP_PATH" \
  --tasks "$TASKS" \
  --mode both \
  --ae_hidden_dim 128 \
  --output_root "$OUT_ROOT" \
  --ae_epochs 3 \
  --ae_lr 1e-3 \
  --batch_size_feat 64 \
  --batch_size_ae 256 \
  --num_workers 8 \
  --device cuda
