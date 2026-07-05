#!/usr/bin/env bash
set -e

# 选择训练顺序：order1..order6
ORDER="${1:-order1}"
ADAPTER_ROOT=$2

# ======= YOU MUST SET THESE =======
BASE_MODEL="/home/houzhiyan/Qwen2.5-VL-3B-Instruct"   # 原始 base 模型
# 持续学习输出根目录（含 1,2,3... 或 1-sft...）

# 推理参数
BS=16

SAMPLE_NUM=0   # 0=全量；比如 200=抽样200
DTYPE=bf16

export CUDA_VISIBLE_DEVICES=0

python /home/houzhiyan/qwen/src/src/open_r1/evaluate_cl.py \
  --base_model "${BASE_MODEL}" \
  --adapter_root "${ADAPTER_ROOT}" \
  --order "${ORDER}" \
  --bs ${BS} \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  --sample_num ${SAMPLE_NUM} \
  --dtype ${DTYPE} \
  --attn_impl flash_attention_2 \
  --device cuda:0 \
  --do_post \
  --do_last
