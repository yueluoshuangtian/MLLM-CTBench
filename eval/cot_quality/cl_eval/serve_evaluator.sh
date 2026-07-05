#!/usr/bin/env bash
# 用 claw_env_q35 的 sglang 把 Qwen3.5-27B 评估器起成 OpenAI 兼容服务。
# 该 env: transformers 5.8.1（认识 qwen3_5/qwen3_vl），sglang 已装。
set -e

ENV_DIR=/mnt/cxzx/workspace/data_transfer/houzhiyan/miniconda3/envs/claw_env_q35
ENV_PY=$ENV_DIR/bin/python
export PATH=$ENV_DIR/bin:$PATH          # 让 scheduler 子进程能按裸名找到 ninja 等二进制
MODEL=/mnt/cxzx/share/bj_share/anhongyan/personal_datas/zhiyan/models/Qwen3.5-27B
PORT=${PORT:-30000}
DP=${DP:-8}                 # 数据并行副本数（全部 8 卡，27B bf16≈54G 单卡放得下）
TP=${TP:-1}                 # 每副本张量并行；27B 单卡足够，故 tp=1
GPUS=${GPUS:-0,1,2,3,4,5,6,7}

export CUDA_VISIBLE_DEVICES=$GPUS
export SGLANG_DISABLE_CUDNN_CHECK=1   # 跳过 sglang 对 torch2.9.1+cudnn 的过度保守预检（本任务不涉及 Conv3d）
echo "[serve] sglang 启动 Qwen3.5-27B  port=$PORT dp=$DP tp=$TP gpus=$GPUS"

# --served-model-name 要与 cl_config.EVAL_MODEL 一致（默认 Qwen3.5-27B）
$ENV_PY -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name Qwen3.5-27B \
  --host 127.0.0.1 --port "$PORT" \
  --data-parallel-size "$DP" \
  --tensor-parallel-size "$TP" \
  --mem-fraction-static 0.85 \
  --trust-remote-code
# 多模态/对话模板由模型目录的 chat_template.jinja 自动加载；
# 若 sglang 报 chat-template 缺失，加: --chat-template "$MODEL/chat_template.jinja"
