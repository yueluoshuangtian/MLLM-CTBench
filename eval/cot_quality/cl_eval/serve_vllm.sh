#!/usr/bin/env bash
# 用 vllm_cu128 venv 起 Qwen3.5-27B 评估器（本机驱动535可跑，沿用 zhiyan/serve_qwen.sh 配方）。
# 与原配方差异：开启图片输入(image:1)，因为本任务 VQA 需要 grounding 维度。
set -e

VENV=/mnt/cxzx/share/bj_share/anhongyan/personal_datas/zhiyan/models/vllm_cu128
PY=$VENV/bin/python
export PATH=$VENV/bin:$PATH          # 让 worker 子进程找到 ninja 等
MODEL=/mnt/cxzx/share/bj_share/anhongyan/personal_datas/zhiyan/models/Qwen3.5-27B
PORT=${PORT:-30000}
TP=${TP:-4}                          # 与他们一致：每副本 TP=4
GPUS=${GPUS:-0,1,2,3}
MAXLEN=${MAXLEN:-32768}

export CUDA_VISIBLE_DEVICES=$GPUS
echo "[serve] vllm_cu128 启动 Qwen3.5-27B  port=$PORT tp=$TP gpus=$GPUS maxlen=$MAXLEN"

$PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name Qwen3.5-27B \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization 0.88 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --reasoning-parser qwen3 \
  --trust-remote-code
