#!/bin/bash
export PYTHONPATH=/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
#task_ids = ["numglue","art","math","fomc","medical","OCR","science"]
port=$(shuf -i25001-30000 -n1)
# deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port $port
python llava/eval/model_vqa_loader_mulGPU.py \
    --task-id "art" \
    --task-name "baseline" \
    --task-order "seqft" \
    --model-path "/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/seqft/llava_art_reasoning_3epoch" \
    --answers-file "/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/checkpoints/llava-v1.5-7b" \
    --conv-mode "vicuna_v1" \
    --temperature 1 \
    --num_beams 1 \
    --max_new_tokens 4096 \
