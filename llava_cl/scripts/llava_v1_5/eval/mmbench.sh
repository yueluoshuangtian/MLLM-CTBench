#!/bin/bash

gpu=$3

method=$1
task=$2
model_name="llava-v1.5-7b-$method-$task"

SPLIT="mmbench_dev_20230712"
result_file="./playground/data/eval/mmbench/answers/$SPLIT/$model_name.jsonl"

export CUDA_VISIBLE_DEVICES=$gpu

#python -m llava.eval.model_vqa_mmbench \
#    --model-path checkpoints/$model_name \
#    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
#    --answers-file $result_file \
#    --single-pred-prompt \
#    --temperature 0 \
#    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $model_name
