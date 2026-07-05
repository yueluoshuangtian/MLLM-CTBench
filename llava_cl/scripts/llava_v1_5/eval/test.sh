#!/bin/bash

model_name="ep-0flickr30k-1vizwiz-2textvqa"

#### Flickr30k
#dataset="flickr30k"
#echo "Testing on $dataset"
#python -m llava.eval.model_vqa_loader \
#    --model-base liuhaotian/llava-v1.5-7b \
#    --model-path checkpoints/$model_name \
#    --question-file ./playground/data/eval/$dataset/llava_${dataset}_val.jsonl \
#    --image-folder ./playground/data/flickr30k \
#    --answers-file ./playground/data/eval/$dataset/answers/$model_name.jsonl \
#    --temperature 0 \
#    --conv-mode vicuna_v1
#
#python -m llava.eval.eval_caption \
#    --annotation-file ./playground/data/eval/$dataset/gt_${dataset}_val.json \
#    --result-file ./playground/data/eval/$dataset/answers/$model_name.jsonl \
#    --dataset $dataset

### TextVQA
dataset="textvqa"
echo "Testing on $dataset"
python -m llava.eval.model_vqa_loader \
    --model-base liuhaotian/llava-v1.5-7b \
    --model-path checkpoints/$model_name \
    --question-file ./playground/data/eval/$dataset/llava_${dataset}_val_v051_ocr.jsonl \
    --image-folder ./playground/data/textvqa/train_images \
    --answers-file ./playground/data/eval/$dataset/answers/$model_name.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
#
python -m llava.eval.eval_vqa \
    --annotation-file ./playground/data/eval/$dataset/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/$dataset/answers/$model_name.jsonl \
    --dataset $dataset

#### VizWiz
#dataset="vizwiz"
#echo "Testing on $dataset"
#python -m llava.eval.model_vqa_loader \
#    --model-base liuhaotian/llava-v1.5-7b \
#    --model-path checkpoints/$model_name \
#    --question-file ./playground/data/eval/$dataset/llava_${dataset}_val.jsonl \
#    --image-folder ./playground/data/vizwiz/images \
#    --answers-file ./playground/data/eval/$dataset/answers/$model_name.jsonl \
#    --temperature 0 \
#    --conv-mode vicuna_v1
#
#python -m llava.eval.eval_vqa \
#    --annotation-file ./playground/data/eval/$dataset/gt_${dataset}_val.json \
#    --result-file ./playground/data/eval/$dataset/answers/$model_name.jsonl \
#    --dataset $dataset

#### OKVQA
#dataset="okvqa"
#echo "Testing on $dataset"
#python -m llava.eval.model_vqa_loader \
#    --model-base liuhaotian/llava-v1.5-7b \
#    --model-path checkpoints/$model_name \
#    --question-file ./playground/data/eval/$dataset/llava_${dataset}_val.jsonl \
#    --image-folder ./playground/data/coco \
#    --answers-file ./playground/data/eval/$dataset/answers/$model_name.jsonl \
#    --temperature 0 \
#    --conv-mode vicuna_v1
#
#python -m llava.eval.eval_vqa \
#    --annotation-file ./playground/data/eval/$dataset/gt_${dataset}_val.json \
#    --result-file ./playground/data/eval/$dataset/answers/$model_name.jsonl \
#    --dataset $dataset
#
#### AOKVQA
#dataset="aokvqa"
#echo "Testing on $dataset"
#python -m llava.eval.model_vqa_loader \
#    --model-base liuhaotian/llava-v1.5-7b \
#    --model-path checkpoints/$model_name \
#    --question-file ./playground/data/eval/$dataset/llava_${dataset}_val.jsonl \
#    --image-folder ./playground/data/coco \
#    --answers-file ./playground/data/eval/$dataset/answers/$model_name.jsonl \
#    --temperature 0 \
#    --conv-mode vicuna_v1
#
#python -m llava.eval.eval_vqa \
#    --annotation-file ./playground/data/eval/$dataset/gt_${dataset}_val.json \
#    --result-file ./playground/data/eval/$dataset/answers/$model_name.jsonl \
#    --dataset $dataset
#
#### TextCaps
#dataset="textcaps"
#echo "Testing on $dataset"
#python -m llava.eval.model_vqa_loader \
#    --model-base liuhaotian/llava-v1.5-7b \
#    --model-path checkpoints/$model_name \
#    --question-file ./playground/data/eval/$dataset/llava_${dataset}_val.jsonl \
#    --image-folder ./playground/data/textvqa \
#    --answers-file ./playground/data/eval/$dataset/answers/$model_name.jsonl \
#    --temperature 0 \
#    --conv-mode vicuna_v1
#
#python -m llava.eval.eval_caption \
#    --annotation-file ./playground/data/eval/$dataset/gt_${dataset}_val.json \
#    --result-file ./playground/data/eval/$dataset/answers/$model_name.jsonl \
#    --dataset $dataset
#
#### VQAv2
#dataset="vqav2"
#echo "Testing on $dataset"
#python -m llava.eval.model_vqa_loader \
#    --model-base liuhaotian/llava-v1.5-7b \
#    --model-path checkpoints/$model_name \
#    --question-file ./playground/data/eval/$dataset/llava_${dataset}_val.jsonl \
#    --image-folder ./playground/data/coco/ \
#    --answers-file ./playground/data/eval/$dataset/answers/$model_name.jsonl \
#    --temperature 0 \
#    --conv-mode vicuna_v1
#
#python -m llava.eval.eval_vqa \
#    --annotation-file ./playground/data/eval/$dataset/gt_${dataset}_val.json \
#    --result-file ./playground/data/eval/$dataset/answers/$model_name.jsonl \
#    --dataset $dataset
#
#### GQA
#dataset="gqa"
#echo "Testing on $dataset"
#python -m llava.eval.model_vqa_loader \
#    --model-base liuhaotian/llava-v1.5-7b \
#    --model-path checkpoints/$model_name \
#    --question-file ./playground/data/eval/$dataset/llava_${dataset}_testdev_balanced.jsonl \
#    --image-folder ./playground/data/gqa/images \
#    --answers-file ./playground/data/eval/$dataset/answers/$model_name.jsonl \
#    --temperature 0 \
#    --conv-mode vicuna_v1
#
#python -m llava.eval.eval_vqa \
#    --annotation-file ./playground/data/eval/$dataset/gt_${dataset}_testdev_balanced.json \
#    --result-file ./playground/data/eval/$dataset/answers/$model_name.jsonl \
#    --dataset $dataset
#
#### OCRVQA
#dataset="ocrvqa"
#echo "Testing on $dataset"
#python -m llava.eval.model_vqa_loader \
#    --model-base liuhaotian/llava-v1.5-7b \
#    --model-path checkpoints/$model_name \
#    --question-file ./playground/data/eval/$dataset/llava_${dataset}_val.jsonl \
#    --image-folder ./playground/data/ocrvqa/images \
#    --answers-file ./playground/data/eval/$dataset/answers/$model_name.jsonl \
#    --temperature 0 \
#    --conv-mode vicuna_v1
#
#python -m llava.eval.eval_vqa \
#    --annotation-file ./playground/data/eval/$dataset/gt_${dataset}_val.json \
#    --result-file ./playground/data/eval/$dataset/answers/$model_name.jsonl \
#    --dataset $dataset
