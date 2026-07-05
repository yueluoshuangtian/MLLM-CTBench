#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#model_base="liuhaotian/llava-v1.5-7b"

model_path=$1
output_dir=$(dirname $model_path)
round=$(basename $model_path)
output_file=$output_dir/predictions/$round-vizwiz-test.jsonl

echo "Testing on VizWiz"

if [ ! -f $output_file ]; then

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do

python -m llava.eval.model_vqa_loader \
    --model-path $model_path \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file $output_dir/predictions/$round-vizwiz-test/${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1 &
done

wait

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $output_dir/predictions/$round-vizwiz-test/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

fi

rm -r $output_dir/predictions/$round-vizwiz-test


python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file $output_file \
    --result-upload-file $output_dir/predictions/$round-vizwiz-test-upload.jsonl
