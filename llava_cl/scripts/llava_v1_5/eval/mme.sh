#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export $(cat .env | xargs -d '\r\n')

model_base=$LLAVA_V1_5_7B

model_path=$1
output_dir=$(dirname $model_path)
round=$(basename $model_path)
output_file=$output_dir/predictions/$round-mme.jsonl

echo "Testing on MME"

if [ ! -f $output_file ]; then

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    if [[ $model_path == *"eproj"* ]]; then
      CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
          --model-base $model_base \
          --model-path $model_path \
          --question-file ./playground/data/eval/MME/llava_mme.jsonl \
          --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
          --answers-file $output_dir/predictions/$round-mme/${CHUNKS}_${IDX}.jsonl \
          --num-chunks $CHUNKS \
          --chunk-idx $IDX \
          --temperature 0 \
          --conv-mode vicuna_v1 &
    else
      CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
          --model-path $model_path \
          --question-file ./playground/data/eval/MME/llava_mme.jsonl \
          --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
          --answers-file $output_dir/predictions/$round-mme/${CHUNKS}_${IDX}.jsonl \
          --num-chunks $CHUNKS \
          --chunk-idx $IDX \
          --temperature 0 \
          --conv-mode vicuna_v1 &
    fi
done

wait

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $output_dir/predictions/$round-mme/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

rm -r $output_dir/predictions/$round-mme

fi

python ./playground/data/eval/MME/convert_answer_to_mme.py \
  --output_file $output_file \
  --results_dir $output_dir/predictions/$round-mme

python ./playground/data/eval/MME/eval_tool/calculation.py \
  --results_dir $output_dir/predictions/$round-mme
