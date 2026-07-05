#!/bin/bash
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=False
export PYTHONPATH=/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM:$PYTHONPATH
export $(cat .env | xargs -d '\r\n')

initial_tasks="numglue,art,math,fomc,medical,OCR,science"
# initial_tasks="science,OCR,medical,fomc,math,art,numglue"

order=$1          # order1 / order2 / order3
method=$2         # moelora / eproj / ...
expert_num=$3     # 比如 2 或 4
gate_rank=$4      # 比如 16
orth_lambda=$5
# B_reg=$6

eproj_key_lr=5e-3
ewc_lbd=1e8
replay_ratio=0

if [[ $order == order1 ]]; then
    tasks="numglue,art,math,fomc,medical,OCR,science"
elif [[ $order == order2 ]]; then
    tasks="science,OCR,medical,fomc,math,art,numglue"
elif [[ "$order" == order3 ]]; then
  tasks="ScienceQA,TextVQA,ImageNet,GQA,VizWiz,Grounding,VQAv2,OCRVQA"
elif [[ "$order" == order4 ]]; then
  tasks="ScienceQA,ImageNet,VizWiz,Grounding,TextVQA,GQA,VQAv2,OCRVQA"
else
    echo "Unknown order: $order"
    exit 1
fi

# ---- 输出目录：编码 expert_num / gate_rank  ----
# output_dir="./without_reasoning/llava-v1.5-7b/moelora_forget/${order}/${method}Arouter_E${expert_num}_gr${gate_rank}_${orth_lambda}_B_reg${B_reg}"
output_dir="./without_reasoning/llava-v1.5-7b/moelora_forget/${order}/${method}Arouter_E${expert_num}_gr${gate_rank}_${orth_lambda}_B"
if [[ $method == eproj ]]; then
    lr=1e-3
else
    lr=2e-5
fi

if [ ! -d "$output_dir/predictions" ]; then
    mkdir -p "$output_dir/predictions"
fi

set -o pipefail
retVal=1

# OOM 重试循环
while [ $retVal -ne 0 ];
do
    port=$(shuf -i25000-30000 -n1)
    echo "Hello, world! port=$port"

    deepspeed  --include=localhost:0,1,2,3,4,5,6,7 --master_port $port llava/train/train_cl_lora_test.py \
        --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
        --expert_num ${expert_num} --gate_rank ${gate_rank} --moelora_orth_lambda ${orth_lambda} \
        --gate_router a --moelora_reg_enabl True \
        --moelora_importance_top_p 0.4 --moelora_B_reg_scale 0.1 \
        --deepspeed ./scripts/zero2.json \
        --model_name_or_path /mnt/cxzx/workspace/data_transfer/houzhiyan/models/llava-v1.5-7b \
        --version v1 \
        --initial_tasks $initial_tasks \
        --tasks $tasks \
        --image_folder /mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark \
        --vision_tower /mnt/cxzx/workspace/data_transfer/houzhiyan/models/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir $output_dir \
        --num_train_epochs 3 \
        --per_device_train_batch_size 12 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "no" \
        --save_total_limit 2 \
        --learning_rate $lr \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 1536 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to wandb \
        --cl $method \
        --eproj_lbd 1 \
        --eproj_key_lr $eproj_key_lr \
        --seed 42 \
        --replay_ratio $replay_ratio 2>&1 | tee >(sed -r 's/\x1B\[[0-9;]*[mK]//g' >> "$output_dir/log.txt")

    retVal=$?
done

if [ -f "$output_dir/cl_states.bin" ]; then
    rm "$output_dir/cl_states.bin"
fi
