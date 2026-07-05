#!/bin/bash
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=False
export PYTHONPATH=/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM:$PYTHONPATH
export $(cat .env | xargs -d '\r\n')

initial_tasks="numglue,art,math,fomc,medical,OCR,science"

order=$1
method=$2
hyperparam=$3
model=$4

eproj_key_lr=5e-3
ewc_lbd=1e8
replay_ratio=0

if [[ $order == order1 ]]; then
tasks="numglue,art,math,fomc,medical,OCR,science"
elif [[ $order == order2 ]]; then
tasks="science,OCR,meciacl,fomc,math,art,numglue"
else
return 1
fi

output_dir="./checkpoints/llava-v1.5-7b/${order}/${method}_${hyperparam}"

if [[ $method == eproj ]]; then
eproj_key_lr=$hyperparam
elif [[ $method == ewc ]] || [[ $method == mas ]] || [[ $method == *"tir"* ]]; then
ewc_lbd=$hyperparam
elif [[ $method == replay ]]; then
replay_ratio=$hyperparam
elif [[ $method == lota ]]; then
lota_ratio=$hyperparam
else
output_dir="./checkpoints/llava-v1.5-7b/${order}/${method}"
fi

if [[ $method == eproj ]]; then
lr=1e-3
else
lr=2e-5
fi

if [ ! -d $output_dir/predictions ]; then
mkdir -p $output_dir/predictions
cp ./checkpoints/llava-v1.5-7b/seq_0/predictions/0-* $output_dir/predictions
fi

retVal=1

while [ $retVal -ne 0 ];
## this loop is a compromise for OOM error
do

port=$(shuf -i25000-30000 -n1)

deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port $port llava/train/test_cl.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $model \
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
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --ewc_lbd $ewc_lbd \
    --eproj_lbd 1 \
    --eproj_key_lr $eproj_key_lr \
    --start_task_id 7 \
    --test_checkpoint_sequence 3 \
    --replay_ratio $replay_ratio \
# >> $output_dir/log_infer.txt 2>&1 
retVal=$?

done

if [ -f $output_dir/cl_states.bin ]; then
rm $output_dir/cl_states.bin
fi

echo "art is done..."