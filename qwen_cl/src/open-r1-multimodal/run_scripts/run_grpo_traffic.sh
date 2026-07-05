cd src/open-r1-multimodal

export DEBUG_MODE="true"

export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$PYTHONPATH:/public/home/houzhiyan/VLM-R1-main/src/open-r1-multimodal/src/"
RUN_NAME="Qwen2.5-VL-3B-GRPO-MAPLM"
export LOG_PATH="./debug_log_$RUN_NAME.txt"
export NCCL_DEBUG=INFO
torchrun --nproc_per_node=8 src/open_r1/grpo_traffic.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path /mnt/train_data/xjdu/model/Qwen2.5-VL-3B-Instruct \
    --dataset_name data_config/maplm.yaml \
    --image_root /mnt/train_data/xjdu/MAPLM-V2/image_concatenation \
    --max_prompt_length 1024 \
    --num_generations 4 \
    --per_device_train_batch_size 144 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true
