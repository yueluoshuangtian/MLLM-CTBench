
export WANDB_MODE=offline
export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0

RUN_NAME="Qwen2.5-VL-3B-GRPO-OCR_lora_freeze_vit"
export LOG_PATH="src/open-r1-multimodal_GRPO_lora_freeze_vit/debug_log_$RUN_NAME.txt"
rm $LOG_PATH
export LOG_PATH="src/open-r1-multimodal_GRPO_lora_freeze_vit/debug_log_$RUN_NAME.txt"
ALL_PATH="./src/open-r1-multimodal_GRPO_lora_freeze_vit/log_$RUN_NAME.txt"
# Check if the directory exists
DIR_PATH=$(dirname "$LOG_PATH")
if [ ! -d "$DIR_PATH" ]; then
  # Create the directory if it doesn't exist
  mkdir -p "$DIR_PATH"
fi

# Check if the file exists
if [ ! -f "$LOG_PATH" ]; then
  # Create the file if it doesn't exist
  touch "$LOG_PATH"
  echo "Created new log file: $LOG_PATH"
else
  echo "Log file already exists: $LOG_PATH"
fi
touch "./src/open-r1-multimodal_GRPO_lora_freeze_vit/log_$RUN_NAME.txt"
# echo "开始运行模型"
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_port="15367" \
     /public/home/houzhiyan/VLM-R1-main/src/src/open_r1/grpo_rec.py \
    --deepspeed ./src/open-r1-multimodal/local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path /public/home/houzhiyan/Qwen2.5-VL-3B-Instruct \
    --dataset_name /public/home/houzhiyan/VLM-R1-main/RLCL_datasets/train/OCR.json \
    --image_root /home/houzhiyan/dataset/images \
    --max_prompt_length 2048 \
    --num_generations 4 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 5 \
    --run_name $RUN_NAME \
    --save_only_model true \
    --save_steps 200 \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true \
    >> $ALL_PATH 2>&1
        # --save_steps 100 \