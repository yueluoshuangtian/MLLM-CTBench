export WANDB_MODE=offline
export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0
RUN_NAME="Qwen2.5-VL-3B_sft-numglue"
export LOG_PATH="src/open-r1-multimodal_sft/debug_log_$RUN_NAME.txt"
ALL_PATH="./src/open-r1-multimodal_sft/log_$RUN_NAME.txt"
# export LOG_PATH="src/sft_test/debug_log_$RUN_NAME.txt"
# ALL_PATH="./src/sft_test/log_$RUN_NAME.txt"
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
touch "./src/open-r1-multimodal/log_$RUN_NAME.txt"

accelerate launch --config_file=src/open-r1-multimodal/configs/zero3.yaml src/src/open_r1/sft.py \
    --model_name_or_path /public/home/houzhiyan/Qwen2.5-VL-3B-Instruct \
    --dataset_name /public/home/houzhiyan/VLM-R1-main/RLCL_datasets/train/numglue.json \
    --image_root /home/houzhiyan/dataset/images \
    --learning_rate 1.0e-5 \
    --num_train_epochs 5 \
    --packing \
    --max_length 4096 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --output_dir output_dirSFT/$RUN_NAME \
    --is_vqa false \
    --cur_task numglue \
    >> $ALL_PATH 2>&1
    # --output_dir output_dirSFT/$RUN_NAME \
         
        # --eval_strategy steps \
    # --eval_steps 100 \
