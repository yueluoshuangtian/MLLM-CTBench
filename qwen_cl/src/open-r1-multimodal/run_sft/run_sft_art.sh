export WANDB_MODE=offline
export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=7
#0,1,2,3,4,5,6,
# export CUDA_VISIBLE_DEVICES=0
RUN_NAME="Qwen2.5-VL-3B_sft-art"
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

accelerate launch --config_file=src/open-r1-multimodal/configs/test_zero3.yaml --main_process_port 29505 src/src/open_r1/sft.py \
    --model_name_or_path /public/home/houzhiyan/VLM-R1-main/cl_output/seqft/Qwen2.5-VL-3B_seqft_0/max_merge \
    --dataset_name /public/home/houzhiyan/VLM-R1-main/RLCL_datasets/train/art.json \
    --image_root /home/houzhiyan/dataset/images \
    --learning_rate 1.0e-5 \
    --num_train_epochs 3 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 5 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --output_dir output/123 \
    --seed 42 \
    --cur_task art \
    # >> $ALL_PATH 2>&1
        # --eval_strategy steps \
    # --eval_steps 100 \
