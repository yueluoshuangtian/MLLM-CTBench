export WANDB_MODE=offline
export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=0
seqft_tasks="numglue,art,math,fomc,medical,OCR,science"
seqftback_tasks="science,OCR,medical,fomc,math,art,numglue"

method=$1
hyperparam=$2
# export CUDA_VISIBLE_DEVICES=0
RUN_NAME="Qwen2.5-VL-3B_seqft_test"
export LOG_PATH="cl_output/debug_log_$RUN_NAME.txt"
ALL_PATH="./cl_output/$method/log_$RUN_NAME.txt"
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
touch "./cl_output/log_$RUN_NAME.txt"



retVal=1
while [ $retVal -ne 0 ];
## this loop is a compromise for OOM error
do
accelerate launch --config_file=src/open-r1-multimodal/configs/zero3_test.yaml src/src/open_r1/sft_CL.py \
    --model_name_or_path /public/home/houzhiyan/Qwen2.5-VL-3B-Instruct \
    --image_root /home/houzhiyan/dataset/images \
    --dataset_name /public/home/houzhiyan/VLM-R1-main/RLCL_datasets/train_for_test \
    --learning_rate 1.0e-5 \
    --num_train_epochs 3 \
    --packing \
    --tasks $seqft_tasks \
    --max_seq_length 4096 \
    --per_device_train_batch_size 5 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --bf16 True \
    --logging_steps 5 \
    --cl_method $method \
    --cl_method_alptha $hyperparam \
    --output_dir cl_output/$method/$RUN_NAME \
    # >> $ALL_PATH 2>&1

retVal=$?

done