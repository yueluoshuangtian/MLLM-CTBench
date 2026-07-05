export WANDB_MODE=offline
export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=1

method=$1
hyperparam=$2
order=$3

RUN_NAME="Qwen2.5-VL-3B_${method}_${hyperparam}"
export LOG_PATH="cl_output/$order/$RUN_NAME/debug_$RUN_NAME.txt"

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


seqft_tasks="numglue,art,math,fomc,medical,OCR,science"
seqftback_tasks="science,OCR,medical,fomc,math,art,numglue"
if [ "$order" == "seqft" ]; then
    tasks=$seqft_tasks
elif [ "$order" == "seqftback" ]; then
    tasks=$seqftback_tasks
else
    echo "❌ 错误：未知的任务顺序参数 '$order'"
    exit 1
fi

retVal=1
while [ $retVal -ne 0 ];
## this loop is a compromise for OOM error
do
accelerate launch --config_file=src/open-r1-multimodal/configs/zero2.yaml src/src/open_r1/sft_replay.py \
    --model_name_or_path /home/houzhiyan/Qwen2.5-VL-3B-Instruct \
    --image_root /home/houzhiyan/dataset/images \
    --dataset_name /home/houzhiyan/dataset/qwen/train_for_test \
    --learning_rate 1.0e-5 \
    --num_train_epochs 4 \
    --packing \
    --tasks $tasks \
    --save_steps 2000 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 3 \
    --gradient_checkpointing \
    --bf16 True \
    --logging_steps 5 \
    --cl_method $method \
    --replay_ratio $hyperparam \
    --output_dir cl_output/$order/$RUN_NAME \
    >> $LOG_PATH 2>&1

retVal=$?

done