export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH=/home/houzhiyan/qwen:$PYTHONPATH
method=$1
hyperparam=$2
order=$3

RUN_NAME="Qwen2.5-VL-3B_${method}_${hyperparam}"
ALL_PATH="grpo_cl_output/$order/$RUN_NAME/eval_log_$RUN_NAME.txt"
touch "grpo_cl_output/$order/$RUN_NAME/eval_log_$RUN_NAME.txt"
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

python src/src/open_r1/evaluate_cl.py \
    --model_paths grpo_cl_output/$order/$RUN_NAME \
    --bs 30 \
    --tasks $tasks \
    --sample_num 1500 \
    --max_new_tokens 4096 \


