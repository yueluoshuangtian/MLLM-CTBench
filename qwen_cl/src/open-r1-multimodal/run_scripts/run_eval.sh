export CUDA_VISIBLE_DEVICES=0
cd /public/home/houzhiyan/VLM-R1-main
EVAL_NAME="sft_test_checkpoint500"
#test_tasks:art、fomc、math、medical、numglue、object、ocr、science
python ./src/eval/test_rec_baseline.py \
    --model_path /public/home/houzhiyan/VLM-R1-main/output_dirSFT/Qwen2.5-VL-3B_sft-math/checkpoint-500 \
    --bs 48 \
    --output_dir /public/home/houzhiyan/VLM-R1-main/RLCL_datasets/$EVAL_NAME \
    --image_root /home/houzhiyan/dataset/images \
    --data_root /public/home/houzhiyan/VLM-R1-main/RLCL_datasets/test \
    --test_datasets_task math \
    --sample_num 1200 \
    --max_new_tokens 4096 \
