export CUDA_VISIBLE_DEVICES=1
cd /public/home/houzhiyan/VLM-R1-main
EVAL_NAME="ori_test"
#test_tasks:art、fomc、math、medical、numglue、object、ocr、science
python ./src/eval/test_rec_baseline.py \
    --model_path /public/home/houzhiyan/VLM-R1-main/cl_output/seqft/Qwen2.5-VL-3B_seqft_0/max_merge \
    --bs 32 \
    --output_dir /public/home/houzhiyan/VLM-R1-main/RLCL_datasets/$EVAL_NAME \
    --image_root /home/houzhiyan/dataset/images \
    --data_root /public/home/houzhiyan/VLM-R1-main/RLCL_datasets/test \
    --test_datasets_task math \
    --sample_num 1200 \
    --max_new_tokens 1024 \
