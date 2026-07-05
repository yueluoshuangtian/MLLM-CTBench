export CUDA_VISIBLE_DEVICES=0
cd /public/home/houzhiyan/VLM-R1-main

#test_tasks:art、fomc、math、medical、numglue、object、ocr、science
python ./src/eval/test_rec_baseline.py \
    --model_path /public/home/houzhiyan/VLM-R1-main/output_dirSFT/Qwen2.5-VL-3B_sft-science \
    --bs 32 \
    --output_dir /public/home/houzhiyan/VLM-R1-main/RLCL_datasets/RL_test \
    --image_root /home/houzhiyan/dataset/images \
    --data_root /public/home/houzhiyan/VLM-R1-main/RLCL_datasets/test \
    --test_datasets_task science \
    --sample_num 1200 \
    --max_new_tokens 1024 \
