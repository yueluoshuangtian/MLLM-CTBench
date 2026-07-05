export CUDA_VISIBLE_DEVICES=0
cd /public/home/houzhiyan/VLM-R1-main
# model=$1
# save_path=$2
# task=$3
# is_vqa=$is_vqa
#test_tasks:art、fomc、math、medical、numglue、object、ocr、science
python ./src/eval/test_rec_baseline.py \
    --model_path /public/home/houzhiyan/VLM-R1-main/cl_output/seqftback/Qwen2.5-VL-3B_lora_0/4-lora \
    --bs 32 \
    --output_dir /public/home/houzhiyan/VLM-R1-main/cl_output/seqftback/Qwen2.5-VL-3B_lora_0/prediction \
    --image_root /home/houzhiyan/dataset/images \
    --data_root /public/home/houzhiyan/VLM-R1-main/RLCL_datasets/test \
    --test_datasets_task fomc \
    --sample_num 1200 \
    --max_new_tokens 1024 \
    --is_vqa false \