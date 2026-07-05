export CUDA_VISIBLE_DEVICES=0
cd /public/home/houzhiyan/VLM-R1-main
model_path=$1
output_dir=$2
task=$3
#test_tasks:art、fomc、math、medical、numglue、object、ocr、science
python ./src/eval/test_rec_baseline.py \
    --model_path $model_path \
    --bs 64 \
    --output_dir $output_dir \
    --image_root /home/houzhiyan/dataset/images \
    --data_root /public/home/houzhiyan/VLM-R1-main/RLCL_datasets/test \
    --test_datasets_task $task \
    --sample_num 1200 \
    --max_new_tokens 1024 \
    --is_vqa false \