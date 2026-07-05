
export WANDB_MODE=offline
export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
order=$1

accelerate launch --config_file=src/open-r1-multimodal/configs/zero1.yaml src/src/open_r1/max_merge_model.py \
    --model_name_or_path /public/home/houzhiyan/Qwen2.5-VL-3B-Instruct \
    --ft_model_paths cl_output/${order}/Qwen2.5-VL-3B_${order}_0 \
    --image_root /home/houzhiyan/dataset/images \
    --dataset_name /public/home/houzhiyan/VLM-R1-main/RLCL_datasets/train/art.json \
    --learning_rate 1.0e-5 \
    --num_train_epochs 8 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 5 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --output_dir cl_output/${order}/Qwen2.5-VL-3B_${order}_0/max_merge \
    --seed 42 \
    --cur_task art \
    --alpha 0.8 \
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 bash /public/home/houzhiyan/VLM-R1-main/src/scripts_cl/merge_max_model.sh seqft