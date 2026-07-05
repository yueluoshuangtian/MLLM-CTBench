export WANDB_MODE=offline
export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0


accelerate launch --config_file=src/open-r1-multimodal/configs/zero3.yaml src/src/open_r1/sft.py \
    --model_name_or_path /public/home/houzhiyan/VLM-R1-main/cl_output/seqftback/Qwen2.5-VL-3B_lora_0/3 \
    --dataset_name /public/home/houzhiyan/VLM-R1-main/RLCL_datasets/train/fomc.json \
    --image_root /home/houzhiyan/dataset/images \
    --learning_rate 1e-5 \
    --num_train_epochs 7 \
    --output_dir /public/home/houzhiyan/VLM-R1-main/cl_output/seqftback/Qwen2.5-VL-3B_lora_0/4-lora \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --logging_steps 5 \
    --bf16 \
    --data_seed 42 \
    --report_to wandb \
    --attn_implementation flash_attention_2 \
    --save_only_model true \
    --save_steps 500 \
    --lora_target_modules q_proj v_proj visual \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --cur_task fomc \

python /public/home/houzhiyan/VLM-R1-main/src/src/open_r1/lora_merge.py \
    --base_model_path /public/home/houzhiyan/VLM-R1-main/cl_output/seqftback/Qwen2.5-VL-3B_lora_0/3 \
    --lora_path /public/home/houzhiyan/VLM-R1-main/cl_output/seqftback/Qwen2.5-VL-3B_lora_0/4-lora \
    --save_path /public/home/houzhiyan/VLM-R1-main/cl_output/seqftback/Qwen2.5-VL-3B_lora_0/4 \
