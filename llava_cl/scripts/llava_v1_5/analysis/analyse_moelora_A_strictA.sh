#!/bin/bash
set -e

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=False
export PYTHONPATH=/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM:$PYTHONPATH

order=$1
expert_num=$2
gate_rank=$3
orth_lambda=$4
old_task=$5

run_root="/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/without_reasoning/llava-v1.5-7b/moelora_forget/${order}/moeloraArouter_E${expert_num}_gr${gate_rank}_${orth_lambda}"
analysis_root="/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/without_reasoning/moelora_analysis_strictA/${order}_E${expert_num}_gr${gate_rank}_${orth_lambda}_old${old_task}"

mkdir -p "${analysis_root}"

echo "[Analysis-Bash] run_root=${run_root}"
echo "[Analysis-Bash] analysis_root=${analysis_root}"

port=$(shuf -i25000-30000 -n1)
echo "[Analysis-Bash] port=${port}"

deepspeed --include=localhost:0 --master_port "${port}" \
  llava/train/moelora_offline_analysis_strictA.py \
  --run_root "${run_root}" \
  --analysis_root "${analysis_root}" \
  --old_task_name "${old_task}" \
  --batch_size 4 \
  --max_old_batches 50 \
  --task_id_1 1 \
  --task_id_T 7 \
  --draw_I_tasks "1,4,7" \
  --output_dir "${run_root}" \
  --lora_enable True \
  --lora_r 128 \
  --lora_alpha 256 \
  --lora_dropout 0.0 \
  --lora_bias none \
  --expert_num "${expert_num}" \
  --gate_rank "${gate_rank}" \
  --moelora_orth_lambda "${orth_lambda}" \
  --gate_router a \
  --moelora_reg_enable True \
  --moelora_importance_top_p 0.4 \
  --moelora_B_reg_scale 0.1 \
  --deepspeed ./scripts/zero2.json \
  --model_name_or_path /mnt/cxzx/workspace/data_transfer/houzhiyan/models/llava-v1.5-7b \
  --version v1 \
  --initial_tasks "numglue,art,math,fomc,medical,OCR,science" \
  --tasks "numglue,art,math,fomc,medical,OCR,science" \
  --image_folder //mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark \
  --vision_tower /mnt/cxzx/workspace/data_transfer/houzhiyan/models/clip-vit-large-patch14-336 \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --model_max_length 1536 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --cl moelora \
  2>&1 | tee "${analysis_root}/log.txt"
