#!/bin/bash
# 子进程逐任务 launcher: 每个任务一个独立 deepspeed 进程, 训完+评估完退出,
# OS 回收全部显存 → 杜绝跨任务 reload 显存泄漏 (in-process 模式会泄漏 ~17.7GB/任务).
# 下一任务由 train_cl 的 start_task_id 逻辑自动从上一任务 checkpoint 续训.
#
# Usage: bash scripts/llava_v1_5/run_method_subproc.sh <method> [<hyperparam>] [<order>]
set -u
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=False
export DS_SKIP_CUDA_CHECK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY
source /mnt/cxzx/workspace/data_transfer/houzhiyan/miniconda3/etc/profile.d/conda.sh
conda activate llava

REPO=/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM
MODEL_DIR=/mnt/cxzx/workspace/data_transfer/houzhiyan/models
IMG_DIR=/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark
DATA_DIR=/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/llava_train
export PYTHONPATH=$REPO:${PYTHONPATH:-}

method=${1:?missing method}
hyperparam=${2:-default}
order=${3:-order3}

case $order in
  order1) tasks="numglue,art,math,fomc,medical,OCR,science" ;;
  order2) tasks="science,OCR,medical,fomc,math,art,numglue" ;;
  order3) tasks="art,OCR,fomc,science,numglue,math,medical" ;;
  *) echo "unknown order: $order"; exit 1 ;;
esac
initial_tasks="numglue,art,math,fomc,medical,OCR,science"
NUM_TASKS=$(echo "$tasks" | awk -F',' '{print NF}')

ewc_lbd=1e8; lwf_lbd=0.2; replay_ratio=0; extra_args=""
# 显式判 "default" 字面值. 见 ERRORS_MACHINE_B.md §1 (run_method.sh 同样 bug 已修).
case $method in
  none|seqft) method=none ;;
  ewc)        [ "$hyperparam" = "default" ] && ewc_lbd=1e8 || ewc_lbd=$hyperparam ;;
  # MAS 用 |grad| 而不是 grad^2 累积 importance, 量级比 EWC 大 ~1000x; 共用 λ=1e8 会让 reg 项
  # 远大于 sup_loss → 反传梯度爆 → NaN. 默认降到 1e6, 与 ewc.py loss() 的 NaN guard 配合.
  # 见 ERRORS_MACHINE_B.md MAS λ scaling 复盘.
  mas)        [ "$hyperparam" = "default" ] && ewc_lbd=1e6 || ewc_lbd=$hyperparam ;;
  lwf)        [ "$hyperparam" = "default" ] && lwf_lbd=0.2 || lwf_lbd=$hyperparam ;;
  der)        if [ "$hyperparam" = "default" ]; then lwf_lbd=0.5; replay_ratio=0.05; else lwf_lbd=$hyperparam; replay_ratio=0.05; fi ;;
  replay)     [ "$hyperparam" = "default" ] && replay_ratio=0.2 || replay_ratio=$hyperparam ;;
  freeze|max_merge|l2p) ;;
  *) echo "unknown method: $method"; exit 1 ;;
esac

# 单 task 最多 retry 几次. 推理时 NCCL timeout / OOM cascade 经常因为不同 rank
# 分到的数据长短不一, 重试一次很可能就过. 见 ERRORS_MACHINE_B.md §2.
MAX_RETRY=${MAX_RETRY:-3}

output_dir="$REPO/checkpoints/llava-v1.5-7b/${order}/${method}_${hyperparam}"
mkdir -p "$output_dir/predictions"
LR=2e-5

echo "[$(date -Iseconds)] SUBPROC launcher method=$method order=$order tasks=$tasks ($NUM_TASKS)"

for (( N=1; N<=NUM_TASKS; N++ )); do
    # resume: 若该任务的 merged predictions 已存在 (非 chunk 子目录), 视为已完成, 跳过.
    # (中途断点/数据移动后重启时不重训已完成的任务)
    if ls "$output_dir/$N/predictions/$N"-*.jsonl >/dev/null 2>&1; then
        echo "================ [$method] task $N/$NUM_TASKS already done (predictions exist), SKIP ================"
        continue
    fi

    # 重试循环: 同一任务最多 MAX_RETRY 次, 直到 predictions 落盘为止.
    # 失败原因常是推理阶段多 rank 数据长度不齐 → NCCL timeout / OOM cascade,
    # 这类错误重启同样的任务大概率能过 (随机数据顺序不同, KV cache 峰值不同).
    # 见 ERRORS_MACHINE_B.md §2/§3.
    attempt=0
    while [ $attempt -lt $MAX_RETRY ]; do
        attempt=$((attempt+1))
        echo "================ [$method] task $N/$NUM_TASKS attempt $attempt/$MAX_RETRY start $(date -Iseconds) ================"

        # 清掉本任务可能的残留 dir, 保证 start_task_id 计算成 N-1 (从上一任务 checkpoint 续训)
        rm -rf "$output_dir/$N"

        port=$(shuf -i25000-30000 -n1)
        # 注意: deepspeed 整个 rc 我们不强求 0, 只看 predictions 是否落盘.
        # `set -u` 不影响 pipefail 与否 (本脚本未开 pipefail), 用 || true 兜底 deepspeed crash.
        deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port $port $REPO/llava/train/train_cl.py \
            --deepspeed $REPO/scripts/zero2.json \
            --model_name_or_path $MODEL_DIR/llava-v1.5-7b \
            --vision_tower $MODEL_DIR/clip-vit-large-patch14-336 \
            --version v1 \
            --tasks $tasks --initial_tasks $initial_tasks \
            --data_dir $DATA_DIR --image_folder $IMG_DIR \
            --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2 \
            --mm_use_im_start_end False --mm_use_im_patch_token False \
            --image_aspect_ratio pad --group_by_modality_length True \
            --bf16 True --output_dir $output_dir \
            --num_train_epochs 3 \
            --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 4 \
            --evaluation_strategy no --save_strategy steps --save_steps 50000 --save_total_limit 1 \
            --learning_rate $LR --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type cosine \
            --logging_steps 5 --tf32 True \
            --model_max_length 2048 --max_new_tokens 2048 \
            --gradient_checkpointing True --dataloader_num_workers 4 --lazy_preprocess True \
            --report_to none \
            --cl $method --only_task_id $N --ewc_lbd $ewc_lbd --lwf_lbd $lwf_lbd \
            --replay_ratio $replay_ratio --optim adamw_bnb_8bit --seed 42 \
            $extra_args \
            2>&1 | tee -a "$output_dir/log_task${N}.txt" || true

        if ls "$output_dir/$N/predictions/"*.jsonl >/dev/null 2>&1; then
            echo "================ [$method] task $N/$NUM_TASKS done at attempt $attempt $(date -Iseconds) ================"
            break
        fi

        echo "[$method] task $N attempt $attempt FAILED (no predictions). retry after 30s..."
        sleep 30
    done

    # 所有 retry 用尽仍没 predictions: 不 abort, 让下一 task 继续 (用户决议: 推理类错误忽略).
    # 注意: 下一 task 会用本 task 的 model checkpoint 续训; 如果本 task 训练阶段也没成功
    # (即 $output_dir/$N/ 整个目录不存在), train_cl 的 start_task_id 会停在 N-1, 续训仍是 N-1 → N,
    # 等于重做本 task. 这是预期行为.
    if ! ls "$output_dir/$N/predictions/"*.jsonl >/dev/null 2>&1; then
        echo "[$method] task $N permanently FAILED after $MAX_RETRY attempts. SKIPPING to next task."
    fi

    # 磁盘清理: 删掉 task N-2 的模型权重 (保留 predictions/evaluations + task N-1 供续训 + max_merge snapshots)
    OLD=$((N-2))
    if [ $OLD -ge 1 ]; then
        find "$output_dir/$OLD" -maxdepth 1 -type f \
             \( -name "*.bin" -o -name "*.safetensors" -o -name "pytorch_model*" \) -delete 2>/dev/null
        echo "[cleanup] removed model weights of task $OLD (kept predictions)"
    fi
done

echo "================ [$method] ALL $NUM_TASKS tasks done $(date -Iseconds) ================"
