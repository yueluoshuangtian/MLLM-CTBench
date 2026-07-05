"""
Max-merge: 模型融合 baseline.
每个任务独立训练，最后通过逐参数 element-wise max 把所有任务 checkpoint 融合成一个模型。
论文 [30] 的实现。

实现策略 (受限于 train_cl.py 主循环逐任务训练 + 评测的设计):
- 训练阶段: 不做任何正则 (等价 SFT)
- after_train 阶段: 把当前任务训练完的模型 state_dict 快照存到 output_dir/<task_id>/max_merge_snapshot.pt
- 最后一个任务完成后: 遍历所有快照, 对每个参数取 element-wise max, 写回模型并保存
"""
import os
import torch
import torch.distributed as dist
from transformers.modeling_utils import unwrap_model
from .base import BaseCLearner


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class MaxMergeLearner(BaseCLearner):
    """
    Element-wise Max merging of per-task checkpoints.
    训练流程与 SFT 完全相同, 仅在每个 task 结束后保存快照,
    最后一个 task 完成后做 max-merge.
    """

    def __init__(self, rank, output_dir):
        global local_rank
        local_rank = rank
        self.output_dir = output_dir
        self.snapshot_dir = os.path.join(output_dir, "max_merge_snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.snapshots = []  # list of paths

    def before_train(self, task_id, model, tokenizer=None, **kwargs):
        pass

    def after_train(self, task_id, model, tokenizer=None, **kwargs):
        """每个 task 训完, rank0 保存 trainable params 快照."""
        snapshot_path = os.path.join(self.snapshot_dir, f"task_{task_id}.pt")
        if local_rank in (0, -1):
            base_model = unwrap_model(model)
            state = {
                n: p.detach().cpu().clone()
                for n, p in base_model.named_parameters()
                if p.requires_grad
            }
            torch.save(state, snapshot_path)
            rank0_print(f"[MaxMerge] saved snapshot for task {task_id} → {snapshot_path}")
        if dist.is_initialized():
            dist.barrier()
        self.snapshots.append(snapshot_path)

    def loss(self, loss, **kwargs):
        return loss

    def finalize(self, model):
        """
        所有任务训练完成后调用, 把 snapshot 逐参数取 max 写回 model.
        train_cl.py 的主循环需要在 for-loop 之后手动调一次 cl_learner.finalize(model).
        """
        # 从磁盘 glob 所有 task_*.pt, 而不是依赖内存里的 self.snapshots.
        # (子进程逐任务模式下每个进程的 self.snapshots 只含当前任务那一个,
        #  必须扫磁盘才能拿全所有任务的快照.)
        import glob as _glob
        snap_paths = sorted(
            _glob.glob(os.path.join(self.snapshot_dir, "task_*.pt")),
            key=lambda p: int(os.path.basename(p)[len("task_"):-len(".pt")])
        )
        if not snap_paths:
            snap_paths = list(self.snapshots)
        if not snap_paths:
            rank0_print("[MaxMerge] no snapshots, skipping finalize.")
            return
        rank0_print(f"[MaxMerge] merging {len(snap_paths)} snapshots with element-wise max: {[os.path.basename(p) for p in snap_paths]}")

        # 把 snapshot 文件都读到内存
        all_states = [torch.load(p, map_location="cpu") for p in snap_paths]

        base_model = unwrap_model(model)
        for n, p in base_model.named_parameters():
            if not p.requires_grad:
                continue
            vals = [s[n] for s in all_states if n in s]
            if not vals:
                continue
            stacked = torch.stack(vals, dim=0)
            merged = stacked.max(dim=0).values
            p.data.copy_(merged.to(p.device, dtype=p.dtype))

        rank0_print("[MaxMerge] merge done.")
