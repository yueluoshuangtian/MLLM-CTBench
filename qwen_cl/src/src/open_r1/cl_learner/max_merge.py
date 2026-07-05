"""
MagMaX (Marczak 2024): per-task independent SFT then element-wise MAX merge.

Run-time behavior:
- Training is identical to Sequential SFT, EXCEPT each task starts from the
  base model (no `--continue_lora_path`), i.e. learns its own LoRA delta.
- After each task, save a snapshot of trainable LoRA params to disk.
- After the FINAL task: scan all task_*.pt snapshots, take element-wise max
  per parameter (treating each tensor as N candidate values, picking max),
  load merged values into the model, save the merged adapter.

`finalize(model)` is called from the main loop after the last task only.
"""

import os
import glob
import torch

from .base import BaseCLearner, rank0_print, barrier


class MaxMergeLearner(BaseCLearner):

    name = "max_merge"

    def __init__(self, output_dir, **kwargs):
        super().__init__()
        # output_dir is the per-task dir, e.g. <root>/<step>-sft. Snapshots go
        # under <root>/max_merge_snapshots/.
        self.task_output_dir = output_dir
        self.snapshot_dir = os.path.join(
            os.path.dirname(output_dir.rstrip("/")),
            "max_merge_snapshots",
        )
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def before_train(self, task_id, model, tokenizer=None, **kwargs):
        pass

    def after_train(self, task_id, model, tokenizer=None, **kwargs):
        snap_path = os.path.join(self.snapshot_dir, f"task_{task_id}.pt")
        from .base import _is_rank0
        if _is_rank0():
            try:
                from transformers.modeling_utils import unwrap_model
                base = unwrap_model(model)
            except Exception:
                base = model
            state = {
                n: p.detach().cpu().clone()
                for n, p in base.named_parameters()
                if p.requires_grad
            }
            torch.save(state, snap_path)
            rank0_print(f"[MaxMerge] saved snapshot task {task_id} -> {snap_path}")
        barrier()

    def finalize(self, model, **kwargs):
        snap_paths = sorted(
            glob.glob(os.path.join(self.snapshot_dir, "task_*.pt")),
            key=lambda p: int(os.path.basename(p)[len("task_"):-len(".pt")]),
        )
        if not snap_paths:
            rank0_print("[MaxMerge] no snapshots, skipping finalize")
            return
        rank0_print(
            f"[MaxMerge] merging {len(snap_paths)} snapshots (element-wise max): "
            f"{[os.path.basename(p) for p in snap_paths]}"
        )
        all_states = [torch.load(p, map_location="cpu") for p in snap_paths]
        try:
            from transformers.modeling_utils import unwrap_model
            base = unwrap_model(model)
        except Exception:
            base = model
        n_merged = 0
        for n, p in base.named_parameters():
            if not p.requires_grad:
                continue
            vals = [s[n] for s in all_states if n in s]
            if not vals:
                continue
            stacked = torch.stack(vals, dim=0)
            merged = stacked.max(dim=0).values
            p.data.copy_(merged.to(p.device, dtype=p.dtype))
            n_merged += 1
        rank0_print(f"[MaxMerge] merged {n_merged} params")
