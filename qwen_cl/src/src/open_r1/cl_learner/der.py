"""
DER (Dark Experience Replay, Yan & Wang variant).

Strategy:
- During training of task k, also forward the **prior task's frozen model** on
  the same batch; cache its logits z*. Add a per-token distillation loss:
      L_distill = ||student_logits - z*||^2  on labels != -100 tokens.

Drop-in for Qwen2.5-VL + LoRA: we use `copy.deepcopy(model)` after the prior
adapter is loaded (same trick as LwF). This is an APPROXIMATION of the
canonical DER (which stores logits on a sampled replay buffer); since we
already do CL on the **current task's** data and there's no separate replay
buffer in this codebase, this learner effectively becomes "LwF with MSE on
logits" — a reasonable variant matching the spirit of DER on a sequential
pipeline without external buffer.

To get the canonical "logits on buffered prior-task samples" form, combine
this learner with --cl_method=der --replay_ratio>0; the replay_paths samples
will be forwarded too and the same teacher will be applied to them.
"""

import copy
import torch
from .base import BaseCLearner, rank0_print


class DERLearner(BaseCLearner):

    name = "der"

    def __init__(self, lbd, **kwargs):
        super().__init__()
        self.lbd = float(lbd)
        self.teacher = None
        self._is_first_task = True

    def before_train(self, task_id, model, tokenizer=None, **kwargs):
        self._is_first_task = (task_id == 0)
        if self._is_first_task:
            self.teacher = None
            return
        rank0_print(f"[DER] snapshot teacher (frozen prior-task model) at task {task_id}")
        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.train()

    def after_train(self, task_id, model, tokenizer=None, **kwargs):
        if self.teacher is not None:
            del self.teacher
            self.teacher = None
            import gc, torch as _t
            gc.collect()
            if _t.cuda.is_available():
                _t.cuda.empty_cache()

    def loss(self, loss, model=None, inputs=None, outputs=None, **kwargs):
        if self._is_first_task or self.teacher is None or inputs is None:
            return loss
        labels = inputs.get("labels", None)
        if labels is None or outputs is None or getattr(outputs, "logits", None) is None:
            return loss

        # Lazy: move teacher to inputs' device + bf16 dtype (same as LwF)
        target_dev = None
        target_dtype = torch.bfloat16
        for v in inputs.values():
            if torch.is_tensor(v):
                target_dev = v.device
                break
        if target_dev is not None:
            t_params = next(self.teacher.parameters())
            if t_params.device != target_dev or t_params.dtype != target_dtype:
                self.teacher = self.teacher.to(device=target_dev, dtype=target_dtype)

        teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        with torch.no_grad():
            teacher_out = self.teacher(**teacher_inputs)
        teacher_logits = teacher_out.logits

        s_logits = outputs.logits[..., :-1, :].contiguous()
        t_logits = teacher_logits[..., :-1, :].contiguous()
        labels_shift = labels[..., 1:].contiguous()

        token_indices = torch.nonzero(labels_shift != -100, as_tuple=True)
        if token_indices[0].numel() == 0:
            return loss

        s_tok = s_logits[token_indices].float()
        t_tok = t_logits[token_indices].float()
        # MSE on logits (canonical DER form: ||z_s - z_t||^2, mean over tokens)
        distill = self.lbd * torch.nn.functional.mse_loss(s_tok, t_tok)
        try:
            rank0_print(f"[DER] sup={float(loss):.4f}  distill_mse={float(distill):.4f}")
        except Exception:
            pass
        return loss + distill
