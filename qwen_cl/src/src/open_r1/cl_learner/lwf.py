"""
LwF (Li & Hoiem) on Qwen2.5-VL + PEFT-LoRA.

Strategy:
- After task k finishes, save the merged (base + LoRA-k) weights as a teacher
  snapshot (or keep it in RAM if memory allows). Since Qwen2.5-VL-3B is 6GB
  bf16, deepcopy is feasible on 80GB cards but not ideal.
- During task k+1, on each batch we additionally forward the **frozen teacher**
  (with the previous task's adapter loaded) on the same `inputs`, take its
  logits, and add KL(student || teacher) on `labels != -100` tokens.

Implementation detail (LoRA-only):
- The cheapest teacher is the **previous-task adapter merged with the base
  model**. We keep one `teacher_lora_path` (the previous task's adapter dir)
  and **clone the student model + reload the previous adapter** at task start.
  Since both teacher and student share the same frozen base, only one extra
  set of LoRA weights is in memory.

To keep this drop-in compatible with the existing `sft.py` (single model
construction), we approximate by **using a copy.deepcopy of the model before
the new adapter starts diverging**. For Qwen-VL 3B bf16, that's ~6-8 GB per
GPU — acceptable on 80GB.
"""

import copy
import torch
from .base import BaseCLearner, rank0_print


class LwFLearner(BaseCLearner):

    name = "lwf"

    def __init__(self, lbd, **kwargs):
        super().__init__()
        self.lbd = float(lbd)
        self.teacher = None  # deepcopy of model AFTER prior adapter loaded
        self._is_first_task = True

    # =========================================================================
    def before_train(self, task_id, model, tokenizer=None, **kwargs):
        # task_id is 0-based
        self._is_first_task = (task_id == 0)
        if self._is_first_task:
            # No teacher for task 1 — first task is pure SFT
            self.teacher = None
            return
        rank0_print(f"[LwF] snapshot teacher from model state at start of task {task_id}")
        # At this point the model has already loaded the prior adapter via
        # `--continue_lora_path`. Deepcopy preserves that exact state.
        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        # Keep teacher in train() mode so it goes through flash-attn correctly
        # (matches the CLMM trick).
        self.teacher.train()

    def after_train(self, task_id, model, tokenizer=None, **kwargs):
        # Release teacher to free memory before next task
        if self.teacher is not None:
            del self.teacher
            self.teacher = None
            import gc, torch as _t
            gc.collect()
            if _t.cuda.is_available():
                _t.cuda.empty_cache()

    # =========================================================================
    def loss(self, loss, model=None, inputs=None, outputs=None, **kwargs):
        if self._is_first_task or self.teacher is None or inputs is None:
            return loss
        labels = inputs.get("labels", None)
        if labels is None or outputs is None or getattr(outputs, "logits", None) is None:
            return loss

        # Lazy: move teacher to inputs' device + bf16 dtype on first call
        # (deepcopy in before_train happened before deepspeed dispatched the
        # student to GPU + cast to bf16, so the teacher may still be on CPU/fp32).
        any_v = next((v for v in inputs.values() if v is not None and torch.is_tensor(v) and v.is_floating_point()), None)
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

        # Strip labels from teacher inputs so we don't compute teacher loss
        teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        with torch.no_grad():
            teacher_out = self.teacher(**teacher_inputs)
        teacher_logits = teacher_out.logits

        student_logits = outputs.logits

        # next-token alignment
        student_logits = student_logits[..., :-1, :].contiguous()
        teacher_logits = teacher_logits[..., :-1, :].contiguous()
        labels_shift = labels[..., 1:].contiguous()

        token_indices = torch.nonzero(labels_shift != -100, as_tuple=True)
        if token_indices[0].numel() == 0:
            return loss

        s_tok = student_logits[token_indices]
        t_tok = teacher_logits[token_indices]

        log_p = torch.log_softmax(s_tok, dim=-1)
        q = torch.softmax(t_tok, dim=-1)
        distill = self.lbd * torch.nn.functional.kl_div(
            log_p, q, reduction="batchmean", log_target=False
        )
        try:
            rank0_print(f"[LwF] sup={float(loss):.4f}  distill={float(distill):.4f}  lbd={self.lbd}")
        except Exception:
            pass
        return loss + distill
