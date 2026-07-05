"""
EWC (Kirkpatrick 2017) on Qwen2.5-VL + PEFT-LoRA.

Online importance estimation: register backward hooks on LoRA params during
the current task's training; accumulate grad^2 per parameter into a CPU buffer.
At the end of each task: mean-normalize + top-k sparsify the importance, blend
with historical importance via task-count weighted average, and save a snapshot
of current LoRA params as Whist (anchor for next task's regularizer).

Reg loss at step t (task_id >= 1):
    L_total = L_sup + (lbd / sqrt(task_id+1)) * 0.5 * sum_i Ipt_i * (p_i - Whist_i)^2

Ported & adapted from CLMM/cl_learner/ewc.py.
On LoRA-only training, mem footprint is small (~50-200 MB total), so we keep
Ipt/Whist on CPU and stream to GPU per-param via a custom autograd Function.
"""

import os
import copy
import re
import torch
import torch.distributed as dist

from .base import (
    BaseCLearner,
    rank0_print,
    select_trainable_lora_params,
    barrier,
    is_lora_param,
)


_ADAPTER_RE = re.compile(r"(lora_[AB])\.[^.]+\.weight$")


def _norm_key(n):
    """Strip the adapter name so keys are stable across tasks.
    'base_model....lora_A.task_3.weight' → 'base_model....lora_A.weight'
    """
    return _ADAPTER_RE.sub(r"\1.weight", n)


class _EWCRegFn(torch.autograd.Function):
    """
    Forward computes 0.5 * Ipt * (p - Whist)^2 without retaining the intermediate
    tensors in the autograd graph (no_grad). Backward returns the analytical
    gradient: lbd * Ipt * (p - Whist).
    """

    @staticmethod
    def forward(ctx, param, ipt_cpu, whist_cpu, lbd_scalar):
        ctx.save_for_backward(param)
        ctx.ipt_cpu = ipt_cpu
        ctx.whist_cpu = whist_cpu
        ctx.lbd = lbd_scalar
        with torch.no_grad():
            ipt = ipt_cpu.to(param.device, dtype=param.dtype, non_blocking=True)
            whist = whist_cpu.to(param.device, dtype=param.dtype, non_blocking=True)
            val = (ipt * (param - whist).pow(2)).sum() * 0.5
            del ipt, whist
        return val

    @staticmethod
    def backward(ctx, grad_output):
        (param,) = ctx.saved_tensors
        ipt = ctx.ipt_cpu.to(param.device, dtype=param.dtype, non_blocking=True)
        whist = ctx.whist_cpu.to(param.device, dtype=param.dtype, non_blocking=True)
        grad_param = ctx.lbd * grad_output * ipt * (param - whist)
        del ipt, whist
        return grad_param, None, None, None


class EWCLearner(BaseCLearner):
    """
    Args:
      lbd: base regularization weight (paper: 1e8 for full-FT — for LoRA r=64
        a much smaller value (1e2..1e4) is typical, but we expose it as CLI arg).
      output_dir: where to save cl_states_lora.bin (Ipt + task counter).
      ipt_sparsity: keep top (1 - sparsity) fraction of importance per param tensor.
    """

    name = "ewc"

    Ipt = None
    Whist = None
    num_seen_tasks = 0

    def __init__(self, lbd, output_dir, ipt_sparsity=0.9, grad_sq_clip=1e2, **kwargs):
        super().__init__()
        self.lbd = float(lbd)
        self.output_dir = output_dir
        self.save_file = os.path.join(output_dir, "cl_states_lora.bin")
        # Note: per-task subprocess mode (run_order3.sh re-launches sft.py for each
        # task) means we MUST persist Ipt to disk between tasks, and reload on
        # each new process.
        self._ipt_sparsity = float(ipt_sparsity)
        self._grad_sq_clip = float(grad_sq_clip)
        self._min_eps = 1e-8

        self.list_of_hooks = []
        self.curIpt = None
        self.iters = None
        self.load_num_seen_tasks = 0
        self._record_on = True  # gradient hooks always fire during training_step

    # =========================================================================
    # Reg loss (called by CustomSFTTrainer.compute_loss)
    # =========================================================================
    def loss(self, loss, model=None, inputs=None, outputs=None, **kwargs):
        if self.Ipt is None or model is None:
            return loss
        cur_task_id = self.num_seen_tasks + 1  # 1-based for the schedule
        effective_lbd = self.lbd / (cur_task_id ** 0.5 if cur_task_id > 0 else 1.0)

        reg_total = loss.new_zeros(())
        n_applied = 0
        for n, p in select_trainable_lora_params(model):
            key = _norm_key(n)
            if key not in self.Ipt or key not in self.Whist:
                continue
            reg_total = reg_total + _EWCRegFn.apply(
                p, self.Ipt[key], self.Whist[key], effective_lbd
            )
            n_applied += 1

        if n_applied > 0:
            rank0_print(
                f"[EWC] sup={float(loss):.4f}  reg={float(reg_total):.4f}  "
                f"lbd_eff={effective_lbd:.4g}  params={n_applied}"
            )
        return loss + reg_total

    # =========================================================================
    # Lifecycle
    # =========================================================================
    def before_train(self, task_id, model, tokenizer=None, **kwargs):
        # task_id is 0-based: first task -> 0, second -> 1
        self.load()
        if task_id > 0 and self.num_seen_tasks >= self.load_num_seen_tasks and self.Ipt is not None:
            self.get_Whist(model)
        self.register_ipt_hooks(model)

    def after_train(self, task_id, model, tokenizer=None, **kwargs):
        self.remove_ipt_hooks()
        if self.num_seen_tasks >= self.load_num_seen_tasks:
            self.Whist = None
            if self.curIpt is not None and self.iters is not None and self.iters > 0:
                rank0_print(f"[EWC] update Ipt after task {task_id}, {self.iters} steps")
                self.update_Ipt()
                self.Ipt = {n: p.cpu() for n, p in self.Ipt.items()}
                self.save()
            else:
                rank0_print(f"[EWC][WARN] no grad stats for task {task_id}, skip Ipt update")
        self.curIpt = None
        self.iters = None
        self.num_seen_tasks += 1

    # =========================================================================
    # Whist snapshot (anchor params)
    # =========================================================================
    def get_Whist(self, model):
        rank0_print("[EWC] snapshot Whist on CPU (normalized keys)")
        self.Whist = {
            _norm_key(n): p.detach().clone().cpu().to(dtype=torch.bfloat16)
            for n, p in select_trainable_lora_params(model)
        }

    # =========================================================================
    # Update Ipt: mean-normalize + top-k sparsify + task-count weighted blend
    # =========================================================================
    def update_Ipt(self):
        if self.curIpt is None or self.iters is None or self.iters <= 0:
            return
        cur_ipt_cpu = {}
        for n, p in self.curIpt.items():
            p = p.cpu() / max(1, self.iters)
            mean_val = p.abs().mean()
            if mean_val > 0:
                p = p / (mean_val + self._min_eps)
            if 0.0 < self._ipt_sparsity < 1.0:
                flat = p.view(-1)
                k = int((1.0 - self._ipt_sparsity) * flat.numel())
                if k > 0 and k < flat.numel():
                    thresh = flat.abs().kthvalue(max(1, flat.numel() - k))[0]
                    p = p * (p.abs() >= thresh)
            cur_ipt_cpu[n] = p

        if self.Ipt is not None:
            for n, p in cur_ipt_cpu.items():
                if n in self.Ipt:
                    self.Ipt[n] = (
                        self.Ipt[n] * self.num_seen_tasks + p
                    ) / (self.num_seen_tasks + 1)
                else:
                    self.Ipt[n] = p
        else:
            self.Ipt = cur_ipt_cpu

        self.curIpt = None

    # =========================================================================
    # Save / load (persist across subprocess task launches)
    # =========================================================================
    def save(self):
        from .base import _is_rank0
        if _is_rank0():
            os.makedirs(os.path.dirname(self.save_file), exist_ok=True)
            cpu_Ipt = {n: p.to(torch.float32) for n, p in self.Ipt.items()}
            torch.save(
                {"Ipt": cpu_Ipt, "num_seen_tasks": self.num_seen_tasks + 1},
                self.save_file,
            )
            rank0_print(f"[EWC] saved Ipt -> {self.save_file}")
        barrier()

    def load(self):
        if os.path.isfile(self.save_file) and self.Ipt is None:
            ckpt = torch.load(self.save_file, map_location="cpu")
            self.Ipt = {n: p.to(torch.bfloat16) for n, p in ckpt["Ipt"].items()}
            self.load_num_seen_tasks = ckpt["num_seen_tasks"]
            self.num_seen_tasks = ckpt["num_seen_tasks"]
            rank0_print(
                f"[EWC] loaded Ipt from {self.save_file} (num_seen_tasks={self.num_seen_tasks})"
            )
            del ckpt

    # =========================================================================
    # Online grad-square hooks
    # =========================================================================
    def cal_ipt(self, name):
        def hook(grad):
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_sq = grad.pow(2)
            if self._grad_sq_clip is not None:
                grad_sq = torch.clamp(grad_sq, max=self._grad_sq_clip)
            self.curIpt[name].add_(grad_sq.detach().to("cpu", dtype=torch.bfloat16))
        return hook

    def rec_iter(self):
        def hook(_grad):
            self.iters += 1
        return hook

    def register_ipt_hooks(self, model):
        """Register `register_post_accumulate_grad_hook` on each LoRA param.
        This newer PyTorch hook (2.0+) fires AFTER gradient accumulation is
        complete (deepspeed has reduced/synced), so .grad is reliably available
        on the param at the moment the hook runs. This avoids the unreliability
        of tensor.register_hook under DeepSpeed ZeRO-2 + PEFT.
        """
        rank0_print("[EWC] register post_accumulate_grad hooks on LoRA params")
        self.iters = 0
        params = list(select_trainable_lora_params(model))
        if not params:
            rank0_print("[EWC][WARN] no LoRA trainable params; reg is no-op")
            return
        # curIpt keyed by NORMALIZED names so keys are stable across tasks
        self.curIpt = {
            _norm_key(n): torch.zeros(p.shape, device="cpu", dtype=torch.bfloat16)
            for n, p in params
        }
        self._counter_name = _norm_key(params[0][0])

        # Prefer the classic register_hook (fires inside backward, before any
        # potential deepspeed partition). Under ZeRO-1 gradients remain on the
        # params, so this is the most reliable signal.
        def _make_hook(name_norm):
            def _hook(grad):
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                self._accumulate_one(name_norm, grad)
                if name_norm == self._counter_name:
                    self.iters = (self.iters or 0) + 1
                    if self.iters <= 3:
                        rank0_print(f"[EWC][DEBUG] hook fired #{self.iters} grad.shape={tuple(grad.shape)}")
                return grad
            return _hook
        for n, p in params:
            kn = _norm_key(n)
            h = p.register_hook(_make_hook(kn))
            self.list_of_hooks.append(h)
        rank0_print(f"[EWC] registered {len(self.list_of_hooks)} hooks on LoRA params")

    def remove_ipt_hooks(self):
        # legacy no-op (hooks removed)
        for h in self.list_of_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.list_of_hooks = []

    def post_backward(self, model, cl_task_progress=None, **kwargs):
        """Fallback path: if register_hook didn't fire (some DeepSpeed configs),
        capture grad directly from .grad here. Called after accelerator.backward().
        Uses normalized keys to match curIpt.
        """
        if self.curIpt is None:
            return
        applied = 0
        for n, p in select_trainable_lora_params(model):
            key = _norm_key(n)
            if key not in self.curIpt:
                continue
            grad = getattr(p, "grad", None)
            if grad is None:
                continue
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            self._accumulate_one(key, grad)
            applied += 1
        if applied > 0:
            # Only increment iters if classic hook didn't already
            # (avoid double counting). Detect by checking whether iters was
            # already advanced this step: simple heuristic — use a separate
            # _pb_count that increments only here.
            self._pb_count = getattr(self, "_pb_count", 0) + 1
            if self.iters == 0 or self.iters is None:
                self.iters = self._pb_count
            if self._pb_count <= 3:
                rank0_print(f"[EWC][DEBUG] post_backward applied={applied} pb_count={self._pb_count} iters={self.iters}")

    def _accumulate_one(self, name, grad):
        """EWC: grad^2 with clip. Overridden by MAS to use |grad|."""
        grad_sq = grad.pow(2)
        if self._grad_sq_clip is not None:
            grad_sq = torch.clamp(grad_sq, max=self._grad_sq_clip)
        self.curIpt[name].add_(grad_sq.detach().to("cpu", dtype=torch.bfloat16))
