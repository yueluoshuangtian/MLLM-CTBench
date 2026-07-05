import os
import gc
import copy
import torch
import torch.distributed as dist
from transformers.modeling_utils import unwrap_model
from .base import BaseCLearner

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def release_memory():
    gc.collect()
    torch.cuda.empty_cache()
    memory_stats()


def memory_stats():
    rank0_print(f"memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2}")
    rank0_print(f"memory reserved:  {torch.cuda.memory_reserved()  / 1024 ** 2}")


def _is_lora_param(name: str) -> bool:
    """LoRA 的 A/B 权重。仅当模型确实含 LoRA 时用作过滤。"""
    return ("lora_A" in name) or ("lora_B" in name)


class _EWCRegFn(torch.autograd.Function):
    """
    定制 autograd Function: forward 计算 reg 标量但不保留中间 tensor (no_grad 算),
    backward 解析地返回 grad_p = λ·ipt·(p - whist).
    避免 (ipt * (p-whist)²).sum() 这种写法在 7B 全参 EWC 时保留几十 GB 中间 tensor → OOM.

    单次 backward + 通过 AccumulateGrad(p) 走一条边汇入 sup_loss 的边, ZeRO-2 hook 只 fire 一次.
    """
    @staticmethod
    def forward(ctx, param, ipt_cpu, whist_cpu, lbd_scalar):
        ctx.save_for_backward(param)
        ctx.ipt_cpu = ipt_cpu      # CPU buffer, detached
        ctx.whist_cpu = whist_cpu  # CPU buffer, detached
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
        # ∂(0.5·ipt·(p-whist)²)/∂p = ipt·(p-whist)
        grad_param = ctx.lbd * grad_output * ipt * (param - whist)
        del ipt, whist
        return grad_param, None, None, None


def _select_ewc_params(model):
    """
    选择需要做 EWC 的参数:
      - 若模型含 LoRA (lora_A/lora_B 至少有 1 个 requires_grad=True) → 只对 LoRA 做 EWC
      - 否则 (full-FT) → 对所有 requires_grad=True 的参数做 EWC
    返回 (name, param) iterator。
    """
    from transformers.modeling_utils import unwrap_model
    base_model = unwrap_model(model)
    has_lora = any(
        p.requires_grad and _is_lora_param(n)
        for n, p in base_model.named_parameters()
    )
    for n, p in base_model.named_parameters():
        if not p.requires_grad:
            continue
        if has_lora and not _is_lora_param(n):
            continue
        yield n, p


class EWCLearner(BaseCLearner):
    """
    LoRA 版 EWC（在线 + 工程增强版）：
    - 只对 LoRA 参数做重要性估计 & 正则；
    - 重要性 Ipt 在“正常训练过程中”通过 grad^2 统计（不再额外起 0-lr Trainer），兼容 DeepSpeed；
    - 统计时做梯度裁剪、防 NaN、归一化、Top-k 稀疏；
    - λ 做简单的任务相关调度（越后面的任务约束略弱一点）；
    - 不增加任何可训练参数，只调整正则项的工程实现。
    """

    # 类级别默认（实际运行时用的是实例属性）
    Ipt = None
    Whist = None
    num_seen_tasks = 0

    def __init__(self, trainer_cls, lbd, rank, output_dir, training_args):
        """
        trainer_cls: 外部传进来的 Trainer 类（接口保持一致，但这里不会再 new Trainer）
        lbd:         原始 EWC 系数（外部 --ewc_lbd）
        rank:        local_rank
        output_dir:  用于保存 Ipt 状态
        training_args: 仅作备份/接口兼容，不会再改 learning_rate
        """
        global local_rank
        local_rank = rank

        self.list_of_hooks = []
        self.curIpt = None         # 当前任务的 grad^2 累积
        self.iters = None          # 当前任务的 backward 次数（步数）
        self.load_num_seen_tasks = 0

        self.lbd = lbd
        self.device = torch.device("cuda", local_rank)
        self.save_file = os.path.join(output_dir, "cl_states_lora.bin")

        self.training_args = copy.deepcopy(training_args)
        self.trainer_cls = trainer_cls

        # 一些工程常数（不是可训练参数）
        self._grad_sq_clip_max = 1e2      # grad^2 裁剪上限，防止极端值
        self._ipt_sparsity = 0.9         # 每个张量只保留 10% 最大重要性（Top-k 稀疏）
        self._min_eps = 1e-8

    # ------------------- 正则 loss -------------------
    def loss(self, loss, model=None, **kwargs):
        """
        总 loss = sup_loss + Σ _EWCRegFn(pᵢ, Iᵢ, p*ᵢ, λ_eff)
        每个 _EWCRegFn 内部 no_grad 计算 reg 值 + 解析 backward 给出 grad_p,
        避免几百个 named param 的 (p-whist)² 中间 tensor 撑爆 GPU (7B 全参 EWC ~50GB autograd 图).

        λ 任务相关调度: λ_eff = λ / sqrt(task_id)
        Whist / Ipt CPU buffers, 按需 stream 到 GPU.
        """
        if self.Ipt is None:
            return loss

        cur_task_id = self.num_seen_tasks + 1
        effective_lbd = float(self.lbd) / (cur_task_id ** 0.5 if cur_task_id > 0 else 1.0)

        reg_total = loss.new_zeros(())
        n_applied = 0
        for n, p in _select_ewc_params(model):
            if n not in self.Ipt or n not in self.Whist:
                continue
            # _EWCRegFn 单 step 只持有当前 param 的 ipt/whist GPU 临时, backward 时再算一次
            reg_total = reg_total + _EWCRegFn.apply(p, self.Ipt[n], self.Whist[n], effective_lbd)
            n_applied += 1

        rank0_print(
            f"[EWC] Sup: {loss.item():.4f}  "
            f"Reg(value): {reg_total.item():.4f}  "
            f"lbd_eff={effective_lbd:.4g}  params={n_applied}"
        )
        # NaN/Inf/巨值 guard: λ 调不准 / Ipt 累积过大 时, reg_total 可能 NaN/Inf 或远大于 sup,
        # 加上后梯度会爆 → 后续 step NaN → 整 run 崩. 见 ERRORS_MACHINE_B.md MAS λ scaling 复盘.
        # 这里只在异常时丢弃本 step 的 reg, 不动 sup_loss, 让训练继续推进.
        _reg_val = reg_total.detach()
        if torch.isnan(_reg_val) or torch.isinf(_reg_val) or _reg_val.abs().item() > 100.0 * max(loss.detach().abs().item(), 1e-6):
            rank0_print(
                f"[EWC][WARN] reg unhealthy (reg={_reg_val.item():.4g}, sup={loss.item():.4g}, "
                f"ratio={_reg_val.item()/max(loss.item(),1e-6):.2g}) → skip reg this step"
            )
            return loss
        return loss + reg_total

    # ------------------- 每个任务前 -------------------
    def before_train(self, task_id, model, tokenizer=None, **kwargs):
        """
        任务开始前：
        1. 尝试加载历史 Ipt；
        2. 记录上一任务结束时的 LoRA 参数快照 Whist（供正则使用）；
        3. 把 Ipt 搬到当前 device；
        4. 在当前模型上挂 Ipt 统计的 hooks（在线统计 grad^2）。
        """
        # （1）加载历史 Ipt（只加载一次）
        self.load()

        if task_id > 0 and self.num_seen_tasks >= self.load_num_seen_tasks and self.Ipt is not None:
            # （2）记录上一任务结束时 LoRA 参数快照
            self.get_Whist(model)
            # Ipt 保持在 CPU, loss() 用时按需移到 GPU (避免 7B full-FT OOM)

        # （4）为当前任务注册 grad^2 hooks
        self.register_ipt_hooks(model)

    # ------------------- 每个任务后 -------------------
    def after_train(self, task_id, model, tokenizer=None,
                    train_dataset=None, data_collator=None, **kwargs):
        """
        任务结束后：
        1. 移除 hooks；
        2. 若有有效统计，则更新 Ipt（归一化 + 稀疏化）；
        3. 把 Ipt 移回 CPU 并保存；
        4. 任务计数 +1。
        """
        # 先卸载 hook，避免评估/下个任务还在写 curIpt
        self.remove_ipt_hooks()

        if self.num_seen_tasks >= self.load_num_seen_tasks:
            # 用完 Whist 可以丢掉，省显存
            self.Whist = None

            # 若整个任务期间确实累积了 grad^2，则更新 Ipt
            if self.curIpt is not None and self.iters is not None and self.iters > 0:
                rank0_print(
                    f"[EWC-LoRA] Updating Ipt after task {task_id} "
                    f"with {self.iters} steps."
                )
                self.update_Ipt()
                # Ipt 统一移回 CPU 存储，下一任务前再搬回 GPU
                self.Ipt = {n: p.cpu() for n, p in self.Ipt.items()}
                self.save()
            else:
                rank0_print(
                    f"[EWC-LoRA][WARN] No grad statistics collected for task {task_id}, "
                    f"skip Ipt update."
                )

        # 清理临时缓存
        self.curIpt = None
        self.iters = None

        # 已完成任务数 +1
        self.num_seen_tasks += 1

    # ------------------- 记录上一任务参数 -------------------
    def get_Whist(self, model):
        # 存 CPU 避免 GPU OOM (7B full-FT Whist 占 14GB). loss() 用时按需移到 GPU.
        rank0_print("[EWC] Get Whist (on CPU) for regularization.")
        self.Whist = {
            n: p.detach().clone().cpu().to(dtype=torch.bfloat16)
            for n, p in _select_ewc_params(model)
        }

    # ------------------- 更新 Ipt -------------------
    def update_Ipt(self):
        """
        用当前任务累积的 grad^2 更新全局 Ipt：
        - 先对当前任务的 Ipt 做均值归一化；
        - 然后做 Top-k 稀疏化（只保留每个张量中最重要的那部分）；
        - 再和历史 Ipt 按任务数加权平均。
        """
        rank0_print("[EWC-LoRA] Updating Ipt.")
        if self.curIpt is None or self.iters is None or self.iters <= 0:
            rank0_print("[EWC-LoRA][WARN] curIpt is None or iters <= 0, skip update.")
            return

        cur_ipt_cpu = {}
        for n, p in self.curIpt.items():
            # 先做平均
            p = p.cpu() / max(1, self.iters)

            # （1）均值归一化，稳定尺度
            mean_val = p.abs().mean()
            if mean_val > 0:
                p = p / (mean_val + self._min_eps)

            # （2）Top-k 稀疏：只保留 (1 - sparsity) 比例的最大元素
            if 0.0 < self._ipt_sparsity < 1.0:
                flat = p.view(-1)
                k = int((1 - self._ipt_sparsity) * flat.numel())
                if k > 0:
                    # kthvalue 的 index 从 1 开始，因此取 numel - k
                    thresh = flat.abs().kthvalue(
                        max(1, flat.numel() - k)
                    )[0]
                    mask = (p.abs() >= thresh)
                    p = p * mask

            cur_ipt_cpu[n] = p

        # 和历史 Ipt 做任务数加权平均
        if self.Ipt is not None:
            for n, p in cur_ipt_cpu.items():
                if n in self.Ipt:
                    self.Ipt[n] = (self.Ipt[n] * self.num_seen_tasks + p) / (
                        self.num_seen_tasks + 1
                    )
                else:
                    # 新出现的 LoRA 参数
                    self.Ipt[n] = p
        else:
            self.Ipt = cur_ipt_cpu

        self.curIpt = None

    # ------------------- 存 / 读 -------------------
    def save(self):
        rank0_print("[EWC-LoRA] Saving Ipt.")
        if local_rank <= 0:
            os.makedirs(os.path.dirname(self.save_file), exist_ok=True)
            # Ipt 用 float32 存盘，数值更稳
            cpu_Ipt = {n: p.to(torch.float32) for n, p in self.Ipt.items()}
            torch.save(
                {"Ipt": cpu_Ipt, "num_seen_tasks": self.num_seen_tasks + 1},
                self.save_file,
            )
        if dist.is_initialized():
            dist.barrier()

    def load(self):
        if os.path.isfile(self.save_file) and self.Ipt is None:
            rank0_print(f"[EWC-LoRA] Loading Ipt from {self.save_file} ...")
            ckpt = torch.load(self.save_file, map_location="cpu")
            # 读进来先用 bfloat16 存着，必要时再搬到 GPU
            self.Ipt = {n: p.to(torch.bfloat16) for n, p in ckpt["Ipt"].items()}
            self.load_num_seen_tasks = ckpt["num_seen_tasks"]
            # 子进程逐任务模式: 新进程 num_seen_tasks 默认 0, 必须从磁盘重建,
            # 否则 before_train 会跳过 get_Whist + loss() 的 λ 调度算错 → EWC/MAS 静默失效.
            self.num_seen_tasks = ckpt["num_seen_tasks"]
            del ckpt

    # ------------------- hook 部分（在线统计 grad^2） -------------------
    def cal_ipt(self, name):
        """
        每次 backward 时，参数的 grad 触发这个 hook:
        - NaN 防护;
        - grad^2 裁剪;
        - 在 GPU 上算完立即 .cpu() 累加到 CPU 的 curIpt[name] (避免 7B full-FT 时
          全参 curIpt 占满 GPU 14GB → backward OOM).
        """
        def hook(grad):
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_sq = grad.pow(2)
            if self._grad_sq_clip_max is not None:
                grad_sq = torch.clamp(grad_sq, max=self._grad_sq_clip_max)
            self.curIpt[name].add_(grad_sq.detach().to("cpu", dtype=torch.bfloat16))
        return hook

    def rec_iter(self):
        """
        用于记录 backward 的次数（迭代次数），借助某个参数的 grad hook。
        """
        def hook(_grad):
            self.iters += 1
        return hook

    def register_ipt_hooks(self, model):
        """
        在当前模型上注册 grad hook：
        - 选 EWC 目标参数 (LoRA 优先，否则全部可训练参数)
        - 挑第一个挂 rec_iter, 全部挂 cal_ipt
        """
        rank0_print("[EWC] Registering hooks for Ipt (online).")
        self.iters = 0

        ewc_params = list(_select_ewc_params(model))
        if not ewc_params:
            rank0_print("[EWC][WARN] No params selected, EWC will be a no-op.")
            return

        # 挑一个挂 rec_iter
        first_n, first_p = ewc_params[0]
        h = first_p.register_hook(self.rec_iter())
        self.list_of_hooks.append(h)

        # curIpt 放 CPU 累加, 避免 7B full-FT 时 14GB curIpt 撑爆 backward (OOM).
        # hook 里把 grad_sq.cpu() 累加进来.
        self.curIpt = {
            n: torch.zeros(p.shape, device="cpu", dtype=torch.bfloat16)
            for n, p in ewc_params
        }

        # 挂 cal_ipt
        for n, p in ewc_params:
            h = p.register_hook(self.cal_ipt(n))
            self.list_of_hooks.append(h)

    def remove_ipt_hooks(self):
        rank0_print("[EWC-LoRA] Removing hooks for Ipt.")
        for h in self.list_of_hooks:
            h.remove()
        self.list_of_hooks = []
