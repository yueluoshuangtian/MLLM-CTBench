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
    """只把 LoRA 的 A/B 权重当成 EWC 对象。"""
    return ("lora_A" in name) or ("lora_B" in name)


class EWCLoraLearner(BaseCLearner):
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
        总 loss = 监督 loss + 有效 λ * EWC 正则。
        有效 λ 会根据已见任务数做简单调度：lambda_t = lambda / sqrt(t)
        """
        reg_loss = 0.0
        if self.Ipt is not None:
            base_model = unwrap_model(model)
            for n, p in base_model.named_parameters():
                if not p.requires_grad:
                    continue
                if not _is_lora_param(n):
                    continue
                if n not in self.Ipt or n not in self.Whist:
                    # 新出现的 LoRA 参数（结构变化），没有历史 Ipt/Whist，跳过
                    continue
                reg_loss = reg_loss + (self.Ipt[n] * (p - self.Whist[n]).pow(2) / 2).sum()

            # 任务相关 λ 调度：越往后任务，约束略弱，避免完全学不动
            # 当前正在训练的任务 id ≈ 已完成任务数 + 1
            cur_task_id = self.num_seen_tasks + 1
            effective_lbd = self.lbd / (cur_task_id ** 0.5 if cur_task_id > 0 else 1.0)

            reg_loss = effective_lbd * reg_loss
            rank0_print(
                f"[EWC-LoRA] Supervise Loss: {loss}  "
                f"Reg Loss: {reg_loss}  lambda_eff={effective_lbd:.4f}"
            )

        return loss + reg_loss

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
            # （3）把 Ipt 搬到当前 device
            self.Ipt = {n: p.to(self.device) for n, p in self.Ipt.items()}

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
        rank0_print("[EWC-LoRA] Get Whist for regularization.")
        base_model = unwrap_model(model)
        self.Whist = {
            n: p.detach().clone().to(device=self.device, dtype=torch.bfloat16)
            for n, p in base_model.named_parameters()
            if p.requires_grad and _is_lora_param(n)
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
            del ckpt

    # ------------------- hook 部分（在线统计 grad^2） -------------------
    def cal_ipt(self, name):
        """
        每次 backward 时，LoRA 参数的 grad 会触发这个 hook：
        - 做 NaN 防护；
        - grad^2 裁剪，避免极端值；
        - 累加到 self.curIpt[name]。
        """
        def hook(grad):
            # 防止 NaN / Inf
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_sq = grad.pow(2)

            # 简单裁剪，避免少数极端 batch 主宰 Ipt
            if self._grad_sq_clip_max is not None:
                grad_sq = torch.clamp(grad_sq, max=self._grad_sq_clip_max)

            self.curIpt[name] += grad_sq
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
        - 先找一个 LoRA 参数挂 rec_iter hook 统计 step 数；
        - 为所有 LoRA 参数分配 curIpt 累积张量；
        - 为所有 LoRA 参数挂 cal_ipt hook，累积 grad^2。
        """
        rank0_print("[EWC-LoRA] Registering hooks for Ipt (online).")
        base_model = unwrap_model(model)

        # 1) 初始化计数器
        self.iters = 0

        # 2) 挑一个 LoRA 参数挂 rec_iter，用于统计迭代次数
        for n, p in base_model.named_parameters():
            if p.requires_grad and _is_lora_param(n):
                h = p.register_hook(self.rec_iter())
                self.list_of_hooks.append(h)
                break

        # 3) 为每个 LoRA 参数分配一个累积张量
        self.curIpt = {
            n: torch.zeros_like(p).to(device=self.device, dtype=torch.bfloat16)
            for n, p in base_model.named_parameters()
            if p.requires_grad and _is_lora_param(n)
        }

        # 4) 为每个 LoRA 参数挂 grad^2 累积 hook
        for n, p in base_model.named_parameters():
            if p.requires_grad and _is_lora_param(n):
                h = p.register_hook(self.cal_ipt(n))
                self.list_of_hooks.append(h)

    def remove_ipt_hooks(self):
        rank0_print("[EWC-LoRA] Removing hooks for Ipt.")
        for h in self.list_of_hooks:
            h.remove()
        self.list_of_hooks = []
