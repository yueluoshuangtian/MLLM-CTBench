import copy
import torch
from .base import BaseCLearner

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class OLoRALearner(BaseCLearner):
    """
    O-LoRA 连续学习策略：
    - 冻结的旧 LoRA_A 作为 {A_i}（历史子空间，使用 clone() 快照，不参与梯度）
    - 当前任务可训练的 LoRA_A 作为 A_t（直接引用 param，本身参与反向传播）
    - 在 loss 中加入正交约束：
        L_total = L_sup + λ * mean_{i,t} || A_i^T A_t ||_F^2
    """

    def __init__(self, lbd, rank, model, lora_r):
        """
        保持原有签名：__init__(self, lbd, rank, model, lora_r)
        注意：这里不再 deepcopy model，也不修改 model，只保存超参。
        """
        global local_rank
        local_rank = rank

        self.lbd = float(lbd)
        self.device = torch.device("cuda", rank)
        self.lora_r = lora_r

        # 这两个字典在每个任务开始前，通过 before_train() 重新构建
        self.prev_lora_As = {}  # 冻结的旧 LoRA_A（快照）
        self.now_lora_At = {}   # 当前任务的 LoRA_A（参数引用）

    # --------- 每个任务开始前调用，收集 LoRA 参数 ---------
    def before_train(self, task_id, model, tokenizer=None, **kwargs):
        """
        在主训练循环中，每个 task 进入训练前会调用：
            cl_learner.before_train(task_id, model, tokenizer, **data_module)

        这里根据当前 model 的 requires_grad 状态，区分：
        - prev_lora_As：所有冻结的 lora_A（旧任务）
        - now_lora_At：所有可训练的 lora_A（当前任务）
        """
        self.prev_lora_As = {}
        self.now_lora_At = {}

        for name, param in model.named_parameters():
            if "lora_A" not in name:
                continue

            # 旧任务 LoRA：冻结（requires_grad=False），只需要快照
            if not param.requires_grad:
                self.prev_lora_As[name] = param.detach().clone()

            # 当前任务 LoRA：可训练（requires_grad=True），需要保留引用以参与梯度
            else:
                self.now_lora_At[name] = param  # 不要 clone/detach

        if local_rank in (0, -1):
            rank0_print(
                f"[OLoRA][before_train] task_id={task_id} "
                f"prev_lora_As={len(self.prev_lora_As)}, now_lora_At={len(self.now_lora_At)}"
            )

    # --------- 在 Trainer 的 loss 计算阶段调用 ---------
    def loss(self, loss, **kwargs):
        """
        Trainer 每个 step 会调用 cl_learner.loss(original_loss, **extra),
        这里在原有监督 loss 基础上加上正交约束项。

        参数:
            loss: 原始监督损失 (一个标量 tensor)
            其它 kwargs: 保持兼容（logits, inputs, labels 等），这里不强依赖
        """
        # 没有历史 LoRA（第一个任务），只返回监督 loss
        if len(self.prev_lora_As) == 0 or len(self.now_lora_At) == 0:
            rank0_print(f"[OLoRA] no previous LoRA, Supervise Loss: {loss}")
            return loss

        total_orth_loss = 0.0
        num_pairs = 0

        # 遍历当前任务的每个 LoRA_A^t
        for name_t, A_t in self.now_lora_At.items():
            # name 类似 "...layer...something.lora_A.weight"，
            # 用前缀匹配同一层的旧 LoRA
            layer_prefix = name_t.split(".lora_A")[0]

            for name_i, A_i in self.prev_lora_As.items():
                if not name_i.startswith(layer_prefix):
                    continue

                # A_i: [out_dim, r], A_t: [out_dim, r]
                # O_{i,t} = A_i^T A_t : [r, r]
                O_i_t = torch.matmul(A_i.T, A_t)
                total_orth_loss = total_orth_loss + (O_i_t ** 2).sum()
                num_pairs += 1

        if num_pairs > 0:
            total_orth_loss = total_orth_loss / num_pairs

        orth_loss = self.lbd * total_orth_loss

        rank0_print(f"[OLoRA] Supervise Loss: {loss}  Orth Loss: {orth_loss}")
        return loss + orth_loss

    # --------- 每个任务训练结束后调用 ---------
    def after_train(self, task_id, model, data_modules, **kwargs):
        """
        与其它 CLearner 保持接口一致。
        你原来在这里把 training_args.lora_enable 置 False，
        如果希望最后任务后禁止继续加新 LoRA，可以保留；不需要也可以删掉。
        """
        training_args = kwargs.get("training_args", None)
        if task_id == len(data_modules):
            # 可选：最后一个任务后不再继续启用 LoRA
            # 如果后续还要接着做别的任务，可以注释掉这一行
            training_args.lora_enable = False

        # 清理缓存的快照，防止显存/内存堆积
        self.prev_lora_As = {}
        self.now_lora_At = {}
