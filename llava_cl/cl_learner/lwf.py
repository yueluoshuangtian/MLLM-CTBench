import copy
import torch
from .base import BaseCLearner


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class LwFLearner(BaseCLearner):

    def __init__(self, lbd, rank, model):
        """
        lbd: 蒸馏 loss 系数
        rank: local_rank（哪块 GPU）
        model: 当前的 student 模型（带 LoRA / 量化等）
        """
        global local_rank
        local_rank = rank

        self.lbd = lbd
        self.device = torch.device("cuda", rank)

        # 深拷贝一份 teacher 模型
        self.model = copy.deepcopy(model)
        self.model.to(device=self.device, dtype=torch.bfloat16)

        # 冻结 teacher 参数（不参与梯度更新）
        for p in self.model.parameters():
            p.requires_grad_(False)

        # 注意：这里**不**调用 self.model.eval()
        # 让 teacher 保持 training=True，这样在你的 flash-attn 补丁里
        # 会走 _fa_forward_train_only 的 FlashAttention 分支，
        # 而不会走 fallback 的 _ORIG_FORWARD(要求 4D mask)。

    def update_model(self, latest_model):
        """
        在每个任务训练结束后，用最新的 student 权重刷新 teacher。
        """
        latest_state = latest_model.state_dict()
        for name, param in self.model.named_parameters():
            if name in latest_state:
                param.data.copy_(latest_state[name].data)
            else:
                # 正常情况下不会出现，不匹配时打印一下但不中断训练
                rank0_print(f"[LwF] warning: {name} not found in latest_model.state_dict()")

    def loss(self, loss, inputs=None, logits=None, is_first_iteration=False, labels=None, **kwargs):
        """
        loss: 当前 supervised loss（CE + label_smoother 后）
        inputs: Trainer 传入的 batch（包含 input_ids / attention_mask / images / convs 等）
        logits: student 模型的输出 logits
        is_first_iteration: LwF 的第一个任务是否只用 supervised loss
        labels: shift 前的 labels（含 -100）
        """
        kldiv_fct = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)

        # 第一个任务：只用监督 loss，不做蒸馏（避免 teacher = random）
        if is_first_iteration:
            return loss

        # 如果缺少 logits 或 labels 就没法蒸馏，直接返回 supervised loss
        if logits is None or labels is None:
            return loss

        # ---------- 1) teacher 前向 ----------
        # 浅拷贝一份 inputs，避免修改 Trainer 传入的原始 batch
        teacher_inputs = dict(inputs) if inputs is not None else {}

        # 明确保证 teacher 在前向时处于 train 模式（training=True）
        # 这样 llama_flash_attn_monkey_patch._fa_forward_train_only 会走
        # FlashAttention 分支，而不是 fallback 到 _ORIG_FORWARD（4D mask）。
        old_mode = self.model.training
        self.model.train()  # training = True

        with torch.no_grad():
            teacher_logits = self.model(**teacher_inputs).logits

        # 恢复之前的模式（一般来说 old_mode 也是 True，但这里写严谨一点）
        self.model.train(old_mode)

        # ---------- 2) 对齐 student / teacher 的 token 位置 ----------
        # 和你原来的实现保持一致：右移 / 左移对齐
        # student / teacher logits 对应到 labels[..., 1:]
        logits = logits[..., :-1, :].contiguous()
        teacher_logits = teacher_logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        # 只在 labels != -100 的 token 上做 KL 蒸馏
        token_indices = torch.nonzero(labels != -100, as_tuple=True)

        if token_indices[0].numel() == 0:
            # 这一 batch 没有监督 token（可能被截断成全 -100），跳过蒸馏
            rank0_print("[LwF] no valid tokens for distillation in this batch.")
            return loss

        # [N, vocab_size]
        student_tok = logits[token_indices]
        teacher_tok = teacher_logits[token_indices]

        log_p = torch.log_softmax(student_tok, dim=-1)
        q = torch.softmax(teacher_tok, dim=-1)

        distill_loss = self.lbd * kldiv_fct(log_p, q)

        try:
            sup_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
            dist_val = distill_loss.item()
            rank0_print(f"Supervise Loss: {sup_val:.6f}  Distill Loss: {dist_val:.6f}")
        except Exception:
            rank0_print(f"Supervise Loss: {loss}  Distill Loss: {distill_loss}")

        return loss + distill_loss

    def after_train(self, task_id, model, tokenizer=None, **kwargs):
        """
        每个任务训练结束后，用最新的 student 更新 teacher。
        """
        if task_id > 0:
            self.update_model(model)
