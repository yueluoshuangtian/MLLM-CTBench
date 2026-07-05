# cl_learner/moelora.py
import os
import torch

from .base import BaseCLearner
from CoIN.peft.tuners.coinmoelora import CoINMOELoraLinear


class moeloraLearner(BaseCLearner):
    """
    MoE-LoRA 持续学习 Learner（ZeRO2/ZeRO3 友好版思路）：

    - 用 router_full[e] * rank_energy[e,j] 构造 rank 重要性 I_t(e,j)
    - I_t 用 EMA 累积为当前任务的重要性
    - 任务结束后聚合到 I_agg，并保存 A/B 的参考快照
    - 训练时不在 loss 图里加正则项，而是在 backward 之后通过 post_backward()
      直接往 .grad 里加正则梯度，避免 ZeRO2 “重复 reduce 梯度”的问题

    额外支持：
    - moelora_reg_random_importance=True 时，对 I_agg 随机打散作为对照
    """

    def __init__(
        self,
        training_args,
        start_task_id,
        model,
        snapshot_every_n_steps: int = 0,
        snapshot_on_epoch_end: bool = True,
    ):
        self.training_args = training_args
        self.model = model
        self.start_task_id = int(start_task_id) if start_task_id is not None else -1

        # === 超参 ===
        self.lambda_stab = float(getattr(training_args, "moelora_orth_lambda", 0.0))
        self.ema_decay = float(getattr(training_args, "moelora_ema_decay", 0.9))

        # 旧：step-based warmup（作为回退/兼容）
        self.reg_warmup_steps = int(getattr(training_args, "moelora_reg_warmup_steps", 0))
        # 新：按“任务进度比例”warmup（例如 0.3 表示前 30% 进度）
        self.reg_warmup_ratio = float(getattr(training_args, "moelora_reg_warmup_ratio", 0.3))

        self.B_reg_scale = float(getattr(training_args, "moelora_B_reg_scale", 0.05))
        self.importance_clip = float(getattr(training_args, "moelora_importance_clip", 10.0))
        self.importance_top_p = float(getattr(training_args, "moelora_importance_top_p", 0.3))

        # 正则化总开关（默认 True，保持兼容）
        self.reg_enable = bool(getattr(training_args, "moelora_reg_enable", True))

        # 是否使用“随机 rank 重要性”（与真实 I_agg 分布相同，只是打乱位置）
        self.random_importance = bool(getattr(training_args, "moelora_reg_random_importance", False))

        # 记录所有 CoINMOELoraLinear 模块
        self._modules = []
        for name, m in model.named_modules():
            if isinstance(m, CoINMOELoraLinear):
                m._module_name = name
                m._record_enabled = False
                m.usage_recorder = self._record_usage
                self._modules.append((name, m))

        # ====== 聚合的“旧任务参考”（跨任务持久化） ======
        self._I_agg = {}   # module_name -> [E, r_single]
        self._A_ref = {}   # module_name -> [E, r_single, d_in]
        self._B_ref = {}   # module_name -> [E, d_out, r_single]

        # ====== 当前任务运行时统计 ======
        self._cur_I_ema = {}          # 当前任务 EMA(I_t)
        self._cur_I_steps = 0         # 有多少 batch 参与统计
        self._loss_calls_in_task = 0  # 正则应用次数（post_backward 内增加）

        # ====== CL 状态保存根目录 ======
        # ✅ FIX: run.sh 每个 task 的 output_dir 是 "<ROOT>/<k>-sft"
        # 若直接用 output_dir，会导致每个 task 存到不同目录，下一任务读不到上一任务状态
        # 所以这里固定为 output_dir 的父目录 "<ROOT>"
        out_dir = getattr(training_args, "output_dir", ".") or "."
        self._cl_state_root = os.path.dirname(out_dir.rstrip("/")) or "."

        # ====== 打印标记 ======
        self._reg_active_msg_printed = False
        self._reg_inactive_msg_printed = False
        self._reg_disabled_msg_printed = False

        # 尝试加载之前任务的聚合状态
        self._load_prev_cl_state(self.start_task_id)

    # ========== CL 状态持久化 ==========

    def _cl_state_latest_path(self):
        return os.path.join(self._cl_state_root, "moelora_cl_state_latest.pt")

    def _cl_state_task_path(self, task_id: int):
        return os.path.join(self._cl_state_root, f"moelora_cl_state_task_{task_id}.pt")

    def _load_prev_cl_state(self, start_task_id: int):
        """
        start_task_id：上一任务的 task_id（0-based）。允许为 -1（表示没有上一任务）。
        """
        state_path = None
        start_task_id = int(start_task_id)

        # 优先找指定 task 的快照；找不到再退化到 latest
        if start_task_id >= 0:
            cand = self._cl_state_task_path(start_task_id)
            if os.path.exists(cand):
                state_path = cand
            else:
                cand2 = self._cl_state_latest_path()
                if os.path.exists(cand2):
                    state_path = cand2
        else:
            cand = self._cl_state_latest_path()
            if os.path.exists(cand):
                state_path = cand

        if state_path is not None:
            try:
                state = torch.load(state_path, map_location="cpu")
                self._I_agg = state.get("I_agg", {})
                self._A_ref = state.get("A_ref", {})
                self._B_ref = state.get("B_ref", {})
                print(
                    f"[MoE-LoRA CL] Loaded CL state from: {state_path} "
                    f"(modules={len(self._I_agg)})"
                )
            except Exception as e:
                print(f"[MoE-LoRA CL][WARN] Failed to load CL state from {state_path}: {repr(e)}")
        else:
            print("[MoE-LoRA CL] No previous CL state found, start from scratch.")

    def _save_cl_state(self, task_id: int):
        os.makedirs(self._cl_state_root, exist_ok=True)
        state = {
            "I_agg": self._I_agg,
            "A_ref": self._A_ref,
            "B_ref": self._B_ref,
        }
        path_task = self._cl_state_task_path(task_id)
        path_latest = self._cl_state_latest_path()
        try:
            torch.save(state, path_task)
            torch.save(state, path_latest)
            print(
                f"[MoE-LoRA CL] Saved CL state after task {task_id} to:\n"
                f"  - {path_task}\n"
                f"  - {path_latest}"
            )
        except Exception as e:
            print(f"[MoE-LoRA CL][WARN] Failed to save CL state for task {task_id}: {repr(e)}")

    # ========== Trainer 前向里会调用的接口 ==========

    def enable_record(self, flag: bool):
        """
        由 Trainer.compute_loss 在每个前向前/后调用：
        True 时开启 rank_energy/router_full 统计，False 时关闭。
        """
        for _, m in self._modules:
            m._record_enabled = flag

    def loss(self, loss, **kwargs):
        """
        不在 loss 图里叠加正则，正则在 post_backward() 修改 .grad。
        """
        return loss

    # ========== post-backward：在梯度上加正则 ==========

    def post_backward(self, model, cl_task_progress=None):
        """
        在 backward 之后、optimizer.step 之前调用：
        - 直接对 LoRA A/B 的 .grad 加稳定化梯度
        - 不创建新的 autograd 图
        """
        # 1) 开关条件
        if not self.reg_enable:
            if not self._reg_disabled_msg_printed:
                print("[MoE-LoRA CL] Stability regularization DISABLED by `moelora_reg_enable=False`.")
                self._reg_disabled_msg_printed = True
            return

        if self.lambda_stab <= 0.0:
            if not self._reg_disabled_msg_printed:
                print(
                    f"[MoE-LoRA CL] Stability regularization DISABLED "
                    f"(lambda_stab={self.lambda_stab} <= 0)."
                )
                self._reg_disabled_msg_printed = True
            return

        if len(self._I_agg) == 0:
            if not self._reg_inactive_msg_printed:
                print(
                    "[MoE-LoRA CL] Stability regularization NOT ACTIVE yet: "
                    "no previous importance statistics (probably first task)."
                )
                self._reg_inactive_msg_printed = True
            return

        self._loss_calls_in_task += 1
        eps = 1e-6

        # 2) warmup factor（优先 task_progress，其次 step-based）
        if cl_task_progress is not None and self.reg_warmup_ratio > 0.0:
            warmup_factor = min(1.0, float(cl_task_progress) / float(self.reg_warmup_ratio))
        elif self.reg_warmup_steps > 0:
            warmup_steps = max(1, self.reg_warmup_steps)
            warmup_factor = min(1.0, self._loss_calls_in_task / float(warmup_steps))
        else:
            warmup_factor = 1.0

        lambda_eff = self.lambda_stab * warmup_factor
        if lambda_eff <= 0.0:
            return

        # 3) 预处理 I（归一化 + clip + top-p；可选随机打乱）
        with torch.no_grad():
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            module_stats = []  # (m, adapter, I, A_ref_all, B_ref_all, E)
            total_cnt = 0

            for name, m in self._modules:
                if name not in self._I_agg:
                    continue

                I_raw = self._I_agg[name].to(device)  # [E, r]
                if I_raw.numel() == 0:
                    continue

                if self.random_importance:
                    flat = I_raw.view(-1)
                    perm = torch.randperm(flat.numel(), device=flat.device)
                    I_raw = flat[perm].view_as(I_raw)

                I_raw = torch.clamp(I_raw, min=0.0)
                mean_I = I_raw.mean()
                if mean_I.item() < eps:
                    continue

                I_norm = I_raw / (mean_I + eps)
                I_norm = torch.clamp(I_norm, max=self.importance_clip)

                top_p = self.importance_top_p
                if 0.0 < top_p < 1.0:
                    flat2 = I_norm.view(-1)
                    if flat2.numel() > 0:
                        k = int(max(1, round(flat2.numel() * top_p)))
                        if k < flat2.numel():
                            _, idx = torch.topk(flat2, k=k, largest=True, sorted=False)
                            mask_flat = torch.zeros_like(flat2, dtype=torch.bool)
                            mask_flat[idx] = True
                            flat2 = flat2 * mask_flat.to(flat2.dtype)
                            I_norm = flat2.view_as(I_norm)

                I = I_norm

                adapter = m.active_adapter
                E = m._current_E()

                A_ref_all = self._A_ref[name].to(device)  # [E, r, d_in]
                B_ref_all = self._B_ref[name].to(device)  # [E, d_out, r]

                module_stats.append((m, adapter, I, A_ref_all, B_ref_all, E))
                total_cnt += I.numel()

            if total_cnt == 0:
                if not self._reg_inactive_msg_printed:
                    print(
                        "[MoE-LoRA CL][WARN] Importance stats exist but total_cnt==0; "
                        "stability regularization has no effect for this step."
                    )
                    self._reg_inactive_msg_printed = True
                return

            coef = lambda_eff * (2.0 / float(total_cnt))

            # 4) 往 .grad 里加正则梯度
            for (m, adapter, I, A_ref_all, B_ref_all, E) in module_stats:
                for e in range(E):
                    I_e = I[e]  # [r]
                    if I_e.numel() == 0:
                        continue

                    A_cur = m.lora_A[adapter].loraA[e].mlp.weight  # [r, d_in]
                    B_cur = m.lora_B[adapter].loraB[e].mlp.weight  # [d_out, r]

                    if A_cur.grad is None and B_cur.grad is None:
                        continue

                    A_ref = A_ref_all[e].to(A_cur.device)
                    B_ref = B_ref_all[e].to(B_cur.device)

                    I_A = I_e.view(-1, 1).to(A_cur.device, dtype=A_cur.dtype)
                    I_B = I_e.view(1, -1).to(B_cur.device, dtype=B_cur.dtype)

                    if A_cur.grad is not None:
                        dA = coef * I_A * (A_cur - A_ref)
                        A_cur.grad.add_(dA)

                    if B_cur.grad is not None and self.B_reg_scale != 0.0:
                        dB = coef * self.B_reg_scale * I_B * (B_cur - B_ref)
                        B_cur.grad.add_(dB)

            if not self._reg_active_msg_printed:
                print(
                    f"[MoE-LoRA CL] Stability regularization (post-backward) ACTIVE: "
                    f"lambda_stab={self.lambda_stab}, eff={lambda_eff:.4g}, "
                    f"total_cnt={total_cnt}, "
                    f"warmup_ratio={self.reg_warmup_ratio}, "
                    f"warmup_steps={self.reg_warmup_steps}, "
                    f"random_importance={self.random_importance}."
                )
                self._reg_active_msg_printed = True

    # ========== Trainer 生命周期调用接口 ==========

    def before_train(self, task_id, model, tokenizer=None, **kwargs):
        self._cur_I_ema = {}
        self._cur_I_steps = 0
        self._loss_calls_in_task = 0
        self._reg_active_msg_printed = False
        self._reg_inactive_msg_printed = False
        self._reg_disabled_msg_printed = False

    def after_train(self, task_id, model, tokenizer=None, **kwargs):
        if self._cur_I_steps == 0:
            print(f"[MoE-LoRA CL] after_train: no rank_energy collected for task {task_id}.")
            return

        eps = 1e-6

        for name, m in self._modules:
            if name not in self._cur_I_ema:
                continue

            I_t = self._cur_I_ema[name]  # [E, r]
            adapter = m.active_adapter
            E = m._current_E()

            A_list = [m.lora_A[adapter].loraA[e].mlp.weight.detach().cpu() for e in range(E)]
            B_list = [m.lora_B[adapter].loraB[e].mlp.weight.detach().cpu() for e in range(E)]
            A_t = torch.stack(A_list, dim=0)  # [E, r, d_in]
            B_t = torch.stack(B_list, dim=0)  # [E, d_out, r]

            if name not in self._I_agg:
                self._I_agg[name] = I_t.detach().cpu()
                self._A_ref[name] = A_t.clone()
                self._B_ref[name] = B_t.clone()
            else:
                I_prev = self._I_agg[name]
                A_prev = self._A_ref[name]
                B_prev = self._B_ref[name]

                I_new = I_prev + I_t
                w_prev_A = (I_prev / (I_new + eps)).unsqueeze(-1)
                w_t_A = (I_t / (I_new + eps)).unsqueeze(-1)
                A_new = w_prev_A * A_prev + w_t_A * A_t

                w_prev_B = (I_prev / (I_new + eps)).unsqueeze(1)
                w_t_B = (I_t / (I_new + eps)).unsqueeze(1)
                B_new = w_prev_B * B_prev + w_t_B * B_t

                self._I_agg[name] = I_new.detach().cpu()
                self._A_ref[name] = A_new.detach().cpu()
                self._B_ref[name] = B_new.detach().cpu()

        self._cur_I_ema = {}
        self._cur_I_steps = 0

        self._save_cl_state(int(task_id))

        # 单独保存 I_agg 快照（可视化）
        try:
            viz_root = os.path.join(self._cl_state_root, "I_snapshots")
            os.makedirs(viz_root, exist_ok=True)
            snap = {name: I.clone() for name, I in self._I_agg.items()}
            out_path = os.path.join(viz_root, f"I_agg_task{task_id}.pt")
            torch.save(snap, out_path)
            print(f"[MoE-LoRA CL] Saved I_agg snapshot for task {task_id} to {out_path}")
        except Exception as e:
            print(f"[MoE-LoRA CL][WARN] failed to save I_agg snapshot for task {task_id}: {repr(e)}")

    # ========== forward 内 usage_recorder 回调 ==========

    def _record_usage(self, payload: dict):
        """
        payload:
          - module: 模块名
          - router_full: [E] batch 平均 gating 概率（softmax 后）
          - rank_energy: [E, r] batch 内 (A_e^T x)^2 的 rank 维均值

        构造 I_t(e,j) ∝ router_full[e] * rank_energy[e,j]，并做 EMA。
        """
        name = payload.get("module", None)
        rank_energy = payload.get("rank_energy", None)   # [E, r]
        router_full = payload.get("router_full", None)   # [E]

        if name is None or rank_energy is None:
            return

        if router_full is not None:
            try:
                rf = router_full.to(rank_energy.device).to(rank_energy.dtype)
                if rf.dim() == 1:
                    rf = rf.view(-1, 1)  # [E, 1]
                if rf.size(0) == rank_energy.size(0):
                    rank_energy = rank_energy * rf
                else:
                    print(
                        f"[MoE-LoRA CL][WARN] router_full.shape={tuple(router_full.shape)} "
                        f"does not match rank_energy.shape={tuple(rank_energy.shape)}; "
                        f"skip expert-level weighting for module {name}."
                    )
            except Exception as e:
                print(
                    f"[MoE-LoRA CL][WARN] failed to apply router_full weighting in _record_usage "
                    f"for module {name}: {repr(e)}"
                )

        if name not in self._cur_I_ema:
            self._cur_I_ema[name] = rank_energy.clone()
        else:
            beta = self.ema_decay
            self._cur_I_ema[name] = beta * self._cur_I_ema[name] + (1.0 - beta) * rank_energy.clone()

        self._cur_I_steps += 1
