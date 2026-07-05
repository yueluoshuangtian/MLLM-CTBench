import os
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.utils.data import Sampler
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
    unwrap_model,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)
from transformers import TrainerCallback

from llava.data.dataloader import LengthGroupedSampler


# ============================================================
#              Zero-3 相关辅助（mm_projector 保存）
# ============================================================
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


# ============================================================
#            MoE-LoRA Snapshot Callback（按 epoch）
# ============================================================
class MoELoRASnapshotCallback(TrainerCallback):
    """
    训练过程中在每个 epoch 结束时，调用 learner.on_epoch_end
    用于 MoE-LoRA 的参数 / 统计快照（比如保存 I_{e,j}(t) 等）。
    """

    def __init__(self, learner):
        self.learner = learner

    def on_step_end(self, args, state, control, **kwargs):
        # 目前不在 step 级别做任何事，保留接口占位
        if hasattr(self.learner, "on_step_end"):
            ...
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if hasattr(self.learner, "on_epoch_end") and state.epoch is not None:
            try:
                epoch = int(state.epoch)
            except Exception:
                epoch = None
            if epoch is not None:
                # learner 内部自己决定是否利用 epoch / out_dir 做 snapshot
                self.learner.on_epoch_end(epoch=epoch, out_dir=args.output_dir)
        return control


# ============================================================
#               MoE 辅助：all-reduce + balance loss
# ============================================================
def _all_reduce_mean(v: torch.Tensor):
    # 为了 ZeRO2 安全，可以只在统计/监控场景使用。
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return v
    v = v.clone()
    torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.SUM)
    v = v / torch.distributed.get_world_size()
    return v


def global_mean_with_local_grad(v: torch.Tensor):
    """
    一个 trick：让前向里看到的是“全局均值”，但反向里只对本地 v 有梯度。
    这里简单地做： v + (all_reduce_mean(v).detach() - v.detach())
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return v
    with torch.no_grad():
        g = _all_reduce_mean(v)
    return v + (g - v).detach()


def moe_balance_loss(model, variant="switch", alpha=1e-2, imp_w=1.0, load_w=1.0):
    """
    MoE 负载均衡正则（可选用）：
    - variant="switch": sum_e f_e * p_e
    - variant="cv2": 基于 CV^2 的负载 / 重要度均衡
    """
    aux = torch.zeros([], device=next(model.parameters()).device, dtype=torch.float32)
    n_layers = 0
    for m in model.modules():
        P = getattr(m, "_last_P", None)
        f = getattr(m, "_last_f", None)
        if P is None or f is None:
            continue
        # P 要保留梯度，f 只当统计
        P = global_mean_with_local_grad(P.float())  # 可导
        f = _all_reduce_mean(f.float())             # 不可导统计

        if variant == "switch":
            E = float(P.numel())
            aux = aux + alpha * E * (f * P).sum()
        else:
            def cv2(v):
                m_ = v.mean()
                return (v.var(unbiased=False) / (m_ * m_ + 1e-9))

            E = float(P.numel())
            load = f
            aux = aux + (imp_w * E * cv2(P) + load_w * E * cv2(load))

        n_layers += 1

    if n_layers > 0:
        aux = aux / n_layers
    return aux


# ============================================================
#                     LLaVATrainer
# ============================================================

class LLaVATrainer(Trainer):
    def __init__(self, cl_learner=None, is_first_iteration: bool = False, **kwargs):
        """
        cl_learner: 一个持续学习 learner（EWC/MAS/LwF/eproj/moelora...）
        is_first_iteration: 仅在 LwF 下，第一个任务不使用蒸馏 loss。
        """
        super().__init__(**kwargs)
        self.cl_learner = cl_learner
        self.is_first_iteration = is_first_iteration

        # 若传入了 cl_learner，则挂载 MoE-LoRA snapshot 回调（在 cl == "moelora" 情况下真正会用到）
        if cl_learner is not None:
            try:
                from .callbacks import MoELoRASnapshotCallback
                self.add_callback(MoELoRASnapshotCallback(self.cl_learner))
            except Exception:
                pass

    # ----------------- Sampler -----------------
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if getattr(self.args, "group_by_modality_length", False):
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    # ----------------- Optimizer -----------------
    def create_optimizer(self):
        """
        Setup the optimizer.

        保持你原来项目里的优化器分组逻辑不变：
        - (可选) 对 mm_projector / e_mm_projector 单独设 lr；
        - (可选) 对 eproj 的 key 向量单独设 lr。
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            # do not weight decay the task_key for eproj method since it updates the task_key in all tasks
            decay_parameters = [
                name
                for name in decay_parameters
                if "bias" not in name and "e_img_key" not in name and "e_inst_key" not in name
            ]

            if getattr(self.args, "mm_projector_lr", None) is not None:
                projector_parameters = [
                    name for name, _ in opt_model.named_parameters()
                    if "mm_projector" in name or "e_mm_projector" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            elif getattr(self.args, "eproj_key_lr", None) is not None:
                eproj_key_parameters = [
                    name for name, _ in opt_model.named_parameters()
                    if "e_img_key" in name or "e_inst_key" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in eproj_key_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in eproj_key_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in eproj_key_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.eproj_key_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in eproj_key_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.eproj_key_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            # 8bit Adam 里对 Embedding 做 override（你原来的逻辑）
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        print(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        print(f"bitsandbytes: will optimize {module} in fp32")
                print(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    # ----------------- Checkpoint 保存：mm_projector 特例 -----------------
    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # 只保存 mm_projector / e_mm_projector（以及可选的 embed_tokens / embed_in）
            keys_to_match = ["mm_projector", "e_mm_projector", "vision_resampler"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, "mm_projector.bin"))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            # mm_projector-only 模式下，_save_checkpoint 已经处理，这里跳过
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    # ----------------- Loss 计算：挂接 CL / MoE-LoRA -----------------
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        这里做三件事：
        1) 正常的监督 loss（label_smoother）；
        2) 在 cl_learner.enable_record(True/False) 的包裹下，触发 QRMoE-LoRA 模块的 rank-1 统计；
        3) 把 task loss 交给 cl_learner.loss 做持续学习正则（EWC/MAS/LwF/eproj/moelora...）。
        """
        # ---- 处理 labels（与 HF Trainer 逻辑一致）----
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        new_labels = None

        # ====== 1) MoE-LoRA 统计：开启记录（仅主前向）======
        if hasattr(self, "cl_learner") and hasattr(self.cl_learner, "enable_record"):
            # 只在主前向打开；梯度检查点的重算阶段我们不会再次打开
            self.cl_learner.enable_record(True)

        # ---- 调用模型前向 ----
        if hasattr(self.args, "cl") and self.args.cl == "lwf":
            # LwF: 前向需要输出旧 logits 对应的新 labels
            outputs, new_labels = model(output_labels=True, **inputs)
        else:
            outputs = model(**inputs)

        # ====== 2) MoE-LoRA 统计：关闭记录（避免重算重复累积）======
        if hasattr(self, "cl_learner") and hasattr(self.cl_learner, "enable_record"):
            self.cl_learner.enable_record(False)

        # ====== 3) 基础 task loss（不含任何 CL 正则）======
        if labels is not None:
            valid = (labels != -100).sum().item()
            if valid == 0:
                print("[WARN] no supervised tokens in this batch (likely due to truncation)")

            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                # causal LM: 默认 shift labels
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # ====== 4) 计算当前任务训练进度（0~1），供 CL warmup 使用 ======
        if self.state.epoch is None:
            task_progress = None
        else:
            cur_epoch = float(self.state.epoch)
            num_epochs = float(self.args.num_train_epochs)
            task_progress = max(0.0, min(1.0, cur_epoch / max(num_epochs, 1e-6)))

        # ====== 5) 持续学习 learner 的附加正则 ======
        if hasattr(self, "cl_learner") and self.cl_learner is not None:
            if getattr(self.args, "cl", None) == "lwf":
                # LwF: 继承你原来的接口（增加旧模型蒸馏项）
                loss = self.cl_learner.loss(
                    loss,
                    model=model,
                    inputs=inputs,
                    logits=outputs.logits,
                    labels=new_labels,
                    is_first_iteration=self.is_first_iteration,
                )
            elif getattr(self.args, "cl", None) == "moelora":
                # MoE-LoRA：额外传入当前任务进度，用于 epoch-ratio warmup
                loss = self.cl_learner.loss(
                    loss,
                    model=model,
                    inputs=inputs,
                    logits=outputs.logits,
                    labels=new_labels,
                    cl_task_progress=task_progress,
                )
            else:
                # 其他 CL 策略（EWC/MAS/eproj/OLoRA...）：接口保持不变
                loss = self.cl_learner.loss(
                    loss,
                    model=model,
                    inputs=inputs,
                    logits=getattr(outputs, "logits", None),
                    labels=labels,
                )

        return (loss, outputs) if return_outputs else loss

    # --------- 训练 step：只对 moelora 做自定义 backward + post_backward ---------
    def training_step(self, model, inputs):
        """
        在 HF Trainer 的 training_step 基础上，仅在 cl == "moelora" 时：
        - 自己调用 backward；
        - 在 backward 之后调用 moeloraLearner.post_backward 加正则梯度。

        对于 OLoRA / EWC / MAS / LwF / eproj 等其他 cl 方法：
        - 直接回退到 super().training_step，完全沿用官方逻辑，
          避免 deepspeed ZeRO 出现“梯度重复归约”的断言。
        """
        # 非 moelora：走原生 HF Trainer 的 training_step（不会触发重复 all-reduce）
        if not (
            getattr(self, "cl_learner", None) is not None
            and getattr(self.args, "cl", None) == "moelora"
            and hasattr(self.cl_learner, "post_backward")
        ):
            return super().training_step(model, inputs)

        # ====== 以下逻辑仅在 cl == "moelora" 时执行 ======
        model.train()
        inputs = self._prepare_inputs(inputs)

        # 标准 HF / Accelerate 损失计算
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        # 标准 backward（ZeRO2 / fp16 等由 accelerate/deepspeed 接管）
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            self.accelerator.backward(loss)

        # 计算任务进度（与 compute_loss 中保持一致）
        if self.state.epoch is None:
            task_progress = None
        else:
            cur_epoch = float(self.state.epoch)
            num_epochs = float(self.args.num_train_epochs)
            task_progress = max(0.0, min(1.0, cur_epoch / max(num_epochs, 1e-6)))

        # 解包底层模型（兼容 Deepspeed / DDP / FSDP）
        raw_model = unwrap_model(model)

        # 在已经完成一次标准 backward 后，对 MoE-LoRA 的参数显式加正则梯度
        self.cl_learner.post_backward(raw_model, cl_task_progress=task_progress)

        # 与 HF Trainer 一致，返回归一化 loss
        return loss.detach() / self.args.gradient_accumulation_steps
