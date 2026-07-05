# -*- coding: utf-8 -*-
import warnings
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
    ModulesToSaveWrapper,
)
from .lora import (
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
    Embedding,
    Conv2d,
)

try:
    from .lora import Linear8bitLt, Linear4bit
    _HAS_BNB = True
except Exception:
    Linear8bitLt = None
    Linear4bit = None
    _HAS_BNB = False

from ..import_utils import is_bnb_4bit_available, is_bnb_available

if is_bnb_available():
    import bitsandbytes as bnb


# =========================================================
# CoIN-MoE-LoRA Config（多“整 LoRA”专家）
# =========================================================
@dataclass
class CoINMOELoraConfig(LoraConfig):
    """
    Configuration for CoIN-MoE-LoRA with QR/A/MLP-based gating.

    和原版区别：
    - 不再把 rank=r 按秩均分，而是每个 expert 都是完整的 rank=r LoRA；
    - expert_num 控制“整 LoRA 专家”的个数。
    """

    expert_num: int = field(default=2)   # LoRA 专家个数（每个是 rank=r）
    gate_rank: int = field(default=16)   # 门控子空间列数 / MLP hidden dim
    gate_tau: float = field(default=1.6) # softmax 温度
    gate_tie: str = field(default="A_T") # "A_T" 或 "free"
    gate_router: str = field(default="qr")  # "qr" | "a" | "mlp"

    def __post_init__(self):
        self.peft_type = PeftType.MOE_LORA_CoIN

        if self.expert_num is None or self.expert_num <= 0:
            raise ValueError("CoIN-MoE-LoRA 需要 expert_num 为正整数。")

        # 注意：这里不再要求 r 能被 expert_num 整除，每个 expert 都是完整 rank=r。
        if hasattr(self, "r") and self.r is not None and self.r <= 0:
            raise ValueError("LoRA rank r 必须为正。")

        if self.gate_rank <= 0:
            raise ValueError("`gate_rank` 必须为正数")
        if self.gate_tau <= 0.0:
            raise ValueError("`gate_tau` 必须大于 0")

        # router 模式
        if self.gate_router not in ("qr", "a", "mlp"):
            raise ValueError("`gate_router` 必须为 'qr'、'a' 或 'mlp'")


# =========================================================
# CoIN-MoE-LoRA Model Wrapper
# =========================================================
class CoINMOELoraModel(LoraModel):
    """
    按照 PEFT 框架把 Linear/Conv1D 等模块替换为 CoINMOELoraLinear，
    其内部实现“多个整 LoRA 专家 + QR/A/MLP gating”。
    """

    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    # -------- Adapter 注册与模块替换 --------
    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = (
                self.model.config.to_dict()
                if hasattr(self.model.config, "to_dict")
                else self.model.config
            )
            config = self._prepare_coinmoelora_config(config, model_config)
            self.peft_config[adapter_name] = config

        self._find_and_replace(adapter_name)

        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "CoINMOELoraModel supports only 1 adapter with bias. "
                "When using multiple adapters, set bias to 'none' for all adapters."
            )

        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)

        # gate_tie="free" 时，确保 gate_P 可训练；gate_router="mlp" 时，确保 router_mlp 可训练
        for mod in self.model.modules():
            if isinstance(mod, CoINMOELoraLinear):
                if mod.gate_tie == "free" and hasattr(mod, "gate_P") and mod.gate_P is not None:
                    for p in mod.gate_P:
                        p.requires_grad_(True)
                if getattr(mod, "gate_router", None) == "mlp" and hasattr(mod, "router_mlp") and mod.router_mlp is not None:
                    for p in mod.router_mlp.parameters():
                        p.requires_grad_(True)

        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]

        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue
            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)

            # 已经是 LoraLayer，则复用其包装
            if isinstance(target, LoraLayer) and isinstance(target, nn.Conv2d):
                target.update_layer_conv2d(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer) and isinstance(target, nn.Embedding):
                target.update_layer_embedding(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            else:
                # 用 CoINMOELoraLinear / Conv2d / Embedding 包装
                new_module = self._create_new_module(lora_config, adapter_name, target)
                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model."
            )

    def _create_new_module(self, lora_config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "expert_num": lora_config.expert_num,
            "gate_rank": getattr(lora_config, "gate_rank", 16),
            "gate_tau": getattr(lora_config, "gate_tau", 1.6),
            "gate_tie": getattr(lora_config, "gate_tie", "A_T"),
            "gate_router": getattr(lora_config, "gate_router", "qr"),  # "qr"|"a"|"mlp"
        }

        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        # --- 8bit ---
        if loaded_in_8bit and _HAS_BNB and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
            )
            return new_module

        # --- 4bit ---
        if loaded_in_4bit and is_bnb_4bit_available() and _HAS_BNB and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(
                adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs
            )
            return new_module

        # --- Embedding ---
        if isinstance(target, nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
            new_module.to(device=target.weight.device, dtype=target.weight.dtype)
            with torch.no_grad():
                new_module.weight.copy_(target.weight)
            new_module.weight.requires_grad_(False)
            return new_module

        # --- Conv2d ---
        if isinstance(target, nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(
                adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs
            )
            new_module.to(device=target.weight.device, dtype=target.weight.dtype)
            with torch.no_grad():
                new_module.weight.copy_(target.weight)
                if target.bias is not None and new_module.bias is not None:
                    new_module.bias.copy_(target.bias)
            new_module.weight.requires_grad_(False)
            if new_module.bias is not None:
                new_module.bias.requires_grad_(False)
            return new_module

        # --- Linear / Conv1D ---
        if isinstance(target, nn.Linear):
            in_features, out_features = target.in_features, target.out_features
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is True but target is torch.nn.Linear. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        elif isinstance(target, Conv1D):
            in_features, out_features = (
                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
            )
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is False but target is Conv1D. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Only `torch.nn.Linear` and `Conv1D` are supported."
            )

        new_module = CoINMOELoraLinear(adapter_name, in_features, out_features, bias=bias, **kwargs)
        new_module.to(device=target.weight.device, dtype=target.weight.dtype)
        with torch.no_grad():
            if getattr(new_module, "fan_in_fan_out", False):
                if new_module.weight.shape == target.weight.shape:
                    new_module.weight.copy_(target.weight)
                elif (
                    new_module.weight.shape[0] == target.weight.shape[1]
                    and new_module.weight.shape[1] == target.weight.shape[0]
                ):
                    new_module.weight.copy_(target.weight.T)
                else:
                    raise RuntimeError(
                        f"Shape mismatch when copying weight with fan_in_fan_out=True: "
                        f"new={tuple(new_module.weight.shape)}, old={tuple(target.weight.shape)}"
                    )
            else:
                new_module.weight.copy_(target.weight)
            if bias and getattr(target, "bias", None) is not None and new_module.bias is not None:
                new_module.bias.copy_(target.bias)

        new_module.weight.requires_grad_(False)
        if bias and new_module.bias is not None:
            new_module.bias.requires_grad_(False)
        new_module.to(device=target.weight.device, dtype=target.weight.dtype)
        return new_module

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @staticmethod
    def _prepare_coinmoelora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def _unload_and_optionally_merge(self, merge=True):
        # 与 PEFT 对齐：量化模式下不合并
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in quantized mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                else:
                    bias = target.bias is not None
                    if getattr(target, "is_target_conv_1d_layer", False):
                        new_module = Conv1D(target.out_features, target.in_features)
                    else:
                        new_module = nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    target.merge()

            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model


# =========================================================
# 基础 CoIN-MoE-LoRA 层（多整 LoRA 专家）
# =========================================================
class CoINMOELoraLayer(LoraLayer):
    def __init__(self, in_features: int, out_features: int, expert_num: int):
        super().__init__(in_features, out_features)
        self.expert_num = expert_num

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        # LoRA 超参登记
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha

        # dropout
        lora_dropout_layer = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        # 构建 A/B experts（每个 expert 一个完整 rank=r 的 LoRA）
        if r > 0:
            self.lora_A.update(
                nn.ModuleDict({adapter_name: CoINMOELinearA(self.in_features, r, self.expert_num)})
            )
            self.lora_B.update(
                nn.ModuleDict({adapter_name: CoINMOELinearB(r, self.out_features, self.expert_num)})
            )
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            for i in range(self.expert_num):
                nn.init.normal_(self.lora_A[adapter_name].loraA[i].mlp.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.lora_B[adapter_name].loraB[i].mlp.weight)


# =========================================================
# CoIN-MoE-LoRA Linear：多整 LoRA + QR/A/MLP gating + Rank-1 统计
# =========================================================
class CoINMOELoraLinear(nn.Linear, CoINMOELoraLayer):
    """
    主干 Linear 冻结 + 多整 LoRA 专家 + 基于 QR / A / 向量打分(gate_router='mlp') 的 gating。

    - 每个 expert 是一个完整的 rank=r LoRA；
    - router:
        * "qr": 对每个 expert 的 A(e)^T 或 gate_P(e) 做 QR 得到 Q(e)，gating 用 Q(e)^T x 的能量；
        * "a" : 直接用 A(e)^T x 的能量；
        * "mlp": 现在实现为「每个 expert 一个向量 w_e，score_e(x) = w_e^T x」；
    - 所有 experts 的输出按 softmax 权重加权；
    - forward 末尾记录 rank-1 能量，供持续学习正则使用。
    """

    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        if "expert_num" not in kwargs:
            raise ValueError("`expert_num` must be provided for CoINMOELoraLinear.")
        self.expert_num = int(kwargs.pop("expert_num"))
        self.expert_use_num = int(kwargs.pop("expert_use_num", 2))  # 兼容字段（未使用）

        # QR / A / MLP gating 超参
        self.gate_rank = int(kwargs.pop("gate_rank", 16))
        self.gate_tau = float(kwargs.pop("gate_tau", 1.6))
        self.gate_tie = str(kwargs.pop("gate_tie", "A_T"))  # "A_T" or "free"
        self.gate_router = str(kwargs.pop("gate_router", "qr"))  # "qr" | "a" | "mlp"
        if self.gate_router not in ("qr", "a", "mlp"):
            raise ValueError("`gate_router` must be 'qr', 'a' or 'mlp'.")

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        CoINMOELoraLayer.__init__(self, in_features=in_features, out_features=out_features, expert_num=self.expert_num)

        # 门控底座 P：free 模式显式；A_T 模式 runtime 从 A^T 构造
        if self.gate_tie == "free":
            self.gate_P = nn.ParameterList(
                [
                    nn.Parameter(0.02 * torch.randn(self.in_features, self.gate_rank))
                    for _ in range(self.expert_num)
                ]
            )
        else:
            self.gate_P = None

        # === 新版 "mlp" router：每个 expert 一条向量 w_e，score_e(x) = w_e^T x ===
        if self.gate_router == "mlp":
            # router_vec: [E, d_in]，第 e 行是 w_e
            self.router_vec = nn.Parameter(
                0.02 * torch.randn(self.expert_num, self.in_features)
            )
            # 保留 router_mlp 属性为 None，以兼容外部代码中的 hasattr / is not None 判断
            self.router_mlp = None
        else:
            self.router_mlp = None
            self.router_vec = None

        # 专家 mask：1=可用；0=屏蔽
        self.register_buffer("_expert_mask", torch.ones(self.expert_num))

        # 外部 learner 注入的统计回调
        self.usage_recorder = None
        self._record_enabled = False   # 由 learner 控制
        self._module_name = None       # 由 learner 写入模块名（model.named_modules 的 key）

        # 冻结主干 Linear
        self.weight.requires_grad = False
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        # 构建 LoRA experts
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name
        self.merged = False

        # 监控缓存（可用于 moe_balance_loss 或可视化）
        self._last_P = None
        self._last_f = None

        # 原 warning：qr 模式下 gate_rank>d_in 有点浪费，这里保持不变
        if self.gate_rank > self.in_features and self.gate_router in ("qr", "mlp"):
            warnings.warn(
                f"[CoIN-MoE-LoRA] gate_rank={self.gate_rank} > d_in={self.in_features}; "
                f"QR/MLP 实际只会使用前 d_in 维。"
            )

    # ---------- helpers ----------

    def _current_E(self) -> int:
        return len(self.lora_A[self.active_adapter].loraA)

    def set_mask(self, mask: torch.Tensor):
        self._expert_mask = mask.detach().to(self._expert_mask.device)

    def As_weight(self, e: int):
        # 第 e 个 expert 的 LoRA-A 权重：[r, d_in]
        return self.lora_A[self.active_adapter].loraA[e].mlp.weight

    def Bs_weight(self, e: int):
        # 第 e 个 expert 的 LoRA-B 权重：[d_out, r]
        return self.lora_B[self.active_adapter].loraB[e].mlp.weight

    def _qr_Q_list(self, x_dtype: torch.dtype):
        """
        为每个 expert 生成 Q(e) 的前 gate_rank 列：
        - gate_tie == "A_T": 对 A(e)^T 做 QR；
        - gate_tie == "free": 对 trainable gate_P[e] 做 QR。
        QR 在 float32 上做，返回时转回 x_dtype。
        """
        Qs = []
        E = self._current_E()
        m = int(self.gate_rank)

        for e in range(E):
            if self.gate_tie == "A_T":
                A_e = self.As_weight(e)          # [r, d_in]
                Pe = A_e.t()                     # [d_in, r]
                rg_eff = min(m, Pe.size(1))
                Pe = Pe[:, :rg_eff]              # [d_in, rg_eff]
            else:
                Pe = self.gate_P[e]              # [d_in, r_g]
                rg_eff = min(m, Pe.size(1))
                Pe = Pe[:, :rg_eff]

            Pe32 = Pe.to(torch.float32)
            Q_full32, _ = torch.linalg.qr(Pe32, mode="reduced")  # [d_in, rg_eff]
            Qe32 = Q_full32[:, :rg_eff]
            Qs.append(Qe32.to(x_dtype))

        return Qs

    # ---------- forward ----------

    def forward(self, x: torch.Tensor, **kwargs):
        previous_dtype = x.dtype

        # 关闭 LoRA 时退化为原线性
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if self.r[self.active_adapter] <= 0:
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        # 1) 主干输出
        result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        # 2) flatten + dtype 对齐
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.size(-1))  # [N, d_in]
        x_flat = x_flat.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)

        E = self._current_E()
        mask = self._expert_mask[:E].to(device=x_flat.device)

        # 若所有专家被屏蔽，直接返回主干输出
        if (mask > 0.5).sum().item() == 0:
            return result.to(previous_dtype)

        # 3) Router 得分 S: [N, E]
        if self.gate_router == "qr":
            # QR-based router：Q(e)^T x 的能量
            Qs = self._qr_Q_list(x_dtype=x_flat.dtype)   # list of [d_in, r_g_eff]
            Qcat = torch.cat(Qs, dim=1)                  # [d_in, sum r_g_eff]
            Z = x_flat @ Qcat                            # [N, sum r_g_eff]

            rg_list = [q.size(1) for q in Qs]
            S_chunks = []
            col0 = 0
            for rg_eff in rg_list:
                Zi = Z[:, col0:col0 + rg_eff]            # [N, r_g_eff]
                col0 += rg_eff
                S_chunks.append((Zi ** 2).mean(dim=-1, keepdim=True))  # [N,1]
            S = torch.cat(S_chunks, dim=1)               # [N, E]

        elif self.gate_router == "a":
            # A-based router：直接用 A(e)^T x 的能量
            S_chunks = []
            for e in range(E):
                A_e = self.As_weight(e)          # [r, d_in]
                Zi = x_flat @ A_e.t()           # [N, r]
                S_e = (Zi ** 2).mean(dim=-1, keepdim=True)  # [N,1]
                S_chunks.append(S_e)
            S = torch.cat(S_chunks, dim=1)       # [N, E]

        elif self.gate_router == "mlp":
            # 向量 router：每个 expert 一条向量 w_e，score_e(x) = w_e^T x
            # router_vec: [E, d_in]
            W = self.router_vec.to(x_flat.dtype)         # [E, d_in]
            S = x_flat @ W.t()                           # [N, E]
        else:
            raise RuntimeError(f"Unknown gate_router mode: {self.gate_router}")

        # mask + 温度缩放 + 数值稳定 softmax
        S_masked = S.masked_fill(mask.unsqueeze(0) < 0.5, -1e9)
        S_norm = S_masked / max(self.gate_tau, 1e-6)
        S_norm = S_norm - S_norm.max(dim=-1, keepdim=True).values
        G = torch.softmax(S_norm, dim=-1)            # [N, E]

        # 4) LoRA A/B 分支
        x_flat_drop = self.lora_dropout[self.active_adapter](x_flat)
        lora_a_list = self.lora_A[self.active_adapter](x_flat_drop)  # list of [N, r]
        lora_b_list = self.lora_B[self.active_adapter](lora_a_list)  # list of [N, d_out]

        scale = self.scaling[self.active_adapter]
        out_features = self.out_features
        G_shaped = G.view(*orig_shape[:-1], E)

        for i in range(E):
            if mask[i] < 0.5:
                continue
            expert_out = lora_b_list[i].view(*orig_shape[:-1], out_features)
            w = G_shaped[..., i].unsqueeze(-1)
            result = result + expert_out * scale * w

        result = result.to(previous_dtype)

        # 5) router_full / load 统计（用于 balance loss 或可视化）
        with torch.no_grad():
            P = torch.softmax(S_norm.detach(), dim=-1).mean(dim=0)  # [E]
            self._last_P = P

            sel_mask = (G > 0).to(G.dtype)
            f = sel_mask.sum(dim=0)
            f = f / (f.sum() + 1e-9)
            self._last_f = f

        # 6) rank-1 能量统计：供 moeloraLearner 使用（ZeRO2-safe）
        if (
            self.training
            and self.usage_recorder is not None
            and getattr(self, "_record_enabled", False)
        ):
            try:
                with torch.no_grad():
                    N = x_flat_drop.size(0)
                    if N > 0:
                        rank_energy_list = []
                        for e in range(E):
                            A_e = self.As_weight(e)          # [r, d_in]
                            scores = x_flat_drop @ A_e.t()   # [N, r]
                            re = (scores ** 2).mean(dim=0)   # [r]
                            rank_energy_list.append(re.unsqueeze(0))
                        rank_energy = torch.cat(rank_energy_list, dim=0).detach().cpu()  # [E, r]
                        router_full = P.detach().cpu()

                        payload = {
                            "module": getattr(self, "_module_name", None),
                            "router_full": router_full,
                            "rank_energy": rank_energy,
                        }
                        self.usage_recorder(payload)
            except Exception as e:
                print(f"[CoIN-MoE-LoRA] record_usage failed in forward: {repr(e)}")

        return result



# =========================================================
# LoRA A / B 专家容器（每个 expert 一个完整 rank=r LoRA）
# =========================================================
class CoINMOEExpert(nn.Module):
    """每个专家的小线性层，用于 LoRA A/B。"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.mlp = nn.Linear(self.in_features, self.out_features, bias=False)
        self.weight = self.mlp.weight  # 便于外部直接访问 .weight

    def forward(self, x):
        return self.mlp(x)


class CoINMOELinearA(nn.Module):
    """LoRA-A：每个 expert 的 A 都是 [r, d_in] 的完整 LoRA。"""

    def __init__(self, in_features, out_features, expert_num) -> None:
        super().__init__()
        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features  # out_features = r
        self.loraA = nn.ModuleList([])
        self.r = self.out_features  # 每个 expert 的 rank=r
        for _ in range(self.expert_num):
            self.loraA.append(CoINMOEExpert(self.in_features, self.r))

    def forward(self, x):
        # 返回 list：每个 expert 输出 [N, r]
        return [expert(x) for expert in self.loraA]


class CoINMOELinearB(nn.Module):
    """LoRA-B：每个 expert 的 B 都是 [d_out, r] 的完整 LoRA。"""

    def __init__(self, in_features, out_features, expert_num) -> None:
        super().__init__()
        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features  # in_features = r
        self.loraB = nn.ModuleList([])
        self.r = self.in_features  # 每个 expert 的 rank=r
        for _ in range(self.expert_num):
            self.loraB.append(CoINMOEExpert(self.r, self.out_features))

    def forward(self, x_list):
        # 输入/输出均为 list，与 A 分支一一对应
        return [expert(x) for expert, x in zip(self.loraB, x_list)]
