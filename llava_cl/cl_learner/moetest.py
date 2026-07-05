import os
import gc
import copy
from typing import List, Dict, Optional
import math
import torch
import torch.distributed as dist
from transformers.modeling_utils import unwrap_model 
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
# from deepspeed.utils import safe_get_full_grad, safe_set_full_grad
from accelerate.state import AcceleratorState, PartialState
import deepspeed
import torch.nn as nn
from tqdm import tqdm
from .base import BaseCLearner
import sys
local_rank = None

def rank0_print(*args):
    if dist.get_rank() == 0:
        print(*args)


def release_memory():
    gc.collect()
    torch.cuda.empty_cache()
    memory_stats()


def memory_stats():
    rank0_print(f"memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2}")
    rank0_print(f"memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2}")
def maybe_zero_3(param, ignore_status=False, name=None):
    """
    Gather a full parameter tensor under ZeRO-3.
    Returns a **CPU** tensor detached from graph.
    """
    rank = dist.get_rank()
    
    if hasattr(param, "ds_id"):  # ZeRO shard
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE and not ignore_status:
            logging.warning(f"{name}: param.ds_status == NOT_AVAILABLE")
        # shard 参数
        if rank == 0:
            with zero.GatheredParameters([param], modifier_rank=None):
                full = param.data.detach().cpu().clone()
            return full
        else:
            return None
    else:  # 已在当前进程
        return param.detach().cpu().clone() if rank == 0 else None

class MaskedRouter(nn.Module):
    """
    包装原始 nn.Linear 路由器：
    - 前向：对被禁用/裁剪列施加 -inf，使其 softmax 后概率为 0（route_mask）
    - 反向：只允许 trainable 列产生梯度（grad_mask）
    - 统计：用 mask 后的 top-k one-hot 估计专家使用频率 u 的 EMA
    """
    def __init__(self,
                 linear: nn.Linear,
                 route_mask: torch.Tensor,   # [E] 1=可路由, 0=屏蔽
                 grad_mask: torch.Tensor,    # [E] 1=可训练列, 0=列梯度清零
                 stats_ref: dict,            # {'u': Tensor[E], ...}
                 ema_beta: float,
                 top_k: int):
        super().__init__()
        self.linear = linear
        self.route_mask = route_mask
        self.grad_mask = grad_mask
        self.stats_ref = stats_ref
        self.ema_beta = ema_beta
        self.top_k = top_k

        # 保持原参数名，优化器/ckpt 兼容
        for n, p in self.linear.named_parameters(recurse=False):
            self.register_parameter(n, p)

        # 反向列梯度掩码：仅新列可训练
        def _col_grad_mask(g):
            # g: [in_features, E]
            col_mask = self.grad_mask.view(1, -1)
            return g * col_mask
        if hasattr(self, "weight") and isinstance(self.weight, torch.nn.Parameter):
            self.weight.register_hook(_col_grad_mask)
        if getattr(self, "bias", None) is not None:
            self.bias.register_hook(lambda g: g * self.grad_mask)

    def forward(self, x):
        logits = self.linear(x)  # (..., E)

        # 前向硬屏蔽（裁剪/未启用）
        big_neg = -1e9
        masked_logits = logits + (1.0 - self.route_mask)\
            .view(*([1] * (logits.dim() - 1)), -1) * big_neg

        probs = torch.softmax(masked_logits, dim=-1)
        k = min(self.top_k, probs.shape[-1])
        topk_idx = torch.topk(probs, k=k, dim=-1).indices  # (..., k)
        onehot = torch.zeros_like(probs)
        onehot.scatter_(-1, topk_idx, 1.0)

        # 统计 u（基于mask后的真实选择）
        with torch.no_grad():
            flat = onehot.reshape(-1, onehot.shape[-1])  # [Ntok, E]
            mean_use = flat.float().mean(dim=0)          # [E]
            u_old = self.stats_ref["u"]
            self.stats_ref["u"].copy_((1 - self.ema_beta) * u_old
                                      + self.ema_beta * mean_use.to(u_old.dtype))

        return masked_logits

class MoELoRALearner(BaseCLearner):
    """
    动态软剪枝 + 专家保护（仅 LoRA-MoE 子模块）
    - 评分: S_j = u_bar * (alpha * a_bar * (1 + H_act) + beta * g_bar * (1 + H_grad))
    - 周期性软剪: 每次对未保护专家按 S 升序剪 ~10%，每层至少保留 M 个
    - 历史 Top-K 保护: 不剪 + 强正则 (EWC/L2 近似)
    """
    # ====== 可调超参 ======
    def __init__(
        self,
        training_args,
        start_task_id,
        model,
        prune_ratio: float = 0.10,         # 每轮剪掉未保护专家的比例
        ema_beta: float = 0.05,            # EMA 更新系数
        prune_rounds: int = 5,            # 训练全程大致做多少轮剪枝（用于估算周期）
        entropy_eps: float = 1e-6,         # 熵/方差归一化的数值稳定项
    ):
        super().__init__()
        self.model = model
        self.prune_ratio = prune_ratio

        self.ema_beta = ema_beta

        self.prune_rounds = max(prune_rounds, 1)


        self.entropy_eps = entropy_eps

        self.training_args = copy.deepcopy(training_args)
        self.start_task_id = start_task_id
        self.expert_use_num = self.training_args.expert_use_num
        # per-layer 状态
        self.mask = {}           # layer_name -> Tensor[E] (1=keep, 0=masked)
        self.stats = {}          # layer_name -> {'u','a','g'}: Tensor[E]
        self._tmp_act_sum = {}
        self._tmp_act_cnt = {}
        self._tmp_grad_sum = {}
        self._tmp_grad_cnt = {}
        self.params_of_expert = {}     # (layer, j) -> [params]
        self.new_experts = {}          # layer_name -> set(j) 本任务新增专家索引
        self._wrapped_router = {}      # layer_name -> MaskedRouter
        self.layers = [] # [(name, module, expert_num), ...] 
        self.layer_by_name = {}
        self.protected = []
        # EMA 统计
        self.stats = {}  # layer_name -> dict with 'u','a','H_act','g','H_grad'  (每个均为长度E的一维Tensor)

        # 临时缓存(逐step)
        self._tmp_act_sum = {}
        self._tmp_act_sqsum = {}
        self._tmp_act_count = {}
        self._tmp_grad_sum = {}
        self._tmp_grad_sqsum = {}
        self._tmp_grad_count = {}

        # 路由包装器引用，便于更新 mask
        self._wrapped_router = {}  # layer_name -> MaskedRouter

    # ============ 工具：EMA ============
    def _ema_update_(self, old: torch.Tensor, new: torch.Tensor):
        if old is None:
            return new.clone()
        return (1 - self.ema_beta) * old + self.ema_beta * new

    # ============ 查找 & 包装模块 ============
    def _iter_coinmoe_linear(self, model):
        for n, m in model.named_modules():
            # 你的打印中类型名是 CoINMOELoraLinear
            if m.__class__.__name__ == "CoINMOELoraLinear":
                # 该模块上有属性 expert_num
                expert_num = getattr(m, "expert_num", None)
                if expert_num is None:  # 容错
                    continue
                yield n, m, int(expert_num)

    def _register_hooks_for_layer(self, layer_name: str, model, expert_num: int):
        """为每个专家注册 forward/grad 钩子; 包装路由以支持 mask 与统计使用强度"""
        device = next(model.parameters()).device

        # 初始化容器
        self.stats[layer_name] = {
            "u": torch.zeros(expert_num, device=device),
            "a": torch.zeros(expert_num, device=device),
            "H_act": torch.zeros(expert_num, device=device),
            "g": torch.zeros(expert_num, device=device),
            "H_grad": torch.zeros(expert_num, device=device),
        }
        self._tmp_act_sum[layer_name] = torch.zeros(expert_num, device=device)
        self._tmp_act_sqsum[layer_name] = torch.zeros(expert_num, device=device)
        self._tmp_act_count[layer_name] = torch.zeros(expert_num, device=device)
        self._tmp_grad_sum[layer_name] = torch.zeros(expert_num, device=device)
        self._tmp_grad_sqsum[layer_name] = torch.zeros(expert_num, device=device)
        self._tmp_grad_count[layer_name] = torch.zeros(expert_num, device=device)
        
        # 初始化 mask / 保护集
        self.mask[layer_name] = torch.ones(expert_num, device=device)
        self.protected[layer_name] = set()

        # ---- 1) 包装路由，统计使用强度 u，并应用 mask（软剪）----
        linear_router = model.lora_router[self.adapter_key]
        wrapped = MaskedRouter(linear_router, self.mask[layer_name], self.stats[layer_name], self.ema_beta,self.expert_use_num)
        model.lora_router[self.adapter_key] = wrapped  # 替换
        self._wrapped_router[layer_name] = wrapped

        # ---- 2) 注册 forward hook 统计激活强度/熵：选择 B_i.mlp 的输出 ----
        # 结构：m.lora_B['default'].loraB[i].mlp 是一个 nn.Linear
        for j in range(expert_num):  # 遍历每个专家 j
            b_linear = model.lora_B[self.adapter_key].loraB[j].mlp  # 获取专家 j 的LoRA-B输出层（nn.Linear）

            # 生成前向钩子函数（闭包，绑定当前 layer_name 和专家索引 j）
            def _mk_fwd_hook(layer=layer_name, idx=j):
                def _hook(_module, inp, out):  # 钩子函数：在专家前向传播后调用
                    # out: 专家 j 的输出张量，形状为 (..., out_features)
                    with torch.no_grad():  # 统计不参与梯度计算
                        o = out.detach()  # 剥离计算图
                        # 激活强度 a：输出的L2范数均值（衡量激活大小）
                        val = o.pow(2).mean()
                        self._tmp_act_sum[layer][idx] += val  # 累加激活强度
                        # 激活熵 H_act：用输出平方的均值作为代理（衡量激活分布的分散度）
                        self._tmp_act_sqsum[layer][idx] += (o.pow(2).mean())
                        self._tmp_act_count[layer][idx] += 1  # 累加统计次数
                return _hook

            # 为专家 j 的输出层注册前向钩子
            b_linear.register_forward_hook(_mk_fwd_hook())

            # ---- 3) 注册梯度 hook 统计梯度强度与“熵/分散度” ----
            # 对 A_i、B_i 的权重都做
            a_w = model.lora_A[self.adapter_key].loraA[j].mlp.weight
            b_w = b_linear.weight

            def _mk_grad_hook(layer=layer_name, idx=j):
                def _gh(grad):
                    with torch.no_grad():
                        g = grad.detach()
                        g_abs = g.abs()
                        self._tmp_grad_sum[layer][idx] += g_abs.mean()
                        # “熵/分散度”用方差/均值近似
                        var_proxy = (g_abs.var(unbiased=False) / (g_abs.mean() + self.entropy_eps))
                        self._tmp_grad_sqsum[layer][idx] += var_proxy
                        self._tmp_grad_count[layer][idx] += 1
                    return grad
                return _gh

            a_w.register_hook(_mk_grad_hook())
            b_w.register_hook(_mk_grad_hook())

            # 记录该专家参数（用于保护正则）
            self.params_of_expert[(layer_name, j)] = [a_w, b_w]

    def _estimate_period(self, total_steps_guess: int):

        self.period = max(50, total_steps_guess // self.prune_rounds)

    # ============ CLearner 接口 ============
    def before_train(self, task_id, model, tokenizer=None, **kwargs):
        # 1) 发现所有 CoINMOELoraLinear 层
        self.layers.clear()
        self.layer_by_name.clear()
        for name, mod, expert_num in self._iter_coinmoe_linear(model):
            self.layers.append((name, mod, expert_num))
            self.layer_by_name[name] = mod
            self._register_hooks_for_layer(name, mod, expert_num)

        # 2) 估算训练总步数并设置剪枝周期
        #    优先从 kwargs 里估算（若提供了 train_dataset）
        total_steps_guess = 5000
        try:
            train_ds = kwargs.get("train_dataset", None)
            if train_ds is not None:
                bs = kwargs.get("per_device_train_batch_size", 1)
                gas = getattr(kwargs.get("args", None), "gradient_accumulation_steps", 1) if kwargs.get("args", None) else 1
                world = dist.get_world_size() if dist.is_initialized() else 1
                epochs = getattr(kwargs.get("args", None), "num_train_epochs", 1) if kwargs.get("args", None) else 1
                steps_per_epoch = max(1, math.ceil(len(train_ds) / (bs * world * gas)))
                total_steps_guess = int(steps_per_epoch * epochs)
        except Exception:
            pass

        self._estimate_period(total_steps_guess)

        if self.verbose and dist.get_rank() in (None, 0):
            print(f"[MoELoRALearner] Found {len(self.layers)} CoINMoE-LoRA layers.")
            print(f"[MoELoRALearner] Prune every {self.period} steps "
                  f"(target ~{self.prune_rounds} rounds over ~{total_steps_guess} steps).")

    def after_train(self, task_id, model, tokenizer=None, **kwargs):
        # 在任务结束时，确定“历史 Top-K 保护”并保存 reference（用于正则）
        for name, _, E in self.layers:
            S = self._compute_scores_per_layer(name)  # Tensor[E]
            topk = min(self.protect_top_k, E)
            if topk <= 0:
                continue
            keep_idx = torch.topk(S, k=topk, largest=True).indices.tolist()
            self.protected[name].update(keep_idx)

            # 为保护的专家保存快照
            for j in keep_idx:
                for p in self.params_of_expert[(name, j)]:
                    self.snapshots[p] = p.detach().clone()

        if self.verbose and dist.get_rank() in (None, 0):
            msg = {ln: sorted(list(s)) for ln, s in self.protected.items()}
            print(f"[MoELoRALearner] Protected experts after task {task_id}: {msg}")

    def loss(self, loss, model=None, **kwargs):
        """每步调用：1) 累计 step；2) 提交统计到 EMA；3) 周期性剪枝；4) 保护正则"""
        self.step += 1
        # 提交统计（把逐步缓存推进到 EMA）
        self._commit_step_stats()

        # 周期性软剪枝
        if self.step % self.period == 0:
            self._prune_once()

        # 保护正则：对被保护专家的参数加 L2 到其 snapshot
        if self.reg_lambda > 0.0 and len(self.snapshots) > 0:
            reg = 0.0
            for p, ref in self.snapshots.items():
                reg = reg + (p - ref).pow(2).mean()
            loss = loss + self.reg_lambda * reg

        return loss

    # ============ 统计提交 & 评分 ============
    def _commit_step_stats(self):
        """把 _tmp_* 累积的 step 统计写入 EMA"""
        for name, _, E in self.layers:
            # 激活
            cnt = self._tmp_act_count[name].clamp_min(1.0)
            a_now = self._tmp_act_sum[name] / cnt
            Hact_now = self._tmp_act_sqsum[name] / cnt

            self.stats[name]["a"] = self._ema_update_(self.stats[name]["a"], a_now)
            # 归一化到 [0,1]（层内 min-max），数值更稳
            self.stats[name]["H_act"] = self._ema_update_(self.stats[name]["H_act"], Hact_now)

            # 梯度
            gcnt = self._tmp_grad_count[name].clamp_min(1.0)
            g_now = self._tmp_grad_sum[name] / gcnt
            Hgrad_now = self._tmp_grad_sqsum[name] / gcnt
            self.stats[name]["g"] = self._ema_update_(self.stats[name]["g"], g_now)
            self.stats[name]["H_grad"] = self._ema_update_(self.stats[name]["H_grad"], Hgrad_now)

            # 清空 step 累积
            self._tmp_act_sum[name].zero_()
            self._tmp_act_sqsum[name].zero_()
            self._tmp_act_count[name].zero_()
            self._tmp_grad_sum[name].zero_()
            self._tmp_grad_sqsum[name].zero_()
            self._tmp_grad_count[name].zero_()

    def _minmax01(self, x: torch.Tensor):
        m, M = torch.quantile(x, 0.05), torch.quantile(x, 0.95)  # 分位点更稳
        den = (M - m).clamp_min(1e-12)
        return ((x - m) / den).clamp(0, 1)

    def _compute_scores_per_layer(self, layer_name: str) -> torch.Tensor:
        st = self.stats[layer_name]
        u = self._minmax01(st["u"])
        a = self._minmax01(st["a"])
        Hact = self._minmax01(st["H_act"])
        g = self._minmax01(st["g"])
        Hgrad = self._minmax01(st["H_grad"])
        S = u * (self.alpha * a * (1 + Hact) + self.beta * g * (1 + Hgrad))
        return S

    # ============ 软剪枝 ============
    def _prune_once(self):
        pruned_total = 0
        for name, mod, E in self.layers:
            keep_mask = self.mask[name]          # Tensor[E] (0/1)
            prot = self.protected[name]          # set
            already_off = (keep_mask == 0).nonzero(as_tuple=False).flatten().tolist()

            # 候选 = 未保护 & 未被剪
            cand = [j for j in range(E) if (j not in prot) and (j not in already_off)]
            if len(cand) == 0:
                continue

            S = self._compute_scores_per_layer(name)
            # 每层至少保留 self.min_keep
            can_prune_max = max(0, (E - self.min_keep) - (len(already_off)))
            if can_prune_max <= 0:
                continue

            k = max(1, int(round(self.prune_ratio * len(cand))))
            k = min(k, can_prune_max)
            # 在 cand 中选 S 最低的 k 个
            cand_scores = S[cand]
            prune_rel_idx = torch.topk(cand_scores, k=k, largest=False).indices.tolist()
            prune_idx = [cand[i] for i in prune_rel_idx]

            # 软剪: router logits 上打 -inf（通过 mask=0），并可选冻结 A/B 参数
            for j in prune_idx:
                keep_mask[j] = 0.0
                # 冻结该专家 LoRA A/B
                for p in self.params_of_expert[(name, j)]:
                    p.requires_grad_(False)
                self.frozen[name].add(j)

            pruned_total += len(prune_idx)

        if self.verbose and pruned_total > 0 and dist.get_rank() in (None, 0):
            print(f"[MoELoRALearner] Step {self.step}: soft-pruned {pruned_total} experts across layers.")


