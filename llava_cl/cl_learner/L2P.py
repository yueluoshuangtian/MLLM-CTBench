# /mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/cl_learner/L2P.py

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
import json
from .base import BaseCLearner
from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
# ----------------------- small helpers -----------------------
def _dtype_like_model(model: nn.Module):
    """获取与模型参数相同的数据类型(dtype)"""
    for p in model.parameters():
        return p.dtype
    return torch.float32  # 若模型无参数，默认返回float32

def _device_like_model(model: nn.Module):
    """获取与模型参数相同的设备(device)"""
    for p in model.parameters():
        return p.device
    return torch.device("cpu")  # 若模型无参数，默认返回CPU


# ==================== L2P: Prompt-Pool + Learner ====================
class L2PPromptPool(nn.Module):
    """
    提示池(文本/视觉各实例化一套)。
    - selection_mode="semantic"：保留你前面可微 top-k 方案(keys 可学习)；
    - selection_mode="attn"：改为跨模态注意力融合：
        q_side = w ⊙ Attn_side(Q=F_side, K=F_other, V=F_other) + F_side
        c_side = pool(q_side)   # [D]
        logits = (P W_p)^mean_L @ (MLP(c_side) W_q)    # [M]
        attn over prompts → 融合为 [L, D] 一段综合提示。
    """
    def __init__(
        self,
        hidden_size: int,  # 模型隐藏层维度(与提示维度一致)
        num_prompts: int = 64,  # 提示池中提示的数量(M)
        prompt_len: int = 8,  # 每个提示的长度(token数，L)
        key_dim: int = 256,  # 语义匹配中key的维度(K)
        selection_mode: str = "semantic",  # 提示选择模式："semantic" | "attn"
        top_k: int = 4,  # semantic模式下选择的提示数量
        pool_type: str = "max",  # 特征池化方式："mean"(均值) | "max"(最大值)
        init_range: float = 0.02,  # 参数初始化的标准差范围
        # ---- semantic模式(可学习key)相关 ----
        learn_keys: bool = True,  # 是否让key可学习
        semantic_variant: str = "soft_topk",  # semantic模式下的选择变体："soft_topk"(软选择) | "gumbel_st"(Gumbel-softmax)
        tau: float = 0.07,  # soft_topk中的温度参数(控制分布平滑度)
        eval_hard: bool = True,  # 评估时是否使用硬选择(直接选top-k，非可微)
        gumbel_tau: float = 1.0,  # Gumbel-softmax的温度参数
        eps: float = 1e-12,  # 数值稳定性参数
        # ---- 跨模态注意力相关 ----
        attn_heads: int = 8,  # 多头注意力头数
        attn_dropout: float = 0.0,  # 注意力dropout概率
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_prompts = num_prompts
        self.prompt_len = prompt_len
        self.key_dim = key_dim
        self.selection_mode = selection_mode
        self.top_k = top_k
        self.pool_type = pool_type

        # semantic 控制
        self.semantic_variant = semantic_variant
        self.tau = tau
        self.gumbel_tau = gumbel_tau
        self.eval_hard = eval_hard
        self.eps = eps

        # 提示参数：[M, L, D]，M个提示，每个提示含L个token，每个token维度为D(hidden_size)
        self.prompts = nn.Parameter(torch.empty(num_prompts, prompt_len, hidden_size))
        # Key参数：[M, K]，每个提示对应一个可学习的key(semantic模式使用)
        self.keys = nn.Parameter(torch.empty(num_prompts, key_dim), requires_grad=learn_keys)
        
        # 查询映射网络：将输入特征(文本/视觉)映射到key空间(用于匹配提示)
        self.query_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, key_dim),
        )


        # 提示侧投影矩阵(用于attn模式下的提示权重计算)
        self.w_q = nn.Linear(key_dim, key_dim, bias=False)  # 查询投影
        self.w_p = nn.Linear(hidden_size, key_dim, bias=False)  # 提示投影

        # ====== 新增：跨模态注意力模块(batch_first) & 维度权重向量 w(weight modulation)======
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=attn_heads,
                                         dropout=attn_dropout, batch_first=True)
        self.w_mod = nn.Parameter(torch.ones(hidden_size))  # w ∈ R^d

        # 初始化
        nn.init.trunc_normal_(self.prompts, std=init_range)
        nn.init.trunc_normal_(self.keys, std=init_range)
        for m in self.query_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_p.weight)

    # ---------- utils ----------
    def _pool_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """x: [T, D] -> [D]"""
        if x.numel() == 0:
            return torch.zeros(self.hidden_size, device=x.device, dtype=x.dtype)
        if self.pool_type == "max":
            return x.max(dim=0).values
        return x.mean(dim=0)

    @staticmethod
    def _cosine_sim(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        q = F.normalize(query, dim=-1)   # [K]
        k = F.normalize(keys, dim=-1)    # [M, K]
        return torch.matmul(k, q)        # [M]

    @staticmethod
    def _topk_mask(sim: torch.Tensor, k: int):
        k = min(k, sim.size(0))
        idx = torch.topk(sim, k=k, dim=0).indices
        mask = torch.zeros_like(sim, dtype=torch.bool); mask[idx] = True
        return idx, mask

    @staticmethod
    def _sample_gumbel(shape, device, eps=1e-12):
        u = torch.rand(shape, device=device)
        return -torch.log(-torch.log(u + eps) + eps)

    # ---------- 语义模式(保持不变：可微 top-k) ----------
    def _semantic_select(self, seg_embed: torch.Tensor) -> torch.Tensor:
        pooled = self._pool_tokens(seg_embed)       # [D]
        query = self.query_mlp(pooled)              # [K]
        sim = self._cosine_sim(query, self.keys)    # [M]
        idx, mask = self._topk_mask(sim, self.top_k)

        if (not self.training) and self.eval_hard:
            chosen = self.prompts.index_select(0, idx)         # [k, L, D]
            return chosen.reshape(-1, self.hidden_size)        # [k*L, D]

        if self.semantic_variant == "soft_topk":
            topk_sim = sim[idx]                                # [k]
            w = F.softmax(topk_sim / self.tau, dim=0)          # [k]
            chosen = self.prompts.index_select(0, idx)         # [k, L, D]
            return (chosen * w.view(-1,1,1)).reshape(-1, self.hidden_size)

        # gumbel-st
        g = self._sample_gumbel(sim.shape, device=sim.device, eps=self.eps)
        sim_g = (sim + g) / max(self.gumbel_tau, self.eps)
        hard_idx, hard_mask = self._topk_mask(sim_g, self.top_k)
        hard_chosen = self.prompts.index_select(0, hard_idx)   # [k, L, D]
        soft = F.softmax(sim / max(self.tau, self.eps), dim=0)
        soft = soft * hard_mask.float()
        soft = soft / (soft.sum() + self.eps)
        soft_topk = soft[hard_idx]                             # [k]
        w = (hard_mask.float()[hard_idx] - soft_topk).detach() + soft_topk
        return (hard_chosen * w.view(-1,1,1)).reshape(-1, self.hidden_size)

    # ---------- 跨模态注意力路由 ----------
    def _attn_xmodal(self, self_seg: torch.Tensor, other_seg: torch.Tensor) -> torch.Tensor:
        """
        self_seg:  当前侧序列 F_side, [T_side, D]
        other_seg: 另一侧序列 F_other, [T_other, D]
        return: 融合后的综合提示，[L, D]
        """
        # 1) Cross-Attention: Q=self_seg, K=V=other_seg
        #    形状要求 [B, T, D]，这里 B=1
        q = self_seg.unsqueeze(0)      # [1, Tq, D]
        k = other_seg.unsqueeze(0)     # [1, Tk, D]
        v = other_seg.unsqueeze(0)     # [1, Tk, D]
        attn_out, _ = self.mha(q, k, v, need_weights=False)  # [1, Tq, D]
        attn_out = attn_out.squeeze(0)                       # [Tq, D]

        # 2) 维度权重调节 + 残差：q_side = w ⊙ attn_out + F_side
        q_side = attn_out * self.w_mod.view(1, -1) + self_seg  # [Tq, D]

        # 3) 池化得到全局上下文向量 c_side ∈ R^D
        c_side = self._pool_tokens(q_side)                     # [D]

        # 4) 用 c_side 做 query，计算 prompts 的权重并融合为一段 [L, D]
        q_vec = self.query_mlp(c_side)                         # [K]
        q_proj = self.w_q(q_vec)                               # [K]
        P = self.prompts                                       # [M, L, D]
        P_proj = self.w_p(P)                                   # [M, L, K]
        P_key = P_proj.mean(dim=1)                             # [M, K]
        logits = torch.matmul(P_key, q_proj)                   # [M]
        attn = logits.softmax(dim=0)                           # [M]
        fused = torch.einsum("m, mld -> ld", attn, P)          # [L, D]
        return fused

    # ---------- 统一入口 ----------
    def select_prompts(self, seg_embed: torch.Tensor,
                       other_embed: torch.Tensor = None) -> torch.Tensor:
        """
        - semantic：忽略 other_embed，top-k 拼接(可微)。
        - attn：若提供 other_embed → 跨模态；否则退化为单模态(Q=K=V=self)。
        """
        if self.selection_mode == "semantic":
            return self._semantic_select(seg_embed)

        # 注意力路由：cross-modal / uni-modal
        if other_embed is None:
            other_embed = seg_embed
        return self._attn_xmodal(seg_embed, other_embed)


class L2PLearner(BaseCLearner):
    """
    - 训练阶段：为“当前任务”创建并仅训练该任务的 text/vision 两套 L2PPromptPool(含其 query_mlp/keys/prompts)。
      同时**冻结整模型骨干**(LLM/vision/projector/LoRA等)。
    - 测试阶段：构造“合并视图池”(MergedPromptPool)，内部持有历次任务的子池，按各自的 query_mlp/keys 计算，
      再做全局 top-k 选择，避免后续任务改写旧任务查询语义，满足持续学习评测。
    - 多模态：通过 patch `prepare_inputs_labels_for_multimodal` 在文本段/图像块前插入 prompts；
      纯文本模型：patch `forward`，仅插入文本 prompts。
    """

    # ====== 测试阶段用：合并视图池(只读/不训练) ======
    class MergedPromptPool(nn.Module):
        """
        持有多个冻结的 L2PPromptPool 子池；select_prompts() 时：
        - semantic：各子池独立计算相似度，收集候选并做**全局 top-k**，返回 [k*L, D]。
        - attn：各子池先各自融合出 [L,D]，再按一个轻量分数 softmax 加权求和，返回 [L,D]。
        仅用于评测阶段(no_grad)。
        """
        def __init__(self, subpools: List['L2PPromptPool'], selection_mode: str, global_top_k: int):
            super().__init__()
            self.subpools = nn.ModuleList(subpools)
            for sp in self.subpools:
                for p in sp.parameters():
                    p.requires_grad = False
            self.selection_mode = selection_mode
            self.global_top_k = int(global_top_k)

        @torch.no_grad()
        def select_prompts(self, seg_embed: torch.Tensor, other_embed: torch.Tensor = None) -> torch.Tensor:
            device = seg_embed.device
            D = seg_embed.size(-1)

            if self.selection_mode == "semantic":
                # —— 收集所有子池候选(尽量少算：每个子池取局部 top-k，再全局 top-k)——
                block_list = []   # [(score, block=[L,D], L), ...]
                for sp in self.subpools:
                    # 计算该子池的 logits(用它自己的 query/keys)
                    pooled = seg_embed.mean(dim=0)                              # [D]
                    q = sp.query_mlp(pooled)                                    # [K]
                    # 余弦相似度：
                    qn = F.normalize(q, dim=-1)                                 # [K]
                    kn = F.normalize(sp.keys, dim=-1)                           # [M_sp, K]
                    sim = torch.matmul(kn, qn)                                   # [M_sp]

                    # 取子池局部 top-k'(不大于 global_top_k)
                    k_sp = min(int(getattr(sp, "top_k", self.global_top_k)), sim.numel(), self.global_top_k)
                    if k_sp <= 0:
                        continue
                    vals, idx = torch.topk(sim, k=k_sp, dim=0)                   # [k_sp]
                    # 将每个提示段的 token 块与一个代表分数入候选池
                    Lp = int(getattr(sp, "prompt_len", 0))
                    if Lp <= 0:
                        continue
                    chosen = sp.prompts.index_select(0, idx)                     # [k_sp, Lp, D]
                    # 记录(score, block)：用 sim 作为排序分数(简单有效)
                    for j in range(k_sp):
                        block_list.append((vals[j].item(), chosen[j], Lp))

                if len(block_list) == 0:
                    # 回退：空则给一段全零提示，避免崩溃
                    return torch.zeros(self.global_top_k, 1 if self.global_top_k == 0 else D, device=device).reshape(0, D)

                # 全局 top-k
                block_list.sort(key=lambda x: float(x[0]), reverse=True)
                top = block_list[:min(self.global_top_k, len(block_list))]
                # 拼接为 [k*L, D]
                out = torch.cat([b for _, b, _ in top], dim=0)                   # [sum Lp, D](不同池 Lp 可相同)
                return out

            # —— 注意力路由：各子池各自融合 -> [L,D]，再做加权求和 —— 
            fused_list, scores = [], []
            for sp in self.subpools:
                f = sp._attn_xmodal(seg_embed, other_embed if other_embed is not None else seg_embed)  # [L,D]
                fused_list.append(f.unsqueeze(0))                                                       # [1,L,D]
                # 简单分数：两者均值向量的点积
                s = (seg_embed.mean(dim=0, keepdim=True) @ f.mean(dim=0, keepdim=True).T).squeeze()
                scores.append(s)
            F_cat = torch.cat(fused_list, dim=0)                       # [S, L, D]
            W = torch.stack(scores, dim=0).softmax(dim=0)              # [S]
            out = torch.einsum("s,sld->ld", W, F_cat)                  # [L,D]
            return out

    # ====== Learner 主体 ======
    def __init__(
        self,
        training_args=None,
        text_num_prompts=64,
        text_prompt_len=8,
        vision_num_prompts=64,
        vision_prompt_len=8,
        key_dim=256,
        text_top_k=4,
        vision_top_k=4,
        selection_mode="semantic",   # "semantic" | "attn"
        pool_type="mean",
        reg_lambda: float = 0.0,     # Prompt L2 正则(当前训练池)
    ):
        super().__init__()
        self.training_args = training_args
        self.cfg = dict(
            text_num_prompts=text_num_prompts,
            text_prompt_len=text_prompt_len,
            vision_num_prompts=vision_num_prompts,
            vision_prompt_len=vision_prompt_len,
            key_dim=key_dim,
            text_top_k=text_top_k,
            vision_top_k=vision_top_k,
            selection_mode=selection_mode,
            pool_type=pool_type,
            reg_lambda=reg_lambda,
        )

        # 首次 patch 标记 & 原函数缓存
        self._patched = False
        self._orig_prepare_fn = None
        self._orig_forward_fn = None

        # 每任务的“专属子池”(含其 query_mlp/keys/prompts)
        self.task_txt_pools: dict[int, L2PPromptPool] = {}
        self.task_vis_pools: dict[int, L2PPromptPool] = {}

        # 当前激活的池(训练期=当前任务子池；测试期=合并视图池)
        self.txt_pool: nn.Module = None
        self.vis_pool: nn.Module = None

        self.current_task_id = None

    # ---------- 工具 ----------
    def _device_of(self, model: nn.Module):
        for p in model.parameters():
            return p.device
        return torch.device("cpu")

    def _dtype_of(self, model: nn.Module):
        for p in model.parameters():
            return p.dtype
        return torch.float32

    def _make_pool(self, model, is_text: bool, num_prompts: int, prompt_len: int, top_k: int) -> 'L2PPromptPool':
        hidden = model.config.hidden_size
        pool = L2PPromptPool(
            hidden_size=hidden,
            num_prompts=num_prompts,
            prompt_len=prompt_len,
            key_dim=self.cfg["key_dim"],
            selection_mode=self.cfg["selection_mode"],
            top_k=top_k,
            pool_type=self.cfg["pool_type"],
            # 保持你前面定义的可微 key/语义模式默认值(如 learn_keys=True, semantic_variant="soft_topk"...)
        ).to(self._device_of(model)).to(dtype=self._dtype_of(model))
        return pool

    def _bind_active_pools_to_model(self, model):
        # 将“当前激活池”挂在 model 上，便于 Trainer/保存
        model.l2p_txt_pool = self.txt_pool
        model.l2p_vis_pool = self.vis_pool

    # ---------- 生命周期钩子 ----------
    def before_train(self, task_id, model, tokenizer=None, **kwargs):
        """
        - 首次调用时 patch 模型前向；
        - 冻结整模型，仅训练当前任务的子池；
        - 将当前激活池设置为该任务的子池。
        """
        self.current_task_id = task_id

        # 首次 patch
        if not self._patched:
            if hasattr(model, "prepare_inputs_labels_for_multimodal"):
                self._patch_llava_prepare(model)
            else:
                self._patch_llama_forward(model)
            self._patched = True

        # 冻结全模型
        for _, p in model.named_parameters():
            p.requires_grad = False

        # 为该任务创建子池(若不存在)，并激活它
        if task_id not in self.task_txt_pools:
            self.task_txt_pools[task_id] = self._make_pool(
                model, is_text=True,
                num_prompts=self.cfg["text_num_prompts"],
                prompt_len=self.cfg["text_prompt_len"],
                top_k=self.cfg["text_top_k"],
            )
        if task_id not in self.task_vis_pools:
            self.task_vis_pools[task_id] = self._make_pool(
                model, is_text=False,
                num_prompts=self.cfg["vision_num_prompts"],
                prompt_len=self.cfg["vision_prompt_len"],
                top_k=self.cfg["vision_top_k"],
            )

        # 当前任务 = 训练池(只训练这两套参数)
        self.txt_pool = self.task_txt_pools[task_id]
        self.vis_pool = self.task_vis_pools[task_id]
        for p in self.txt_pool.parameters(): p.requires_grad = True
        for p in self.vis_pool.parameters(): p.requires_grad = True

        self._bind_active_pools_to_model(model)

    def after_train(self, task_id, model, tokenizer=None, **kwargs):
        """
        - 训练一个任务结束，为接下来“对所有任务的评测”切换到“合并视图池”。
        - 下一任务开始时，before_train 会再切回该任务的子池。
        """
        txt_list = [self.task_txt_pools[k] for k in sorted(self.task_txt_pools.keys())]
        vis_list = [self.task_vis_pools[k] for k in sorted(self.task_vis_pools.keys())]

        self.txt_pool = self.MergedPromptPool(
            subpools=txt_list,
            selection_mode=self.cfg["selection_mode"],
            global_top_k=self.cfg["text_top_k"],
        ).to(self._device_of(model))
        self.vis_pool = self.MergedPromptPool(
            subpools=vis_list,
            selection_mode=self.cfg["selection_mode"],
            global_top_k=self.cfg["vision_top_k"],
        ).to(self._device_of(model))

        # 评测池全冻结
        for p in self.txt_pool.parameters(): p.requires_grad = False
        for p in self.vis_pool.parameters(): p.requires_grad = False

        self._bind_active_pools_to_model(model)

    def loss(self, loss, **kwargs):
        """
        仅对“当前训练池”的参数施加 L2 正则(避免评测池/旧池被动受影响)。
        """
        reg_lambda = self.cfg["reg_lambda"]
        if reg_lambda > 0.0 and isinstance(self.txt_pool, L2PPromptPool) and isinstance(self.vis_pool, L2PPromptPool):
            reg = 0.0
            for p in list(self.txt_pool.parameters()) + list(self.vis_pool.parameters()):
                reg = reg + p.pow(2).sum()
            loss = loss + reg_lambda * reg
        return loss

    # ---------- patchers ----------
    def _patch_llava_prepare(self, model):
        """
        多模态：在 prepare_inputs_labels_for_multimodal 中插队，分别在每段文本/图像前插入
        由当前激活池(训练池或合并测试池)产生的 prompts。
        """
        self._orig_prepare_fn = model.prepare_inputs_labels_for_multimodal

        def _patched_prepare(this, convs, input_ids, position_ids, attention_mask, past_key_values, labels, images):
            vision_tower = this.get_vision_tower() if hasattr(this, "get_vision_tower") else None

            # 推理 mode (labels=None, 含 generate prefill 和 autoregressive): 整体跳过 L2P prompt 注入,
            # 走原始函数. 否则 prefill 注入 prompts 把 KV cache 扩长 64, HF 的 attention_mask 没跟上,
            # autoregressive 步会报 "Attention mask should be of size X but is Y".
            # ⚠️ 这意味着训练学到的 prompts 在 eval 阶段不发挥作用 — smoke wiring OK 但 L2P 数值不正确.
            # §7.2 要 L2P 数值正确, 需要 prefill 注入 prompts 后同步扩展 attention_mask / position_ids
            # 并让 HF generate 内部循环也用扩展后的长度. 留作后续工作.
            if labels is None:
                return self._orig_prepare_fn(convs, input_ids, position_ids, attention_mask, past_key_values, labels, images)

            attention_mask_ = attention_mask
            position_ids_ = position_ids

            _labels = labels
            _position_ids = position_ids_
            _attention_mask = attention_mask_

            if attention_mask_ is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                attention_mask = attention_mask_.bool()
            # generate 自回归步: attention_mask 是累积长度 (含 KV cache), input_ids 只有新 token.
            # 把 mask 尾部对齐到 input_ids 长度, 否则 ids[mask] IndexError.
            if attention_mask.shape[-1] != input_ids.shape[-1]:
                attention_mask = attention_mask[..., -input_ids.shape[-1]:]
            if position_ids_ is None:
                position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            else:
                position_ids = position_ids_
            if labels is None:
                labels = torch.full_like(input_ids, IGNORE_INDEX)
            # 去 padding
            input_ids_list = [ids[mask] for ids, mask in zip(input_ids, attention_mask)]
            labels_list = [lab[mask] for lab, mask in zip(labels, attention_mask)]

            new_input_embeds, new_labels = [], []

            # 编码图像(若有)
            if images is not None and vision_tower is not None:
                if isinstance(images, list) or images.ndim == 5:
                    concat_images = torch.cat([img for img in images], dim=0)
                    image_features = this.encode_images(concat_images, convs)
                    split_sizes = [img.shape[0] for img in images]
                    image_features = torch.split(image_features, split_sizes, dim=0)
                    image_features = [x.flatten(0, 1).to(this.device) for x in image_features]
                else:
                    image_features = this.encode_images(images, convs).to(this.device)
            else:
                image_features = None

            cur_image_idx = 0
            for bidx, cur_input_ids in enumerate(input_ids_list):
                cur_labels = labels_list[bidx]

                if image_features is None:
                    # 纯文本：文本段 + 文本提示
                    cur_text_embeds = this.get_model().embed_tokens(cur_input_ids)   # [T,D]
                    txt_prompt = self.txt_pool.select_prompts(cur_text_embeds)       # [kL,D] or [L,D]
                    cur_embeds = torch.cat([txt_prompt, cur_text_embeds], dim=0)
                    cur_labs = torch.cat([
                        torch.full((txt_prompt.size(0),), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype),
                        cur_labels
                    ], dim=0)
                    new_input_embeds.append(cur_embeds)
                    new_labels.append(cur_labs)
                    continue

                # 多模态：按 IMAGE_TOKEN_INDEX 分段交替插入
                num_images = int((cur_input_ids == IMAGE_TOKEN_INDEX).sum().item()
                                 if hasattr((cur_input_ids == IMAGE_TOKEN_INDEX).sum(), "item")
                                 else (cur_input_ids == IMAGE_TOKEN_INDEX).sum())
                image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]

                # 文本段(去掉 image token)
                cur_labels_noim = [cur_labels[image_token_indices[i]+1:image_token_indices[i+1]] for i in range(len(image_token_indices)-1)]
                split_sizes = [x.shape[0] for x in cur_labels_noim]
                cur_text_embeds_all = this.get_model().embed_tokens(
                    torch.cat([cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]] for i in range(len(image_token_indices)-1)])
                )
                cur_text_embeds_no_im = torch.split(cur_text_embeds_all, split_sizes, dim=0)

                cur_new_embeds, cur_new_labels = [], []
                for seg_i in range(num_images + 1):
                    # 文本段 + 文本池(允许跨模态注意力：other=paired_img)
                    txt_seg = cur_text_embeds_no_im[seg_i]
                    paired_img = image_features[cur_image_idx] if (image_features is not None and seg_i < num_images) else None
                    txt_prompt = self.txt_pool.select_prompts(txt_seg, other_embed=paired_img)
                    cur_new_embeds.append(txt_prompt)
                    cur_new_labels.append(torch.full((txt_prompt.size(0),), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

                    cur_new_embeds.append(txt_seg)
                    cur_new_labels.append(cur_labels_noim[seg_i])

                    if seg_i < num_images:
                        # 图像块 + 视觉池(other=txt_seg)
                        img_feat = image_features[cur_image_idx]
                        cur_image_idx += 1
                        vis_prompt = self.vis_pool.select_prompts(img_feat, other_embed=txt_seg)
                        cur_new_embeds.append(vis_prompt)
                        cur_new_labels.append(torch.full((vis_prompt.size(0),), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

                        cur_new_embeds.append(img_feat)
                        cur_new_labels.append(torch.full((img_feat.size(0),), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

                cat_embeds = torch.cat(cur_new_embeds, dim=0)
                cat_labels = torch.cat(cur_new_labels, dim=0)
                new_input_embeds.append(cat_embeds)
                new_labels.append(cat_labels)

            # 截断 & pad
            tokenizer_model_max_length = getattr(this.config, 'tokenizer_model_max_length', None)
            if tokenizer_model_max_length is not None:
                new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
                new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

            max_len = max(x.shape[0] for x in new_input_embeds)
            bsz = len(new_input_embeds)
            dtype = new_input_embeds[0].dtype
            dev = new_input_embeds[0].device

            new_input_embeds_padded = []
            new_labels_padded = torch.full((bsz, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=dev)
            attention_mask_out = torch.zeros((bsz, max_len), dtype=torch.bool, device=dev)
            position_ids_out = torch.zeros((bsz, max_len), dtype=torch.long, device=dev)

            padding_side = getattr(this.config, 'tokenizer_padding_side', 'right')
            for i, (emb_i, lab_i) in enumerate(zip(new_input_embeds, new_labels)):
                L = emb_i.size(0)
                pad = torch.zeros((max_len - L, emb_i.size(1)), dtype=dtype, device=dev)
                if padding_side == "left":
                    new_input_embeds_padded.append(torch.cat([pad, emb_i], dim=0))
                    if L > 0:
                        new_labels_padded[i, -L:] = lab_i
                        attention_mask_out[i, -L:] = True
                        position_ids_out[i, -L:] = torch.arange(0, L, device=dev)
                else:
                    new_input_embeds_padded.append(torch.cat([emb_i, pad], dim=0))
                    if L > 0:
                        new_labels_padded[i, :L] = lab_i
                        attention_mask_out[i, :L] = True
                        position_ids_out[i, :L] = torch.arange(0, L, device=dev)

            new_input_embeds_out = torch.stack(new_input_embeds_padded, dim=0)
            new_labels_out = None if _labels is None else new_labels_padded
            attention_mask_final = None if _attention_mask is None else attention_mask_out.to(dtype=_attention_mask.dtype)
            position_ids_final = None if _position_ids is None else position_ids_out

            return None, position_ids_final, attention_mask_final, past_key_values, new_input_embeds_out, new_labels_out

        model.prepare_inputs_labels_for_multimodal = _patched_prepare.__get__(model, model.__class__)

    def _patch_llama_forward(self, model):
        """
        纯文本模型：在 forward 中构造并插入基于当前激活池生成的文本 prompts。
        """
        self._orig_forward_fn = model.forward

        def _patched_forward(this, input_ids=None, attention_mask=None, position_ids=None,
                             past_key_values=None, inputs_embeds=None, labels=None, **kwargs):
            if inputs_embeds is not None or input_ids is None:
                return self._orig_forward_fn(input_ids=input_ids, attention_mask=attention_mask,
                                             position_ids=position_ids, past_key_values=past_key_values,
                                             inputs_embeds=inputs_embeds, labels=labels, **kwargs)

            if attention_mask is None:
                attention_mask_ = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                attention_mask_ = attention_mask.bool()
            if position_ids is None:
                position_ids_ = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            else:
                position_ids_ = position_ids
            if labels is None:
                labels_ = torch.full_like(input_ids, IGNORE_INDEX)
            else:
                labels_ = labels

            ids_list = [ids[mask] for ids, mask in zip(input_ids, attention_mask_)]
            labs_list = [lab[mask] for lab, mask in zip(labels_, attention_mask_)]

            embeds_list, new_labels = [], []
            embed_fn = this.get_input_embeddings()

            for ids_i, lab_i in zip(ids_list, labs_list):
                txt_emb = embed_fn(ids_i)                              # [T,D]
                txt_prompt = self.txt_pool.select_prompts(txt_emb)     # [kL,D] or [L,D]
                emb_i = torch.cat([txt_prompt, txt_emb], dim=0)
                lab_i = torch.cat([
                    torch.full((txt_prompt.size(0),), IGNORE_INDEX, device=lab_i.device, dtype=lab_i.dtype),
                    lab_i
                ], dim=0)
                embeds_list.append(emb_i)
                new_labels.append(lab_i)

            tokenizer_model_max_length = getattr(this.config, 'tokenizer_model_max_length', None)
            if tokenizer_model_max_length is not None:
                embeds_list = [x[:tokenizer_model_max_length] for x in embeds_list]
                new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

            max_len = max(x.size(0) for x in embeds_list)
            bsz = len(embeds_list)
            dtype = embeds_list[0].dtype
            dev = embeds_list[0].device

            new_input_embeds_padded = []
            new_labels_padded = torch.full((bsz, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=dev)
            attention_mask_out = torch.zeros((bsz, max_len), dtype=torch.bool, device=dev)
            position_ids_out = torch.zeros((bsz, max_len), dtype=torch.long, device=dev)

            padding_side = getattr(this.config, 'tokenizer_padding_side', 'right')
            for i, (emb_i, lab_i) in enumerate(zip(embeds_list, new_labels)):
                L = emb_i.size(0)
                pad = torch.zeros((max_len - L, emb_i.size(1)), dtype=dtype, device=dev)
                if padding_side == "left":
                    new_input_embeds_padded.append(torch.cat([pad, emb_i], dim=0))
                    if L > 0:
                        new_labels_padded[i, -L:] = lab_i
                        attention_mask_out[i, -L:] = True
                        position_ids_out[i, -L:] = torch.arange(0, L, device=dev)
                else:
                    new_input_embeds_padded.append(torch.cat([emb_i, pad], dim=0))
                    if L > 0:
                        new_labels_padded[i, :L] = lab_i
                        attention_mask_out[i, :L] = True
                        position_ids_out[i, :L] = torch.arange(0, L, device=dev)

            new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

            return self._orig_forward_fn(
                input_ids=None,
                attention_mask=attention_mask_out.to(dtype=attention_mask.dtype) if attention_mask is not None else attention_mask_out,
                position_ids=position_ids_out,
                past_key_values=past_key_values,
                inputs_embeds=new_input_embeds,
                labels=new_labels_padded if labels is not None else None,
                **kwargs
            )

        model.forward = _patched_forward.__get__(model, model.__class__)

    # -------- 序列化工具 --------
    def bind_model(self, model: torch.nn.Module):
        """
        在创建 learner 后、进入任务循环前调用一次。
        便于保存/加载时知道该把参数放到哪个 device/dtype。
        """
        self._bound_model = model

    def _pool_save_dir(self, output_dir: str, task_id: int) -> str:
        return os.path.join(output_dir, str(task_id), "l2p_pools")

    def _pool_meta_from_instance(self, pool: 'L2PPromptPool') -> dict:
        # 把必要结构超参写入 meta，方便恢复时一致
        meta = dict(
            version="v1",
            hidden_size=int(getattr(pool, "hidden_size", 0)),
            num_prompts=int(getattr(pool, "num_prompts", 0)),
            prompt_len=int(getattr(pool, "prompt_len", 0)),
            key_dim=int(getattr(pool, "key_dim", 256)),
            selection_mode=str(getattr(pool, "selection_mode", "semantic")),
            top_k=int(getattr(pool, "top_k", 4)),
            pool_type=str(getattr(pool, "pool_type", "mean")),
            # 下面是注意力/可微选择相关，可不存在则回退默认
            semantic_variant=str(getattr(pool, "semantic_variant", "soft_topk")),
            tau=float(getattr(pool, "tau", 0.07)),
            gumbel_tau=float(getattr(pool, "gumbel_tau", 1.0)),
            eval_hard=bool(getattr(pool, "eval_hard", True)),
            attn_heads=int(getattr(pool, "attn_heads", 8)) if hasattr(pool, "attn_heads") else 8,
            attn_dropout=float(getattr(pool, "attn_dropout", 0.0)) if hasattr(pool, "attn_dropout") else 0.0,
        )
        return meta

    def _build_pool_from_meta(self, meta: dict, is_text: bool) -> 'L2PPromptPool':
        """
        用 meta 恢复一个空的 L2PPromptPool 实例（随后会 load_state_dict）。
        如果 meta 缺字段，使用当前 learner 的 cfg 作为回退。
        """
        # 回退默认从 self.cfg 取（保证兼容老快照）
        def _get(k, default):
            return meta.get(k, self.cfg.get(k, default)) if isinstance(self.cfg, dict) else meta.get(k, default)

        hidden = getattr(self._bound_model.config, "hidden_size", _get("hidden_size", 4096))
        pool = L2PPromptPool(
            hidden_size=hidden,
            num_prompts=int(_get("num_prompts", 64 if is_text else 64)),
            prompt_len=int(_get("prompt_len", 8 if is_text else 8)),
            key_dim=int(_get("key_dim", 256)),
            selection_mode=str(_get("selection_mode", "semantic")),
            top_k=int(_get("top_k", 4)),
            pool_type=str(_get("pool_type", "mean")),
        ).to(next(self._bound_model.parameters()).device
        ).to(dtype=next(self._bound_model.parameters()).dtype)
        # 若需要严格还原可微选择/注意力的细节，也可以在这里把 semantic_variant / tau / attn_heads 等设置回去
        if hasattr(pool, "semantic_variant"): pool.semantic_variant = str(_get("semantic_variant", "soft_topk"))
        if hasattr(pool, "tau"): pool.tau = float(_get("tau", 0.07))
        if hasattr(pool, "gumbel_tau"): pool.gumbel_tau = float(_get("gumbel_tau", 1.0))
        if hasattr(pool, "eval_hard"): pool.eval_hard = bool(_get("eval_hard", True))
        if hasattr(pool, "attn_heads"): pool.attn_heads = int(_get("attn_heads", 8))
        if hasattr(pool, "attn_dropout"): pool.attn_dropout = float(_get("attn_dropout", 0.0))
        return pool

    # -------- 对外保存接口（每个任务结束后调用） --------
    def save_task_pools(self, output_dir: str, task_id: int):
        """
        将【当前任务】的子池快照到磁盘（只保存该任务的 text/vision 子池）。
        路径：{output_dir}/{task_id}/l2p_pools/{txt_pool.pt, vis_pool.pt, meta.json}
        """
        assert task_id in self.task_txt_pools and task_id in self.task_vis_pools, \
            f"[L2P] 当前任务 {task_id} 无可保存的池。"

        save_dir = self._pool_save_dir(output_dir, task_id)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # 以 state_dict 形式保存参数（权重较小，易于跨设备加载）
        txt_pool = self.task_txt_pools[task_id]
        vis_pool = self.task_vis_pools[task_id]
        torch.save(txt_pool.state_dict(), os.path.join(save_dir, "txt_pool.pt"))
        torch.save(vis_pool.state_dict(), os.path.join(save_dir, "vis_pool.pt"))

        # 保存结构元信息（还原必要构造参数）
        meta = dict(
            version="v1",
            text=self._pool_meta_from_instance(txt_pool),
            vision=self._pool_meta_from_instance(vis_pool),
        )
        with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[L2P] 任务 {task_id} 的 L2P 池已保存到：{save_dir}")

    # -------- 对外加载接口（训练循环前调用一次） --------
    def load_existing_pools(self, output_dir: str):
        """
        在 resume/断点续训 或 start_task_id>0 时，从 {output_dir} 下扫描历史任务目录，
        将各任务的子池加载到内存，并构建合并视图池用于“全任务评测”。
        """
        assert hasattr(self, "_bound_model"), \
            "[L2P] 请先调用 bind_model(model) 再 load_existing_pools(...)。"

        base = Path(output_dir)
        if not base.exists():
            return

        loaded_any = False
        for sub in sorted(base.iterdir(), key=lambda p: p.name):
            if not (sub.is_dir() and sub.name.isdigit()):
                continue
            task_id = int(sub.name)
            save_dir = sub / "l2p_pools"
            if not save_dir.exists():
                continue

            meta_path = save_dir / "meta.json"
            txt_path = save_dir / "txt_pool.pt"
            vis_path = save_dir / "vis_pool.pt"
            if not (meta_path.exists() and txt_path.exists() and vis_path.exists()):
                continue

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # 构造空池并加载权重
            txt_meta = meta.get("text", {})
            vis_meta = meta.get("vision", {})
            txt_pool = self._build_pool_from_meta(txt_meta, is_text=True)
            vis_pool = self._build_pool_from_meta(vis_meta, is_text=False)

            device = next(self._bound_model.parameters()).device
            map_loc = device

            state_txt = torch.load(txt_path, map_location=map_loc)
            state_vis = torch.load(vis_path, map_location=map_loc)
            txt_pool.load_state_dict(state_txt, strict=True)
            vis_pool.load_state_dict(state_vis, strict=True)

            # 冻结历史任务池
            for p in txt_pool.parameters(): p.requires_grad = False
            for p in vis_pool.parameters(): p.requires_grad = False

            # 登记
            self.task_txt_pools[task_id] = txt_pool.to(device)
            self.task_vis_pools[task_id] = vis_pool.to(device)
            loaded_any = True

            print(f"[L2P] 已加载任务 {task_id} 的 L2P 池。")

        # 若加载到历史池，立刻构建合并视图并绑定到模型（便于随后全任务评测）
        if loaded_any:
            self.rebuild_merged_pools()
            print("[L2P] 历史任务池已合并为测试视图。")

    def rebuild_merged_pools(self):
        """
        根据当前的 task_*_pools 重建合并视图池，并绑定到 model（仅推理用）。
        可在你动态增删子池后手动调用。
        """
        assert hasattr(self, "_bound_model"), \
            "[L2P] 请先调用 bind_model(model) 再 rebuild_merged_pools()。"

        device = next(self._bound_model.parameters()).device

        if len(self.task_txt_pools) > 0:
            txt_list = [self.task_txt_pools[k] for k in sorted(self.task_txt_pools)]
            self.txt_pool = self.MergedPromptPool(
                subpools=txt_list,
                selection_mode=self.cfg["selection_mode"],
                global_top_k=self.cfg["text_top_k"],
            ).to(device)
        if len(self.task_vis_pools) > 0:
            vis_list = [self.task_vis_pools[k] for k in sorted(self.task_vis_pools)]
            self.vis_pool = self.MergedPromptPool(
                subpools=vis_list,
                selection_mode=self.cfg["selection_mode"],
                global_top_k=self.cfg["vision_top_k"],
            ).to(device)

        # 绑定到模型用于 generate()
        self._bind_active_pools_to_model(self._bound_model)

        # 评测池全冻结
        if self.txt_pool is not None:
            for p in self.txt_pool.parameters(): p.requires_grad = False
        if self.vis_pool is not None:
            for p in self.vis_pool.parameters(): p.requires_grad = False