# llama_flash_attn_monkey_patch.py
from typing import Optional, Tuple
import warnings
import torch
import transformers

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)

# 训练/无 cache 的 qkvpacked 快路径
try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
except ImportError:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func

from flash_attn.bert_padding import unpad_input, pad_input

# ===== 保存原始方法，便于恢复 =====
_ORIG_FORWARD = None
_ORIG_PREPARE_MASK = None
_PATCHED = False


def _ensure_bool_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    return mask.to(torch.bool) if mask.dtype is not torch.bool else mask


def _prepare_decoder_attention_mask_passthrough(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # 训练时把 2D key_padding_mask 原样传下去给 FA，用不到 HF 的复杂mask逻辑
    return attention_mask


def _fa_forward_train_only(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
):
    """
    仅在 (self.training == True 且 past_key_value is None) 时走 FlashAttention。
    其他场景一律回退到 HF 原生 forward（含推理/生成）。
    """
    if (not self.training) or (past_key_value is not None):
        return _ORIG_FORWARD(self, hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache)

    if output_attentions:
        warnings.warn("Patched LlamaAttention 不支持返回注意力权重，恒为 None。")

    bsz, q_len, _ = hidden_states.size()

    # 1) Q/K/V
    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2).contiguous()  # [B, nH, S, D]
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2).contiguous()  # [B, nKV, S, D]
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2).contiguous()  # [B, nKV, S, D]
    )

    # 2) RoPE（训练无 cache：kv_len=S）
    cos, sin = self.rotary_emb(value_states, seq_len=key_states.shape[-2])
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # 3) repeat_kv 到 nH
    key_states = repeat_kv(key_states, self.num_key_value_groups)     # [B, nH, S, D]
    value_states = repeat_kv(value_states, self.num_key_value_groups) # [B, nH, S, D]

    # 4) FlashAttention（qkvpacked）
    key_padding_mask = _ensure_bool_mask(attention_mask)
    qkv = torch.stack([query_states, key_states, value_states], dim=2).transpose(1, 3).contiguous()  # [B, S, 3, nH, D]

    if key_padding_mask is None:
        qkv_flat = qkv.view(-1, 3, self.num_heads, self.head_dim)  # [B*S, 3, nH, D]
        cu_q_lens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device)
        max_s = q_len
        out_unpad = flash_attn_unpadded_qkvpacked_func(
            qkv_flat, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )  # [B*S, nH, D]
        out = out_unpad.view(bsz, q_len, -1)
    else:
        qkv_flat = qkv.view(bsz, q_len, -1)  # [B, S, 3*nH*D]
        _res = unpad_input(qkv_flat, key_padding_mask)
        if isinstance(_res, (tuple, list)):
            if len(_res) == 4:
                qkv_unpad, indices, cu_q_lens, max_s = _res
            elif len(_res) == 5:
                qkv_unpad, indices, cu_q_lens, max_s, _ = _res
            else:
                raise RuntimeError(f"unpad_input 返回 {len(_res)} 个值，需 4 或 5。")
        else:
            raise RuntimeError("unpad_input 未返回 tuple/list。")

        qkv_unpad = qkv_unpad.view(-1, 3, self.num_heads, self.head_dim)
        out_unpad = flash_attn_unpadded_qkvpacked_func(
            qkv_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        out_unpad = out_unpad.view(-1, self.num_heads * self.head_dim)
        out = pad_input(out_unpad, indices, bsz, q_len)  # [B, S, nH*D]

    return self.o_proj(out), None, None


def replace_llama_attn_with_flash_attn():
    """
    保持你原来的函数名：全局打补丁，但实际只在“训练且无 cache”时走 FlashAttention。
    """
    enable_flash_attn()  # 调到同文件的开关函数


def enable_flash_attn():
    """
    开补丁（训练-Only生效）。
    """
    global _ORIG_FORWARD, _ORIG_PREPARE_MASK, _PATCHED
    if _PATCHED:
        return
    LModel = transformers.models.llama.modeling_llama.LlamaModel
    LAttn = transformers.models.llama.modeling_llama.LlamaAttention

    # transformers >= 4.36 删除了 _prepare_decoder_attention_mask, 用 _prepare_4d_causal_attention_mask 代替.
    # 这里仅在旧版上 patch, 新版上跳过 (HF 内置 4D mask 与 FlashAttention 兼容).
    _ORIG_PREPARE_MASK = getattr(LModel, "_prepare_decoder_attention_mask", None)
    _ORIG_FORWARD = LAttn.forward

    cuda_major, _ = torch.cuda.get_device_capability()
    if cuda_major < 8:
        warnings.warn(
            "FlashAttention 训练建议 A100/H100（head_dim>64 的反传限制）"
        )

    if _ORIG_PREPARE_MASK is not None:
        LModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask_passthrough
    LAttn.forward = _fa_forward_train_only
    _PATCHED = True


def restore_llama_attn_to_hf():
    """
    关补丁：彻底恢复 HF 原生 Attention（推理前调用，推理完后可再 enable）。
    """
    global _ORIG_FORWARD, _ORIG_PREPARE_MASK, _PATCHED
    if not _PATCHED:
        return
    LModel = transformers.models.llama.modeling_llama.LlamaModel
    LAttn = transformers.models.llama.modeling_llama.LlamaAttention

    if _ORIG_PREPARE_MASK is not None:
        LModel._prepare_decoder_attention_mask = _ORIG_PREPARE_MASK
    if _ORIG_FORWARD is not None:
        LAttn.forward = _ORIG_FORWARD
    _PATCHED = False


# 一个方便的上下文管理器：在 with 内用原生 HF，离开后自动恢复 FlashAttention
class original_llama_attn_ctx:
    def __enter__(self):
        restore_llama_attn_to_hf()
    def __exit__(self, exc_type, exc, tb):
        enable_flash_attn()
