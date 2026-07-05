# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import glob
import json
import time
import gc
import shutil
import pandas as pd
import shortuuid
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
import logging
import pathlib
from typing import Optional, List
import torch
import torch.distributed as dist
from ddp_debug_tools import attach_debug_probes
import transformers
import tokenizers
from llava_trainer import LLaVATrainer
from llava.data.dataset import DataArguments, make_cl_data_module, smart_tokenizer_and_embedding_resize
from llava.eval.model_vqa_loader import create_data_loader, get_chunk, KeywordsStoppingCriteria
from llava.eval.eval_vqa import eval_single as eval_single_vqa
from llava.eval.eval_caption import eval_single as eval_single_caption
from llava.conversation import conv_templates, SeparatorStyle

from llava import conversation as conversation_lib
from llava.model import *
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn, restore_llama_attn_to_hf, original_llama_attn_ctx, enable_flash_attn
from cl_learner import *

import deepspeed
import sys
import os

import sys

# ===== 原有评测目录 =====
LLAVA_EVAL_DIR = "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/new_eval_tool"
if LLAVA_EVAL_DIR not in sys.path:
    sys.path.insert(0, LLAVA_EVAL_DIR)
# ===== 原有 =====
import evaluate_art_long_sentences,evaluate_FOMC,evaluate_math,evaluate_numglue,evaluate_object,evaluate_science,evaluate_medical_long_sentences,evaluate_ocr

# ===== CoIN 新增评测目录 =====
COIN_EVAL_DIR = "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN/eval"
if COIN_EVAL_DIR not in sys.path:
    sys.path.insert(0, COIN_EVAL_DIR)

# ===== 新增 =====
import eval_science_qa,eval_textvqa,eval_ImagetNet,eval_gqa,eval_vizwiz,eval_grounding,eval_vqav2,eval_ocrvqa

local_rank = None
#question_word
# question_file_dict = {
#     "numglue": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test_for_test/numglue.json",
#     "art": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test_for_test/art.json",
#     "math": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test_for_test/math.json",
#     "fomc": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test_for_test/fomc.json",
#     "medical": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test_for_test/medical.json",
#     "OCR": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test_for_test/OCR.json",
#     "science": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test_for_test/science.json"
#     # "object_relative":"/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/infer_answers/object/test_infer_reasoning_use_relative.json"
# }
question_file_dict = {
    "numglue": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/numglue.json",
    "art": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/art.json",
    "math": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/math.json",
    "fomc": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/fomc.json",
    "medical": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/medical.json",
    "OCR": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/OCR.json",
    "science": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/science.json",
    'ScienceQA':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN/Instructions_10Type/ScienceQA/test.json',
    'TextVQA':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN/Instructions_10Type/TextVQA/test.json',
    'ImageNet':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN/Instructions_10Type/ImageNet/test.json',
    'GQA':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN/Instructions_10Type/GQA/test.json',
    'VizWiz':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN/Instructions_10Type/VizWiz/test.json',
    'Grounding':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN/Instructions_10Type/Grounding/test.json',
    'VQAv2':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN/Instructions_10Type/VQAv2/test.json',
    'OCRVQA':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN/Instructions_10Type/OCRVQA/test.json'
    
}
annotation_file_dict = {
    "numglue": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/infer_answers/numglue/NumGLUE_test_answer.json",
    "art": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/infer_answers/art/test_infer_answer.json",
    "math": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/infer_answers/math/test_infer_answer.json",
    "fomc": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/infer_answers/fomc/test_infer_answer.json",
    "medical": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/infer_answers/medical/test_infer_answer.json",
    "OCR": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/infer_answers/ocr/test_infer_use_reasoning.json",
    "science": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/infer_answers/science/test_ann_infer_answer.json",
    'ScienceQA':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN/Instructions_10Type/ScienceQA/test.json',
    'TextVQA':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN/Instructions_10Type/TextVQA/test.json',
    'ImageNet':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN/Instructions_10Type/ImageNet/test.json',
    'GQA':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN/Instructions_10Type/GQA/test.json',
    'VizWiz':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN/Instructions_10Type/VizWiz/test.json',
    'Grounding':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN/Instructions_10Type/Grounding/test.json',
    'VQAv2':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN/Instructions_10Type/VQAv2/test.json',
    'OCRVQA':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN/Instructions_10Type/OCRVQA/test.json'
}
#question_iamge
image_folder_dict = {
    "numglue": "",
    "art": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "math": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "fomc": "",
    "medical": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "OCR": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "science": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    'ScienceQA':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN',
    'TextVQA':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN',
    'ImageNet':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN',
    'GQA':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN',
    'VizWiz':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN',
    'Grounding':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN',
    'VQAv2':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN',
    'OCRVQA':'/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/CoIN'
} 

def release_memory():
    gc.collect()
    torch.cuda.empty_cache()
    memory_stats()


def memory_stats():
    rank0_print(f"memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2}")
    rank0_print(f"memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2}")


def list_of_ints(arg):
    return [p.strip() for p in arg.split(',') if len(p.strip()) > 0]

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

from packaging import version

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

@dataclass
class TestingArguments:
    conv_mode: str = "vicuna_v1"
    num_beams: int = 1
    max_new_tokens: int = 4096

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    continue_lora_enable:bool = field(default=False)
    continue_lora_path: list[str] = field(default_factory=list)
    # === 可微 QR 门控超参 ===
    # 总秩 r = lora_r，被拆成 G = r / r_g 个 blocks
    gate_rank: int = field(
        default=16,  # 对应 lora_r=128 时，G=4 blocks
        metadata={
            "help": (
                "每个 LoRA block 使用的秩 r_g"
            )
        },
    )

    # 当前 block-level 实现中已经不用 softmax gating 了，这个字段只是为了保持兼容
    gate_tau: float = field(
        default=1.6,
        metadata={
            "help": (
                "历史字段：旧版可微 QR softmax gating 使用的温度；"
                "在当前 CoIN-MoE-LoRA block-level 实现中不再使用，仅作兼容占位。"
            )
        },
    )

    # 同上：现在总是用 A^T 的列空间做 QR，这个参数不会影响行为
    gate_tie: str = field(
        default="A_T",
        metadata={
            "help": (
                "历史字段：门控子空间来源（如 'free' 或 'A_T'）；"
                "当前实现中始终使用 A^T 的列空间做 QR，本参数仅为兼容保留。"
            )
        },
    )
    
    cl: str = "none"
    ewc_lbd: float = 1e8
    lwf_lbd: float = 0.2
    eproj_lbd: float = 1.
    OLoRA_lbd:float = 0.1
    eproj_key_lr: Optional[float] = 1e-2
    eproj_key_steps: Optional[int] = 2000
    pruning_fn: Optional[List[str]] = field(default_factory=lambda: ['mm_projector', 'vision_tower', 'vision_resampler'])
    sparsity_ratios: Optional[List[float]] = field(default_factory=lambda: [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    experts_sparsity_ratios: Optional[float] = 0.1
    #moe balanced loss
    moe_balance_variant: str = field(default="switch", metadata={"help": "switch|cv2"})
    moe_balance_alpha: float = field(default=5e-3)
    moe_imp_w: float = field(default=1.0)
    moe_load_w: float = field(default=1.0)      


    # === MoE-LoRA 子空间正交正则 ===
    moelora_orth_lambda: float = field(
        default=1e-4,
        metadata={"help": "MoE-LoRA Rank-1 稳定化正则系数 λ，用于对旧任务重要方向的 A/B 施加 L2 约束"}
    )
    expert_num:int= field(
        default=2,
        metadata={"help": " 专家数量"}
    )
    # === 重要性统计 & 正则 warmup ===
    moelora_ema_decay: float = field(
        default=0.9,
        metadata={
            "help": (
                "统计 rank-1 重要性 I_t(e,j) 时的 EMA 衰减系数 β："
                "I_t = β * I_{t-1} + (1-β) * current_batch_rank_energy。"
                "越接近 1，越平滑（抗噪更强，但响应更慢）。"
            )
        },
    )

    moelora_reg_warmup_ratio: float = field(
        default=0.2,
        metadata={
            "help": (
                "按当前任务训练进度（0~1）做正则 warmup 的比例："
                "在任务前 warmup_ratio 的训练过程中，正则系数从 0 线性升到 moelora_orth_lambda；"
                "之后保持为 moelora_orth_lambda。对不同数据集/步数自适应。"
            )
        },
    )

    moelora_reg_warmup_steps: int = field(
        default=0,
        metadata={
            "help": (
                "备用：按 step 数做 warmup 的步数（loss 调用次数），"
                "仅在拿不到任务进度 cl_task_progress 时使用。"
                "设为 0 表示只用 moelora_reg_warmup_ratio，不按步数 warmup。"
            )
        },
    )

    # === A/B 分支权重 & 重要性裁剪 ===
    moelora_B_reg_scale: float = field(
        default=0.1,
        metadata={
            "help": (
                "LoRA-B 分支相对 LoRA-A 的正则缩放系数："
                "reg = sum(I * (A-A_ref)^2) + B_reg_scale * sum(I * (B-B_ref)^2)。"
                "通常设得较小（如 0.05），意味着强约束方向 A，弱约束 B，"
                "给当前任务留一点适配空间。"
            )
        },
    )

    moelora_importance_clip: float = field(
        default=10.0,
        metadata={
            "help": (
                "对归一化后的 rank-1 重要性 I 做上界裁剪："
                "I_norm = clamp(I / mean(I), max = importance_clip)。"
                "防止某些维度的 I 极端大导致梯度过分集中，训练不稳定。"
            )
        },
    )
    moelora_importance_top_p: float = field(
        default=0.4,
        metadata={
            "help": (
                "EWC 正则只作用在最重要的 top-p 比例 rank 方向上："
                "先把 I 展平成一维并排序，只保留累积比例为 top_p 的最大值，其余置 0，"
                "再 reshape 回 [E, r]。例如 0.3 表示只约束最重要的 30%% 方向，"
                "其余 70%% 自由度更大，有利于适应新任务。"
            )
        },
    )   
    gate_router: str = field(default="qr")  #隐式router的方式
    moelora_reg_enable:bool = field(default=True)
    #正则化控制参数,控制是
    moelora_reg_random_importance: bool = field(
        default=False,
        metadata={"help": "If True, use randomly shuffled rank importance for MoE-LoRA regularization (ablation)."},
    )     

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias,adapter_task_name):
    if bias == "none":
        
        to_return = {k: t for k, t in named_params if ("lora_" in k) and (adapter_task_name in k)}
    elif bias == "all":
        
        to_return = {k: t for k, t in named_params if ("lora_" in k or "bias" in k)and (adapter_task_name in k)}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        
        for k, t in named_params:
            if ("lora_" in k) and (adapter_task_name in k):
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif ("bias" in k) and (adapter_task_name in k):
                maybe_lora_bias[k] = t
        
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def _append_time_log(record, log_path):
    """
    追加一条时间记录到 log_path (JSONL)。
    record: 一个 dict，例如 {"phase": "train", "task_id": 1, "elapsed_sec": 123.4, ...}
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match, keys_not_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match) and not any(key_match in k for key_match in keys_not_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        # 排除 Dropout 层
        if isinstance(module, torch.nn.Dropout):
            continue
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False) or hasattr(trainer.model.get_model(), "e_mm_projector"):
        # Only save Adapter
        keys_to_match = ['mm_projector', 'e_mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])
        keys_not_match = ['task_encoder']

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match, keys_not_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

# —— 工具函数 —— #
def _touch(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a"):
        os.utime(path, None)

def _file_ready(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0

def _wait_for_flags(dir_path: str, patterns: list[str], timeout_s: int, poll_s: float = 2.0):
    """
    轮询等待 dir_path 下所有 pattern 文件都存在(不要求非空)，超时返回 False。
    """
    end_t = time.time() + timeout_s
    while time.time() < end_t:
        ok = True
        for pat in patterns:
            if not os.path.exists(os.path.join(dir_path, pat)):
                ok = False
                break
        if ok:
            return True
        time.sleep(poll_s)
    return False

def _wait_and_merge_chunks_rank0(chunks_dir: str, answers_file: str, num_chunks: int, timeout_s: int = 600):
    """
    rank0：等待所有 rank 写完 done.<rank>(有超时)，然后合并“已有”的分片(缺的跳过)，并写 MERGED.ok
    """
    # 1) 等 done flags(这一步只证明 rank 循环完成，不保证文件非空)
    done_flags = [f"done.{i}" for i in range(num_chunks)]
    ok = _wait_for_flags(chunks_dir, done_flags, timeout_s=timeout_s)
    if not ok:
        print(f"[EVAL][merge][TIMEOUT] waited {timeout_s}s for done flags; will merge whatever exists.")

    # 2) 合并已有分片
    parts = [os.path.join(chunks_dir, f"{num_chunks}-{i}.jsonl") for i in range(num_chunks)]
    with open(answers_file, "w") as out:
        merged = 0
        for p in parts:
            if _file_ready(p):
                with open(p, "r") as fin:
                    for line in fin:
                        out.write(line)
                merged += 1
    print(f"[EVAL][merge] merged {merged}/{num_chunks} chunks into {answers_file}")

    # 3) 写 MERGED.ok，通知其余 rank 可以进入下一评测
    _touch(os.path.join(chunks_dir, "MERGED.ok"))

def evaluate(args, tokenizer, model, model_task_id, task_id, task_val, output_dir):

    from CoIN.peft.tuners.coinmoelora import CoINMOELoraLinear
    def call_eval(module, ann, pred, out=None):
        if hasattr(module, "eval_results"):
            try:
                return module.eval_results(ann, pred, out)
            except TypeError:
                return module.eval_results(ann, pred)
        if hasattr(module, "eval_single"):
            return module.eval_single(ann, pred)
        raise AttributeError(f"[EVAL] {module.__name__} has neither eval_results nor eval_single")

    def _load_questions_auto(path):
        """兼容 jsonl(逐行) 和 json(整体 list/dict)。"""
        with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
            first_nonempty = ""
            # 探测首个非空白字符
            pos = f.tell()
            while True:
                ch = f.read(1)
                if ch == "":
                    break
                if not ch.isspace():
                    first_nonempty = ch
                    break
            f.seek(pos)

            if first_nonempty == "[" or first_nonempty == "{":
                # json 文件（整体）
                return json.load(f)
            else:
                # jsonl 文件（逐行）
                f.seek(0)
                data = []
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    data.append(json.loads(s))
                return data

    def _sanitize_jsonl_inplace(path, verbose=True):
        """
        清洗 jsonl：去掉空行/坏行，保证后续 eval 的 json.loads(line) 不会炸。
        """
        if (not os.path.isfile(path)) or os.path.getsize(path) == 0:
            return

        good_lines = []
        bad_cnt = 0
        empty_cnt = 0

        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    empty_cnt += 1
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    bad_cnt += 1
                    if verbose:
                        print(f"[EVAL][sanitize][WARN] drop bad json line {ln}: {s[:120]}")
                    continue
                good_lines.append(json.dumps(obj, ensure_ascii=False))

        # 覆写回去（保证每行一个 JSON）
        with open(path, "w", encoding="utf-8") as f:
            for s in good_lines:
                f.write(s + "\n")

        if verbose and (empty_cnt > 0 or bad_cnt > 0):
            print(f"[EVAL][sanitize] cleaned {path}: drop empty={empty_cnt}, bad={bad_cnt}, keep={len(good_lines)}")

    rank0_print(f"Testing model:{model_task_id} on {task_val}")

    answers_file = os.path.join(output_dir, "predictions", f"{task_id}-{task_val}.jsonl")
    print(f"这是第{model_task_id}个模型的第{task_id}个任务的推理，任务的名称是:{task_val}")
    evaluation_results_file = os.path.join(output_dir, "evaluations", f"{task_id}-{task_val}.jsonl")

    router_root_dir = os.path.join(output_dir, "router")
    os.makedirs(router_root_dir, exist_ok=True)
    router_file_merged = os.path.join(router_root_dir, f"{task_id}-{task_val}.jsonl")

    exp_root = os.path.dirname(output_dir)
    eval_time_log_path = os.path.join(exp_root, "time_eval_log.jsonl")

    acc = 0.0

    try:
        # === 1) 推理（answers_file 不存在或为空才推）===
        if (not os.path.isfile(answers_file)) or os.path.getsize(answers_file) == 0:
            infer_start = time.time()

            model.config.tokenizer_padding_side = "left"
            tokenizer.padding_side = "left"

            image_processor = model.get_vision_tower().image_processor
            num_chunks = int(os.getenv("WORLD_SIZE", "1"))
            chunk_idx = local_rank

            question_file = question_file_dict[task_val]
            image_folder = image_folder_dict[task_val]

            pred_chunks_dir = os.path.join(output_dir, "predictions", f"{task_id}-{task_val}")
            answers_chunk_file = os.path.join(pred_chunks_dir, f"{num_chunks}-{chunk_idx}.jsonl")

            router_chunks_dir = os.path.join(output_dir, "router", f"{task_id}-{task_val}")
            router_chunk_file = os.path.join(router_chunks_dir, f"{num_chunks}-{chunk_idx}.jsonl")

            questions = _load_questions_auto(question_file)
            questions = get_chunk(questions, num_chunks, chunk_idx)

            data_loader = create_data_loader(
                questions, image_folder, tokenizer, image_processor, model.config, args.conv_mode
            )

            os.makedirs(os.path.dirname(answers_chunk_file), exist_ok=True)
            os.makedirs(os.path.dirname(router_chunk_file), exist_ok=True)

            ans_file = open(answers_chunk_file, "w", encoding="utf-8")
            router_file = open(router_chunk_file, "w", encoding="utf-8")

            if hasattr(model.get_model(), "e_mm_projector"):
                model.get_model().e_mm_projector.init_task_id_retrieve_acc()

            conv = conv_templates[args.conv_mode].copy()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]

            model.gradient_checkpointing_disable()
            model.eval()

            moe_layers = []
            for name, module in model.named_modules():
                if isinstance(module, CoINMOELoraLinear):
                    module.analysis_collect = True
                    module.reset_analysis_buffer()
                    if getattr(module, "analysis_id", None) is None:
                        module.analysis_id = name
                    moe_layers.append((name, module))

            if local_rank == 0:
                print(f"[EVAL] Found {len(moe_layers)} CoIN-MoE-LoRA layers for router logging (mean mode).")

            for i, batch in tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                miniters=max(1, len(data_loader)//10),
                disable=(local_rank != 0)
            ):
                input_ids = batch["input_ids"]
                convs = batch["convs"]
                attn_mask = batch["attention_mask"]
                question_ids = batch["question_ids"]
                prompts = batch["prompts"]

                batch_size = input_ids.size(0)

                image_tensor = None
                if 'images' in batch:
                    image_tensor = batch["images"].to(device='cuda', non_blocking=True, dtype=torch.bfloat16)

                input_ids = input_ids.to(device='cuda', non_blocking=True)
                attn_mask = attn_mask.to(device='cuda', non_blocking=True)
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                for _, module in moe_layers:
                    module.reset_analysis_buffer()

                try:
                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids=input_ids,
                            attention_mask=attn_mask,
                            images=image_tensor,
                            convs=convs,
                            do_sample=False,
                            temperature=0.0,
                            top_p=1.0,
                            num_beams=args.num_beams,
                            max_new_tokens=args.max_new_tokens,
                            min_new_tokens=1,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                        )
                except Exception as e:
                    print(f"[EVAL][gen] step {i} failed on rank {local_rank}: {repr(e)}; skip this batch.")
                    continue

                router_per_sample = [dict() for _ in range(batch_size)]
                for layer_name, module in moe_layers:
                    stats = module.get_analysis_stats()
                    G_mean = stats.get("G_mean", None)
                    ent_mean = stats.get("entropy_mean", None)

                    if G_mean is None or G_mean.size(0) != batch_size:
                        continue

                    G_list = G_mean.tolist()
                    ent_list = ent_mean.tolist() if (ent_mean is not None and ent_mean.size(0) == batch_size) else [None]*batch_size

                    for b in range(batch_size):
                        router_per_sample[b][layer_name] = {
                            "G_mean": G_list[b],
                            "entropy_mean": ent_list[b],
                        }

                input_token_len = input_ids.shape[1]
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)

                ans_id = shortuuid.uuid()
                for b_idx, (qid, prompt, output) in enumerate(zip(question_ids, prompts, outputs)):
                    qid_str = str(qid)
                    pred_text = output.strip()

                    ans_file.write(json.dumps({
                        "question_id": qid_str,
                        "text": pred_text,

                        # 兼容字段
                        "questionId": qid_str,
                        "prediction": pred_text,

                        "prompt": prompt,
                        "answer_id": ans_id,
                        "model_id": str(task_id),
                        "metadata": {}
                    }, ensure_ascii=False) + "\n")

                    router_record = {
                        "question_id": qid_str,
                        "model_task_id": model_task_id,
                        "eval_task_id": task_id,
                        "eval_task_name": task_val,
                        "router": router_per_sample[b_idx],
                    }
                    router_file.write(json.dumps(router_record, ensure_ascii=False) + "\n")

                ans_file.flush()
                router_file.flush()

            ans_file.close()
            router_file.close()

            pred_my_done_flag = os.path.join(pred_chunks_dir, f"done.{chunk_idx}")
            _touch(pred_my_done_flag)
            router_my_done_flag = os.path.join(router_chunks_dir, f"done.{chunk_idx}")
            _touch(router_my_done_flag)

            if local_rank == 0:
                try:
                    _wait_and_merge_chunks_rank0(pred_chunks_dir, answers_file, num_chunks, timeout_s=600)
                except Exception as e:
                    print(f"[EVAL][merge] unexpected error during merge on rank0 (pred): {repr(e)}; still mark ok.")
                    _touch(os.path.join(pred_chunks_dir, "MERGED.ok"))

                try:
                    _wait_and_merge_chunks_rank0(router_chunks_dir, router_file_merged, num_chunks, timeout_s=600)
                except Exception as e:
                    print(f"[EVAL][merge] unexpected error during merge on rank0 (router): {repr(e)}; still mark ok.")
                    _touch(os.path.join(router_chunks_dir, "MERGED.ok"))
            else:
                _wait_for_flags(pred_chunks_dir, ["MERGED.ok"], timeout_s=600)
                _wait_for_flags(router_chunks_dir, ["MERGED.ok"], timeout_s=600)

            infer_elapsed = time.time() - infer_start
            if local_rank in (0, -1):
                _append_time_log(
                    {
                        "phase": "inference",
                        "model_task_id": model_task_id,
                        "eval_task_id": task_id,
                        "eval_task_name": task_val,
                        "elapsed_sec": infer_elapsed,
                        "world_size": int(os.getenv("WORLD_SIZE", "1")),
                        "local_rank": int(local_rank),
                    },
                    eval_time_log_path,
                )
        else:
            print(f"推理答案 {answers_file} 已存在或非空，跳过推理。")

        # === 2) rank0 做评测 ===
        if local_rank == 0:
            # ★关键：评测前清洗 answers_file，彻底解决 JSONDecodeError
            _sanitize_jsonl_inplace(answers_file, verbose=True)

            pred_subdir = os.path.join(output_dir, "predictions", f"{task_id}-{task_val}")
            if os.path.isdir(pred_subdir):
                shutil.rmtree(pred_subdir, ignore_errors=True)

            router_subdir = os.path.join(output_dir, "router", f"{task_id}-{task_val}")
            if os.path.isdir(router_subdir):
                shutil.rmtree(router_subdir, ignore_errors=True)

            eval_metric_start = time.time()

            if task_val == "art":
                acc = call_eval(evaluate_art_long_sentences, annotation_file_dict[task_val], answers_file, evaluation_results_file)
            elif task_val == "numglue":
                acc = call_eval(evaluate_numglue, annotation_file_dict[task_val], answers_file, evaluation_results_file)
            elif task_val == "math":
                acc = call_eval(evaluate_math, annotation_file_dict[task_val], answers_file, evaluation_results_file)
            elif task_val == "fomc":
                acc = call_eval(evaluate_FOMC, annotation_file_dict[task_val], answers_file, evaluation_results_file)
            elif task_val == "medical":
                acc = call_eval(evaluate_medical_long_sentences, annotation_file_dict[task_val], answers_file, evaluation_results_file)
            elif task_val == "OCR":
                acc = call_eval(evaluate_ocr, annotation_file_dict[task_val], answers_file, evaluation_results_file)
            elif task_val == "science":
                acc = call_eval(evaluate_science, annotation_file_dict[task_val], answers_file, evaluation_results_file)
            elif task_val == "object":
                acc = call_eval(evaluate_object, annotation_file_dict[task_val], answers_file, evaluation_results_file)

            elif task_val == "ScienceQA":
                acc = call_eval(eval_science_qa, annotation_file_dict[task_val], answers_file, evaluation_results_file)
            elif task_val == "TextVQA":
                acc = call_eval(eval_textvqa, annotation_file_dict[task_val], answers_file, evaluation_results_file)
            elif task_val == "ImageNet":
                acc = call_eval(eval_ImagetNet, annotation_file_dict[task_val], answers_file, evaluation_results_file)
            elif task_val == "GQA":
                acc = call_eval(eval_gqa, annotation_file_dict[task_val], answers_file, evaluation_results_file)
            elif task_val == "VizWiz":
                acc = call_eval(eval_vizwiz, annotation_file_dict[task_val], answers_file, evaluation_results_file)
            elif task_val == "Grounding":
                acc = call_eval(eval_grounding, annotation_file_dict[task_val], answers_file, evaluation_results_file)
            elif task_val == "VQAv2":
                acc = call_eval(eval_vqav2, annotation_file_dict[task_val], answers_file, evaluation_results_file)
            elif task_val == "OCRVQA":
                acc = call_eval(eval_ocrvqa, annotation_file_dict[task_val], answers_file, evaluation_results_file)
            else:
                raise ValueError(f"Unknown task : {task_val}")

            eval_metric_elapsed = time.time() - eval_metric_start
            _append_time_log(
                {
                    "phase": "eval_metric",
                    "model_task_id": model_task_id,
                    "eval_task_id": task_id,
                    "eval_task_name": task_val,
                    "elapsed_sec": eval_metric_elapsed,
                    "world_size": int(os.getenv("WORLD_SIZE", "1")),
                    "local_rank": int(local_rank),
                },
                eval_time_log_path,
            )

            model.config.tokenizer_padding_side = "right"
            tokenizer.padding_side = "right"
            print(f"任务 {task_val} 推理结果 {answers_file} 的正确率是 {acc}")
            print(f"任务 {task_val} 的 router(mean) 统计已保存到 {router_file_merged}")
            return acc

        else:
            model.config.tokenizer_padding_side = "right"
            tokenizer.padding_side = "right"
            return 0.0

    except Exception:
        try:
            model.config.tokenizer_padding_side = "right"
            tokenizer.padding_side = "right"
        except Exception:
            pass
        raise


def load_lora_parameter(model, training_args):
    """
    普通 LoRA 注入/续训入口。

    设计目标（与主代码的调用方式兼容）：
    - 主代码只在 train() 开头调用一次：
        if cl in ["moelora", ...]: load_moe_lora(...)
        else: load_lora_parameter(...)

    - 对 cl == "olora"：
        借鉴 load_moe_lora 的“prev_cnt + 1”方式：
        * 依赖 continue_lora_path 代表“已完成任务数”
        * 当前任务 adapter = task_{prev_cnt+1}
        * 加载历史 adapters
        * 冻结历史 LoRA，只训练当前 LoRA

      这种写法 **适用于“每个任务训练完会重启脚本”** 的运行范式。

    - 对其它 cl：
        保持你旧逻辑：本次 run 只初始化一个新 adapter，
        并可选加载/冻结历史 adapters。
    """
    import os
    import torch
    from peft import LoraConfig, get_peft_model, PeftModel

    # ---------- 统一构造 LoRA Config ----------
    def _build_lora_config():
        return LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=getattr(training_args, "lora_dropout", 0.0),
            bias=getattr(training_args, "lora_bias", "none"),
            task_type="CAUSAL_LM",
        )

    # ---------- 精度设置 ----------
    def _apply_dtype(m):
        if getattr(training_args, "bits", 0) == 16:
            if getattr(training_args, "bf16", False):
                m.to(torch.bfloat16)
            elif getattr(training_args, "fp16", False):
                m.to(torch.float16)

    # ============================================================
    #  A) OLoRA 模式：模仿 load_moe_lora 的任务编号推断
    # ============================================================
    if getattr(training_args, "cl", None) == "olora":
        _rank0_print = globals().get("rank0_print", print)

        if not getattr(training_args, "lora_enable", False):
            _rank0_print("[OLoRA] lora_enable=False, skip injection.")
            return model

        prev_cnt = len(getattr(training_args, "continue_lora_path", []))
        cur_task_id = prev_cnt + 1
        cur_adapter_name = f"task_{cur_task_id}"

        _rank0_print(
            f"[OLoRA] detected {prev_cnt} finished tasks, "
            f"current training task_id={cur_task_id}, adapter={cur_adapter_name}"
        )

        lora_config = _build_lora_config()

        # 1) 注入当前任务 adapter
        #    - 大多数情况下此时 model 还是“裸模型”，用 get_peft_model 即可
        #    - 若你未来改成同进程多任务，也能兼容 PeftModel
        if not isinstance(model, PeftModel):
            model = get_peft_model(model, lora_config, adapter_name=cur_adapter_name)
        else:
            peft_cfg = getattr(model, "peft_config", {})
            if cur_adapter_name not in peft_cfg:
                model.add_adapter(cur_adapter_name, lora_config)

        _apply_dtype(model)

        # 2) 加载历史 adapters（按 continue_lora_path）
        if getattr(training_args, "continue_lora_enable", False):
            paths = getattr(training_args, "continue_lora_path", [])
            if len(paths) > 0:
                _rank0_print("[OLoRA] loading previous LoRA adapters...")
            for task_ada_id, adapter_path in enumerate(paths):
                task_ada_id += 1
                old_adapter_name = f"task_{task_ada_id}"

                # 已存在则跳过
                peft_cfg = getattr(model, "peft_config", {})
                if old_adapter_name in peft_cfg:
                    continue

                if adapter_path is not None and os.path.isdir(adapter_path):
                    _rank0_print(f"[OLoRA] load adapter {old_adapter_name} from {adapter_path}")
                    try:
                        model.load_adapter(adapter_path, adapter_name=old_adapter_name)
                    except Exception as e:
                        _rank0_print(f"[OLoRA][WARN] failed to load {old_adapter_name}: {repr(e)}")

        # 3) 冻结旧 LoRA，仅训练当前 adapter 的 LoRA_A/B
        for n, p in model.named_parameters():
            if ("lora_A" in n) or ("lora_B" in n):
                if cur_adapter_name in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        # 4) 激活当前 adapter
        try:
            model.set_adapter(cur_adapter_name)
        except Exception:
            pass

        model.train()
        _rank0_print(f"[OLoRA] adapter ready for training: {cur_adapter_name}")
        return model

    # ============================================================
    #  B) 非 OLoRA：保持你原有“单次 run 注入一个新 LoRA”的逻辑
    # ============================================================
    from peft import set_peft_model_state_dict  # 保持你原 import 兼容

    task_ada_max_id = len(getattr(training_args, "continue_lora_path", [])) + 1

    if training_args.lora_enable:
        new_task_name = f"task_{task_ada_max_id}"
        lora_config = _build_lora_config()

        _apply_dtype(model)

        rank0_print("Adding initial LoRA adapters...")
        model = get_peft_model(model, lora_config, adapter_name=new_task_name)

        if getattr(training_args, "continue_lora_enable", False):
            paths = getattr(training_args, "continue_lora_path", [])
            if len(paths) > 0:
                rank0_print("Loading multiple existing LoRA adapters...")
                for task_ada_id, adapter_path in enumerate(paths):
                    task_ada_id += 1
                    adapter_name = f"task_{task_ada_id}"
                    rank0_print(f"Loading LoRA adapter from {adapter_path}...")
                    model.load_adapter(adapter_path, adapter_name)

            old_lora_names = {
                name for name, _ in model.named_parameters()
                if ("lora_A" in name or "lora_B" in name)
                and (f"task_{task_ada_max_id}" not in name)
            }

            for name, param in model.named_parameters():
                if name in old_lora_names:
                    param.requires_grad = False
            rank0_print("Frozen all loaded LoRA adapters. Only new LoRA will be trained.")

            model.set_adapter(new_task_name)
    else:
        rank0_print("LoRA is not enabled. Skipping LoRA adapter addition.")

    return model


def load_moe_lora(model, training_args):
    """
    注入 / 续接 CoIN-MoE-LoRA 适配器（用于 cl == "moelora"）。

    设计要点：
    - 使用适配器名 task_{k}（与 train() 中的保存逻辑完全兼容）；
    - 第一个任务（无历史路径）时，从零初始化 task_1；
    - 后续任务 k>1 时：
        * 从上一个任务目录 <output_dir>/<k-1>/task_{k-1}/adapter_model.* 载入权重；
        * 将 key 中的 "task_{k-1}" 替换为 "task_{k}"，加载到当前模型；
        * 相当于“续接”上一任务的 LoRA 参数，然后在其上继续做 CL + 正则；
    - 只保证当前 adapter 的 LoRA / gate 参数是 requires_grad=True，其他 LoRA 参数全部冻结。
    """


    from CoIN.peft import get_peft_model, TaskType
    from CoIN.peft.tuners.coinmoelora import CoINMOELoraConfig, CoINMOELoraLinear


    # rank0_print 兼容（如果外面有定义就用外面的）
    _rank0_print = globals().get("rank0_print", print)

    # ===== 0. 如果没开 LoRA，直接返回 =====
    if not getattr(training_args, "lora_enable", False):
        _rank0_print("[MoE-LoRA] lora_enable=False，跳过注入。")
        return model

    # ===== 1. 当前要训练的是第几个任务？ =====
    prev_cnt = len(getattr(training_args, "continue_lora_path", []))  # 已完成任务数
    cur_task_id = prev_cnt + 1                                       # 当前任务编号（1-based）
    cur_adapter_name = f"task_{cur_task_id}"

    _rank0_print(
        f"[MoE-LoRA] detected {prev_cnt} finished tasks, "
        f"current training task_id={cur_task_id}, adapter={cur_adapter_name}"
    )

    # ===== 2. 构造 CoIN-MoE-LoRA Config =====
    cfg = CoINMOELoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type=TaskType.CAUSAL_LM_CoIN,
        # QR 门控超参：来自 TrainingArguments
        gate_rank=int(getattr(training_args, "gate_rank", 4)),
        gate_tau=float(getattr(training_args, "gate_tau", 1.6)),
        gate_tie=str(getattr(training_args, "gate_tie", "A_T")),
        gate_router=str(getattr(training_args, "gate_router", "qr")),  # <<< 新增：选择 router 模式 "qr"|"a"
        # expert_num 建议直接用 num_experts（如果你在 args 里这么定义的）
        expert_num=int(getattr(training_args, "num_experts", 2)),
    )

    # ===== 3. 注入当前任务的 MoE-LoRA 适配器 =====
    # 注意：此时 adapter=cur_adapter_name 里还是随机初始化的权重
    model = get_peft_model(model, cfg, adapter_name=cur_adapter_name)

    # 精度设置：与 train() 中的一致（bits=16 时，按 fp16/bf16 走）
    if getattr(training_args, "bits", 0) == 16:
        if getattr(training_args, "bf16", False):
            model.to(torch.bfloat16)
        elif getattr(training_args, "fp16", False):
            model.to(torch.float16)

    # 尝试把当前 adapter 设为激活 adapter
    try:
        model.set_adapter(cur_adapter_name)
    except Exception:
        pass

    # ===== 4. 如果有历史任务：从“最后一个任务”的 LoRA 快照续接参数 =====
    if prev_cnt > 0 and getattr(training_args, "continue_lora_enable", False):
        last_task_id = prev_cnt
        last_task_dir = training_args.continue_lora_path[-1]  # 例如 .../<output_root>/3
        last_adapter_name = f"task_{last_task_id}"

        adapter_dir = os.path.join(last_task_dir, last_adapter_name)
        _rank0_print(
            f"[MoE-LoRA] trying to init {cur_adapter_name} "
            f"from previous adapter {last_adapter_name} in: {adapter_dir}"
        )

        if os.path.isdir(adapter_dir):
            # 先尝试 safetensors，再尝试 bin
            weight_file = None
            st_path = os.path.join(adapter_dir, "adapter_model.safetensors")
            bin_path = os.path.join(adapter_dir, "adapter_model.bin")
            if os.path.exists(st_path):
                weight_file = st_path
                use_safe = True
            elif os.path.exists(bin_path):
                weight_file = bin_path
                use_safe = False

            if weight_file is not None:
                try:
                    if use_safe:
                        from safetensors.torch import load_file as safe_load
                        prev_state = safe_load(weight_file)
                    else:
                        prev_state = torch.load(weight_file, map_location="cpu")

                    remapped_state = {}
                    old_tag = last_adapter_name
                    new_tag = cur_adapter_name
                    for k, v in prev_state.items():
                        # 把 key 里的 task_{last} 改成 task_{cur}
                        new_k = k.replace(old_tag, new_tag)
                        remapped_state[new_k] = v

                    missing, unexpected = model.load_state_dict(remapped_state, strict=False)

                    _rank0_print(
                        f"[MoE-LoRA] loaded previous LoRA weights from {weight_file} "
                        f"into adapter {cur_adapter_name} (from {last_adapter_name}).\n"
                        f"  - missing keys: {len(missing)}\n"
                        f"  - unexpected keys: {len(unexpected)}"
                    )
                except Exception as e:
                    _rank0_print(
                        f"[MoE-LoRA][WARN] failed to load previous adapter weights "
                        f"from {weight_file}: {repr(e)}"
                    )
            else:
                _rank0_print(
                    f"[MoE-LoRA][WARN] no adapter_model.[safetensors|bin] "
                    f"found in {adapter_dir}, start from random init."
                )
        else:
            _rank0_print(
                f"[MoE-LoRA][WARN] adapter_dir {adapter_dir} does not exist, "
                f"start {cur_adapter_name} from random init."
            )
    else:
        _rank0_print("[MoE-LoRA] no previous task to continue from, start from fresh LoRA (task_1).")

    # ===== 5. 只训练“当前 adapter”的 LoRA / gate 参数，其他 LoRA 参数全部冻结 =====
    for n, p in model.named_parameters():
        # LoRA / gate 相关参数的粗略判定
        is_lora_param = ("lora_A" in n) or ("lora_B" in n) or ("gate_P" in n) or ("gate_P_map" in n)  or ("lora_router" in n)
        if is_lora_param:
            if cur_adapter_name in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

    # 同时，对于 CoINMOELoraLinear 中的 gate_P（free 模式），再保险处理一下：
    try:
        for mod in model.modules():
            if isinstance(mod, CoINMOELoraLinear):
                # 只允许当前 adapter 的门控参数训练
                if getattr(mod, "gate_tie", "A_T") == "free" and hasattr(mod, "gate_P") and mod.gate_P is not None:
                    for p in mod.gate_P:
                        p.requires_grad_(True)
    except Exception:
        pass

    model.train()
    _rank0_print(
        f"[MoE-LoRA] adapter ready for training: {cur_adapter_name} "
        f"(cl=moelora, prev_cnt={prev_cnt})."
    )

    return model



# ===== GateTauSchedulerCallback: 动态退火 gate_tau =====
import math
from transformers import TrainerCallback

def _set_all_gate_tau(model, tau: float, verbose: bool = False):
    """把模型里所有带 gate_tau 属性的模块都更新为 tau。"""
    n = 0
    for m in model.modules():
        if hasattr(m, "gate_tau"):
            try:
                m.gate_tau = float(tau)
                n += 1
            except Exception:
                pass
    return n

class GateTauSchedulerCallback(TrainerCallback):
    """
    在训练过程中为可微 QR 门控的 CoIN-MoE-LoRA 做 gate_tau 退火：
    - 线性: tau(s) = start + (end - start) * progress
    - 余弦: tau(s) = end + 0.5*(start - end)*(1 + cos(pi * progress))
    支持只在 [start_step, end_step] 区间内生效，区间外固定边界值。
    """
    def __init__(
        self,
        start_tau: float = 1.5,
        end_tau: float = 0.7,
        total_steps: int = None,      # 若为 None，则在首次 on_step_end 时用 state.max_steps 自动填充
        mode: str = "linear",         # "linear" | "cosine"
        start_step: int = 0,
        end_step: int = None,         # 若为 None，默认为 total_steps
        log_every_n_steps: int = 100, # 记录频率
    ):
        assert mode in ("linear", "cosine")
        self.start_tau = float(start_tau)
        self.end_tau = float(end_tau)
        self.total_steps = total_steps
        self.mode = mode
        self.start_step = int(start_step)
        self.end_step = end_step
        self.log_every_n_steps = int(log_every_n_steps)

        # 运行时填充
        self._inited = False

    def _compute_tau(self, step: int):
        # 边界外：固定
        if step <= self.start_step:
            return self.start_tau
        if step >= self.end_step:
            return self.end_tau
        # 区间内：按 progress 退火
        span = max(1, self.end_step - self.start_step)
        progress = (step - self.start_step) / span  # [0,1]
        if self.mode == "linear":
            tau = self.start_tau + (self.end_tau - self.start_tau) * progress
        else:  # cosine
            tau = self.end_tau + 0.5 * (self.start_tau - self.end_tau) * (1.0 + math.cos(math.pi * progress))
        return tau

    def _maybe_init(self, state):
        if not self._inited:
            if self.total_steps is None:
                # Trainer 会在开训时给出 max_steps
                self.total_steps = int(getattr(state, "max_steps", 0)) or None
            if self.end_step is None:
                self.end_step = self.total_steps if self.total_steps is not None else self.start_step
            self._inited = True

    def on_step_end(self, args, state, control, **kwargs):
        self._maybe_init(state)
        if self.total_steps is None:
            # 还拿不到 max_steps(极少数情况)，先跳过
            return control

        step = int(state.global_step)
        tau = self._compute_tau(step)

        # 更新模型里的所有 gate_tau
        model = kwargs.get("model", None)
        if model is not None:
            changed = _set_all_gate_tau(model, tau)
            if changed > 0 and (step % self.log_every_n_steps == 0) and (args.local_rank in (0, -1)):
                print(f"[GateTauScheduler] step={step} tau={tau:.4f} (updated {changed} modules)")

        return control

    # 可选：在每个 epoch 开始时也强制同步一次，避免 resume 时错位
    def on_epoch_begin(self, args, state, control, **kwargs):
        self._maybe_init(state)
        if self.total_steps is None:
            return control
        step = int(state.global_step)
        tau = self._compute_tau(step)
        model = kwargs.get("model", None)
        if model is not None:
            _set_all_gate_tau(model, tau)
        return control

    
def print_trainable_params(m):
    trainables = 0
    total = 0
    for n, p in m.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainables += p.numel()
    rank0_print(f"[DEBUG] trainable params: {trainables:,} / {total:,} "
          f"({100.0*trainables/total:.4f}%)")    

# def _answers_file_path(base_dir: str, eval_idx: int, eval_name: str) -> str:
#     # base_dir 通常是 <output_dir>/<task_id>
#     return os.path.join(base_dir, "predictions", f"{eval_idx}-{eval_name}.jsonl")

def train(attn_implementation=None):
    global local_rank
    replace_llama_attn_with_flash_attn()

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, TestingArguments))
    print("参数加载完毕")
    model_args, data_args, training_args, testing_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # ========= 新增：补推理模式开关（不改 dataclass，走环境变量） =========
    infer_only = os.getenv("INFER_ONLY", "0") == "1"
    infer_only_max_one = os.getenv("INFER_ONLY_MAX_ONE", "0") == "1"

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # 2) bitsandbytes 量化
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type
            )
        ))

    # 13) 任务列表
    tasks = [task.strip() for task in data_args.tasks.split(',') if len(task.strip()) > 0]


    # ========= 新增：训练完成判定（不依赖 predictions） =========
    def _has_any_checkpoint_files(task_dir: str) -> bool:
        """
        只用于判断“该 task_id 是否已经训练产出可用于推理的权重”。
        兼容：
          - 非 moelora：权重可能在 task_dir 根目录（你 move adapter_dir 到根目录了）
          - moelora：权重可能在 task_dir/task_k/ 下
        """
        if not os.path.isdir(task_dir):
            return False

        # 常见文件名兜底（按你保存逻辑）
        candidates_root = [
            "adapter_model.safetensors",
            "adapter_model.bin",
            "non_lora_trainables.bin",
            "pytorch_model.bin",
            "model.safetensors",
            "config.json",
        ]
        for fn in candidates_root:
            if os.path.isfile(os.path.join(task_dir, fn)):
                return True

        # moelora 情况：task_dir/task_k/adapter_model.*
        base = os.path.basename(task_dir)
        if base.isdigit():
            k = int(base)
            sub = os.path.join(task_dir, f"task_{k}")
            for fn in ["adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"]:
                if os.path.isfile(os.path.join(sub, fn)):
                    return True

        # 兜底：目录里有 checkpoint-* 也算训练产出
        if list(pathlib.Path(task_dir).glob("checkpoint-*")):
            return True

        return False

    # ========= 新增：推理完成判定 =========
    def _answers_file(output_dir_for_model: str, eval_task_id: int, eval_name: str) -> str:
        return os.path.join(output_dir_for_model, "predictions", f"{eval_task_id}-{eval_name}.jsonl")

    def _is_infer_done(output_dir_for_model: str, model_task_id: int, eval_task_id: int, eval_name: str) -> bool:
        """
        单个 (model_task_id -> eval_task_id) 的推理是否完成：answers_file 存在且非空。
        """
        af = _answers_file(output_dir_for_model, eval_task_id, eval_name)
        return os.path.isfile(af) and os.path.getsize(af) > 0

    def _is_last_model_all_infer_done(output_dir_for_last_model: str, expected_n: int) -> bool:
        """
        对“最后一个模型”的整体完成判定：predictions/ 直属 jsonl 文件数量 == expected_n
        （子目录不算）
        """
        pred_dir = os.path.join(output_dir_for_last_model, "predictions")
        if not os.path.isdir(pred_dir):
            return False
        files = [
            f for f in os.listdir(pred_dir)
            if f.endswith(".jsonl") and os.path.isfile(os.path.join(pred_dir, f))
        ]
        return len(files) == expected_n

    # ========= 改造：扫描已训练 task（用于推理），不再用 predictions =========
    trained_task_ids = []
    for d in glob.glob(os.path.join(training_args.output_dir, "*")):
        if os.path.isdir(d) and os.path.basename(d).isdigit():
            if _has_any_checkpoint_files(d):
                trained_task_ids.appensd(int(os.path.basename(d)))

    trained_task_ids = sorted(set(trained_task_ids))
    max_trained_task_id = max(trained_task_ids) if len(trained_task_ids) > 0 else 0

    # 保持你原逻辑：为加载旧 LoRA 准备 continue_lora_path（但在 infer_only 下只是用于兼容 load_* 逻辑）
    training_args.continue_lora_path = []
    if max_trained_task_id > 0 and training_args.cl != "eproj":
        for completed_task_id in range(1, max_trained_task_id + 1):
            training_args.continue_lora_path.append(os.path.join(training_args.output_dir, str(completed_task_id)))
        training_args.continue_lora_enable = True

    # 4) 加载模型
    if model_args.vision_tower is not None:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            temperature=1.0,
            top_p=1.0,
            **bnb_model_from_pretrained_args
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # 8) 注入/加载 LoRA 或 MoE-LoRA
    if training_args.cl in ["moelora", "qsubspace_moelora"]:
        model = load_moe_lora(model, training_args)
    else:
        model = load_lora_parameter(model, training_args)

    # 9) tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # 10) pad_token 等
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # 11) 多模态模块初始化
    if model_args.vision_tower is not None:
        if training_args.cl == "eproj":
            model_args.mm_vision_select_feature = "cls_patch"

        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
            if hasattr(model.get_model(), "e_mm_projector"):
                for p in model.get_model().e_mm_projector.parameters():
                    p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
            if hasattr(model.get_model(), "e_mm_projector"):
                model.get_model().e_mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # 12) bits 模式的 dtype 修正
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # data_modules 仍然构建（不破坏你整体流程）；推理只实际用到 tasks/question_file_dict 等
    data_modules = make_cl_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        task_list=data_args.tasks,
        rank=local_rank
    )

    deepspeed.init_distributed()

    # CL learner 初始化（infer_only 下基本不使用，但保留不破坏结构）
    if training_args.cl == "tir+ewc":
        task_encoder = TaskEncoder(model.get_model().get_vision_tower(),
                                   training_args.local_rank,
                                   training_args.output_dir)
        cl_learner = TIREWCLearner(task_encoder,
                                   LLaVATrainer,
                                   training_args.ewc_lbd,
                                   training_args.local_rank,
                                   training_args.output_dir,
                                   training_args)
    elif training_args.cl == "tir+mas":
        task_encoder = TaskEncoder(model.get_model().get_vision_tower(),
                                   training_args.local_rank,
                                   training_args.output_dir)
        cl_learner = TIRMASLearner(task_encoder,
                                   LLaVATrainer,
                                   training_args.ewc_lbd,
                                   training_args.local_rank,
                                   training_args.output_dir,
                                   training_args)
    elif training_args.cl == "ewc":
        if training_args.lora_enable:
            cl_learner = EWCLoraLearner(LLaVATrainer,
                                        training_args.ewc_lbd,
                                        training_args.local_rank,
                                        training_args.output_dir,
                                        training_args)
        else:
            cl_learner = EWCLearner(LLaVATrainer,
                                    training_args.ewc_lbd,
                                    training_args.local_rank,
                                    training_args.output_dir,
                                    training_args)
    elif training_args.cl == "mas":
        cl_learner = MASLearner(LLaVATrainer,
                                training_args.ewc_lbd,
                                training_args.local_rank,
                                training_args.output_dir,
                                training_args)
    elif training_args.cl == "lwf":
        cl_learner = LwFLearner(training_args.lwf_lbd,
                                training_args.local_rank,
                                model)
    elif training_args.cl == "eproj":
        cl_learner = EprojLearner(LLaVATrainer,
                                  training_args.eproj_lbd,
                                  training_args.num_experts,
                                  training_args.eproj_key_steps,
                                  training_args.local_rank,
                                  training_args.output_dir,
                                  training_args,
                                  model)
    elif training_args.cl == "olora":
        cl_learner = OLoRALearner(
            lbd=training_args.OLoRA_lbd,
            rank=training_args.local_rank,
            model=model,
            lora_r=training_args.lora_r
        )
    elif training_args.cl == "lota":
        cl_learner = LoTALearner(LLaVATrainer,
                                 training_args.sparsity_ratios,
                                 training_args.pruning_fn,
                                 training_args,
                                 max_trained_task_id,
                                 model)
    elif training_args.cl == "moelora":
        cl_learner = moeloraLearner(
            training_args=training_args,
            start_task_id=max_trained_task_id,
            model=model,
            snapshot_every_n_steps=0,
            snapshot_on_epoch_end=True,
        )
    else:
        cl_learner = BaseCLearner()

    model.to(device=torch.device(f"cuda:{local_rank}"), dtype=torch.bfloat16)

    # acc_matrix 结构保留
    num_initial_tasks = len(list_of_ints(data_args.initial_tasks))
    num_tasks_total = len(data_modules)
    acc_matrix = np.zeros((num_tasks_total + 1, num_initial_tasks + num_tasks_total))
    output_dir_root = training_args.output_dir

    dist.barrier()

    data_modules_list = list(data_modules)

    # ========= 新增：infer-only 时，只跑已训练的 model_task_id =========
    # 若 max_trained_task_id == 0，直接结束
    if max_trained_task_id == 0:
        rank0_print("[INFER_ONLY] No trained task directories found; nothing to infer.")
        return

    for train_idx, data_module in enumerate(data_modules_list):
        model_task_id = train_idx + 1
        model_task_name = tasks[train_idx]

        # infer-only：只处理已经训练产出的模型
        if model_task_id > max_trained_task_id:
            break

        # ----------- 关键改动：不再因为 continue_lora_path 而跳过推理 -----------
        # 原逻辑：if not task_id > len(training_args.continue_lora_path): continue
        # 现在：推理补全必须允许进入

        # 仍然设置当前模型目录（与原流程一致）
        training_args.output_dir = os.path.join(output_dir_root, str(model_task_id))
        os.makedirs(training_args.output_dir, exist_ok=True)

        # infer-only：跳过 before_train / after_train，避免对已训练模型做额外状态修改
        # （如果你确信 after_train 不会动权重，也可以保留；这里按“最安全”处理）
        # cl_learner.before_train(model_task_id, model, tokenizer, **data_module)

        rank0_print(f"[INFER_ONLY={infer_only}] Start inference check for model task {model_task_id}: {model_task_name}")

        # ========= 新增：为当前 model_task_id 构造“应该补的 eval 列表” =========
        num_tasks_all = len(tasks)
        expected_evals = []
        if model_task_id < num_tasks_all:
            # 前 N-1：只测当前任务
            expected_evals.append((model_task_id, tasks[model_task_id - 1]))
        else:
            # 最后一个：测全部任务
            for eval_idx, eval_name in enumerate(tasks, start=1):
                expected_evals.append((eval_idx, eval_name))

        cur_model_infer_path = os.path.join(output_dir_root, f"{model_task_id}")
        os.makedirs(cur_model_infer_path, exist_ok=True)

        # 最后一个模型：如果 predictions/ 直属 jsonl 数量已经齐了，可直接跳过
        if model_task_id == num_tasks_all:
            if _is_last_model_all_infer_done(cur_model_infer_path, expected_n=num_tasks_all):
                rank0_print(f"[INFER] Last model {model_task_id} already has {num_tasks_all} direct jsonl in predictions/, skip all.")
                continue

        # 找缺失项
        missing = []
        for (eval_task_id, eval_name) in expected_evals:
            if not _is_infer_done(cur_model_infer_path, model_task_id, eval_task_id, eval_name):
                missing.append((eval_task_id, eval_name))

        if len(missing) == 0:
            rank0_print(f"[INFER] model {model_task_id}: all expected inference done. skip.")
            continue

        # ========= 推理补全：这里保持你原来的 evaluate 流程 =========
        restore_llama_attn_to_hf()

        # 如果你使用的是 PEFT 多 adapter，并且需要显式切换当前 adapter，可在这里尝试 set_adapter
        # （不确定你 load_* 的内部实现，这里做“安全尝试”，失败不阻断）
        try:
            if hasattr(model, "set_adapter"):
                model.set_adapter(f"task_{model_task_id}")
        except Exception as e:
            if local_rank == 0:
                print(f"[INFER][WARN] set_adapter(task_{model_task_id}) failed: {repr(e)}")

        # 逐个补缺失推理（保序执行，便于你外层 sh 循环稳定推进）
        for (eval_task_id, eval_name) in missing:
            try:
                acc = evaluate(
                    testing_args, tokenizer, model,
                    model_task_id,          # model_task_id
                    eval_task_id,           # task_id (eval task id)
                    eval_name,              # task_val (eval task name)
                    cur_model_infer_path
                )
            except Exception as e:
                print(f"[EVAL][model {model_task_id} -> {eval_name}] failed: {repr(e)}; continue.")
                acc = float("nan")

            if local_rank == 0:
                acc_matrix[model_task_id - 1, eval_task_id - 1] = acc

            # 可选：一次只补 1 个缺失项就退出，配合你的 sh 循环更稳
            if infer_only and infer_only_max_one:
                rank0_print("[INFER_ONLY] Completed one missing inference; exit for next loop round.")
                return

        rank0_print(acc_matrix)
        enable_flash_attn()

    rank0_print(acc_matrix)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
