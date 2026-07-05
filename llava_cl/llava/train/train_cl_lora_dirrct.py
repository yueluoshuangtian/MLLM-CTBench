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

directory_path = "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/new_eval_tool"
sys.path.insert(0, directory_path)
import evaluate_art_long_sentences,evaluate_FOMC,evaluate_math,evaluate_numglue,evaluate_object,evaluate_science,evaluate_medical_long_sentences,evaluate_ocr


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
    "science": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/science.json"
    # "object_relative":"/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark/infer_answers/object/test_infer_reasoning_use_relative.json"
}
annotation_file_dict = {
    "numglue": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/infer_answers/numglue/NumGLUE_test_answer.json",
    "art": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/infer_answers/art/test_infer_answer.json",
    "math": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/infer_answers/math/test_infer_answer.json",
    "fomc": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/infer_answers/fomc/test_infer_answer.json",
    "medical": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/infer_answers/medical/test_infer_answer.json",
    "OCR": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/infer_answers/ocr/test_infer_use_reasoning.json",
    "science": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/infer_answers/science/test_ann_infer_answer.json"
}
#question_iamge
image_folder_dict = {
    "numglue": "",
    "art": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "math": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "fomc": "",
    "medical": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "OCR": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "science": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark"
    # "object_relative":"/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark"
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
    expert_num: Optional[int] = None
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    continue_lora_enable:bool = field(default=False)
    continue_lora_path: list[str] = field(default_factory=list)
    # === 可微 QR 门控超参 ===
    gate_rank: int = 24        # 每专家门控子空间列数 r_g(建议 ≤ lora_r/expert_num)
    gate_tau: float = 1.6    # softmax 温度
    gate_tie: str = "A_T"     # "free" 或 "A_T"
    
    cl: str = "none"
    ewc_lbd: float = 1e8
    lwf_lbd: float = 0.2
    eproj_lbd: float = 1.
    OLoRA_lbd:float = 0.1
    num_experts: int = 9
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

    #moelora
    rank_level_gate: bool = field(default=True)
    gate_top_k: Optional[int] = field(default=None, metadata={"help": "仅 frozen_mix 生效；None=全专家软权重"})
    # === MoE-LoRA 子空间正交正则 ===
    moelora_orth_lambda: float = field(
        default=1e-4,
        metadata={"help": "q-subspace MoE-LoRA 的子空间正交正则系数 μ，用于 L_orth"}
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
import time
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
    rank0_print(f"Testing model:{model_task_id} on {task_val}")
    answers_file = os.path.join(output_dir, "predictions", f"{task_id}-{task_val}.jsonl")
    print(f"这是第{model_task_id}个模型的第{task_id}个任务的推理，任务的名称是:{task_val}")
    evaluation_results_file = os.path.join(output_dir, "evaluations", f"{task_id}-{task_val}.jsonl")

    # === 1) 仅当答案文件不存在或为空时才执行推理 ===
    if (not os.path.isfile(answers_file)) or os.path.getsize(answers_file) == 0:
        acc = 0.0

        model.config.tokenizer_padding_side = "left"
        tokenizer.padding_side = "left"

        image_processor = model.get_vision_tower().image_processor
        num_chunks = int(os.getenv('WORLD_SIZE', '1'))
        chunk_idx = local_rank
        question_file = question_file_dict[task_val]
        image_folder = image_folder_dict[task_val]
        chunks_dir = os.path.join(output_dir, "predictions", f"{task_id}-{task_val}")
        answers_chunk_file = os.path.join(chunks_dir, f"{num_chunks}-{chunk_idx}.jsonl")

        questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
        questions = get_chunk(questions, num_chunks, chunk_idx)
        data_loader = create_data_loader(questions, image_folder, tokenizer, image_processor, model.config, args.conv_mode)

        os.makedirs(os.path.dirname(answers_chunk_file), exist_ok=True)
        ans_file = open(answers_chunk_file, "w")

        if hasattr(model.get_model(), "e_mm_projector"):
            model.get_model().e_mm_projector.init_task_id_retrieve_acc()

        conv = conv_templates[args.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]

        model.gradient_checkpointing_disable()
        model.eval()

        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), miniters=max(1, len(data_loader)//10), disable=(local_rank != 0)):
            input_ids = batch["input_ids"]
            convs = batch["convs"]
            attn_mask = batch["attention_mask"]

            question_ids = batch["question_ids"]
            prompts = batch["prompts"]

            if 'images' in batch:
                image_tensor = batch["images"].to(device='cuda', non_blocking=True, dtype=torch.bfloat16)
            else:
                image_tensor = None

            input_ids = input_ids.to(device='cuda', non_blocking=True)
            attn_mask = attn_mask.to(device='cuda', non_blocking=True)
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            # —— 生成阶段兜底 —— #
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

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)

            ans_id = shortuuid.uuid()
            for idx, prompt, output in zip(question_ids, prompts, outputs):
                ans_file.write(json.dumps({
                    "question_id": idx,
                    "prompt": prompt,
                    "text": output.strip(),
                    "answer_id": ans_id,
                    "model_id": str(task_id),
                    "metadata": {}
                }) + "\n")
            if hasattr(model.get_model(), "e_mm_projector") and (i % 50 == 0 or i == len(questions) - 1):
                print(model.get_model().e_mm_projector.cal_task_id_retrieve_acc())
            ans_file.flush()

        ans_file.close()

        # —— 软同步：用旗标文件协调 —— #
        chunks_dir = os.path.join(output_dir, "predictions", f"{task_id}-{task_val}")
        my_done_flag = os.path.join(chunks_dir, f"done.{chunk_idx}")
        _touch(my_done_flag)  # 本 rank 声明“我的分片已完成”

        if local_rank == 0:
            # rank0：等待 done.*(超时后也继续)，合并已有分片，并写 MERGED.ok
            try:
                _wait_and_merge_chunks_rank0(chunks_dir, answers_file, num_chunks, timeout_s=600)
            except Exception as e:
                print(f"[EVAL][merge] unexpected error during merge on rank0: {repr(e)}; will still write MERGED.ok.")
                _touch(os.path.join(chunks_dir, "MERGED.ok"))
        else:
            # 非 rank0：等待 MERGED.ok(最多等一段时间，避免抢跑)
            ok = _wait_for_flags(chunks_dir, ["MERGED.ok"], timeout_s=600)
            if not ok:
                print(f"[EVAL][rank {local_rank}] MERGED.ok not seen within timeout; proceed anyway.")

    else:
        print(f"推理答案 {answers_file} 已存在或非空，跳过推理。")

    # === rank0 做评测，其他 rank 直接返回(不再 barrier) ===
    if local_rank == 0:
        # 清理分片目录(若存在)
        pred_subdir = os.path.join(output_dir, "predictions", f"{task_id}-{task_val}")
        if os.path.isdir(pred_subdir):
            shutil.rmtree(pred_subdir, ignore_errors=True)

        if task_val == "art":
            acc = evaluate_art_long_sentences.eval_results(annotation_file_dict[task_val], answers_file, evaluation_results_file, is_reasoning=False)
        elif task_val == "numglue":
            acc = evaluate_numglue.eval_results(annotation_file_dict[task_val], answers_file, evaluation_results_file, is_reasoning=False)
        elif task_val == "math":
            acc = evaluate_math.eval_results(annotation_file_dict[task_val], answers_file, evaluation_results_file, is_reasoning=False)
        elif task_val == "fomc":
            acc = evaluate_FOMC.eval_results(annotation_file_dict[task_val], answers_file, evaluation_results_file, is_reasoning=False)
        elif task_val == "medical":
            acc = evaluate_medical_long_sentences.eval_results(annotation_file_dict[task_val], answers_file, evaluation_results_file, is_reasoning=False)
        elif task_val == "OCR":
            acc = evaluate_ocr.eval_results(annotation_file_dict[task_val], answers_file, evaluation_results_file, is_reasoning=False)
        elif task_val == "science":
            acc = evaluate_science.eval_results(annotation_file_dict[task_val], answers_file, evaluation_results_file, is_reasoning=False)
        elif task_val == "object":
            acc = evaluate_object.eval_results(annotation_file_dict[task_val], answers_file, evaluation_results_file)
        else:
            raise ValueError(f"Unknown task : {task_val}")

        model.config.tokenizer_padding_side = "right"
        tokenizer.padding_side = "right"
        print(f"任务 {task_val} 推理结果 {answers_file} 的正确率是 {acc}")
        return acc
    else:
        # 非 rank0：返回占位值，训练主循环里已 try/except 包裹，不影响后续
        model.config.tokenizer_padding_side = "right"
        tokenizer.padding_side = "right"
        return 0.0
    

def load_lora_parameter(model, training_args):
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
    # 1. 计算新LoRA适配器的ID(确保命名唯一，基于已加载的LoRA数量)
    # task_ada_max_id：新适配器的编号(如已加载2个旧LoRA，新适配器为task_3)
    task_ada_max_id = len(training_args.continue_lora_path) + 1

    # 2. 若启用LoRA(training_args.lora_enable=True)，初始化新LoRA适配器
    if training_args.lora_enable:
        new_task_name = f"task_{task_ada_max_id}"  # 新LoRA适配器命名(如task_1)
        # 配置LoRA参数
        lora_config = LoraConfig(
            r=training_args.lora_r,  # LoRA秩(秩越小，参数越少)
            lora_alpha=training_args.lora_alpha,  # LoRA缩放因子(控制LoRA影响强度)
            target_modules=find_all_linear_names(model),  # 目标模块(通常是模型所有线性层，需提前实现find_all_linear_names)
            lora_dropout=training_args.lora_dropout,  # LoRA层 dropout(防止过拟合)
            bias=training_args.lora_bias,  # 是否训练bias(如"none"不训练，"all"训练所有bias)
            task_type="CAUSAL_LM",  # 任务类型(因果语言模型，适配LLaMA/LLaVA)
        )

        # 根据精度配置调整模型 dtype(16位训练，降低显存占用)
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)  # 若启用bf16，转为bf16精度
            if training_args.fp16:
                model.to(torch.float16)   # 若启用fp16，转为fp16精度

        rank0_print("Adding initial LoRA adapters...")  # 仅主进程打印日志
        # 为模型添加新LoRA适配器(get_peft_model是PEFT库核心函数，包装模型为PeftModel)
        model = get_peft_model(model, lora_config, adapter_name=new_task_name)

    # 3. 若启用“继续加载旧LoRA”(training_args.continue_lora_enable=True)，加载并冻结旧适配器
    if training_args.continue_lora_enable:
        # 遍历所有旧LoRA适配器路径，依次加载(支持多任务LoRA累加)
        if len(training_args.continue_lora_path) > 0:
            rank0_print("Loading multiple existing LoRA adapters...")
            for task_ada_id, adapter_path in enumerate(training_args.continue_lora_path):
                task_ada_id += 1  # 适配器编号从1开始(如第一个旧适配器为task_1)
                adapter_name = f"task_{task_ada_id}"  # 旧适配器命名(与保存时一致)
                rank0_print(f"Loading LoRA adapter from {adapter_path}...")
                # 加载旧LoRA适配器到模型(PeftModel的load_adapter方法)
                model.load_adapter(adapter_path, adapter_name)

        # 筛选所有“旧LoRA适配器”的参数名(lora_A/lora_B是LoRA核心参数，排除新适配器)
        old_lora_names = {
            name for name, _ in model.named_parameters()
            if ("lora_A" in name or "lora_B" in name)  # 仅匹配LoRA参数
            and (f"task_{task_ada_max_id}" not in name)  # 排除新适配器参数
        }

        # 冻结旧LoRA适配器参数(禁止反向传播更新，避免遗忘旧任务)
        for name, param in model.named_parameters():
            if name in old_lora_names:
                param.requires_grad = False  # 冻结旧LoRA，仅新LoRA可训练
        rank0_print("Frozen all loaded LoRA adapters. Only new LoRA will be trained.")

        # 激活新LoRA适配器(指定模型当前使用的适配器为新任务适配器)
        model.set_adapter(new_task_name)
    else:
        # 若未启用LoRA，打印提示信息
        rank0_print("LoRA is not enabled. Skipping LoRA adapter addition.")
    
    return model  # 返回添加/加载LoRA后的模型


def load_moe_lora(model, training_args):
    """
    注入当前任务的 CoIN-MoE-LoRA 适配器；训练时不加载历史适配器。
    cl == "moelora" 时使用。
    """
    try:
        from CoIN.peft import get_peft_model, TaskType
        from CoIN.peft.tuners.coinmoelora import CoINMOELoraConfig, CoINMOELoraLinear
    except Exception:
        from peft import get_peft_model
        try:
            from peft import TaskType
        except Exception:
            class _TaskType:
                CAUSAL_LM_CoIN = "CAUSAL_LM_CoIN"
            TaskType = _TaskType()
        from peft.tuners.coinmoelora import CoINMOELoraConfig, CoINMOELoraLinear

    _rank0_print = globals().get("rank0_print", print)

    if not getattr(training_args, "lora_enable", False):
        _rank0_print("[MoE-LoRA] lora_enable=False，跳过注入。")
        return model

    # === 1) 新适配器名：仍然基于 continue_lora_path 计数，保证目录名 task_k 对齐 ===
    prev_cnt = len(getattr(training_args, "continue_lora_path", []))
    cur_adapter_name = f"task_{prev_cnt + 1}"

    # === 2) MoE-LoRA 配置 ===
    cfg = CoINMOELoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type=TaskType.CAUSAL_LM_CoIN,
        expert_num=int(getattr(training_args, "expert_num", 4)),
        gate_rank=int(getattr(training_args, "gate_rank", 4)),
        gate_tau=float(getattr(training_args, "gate_tau", 0.7)),
        gate_tie=str(getattr(training_args, "gate_tie", "free")),
        # ★ 强制启用逐秩 MoE(Q^T x top-K 硬路由)
        rank_level_gate=True,
        gate_top_k=int(getattr(training_args, "gate_top_k", 64)),
    )

    # === 3) 精度设置 ===
    if getattr(training_args, "bits", 0) == 16:
        if getattr(training_args, "bf16", False):
            model.to(torch.bfloat16)
        elif getattr(training_args, "fp16", False):
            model.to(torch.float16)

    # === 4) 注入当前适配器 ===
    model = get_peft_model(model, cfg, adapter_name=cur_adapter_name)
    try:
        model.set_adapter(cur_adapter_name)
    except Exception:
        pass

    _rank0_print("[MoE-LoRA] 训练期不加载历史适配器；仅注入当前适配器。")

    # === 5) 冻结非当前适配器的 LoRA / gate 参数(防御性，正常情况下不存在旧适配器)=== 
    for n, p in model.named_parameters():
        is_lora_param = ("lora_A" in n) or ("lora_B" in n) or ("gate_P" in n) or ("gate_P_map" in n)
        if is_lora_param and (cur_adapter_name not in n):
            p.requires_grad = False

    # === 6) 把 gate_top_k 从 TrainingArguments 传给每个 CoINMOELoraLinear ===
    gate_top_k = getattr(training_args, "gate_top_k", None)

    try:
        for mod in model.modules():
            if isinstance(mod, CoINMOELoraLinear):
                # 训练时只走“当前适配器”的 MoE
                setattr(mod, "train_forward_mode", "single")
                setattr(mod, "train_mix_adapters", None)
                setattr(mod, "gate_top_k", gate_top_k)
                setattr(mod, "multi_active_adapters", None)
                setattr(mod, "multi_top_k", None)
                setattr(mod, "active_adapter", cur_adapter_name)
    except Exception:
        pass

    model.train()
    _rank0_print(
        f"[MoE-LoRA] 已注入新适配器：{cur_adapter_name}(cl=moelora)；训练阶段不加载旧适配器。"
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


# === 新增：每个任务单独初始化 LoRA 适配器 ===
def init_single_task_lora(model, training_args, task_id: int):
    """
    为当前任务 task_id 初始化“专用 LoRA 适配器”，实现：
    - 每个任务使用独立的 adapter: adapter_name = f"task_{task_id}"
    - 冻结其它任务的 LoRA 参数(不再继续更新旧任务 LoRA)
    - 只训练当前任务对应的 LoRA
    """
    from peft import LoraConfig, get_peft_model
    try:
        from peft import PeftModel
    except Exception:
        PeftModel = None

    adapter_name = f"task_{task_id}"

    # LoRA 配置（沿用你的配置）
    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )

    # 精度配置
    if training_args.bits == 16:
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)

    # 1）若当前还不是 PeftModel，第一次注入 LoRA
    if PeftModel is not None and isinstance(model, PeftModel):
        # 已经是 PeftModel：如果当前适配器不存在，就 add_adapter
        peft_cfg = getattr(model, "peft_config", {})
        if adapter_name not in peft_cfg:
            rank0_print(f"[LoRA] add_adapter for new task adapter: {adapter_name}")
            model.add_adapter(adapter_name, lora_config)
    else:
        # 不是 PeftModel：第一次注入 LoRA
        rank0_print(f"[LoRA] get_peft_model for first adapter: {adapter_name}")
        model = get_peft_model(model, lora_config, adapter_name=adapter_name)

    # 2）冻结所有 LoRA 参数，只放开当前任务的 LoRA
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            if adapter_name in name:
                param.requires_grad = True   # 当前任务的 LoRA 可训练
            else:
                param.requires_grad = False  # 其它任务 LoRA 冻结

    # 3）激活当前任务适配器
    try:
        model.set_adapter(adapter_name)
    except Exception:
        pass

    rank0_print(f"[LoRA] 已初始化当前任务专用 LoRA 适配器: {adapter_name}")
    return model


def train(attn_implementation=None):
    global local_rank  # 全局变量：分布式训练的本地进程ID(如GPU编号)
    replace_llama_attn_with_flash_attn()
    # 1. 解析命令行参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, TestingArguments))
    print("参数加载完毕")
    model_args, data_args, training_args, testing_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank  # 初始化本地进程ID
    
    # 确定计算精度
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # 2. 配置4/8位量化参数(BitsAndBytes)
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

    # 3. 检查已完成的任务(用于断点续训，避免重复训练)
    start_task_id = 0  # 初始任务ID(从1开始)
    
    # 遍历输出目录，找到所有以数字命名的任务目录(如1/2/3，代表已完成的任务1-3)
    for d in glob.glob(os.path.join(training_args.output_dir, "*")): 
        if os.path.isdir(d) and os.path.basename(d).isdigit() and len(os.listdir(d)) > 0:
            start_task_id = max(int(os.path.basename(d)), start_task_id)

    # === 重要修改：不再使用 continue_lora_path 做历史 LoRA 继承 ===
    # 对“每个任务单独 LoRA 微调”来说，不需要加载以前任务的 LoRA
    training_args.continue_lora_path = []
    training_args.continue_lora_enable = False

    # 4. 加载模型(区分多模态LLaVA和纯语言Llama)
    if model_args.vision_tower is not None:
        # LLaVA
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            temperature=1.0,
            top_p=1.0,
            **bnb_model_from_pretrained_args
        )
    else:
        # LLaMA
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )

    model.config.use_cache = False

    # 5. 冻结backbone(仅训练LoRA或投影层，可选)
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    # 6. 为量化模型准备训练
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # 7. 启用梯度检查点
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # 8. 调用LoRA管理函数：
    #    - MoE-LoRA 仍然使用原来的逻辑
    #    - 普通 LoRA 的初始化移到“任务循环内部”做 per-task LoRA
    if training_args.cl in ["moelora", "qsubspace_moelora"]:
        model = load_moe_lora(model, training_args)

    # 9. 加载Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # 10. 配置Tokenizer pad_token
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

    # 11. 多模态模型专属配置
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

    # 12. 量化模型中，调整LoRA层和特殊层的精度
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

    # 13. 准备任务列表和数据模块
    tasks = [task.strip() for task in data_args.tasks.split(',') if len(task.strip()) > 0]
    data_modules = make_cl_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        task_list=data_args.tasks,
        rank=local_rank
    )

    # 14. 初始化分布式训练
    deepspeed.init_distributed()

    # 15. 初始化持续学习策略
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
    elif training_args.cl == "lota":
        cl_learner = LoTALearner(LLaVATrainer,
                                 training_args.sparsity_ratios,
                                 training_args.pruning_fn,
                                 training_args,
                                 start_task_id,
                                 model
        )
    elif training_args.cl == "moelora":
        cl_learner = moeloraLearner(
            training_args=training_args,
            start_task_id=start_task_id,
            model=model,
            snapshot_every_n_steps=0,
            snapshot_on_epoch_end=True,
        )
    else:
        cl_learner = BaseCLearner()

    # 16. 将模型移动到指定GPU和精度
    model.to(device=torch.device(f"cuda:{local_rank}"), dtype=torch.bfloat16)

    # 17. 初始化准确率矩阵
    num_initial_tasks = len(list_of_ints(data_args.initial_tasks))
    num_tasks = len(data_modules)
    acc_matrix = np.zeros((num_tasks + 1, num_initial_tasks + num_tasks))
    output_dir = training_args.output_dir

    dist.barrier()

    # 18. 持续学习核心循环
    data_modules_list = list(data_modules)

    for train_idx, data_module in enumerate(data_modules_list):
        task_id = train_idx + 1
        task_val = tasks[train_idx]

        # === 基于 start_task_id 跳过已完成任务 ===
        if task_id <= start_task_id:
            rank0_print(f"[Train] 任务 {task_id} 已存在输出目录，视为已完成，跳过训练。")
            continue

        # === 每个任务：单独初始化当前任务的 LoRA 适配器（普通 LoRA 情况） ===
        if training_args.lora_enable and training_args.cl not in ["moelora", "qsubspace_moelora"]:
            model = init_single_task_lora(model, training_args, task_id)

        # 18.1 训练前准备
        cl_learner.before_train(task_id, model, tokenizer, **data_module)

        # 18.2 设置当前任务的输出目录
        training_args.output_dir = os.path.join(output_dir, str(task_id))
        os.makedirs(training_args.output_dir, exist_ok=True)
        task = list_of_ints(data_args.tasks)[train_idx]

        # 18.3 训练当前任务
        rank0_print(f"Start training on task {task_id}: {task}")
        release_memory()
        print_trainable_params(model)

        if training_args.cl == "lwf":
            if task_id == 1:
                rank0_print(f"对于lwf的第一个任务{task},使用正常的loss")
                is_first_iteration = True
            else:
                rank0_print(f"对于lwf的第{task_id}个任务{task},使用lwf的loss")
                is_first_iteration = False

            trainer = LLaVATrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                cl_learner=cl_learner,
                is_first_iteration=is_first_iteration,
                **data_module
            )
        else:
            trainer = LLaVATrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                cl_learner=cl_learner,
                **data_module
            )

        if training_args.cl == "moelora":
            trainer.add_callback(GateTauSchedulerCallback(
                start_tau=1.8,
                end_tau=0.9,
                mode="cosine",
                start_step=0,
                end_step=int(0.7 * trainer.state.max_steps) if trainer.state.max_steps else None,
                log_every_n_steps=100,
            ))            

        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            attach_debug_probes(
                trainer,
                training_args,
                n_timer_steps=5,
                enable_tiny_profiler=True
            )
            trainer.train()

        trainer.save_state()
        model.config.use_cache = True

        if not training_args.cl == "moelora":
            # 18.4 保存当前任务的 LoRA 适配器
            if training_args.lora_enable:
                state_dict = get_peft_state_maybe_zero_3(
                    model.named_parameters(), training_args.lora_bias, f"task_{task_id}"
                )
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                    model.named_parameters()
                )

                if training_args.local_rank == 0 or training_args.local_rank == -1:
                    model.config.save_pretrained(training_args.output_dir)
                    model.save_pretrained(training_args.output_dir, state_dict=state_dict)
                    torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))

                    adapter_dir = os.path.join(training_args.output_dir, f"task_{task_id}")
                    target_dir = training_args.output_dir
                    if os.path.exists(adapter_dir):
                        for filename in os.listdir(adapter_dir):
                            src_file = os.path.join(adapter_dir, filename)
                            dst_file = os.path.join(target_dir, filename)
                            if os.path.exists(dst_file):
                                print(f"Warning: {dst_file} already exists. Skipping...")
                            else:
                                shutil.move(src_file, dst_file)
                        os.rmdir(adapter_dir)
                        print(f"Moved all files from {adapter_dir} to {target_dir} and deleted {adapter_dir}.")
                    else:
                        print(f"Warning: Adapter directory {adapter_dir} does not exist.")
            else:
                safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        else:
            # 18.4 moelora 保存逻辑(原样保留)
            if training_args.lora_enable:
                cur_adapter = f"task_{task_id}"

                state_dict = get_peft_state_maybe_zero_3(
                    model.named_parameters(), training_args.lora_bias, cur_adapter
                )
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                    model.named_parameters()
                )

                if training_args.local_rank == 0 or training_args.local_rank == -1:
                    model.config.save_pretrained(training_args.output_dir)

                    adapter_dir = os.path.join(training_args.output_dir, cur_adapter)
                    os.makedirs(adapter_dir, exist_ok=True)

                    try:
                        peft_cfg = getattr(model, "peft_config", None)
                        if peft_cfg is not None and cur_adapter in peft_cfg:
                            peft_cfg[cur_adapter].save_pretrained(adapter_dir)
                        else:
                            import json
                            with open(os.path.join(adapter_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
                                json.dump({"adapter_name": cur_adapter, "peft_type": "LORA"}, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass

                    try:
                        from safetensors.torch import save_file as safe_save
                        safe_save(state_dict, os.path.join(adapter_dir, "adapter_model.safetensors"))
                    except Exception:
                        torch.save(state_dict, os.path.join(adapter_dir, "adapter_model.bin"))

                    torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
            else:
                safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

        del trainer
        release_memory()

        # 18.5 训练后处理
        cl_learner.after_train(task_id, model, data_modules_list, **data_module)
        rank0_print(f"Saving model for task {task_id} to: {training_args.output_dir}")
        rank0_print(f"Start inference after task {task_id}: {task}")

        # 评估当前及之前的全部任务（这里仍然用“当前任务的 LoRA”评估多任务表现）
        cur_model_infer_path = os.path.join(output_dir, f"{task_id}")
        os.makedirs(cur_model_infer_path, exist_ok=True)
        restore_llama_attn_to_hf() 
        for eval_idx, eval_name in enumerate(tasks, start=1):
            if eval_idx == task_id:
                try:
                    acc = evaluate(
                        testing_args, tokenizer, model,
                        task_id,          # model_task_id == 当前任务的 LoRA
                        eval_idx,         # 被评估任务序号
                        eval_name,        # 被评估任务名称
                        cur_model_infer_path
                    )
                    if local_rank == 0:
                        acc_matrix[task_id - 1, eval_idx - 1] = acc
                except Exception as e:
                    print(f"[EVAL][task {task_id} -> {eval_name}] failed with error: {repr(e)}; continue to next eval.")
                    if local_rank == 0:
                        acc_matrix[task_id - 1, eval_idx - 1] = float("nan")
                    continue

        rank0_print(acc_matrix)
        enable_flash_attn()

    rank0_print(acc_matrix)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
