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
from datetime import timedelta
import os

# 强制把 torch.utils.checkpoint.checkpoint 改成 non-reentrant.
# 否则 LLaMA 内部 (transformers 4.31 modeling_llama.py:685) 调 checkpoint 默认 use_reentrant=True,
# backward 时嵌套发起一次独立的 autograd.backward, 让 ZeRO-2 的 reduce hook 在同一参数上 fire 两次
# 撞 "The parameter X has already been reduced" assert. 详见 EWC_MAS_ZeRO2_fix.md 方案 1.
import torch.utils.checkpoint as _ckpt_mod
_original_checkpoint = _ckpt_mod.checkpoint
def _checkpoint_non_reentrant(*args, **kwargs):
    if "use_reentrant" not in kwargs:
        kwargs["use_reentrant"] = False
    return _original_checkpoint(*args, **kwargs)
_ckpt_mod.checkpoint = _checkpoint_non_reentrant

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
import datetime
import transformers
import tokenizers
from cl_learner.eproj import EprojLearner
from llava_trainer import LLaVATrainer
from llava.data.dataset import DataArguments, make_cl_data_module, smart_tokenizer_and_embedding_resize
from llava.eval.model_vqa_loader import create_data_loader, get_chunk, KeywordsStoppingCriteria
from llava.conversation import conv_templates, SeparatorStyle
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn, restore_llama_attn_to_hf, original_llama_attn_ctx, enable_flash_attn

from llava import conversation as conversation_lib
from llava.model import *

from cl_learner import *
import pdb
import deepspeed
import sys
import os

directory_path = "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/new_eval_tool"
# 注: new_eval_tool 是简化版评分模块, 提供 loose substring 匹配。
# 论文/原作者评分脚本仍需自行替换以获得严格分数
sys.path.insert(0, directory_path)
import evaluate_art_long_sentences,evaluate_FOMC,evaluate_math,evaluate_numglue,evaluate_object,evaluate_science,evaluate_medical_long_sentences,evaluate_ocr

local_rank = None
#question_word
question_file_dict = {
    "numglue": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/llava_eval/numglue.jsonl",
    "art":     "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/llava_eval/art.jsonl",
    "math":    "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/llava_eval/math.jsonl",
    "fomc":    "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/llava_eval/fomc.jsonl",
    "medical": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/llava_eval/medical.jsonl",
    "OCR":     "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/llava_eval/OCR.jsonl",
    "science": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/llava_eval/science.jsonl",
}
annotation_file_dict = {
    "numglue": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/numglue.json",
    "art":     "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/art.json",
    "math":    "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/math.json",
    "fomc":    "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/fomc.json",
    "medical": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/medical.json",
    "OCR":     "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/OCR.json",
    "science": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/science.json"
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


def deep_clean_model_state(model):
    """
    在两个任务之间清理梯度并回收显存. 不动 deepspeed 的内部属性
    (ds_active_sub_modules / ds_id 等), 那些是 deepspeed.initialize 会用的.
    """
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
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
    eval_max_samples: int = 0  # >0 时截断推理 questions, 用于 smoke test
    

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
    expert_use_num: Optional[int] = None
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

    cl: str = "none"
    # >0 时: 只训练+评估第 N 个任务 (1-indexed) 然后退出, 用于子进程逐任务模式
    # (规避 in-process 跨任务 reload 显存泄漏). 由 run_method_subproc.sh 驱动.
    only_task_id: int = 0
    ewc_lbd: float = 1e8
    lwf_lbd: float = 0.2
    eproj_lbd: float = 1.
    num_experts: int = 9
    eproj_key_lr: Optional[float] = 1e-2
    eproj_key_steps: Optional[int] = 2000
    
    l2p_pool_size : Optional[int] = 512
    l2p_prompt_length : Optional[int] = 4
    top_k : Optional[int] = 4
    
    pruning_fn: Optional[List[str]] = field(default_factory=lambda: ['mm_projector', 'vision_tower', 'vision_resampler'])
    sparsity_ratios: Optional[List[float]] = field(default_factory=lambda: [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    experts_sparsity_ratios: Optional[float] = 0.1



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
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
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
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    if trainer.deepspeed:
        # 确保DeepSpeed收集所有进程的权重
        trainer.save_model(output_dir)
        # 手动保存tokenizer（DeepSpeed可能漏保存）
        if trainer.args.should_save:
            trainer.tokenizer.save_pretrained(output_dir)
        # 保存视觉投影层（若存在）
        if getattr(trainer.args, "tune_mm_mlp_adapter", False):
            mm_state = {k: v.cpu() for k, v in trainer.model.named_parameters() if "mm_projector" in k}
            torch.save(mm_state, os.path.join(output_dir, "mm_projector.bin"))
    else:
        # 非DeepSpeed场景正常保存
        trainer.save_model(output_dir)
        if trainer.args.should_save:
            trainer.tokenizer.save_pretrained(output_dir)



def evaluate(args, tokenizer, model, model_task_id,task_id, task_val, output_dir):

   
    #此处task_id是任务在序列中顺序+1，task_val是任务的名称
    rank0_print(f"Testing model:{model_task_id}on {task_val}")
    answers_file = os.path.join(output_dir, "predictions", f"{task_id}-{task_val}.jsonl")
    print(f"这是第{model_task_id}个模型的第{task_id}个任务的推理，任务的名称是:{task_val}")
    evaluation_results_file = os.path.join(output_dir, "evaluations", f"{task_id}-{task_val}.jsonl")
    if (not os.path.isfile(answers_file)) or os.path.getsize(answers_file) > 0:

        acc = 0.

        if not os.path.isfile(answers_file):
            model.config.tokenizer_padding_side = "left"
            tokenizer.padding_side = "left"

            image_processor = model.get_vision_tower().image_processor
            num_chunks = int(os.getenv('WORLD_SIZE', '1'))
            chunk_idx = local_rank
            question_file = question_file_dict[task_val] #此处question_file_dict字典中的，应与initial_tasks中的任务对应
            image_folder = image_folder_dict[task_val]
            answers_chunk_file = os.path.join(output_dir, "predictions", f"{task_id}-{task_val}",
                                                f"{num_chunks}-{chunk_idx}.jsonl")
            questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r", encoding="utf-8")]
            if getattr(args, "eval_max_samples", 0) and args.eval_max_samples > 0:
                questions = questions[:args.eval_max_samples]
            questions = get_chunk(questions, num_chunks, chunk_idx)
            data_loader = create_data_loader(questions, image_folder, tokenizer, image_processor, model.config,
                                                args.conv_mode)
            answers_chunk_file = os.path.expanduser(answers_chunk_file)
            os.makedirs(os.path.dirname(answers_chunk_file), exist_ok=True)
            ans_file = open(answers_chunk_file, "w", encoding="utf-8")

            if hasattr(model.get_model(), "e_mm_projector"):
                model.get_model().e_mm_projector.init_task_id_retrieve_acc()

            conv = conv_templates[args.conv_mode].copy()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]

            model.gradient_checkpointing_disable()
            model.eval()
            for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), miniters=len(data_loader)//10, disable=(local_rank != 0)):
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

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
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

                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)

                ans_id = shortuuid.uuid()
                for idx, prompt, output in zip(question_ids, prompts, outputs):
                    ans_file.write(json.dumps({"question_id": idx,
                                                "prompt": prompt,
                                                "text": output.strip(),
                                                "answer_id": ans_id,
                                                "model_id": str(task_id),
                                                "metadata": {}}) + "\n")

                if hasattr(model.get_model(), "e_mm_projector") and (i % 50 == 0 or i == len(questions) - 1):
                    print(model.get_model().e_mm_projector.cal_task_id_retrieve_acc())

                ans_file.flush()
            ans_file.close()
            dist.barrier()  # wait for all chunks are completed
            
            if local_rank == 0:
                ans_file = open(answers_file, "w", encoding="utf-8")
                for i in range(num_chunks):
                    for q in open(
                            os.path.join(output_dir, "predictions", f"{task_id}-{task_val}", f"{num_chunks}-{i}.jsonl"),
                            encoding="utf-8"):
                        ans_file.write(q)
                        ans_file.flush()
                ans_file.close()


    else:
        print( f"推理答案{answers_file}已存在")

    if local_rank == 0:
        if os.path.isdir(os.path.join(output_dir, "predictions", f"{task_id}-{task_val}")):
            shutil.rmtree(os.path.join(output_dir, "predictions", f"{task_id}-{task_val}"))
        if task_val == "art":
            
            acc = evaluate_art_long_sentences.eval_results(annotation_file_dict[task_val], answers_file,evaluation_results_file,is_reasoning=True)
        elif task_val == "numglue":
            
            acc = evaluate_numglue.eval_results(annotation_file_dict[task_val], answers_file,evaluation_results_file,is_reasoning=True)
        elif task_val == "math":
            acc = evaluate_math.eval_results(annotation_file_dict[task_val], answers_file,evaluation_results_file,is_reasoning=True)
        elif task_val == "fomc":
            acc = evaluate_FOMC.eval_results(annotation_file_dict[task_val], answers_file,evaluation_results_file,is_reasoning=True)
        elif task_val == "medical":
            acc = evaluate_medical_long_sentences.eval_results(annotation_file_dict[task_val], answers_file,evaluation_results_file,is_reasoning=True)
        elif task_val == "OCR":
            acc = evaluate_ocr.eval_results(annotation_file_dict[task_val], answers_file,evaluation_results_file,is_reasoning=True)
        elif task_val == "science":
            acc = evaluate_science.eval_results(annotation_file_dict[task_val], answers_file,evaluation_results_file,is_reasoning=True)
        elif task_val == "object":
            acc = evaluate_object.eval_results(annotation_file_dict[task_val], answers_file,evaluation_results_file)
        else:
            raise ValueError(f"Unknown task : {task_val}")

    dist.barrier()

    model.config.tokenizer_padding_side = "right"
    tokenizer.padding_side = "right"
    print( f"任务{task_val}推理结果{answers_file}的正确率是{acc}")

    return acc


def train(attn_implementation=None):
    global local_rank
    replace_llama_attn_with_flash_attn()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, TestingArguments))
    print("参数加载完毕")
    model_args, data_args, training_args, testing_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    if training_args.cl == 'replay_fixed':
        data_args.non_duplicates = False
    if training_args.cl == 'replay_fixed_same':
        data_args.non_duplicates = False
        data_args.same_data_range = True
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
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    # resume from task，前面的checkpoint可以删掉
    start_task_id = 0
    for d in glob.glob(os.path.join(training_args.output_dir, "*")): 
        if os.path.isdir(d) and os.path.basename(d).isdigit() and len(os.listdir(d)) > 0:
            start_task_id = max(int(os.path.basename(d)), start_task_id)
    if start_task_id > 0 and training_args.cl != "eproj":
        model_args.model_name_or_path = os.path.join(training_args.output_dir, str(start_task_id))
    rank0_print(f"-----------------------------------------任务{start_task_id}-------------------------------")

    if model_args.vision_tower is not None:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            # attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            temperature=1.0,    # unset temperature and top_p to be incompatible with do_sample=False
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
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    if training_args.lora_enable:
        if training_args.expert_num == None:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )

        else:
            from CoIN.peft import PeftModel, TaskType, get_peft_model, CoINMOELoraConfig, WEIGHTS_NAME, set_peft_model_state_dict
            
            kwargs = { 
                # "task_embedding_dim": training_args.task_embedding_dim,
                "expert_num": training_args.expert_num,
                'expert_use_num':training_args.expert_use_num
            }
            lora_config = CoINMOELoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type=TaskType.CAUSAL_LM_CoIN,
                **kwargs
            )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

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
    tasks = [task.strip() for task in data_args.tasks.split(',') if len(task.strip()) > 0]
    #一个包含不同字典（字典由不同任务的数据和函数组成）
    data_modules = make_cl_data_module(tokenizer=tokenizer,
                                       data_args=data_args,
                                       task_list=data_args.tasks,
                                       rank=local_rank)
    
    # initial_data_modules = make_cl_data_module(tokenizer=tokenizer,
    #                                            data_args=data_args,
    #                                            task_list=data_args.initial_tasks,
    #                                            rank=local_rank)
    
    deepspeed.init_distributed()

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
    elif training_args.cl == "der":
        cl_learner = DERLearner(LLaVATrainer,
                                training_args.lwf_lbd,
                                training_args.local_rank,
                                training_args.output_dir,
                                training_args)
    elif training_args.cl == "freeze":
        # 论文 Freeze baseline: 冻结 backbone，只训练 mm_projector
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        cl_learner = BaseCLearner()
    elif training_args.cl == "max_merge":
        # 每任务独立训练 → 最后合并 (after-all-tasks 后处理)
        from cl_learner.max_merge import MaxMergeLearner
        cl_learner = MaxMergeLearner(training_args.local_rank,
                                     training_args.output_dir)
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
        cl_learner =LoTALearner(LLaVATrainer,
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
            prune_ratio=training_args.experts_sparsity_ratios  
        )
    elif training_args.cl == "l2p":
        cl_learner = L2PLearner(
            training_args=training_args,
            text_num_prompts=64,
            text_prompt_len=16,
            vision_num_prompts=64,
            vision_prompt_len=16,
            key_dim=256,
            text_top_k=4,
            vision_top_k=4,
            selection_mode="semantic",  # or "attn"
            pool_type="max",           # or "max"
            reg_lambda=0.2,             # 可调
        )   
        # 绑定模型，便于确定保存/加载的 device/dtype
        cl_learner.bind_model(model) 
        # 尝试加载历史任务的 L2P 池（断点/续训/start_task_id>0）
        cl_learner.load_existing_pools(training_args.output_dir)
    else:
        cl_learner = BaseCLearner()
    
    model.to(device=torch.device(f"cuda:{local_rank}"), dtype=torch.bfloat16)

    num_initial_tasks = len(list_of_ints(data_args.initial_tasks))
    num_tasks = len(data_modules)
    acc_matrix = np.zeros((num_tasks + 1, num_initial_tasks + num_tasks))
    output_dir = training_args.output_dir

    dist.barrier()

    only_task_id = int(getattr(training_args, "only_task_id", 0) or 0)

    for task_id, data_module in enumerate(data_modules):

        task_val = tasks[task_id]
        task_id += 1

        # 子进程逐任务模式: 只处理第 only_task_id 个任务, 其余跳过 (不训练/不评估,
        # 避免对已完成任务用错误模型重复评估). 模型由 start_task_id 逻辑从上一任务 checkpoint 载入.
        if only_task_id > 0 and task_id != only_task_id:
            continue

        before_train_kwargs = dict(data_module)
        if training_args.cl == "der" and task_id >= 2:
            before_train_kwargs["replay_dataset"] = data_modules[task_id - 2]["train_dataset"]

        cl_learner.before_train(task_id, model, tokenizer, **before_train_kwargs)
        
        training_args.output_dir = os.path.join(output_dir, str(task_id))
        os.makedirs(training_args.output_dir, exist_ok=True)
        task = list_of_ints(data_args.tasks)[task_id - 1]
        
        # Train task
        if task_id > start_task_id:
            rank0_print(f"Start training on task {task_id}: {task}")
            # 在 trainer 创建前彻底清理上一任务的 deepspeed/optimizer 状态
            deep_clean_model_state(model)
            if training_args.cl == "lwf":
                if task_id == 1 :
                    rank0_print(f"对于lwf的第一个任务{task},使用正常的loss")
                    is_first_iteration = True
                else:
                    rank0_print(f"对于lwf的第{task_id}个任务{task},使用lwf的loss")
                    is_first_iteration = False
                trainer = LLaVATrainer(model=model,
                                    tokenizer=tokenizer,
                                    args=training_args,
                                    cl_learner=cl_learner,
                                    is_first_iteration=is_first_iteration,
                                    **data_module)
            else:
                trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    cl_learner=cl_learner,
                    **data_module)

            if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
                trainer.train(resume_from_checkpoint=True)
            else:
                trainer.train()
            trainer.save_state()
            cache_state = model.config.use_cache
            model.config.use_cache = True

            if training_args.lora_enable:
                state_dict = get_peft_state_maybe_zero_3(
                    model.named_parameters(), training_args.lora_bias
                )
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                    model.named_parameters()
                )
                if training_args.local_rank == 0 or training_args.local_rank == -1:
                    model.config.save_pretrained(training_args.output_dir)
                    model.save_pretrained(training_args.output_dir, state_dict=state_dict)
                    torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            else:
                safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

            # 释放 accelerator/deepspeed engine 引用的 fp32 master / optimizer state
            try:
                trainer.accelerator.free_memory()
            except Exception as e:
                rank0_print(f"[WARN] accelerator.free_memory failed: {e}")
            if hasattr(trainer, "optimizer"):
                trainer.optimizer = None
            if hasattr(trainer, "lr_scheduler"):
                trainer.lr_scheduler = None
            if hasattr(trainer, "deepspeed"):
                trainer.deepspeed = None
            if hasattr(trainer, "model_wrapped"):
                trainer.model_wrapped = None
            del trainer

            deep_clean_model_state(model)

        cl_learner.after_train(task_id, model, tokenizer, **data_module)
        if training_args.cl == "l2p":
            cl_learner.save_task_pools(output_dir, task_id)
        rank0_print(f"Saving model for task {task_id} to: {training_args.output_dir}")
        rank0_print(f"Start inference after task {task_id}: {task}")
        cur_model_infer_path = os.path.join(output_dir,f'{task_id}')
        os.makedirs(cur_model_infer_path,exist_ok=True)
        restore_llama_attn_to_hf()
        # 强制推理走 eager attention, 避免 sdpa 与 LLaVA mm-prepare 的 4D mask 尺寸不匹配
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
        # 评估策略 (2026-06-07 改):
        # - 训完中间任务 (task_id < T): 只评当前任务 → 填对角线一个 cell
        # - 训完最后一个任务 (task_id == T): 评所有 task → 填最后一行
        # 这样产出的矩阵正好是 diag + last row, 足够算 AA / BWT / F_mean (见 evaluate_matrix.py).
        # 比起原来的累积评估省 21 次 eval / method (T=7), 大幅节省时间.
        is_last = (task_id == len(data_modules))
        for cur_val_id, cur_task_val in enumerate(data_modules):
            cur_task_val = tasks[cur_val_id]
            cur_val_id += 1
            if is_last:
                eval_this = (cur_val_id <= task_id)        # 最后一个任务: 全评
            else:
                eval_this = (cur_val_id == task_id)        # 中间任务: 只评自己
            if eval_this:
                # 包 try/except: 多机多卡场景下若一个 rank 推完很快、另一 rank 慢, 等待超时会抛
                # (NCCL/distributed timeout, 或 OOM, 或 sdpa 4D mask 不匹配等). 按用户决定:
                # 这类错误 nan 占位忽略, 继续下一个评估任务和后续训练, 不让单次 eval 失败把整 method 拖崩.
                try:
                    acc_matrix[task_id-1, cur_val_id-1] = evaluate(testing_args, tokenizer, model, task_id,cur_val_id, cur_task_val, cur_model_infer_path)
                except Exception as _eval_err:
                    rank0_print(f"[WARN] evaluate failed for task {cur_val_id} ({cur_task_val}) "
                                f"after training task {task_id}: "
                                f"{type(_eval_err).__name__}: {_eval_err}")
                    rank0_print(f"       已用 nan 占位, 继续下一个评估任务 (按 ERRORS_MACHINE_B.md §2 决议)")
                    acc_matrix[task_id-1, cur_val_id-1] = float("nan")
                    try:
                        import gc as _gc
                        _gc.collect()
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
        rank0_print(acc_matrix)
        enable_flash_attn()

        # 子进程逐任务模式: 这个任务训完+评估完, 直接退出进程 (OS 回收全部显存,
        # 杜绝跨任务 reload 泄漏). 下一任务由 wrapper 起新进程, start_task_id 自动续上.
        if only_task_id > 0:
            rank0_print(f"[SingleTask] task {task_id} done, exiting process for clean GPU reclaim.")
            break

        # === 任务间 model 重载: 把 deepspeed engine / optimizer state 整个清掉
        # 否则跨任务 deepspeed re-init 会 OOM (full-FT 7B on 4 GPUs)
        last_iter = (task_id == len(data_modules))
        if not last_iter:
            saved_path = os.path.join(output_dir, str(task_id))
            rank0_print(f"[Reload] del model + reload from {saved_path}")
            # 保留 L2P 等 learner 中对 model 的引用前先 detach
            if training_args.cl == "l2p" and hasattr(cl_learner, "txt_pool"):
                # L2P 的池子参数已挂在 model 上, 重载后需 re-bind
                _l2p_unbind = True
            else:
                _l2p_unbind = False

            # 彻底释放上一任务 GPU 显存. 否则每次 reload 泄漏 ~14GB 一个模型副本
            # (悬挂引用: accelerate AcceleratorState 单例 / deepspeed engine 持有旧 model),
            # 累积到第 5 个任务 reload 时 OOM. 先把旧 model 搬 CPU 保证 GPU 一定释放,
            # 再重置 accelerate 单例断开 engine 引用.
            try:
                model.to("cpu")
            except Exception as e:
                rank0_print(f"[Reload] move old model to cpu skip: {e}")
            vision_tower = None
            del model
            try:
                from accelerate.state import AcceleratorState
                AcceleratorState._reset_state()
            except Exception as e:
                rank0_print(f"[Reload] AcceleratorState reset skip: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            rank0_print(f"[Reload] after free, alloc={torch.cuda.memory_allocated()/1024**2:.0f}MB")

            model = LlavaLlamaForCausalLM.from_pretrained(
                saved_path,
                cache_dir=training_args.cache_dir,
                torch_dtype=torch.bfloat16,
                temperature=1.0, top_p=1.0,
            )
            model.config.use_cache = False
            if training_args.gradient_checkpointing:
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(dtype=torch.bfloat16, device=training_args.device)
            model.to(device=torch.device(f"cuda:{local_rank}"), dtype=torch.bfloat16)
            if _l2p_unbind:
                cl_learner.bind_model(model)
            release_memory()

    # 全部任务训练完成: max_merge 在此处合并所有快照.
    # 子进程模式下只在最后一个任务的进程里 finalize (其余任务进程跳过, 否则会拿不全 snapshot).
    _do_finalize = (only_task_id == 0) or (only_task_id == len(data_modules))
    if _do_finalize and training_args.cl == "max_merge" and hasattr(cl_learner, "finalize"):
        cl_learner.finalize(model)
        if local_rank in (0, -1):
            merged_dir = os.path.join(output_dir, "merged")
            os.makedirs(merged_dir, exist_ok=True)
            model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            rank0_print(f"[MaxMerge] merged model saved to {merged_dir}")

    rank0_print(acc_matrix)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
