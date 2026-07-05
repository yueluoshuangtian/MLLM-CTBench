# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""

import logging
import os
import pdb
from tqdm import tqdm
from grpo_rec import load_json_datas
import datasets
import torch
from torch.utils.data import Dataset
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
from configs import SFTConfig
from utils.callbacks import get_callbacks
import yaml
import json
import math
import random
from PIL import Image
from evaluate_cl import data_model_types
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from dataclasses import field
from qwen_vl_utils import process_vision_info
logger = logging.getLogger(__name__)
from dataclasses import dataclass
import gc
import deepspeed
local_rank = None


@dataclass
class SFTScriptArguments(ScriptArguments):
    image_root: str = field(default=None, metadata={"help": "The root directory of the image."})
    sampling_strategy: str = field(default="all", metadata={"help": "加载数据时采样的策略."})
    cur_task: str = field(default=None, metadata={"help": "任务名称."})
    alpha:float = field(default=0.8,metadata={"help": "模型融合的参数."})
@dataclass
class SFTModelConfig(ModelConfig):
    freeze_vision_modules: bool = False
    ft_model_paths:str = field(default="", metadata={"help": "微调模型根目录"})



from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
def get_model_and_tokenizer(model_args,  training_args,model_name_or_path):
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    if "Qwen2-VL" in model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path, **model_kwargs
        )
    elif "Qwen2.5-VL" in model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, **model_kwargs
        )
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")
    ################
    # Load tokenizer
    ################
    
    if "vl" in model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        # 
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
        )
        logger.info("Using AutoTokenizer for text-only model.")
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    return model,processor
def ensure_cuda_and_dense(tensor, name=""):
    """确保Tensor在CUDA上且是密集的"""
    if not tensor.is_cuda:
        tensor = tensor.cuda()
    if tensor.is_sparse:
        tensor = tensor.to_dense()
    return tensor

def max_matric(ori_model_matric, max_model_matric, contrast_model_matric,alpha):
    # 确保所有输入张量在相同设备上
    device = ori_model_matric.device
    max_model_matric = max_model_matric.to(device)
    contrast_model_matric = contrast_model_matric.to(device)
    
    contrast_minus_ori = contrast_model_matric - ori_model_matric
    max_minus_ori = max_model_matric - ori_model_matric
    
    abs_contrast_minus_ori = torch.abs(contrast_minus_ori)
    abs_max_minus_ori = torch.abs(max_minus_ori)
    
    max_matrix = torch.where(abs_contrast_minus_ori > abs_max_minus_ori, 
                           contrast_model_matric, 
                           max_model_matric)
    max_matrix_contrast = max_matrix - ori_model_matric
    
    max_matrix = ori_model_matric + alpha * max_matrix_contrast
    return max_matrix

def max_merge_contrast_model(ori_model_name_and_parameters, max_merge_model, contrastive_model_name_and_parameters,alpha):
    if contrastive_model_name_and_parameters is None:
        return
    
    # 确保模型在相同设备上
    device = next(max_merge_model.parameters()).device
    
    print("开始模型参数的融合")
    for name, param in tqdm(max_merge_model.named_parameters(), desc="模型融合"):
        if name in ori_model_name_and_parameters and name in contrastive_model_name_and_parameters:
            # 确保所有相关张量在相同设备上
            ori_tensor = ori_model_name_and_parameters[name].to(device)
            contrast_tensor = contrastive_model_name_and_parameters[name].to(device)
            
            # 计算融合后的参数
            merged_param = max_matric(ori_tensor, param.data, contrast_tensor,alpha)
            
            # 更新模型参数
            param.data.copy_(merged_param)
from transformers import GenerationConfig
def max_merge_all_model(script_args, training_args, model_args):
    global local_rank, processor
    import torch.distributed as dist
    # 初始化时设置设备
    device = torch.device(f"cuda:{training_args.local_rank}" if torch.cuda.is_available() else "cpu")
    
    # 加载原始模型
    
    ori_model, processor = get_model_and_tokenizer(model_args,  training_args,model_args.model_name_or_path)
    ori_model = ori_model.to(device)
    
    # 确保模型参数在CUDA上
    ori_model_name_and_parameters = {}
    for name, param in ori_model.named_parameters():
        param_data = param.detach().clone()
        param_data = ensure_cuda_and_dense(param_data, name)
        ori_model_name_and_parameters[name] = param_data
    
    del ori_model
    torch.cuda.empty_cache()
    gc.collect()
    

    
    root = model_args.ft_model_paths
    ft_model_paths = [
        abs_path                                      # 最终保存的是绝对路径
        for entry in os.listdir(root)                 # ① 枚举根目录下的所有条目
        if (
            os.path.isdir(abs_path := os.path.abspath(os.path.join(root, entry)))  # ② 只保留子目录
            and "max_merge" not in abs_path  and "prediction" not in abs_path                                    # ③ 排除包含 max_merge 的目录
            and os.path.basename(abs_path)[-1].isdigit()                          # ④ 目录名最后一个字符为数字
        )
    ]
    print(ft_model_paths)
    if len(ft_model_paths) == 0:
        print("没有微调后的模型来进行融合")
        return None, processor
        
    have_max_merge_model = False
    max_merge_model = None

    try:
        print(f"模型输出的位置{training_args.output_dir}")
        for num_id, ft_model_path in tqdm(enumerate(ft_model_paths),desc='模型融合的进展'):
            if not have_max_merge_model:
                # 所有rank独立加载第一个模型
                print(f"所有rank加载第一个微调模型:{ft_model_path}")
                max_merge_model, _ = get_model_and_tokenizer(model_args,  training_args,ft_model_path)
                max_merge_model = max_merge_model.to(device)
                have_max_merge_model = True
                continue
            
            # 所有rank独立加载对比模型
            print(f"所有rank加载对比模型:{ft_model_path}")
            contrastive_model, _ = get_model_and_tokenizer(model_args,  training_args,ft_model_path)
            contrastive_model = contrastive_model.to(device)
            
            # 确保对比模型参数在CUDA上
            contrastive_model_name_and_parameters = {}
            for name, param in contrastive_model.named_parameters():
                param_data = param.detach().clone()
                param_data = ensure_cuda_and_dense(param_data, name)
                contrastive_model_name_and_parameters[name] = param_data
            
            # 只在rank 0执行融合计算
            if training_args.local_rank == 0:
                max_merge_contrast_model(ori_model_name_and_parameters, max_merge_model, 
                                      contrastive_model_name_and_parameters,script_args.alpha)
            
            del contrastive_model
            torch.cuda.empty_cache()
            gc.collect()
        
            # 使用Deepspeed广播同步融合后的模型
            if dist.is_initialized():
                # 确保所有rank等待rank 0完成融合
                dist.barrier()
                
                # 广播模型参数
                for name, param in max_merge_model.named_parameters():
                    # 确保参数在广播前是CUDA且密集的
                    param.data = ensure_cuda_and_dense(param.data, name)
                    dist.broadcast(param.data, src=0)
            
                # 只在rank 0保存模型
                if training_args.local_rank == 0:
                    
                    # 强烈建议在保存前把 use_cache 重新打开，以便推理
                    os.makedirs(training_args.output_dir, exist_ok=True)
                    
                    # 确保 config 已设置并生效
                    max_merge_model.config.use_cache = True

                    # 若无 generation_config，则补充
                    if getattr(max_merge_model, "generation_config", None) is None:
                        max_merge_model.generation_config = GenerationConfig.from_model_config(
                            max_merge_model.config
                        )

                    # 一次性保存全部模型与 config
                    max_merge_model.save_pretrained(
                        training_args.output_dir,
                        safe_serialization=True
                    )
                    # 显式保存 generation_config
                    max_merge_model.generation_config.save_pretrained(training_args.output_dir)

                    # 只需保存 processor 一次
                    if processor is not None:
                        processor.save_pretrained(training_args.output_dir)
                        logger.info("Processor saved.")
                print("保存融合模型")
                # del trainer
                gc.collect()
            
            print(f"融合了第{num_id+1}个微调后模型:{ft_model_path}")
        return max_merge_model,processor
    except Exception as e:
        print(f"融合过程中发生错误: {str(e)}")
        if dist.is_initialized():
          dist.barrier()
          dist.destroy_process_group()
        raise
def main(script_args, training_args, model_args):
    global local_rank
    # 初始化设备
    torch.cuda.set_device(training_args.local_rank)
    device = torch.device(f"cuda:{training_args.local_rank}")
    # Set seed for reproducibility
    set_seed(training_args.seed)
    try:
        # 初始化分布式训练
        deepspeed.init_distributed()
        
        model,processor = max_merge_all_model(script_args, training_args, model_args)
        
    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        raise



if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, SFTModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    print(script_args)
    main(script_args, training_args, model_args)