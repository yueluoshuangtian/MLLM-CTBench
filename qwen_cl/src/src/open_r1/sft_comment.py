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
from typing import List
import logging
import os
import sys
import pdb
import numpy as np

from grpo_rec import load_json_datas
import datasets
import torch
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint,seed_worker
from configs import SFTConfig
from utils.callbacks import get_callbacks
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
from transformers import TrainerCallback
# from grpo_rec import sft
from peft import PeftModel
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
@dataclass
class SFTScriptArguments(ScriptArguments):
    image_root: str = field(default=None, metadata={"help": "The root directory of the image."})
    sampling_strategy: str = field(default="all", metadata={"help": "加载数据时采样的策略."})

@dataclass
class SFTModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

processor = None
# ==== MONKEY PATCH: make process_vision_info safe for text-only batches ====
from typing import List, Union, Optional
from PIL import Image

# 尽量定位到实际实现所在的模块
try:
    import qwen_vl_utils.vision_process as _qvp
except Exception:  # 某些版本会直接在包顶层暴露
    import qwen_vl_utils as _qvp

def _safe_process_vision_info(
    conversations: Union[List[dict], List[List[dict]]],
    return_video_kwargs: bool = False,
) -> Union[
    tuple[List[Image.Image] | None, List[Union[torch.Tensor, List[Image.Image]]] | None],
    tuple[List[Image.Image] | None, List[Union[torch.Tensor, List[Image.Image]]] | None, dict]
]:
    """稳健版：
    - 无视觉内容 => 返回 (None, None)（或带 kwargs 的三元组）
    - 仅当存在图像时才 resize
    - 跳过文本片段与异常项
    """
    try:
        vision_infos = _qvp.extract_vision_info(conversations)
    except Exception as e:
        logger.warning("extract_vision_info failed, treat as no-vision: %s", e)
        if return_video_kwargs:
            return None, None, {"fps": []}
        return None, None

    images: List[Image.Image] = []
    videos: List[Union[torch.Tensor, List[Image.Image]]] = []
    fps_list: List[float] = []

    if not vision_infos:
        if return_video_kwargs:
            return None, None, {"fps": []}
        return None, None

    for vi in vision_infos:
        if not isinstance(vi, dict):
            continue
        try:
            if "image" in vi or "image_url" in vi:
                img = _qvp.fetch_image(vi)
                if img is None:
                    continue
                if isinstance(img, list):
                    images.extend(img)
                else:
                    images.append(img)
            elif "video" in vi:
                v, fps = _qvp.fetch_video(vi, return_video_sample_fps=True)
                videos.append(v)
                fps_list.append(fps)
            else:
                # 纯文本或未知键，忽略
                continue
        except Exception as e:
            logger.warning("Skip a vision item due to error: %s", e)
            continue

    # 只有有图像时才做 resize
    if len(images) == 0:
        processed_images = None
    else:
        try:
            processed_images = _qvp.resize_image(images)
        except Exception as e:
            logger.warning("resize_image failed, keep originals: %s", e)
            processed_images = images

    if len(videos) == 0:
        videos = None

    if return_video_kwargs:
        return processed_images, videos, {"fps": fps_list}
    return processed_images, videos

# 绑定到包与本地命名空间（双重覆盖，确保本文件后续使用的是安全版）
_qvp.process_vision_info = _safe_process_vision_info
process_vision_info = _safe_process_vision_info
# ==== END MONKEY PATCH ======================================================

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: ScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if data_path.endswith(".json"):
            json_path = data_path
            sampling_strategy = "all" if script_args.sampling_strategy is None else script_args.sampling_strategy
            sampling_number = None
            cur_data_dict = load_json_datas(json_path,sampling_strategy,sampling_number)
            self.list_data_dict.extend(cur_data_dict)

        else:
            raise ValueError(f"Unsupported file type: {data_path}")
        
    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i):
        # 处理批量索引（列表）
        if isinstance(i, list):
            return [self._get_single_item(idx) for idx in i]
        # 处理单个索引（整数）
        return self._get_single_item(i)
    
    def _get_single_item(self, idx):
        # Format into conversation
        def make_conversation(example):
            QUESTION_TEMPLATE = "{Question}.**Task:{process_Question_propmt} First output the thinking process in <think> </think> tags and then {answer_Question_propmt} in <answer> </answer> tags. Output the final answer in JSON format."
            question = example["problem"].split("**Task:")[0].strip()
            process_prompt = example['problem'].split("**Task:")[-1].replace(",and provide an overall score.",".")
            answer_prompt = "Provide the overall score in the format: \"Final Score for Response 1: score\""
            return [{
                "role": "user",
                "content": [{"type": "text", "text": QUESTION_TEMPLATE.format(Question = question,process_Question_propmt = process_prompt,answer_Question_propmt = answer_prompt)}]
                },
                {
                        "role": "assistant",
                        "content": example['solution'],
                }
            ]
        
        # Format into conversation
        def make_conversation_image(example):
            QUESTION_TEMPLATE = "{Question}.**Task:{process_Question_propmt} First output the thinking process in <think> </think> tags and then {answer_Question_propmt} in <answer> </answer> tags. Output the final answer in JSON format."
            question = example["problem"].split("**Task:")[0].strip()
            process_prompt = example['problem'].split("**Task:")[-1].replace(",and provide an overall score.",".")
            answer_prompt = "Provide the overall score in the format: \"Final Score for Response 1: score\""
            image_root = self.script_args.image_root
            # print(111, image_root)
            # print(222, example['image'])
            image_path = os.path.join(image_root, example['image'])

            return  [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": QUESTION_TEMPLATE.format(Question = question,process_Question_propmt = process_prompt,answer_Question_propmt = answer_prompt)}
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": example['solution'],
                    }
                ]

        example = self.list_data_dict[idx]
        if "image" in example.keys():
            example["messages"] = make_conversation_image(example)
        else:
            example['messages'] = make_conversation(example)

        return example

def collate_fn(examples):
    texts = [
        processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
        for example in examples
    ]

    image_inputs = []
    
    for example in examples:
        # print(example["messages"][0]["content"][0])
        imgs, vids = process_vision_info(example["messages"])
        image_inputs.append(imgs)
    
    if all(x is None or len(x) == 0 for x in image_inputs):
        image_inputs = None

    batch = processor(
        max_length = training_args.max_length,
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch
class GroupedHomogeneousBatchSampler(BatchSampler):
    def __init__(
        self,
        qa_idx:   List[int],
        vqa_idx:  List[int],
        batch_size: int,
        world_size: int,
        drop_last: bool = False,
        seed: int = 42,
    ):
        # 调用父类初始化（必须！）
        super().__init__(None, batch_size, drop_last=False)
        
        g = torch.Generator()
        g.manual_seed(seed)
        qa_batches  = _chunk(qa_idx,  batch_size)
        vqa_batches = _chunk(vqa_idx, batch_size)
        #这里打乱qa和vqa内部的顺序
        qa_batches  = [qa_batches[i]  for i in torch.randperm(len(qa_batches),  generator=g)]
        vqa_batches = [vqa_batches[i] for i in torch.randperm(len(vqa_batches), generator=g)]
        #按照进程数划分batch，保证每个iteration中只有相同类型的batch
        qa_batches = _chunk(qa_batches,world_size)
        vqa_batches = _chunk(vqa_batches,world_size)
        
        all_batches = qa_batches + vqa_batches
        all_batches = [all_batches[i] for i in torch.randperm(len(all_batches),generator=g)]
        self.grouped = []

        for batch in all_batches:
            self.grouped.extend(batch)
        



    def __iter__(self):
        
        for batch in self.grouped:
            yield batch  # 每次迭代返回一个批次（索引列表）

    def __len__(self):

        return len(self.grouped)
            
def _chunk(idx, bs):
    
    return [idx[i : i + bs] for i in range(0, len(idx), bs) if len(idx[i : i + bs]) == bs]
  
class HomaBatchTrainer(SFTTrainer):
    def get_train_dataloader(self):
        qa_idx = [i for i, ex in enumerate(self.train_dataset) if "image" not in ex]
        vqa_idx = [i for i, ex in enumerate(self.train_dataset) if "image" in ex]
        
        world_size = self.accelerator.num_processes   # 多卡数量
        batch_sampler = GroupedHomogeneousBatchSampler(
                                                        qa_idx, vqa_idx,
                                                        batch_size=self.args.per_device_train_batch_size,
                                                        world_size=world_size,
                                                        drop_last=False,
                                                        seed=self.args.seed,
                                                        )
        
        dataloader_params = {
            # "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if not isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor    
            dataloader_params["batch_sampler"] = batch_sampler
            
        return self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")
    
    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")
    
    ################
    # Load datasets
    ################

    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)
    # 
    ################
    # Load tokenizer
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        # 
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
        )
        logger.info("Using AutoTokenizer for text-only model.")
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
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
    # training_args.model_init_kwargs = model_kwargs
    from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
    if "Qwen2-VL" in model_args.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    elif "Qwen2.5-VL" in model_args.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    else:
        raise ValueError(f"Unsupported model: {model_args.model_name_or_path}")
    
    if model_args.use_peft:
        
        peft_config=get_peft_config(model_args)
        def find_all_linear_names(model,multimodel_keywords):
            cls = torch.nn.Linear
            lora_module_names = set()
            
            for name, module in model.named_modules():
                
                if any(mm_keyword in name for mm_keyword in multimodel_keywords):
                    continue
                if isinstance(module,cls):
 
                    lora_module_names.add(name)
            for m in lora_module_names:
                if "embed_tokens" in m:
                    lora_module_names.remove(m)
            return list(lora_module_names)
 
        target_modules = find_all_linear_names(model,[model_args.lora_target_modules] if isinstance(model_args.lora_target_modules,str) else model_args.lora_target_modules)
        
        peft_config.target_modules = target_modules
        
    if model_args.freeze_vision_modules:
        for name,module in model.named_modules():
            if any(key in name for key in model_args.lora_target_modules):
                module.requires_grad = False
                
              
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None and model_args.use_peft:
        model = PeftModel.from_pretrained(model, last_checkpoint)
    ############################
    # Initialize the SFT Trainer
    ############################
    class SaveProcessorConfigCallback(TrainerCallback):
        def on_save(self, args, state, control, model=None, tokenizer=None, **kwargs):
            if state.is_world_process_zero:
                if processor is not None:
                    global_step = state.global_step
                    output_dir=f"{args.output_dir}/checkpoint-{global_step}"
                    processor.save_pretrained(output_dir)
                    logger.info("处理器配置文件（preprocessor_config.json）已在检查点保存。")
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False
    trainer = HomaBatchTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=peft_config if model_args.use_peft else None,
        callbacks=get_callbacks(training_args, model_args)+ [SaveProcessorConfigCallback],
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    metrics = train_result.metrics

    metrics["train_samples"] = len(dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    ##################################
    # Save model, processor and config
    ##################################
    logger.info("*** Save model and processor ***")

    # Save model
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save processor (tokenizer or processor with vision-language support)
    if processor is not None:
        processor.save_pretrained(training_args.output_dir)
        logger.info("Processor saved.")

    # Restore config for inference and save
    if hasattr(trainer.model, "config"):
        trainer.model.config.use_cache = True  # re-enable use_cache for fast generation
        trainer.model.config.to_json_file(os.path.join(training_args.output_dir, "config.json"))
        logger.info("Model config updated and saved (use_cache=True).")

    # Save everything else on main process
    kwargs = {
        "model_name": model_args.model_name_or_path,
        "dataset_name": list(script_args.dataset_name),
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)

    #############
    # Push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, SFTModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    print(script_args)
    main(script_args, training_args, model_args)