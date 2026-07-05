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
import sys
import pdb

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

@dataclass
class SFTScriptArguments(ScriptArguments):
    image_root: str = field(default=None, metadata={"help": "The root directory of the image."})
    sampling_strategy: str = field(default="all", metadata={"help": "加载数据时采样的策略."})
    is_vqa: bool = field(default=True, metadata={"help": "训练数据的类型."})


processor = None

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: ScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    cur_data_dict = load_json_datas(json_path,sampling_strategy,sampling_number)
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".json"):
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

        # Format into conversation
        def make_conversation(example):
            QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then {Question_propmt} in <answer> </answer> tags. Output the final answer in JSON format."
            return [{
                "role": "user",
                "content": [{"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"].split('prompt:\n')[0],Question_propmt = example["problem"].split('prompt:\n')[-1].lower().rstrip('.?'))}]
                },
                {
                    
                        "role": "assistant",
                        "content": example['solution'],
                       
                }
            ]
        
        # Format into conversation
        def make_conversation_image(example):
            QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then {Question_propmt} in <answer> </answer> tags. Output the final answer in JSON format."

            image_root = self.script_args.image_root
            # print(111, image_root)
            # print(222, example['image'])
            image_path = os.path.join(image_root, example['image'])

            return  [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"].split('prompt:\n')[0],Question_propmt = example["problem"].split('prompt:\n')[-1].lower().rstrip('.?'))}
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": example['solution'],
                    }
                ]

        example = self.list_data_dict[i]
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
    if script_args.is_vqa:
        for example in examples:
            imgs, vids = process_vision_info(example["messages"])
            

            image_inputs.append(imgs)
    else:
        image_inputs = None
    # pdb.set_trace()
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
    # pdb.set_trace()
    return batch


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
    # pdb.set_trace()
    ################
    # Load tokenizer
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        # pdb.set_trace()
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
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
        # callbacks=get_callbacks(training_args, model_args),
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
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    print(script_args)
    main(script_args, training_args, model_args)