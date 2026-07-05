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
    --resume_from_checkpoint True \
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
import glob
import gc
import datasets
import torch

import transformers
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
from configs import SFTConfig
from utils.callbacks import get_callbacks
from evaluate_cl import evaluate_main,data_model_types
from PIL import Image
from transformers.trainer_utils import get_last_checkpoint,seed_worker
from trl import (
    ModelConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from qwen_vl_utils import process_vision_info
logger = logging.getLogger(__name__)
from torch.utils.data import BatchSampler,DataLoader
from transformers import TrainerCallback
# from grpo_rec import sft
from data.replay_dataset import make_cl_data_module,SFTScriptArguments,freeze_model
from typing import List
import pdb
local_rank = None
processor = None

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
    training_args.packing = False
    # Set seed for reproducibility
    set_seed(training_args.seed)
    ###get tasks
    script_args.tasks = [task.strip() for task in script_args.tasks.split(',') if len(task.strip())>0]
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

    # Check for last tast and it's last checkpoint
    start_task_id = 1
    last_checkpoint = None
    output_dir = training_args.output_dir
    
    os.makedirs(output_dir,exist_ok=True)
    for d in glob.glob(os.path.join(output_dir, "*")):
        if os.path.isdir(d) and os.path.basename(d).isdigit() and len(os.listdir(d)) > 0:
            start_task_id = max(int(os.path.basename(d)), start_task_id)
    config_path = os.path.join(output_dir, str(start_task_id), "preprocessor_config.json")
    
    if os.path.exists(config_path):
        model_args.model_name_or_path = os.path.join(training_args.output_dir, str(start_task_id))
        start_task_id += 1
    else:
        if start_task_id > 1:
            model_args.model_name_or_path = os.path.join(training_args.output_dir, str(start_task_id-1))
    
    if os.path.isdir(os.path.join(output_dir,str(start_task_id))):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")
    print(f'--------------start from the task {script_args.tasks[start_task_id-1]}------------')
    print(f'--------------加载的模型是{model_args.model_name_or_path}------------')

    
    ################
    # Load tokenizer
    ################
    global processor
    def get_processor(model_args):
        if "vl" in model_args.model_name_or_path.lower():
            processor = AutoProcessor.from_pretrained(
                model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
            )
            
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
        return processor
    processor = get_processor(model_args)
    ###################
    # Model init kwargs
    ###################
    def get_accelerete_model(model_args):
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
        return model
    model = get_accelerete_model(model_args)
    if script_args.cl_method == 'freeze_pre_eight_layers':
        layers = range(8)
        freeze_model(model,layers)
    elif script_args.cl_method == 'freeze_last_eight_layers':
        num_total_layers = 36
        num_layers_to_freeze = 8
        layers_to_freeze = list(range(num_total_layers - num_layers_to_freeze, num_total_layers))
        freeze_model(model,layers_to_freeze)
    ################
    # Load CL datasets
    ################
    data_modules = make_cl_data_module(script_args)
    test_filepath = os.path.join(output_dir,"prediction")
    os.makedirs(test_filepath, exist_ok=True)
    ################
    # Initial CL
    ################
    logger.info(f"使用的持续学习方法是{script_args.cl_method},使用的任务{script_args.tasks},方法的参数是{script_args.cl_method_alpha}")
    for task_id, data_module in enumerate(data_modules):

        if start_task_id > 0 and task_id + 1 < start_task_id:
            continue

        script_args.is_vqa = data_model_types[script_args.tasks[task_id]]
        print(f"开始在第{task_id + 1}个任务：{data_module['task_name']}上进行训练")
        logger.info(f"开始在第{task_id + 1}个任务：{data_module['task_name']}上进行训练")
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
        ####保存位置

        training_args.output_dir = os.path.join(output_dir,str(task_id+1))
        print(f'_____________________保存的位置是{training_args.output_dir}____________________________')
        trainer = HomaBatchTrainer(
            model=model,
            args=training_args,
            train_dataset=data_module['train_dataset'],
            eval_dataset=None,
            processing_class=processor.tokenizer,
            data_collator=collate_fn,
            # peft_config=get_peft_config(model_args),
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

        metrics["train_samples"] = len(data_module['train_dataset'])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        ##################################
        # Save model, processor and config
        ##################################
        logger.info("*** Save model and processor ***")

        # Save model
        if trainer.accelerator.is_main_process:
            unwrapped = trainer.accelerator.unwrap_model(trainer.model)
            unwrapped.save_pretrained(training_args.output_dir,
                                    state_dict=unwrapped.state_dict(),
                                    safe_serialization=True)
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
            "dataset_name": data_module['task_name'],
            "tags": ["open-r1"],
        }
        if trainer.accelerator.is_main_process:
            trainer.create_model_card(**kwargs)

        del trainer
        
        if task_id < len(script_args.tasks)-1:
            #给出一个错误结束这次循环
            evaluate_main()
        release_memory()
        #############
        # Push to hub
        #############
        # if training_args.push_to_hub:
        #     logger.info("Pushing to hub...")
        #     trainer.push_to_hub(**kwargs)
def release_memory():
    gc.collect()
    torch.cuda.empty_cache()
    memory_stats()
def memory_stats():
    print(f"memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2}")
    print(f"memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2}")


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    print(script_args)
    main(script_args, training_args, model_args)