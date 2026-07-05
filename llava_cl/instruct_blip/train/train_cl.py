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
from typing import Optional
import torch
import torch.distributed as dist

import transformers
import tokenizers

from instruct_blip.data.dataset import DataArguments, make_cl_data_module, create_test_loader, get_chunk
from instruct_blip.train.blip_trainer import InstructBlipTrainer
from instruct_blip.model.forward import replace_instruct_blip_forward, replace_instruct_blip_generate

from instruct_blip.eval.eval_vqa import eval_single as eval_single_vqa
from instruct_blip.eval.eval_caption import eval_single as eval_single_caption

from cl_learner import *

import deepspeed

local_rank = None

question_file_dict = {
    "flickr30k": "./playground/data/eval/flickr30k/instructblip_flickr30k_val.jsonl",
    "textcaps": "./playground/data/eval/textcaps/instructblip_textcaps_val.jsonl",
    "vqav2": "./playground/data/eval/vqav2/instructblip_vqav2_val.jsonl",
    "ocrvqa": "./playground/data/eval/ocrvqa/instructblip_ocrvqa_val.jsonl",
    "gqa": "./playground/data/eval/gqa/instructblip_gqa_testdev_balanced.jsonl",
    "okvqa": "./playground/data/eval/okvqa/instructblip_okvqa_val.jsonl",
    "aokvqa": "./playground/data/eval/aokvqa/instructblip_aokvqa_val.jsonl",
    "vizwiz": "./playground/data/eval/vizwiz/instructblip_vizwiz_val.jsonl",
    "textvqa": "./playground/data/eval/textvqa/instructblip_textvqa_val_v051_ocr.jsonl",
    "scienceqa": "./playground/data/eval/scienceqa/instructblip_sqa_val.jsonl"
}

annotation_file_dict = {
    "flickr30k": "./playground/data/eval/flickr30k/gt_flickr30k_val.json",
    "textcaps": "./playground/data/eval/textcaps/gt_textcaps_val.json",
    "vqav2": "./playground/data/eval/vqav2/gt_vqav2_val.json",
    "ocrvqa": "./playground/data/eval/ocrvqa/gt_ocrvqa_val.json",
    "gqa": "./playground/data/eval/gqa/gt_gqa_testdev_balanced.json",
    "okvqa": "./playground/data/eval/okvqa/gt_okvqa_val.json",
    "aokvqa": "./playground/data/eval/aokvqa/gt_aokvqa_val.json",
    "vizwiz": "./playground/data/eval/vizwiz/gt_vizwiz_val.json",
    "textvqa": "./playground/data/eval/textvqa/TextVQA_0.5.1_val.json",
    "scienceqa": "./playground/data/eval/scienceqa/gt_sqa_val.json"
}

image_folder_dict = {
    "flickr30k": "./playground/data/flickr30k",
    "textcaps": "./playground/data/textvqa",
    "vqav2": "./playground/data/coco/",
    "ocrvqa": "./playground/data/ocr_vqa/images",
    "gqa": "./playground/data/gqa/images",
    "okvqa": "./playground/data/coco",
    "aokvqa": "./playground/data/coco",
    "vizwiz": "./playground/data/vizwiz/images",
    "textvqa": "./playground/data/textvqa/train_images",
    "scienceqa": "./playground/data/scienceqa/images"
}

replace_instruct_blip_forward()
replace_instruct_blip_generate()


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
    num_beams: int = 1
    max_new_tokens: int = 128


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    pretrain_qformer: Optional[str] = field(default=None)
    freeze_visual_encoder: bool = field(default=True)
    freeze_llm: bool = field(default=True)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
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

    cl: str = "none"
    ewc_lbd: float = 1e8
    lwf_lbd: float = 1.
    eproj_lbd: float = 1.
    num_experts: int = 9
    eproj_key_lr: Optional[float] = 1e-2
    eproj_key_steps: Optional[int] = 2000


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
    multimodal_keywords = ['language_projection', 'vision_model', 'language_model']  # only tune qformer
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
    """Collects the state dict and dump to disk."""

    if hasattr(trainer.model, "e_language_projection"):
        # Only save Adapter
        keys_to_match = ['language_projection', 'e_language_projection']
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

    elif trainer.model.config.freeze_visual_encoder and trainer.model.config.freeze_llm:
        keys_to_match = ['query_tokens', 'qformer', 'language_projection']
        keys_not_match = []

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match, keys_not_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                qformer_folder = os.path.join(parent_folder, "qformer")
                os.makedirs(qformer_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(qformer_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'qformer.bin'))
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


def evaluate(args, processor, model, task_id, task_val, output_dir):
    rank0_print(f"Testing on {task_val}")
    answers_file = os.path.join(output_dir, "predictions", f"{task_id}-{task_val}.jsonl")
    acc = 0.

    if not os.path.isfile(answers_file):
        processor.tokenizer.padding_side = "left"

        num_chunks = int(os.getenv('WORLD_SIZE', '1'))
        chunk_idx = local_rank
        question_file = question_file_dict[task_val]
        image_folder = image_folder_dict[task_val]
        answers_chunk_file = os.path.join(output_dir, "predictions", f"{task_id}-{task_val}",
                                          f"{num_chunks}-{chunk_idx}.jsonl")
        questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
        questions = get_chunk(questions, num_chunks, chunk_idx)
        data_loader = create_test_loader(questions, image_folder, processor)
        answers_chunk_file = os.path.expanduser(answers_chunk_file)
        os.makedirs(os.path.dirname(answers_chunk_file), exist_ok=True)
        ans_file = open(answers_chunk_file, "w")

        if hasattr(model, "e_language_projection"):
            model.e_language_projection.init_task_id_retrieve_acc()

        model.gradient_checkpointing_disable()
        model.eval()
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), miniters=len(data_loader)//10, disable=(local_rank != 0)):
            question_ids = batch.pop("question_ids")
            prompts = batch["prompts"]

            batch = {n: p.to(device="cuda", non_blocking=True) if isinstance(p, torch.Tensor) else p for n, p in batch.items()}
            batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.bfloat16)

            with torch.inference_mode():
                output_ids = model.generate(
                    **batch,
                    do_sample=False,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=False,
                )

            outputs = processor.batch_decode(output_ids, skip_special_tokens=True)

            ans_id = shortuuid.uuid()
            for idx, prompt, output in zip(question_ids, prompts, outputs):
                ans_file.write(json.dumps({"question_id": idx,
                                           "prompt": prompt,
                                           "text": output.strip(),
                                           "answer_id": ans_id,
                                           "model_id": str(task_id),
                                           "metadata": {}}) + "\n")

            if hasattr(model, "e_language_projection") and (i % 50 == 0 or i == len(questions) - 1):
                print(model.e_language_projection.cal_task_id_retrieve_acc())

            ans_file.flush()
        ans_file.close()
        dist.barrier()  # wait for all chunks are completed

        if local_rank == 0:
            ans_file = open(answers_file, "w")
            for i in range(num_chunks):
                for q in open(
                        os.path.join(output_dir, "predictions", f"{task_id}-{task_val}", f"{num_chunks}-{i}.jsonl")):
                    ans_file.write(q)
                    ans_file.flush()
            ans_file.close()

    dist.barrier()  # wait for all chunks are completed
    if local_rank == 0:
        if os.path.isdir(os.path.join(output_dir, "predictions", f"{task_id}-{task_val}")):
            shutil.rmtree(os.path.join(output_dir, "predictions", f"{task_id}-{task_val}"))
        if task_val in ["textcaps", "flickr30k"]:
            acc = eval_single_caption(annotation_file_dict[task_val], answers_file, task_val)
        elif task_val in ["vqav2", "vizwiz", "textvqa", "okvqa", "gqa", "ocrvqa", "aokvqa"]:
            acc = eval_single_vqa(annotation_file_dict[task_val], answers_file, task_val)
        else:
            raise ValueError(f"Unknown task : {task_val}")

    dist.barrier()
    processor.tokenizer.padding_side = "right"

    return acc


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, TestingArguments))
    model_args, data_args, training_args, testing_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

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

    # resume from task
    start_task_id = 0
    for d in glob.glob(os.path.join(training_args.output_dir, "*")):
        if os.path.isdir(d) and os.path.basename(d).isdigit() and len(os.listdir(d)) > 0:
            start_task_id = max(int(os.path.basename(d)), start_task_id)
    if start_task_id > 0 \
            and not (model_args.freeze_visual_encoder and model_args.freeze_llm) \
            and training_args.cl != "eproj":
        model_args.model_name_or_path = os.path.join(training_args.output_dir, str(start_task_id))

    model = transformers.InstructBlipForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **bnb_model_from_pretrained_args
    )
    processor = transformers.InstructBlipProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir
    )

    if model_args.pretrain_qformer is not None:
        qformer_weights = torch.load(model_args.pretrain_qformer, map_location="cpu")["model"]
        qformer_weights_convert = {}
        for name, param in qformer_weights.items():
            name = name.replace("Qformer.bert", "qformer").replace("attention.self", "attention.attention").replace("llm_proj", "language_projection")
            name = name.replace("ln_vision", "vision_model.post_layernorm").replace("qformer.embeddings.LayerNorm", "qformer.embeddings.layernorm")
            qformer_weights_convert[name] = param
        incompatible_keys = model.load_state_dict(qformer_weights_convert, strict=False)
        rank0_print(incompatible_keys)

    if start_task_id > 0 and model_args.freeze_visual_encoder and model_args.freeze_llm and training_args.cl != "eproj":
        qformer_weights = torch.load(os.path.join(training_args.output_dir, str(start_task_id), "qformer.bin"),
                                     map_location="cpu")
        incompatible_keys = model.load_state_dict(qformer_weights, strict=False)
        rank0_print(incompatible_keys)

    model.config.use_cache = False
    model.config.freeze_visual_encoder = model_args.freeze_visual_encoder
    model.config.freeze_llm = model_args.freeze_llm

    if model_args.freeze_visual_encoder:
        model.vision_model.requires_grad_(False)
    if model_args.freeze_llm:
        model.language_model.requires_grad_(False)

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
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

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

    assert model.config.use_decoder_only_language_model, "Data collator only supports decoder only language model"

    data_modules = make_cl_data_module(processor=processor,
                                       data_args=data_args,
                                       task_list=data_args.tasks,
                                       num_query_tokens=model.config.num_query_tokens,
                                       rank=local_rank)
    initial_data_modules = make_cl_data_module(processor=processor,
                                               data_args=data_args,
                                               task_list=data_args.initial_tasks,
                                               num_query_tokens=model.config.num_query_tokens,
                                               rank=local_rank)

    deepspeed.init_distributed()

    if training_args.cl == "tir+ewc":
        task_encoder = TaskEncoder(model.vision_model,
                                   training_args.local_rank,
                                   training_args.output_dir)
        cl_learner = TIREWCLearner(task_encoder,
                                   InstructBlipTrainer,
                                   training_args.ewc_lbd,
                                   training_args.local_rank,
                                   training_args.output_dir,
                                   training_args)
    elif training_args.cl == "tir+mas":
        task_encoder = TaskEncoder(model.vision_model,
                                   training_args.local_rank,
                                   training_args.output_dir)
        cl_learner = TIRMASLearner(task_encoder,
                                   InstructBlipTrainer,
                                   training_args.ewc_lbd,
                                   training_args.local_rank,
                                   training_args.output_dir,
                                   training_args)
    elif training_args.cl == "ewc":
        cl_learner = EWCLearner(InstructBlipTrainer,
                                training_args.ewc_lbd,
                                training_args.local_rank,
                                training_args.output_dir,
                                training_args)
    elif training_args.cl == "mas":
        cl_learner = MASLearner(InstructBlipTrainer,
                                training_args.ewc_lbd,
                                training_args.local_rank,
                                training_args.output_dir,
                                training_args)
    elif training_args.cl == "lwf":
        cl_learner = LwFLearner(training_args.lwf_lbd,
                                training_args.local_rank,
                                model)
    elif training_args.cl == "eproj":
        cl_learner = EprojLearner(InstructBlipTrainer,
                                  training_args.eproj_lbd,
                                  training_args.num_experts,
                                  training_args.eproj_key_steps,
                                  training_args.local_rank,
                                  training_args.output_dir,
                                  training_args,
                                  model)
    else:
        cl_learner = BaseCLearner()

    model.to(device=torch.device(f"cuda:{local_rank}"), dtype=torch.bfloat16)

    num_initial_tasks = len(list_of_ints(data_args.initial_tasks))
    num_tasks = len(data_modules)
    acc_matrix = np.zeros((num_tasks + 1, num_initial_tasks + num_tasks))
    output_dir = training_args.output_dir

    dist.barrier()

    # evaluate on initial tasks
    for val_id, task_val in enumerate(list_of_ints(data_args.initial_tasks)):
        rank0_print(f"Prepare for task 0: {task_val}")
        data_module = initial_data_modules[val_id]
        cl_learner.before_train(0, model, **data_module)
        cl_learner.after_train(0, model, **data_module)
        acc_matrix[0, val_id] = evaluate(testing_args, processor, model, 0, task_val, output_dir)
    for val_id, task_val in enumerate(list_of_ints(data_args.tasks)):
        acc_matrix[0, val_id + num_initial_tasks] = evaluate(testing_args, processor, model, 0, task_val, output_dir)

    for task_id, data_module in enumerate(data_modules):
        task_id += 1
        cl_learner.before_train(task_id, model, **data_module)
        training_args.output_dir = os.path.join(output_dir, str(task_id))
        task = list_of_ints(data_args.tasks)[task_id - 1]

        # Train task
        if task_id > start_task_id:
            rank0_print(f"Start training on task {task_id}: {task}")
            release_memory()

            trainer = InstructBlipTrainer(model=model,
                                          args=training_args,
                                          cl_learner=cl_learner,
                                          **data_module)

            if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
                trainer.train(resume_from_checkpoint=True)
            else:
                trainer.train()
            trainer.save_state()

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
            del trainer

            # delete the model after testing
            if local_rank == 0 and os.path.isdir(os.path.join(output_dir, str(task_id - 1))) and training_args.cl != "eproj":
                shutil.rmtree(os.path.join(output_dir, str(task_id - 1)))

            release_memory()

        cl_learner.after_train(task_id, model, **data_module)

        rank0_print(f"Start inference after task {task_id}: {task}")
        for val_id, task_val in enumerate(list_of_ints(data_args.initial_tasks) + list_of_ints(data_args.tasks)):
            if (val_id-num_initial_tasks+1 == task_id) or (val_id-num_initial_tasks == task_id) or (task_id == num_tasks):
                acc_matrix[task_id, val_id] = evaluate(testing_args, processor, model, task_id, task_val, output_dir)

    aa = acc_matrix[-1, :].mean()
    bwt = (acc_matrix[-1, :] - acc_matrix).min(0).mean()

    acc_matrix = pd.DataFrame(acc_matrix,
                              columns=list_of_ints(data_args.initial_tasks) + list_of_ints(data_args.tasks),
                              index=["initial"] + list_of_ints(data_args.tasks))
    rank0_print(acc_matrix)
    rank0_print(f"Average Accuracy {aa}\nBWT {bwt}")

    if local_rank <= 0:
        acc_matrix["aa"] = np.zeros(acc_matrix.shape[0])
        acc_matrix.iloc[-1, -1] = aa
        acc_matrix["bwt"] = np.zeros(acc_matrix.shape[0])
        acc_matrix.iloc[-1, -1] = bwt
        acc_matrix.to_csv(os.path.join(output_dir, "results.csv"))


if __name__ == "__main__":
    train(attn_implementation="eager")
