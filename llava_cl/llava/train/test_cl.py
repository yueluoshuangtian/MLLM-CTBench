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
from llava_trainer import LLaVATrainer
from llava.data.dataset import DataArguments, make_cl_data_module, smart_tokenizer_and_embedding_resize
from llava.eval.model_vqa_loader import create_data_loader, get_chunk, KeywordsStoppingCriteria
from llava.eval.eval_vqa import eval_single as eval_single_vqa
from llava.eval.eval_caption import eval_single as eval_single_caption
from llava.conversation import conv_templates, SeparatorStyle

from llava import conversation as conversation_lib
from llava.model import *

from cl_learner import *

import deepspeed
import sys
import os


directory_path = "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/new_eval_tool"
sys.path.insert(0, directory_path)
import evaluate_art_long_sentences,evaluate_FOMC,evaluate_math,evaluate_numglue,evaluate_object,evaluate_science,evaluate_medical_long_sentences,evaluate_ocr

local_rank = None
#question_word
question_file_dict = {
    "numglue": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/reasoning_test_for_test/numglue.json",
    "art": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/reasoning_test_for_test/art.json",
    "math": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/reasoning_test_for_test/math.json",
    "fomc": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/reasoning_test_for_test/fomc.json",
    "medical": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/reasoning_test_for_test/medical.json",
    "OCR": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/reasoning_test_for_test/OCR.json",
    "science": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/reasoning_test_for_test/science.json"
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
    # test_task : Optional[str]
    test_checkpoint_sequence : Optional[int]
    conv_mode: str = "vicuna_v1"
    num_beams: int = 1
    max_new_tokens: int = 4096
    start_task_id : int = 0
    # sequence_test : Optional[bool]


    

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
def evaluate(args, tokenizer, model, task_id, task_val, output_dir):
    #此处task_id是任务在序列中顺序+1，task_val是任务的名称
    rank0_print(f"Testing on {task_val}")
    answers_file = os.path.join(output_dir, "predictions_test", f"{task_id}-{task_val}.jsonl")
    os.makedirs(os.path.dirname(answers_file),exist_ok=True)
    print(f"这是第{task_id}个任务的推理，任务的名称是:{task_val}")
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
            answers_chunk_file = os.path.join(output_dir, "predictions_test", f"{task_id}-{task_val}",
                                                f"{num_chunks}-{chunk_idx}.jsonl")
            questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
            questions = get_chunk(questions, num_chunks, chunk_idx)
            data_loader = create_data_loader(questions, image_folder, tokenizer, image_processor, model.config,
                                                args.conv_mode)
            answers_chunk_file = os.path.expanduser(answers_chunk_file)
            os.makedirs(os.path.dirname(answers_chunk_file), exist_ok=True)
            ans_file = open(answers_chunk_file, "w")

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
                        temperature=1.0,
                        top_p=1.0,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        min_new_tokens=1,
                        use_cache=False,
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
                ans_file = open(answers_file, "w")
                for i in range(num_chunks):
                    for q in open(
                            os.path.join(output_dir, "predictions_test", f"{task_id}-{task_val}", f"{num_chunks}-{i}.jsonl")):
                        ans_file.write(q)
                        ans_file.flush()
                ans_file.close()

    else:
        print( f"推理答案{answers_file}已存在")

    if local_rank == 0:
        if os.path.isdir(os.path.join(output_dir, "predictions_test", f"{task_id}-{task_val}")):
            shutil.rmtree(os.path.join(output_dir, "predictions_test", f"{task_id}-{task_val}"))
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
            acc = evaluate_object.eval_results(annotation_file_dict[task_val], answers_file,evaluation_results_file,is_reasoning=True)
        else:
            raise ValueError(f"Unknown task : {task_val}")

    dist.barrier()

    model.config.tokenizer_padding_side = "right"
    tokenizer.padding_side = "right"
    print( f"任务{task_val}推理结果{answers_file}的正确率是{acc}")
    return acc

def evaluate_one_checkpoint_one_task(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, TestingArguments))
    print("参数加载完毕")
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
    start_task_id = testing_args.start_task_id

    model_args.model_name_or_path = os.path.join(training_args.output_dir, str(start_task_id))

    if model_args.vision_tower is not None:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
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
    elif training_args.cl == "eproj":
        cl_learner = EprojLearner(LLaVATrainer,
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

    training_args.output_dir_checkpoint = os.path.join(output_dir, str(testing_args.test_checkpoint_sequence))

    for task_id, data_module in enumerate(data_modules):
        task_val = tasks[task_id]
        task_id += 1
        # if not sequence_test:
        # if not task_val == testing_args.test_task:
        #     rank0_print(f"task_val:{task_val}\n")
        #     rank0_print(f"test_task:{testing_args.test_task}\n")
        #     continue

        
        # cl_learner.before_train(task_id, model, tokenizer, **data_module)
        training_args.output_dir = os.path.join(output_dir, str(task_id))
        task = list_of_ints(data_args.tasks)[task_id - 1]

        # Train task
        # if task_id > start_task_id:
        rank0_print(f"Start inference on task {task_id}: {task}")
        release_memory()

        # trainer = LLaVATrainer(model=model,
        #                     tokenizer=tokenizer,
        #                     args=training_args,
        #                     cl_learner=cl_learner,
        #                     **data_module)
        rank0_print(f"此时任务{task_val}开始：")
        model.config.use_cache = True
        # if os.path.isdir(model_args.model_name_or_path):
        #     rank0_print(f"Loading from checkpoint: {model_args.model_name_or_path}")
        #     trainer.model = trainer.model.from_pretrained(model_args.model_name_or_path)

        # del trainer

        release_memory()

        # cl_learner.after_train(task_id, model, tokenizer, **data_module)

        rank0_print(f"Start inference after task {task_id}: {task}")
        evaluate(testing_args, tokenizer, model, task_id, task_val, output_dir)

import torch.distributed as dist

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

if __name__ == "__main__":
    evaluate_one_checkpoint_one_task(attn_implementation="flash_attention_2")
