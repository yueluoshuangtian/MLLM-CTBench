import os
import gc
import copy
import glob
import torch
import torch.distributed as dist

from functools import partial

from transformers.trainer import unwrap_model
from transformers.models.instructblip import InstructBlipForConditionalGeneration

from .base import BaseCLearner
from llava.model import LlavaLlamaForCausalLM
from llava.model.multimodal_projector.builder import ExpandableProjection as LLaVA_Eproj
from instruct_blip.model.eproj import ExpandableProjection as Blip_Eproj

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def release_memory():
    gc.collect()
    torch.cuda.empty_cache()
    memory_stats()


def memory_stats():
    rank0_print(f"memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2}")
    rank0_print(f"memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2}")


def compute_pull_loss(self, model, inputs, return_outputs=False):
    if isinstance(unwrap_model(model), LlavaLlamaForCausalLM):
        images = inputs["images"]
        convs = inputs["convs"]
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            model.encode_images(concat_images, convs)
        else:
            model.encode_images(images, convs)
    elif isinstance(unwrap_model(model), InstructBlipForConditionalGeneration):
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if "qformer_attention_mask" not in inputs:
            qformer_attention_mask = torch.ones_like(inputs["qformer_input_ids"])
        else:
            qformer_attention_mask = inputs["qformer_attention_mask"]
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = model.qformer(
            input_ids=inputs["qformer_input_ids"],
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        query_output = query_outputs[0][:, : query_tokens.size(1), :]

        model.e_language_projection(query_output, image_embeds[:, 0, :], inputs["prompts"])
    else:
        raise NotImplementedError(f"Unsupported model {type(model)}")

    loss = self.lbd * get_ep(unwrap_model(model)).loss
    return (loss, None) if return_outputs else loss


class EprojLearner(BaseCLearner):
    num_seen_tasks = 0

    def __init__(self, trainer_cls, lbd, num_experts, key_steps, rank, output_dir, training_args, model):
        global local_rank
        local_rank = rank

        self.lbd = lbd
        self.num_experts = num_experts
        self.device = torch.device("cuda", rank)
        self.output_dir = output_dir

        self.training_args = copy.deepcopy(training_args)
        self.training_args.max_steps = key_steps
        self.trainer_cls = trainer_cls

        self.initialize_model(model, num_experts)
        # only tune eproj key and experts
        model.requires_grad_(False)
        for n, p in get_ep(model).named_parameters():
            if "task_encoder" not in n:
                p.requires_grad = True

        self._start_task_id = None
        self.load_model(model)
    #用于计算并返回当前训练任务的起始任务 ID。如果任务已经从某个 ID 开始训练，它会返回任务的最大 ID。
    # 如果没有找到已存在的任务，则返回 0（表示从头开始训练）
    @property
    def start_task_id(self):
        if self._start_task_id is None:
            start_task_id = 0
            for d in glob.glob(os.path.join(self.output_dir, "*")):
                if os.path.isdir(d) \
                        and os.path.basename(d).isdigit() \
                        and os.path.isfile(os.path.join(d, "mm_projector.bin")):
                    start_task_id = max(int(os.path.basename(d)), start_task_id)
            self._start_task_id = start_task_id
        return self._start_task_id

    def load_model(self, model):
        if self.start_task_id > 0:
            model_save_file = os.path.join(self.output_dir, str(self.start_task_id), "mm_projector.bin")
            rank0_print(f"Loading eproj weights from {model_save_file} ...")
            #加载自己训练的映射层的参数
            eproj_weights = torch.load(model_save_file, map_location='cpu')
            #从原模型中寻找其对应映射层的名称
            keyword = get_keyword(model)
            #根据映射层名称找到
            def get_w(weights):
                return {k.split(keyword + '.')[1]: v
                        for k, v in weights.items() if keyword in k}

            incompatible_keys = get_ep(model).load_state_dict(get_w(eproj_weights), strict=False)
            rank0_print(incompatible_keys)
    #该方法用于加载已经保存的专家投影权重（mm_projector.bin）。
    # 如果 start_task_id 大于 0，表示有之前保存的训练结果，需要加载这些权重进行继续训练。
    @staticmethod
    def initialize_model(model, num_experts=None):
        if get_ep(model) is None:
            if isinstance(model, LlavaLlamaForCausalLM):
                model = model.get_model()
                if not hasattr(model.config, "num_experts"):
                    assert num_experts is not None
                    model.config.num_experts = num_experts
                model.e_mm_projector = LLaVA_Eproj(model.config)
                model.e_mm_projector.copy_from_proj(model.mm_projector)
                model.e_mm_projector.to(device=model.mm_projector[0].weight.device, dtype=model.mm_projector[0].weight.dtype)
            elif isinstance(model, InstructBlipForConditionalGeneration):
                if not hasattr(model.config, "num_experts"):
                    assert num_experts is not None
                    model.config.num_experts = num_experts
                model.e_language_projection = Blip_Eproj(model.config)
                model.e_language_projection.copy_from_proj(model.language_projection)
                model.e_language_projection.to(model.language_projection.weight.device, dtype=model.language_projection.weight.dtype)
            else:
                raise RuntimeError(f"Unexpected model type {type(model)}")
    #在训练开始前设置任务ID，并确保只有当前任务的专家模型参数需要更新。
    def before_train(self, task_id, model, tokenizer=None, train_dataset=None, data_collator=None, **kwargs):
        if self.start_task_id == 0 or task_id > self.start_task_id:
            rank0_print(f"Set eproj task id to {self.num_seen_tasks}")
            get_ep(model).set_task_id(self.num_seen_tasks)

            # only train the current expert
            get_ep(model).experts.requires_grad_(False)
            get_ep(model).experts[self.num_seen_tasks].requires_grad_(True)

            self.tune_task_key_only(model, tokenizer, train_dataset, data_collator)

        self.num_seen_tasks += 1
    #使用 trainer_cls（即训练器类）来进行训练，并使用部分应用的 compute_pull_loss 函数来计算损失。
    def tune_task_key_only(self, model, tokenizer, train_dataset, data_collator):
        release_memory()
        trainer = self.trainer_cls(cl_learner=None,
                                   train_dataset=train_dataset,
                                   data_collator=data_collator,
                                   args=self.training_args,
                                   model=model,
                                   tokenizer=tokenizer)
        trainer.lbd = self.lbd
        trainer.compute_loss = partial(compute_pull_loss, trainer)
        trainer.train()
        del trainer
        release_memory()

#根据不同的模型类型返回对应的专家投影层。
def get_ep(model):
    if isinstance(model, LlavaLlamaForCausalLM):
        return getattr(model.get_model(), "e_mm_projector", None)
    elif isinstance(model, InstructBlipForConditionalGeneration):
        return getattr(model, "e_language_projection", None)
    else:
        raise RuntimeError(f"Unexpected model type {type(model)}")

#返回与模型类型对应的关键词，用于加载专家投影层的权重。
def get_keyword(model):
    if isinstance(model, LlavaLlamaForCausalLM):
        return "e_mm_projector"
    elif isinstance(model, InstructBlipForConditionalGeneration):
        return "e_language_projection"
    else:
        raise RuntimeError(f"Unexpected model type {type(model)}")