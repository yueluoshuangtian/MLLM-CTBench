"""
Base interface for continual-learning learners on Qwen2.5-VL + PEFT-LoRA.

Hooks called by `CustomSFTTrainer` (sft.py):
  - enable_record(bool)
  - post_backward(model, cl_task_progress=None)
And by the main loop (sft.py main()):
  - before_train(task_id, model, ...)
  - after_train(task_id, model, ...)
  - loss(loss, model, inputs, outputs, **kwargs) -> loss
  - finalize(model)

`task_id` is 0-based: task 1 in shell ('1-sft') => task_id=0.
"""

import os
import gc
import torch
import torch.distributed as dist


def _is_rank0():
    lr = int(os.environ.get("LOCAL_RANK", "-1"))
    return lr in (-1, 0)


def rank0_print(*args, **kwargs):
    if _is_rank0():
        print(*args, **kwargs)


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def barrier():
    if dist.is_initialized():
        dist.barrier()


def is_lora_param(name: str) -> bool:
    return ("lora_A" in name) or ("lora_B" in name)


def select_trainable_lora_params(model):
    try:
        from transformers.modeling_utils import unwrap_model
        base = unwrap_model(model)
    except Exception:
        base = model
    for n, p in base.named_parameters():
        if not p.requires_grad:
            continue
        if not is_lora_param(n):
            continue
        yield n, p


class BaseCLearner:
    """Default no-op learner = pure sequential SFT."""

    name = "none"

    def __init__(self, *args, **kwargs):
        pass

    # lifecycle
    def before_train(self, task_id, model, tokenizer=None, **kwargs):
        pass

    def after_train(self, task_id, model, tokenizer=None, **kwargs):
        pass

    def finalize(self, model, **kwargs):
        pass

    # per-step
    def enable_record(self, on: bool):
        pass

    def post_backward(self, model, cl_task_progress=None, **kwargs):
        pass

    def loss(self, loss, model=None, inputs=None, outputs=None, **kwargs):
        return loss
