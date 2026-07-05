# Copyright 2025 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
SFT script with switchable PEFT method:
  - --peft_method lora | moelora
  - Supports your custom CoIN-MoE-LoRA (CoIN.peft.tuners.coinmoelora)

Adds "small tricks":
  - mm_projector_lr param group (NOTE: if mm_projector params are frozen, this won't matter; user said ignore)
  - GateTauSchedulerCallback (gate_tau annealing)  ✅ FIXED: do not pre-compute end_step from trainer.state before train
  - Save adapter weights into: <output_dir>/<task_k>/adapter_model.(safetensors|bin)
  - Optional MoE-LoRA regularization:
      * orth regularization on LoRA-A rank directions (row-wise orth)
      * L2 regularization on LoRA-B

Key fixes in this version:
  1) GateTauScheduler end_step computation bug: now let callback self-init from state.max_steps.
  2) create_model_card dataset_name bug: no more list(script_args.dataset_name).
  3) Adapter naming: support task_k by parsing from output_dir "<k>-sft" OR env TASK_K OR --task_k.
     - default remains task_1 for backward compatibility.
  4) Continue training: keep your --continue_lora_path behavior unchanged.
  5) ✅ Integrate continual-learning regularizer via cl_learner/moelora.py
     - enable_record(True/False) around forward
     - post_backward() after backward AND only on sync_gradients step
     - before_train/after_train called per task (per trainer.train())

NEW in this version:
  6) ✅ Per-task train set ratio subsampling:
     - --train_ratio (e.g., 0.4)
     - --train_ratio_strategy random|first|end
     - deterministic when using --data_seed (preferred) or --seed

NEW in this version (your request):
  7) ✅ In-process inference right after training (NO extra eval.py):
     - after each step: ROOT/predictions/<k>-<task>.json
     - if last step: ROOT/last_predictions/<i>-<task_i>.json for i < k
     - with tqdm progress + periodic prints, and NO barrier waits (avoid “卡住”)
"""

import logging
import os
import sys
import math
import random
import json
import yaml
import re
import gc

# ✅ Compat shim: flash-attn 2.8+ uses torch.library.wrap_triton (torch >= 2.6).
# On torch 2.5 we polyfill it as a passthrough so flash_attn's triton kernels
# launch via plain `kernel[grid](...)`. Must run before any flash_attn import.
import torch as _torch_for_shim
if not hasattr(_torch_for_shim.library, "wrap_triton"):
    def _wrap_triton_passthrough(kernel):
        return kernel
    _torch_for_shim.library.wrap_triton = _wrap_triton_passthrough
del _torch_for_shim
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

from tqdm import tqdm

import datasets
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
from transformers import TrainerCallback

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from qwen_vl_utils import process_vision_info


logger = logging.getLogger(__name__)

# ----------------------------
# Task order + test file mapping (same as you provided)
# ----------------------------
ORDER_TO_TASKS: Dict[str, List[str]] = {
    "order1": ["numglue", "art", "math", "fomc", "medical", "OCR", "science"],
    "order2": ["science", "OCR", "medical", "fomc", "math", "art", "numglue"],
    # order3 redefined to MLLM-CTBench paper's difficulty curriculum
    # (matches TMM/CLMM/PROGRESS.md). Original CoIN 8-task order3 was moved to
    # order_coin8 below.
    "order3": ["art", "OCR", "fomc", "science", "numglue", "math", "medical"],
    "order4": ["ScienceQA", "ImageNet", "VizWiz", "Grounding", "TextVQA", "GQA", "VQAv2", "OCRVQA"],
    "order5": ["fomc", "art", "math", "science", "numglue", "OCR", "medical"],
    "order6": ["science", "medical", "fomc", "art", "math", "numglue", "OCR"],
    "order_coin8": ["ScienceQA", "TextVQA", "ImageNet", "GQA", "VizWiz", "Grounding", "VQAv2", "OCRVQA"],
}

DATA_HAS_VISION: Dict[str, bool] = {
    "numglue": False,
    "fomc": False,
    "art": True,
    "math": True,
    "medical": True,
    "OCR": True,
    "science": True,
    "ScienceQA": False,
    "TextVQA": True,
    "ImageNet": True,
    "GQA": True,
    "VizWiz": True,
    "Grounding": True,
    "VQAv2": True,
    "OCRVQA": True,
}

# Paths overridable via env vars QWEN_TEST_ROOT / QWEN_IMAGE_ROOT (default to the
# /mnt/cxzx layout used by this project).
_TEST_ROOT_DEFAULT = "/mnt/cxzx/share/workspace/data_transfer/houzhiyan/TMM/qwen data/test"
_IMG_ROOT_DEFAULT = "/mnt/cxzx/share/workspace/data_transfer/houzhiyan/clmm-benchmark"
_TEST_ROOT = os.environ.get("QWEN_TEST_ROOT", _TEST_ROOT_DEFAULT)
_IMG_ROOT = os.environ.get("QWEN_IMAGE_ROOT", _IMG_ROOT_DEFAULT)

QUESTION_FILE: Dict[str, str] = {
    "numglue": f"{_TEST_ROOT}/numglue.json",
    "art": f"{_TEST_ROOT}/art.json",
    "math": f"{_TEST_ROOT}/math.json",
    "fomc": f"{_TEST_ROOT}/fomc.json",
    "medical": f"{_TEST_ROOT}/medical.json",
    "OCR": f"{_TEST_ROOT}/OCR.json",
    "science": f"{_TEST_ROOT}/science.json",
    "GQA": "/home/houzhiyan/dataset/llava/train/GQA/test.json",
    "Grounding": "/home/houzhiyan/dataset/llava/train/Grounding/test.json",
    "ImageNet": "/home/houzhiyan/dataset/llava/train/ImageNet/test.json",
    "OCRVQA": "/home/houzhiyan/dataset/llava/train/OCRVQA/test.json",
    "ScienceQA": "/home/houzhiyan/dataset/llava/train/ScienceQA/test.json",
    "TextVQA": "/home/houzhiyan/dataset/llava/train/TextVQA/test.json",
    "VizWiz": "/home/houzhiyan/dataset/llava/train/VizWiz/test.json",
    "VQAv2": "/home/houzhiyan/dataset/llava/train/VQAv2/test.json",
}

IMAGE_ROOT: Dict[str, str] = {
    "numglue": "",
    "fomc": "",
    "art": _IMG_ROOT,
    "math": _IMG_ROOT,
    "medical": _IMG_ROOT,
    "OCR": _IMG_ROOT,
    "science": _IMG_ROOT,
    "ScienceQA": "/home/houzhiyan/dataset/CoIN",
    "TextVQA": "/home/houzhiyan/dataset/CoIN",
    "ImageNet": "/home/houzhiyan/dataset/CoIN",
    "GQA": "/home/houzhiyan/dataset/CoIN",
    "VizWiz": "/home/houzhiyan/dataset/CoIN",
    "Grounding": "/home/houzhiyan/dataset/CoIN",
    "VQAv2": "/home/houzhiyan/dataset/CoIN",
    "OCRVQA": "/home/houzhiyan/dataset/CoIN",
}

# =========================
# In-process inference (rank0 only, with progress)
# =========================
def _is_rank0(training_args_, trainer: Optional[SFTTrainer] = None) -> bool:
    lr = int(getattr(training_args_, "local_rank", -1))
    ok = lr in (-1, 0)
    if trainer is not None:
        try:
            ok = ok and trainer.accelerator.is_main_process
        except Exception:
            pass
    return ok

def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def _release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _load_json_list(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _build_messages_for_task(
    data: List[dict],
    task: str,
    *,
    shuffle: bool,
    seed: int,
) -> List[List[dict]]:
    """
    Build Qwen2.5-VL chat messages from test samples.
    Supports keys: problem/solution, or text/answer, optional image.
    """
    if shuffle:
        rnd = random.Random(int(seed))
        rnd.shuffle(data)

    has_vision = bool(DATA_HAS_VISION.get(task, False))
    img_root = str(IMAGE_ROOT.get(task, ""))

    template = "{Question}"
    msgs: List[List[dict]] = []

    for ex in data:
        problem = ex.get("problem") or ex.get("text") or ex.get("question")
        if problem is None:
            raise KeyError(f"[INFER] missing problem/text/question in sample keys={list(ex.keys())}")

        if has_vision and ex.get("image"):
            img_path = os.path.join(img_root, ex["image"]) if img_root else ex["image"]
            msg = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": template.format(Question=problem)},
                ],
            }]
        else:
            msg = [{
                "role": "user",
                "content": [{"type": "text", "text": template.format(Question=problem)}],
            }]
        msgs.append(msg)

    return msgs

def _save_infer_results(data: List[dict], outputs: List[str], out_path: str):
    final = []
    for ex, out in zip(data, outputs):
        final.append({
            "question_id": ex.get("question_id", None),
            "question": ex.get("problem") or ex.get("text") or ex.get("question"),
            "ground_truth": ex.get("solution") or ex.get("answer"),
            "model_output": out,
            "image": ex.get("image", None),
        })
    _safe_mkdir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": final}, f, ensure_ascii=False, indent=2)
    print(f"[INFER][OK] Saved: {out_path}")

@torch.inference_mode()
def _run_inference_one_task_with_progress(
    *,
    model: torch.nn.Module,
    processor_: Any,
    task: str,
    bs: int,
    max_new_tokens: int,
    sample_num: Optional[int],
    shuffle: bool,
    seed: int,
    device: torch.device,
    log_every_batches: int = 20,
) -> Tuple[List[dict], List[str]]:
    """Distributed inference across all ranks. Each rank processes a strided
    slice (data[rank::world_size]); results are all_gathered to rank 0, which
    reassembles and returns them in original order. Other ranks return empty
    lists (and main() should only save from rank 0).
    """
    data_path = QUESTION_FILE[task]
    full_data = _load_json_list(data_path)

    if sample_num is not None and sample_num > 0:
        full_data = full_data[:int(sample_num)]
    if not full_data:
        raise RuntimeError(f"[INFER] No data for task={task} at {data_path}")

    # Distributed setup
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size, rank = 1, 0

    # Strided slice: rank R gets indices [R, R+W, R+2W, ...]
    local_indices = list(range(rank, len(full_data), world_size))
    local_data = [full_data[i] for i in local_indices]

    if not local_data:
        # No work for this rank, but still participate in gather
        local_results = []  # type: List[Tuple[int, str]]
    else:
        messages = _build_messages_for_task(local_data, task, shuffle=shuffle, seed=seed)

        if hasattr(processor_, "tokenizer") and processor_.tokenizer is not None:
            processor_.tokenizer.padding_side = "left"
            if processor_.tokenizer.pad_token is None and processor_.tokenizer.eos_token is not None:
                processor_.tokenizer.pad_token = processor_.tokenizer.eos_token

        model.eval()
        local_outs: List[str] = []
        total_batches = (len(messages) + bs - 1) // bs

        # Only rank 0 prints progress bar to avoid 8x duplicate output
        iterator = range(0, len(messages), bs)
        if rank == 0:
            iterator = tqdm(iterator, total=total_batches, desc=f"[INFER][{task}]", dynamic_ncols=True, leave=True)

        has_vision = bool(DATA_HAS_VISION.get(task, False))

        for bidx, i in enumerate(iterator, start=1):
            batch_messages = messages[i:i + bs]
            texts = [
                processor_.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in batch_messages
            ]

            if has_vision:
                image_inputs, video_inputs = process_vision_info(batch_messages)
            else:
                image_inputs, video_inputs = None, None

            inputs = processor_(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

            gen_ids = model.generate(
                **inputs,
                do_sample=False,
                use_cache=True,
                max_new_tokens=int(max_new_tokens),
            )

            in_ids = inputs["input_ids"]
            trimmed = [out[len(inp):] for inp, out in zip(in_ids, gen_ids)]
            batch_text = processor_.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            local_outs.extend(batch_text)

            if rank == 0 and log_every_batches > 0 and (bidx % log_every_batches == 0):
                print(f"[INFER][{task}] rank0 {bidx}/{total_batches} batches done (×{world_size} ranks)")

        local_results = list(zip(local_indices, local_outs))

    # Gather all ranks' (global_idx, output) tuples
    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_results)
    else:
        gathered = [local_results]

    if rank != 0:
        return [], []  # non-zero ranks return empty; main() only saves from rank 0

    # Reassemble in original order
    merged = {}
    for shard in gathered:
        if shard is None:
            continue
        for global_idx, out in shard:
            merged[global_idx] = out
    ordered_outs = [merged.get(i, "") for i in range(len(full_data))]
    if rank == 0:
        print(f"[INFER][{task}] gathered {sum(1 for o in ordered_outs if o)}/{len(full_data)} outputs from {world_size} ranks")

    return full_data, ordered_outs

def maybe_run_post_and_last_infer(
    trainer: "CustomSFTTrainer",
    model: torch.nn.Module,
    processor_: Any,
    script_args_: "SFTScriptArguments",
    training_args_,
    task_k_1based: int,
):
    """
    Your requirement (in this script, right after training + saving, model still alive):
      1) After each task k: run inference ONLY on current task (script_args.cur_task),
         save to ROOT/predictions/<k>-<task>.json
      2) If this is the last task of the given order: run inference using the current (last) model
         on all previous tasks i < k, save to ROOT/last_predictions/<i>-<task_i>.json
         (skip last task to avoid duplication)
    Notes:
      - rank0 only
      - NO dist barrier to avoid hanging (your “一直卡住” issue)
      - tqdm + periodic prints are enabled
    """
    if not getattr(script_args_, "do_infer", True):
        return
    # NOTE: all ranks participate in inference (data is strided per rank).
    # Saving is rank0-only and is enforced inside _save_infer_results call paths.

    cur_task = getattr(script_args_, "cur_task", None)
    if not cur_task:
        if _is_rank0(training_args_, trainer):
            print("[INFER][SKIP] script_args.cur_task is None")
        return

    # ROOT is parent of "<k>-sft"
    out_dir = str(getattr(training_args_, "output_dir", "") or "")
    root_dir = os.path.dirname(out_dir.rstrip("/"))
    pred_dir = os.path.join(root_dir, "predictions")
    last_dir = os.path.join(root_dir, "last_predictions")
    _safe_mkdir(pred_dir)
    _safe_mkdir(last_dir)

    bs = int(getattr(script_args_, "infer_bs", 16))
    max_new_tokens = int(getattr(script_args_, "infer_max_new_tokens", 256))
    sample_num_raw = int(getattr(script_args_, "infer_sample_num", 0) or 0)
    sample_num = None if sample_num_raw <= 0 else sample_num_raw
    shuffle = bool(getattr(script_args_, "infer_shuffle", False))
    seed = int(getattr(script_args_, "infer_seed", 0))

    # Use local device on rank0
    device = torch.device(getattr(training_args_, "device", "cuda") if torch.cuda.is_available() else "cpu")

    is_main = _is_rank0(training_args_, trainer)

    # (A) POST: current task only (distributed across ranks, rank0 saves)
    try:
        out_path = os.path.join(pred_dir, f"{task_k_1based}-{cur_task}.json")
        if is_main:
            print(f"[INFER][POST] k={task_k_1based} task={cur_task} -> {out_path}")
        data, outs = _run_inference_one_task_with_progress(
            model=model,
            processor_=processor_,
            task=cur_task,
            bs=bs,
            max_new_tokens=max_new_tokens,
            sample_num=sample_num,
            shuffle=shuffle,
            seed=seed,
            device=device,
            log_every_batches=20,
        )
        if is_main:
            _save_infer_results(data, outs, out_path)
    except Exception as e:
        if is_main:
            print(f"[INFER][POST][FAILED] {repr(e)}")

    _release_memory()

    # (B) LAST: only if order provided and we are the last task
    order = getattr(script_args_, "order", None)
    if not order or order not in ORDER_TO_TASKS:
        if is_main:
            print("[INFER][LAST][SKIP] script_args.order not provided or unknown; skip last_predictions.")
        return

    tasks = ORDER_TO_TASKS[order]
    if task_k_1based != len(tasks):
        if is_main:
            print(f"[INFER][LAST][SKIP] not last task: k={task_k_1based}, total={len(tasks)}")
        return

    if is_main:
        print(f"[INFER][LAST] last task reached: order={order}, total_tasks={len(tasks)}")

    # Evaluate previous tasks only: 1..N-1 (distributed, rank0 saves)
    for i, t in enumerate(tasks, start=1):
        if i == len(tasks):
            continue  # skip last task: already saved in POST

        try:
            out_path = os.path.join(last_dir, f"{i}-{t}.json")
            if is_main:
                print(f"[INFER][LAST] eval task={t} with last-adapter -> {out_path}")
            data, outs = _run_inference_one_task_with_progress(
                model=model,
                processor_=processor_,
                task=t,
                bs=bs,
                max_new_tokens=max_new_tokens,
                sample_num=sample_num,
                shuffle=shuffle,
                seed=seed,
                device=device,
                log_every_batches=20,
            )
            if is_main:
                _save_infer_results(data, outs, out_path)
        except Exception as e:
            if is_main:
                print(f"[INFER][LAST][FAILED] task={t} err={repr(e)}")
        finally:
            _release_memory()


# =========================
# Utils: deterministic subsample
# =========================
def _deterministic_subsample(
    data_list: List[Any],
    ratio: float,
    strategy: str,
    seed: int,
) -> List[Any]:
    if ratio is None:
        return data_list
    ratio = float(ratio)
    if ratio >= 1.0:
        return data_list
    if ratio <= 0.0:
        return []

    n = len(data_list)
    k = int(math.ceil(ratio * n))
    k = max(1, min(k, n))

    strategy = (strategy or "random").lower()
    if strategy == "first":
        return data_list[:k]
    if strategy == "end":
        return data_list[-k:]
    if strategy == "random":
        rng = random.Random(int(seed))
        idx = list(range(n))
        rng.shuffle(idx)
        idx = idx[:k]
        return [data_list[i] for i in idx]

    raise ValueError(f"Unknown train_ratio_strategy={strategy}. Use random|first|end.")


# =========================
# Utils: load json/jsonl
# =========================
def load_json_datas(
    json_path: str,
    sampling_strategy: str,
    sampling_number: Optional[int],
    *,
    train_ratio: float = 1.0,
    train_ratio_strategy: str = "random",
    train_ratio_seed: int = 42,
):
    if json_path.endswith(".jsonl"):
        cur_data_dict = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cur_data_dict.append(json.loads(line))
    elif json_path.endswith(".json"):
        with open(json_path, "r", encoding="utf-8") as f:
            cur_data_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported file type: {json_path}")

    if ":" in (sampling_strategy or ""):
        sampling_strategy, sampling_number = sampling_strategy.split(":")
        if "%" in sampling_number:
            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
        else:
            sampling_number = int(sampling_number)

    if sampling_strategy == "first" and sampling_number is not None:
        cur_data_dict = cur_data_dict[:sampling_number]
    elif sampling_strategy == "end" and sampling_number is not None:
        cur_data_dict = cur_data_dict[-sampling_number:]
    elif sampling_strategy == "random" and sampling_number is not None:
        rng = random.Random(int(train_ratio_seed))
        rng.shuffle(cur_data_dict)
        cur_data_dict = cur_data_dict[:sampling_number]

    cur_data_dict = _deterministic_subsample(
        cur_data_dict,
        ratio=float(train_ratio),
        strategy=str(train_ratio_strategy),
        seed=int(train_ratio_seed),
    )

    print(
        f"Loaded {len(cur_data_dict)} samples from {json_path} "
        f"(train_ratio={train_ratio}, strategy={train_ratio_strategy}, seed={train_ratio_seed})"
    )
    return cur_data_dict


# =========================
# Args
# =========================
@dataclass
class SFTScriptArguments(ScriptArguments):
    image_root: str = field(default=None, metadata={"help": "The root directory of the image."})
    sampling_strategy: str = field(default="all", metadata={"help": "加载数据时采样的策略."})
    cur_task: str = field(default=None, metadata={"help": "任务名称."})

    task_k: Optional[int] = field(
        default=None,
        metadata={"help": "Task index k (optional). If None, will infer from output_dir like '<k>-sft' or env TASK_K."},
    )

    train_ratio: float = field(
        default=1.0,
        metadata={"help": "Use only a ratio of the current task training set (e.g., 0.4)."},
    )
    train_ratio_strategy: str = field(
        default="random",
        metadata={"help": "Subsample strategy for train_ratio: random|first|end."},
    )

    # ✅ NEW: order for deciding last-task behavior
    order: Optional[str] = field(
        default=None,
        metadata={"help": "Task order name (order1..order6). Used for last_predictions after final task."},
    )

    # ✅ NEW: inference knobs (in-process)
    do_infer: bool = field(default=True, metadata={"help": "Run inference right after training in the same process."})
    infer_bs: int = field(default=16, metadata={"help": "Inference batch size."})
    infer_max_new_tokens: int = field(default=256, metadata={"help": "Inference max_new_tokens."})
    infer_sample_num: int = field(default=0, metadata={"help": "0 means all test samples; otherwise take first N."})
    infer_shuffle: bool = field(default=False, metadata={"help": "Shuffle test set before inference."})
    infer_seed: int = field(default=0, metadata={"help": "Shuffle seed for inference."})

    # ✅ NEW: unified CL method dispatch (replaces hardcoded moelora-only path)
    cl_method: str = field(default="none", metadata={"help": "none|ewc|mas|lwf|freeze_init|freeze_last|freeze|replay|der|l2p|max_merge|moelora"})
    cl_lbd: float = field(default=1.0, metadata={"help": "Regularization weight for EWC/MAS/LwF/DER (interpretation varies per method)."})
    replay_paths: str = field(default="", metadata={"help": "Colon-separated train json paths of prior tasks (for ER/DER replay buffer)."})
    replay_ratio: float = field(default=0.0, metadata={"help": "Fraction of each prior dataset to mix into current task (ER)."})
    l2p_prompt_len: int = field(default=8, metadata={"help": "L2P per-task prefix prompt length."})
    is_last_task: bool = field(default=False, metadata={"help": "If True, run cl_learner.finalize() after this task (for MagMaX merge)."})


@dataclass
class SFTModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

    peft_method: str = field(default="lora", metadata={"help": "lora | moelora"})

    lora_enable: bool = field(default=True, metadata={"help": "Enable PEFT LoRA/MoE-LoRA"})
    lora_r: int = field(default=128, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=256, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_bias: str = field(default="none", metadata={"help": "LoRA bias: none|all|lora_only"})

    mm_projector_lr: float = field(default=2e-5, metadata={"help": "LR for mm_projector params (if exist)"})

    expert_num: int = field(default=2, metadata={"help": "MoE-LoRA expert_num"})
    gate_rank: int = field(default=4, metadata={"help": "MoE-LoRA gate_rank"})
    gate_tau: float = field(default=1.6, metadata={"help": "MoE-LoRA gate_tau (temperature)"})
    gate_tie: str = field(default="A_T", metadata={"help": "MoE-LoRA gate_tie: A_T|free"})
    gate_router: str = field(default="a", metadata={"help": "MoE-LoRA gate_router: qr|a|mlp|a_cos"})

    moelora_reg_enable: bool = field(default=False, metadata={"help": "Enable MoE-LoRA extra reg"})
    moelora_orth_lambda: float = field(default=0.0, metadata={"help": "Orth reg weight on LoRA-A rank dirs"})
    moelora_B_reg_scale: float = field(default=0.0, metadata={"help": "L2 reg scale for LoRA-B"})
    moelora_importance_top_p: float = field(default=1.0, metadata={"help": "Top-p for importance (reserved)"})

    continue_lora_path: str = field(default=None, metadata={"help": "Path to previous adapter dir to continue (optional)"})

    gate_tau_end: float = field(default=0.7, metadata={"help": "End tau for annealing (moelora only)"})
    gate_tau_mode: str = field(default="cosine", metadata={"help": "linear|cosine"})


processor = None  # global processor
script_args = None
training_args = None


# =========================
# Dataset
# =========================
class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args_: ScriptArguments, train_ratio_seed: int):
        super().__init__()
        self.script_args = script_args_
        self.list_data_dict = []
        self.train_ratio_seed = int(train_ratio_seed)

        ratio = float(getattr(script_args_, "train_ratio", 1.0))
        ratio_strategy = str(getattr(script_args_, "train_ratio_strategy", "random"))

        if data_path.endswith(".yaml"):
            with open(data_path, "r", encoding="utf-8") as file:
                yaml_data = yaml.safe_load(file)
                ds_list = yaml_data.get("datasets")
                for data in ds_list:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    cur_data_dict = load_json_datas(
                        json_path,
                        sampling_strategy,
                        sampling_number,
                        train_ratio=ratio,
                        train_ratio_strategy=ratio_strategy,
                        train_ratio_seed=self.train_ratio_seed,
                    )
                    self.list_data_dict.extend(cur_data_dict)

        elif data_path.endswith(".json") or data_path.endswith(".jsonl"):
            json_path = data_path
            sampling_strategy = "all" if script_args_.sampling_strategy is None else script_args_.sampling_strategy
            sampling_number = None

            cur_data_dict = load_json_datas(
                json_path,
                sampling_strategy,
                sampling_number,
                train_ratio=ratio,
                train_ratio_strategy=ratio_strategy,
                train_ratio_seed=self.train_ratio_seed,
            )
            self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        def make_conversation(example):
            QUESTION_TEMPLATE = "{Question}"
            return [
                {"role": "user", "content": [{"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])}]},
                {"role": "assistant", "content": example["solution"]},
            ]

        def make_conversation_image(example):
            QUESTION_TEMPLATE = "{Question}"
            image_root = self.script_args.image_root
            image_path = os.path.join(image_root, example["image"])
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])} ,
                    ],
                },
                {"role": "assistant", "content": example["solution"]},
            ]

        example = self.list_data_dict[i]
        if "image" in example:
            example["messages"] = make_conversation_image(example)
        else:
            example["messages"] = make_conversation(example)
        return example


# =========================
# Collator
# =========================

def collate_fn(examples):
    # ✅ TRAIN: do NOT add generation prompt (samples already include assistant turns)
    texts = [
        processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        for ex in examples
    ]

    if DATA_HAS_VISION.get(script_args.cur_task, False):
        image_inputs = []
        for i, ex in enumerate(examples):
            try:
                imgs, vids = process_vision_info(ex["messages"])
            except Exception as e:
                print("\n[collate_fn] process_vision_info failed")
                print(f"  cur_task: {getattr(script_args, 'cur_task', None)}")
                print(f"  example_index_in_batch: {i}")
                print("  ex['messages']:")
                print(ex["messages"])
                print("  exception:")
                traceback.print_exc()
                raise
            image_inputs.append(imgs)
    else:
        image_inputs = None

    batch = processor(
        max_length=training_args.max_length,
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    b, L = batch["input_ids"].shape
    assert b == len(examples), "batch 第一维必须等于样本数"
    assert training_args.max_length is None or L <= training_args.max_length, "序列被截断后仍超长"

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    batch["labels"] = labels
    return batch




# =========================
# GateTauSchedulerCallback
# =========================
def _set_all_gate_tau(model, tau: float):
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
    def __init__(
        self,
        start_tau: float = 1.5,
        end_tau: float = 0.7,
        total_steps: int = None,
        mode: str = "cosine",
        start_step: int = 0,
        end_step: int = None,
        log_every_n_steps: int = 100,
    ):
        assert mode in ("linear", "cosine")
        self.start_tau = float(start_tau)
        self.end_tau = float(end_tau)
        self.total_steps = total_steps
        self.mode = mode
        self.start_step = int(start_step)
        self.end_step = end_step
        self.log_every_n_steps = int(log_every_n_steps)
        self._inited = False

    def _maybe_init(self, state):
        if self._inited:
            return
        if self.total_steps is None:
            self.total_steps = int(getattr(state, "max_steps", 0)) or None
        if self.end_step is None:
            self.end_step = self.total_steps if self.total_steps is not None else self.start_step
        self._inited = True

    def _compute_tau(self, step: int):
        if step <= self.start_step:
            return self.start_tau
        if step >= self.end_step:
            return self.end_tau
        span = max(1, self.end_step - self.start_step)
        progress = (step - self.start_step) / span
        if self.mode == "linear":
            tau = self.start_tau + (self.end_tau - self.start_tau) * progress
        else:
            tau = self.end_tau + 0.5 * (self.start_tau - self.end_tau) * (1.0 + math.cos(math.pi * progress))
        return tau

    def on_step_end(self, args, state, control, **kwargs):
        self._maybe_init(state)
        if self.total_steps is None:
            return control

        step = int(state.global_step)
        tau = self._compute_tau(step)
        model = kwargs.get("model", None)
        if model is not None:
            changed = _set_all_gate_tau(model, tau)
            if changed > 0 and (step % self.log_every_n_steps == 0) and (args.local_rank in (0, -1)):
                print(f"[GateTauScheduler] step={step} tau={tau:.4f} (updated {changed} modules)")
        return control


# =========================
# ZeRO3-safe-ish state dict helpers
# =========================
def get_peft_state_maybe_zero_3(named_parameters, bias: str, adapter_name: str) -> Dict[str, torch.Tensor]:
    state = {}
    for n, p in named_parameters:
        if p is None:
            continue
        if adapter_name not in n:
            continue
        if ("lora_" in n) or ("gate_" in n) or ("router_" in n) or ("router_vec" in n) or ("router_mlp" in n):
            try:
                state[n] = p.detach().cpu()
            except Exception:
                pass
        if bias != "none" and n.endswith(".bias") and adapter_name in n:
            try:
                state[n] = p.detach().cpu()
            except Exception:
                pass
    return state


def get_peft_state_non_lora_maybe_zero_3(named_parameters) -> Dict[str, torch.Tensor]:
    state = {}
    for n, p in named_parameters:
        if p is None:
            continue
        if p.requires_grad and ("lora_" not in n) and ("gate_" not in n) and ("router_" not in n):
            try:
                state[n] = p.detach().cpu()
            except Exception:
                pass
    return state


# =========================
# Save callback: save processor at checkpoint
# =========================
class SaveProcessorConfigCallback(TrainerCallback):
    def on_save(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if state.is_world_process_zero and processor is not None:
            global_step = state.global_step
            output_dir = f"{args.output_dir}/checkpoint-{global_step}"
            processor.save_pretrained(output_dir)
            logger.info("Processor config saved into checkpoint.")


# =========================
# Custom Trainer
# =========================
class CustomSFTTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        mm_projector_lr: float = None,
        moelora_reg_enable: bool = False,
        moelora_orth_lambda: float = 0.0,
        moelora_B_reg_scale: float = 0.0,
        peft_method: str = "lora",
        cl_learner=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._mm_projector_lr = mm_projector_lr
        self._moelora_reg_enable = moelora_reg_enable
        self._moelora_orth_lambda = float(moelora_orth_lambda or 0.0)
        self._moelora_B_reg_scale = float(moelora_B_reg_scale or 0.0)
        self._peft_method = peft_method
        self.cl_learner = cl_learner

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        lr = self.args.learning_rate
        wd = self.args.weight_decay
        mm_lr = self._mm_projector_lr if self._mm_projector_lr is not None else lr

        mm_params = []
        base_params = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "mm_projector" in n:
                mm_params.append(p)
            else:
                base_params.append(p)

        param_groups = []
        if len(base_params) > 0:
            param_groups.append({"params": base_params, "lr": lr, "weight_decay": wd})
        if len(mm_params) > 0:
            param_groups.append({"params": mm_params, "lr": mm_lr, "weight_decay": wd})

        optim_cls, optim_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optim_cls(param_groups, **optim_kwargs)
        return self.optimizer

    @staticmethod
    def _orth_reg_A(A: torch.Tensor, eps: float = 1e-6):
        A = A.to(torch.float32)
        A = torch.nn.functional.normalize(A, p=2, dim=1, eps=eps)
        G = A @ A.t()
        I = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
        return torch.mean((G - I) ** 2)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.cl_learner is not None:
            self.cl_learner.enable_record(True)

        try:
            out = model(**inputs)
            loss = out.loss

            # ✅ NEW: unified CL reg loss (EWC/MAS/LwF/DER add reg term here)
            if self.cl_learner is not None and hasattr(self.cl_learner, "loss"):
                try:
                    loss = self.cl_learner.loss(loss, model=model, inputs=inputs, outputs=out)
                except Exception as _e:
                    # Don't crash the run if reg loss has a bug; log and move on
                    if int(os.environ.get("LOCAL_RANK", "-1")) in (-1, 0):
                        print(f"[CL][WARN] cl_learner.loss raised: {repr(_e)}")

            if self._peft_method == "moelora" and self._moelora_reg_enable:
                reg = 0.0
                for m in model.modules():
                    if hasattr(m, "As_weight") and hasattr(m, "Bs_weight"):
                        if self._moelora_orth_lambda > 0:
                            try:
                                E = m._current_E()
                                for e in range(E):
                                    A_e = m.As_weight(e)
                                    reg = reg + self._moelora_orth_lambda * self._orth_reg_A(A_e)
                            except Exception:
                                pass
                        if self._moelora_B_reg_scale > 0:
                            try:
                                E = m._current_E()
                                for e in range(E):
                                    B_e = m.Bs_weight(e)
                                    reg = reg + self._moelora_B_reg_scale * (B_e.to(torch.float32).pow(2).mean())
                            except Exception:
                                pass
                loss = loss + reg

            return (loss, out) if return_outputs else loss
        finally:
            if self.cl_learner is not None:
                self.cl_learner.enable_record(False)

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        if isinstance(loss, (tuple, list)):
            loss = loss[0]

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        # Call post_backward EVERY backward step (not just sync boundaries).
        # EWC/MAS Ipt accumulation needs to fire each backward, since with
        # grad_accumulation_steps>1 only the boundary step has sync_gradients=True
        # which would skip 3/4 of our gradient samples.
        if self.cl_learner is not None:
            max_steps = int(getattr(self.state, "max_steps", 0)) or 0
            cur_step = int(getattr(self.state, "global_step", 0))
            progress = (cur_step / max_steps) if max_steps > 0 else None
            try:
                self.cl_learner.post_backward(model, cl_task_progress=progress)
            except Exception as _e:
                if int(os.environ.get("LOCAL_RANK", "-1")) in (-1, 0):
                    print(f"[CL][WARN] post_backward raised: {repr(_e)}")

        return loss.detach()


# =========================
# PEFT injection: lora | moelora
# =========================
def _freeze_base_model_params(model):
    for p in model.parameters():
        p.requires_grad_(False)


def _enable_trainable_peft_params(model, adapter_name: str):
    """
    Enable trainable params for PEFT.

    IMPORTANT:
      - Do NOT rely on `adapter_name in param_name`, because PEFT param naming differs
        across versions (sometimes uses 'default' or does not include adapter_name in
        the parameter name string).
      - For LoRA/MoE-LoRA we simply unfreeze all lora/gate/router params.
        This is safe because base model params are already frozen by `_freeze_base_model_params`.
    """
    for n, p in model.named_parameters():
        if p is None:
            continue

        is_peft = (
            ("lora_" in n) or ("lora_A" in n) or ("lora_B" in n)
            or ("gate_P" in n) or ("gate_" in n)
            or ("router_vec" in n) or ("router_mlp" in n) or ("router_" in n)
        )

        if is_peft:
            p.requires_grad = True
        # else: keep whatever it is (base params already frozen)


def _infer_task_k(script_args_, training_args_) -> int:
    if getattr(script_args_, "task_k", None) is not None:
        try:
            return int(script_args_.task_k)
        except Exception:
            pass

    env_k = os.environ.get("TASK_K", None)
    if env_k is not None:
        try:
            return int(env_k)
        except Exception:
            pass

    out_dir = getattr(training_args_, "output_dir", "") or ""
    base = os.path.basename(out_dir.rstrip("/"))
    m = re.match(r"^\s*(\d+)\s*-\s*sft\s*$", base)
    if m:
        return int(m.group(1))

    return 1


def inject_peft(model, model_args: SFTModelConfig, adapter_name: str):
    if not model_args.lora_enable:
        logger.info("[PEFT] lora_enable=False, skip PEFT injection.")
        return model, None

    _freeze_base_model_params(model)
    if model_args.peft_method == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        # ✅ PEFT requires target_modules; we auto-detect common Qwen/LLaMA-style proj names
        def _auto_target_modules(_model):
            default_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

            leaf = set()
            for n, _m in _model.named_modules():
                if n:
                    leaf.add(n.split(".")[-1])

            hits = [t for t in default_targets if t in leaf]
            if hits:
                return hits

            # fallback: suffix match
            suffixes = tuple(default_targets)
            hits_set = set()
            for n, _m in _model.named_modules():
                if n.endswith(suffixes):
                    hits_set.add(n.split(".")[-1])
            hits = sorted(hits_set)
            if not hits:
                raise ValueError(
                    "LoRA needs `target_modules`, but no common projection modules were found in model. "
                    "Please set target_modules manually for your backbone."
                )
            return hits

        target_modules = _auto_target_modules(model)

        # ✅ IMPORTANT: do NOT manually freeze/unfreeze here; let PEFT manage it (mimic your MoE-LoRA path)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=int(model_args.lora_r),
            lora_alpha=int(model_args.lora_alpha),
            lora_dropout=float(model_args.lora_dropout),
            bias=str(model_args.lora_bias),
            target_modules=target_modules,
        )

        model = get_peft_model(model, peft_config, adapter_name=adapter_name)

        # ✅ make adapter active/enabled
        try:
            model.set_adapter(adapter_name)
        except Exception:
            pass

        # some PEFT versions expose these toggles
        for fn in ("enable_adapter_layers", "enable_adapters"):
            if hasattr(model, fn):
                try:
                    getattr(model, fn)()
                except Exception:
                    pass

        # ✅ sanity check (rank0): ensure there are trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank in (-1, 0):
            print(f"[LoRA] target_modules={target_modules}")
            print(f"[LoRA] trainable_params={trainable:,} adapter={adapter_name}")

        if trainable == 0:
            # print a few params for debugging
            if local_rank in (-1, 0):
                print("[LoRA][DEBUG] First 50 params requires_grad flags:")
                c = 0
                for n, p in model.named_parameters():
                    if c >= 50:
                        break
                    print(f"  {n}: requires_grad={p.requires_grad}")
                    c += 1
            raise RuntimeError("[LoRA] No trainable parameters after PEFT injection; adapter not active or misconfigured.")

        logger.info(f"[LoRA] injected adapter={adapter_name}")

        # ✅ continue training: load prior adapter weights via direct safetensors
        # read + state_dict copy. PEFT's load_adapter() on transformers 4.49 +
        # peft >=0.10 triggers an internal import of
        # `transformers.integrations.tensor_parallel` which doesn't exist in
        # transformers 4.49 → ModuleNotFoundError. Bypass entirely.
        if model_args.continue_lora_path and os.path.isdir(model_args.continue_lora_path):
            try:
                from safetensors.torch import load_file as _safe_load_file
                ckpt_path = None
                for cand in (
                    os.path.join(model_args.continue_lora_path, "adapter_model.safetensors"),
                    os.path.join(model_args.continue_lora_path, "adapter_model.bin"),
                ):
                    if os.path.isfile(cand):
                        ckpt_path = cand
                        break
                if ckpt_path is None:
                    raise FileNotFoundError(f"no adapter_model.* in {model_args.continue_lora_path}")

                if ckpt_path.endswith(".safetensors"):
                    prior_sd = _safe_load_file(ckpt_path)
                else:
                    prior_sd = torch.load(ckpt_path, map_location="cpu")

                # Map prior state-dict keys to current PEFT model param names.
                # Saved key:  'base_model.model....lora_A.task_<prev>.weight'
                # New model:  'base_model.model....lora_A.<adapter_name>.weight'
                # Adapter name differs per task → rename .lora_{A|B}.<anything>.weight
                # → .lora_{A|B}.<current_adapter>.weight.
                import re as _re
                _adapter_re = _re.compile(r"(lora_[AB])\.[^.]+\.weight$")
                model_state = {n: p for n, p in model.named_parameters()}
                hits, misses = 0, 0
                with torch.no_grad():
                    for k, v in prior_sd.items():
                        candidates = [k]
                        # rewrite '...lora_A.<oldname>.weight' -> '...lora_A.<adapter_name>.weight'
                        m = _adapter_re.search(k)
                        if m is not None:
                            cand = _adapter_re.sub(f"{m.group(1)}.{adapter_name}.weight", k)
                            candidates.append(cand)
                        # also bare 'lora_A.weight' → with adapter
                        for marker in ("lora_A.weight", "lora_B.weight"):
                            if marker in k:
                                cand = k.replace(marker, marker.split(".")[0] + f".{adapter_name}." + marker.split(".")[1])
                                candidates.append(cand)
                        found = next((c for c in candidates if c in model_state), None)
                        if found is None:
                            misses += 1
                            continue
                        target = model_state[found]
                        if target.shape != v.shape:
                            misses += 1
                            continue
                        target.data.copy_(v.to(target.device, dtype=target.dtype))
                        hits += 1
                if local_rank in (-1, 0):
                    print(f"[LoRA] continued from {model_args.continue_lora_path}: hits={hits} misses={misses}")
            except Exception as e:
                logger.warning(f"[LoRA] failed to continue adapter from {model_args.continue_lora_path}: {repr(e)}")
                import traceback; traceback.print_exc()

        return model, adapter_name



    if model_args.peft_method == "moelora":
        from CoIN.peft import get_peft_model as coin_get_peft_model
        from CoIN.peft import TaskType as CoINTaskType
        from CoIN.peft.tuners.coinmoelora import CoINMOELoraConfig

        cfg = CoINMOELoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=None,
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type=CoINTaskType.CAUSAL_LM_CoIN,
            expert_num=int(model_args.expert_num),
            gate_rank=int(model_args.gate_rank),
            gate_tau=float(model_args.gate_tau),
            gate_tie=str(model_args.gate_tie),
            gate_router=str(model_args.gate_router),
        )

        model = coin_get_peft_model(model, cfg, adapter_name=adapter_name)
        try:
            model.set_adapter(adapter_name)
        except Exception:
            pass

        _enable_trainable_peft_params(model, adapter_name)
        logger.info(
            f"[MoE-LoRA] injected adapter={adapter_name} expert_num={model_args.expert_num} router={model_args.gate_router}"
        )

        if model_args.continue_lora_path and os.path.isdir(model_args.continue_lora_path):
            try:
                model.load_adapter(model_args.continue_lora_path, adapter_name=adapter_name)
                model.set_adapter(adapter_name)
                _enable_trainable_peft_params(model, adapter_name)
                logger.info(f"[MoE-LoRA] continued from {model_args.continue_lora_path}")
            except Exception as e:
                logger.warning(f"[MoE-LoRA] failed to continue adapter from {model_args.continue_lora_path}: {repr(e)}")

        return model, adapter_name

    raise ValueError(f"Unknown peft_method={model_args.peft_method}. Use lora|moelora.")


def freeze_vision_modules_(model, verbose: bool = True):
    vision_name_keywords = ("visual", "vision", "vision_tower", "image_encoder", "clip_vision", "vit", "eva")
    peft_name_keywords = ("lora_", "lora_A", "lora_B", "adapter", "gate_", "router_", "router_vec", "router_mlp")

    frozen_params = 0
    total_vision_params = 0

    for module_name, module in model.named_modules():
        name_l = module_name.lower()
        if not any(k in name_l for k in vision_name_keywords):
            continue

        for p_name, p in module.named_parameters(recurse=True):
            full_name = f"{module_name}.{p_name}" if module_name else p_name
            full_l = full_name.lower()

            if any(k in full_l for k in peft_name_keywords):
                continue

            total_vision_params += p.numel()
            if p.requires_grad:
                p.requires_grad_(False)
                frozen_params += p.numel()

    if verbose:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank in (-1, 0):
            print(f"[FreezeVision] total_vision_params={total_vision_params:,} newly_frozen={frozen_params:,}")


# =========================
# Save: adapter snapshot into output_dir/<adapter_name>
# =========================
def save_adapter_snapshot(model, output_dir: str, adapter_name: str, peft_method: str, lora_bias: str, local_rank: int):
    if adapter_name is None:
        return
    if local_rank not in (0, -1):
        return

    os.makedirs(output_dir, exist_ok=True)
    adapter_dir = os.path.join(output_dir, adapter_name)
    os.makedirs(adapter_dir, exist_ok=True)

    try:
        peft_cfg = getattr(model, "peft_config", None)
        if peft_cfg is not None and adapter_name in peft_cfg:
            peft_cfg[adapter_name].save_pretrained(adapter_dir)
        else:
            with open(os.path.join(adapter_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump({"adapter_name": adapter_name, "peft_type": peft_method}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), lora_bias, adapter_name)
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())

    try:
        from safetensors.torch import save_file as safe_save
        safe_save(state_dict, os.path.join(adapter_dir, "adapter_model.safetensors"))
    except Exception:
        torch.save(state_dict, os.path.join(adapter_dir, "adapter_model.bin"))

    torch.save(non_lora_state_dict, os.path.join(output_dir, "non_lora_trainables.bin"))


# =========================
# Main
# =========================
def main(script_args_, training_args_, model_args_):
    global processor, script_args, training_args
    script_args = script_args_
    training_args = training_args_

    set_seed(training_args.seed)

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

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args_}")
    logger.info(f"Script parameters {script_args_}")
    logger.info(f"Training parameters {training_args_}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    train_ratio_seed = int(getattr(training_args, "data_seed", None) or getattr(training_args, "seed", 42))
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args, train_ratio_seed=train_ratio_seed)

    # ✅ NEW: ER (Experience Replay) — mix prior-task samples into current dataset.
    # Triggered by --cl_method=replay AND --replay_paths is non-empty.
    _replay_paths = [p for p in str(getattr(script_args_, "replay_paths", "") or "").split(":") if p.strip()]
    _replay_ratio = float(getattr(script_args_, "replay_ratio", 0.0))
    if str(getattr(script_args_, "cl_method", "none")).lower() in ("replay", "der") \
            and _replay_paths and _replay_ratio > 0:
        rng_replay = random.Random(int(train_ratio_seed) + 1)
        extra = []
        # Filter replay samples to match the CURRENT task's modality, to keep
        # batches homogeneous (the Qwen2.5-VL processor can't ingest a batch
        # with mixed [PIL.Image, None] image entries).
        cur_has_vision = bool(DATA_HAS_VISION.get(getattr(script_args_, "cur_task", ""), False))
        for rp in _replay_paths:
            try:
                d = load_json_datas(
                    rp, "all", None,
                    train_ratio=1.0, train_ratio_strategy="random", train_ratio_seed=train_ratio_seed,
                )
                # Filter by modality compatibility
                if cur_has_vision:
                    d_compat = [s for s in d if s.get("image")]
                else:
                    d_compat = [s for s in d if not s.get("image")]
                n_keep = max(1, int(len(d_compat) * _replay_ratio)) if d_compat else 0
                if n_keep == 0:
                    logger.info(f"[Replay] {rp}: no modality-compatible samples (cur_vision={cur_has_vision}), skip")
                    continue
                idx = list(range(len(d_compat)))
                rng_replay.shuffle(idx)
                idx = idx[:n_keep]
                extra.extend([d_compat[i] for i in idx])
                logger.info(f"[Replay] mixed {n_keep}/{len(d_compat)} (compatible) from {rp} (orig {len(d)})")
            except Exception as e:
                logger.warning(f"[Replay] failed to load {rp}: {repr(e)}")
        if extra:
            dataset.list_data_dict.extend(extra)
            logger.info(f"[Replay] total dataset size after mix: {len(dataset.list_data_dict)}")

    if "vl" in model_args_.model_name_or_path.lower():
        # Cap image size to avoid "Image features and image tokens do not match"
        # errors on very large images (e.g. OCR ChartOCR pages can be 4000+px).
        # 512 * 28 * 28 = 401408 ≈ 640x640 pixels, matches Qwen2.5-VL default.
        _max_px = int(os.environ.get("QWEN_VL_MAX_PIXELS", 512 * 28 * 28))
        _min_px = int(os.environ.get("QWEN_VL_MIN_PIXELS", 4 * 28 * 28))
        processor = AutoProcessor.from_pretrained(
            model_args_.model_name_or_path,
            trust_remote_code=model_args_.trust_remote_code,
            max_pixels=_max_px,
            min_pixels=_min_px,
        )
        logger.info(f"Using AutoProcessor for VLM (max_pixels={_max_px}, min_pixels={_min_px}).")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args_.model_name_or_path,
            trust_remote_code=model_args_.trust_remote_code,
            use_fast=True
        )
        logger.info("Using AutoTokenizer for text-only model.")

    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    torch_dtype = (model_args_.torch_dtype if model_args_.torch_dtype in ["auto", None] else getattr(torch, model_args_.torch_dtype))
    quantization_config = get_quantization_config(model_args_)
    model_kwargs = dict(
        revision=model_args_.model_revision,
        attn_implementation=model_args_.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
    if "Qwen2-VL" in model_args_.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_args_.model_name_or_path, **model_kwargs)
    elif "Qwen2.5-VL" in model_args_.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args_.model_name_or_path,
            trust_remote_code=False,
            **model_kwargs
        )

        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionAttention
        for mod in model.modules():
            if isinstance(mod, Qwen2_5_VLVisionAttention):
                mod.is_causal = False

        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen2_5_vl_mod
        if hasattr(qwen2_5_vl_mod, "apply_rotary_pos_emb_flashatt"):
            _orig_apply = qwen2_5_vl_mod.apply_rotary_pos_emb_flashatt

            def _safe_apply(q, k, cos, sin):
                cos = cos.to(torch.float32)
                sin = sin.to(torch.float32)
                return _orig_apply(q, k, cos, sin)

            qwen2_5_vl_mod.apply_rotary_pos_emb_flashatt = _safe_apply
    else:
        raise ValueError(f"Unsupported model: {model_args_.model_name_or_path}")

    if model_args_.freeze_vision_modules:
        freeze_vision_modules_(model, verbose=True)

    task_k = _infer_task_k(script_args, training_args)
    adapter_name = f"task_{task_k}"

    model, adapter_name = inject_peft(model, model_args_, adapter_name=adapter_name)
    # =========================
    # FIX: LoRA + gradient checkpointing needs input grads, otherwise loss has no grad_fn
    # =========================
    if getattr(training_args, "gradient_checkpointing", False):
        # Frozen base + LoRA + gradient_checkpointing requires:
        # 1. enable gradient_checkpointing for the WHOLE model (LLM blocks)
        # 2. call enable_input_require_grads() so the input_embedding output has
        #    requires_grad=True (otherwise ckpt cannot build grad chain through
        #    a frozen embedding layer)
        # 3. DISABLE gradient_checkpointing on the vision encoder when it's
        #    frozen (it has no trainable params so ckpt would crash with
        #    "element 0 does not require grad").
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            try:
                model.enable_input_require_grads()
            except Exception:
                # Some PEFT-wrapped models expose this only on .base_model
                base = getattr(model, "base_model", None)
                if base is not None and hasattr(base, "enable_input_require_grads"):
                    base.enable_input_require_grads()
            if model_args_.freeze_vision_modules:
                # walk to find a `.visual` (Qwen2.5-VL) or a `.vision_model`
                # submodule and turn off its gradient_checkpointing flag.
                for path in ("base_model.model.visual", "model.visual", "visual",
                             "base_model.model.vision_model", "model.vision_model", "vision_model"):
                    obj = model
                    ok = True
                    for attr in path.split("."):
                        if hasattr(obj, attr):
                            obj = getattr(obj, attr)
                        else:
                            ok = False; break
                    if ok and hasattr(obj, "gradient_checkpointing"):
                        obj.gradient_checkpointing = False
                        if training_args.local_rank in (0, -1):
                            print(f"[GC FIX] disabled gradient_checkpointing on frozen vision module: {path}")
                        break
            if training_args.local_rank in (0, -1):
                print("[GC FIX] gradient_checkpointing enabled + enable_input_require_grads called")
        except Exception as e:
            try:
                emb = model.get_input_embeddings()
                if emb is not None:
                    def _make_require_grads(_module, _inp, out):
                        if torch.is_tensor(out):
                            out.requires_grad_(True)
                    emb.register_forward_hook(_make_require_grads)
                    if training_args.local_rank in (0, -1):
                        print("[GC FIX] fallback forward_hook applied on input embeddings")
            except Exception as e2:
                if training_args.local_rank in (0, -1):
                    print(f"[GC FIX][WARN] failed to enable input grads: {repr(e)} ; fallback failed: {repr(e2)}")

    if getattr(model_args_, "bits", 0) == 16:
        if getattr(training_args, "bf16", False):
            model.to(torch.bfloat16)
        elif getattr(training_args, "fp16", False):
            model.to(torch.float16)

    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.remove_unused_columns = False

    # =========================
    # Unified CL learner dispatch (paper §IV-A baselines + moelora)
    # =========================
    cl_learner = None
    cl_method = str(getattr(script_args_, "cl_method", "none")).lower()
    task_id = int(task_k) - 1  # 0-based

    if cl_method == "moelora":
        # Legacy path kept for the MoE-LoRA work
        setattr(training_args, "moelora_orth_lambda", float(getattr(model_args_, "moelora_orth_lambda", 0.0)))
        setattr(training_args, "moelora_B_reg_scale", float(getattr(model_args_, "moelora_B_reg_scale", 0.0)))
        setattr(training_args, "moelora_importance_top_p", float(getattr(model_args_, "moelora_importance_top_p", 1.0)))
        try:
            from cl_learner.moelora import moeloraLearner
            cl_learner = moeloraLearner(
                training_args=training_args,
                start_task_id=task_id - 1,
                model=model,
            )
            cl_learner.before_train(task_id=task_id, model=model)
            logger.info(f"[CL] moeloraLearner enabled: task_id={task_id}")
        except Exception as e:
            logger.warning(f"[CL] failed to init moeloraLearner: {repr(e)}")
            cl_learner = None
    else:
        # Paper baselines + Sequential FT
        try:
            from cl_learner import build_cl_learner
            # output_dir for cl_learner state files: parent of <step>-sft
            cl_state_dir = os.path.dirname(training_args.output_dir.rstrip("/"))
            replay_paths_list = [
                p for p in str(getattr(script_args_, "replay_paths", "") or "").split(":") if p.strip()
            ]
            cl_learner = build_cl_learner(
                method=cl_method,
                output_dir=cl_state_dir,
                lbd=float(getattr(script_args_, "cl_lbd", 1.0)),
                replay_paths=replay_paths_list,
                replay_ratio=float(getattr(script_args_, "replay_ratio", 0.0)),
                prompt_len=int(getattr(script_args_, "l2p_prompt_len", 8)),
            )
            cl_learner.before_train(task_id=task_id, model=model)
            logger.info(f"[CL] {cl_learner.__class__.__name__} enabled: task_id={task_id}")
        except Exception as e:
            logger.warning(f"[CL] failed to init cl_learner method={cl_method}: {repr(e)}")
            import traceback; traceback.print_exc()
            cl_learner = None

    trainer = CustomSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=getattr(processor, "tokenizer", processor),
        data_collator=collate_fn,
        callbacks=[SaveProcessorConfigCallback],
        mm_projector_lr=float(getattr(model_args_, "mm_projector_lr", training_args.learning_rate)),
        moelora_reg_enable=bool(getattr(model_args_, "moelora_reg_enable", False)),
        moelora_orth_lambda=float(getattr(model_args_, "moelora_orth_lambda", 0.0)),
        moelora_B_reg_scale=float(getattr(model_args_, "moelora_B_reg_scale", 0.0)),
        peft_method=str(getattr(model_args_, "peft_method", "lora")),
        cl_learner=cl_learner,
    )

    if getattr(model_args_, "peft_method", "lora") == "moelora":
        trainer.add_callback(
            GateTauSchedulerCallback(
                start_tau=float(getattr(model_args_, "gate_tau", 1.6)),
                end_tau=float(getattr(model_args_, "gate_tau_end", 0.7)),
                mode=str(getattr(model_args_, "gate_tau_mode", "cosine")),
                start_step=0,
                end_step=None,
                log_every_n_steps=100,
            )
        )

    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    if cl_learner is not None:
        try:
            cl_learner.after_train(task_id=int(task_k) - 1, model=trainer.model)
        except Exception as e:
            logger.warning(f"[CL] after_train failed: {repr(e)}")
        # ✅ NEW: if this is the last task, run finalize() (MagMaX needs it)
        if bool(getattr(script_args_, "is_last_task", False)):
            try:
                cl_learner.finalize(model=trainer.model)
                logger.info("[CL] finalize() called for last task")
            except Exception as e:
                logger.warning(f"[CL] finalize failed: {repr(e)}")

    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Save model and processor ***")
    trainer.save_model(training_args.output_dir)
    if processor is not None:
        processor.save_pretrained(training_args.output_dir)

    if hasattr(trainer.model, "config"):
        trainer.model.config.use_cache = True
        trainer.model.config.to_json_file(os.path.join(training_args.output_dir, "config.json"))

    if model_args_.lora_enable and adapter_name is not None:
        save_adapter_snapshot(
            model=trainer.model,
            output_dir=training_args.output_dir,
            adapter_name=adapter_name,
            peft_method=model_args_.peft_method,
            lora_bias=model_args_.lora_bias,
            local_rank=training_args.local_rank,
        )

    # ✅ NEW: run inference right here (model still alive in memory)
    try:
        maybe_run_post_and_last_infer(
            trainer=trainer,
            model=trainer.model,
            processor_=processor,
            script_args_=script_args_,
            training_args_=training_args,
            task_k_1based=int(task_k),
        )
    except Exception as e:
        logger.warning(f"[INFER] failed unexpectedly: {repr(e)}")

    kwargs = {
        "model_name": model_args_.model_name_or_path,
        "dataset_name": script_args.dataset_name,  # FIXED: keep string
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        try:
            trainer.create_model_card(**kwargs)
        except Exception as e:
            logger.warning(f"create_model_card failed: {repr(e)}")

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    from configs import SFTConfig
    parser = TrlParser((SFTScriptArguments, SFTConfig, SFTModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    print(script_args)
    main(script_args, training_args, model_args)
