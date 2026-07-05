# -*- coding: utf-8 -*-
import os
import gc
import glob
import json
import random
import argparse
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel


# ----------------------------
# Task order (match training)
# ----------------------------
ORDER_TO_TASKS: Dict[str, List[str]] = {
    "order1": ["numglue", "art", "math", "fomc", "medical", "OCR", "science"],
    "order2": ["science", "OCR", "medical", "fomc", "math", "art", "numglue"],
    "order3": ["ScienceQA", "TextVQA", "ImageNet", "GQA", "VizWiz", "Grounding", "VQAv2", "OCRVQA"],
    "order4": ["ScienceQA", "ImageNet", "VizWiz", "Grounding", "TextVQA", "GQA", "VQAv2", "OCRVQA"],
    "order5": ["fomc", "art", "math", "science", "numglue", "OCR", "medical"],
    "order6": ["science", "medical", "fomc", "art", "math", "numglue", "OCR"],
}

# whether task has images
DATA_HAS_VISION = {
    "numglue": False,
    "fomc": False,
    "art": True,
    "math": True,
    "medical": True,
    "OCR": True,
    "science": True,
    "ScienceQA": True,
    "TextVQA": True,
    "ImageNet": True,
    "GQA": True,
    "VizWiz": True,
    "Grounding": True,
    "VQAv2": True,
    "OCRVQA": True,
}

# 路径 overridable via env vars (与 sft.py 一致)：
#   QWEN_TEST_ROOT  — 7 个核心任务的 test json 所在目录
#   QWEN_IMAGE_ROOT — 7 个核心任务的图像 root
# 缺省指向本机真实位置（/mnt/cxzx/workspace/...）
_TEST_ROOT_DEFAULT = "/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/qwen data/test"
_IMG_ROOT_DEFAULT = "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark"
_TEST_ROOT = os.environ.get("QWEN_TEST_ROOT", _TEST_ROOT_DEFAULT)
_IMG_ROOT = os.environ.get("QWEN_IMAGE_ROOT", _IMG_ROOT_DEFAULT)

QUESTION_FILE = {
    "numglue": f"{_TEST_ROOT}/numglue.json",
    "art": f"{_TEST_ROOT}/art.json",
    "math": f"{_TEST_ROOT}/math.json",
    "fomc": f"{_TEST_ROOT}/fomc.json",
    "medical": f"{_TEST_ROOT}/medical.json",
    "OCR": f"{_TEST_ROOT}/OCR.json",
    "science": f"{_TEST_ROOT}/science.json",

    # CoIN 任务（非本次主流程）：保留原始硬编码，调用方负责挂载
    "GQA": "/home/houzhiyan/dataset/qwen/test/GQA.json",
    "Grounding": "/home/houzhiyan/dataset/qwen/test/Grounding.json",
    "ImageNet": "/home/houzhiyan/dataset/qwen/test/ImageNet.json",
    "OCRVQA": "/home/houzhiyan/dataset/qwen/test/OCRVQA.json",
    "ScienceQA": "/home/houzhiyan/dataset/qwen/test/ScienceQA.json",
    "TextVQA": "/home/houzhiyan/dataset/qwen/test/TextVQA.json",
    "VizWiz": "/home/houzhiyan/dataset/qwen/test/VizWiz.json",
    "VQAv2": "/home/houzhiyan/dataset/qwen/test/VQAv2.json",
}

# task-specific image roots
IMAGE_ROOT = {
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


# ----------------------------
# Utils
# ----------------------------
def memory_stats():
    if torch.cuda.is_available():
        print(f"[CUDA] allocated(MB)={torch.cuda.memory_allocated() / 1024 ** 2:.2f} "
              f"reserved(MB)={torch.cuda.memory_reserved() / 1024 ** 2:.2f}")

def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    memory_stats()

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def load_json(path: str) -> List[dict]:
    with open(path, "r") as f:
        return json.load(f)

def detect_trained_k(adapter_root: str) -> int:
    """
    detect max numeric subdir under adapter_root (e.g., 1,2,3,...)
    this matches your previous logic.
    """
    max_k = 0
    for d in glob.glob(os.path.join(adapter_root, "*")):
        if os.path.isdir(d) and os.path.basename(d).isdigit() and len(os.listdir(d)) > 0:
            max_k = max(max_k, int(os.path.basename(d)))
    return max_k

def find_adapter_dir(adapter_root: str, k_1based: int) -> str:
    """
    Find adapter directory for task k.
    Compatible with:
      - <root>/<k>/ (contains adapter_config.json or adapter_model.*)
      - <root>/<k>-sft/task_<k>/...
      - <root>/<k>-sft/...
    """
    candidates = [
        os.path.join(adapter_root, str(k_1based)),
        os.path.join(adapter_root, f"{k_1based}-sft", f"task_{k_1based}"),
        os.path.join(adapter_root, f"{k_1based}-sft"),
        os.path.join(adapter_root, f"task_{k_1based}"),
    ]

    def ok(d: str) -> bool:
        if not os.path.isdir(d):
            return False
        if os.path.exists(os.path.join(d, "adapter_config.json")):
            return True
        if glob.glob(os.path.join(d, "adapter_model.*")):
            return True
        if glob.glob(os.path.join(d, "*.safetensors")) or glob.glob(os.path.join(d, "*.bin")):
            return True
        return False

    for c in candidates:
        if ok(c):
            return c

    deep_root = os.path.join(adapter_root, f"{k_1based}-sft")
    if os.path.isdir(deep_root):
        for d in glob.glob(os.path.join(deep_root, "**"), recursive=True):
            if ok(d):
                return d

    raise FileNotFoundError(f"Cannot find adapter dir for k={k_1based} under {adapter_root}")


# ----------------------------
# Prompt building (NO GT leakage)
# ----------------------------
def build_messages(data: List[dict], task: str, shuffle: bool, seed: int) -> List[List[dict]]:
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(data)

    img_root = IMAGE_ROOT.get(task, "")
    messages: List[List[dict]] = []
    template = "{Question}"

    for ex in data:
        problem = ex.get("problem") or ex.get("text") or ex.get("question")
        if problem is None:
            raise KeyError(f"missing problem/text/question in sample keys={list(ex.keys())}")

        if "image" in ex and ex["image"]:
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

        messages.append(msg)

    return messages


def run_inference(processor: AutoProcessor,
                  model: torch.nn.Module,
                  messages: List[List[dict]],
                  task: str,
                  bs: int,
                  max_new_tokens: int,
                  device: torch.device) -> List[str]:
    # tokenizer padding side must be set on tokenizer, not processor call
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None and processor.tokenizer.eos_token is not None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model.eval()
    outs: List[str] = []

    with torch.inference_mode():
        for i in tqdm(range(0, len(messages), bs), desc=f"🧪 Inference[{task}]"):
            batch_messages = messages[i:i + bs]
            texts = [
                processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in batch_messages
            ]

            if DATA_HAS_VISION.get(task, False):
                image_inputs, video_inputs = process_vision_info(batch_messages)
            else:
                image_inputs, video_inputs = None, None

            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            gen_ids = model.generate(
                **inputs,
                do_sample=False,
                use_cache=True,
                max_new_tokens=max_new_tokens,
            )

            in_ids = inputs["input_ids"]
            trimmed = [out[len(inp):] for inp, out in zip(in_ids, gen_ids)]
            batch_text = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outs.extend(batch_text)

    return outs


def save_results(data: List[dict], outputs: List[str], out_path: str):
    final = []
    for ex, out in zip(data, outputs):
        final.append({
            "question_id": ex.get("question_id", None),
            "question": ex.get("problem") or ex.get("text") or ex.get("question"),
            "ground_truth": ex.get("solution") or ex.get("answer"),
            "model_output": out,
            "image": ex.get("image", None),
        })
    safe_mkdir(os.path.dirname(out_path))
    with open(out_path, "w") as f:
        json.dump({"results": final}, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved: {out_path}")


def evaluate_one_task(base_model_path: str,
                      adapter_dir: str,
                      task: str,
                      out_path: str,
                      bs: int,
                      max_new_tokens: int,
                      sample_num: int,
                      shuffle: bool,
                      seed: int,
                      dtype: str,
                      attn_impl: str,
                      device: torch.device):
    # data
    data_path = QUESTION_FILE[task]
    data = load_json(data_path)
    if sample_num is not None and sample_num > 0:
        data = data[:sample_num]
    if not data:
        raise RuntimeError(f"No data for task={task} at {data_path}")

    messages = build_messages(data, task, shuffle=shuffle, seed=seed)

    # dtype
    torch_dtype = (
        torch.bfloat16 if dtype == "bf16"
        else torch.float16 if dtype == "fp16"
        else torch.float32
    )

    # base + adapter (NO MERGE)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        device_map=None,
    ).to(device)

    processor = AutoProcessor.from_pretrained(base_model_path, use_fast=True)

    # important: for custom MoE-LoRA, ensure your tuner is importable in env
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False).to(device)
    model.eval()

    outputs = run_inference(
        processor=processor,
        model=model,
        messages=messages,
        task=task,
        bs=bs,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    save_results(data, outputs, out_path)

    del model
    del base
    release_memory()


# ----------------------------
# Main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=True, help="Base model path (original Qwen2.5-VL ...)")
    p.add_argument("--adapter_root", type=str, required=True, help="CL root dir that contains task adapters")
    p.add_argument("--order", type=str, required=True, choices=list(ORDER_TO_TASKS.keys()))
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--sample_num", type=int, default=0, help="0 means all")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--attn_impl", type=str, default="flash_attention_2")
    p.add_argument("--device", type=str, default="cuda:0")
    # modes
    p.add_argument("--do_post", action="store_true")
    p.add_argument("--do_last", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tasks = ORDER_TO_TASKS[args.order]

    # how many tasks have actually trained adapters
    trained_k = detect_trained_k(args.adapter_root)
    if trained_k <= 0:
        raise RuntimeError(f"No trained adapters found under: {args.adapter_root}")

    # if training not finished, only evaluate first trained_k tasks
    if trained_k < len(tasks):
        print(f"[WARN] training not finished: trained_k={trained_k} < total_tasks={len(tasks)}")
        tasks = tasks[:trained_k]

    # output dirs
    post_dir = os.path.join(args.adapter_root, "prediction")
    last_dir = os.path.join(args.adapter_root, "last_prediction")
    safe_mkdir(post_dir)
    safe_mkdir(last_dir)

    def already_done(dir_path: str) -> set:
        done = set()
        for f in glob.glob(os.path.join(dir_path, "*_*.json")):
            done.add(os.path.basename(f).rsplit("_", 1)[-1].replace(".json", ""))
        return done

    sample_num = None if (args.sample_num is None or args.sample_num <= 0) else int(args.sample_num)

    # 1) post: after task i, only test task i (your rule)
    if args.do_post:
        done = already_done(post_dir)
        need = [t for t in tasks if t not in done]
        print(f"[POST] need={need}")

        for idx0, task in enumerate(tasks):
            if task not in need:
                continue
            k = idx0 + 1
            adapter_dir = find_adapter_dir(args.adapter_root, k)
            out_path = os.path.join(post_dir, f"{k}_{task}.json")
            print(f"[POST] k={k} task={task} adapter={adapter_dir}")

            evaluate_one_task(
                base_model_path=args.base_model,
                adapter_dir=adapter_dir,
                task=task,
                out_path=out_path,
                bs=args.bs,
                max_new_tokens=args.max_new_tokens,
                sample_num=sample_num,
                shuffle=args.shuffle,
                seed=args.seed,
                dtype=args.dtype,
                attn_impl=args.attn_impl,
                device=device,
            )

    # 2) last: last adapter evaluates all previous tasks (i < n), skip last task to avoid duplication
    if args.do_last:
        done = already_done(last_dir)
        need = [t for t in tasks if t not in done]
        print(f"[LAST] need={need}")

        k_last = len(tasks)
        adapter_last = find_adapter_dir(args.adapter_root, k_last)
        print(f"[LAST] use last adapter k={k_last}: {adapter_last}")

        for idx0, task in enumerate(tasks):
            # skip last task evaluation to avoid duplicate (as you required)
            if idx0 == len(tasks) - 1:
                continue
            if task not in need:
                continue

            out_path = os.path.join(last_dir, f"{idx0+1}_{task}.json")
            print(f"[LAST] eval task={task} with last-adapter={adapter_last}")

            evaluate_one_task(
                base_model_path=args.base_model,
                adapter_dir=adapter_last,
                task=task,
                out_path=out_path,
                bs=args.bs,
                max_new_tokens=args.max_new_tokens,
                sample_num=sample_num,
                shuffle=args.shuffle,
                seed=args.seed,
                dtype=args.dtype,
                attn_impl=args.attn_impl,
                device=device,
            )


if __name__ == "__main__":
    main()
