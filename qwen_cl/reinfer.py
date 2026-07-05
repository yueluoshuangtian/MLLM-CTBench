#!/usr/bin/env python3
"""8卡分布式重推理：用已存 adapter + 大 max_new_tokens 重新生成预测(让模型写完 <answer>)。
复用 sft.py 的分布式推理函数。torchrun --nproc_per_node=8 reinfer.py ...

一个方法跑一次：加载 base 一次，逐 task 切 adapter，重推理 diagonal(1..7) 与 last_row(用末adapter 推 1..6)，
存到 <method_root>/predictions_v2/ 与 last_predictions_v2/。
"""
import os, sys, json, argparse, glob
import torch
import torch.distributed as dist

QWEN = "/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/qwen_cl"
sys.path.insert(0, os.path.join(QWEN, "src/src/open_r1"))
os.environ.setdefault("QWEN_TEST_ROOT", "/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/qwen data/test")
os.environ.setdefault("QWEN_IMAGE_ROOT", "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark")

import re
import sft  # noqa  复用其推理函数与全局(QUESTION_FILE, _build_messages_for_task 等)
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
from safetensors.torch import load_file

ORDER = ["art", "OCR", "fomc", "science", "numglue", "math", "medical"]
BASE = "/mnt/cxzx/workspace/data_transfer/houzhiyan/models/Qwen2.5-VL-3B-Instruct"
TARGET_MODULES = ["q_proj", "v_proj", "down_proj", "k_proj", "up_proj", "o_proj", "gate_proj"]
_RE_AD = re.compile(r"(lora_[AB])\.[^.]+\.weight$")

def is0(): return (not dist.is_initialized()) or dist.get_rank() == 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method_root", required=True)        # .../order3/none_tr1.0
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--sample_num", type=int, default=0)   # 0=全部; 验证时设小
    args = ap.parse_args()

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # processor (与训练一致 max_pixels=512*28*28)
    proc = AutoProcessor.from_pretrained(BASE, max_pixels=512*28*28, min_pixels=256*28*28)
    # base 模型 + 空 LoRA(adapter 名 "ad"), 之后手动拷各 task 权重
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16,
    ).to(device)
    cfg = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, bias="none",
                     task_type=TaskType.CAUSAL_LM, target_modules=TARGET_MODULES)
    model = get_peft_model(model, cfg, adapter_name="ad")
    model.eval()

    root = args.method_root.rstrip("/")
    pred_dir = os.path.join(root, "predictions_v2"); last_dir = os.path.join(root, "last_predictions_v2")
    if is0(): os.makedirs(pred_dir, exist_ok=True); os.makedirs(last_dir, exist_ok=True)

    def adapter_path(k):
        # <root>/<k>-sft/task_<k>/
        p = os.path.join(root, f"{k}-sft", f"task_{k}")
        return p if os.path.isdir(p) else None

    mstate = {n: p for n, p in model.named_parameters()}
    def use_adapter(k):
        """手动把 task_k 的 LoRA 权重拷进 'ad' 槽(绕开 PEFT load_adapter 的 tensor_parallel import bug)。"""
        ap_ = adapter_path(k)
        if ap_ is None:
            if is0(): print(f"[skip] adapter for task {k} missing")
            return None
        sf = os.path.join(ap_, "adapter_model.safetensors")
        if not os.path.isfile(sf):
            if is0(): print(f"[skip] no safetensors task {k}")
            return None
        sd = load_file(sf)
        hits = miss = 0
        with torch.no_grad():
            for kk, v in sd.items():
                cand = _RE_AD.sub(r"\1.ad.weight", kk) if _RE_AD.search(kk) else kk
                tgt = mstate.get(cand)
                if tgt is None or tuple(tgt.shape) != tuple(v.shape):
                    miss += 1; continue
                tgt.data.copy_(v.to(tgt.device, tgt.dtype)); hits += 1
        # 非 LoRA 可训练参数(若有)
        nlt = os.path.join(root, f"{k}-sft", "non_lora_trainables.bin")
        if os.path.isfile(nlt):
            try: model.load_state_dict(torch.load(nlt, map_location=device), strict=False)
            except Exception as e:
                if is0(): print(f"[warn] non_lora load fail k={k}: {e}")
        if is0(): print(f"[adapter k={k}] hits={hits} miss={miss}")
        model.eval()
        return model

    def infer(m, task, out_path):
        data, outs = sft._run_inference_one_task_with_progress(
            model=m, processor_=proc, task=task, bs=args.bs,
            max_new_tokens=args.max_new_tokens,
            sample_num=(args.sample_num or None), shuffle=False, seed=0,
            device=device, log_every_batches=20,
        )
        if is0():
            sft._save_infer_results(data, outs, out_path)
            print(f"[saved] {out_path}  ({sum(1 for o in outs if o)}/{len(data)})")

    # 1) diagonal: task k 用 adapter k
    for k, task in enumerate(ORDER, start=1):
        m = use_adapter(k)
        if m is None: continue
        infer(m, task, os.path.join(pred_dir, f"{k}-{task}.json"))
    # 2) last row: 用最末 adapter(7) 推 1..6
    m = use_adapter(len(ORDER))
    if m is not None:
        for i, task in enumerate(ORDER[:-1], start=1):
            infer(m, task, os.path.join(last_dir, f"{i}-{task}.json"))

    if is0(): print("[REINFER_DONE]", root)

if __name__ == "__main__":
    main()
