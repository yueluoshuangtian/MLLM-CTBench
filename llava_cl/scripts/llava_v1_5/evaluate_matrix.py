"""
持续学习准确率矩阵 + CL 指标 (AA / BWT / Forgetting).
基于 PASs-MoE/without_reasoning/analyze_tool/evaluate_matrix.py 改写, 路径已本地化.

Usage:
    python evaluate_matrix.py <exp_root>
    # 例: python evaluate_matrix.py /path/to/checkpoints/llava-v1.5-7b/order1/ewc_1e8

目录约定 (train_cl.py 产出格式):
exp_root/
  1/                        # 训完任务 1 后的快照
    predictions/
      1-numglue.jsonl       # eval_idx-task_key.jsonl
      2-art.jsonl
      ...
    evaluations/            # 本脚本自动生成
  2/
    ...
"""
import os
import sys
import glob
import re
import json
import numpy as np
from tqdm import tqdm

try:
    import pandas as pd
except ImportError:
    pd = None

# ---- 把简化版 evaluator 模块路径加入 sys.path ----
EVAL_DIR = "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/new_eval_tool"
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

import evaluate_art_long_sentences
import evaluate_FOMC
import evaluate_math
import evaluate_numglue
import evaluate_object
import evaluate_science
import evaluate_medical_long_sentences
import evaluate_ocr


annotation_file_dict = {
    "numglue": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/numglue.json",
    "art":     "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/art.json",
    "math":    "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/math.json",
    "fomc":    "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/fomc.json",
    "medical": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/medical.json",
    "OCR":     "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/OCR.json",
    "science": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/science.json",
}
ANN_PATH = {k.lower(): v for k, v in annotation_file_dict.items()}

EVAL_REGISTRY = {
    "art":     (evaluate_art_long_sentences, {"is_reasoning": False}),
    "fomc":    (evaluate_FOMC,               {"is_reasoning": False}),
    "medical": (evaluate_medical_long_sentences, {"is_reasoning": False}),
    "ocr":     (evaluate_ocr,                {"is_reasoning": False}),
    "math":    (evaluate_math,               {"is_reasoning": False}),
    "numglue": (evaluate_numglue,            {"is_reasoning": False}),
    "science": (evaluate_science,            {"is_reasoning": False}),
    "object":  (evaluate_object,             {"is_reasoning": False}),
}

PRED_PATTERN = re.compile(r"^(\d+)-([a-zA-Z0-9_]+)\.jsonl$")


def call_eval(module, ann, pred, out=None, **kwargs):
    if hasattr(module, "eval_results"):
        try:
            return module.eval_results(ann, pred, out, **kwargs)
        except TypeError:
            return module.eval_results(ann, pred, **kwargs)
    raise AttributeError(f"{module} has no eval_results")


def _sanitize_jsonl(path):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return
    keep, bad, empty = [], 0, 0
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                empty += 1; continue
            try:
                json.loads(s); keep.append(s)
            except Exception:
                bad += 1
    if bad or empty:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(keep) + "\n")
        print(f"[sanitize] {path}: drop empty={empty} bad={bad}")


def list_train_dirs(exp_root):
    out = []
    if not os.path.isdir(exp_root):
        return out
    for e in os.scandir(exp_root):
        if e.is_dir() and e.name.isdigit():
            try:
                next(iter(os.scandir(e.path))); out.append(int(e.name))
            except StopIteration:
                pass
    return sorted(out)


def parse_pred_filename(p):
    m = PRED_PATTERN.match(os.path.basename(p))
    return (int(m.group(1)), m.group(2).lower()) if m else (None, None)


def build_tasks_map(exp_root, train_dirs):
    idx2task = {}
    for tid in train_dirs:
        for p in glob.glob(os.path.join(exp_root, str(tid), "predictions", "*.jsonl")):
            i, t = parse_pred_filename(p)
            if i is None: continue
            idx2task.setdefault(i, t)
    return idx2task


def evaluate_one(pred_path):
    _sanitize_jsonl(pred_path)
    _, task = parse_pred_filename(pred_path)
    if task not in EVAL_REGISTRY or task not in ANN_PATH:
        raise ValueError(f"unknown task '{task}' from {pred_path}")
    mod, kw = EVAL_REGISTRY[task]
    save = pred_path.replace("predictions", "evaluations")
    os.makedirs(os.path.dirname(save), exist_ok=True)
    return float(call_eval(mod, ANN_PATH[task], pred_path, save, **kw))


def cl_metrics(R):
    """AA = mean(last row); BWT = mean(R[-1,i] - R[i,i]); F = max(col) - last(col)."""
    T = R.shape[0]
    AA = float(np.nanmean(R[-1]))
    F = []
    for i in range(T):
        c = R[:, i]
        F.append((np.nanmax(c) - R[-1, i]) if not np.all(np.isnan(c)) else np.nan)
    F = np.array(F, dtype=float)
    BWT = float(np.nanmean([R[-1, i] - R[i, i] for i in range(T - 1)
                            if not (np.isnan(R[-1, i]) or np.isnan(R[i, i]))])) if T > 1 else 0.0
    return dict(AA=AA, BWT=BWT, F_mean=float(np.nanmean(F)))


def get_matrix(exp_root):
    train_dirs = list_train_dirs(exp_root)
    if not train_dirs:
        raise ValueError(f"no digit dirs in {exp_root}")
    T = len(train_dirs)
    idx2task = build_tasks_map(exp_root, train_dirs)
    tasks_list = [idx2task.get(i, f"task{i}") for i in range(1, T + 1)]

    R = np.full((T, T), np.nan, dtype=float)
    row_of = {tid: i for i, tid in enumerate(train_dirs)}
    for tid in train_dirs:
        for p in sorted(glob.glob(os.path.join(exp_root, str(tid), "predictions", "*.jsonl"))):
            i, _ = parse_pred_filename(p)
            if i is None or not 1 <= i <= T: continue
            try:
                acc = evaluate_one(p)
            except Exception as e:
                print(f"[ERR] {p}: {e}"); acc = float("nan")
            R[row_of[tid], i - 1] = acc

    print("\n[ACC MATRIX]")
    if pd is not None:
        df = pd.DataFrame(R, index=[f"after_{t}" for t in train_dirs],
                          columns=[f"{i+1}:{tasks_list[i]}" for i in range(T)])
        print(df.to_string(float_format=lambda x: f"{x:8.4f}"))
    else:
        np.set_printoptions(linewidth=220, suppress=True); print(R)
    return R, tasks_list, train_dirs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_matrix.py <exp_root>")
        sys.exit(1)
    R, tl, td = get_matrix(sys.argv[1])
    print("\n[CL METRICS]", cl_metrics(R))
