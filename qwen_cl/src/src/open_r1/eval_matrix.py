#!/usr/bin/env python3
"""
eval_matrix.py — turn the per-task predictions saved by sft.py into a (T×T)
accuracy matrix and compute the standard continual-learning metrics
(AP, BWT, Forget_j).

Input layout (produced by sft.py's `maybe_run_post_and_last_infer`):

  <root>/predictions/<k>-<task>.json       # post-task k, inference on task k
                                           # gives us the diagonal P_kk
  <root>/last_predictions/<i>-<task_i>.json  # after final task N, inference on
                                             # all prior tasks i < N (last row)

Both files have the shape:
  { "results": [
      {"question_id": ..., "question": ..., "ground_truth": ..., "model_output": ...},
      ...
  ]}

We translate each row to the JSONL format expected by qwen_data/new_eval_tool/_common.py:
  {"question_id": ..., "text": <model_output>}
and call `eval_results(test_json, jsonl)` to get the per-task accuracy.

Output: <root>/metrics.json  with diag, last_row, AP, BWT, Forget_j.

Usage:
  python eval_matrix.py <root_dir> [--order order3]
                                   [--test_root /mnt/.../qwen data/test]
                                   [--eval_tool /mnt/.../new_eval_tool]
"""

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from typing import Dict, List, Optional

DEFAULT_ORDER3 = ["art", "OCR", "fomc", "science", "numglue", "math", "medical"]


def load_predictions(pred_json_path: str) -> List[dict]:
    with open(pred_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognized prediction file shape: {pred_json_path}")


def write_jsonl_for_evaluator(results: List[dict], jsonl_path: str):
    """_common.py expects: {"question_id", "text"} per line, text == model output."""
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ex in results:
            qid = ex.get("question_id") or ex.get("id")
            text = ex.get("model_output") or ex.get("text") or ""
            f.write(json.dumps({"question_id": qid, "text": text}, ensure_ascii=False) + "\n")


def load_eval_results_fn(eval_tool_dir: str):
    """Import `eval_results` from new_eval_tool/_common.py (it's keyed on the
    annotation file basename inside _common, so we can call it directly)."""
    common_path = os.path.join(eval_tool_dir, "_common.py")
    if not os.path.isfile(common_path):
        raise FileNotFoundError(f"_common.py not found at {common_path}")
    spec = importlib.util.spec_from_file_location("eval_common", common_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.eval_results


def score_one(eval_fn, test_json: str, pred_json: str, output_dir: str) -> Optional[float]:
    results = load_predictions(pred_json)
    if not results:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, dir=output_dir) as tmp:
        write_jsonl_for_evaluator(results, tmp.name)
        jsonl = tmp.name
    try:
        acc = eval_fn(test_json, jsonl, output_file=None, is_reasoning=True)
        return float(acc)
    finally:
        try:
            os.unlink(jsonl)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="The per-method run dir (parent of <k>-sft/, predictions/, last_predictions/).")
    parser.add_argument("--order", default="order3", help="task order name (default: order3)")
    parser.add_argument("--test_root", default="/mnt/cxzx/share/workspace/data_transfer/houzhiyan/TMM/qwen data/test")
    parser.add_argument("--eval_tool", default="/mnt/cxzx/share/workspace/data_transfer/houzhiyan/TMM/qwen data/new_eval_tool")
    args = parser.parse_args()

    root = args.root.rstrip("/")
    pred_dir = os.path.join(root, "predictions")
    last_dir = os.path.join(root, "last_predictions")

    if args.order == "order3":
        tasks = DEFAULT_ORDER3
    else:
        raise NotImplementedError(f"order {args.order} not configured here. Edit eval_matrix.py.")

    N = len(tasks)
    test_root = args.test_root
    eval_fn = load_eval_results_fn(args.eval_tool)

    # Per-task diagonal: post-task accuracy P_kk  (from predictions/<k>-<task>.json)
    diag: List[Optional[float]] = [None] * N
    for k, task in enumerate(tasks, start=1):
        pred_json = os.path.join(pred_dir, f"{k}-{task}.json")
        test_json = os.path.join(test_root, f"{task}.json")
        if not (os.path.isfile(pred_json) and os.path.isfile(test_json)):
            print(f"[eval] [diag] miss pred/test for task {k}-{task}: pred={os.path.isfile(pred_json)} test={os.path.isfile(test_json)}")
            continue
        acc = score_one(eval_fn, test_json, pred_json, root)
        diag[k - 1] = acc
        print(f"[eval] diag [{k}-{task}] acc={acc}")

    # Last row: after task N, score on all prior tasks i<N (last_predictions/<i>-<task_i>.json)
    # AND on task N itself (predictions/N-<task_N>.json)
    last_row: List[Optional[float]] = [None] * N
    for i, task in enumerate(tasks[:-1], start=1):
        pred_json = os.path.join(last_dir, f"{i}-{task}.json")
        test_json = os.path.join(test_root, f"{task}.json")
        if not (os.path.isfile(pred_json) and os.path.isfile(test_json)):
            print(f"[eval] [last_row] miss pred for task {i}-{task}")
            continue
        acc = score_one(eval_fn, test_json, pred_json, root)
        last_row[i - 1] = acc
        print(f"[eval] last_row[{i}-{task}] acc={acc}")
    # last_row[N-1] = diag[N-1] (P_NN, the final task scored at end of its own training)
    last_row[N - 1] = diag[N - 1]
    print(f"[eval] last_row[{N}-{tasks[-1]}] acc={last_row[N-1]} (= diagonal)")

    # CL metrics
    # AP = mean(P_Nj), j=1..N
    valid_last = [x for x in last_row if x is not None]
    AP = sum(valid_last) / len(valid_last) if valid_last else None

    # BWT = mean_{j=1..N-1} (P_Nj - P_jj)
    pairs = []
    forgets = {}
    for j in range(N - 1):
        if last_row[j] is not None and diag[j] is not None:
            d = last_row[j] - diag[j]
            pairs.append(d)
            forgets[tasks[j]] = d
    BWT = sum(pairs) / len(pairs) if pairs else None

    print()
    print("=== Order-3 CL metrics ===")
    print(f"  tasks    : {tasks}")
    print(f"  diag P_jj: {[f'{x:.4f}' if x is not None else '-' for x in diag]}")
    print(f"  last  P_Nj: {[f'{x:.4f}' if x is not None else '-' for x in last_row]}")
    print(f"  AP  = {AP:.4f}" if AP is not None else "  AP = N/A")
    print(f"  BWT = {BWT:.4f}" if BWT is not None else "  BWT = N/A")
    print(f"  Forget_j: " + ", ".join(f"{k}:{v:+.4f}" for k, v in forgets.items()))

    metrics = {
        "method_root": root,
        "order": args.order,
        "tasks": tasks,
        "diag_Pjj": diag,
        "last_row_PNj": last_row,
        "AP": AP,
        "BWT": BWT,
        "forget_per_task": forgets,
    }
    out = os.path.join(root, "metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n[eval] wrote {out}")


if __name__ == "__main__":
    main()
