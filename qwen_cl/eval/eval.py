import os
import re
import json
import glob
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from rouge import Rouge


# -----------------------------
# Task rules
# -----------------------------
TASKS_ROUGE = {"art", "medical"}  # rouge-l
TASKS_EXACT_OLD = {"numglue", "math", "fomc", "OCR", "science"}  # exact match

TASKS_EXACT_NEW = {"GQA", "ImageNet", "OCRVQA", "ScienceQA", "TextVQA", "VizWiz", "VQAv2"}
TASK_GROUNDING = "Grounding"

ALL_TASKS = TASKS_ROUGE | TASKS_EXACT_OLD | TASKS_EXACT_NEW | {TASK_GROUNDING}

FILENAME_TASK_RE = re.compile(r"^(?P<order>\d+)[-_](?P<task>[A-Za-z0-9]+)\.json$", re.IGNORECASE)

TASK_ALIAS = {
    "ocr": "OCR",
    "ocrvqa": "OCRVQA",
    "scienceqa": "ScienceQA",
    "textvqa": "TextVQA",
    "vizwiz": "VizWiz",
    "vqav2": "VQAv2",
    "gqa": "GQA",
    "grounding": "Grounding",
    "imagenet": "ImageNet",
    "art": "art",
    "medical": "medical",
    "numglue": "numglue",
    "math": "math",
    "fomc": "fomc",
    "science": "science",
}


# -----------------------------
# Grounding IoU helpers
# -----------------------------
def calculate_iou(b1, b2):
    x1, y1, x2, y2 = b1
    x21, y21, x22, y22 = b2
    inter = max(0, min(x2, x22) - max(x1, x21)) * max(0, min(y2, y22) - max(y1, y21))
    union = (x2 - x1) * (y2 - y1) + (x22 - x21) * (y22 - y21) - inter
    return inter / union if union > 0 else 0.0


_float_pat = re.compile(r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?")

def _extract_4_floats(s: Any) -> Optional[List[float]]:
    if s is None:
        return None
    if not isinstance(s, str):
        s = str(s)
    nums = _float_pat.findall(s)
    if len(nums) < 4:
        return None
    try:
        return [float(nums[i]) for i in range(4)]
    except Exception:
        return None


# -----------------------------
# IO utils
# -----------------------------
def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_task_from_filename(path: str) -> str:
    base = os.path.basename(path)
    m = FILENAME_TASK_RE.match(base)
    if m:
        raw_task = m.group("task")
        key = raw_task.lower()
        if key in TASK_ALIAS:
            task = TASK_ALIAS[key]
            if task in ALL_TASKS:
                return task

    lower = path.lower()
    for key, canonical in TASK_ALIAS.items():
        if key in lower and canonical in ALL_TASKS:
            return canonical

    raise ValueError(f"No matching task found for file: {path}")


# -----------------------------
# Normalization
# -----------------------------
def norm_text(s: str) -> str:
    """默认 exact match：lower + strip"""
    return (s or "").lower().strip()


# ImageNet 专用：去标点后再比
# 保留：中文/英文/数字/希腊字母/空白；其它都当作标点/符号去掉
_imagenet_keep_pat = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9\u0370-\u03FF\s]+")

def normalize_imagenet(s: str) -> str:
    s = (s or "").lower().strip()
    s = _imagenet_keep_pat.sub(" ", s)     # 非保留字符 => 空格
    s = re.sub(r"\s+", " ", s).strip()     # 压缩空白
    return s


# -----------------------------
# Core extraction
# -----------------------------
def extract_pred_and_label(data: Dict[str, Any]) -> tuple[str, str]:
    pred = str(data.get("model_output", "") or "")
    label = str(data.get("ground_truth", "") or "")
    return pred, label


def extract_critic_score(data: Dict[str, Any]) -> Optional[float]:
    text = str(data.get("model_output", "") or "")
    m = re.search(r"\d+(?:\.\d+)?", text)
    return float(m.group()) if m else None


# -----------------------------
# Result struct
# -----------------------------
@dataclass
class EvalResult:
    task: str
    n: int
    metric: str          # "acc" or "rouge-l"
    score_mean: float    # acc: [0,100] ; rouge-l: [0,1]
    critic_avg: Optional[float] = None


# -----------------------------
# Evaluate a file
# -----------------------------
def evaluate_file(
    json_path: str,
    save_dir: Optional[str] = None,
    iou_thr: float = 0.5
) -> EvalResult:
    task = parse_task_from_filename(json_path)
    raw = read_json(json_path)
    datas = raw["results"] if isinstance(raw, dict) and "results" in raw else raw
    if not isinstance(datas, list):
        raise ValueError(f"Invalid json format: {json_path} (expect list or dict with 'results')")

    rouge = Rouge()
    n = len(datas)

    score_sum = 0.0
    critic_sum = 0.0
    critic_cnt = 0
    new_datas: List[Dict[str, Any]] = []

    metric: Optional[str] = None

    for d in datas:
        pred, label = extract_pred_and_label(d)

        # 默认归一化
        pred_n = norm_text(pred)
        label_n = norm_text(label)

        # ImageNet：去标点后再比较
        if task == "ImageNet":
            pred_cmp = normalize_imagenet(pred)
            label_cmp = normalize_imagenet(label)
            d["res_answer"] = pred_cmp
        else:
            pred_cmp = pred_n
            label_cmp = label_n
            d["res_answer"] = pred_n

        # -------- Grounding：IoU -> acc --------
        if task == TASK_GROUNDING:
            b1 = _extract_4_floats(pred)
            b2 = _extract_4_floats(label)
            if b1 is None or b2 is None:
                iou = 0.0
            else:
                iou = float(calculate_iou(b1, b2))
            d["iou"] = iou
            is_right = iou >= float(iou_thr)
            d["is_right"] = bool(is_right)
            score_sum += 1.0 if is_right else 0.0
            metric = "acc"

        # -------- rouge-l tasks --------
        elif task in TASKS_ROUGE:
            try:
                rs = rouge.get_scores(pred_n, label_n, avg=True)
                rl = float(rs["rouge-l"]["f"])
            except Exception:
                rl = 0.0
            d["rouge-l"] = rl
            score_sum += rl
            metric = "rouge-l"

        # -------- exact match tasks (old + new) --------
        elif task in (TASKS_EXACT_OLD | TASKS_EXACT_NEW):
            is_right = (pred_cmp == label_cmp)
            d["is_right"] = bool(is_right)
            score_sum += 1.0 if is_right else 0.0
            metric = "acc"

        else:
            raise ValueError(f"Unexpected task: {task}")

        # 可选记录 critic
        critic = extract_critic_score(d)
        if critic is not None:
            d["critic_final_score"] = critic
            critic_sum += critic
            critic_cnt += 1

        new_datas.append(d)

    # 写逐样本结果
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{task}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(new_datas, f, ensure_ascii=False, indent=2)

    critic_avg = (critic_sum / critic_cnt) if critic_cnt > 0 else None

    if metric == "acc":
        score_mean = (score_sum / n) * 100.0 if n > 0 else 0.0
    else:
        score_mean = (score_sum / n) if n > 0 else 0.0

    return EvalResult(task=task, n=n, metric=metric or "acc", score_mean=score_mean, critic_avg=critic_avg)


# -----------------------------
# Evaluate a dir / root
# -----------------------------
def evaluate_dir(pred_dir: str, save_dir: Optional[str] = None, iou_thr: float = 0.5) -> Dict[str, EvalResult]:
    results: Dict[str, EvalResult] = {}
    for p in sorted(glob.glob(os.path.join(pred_dir, "*.json"))):
        key = os.path.splitext(os.path.basename(p))[0]
        results[key] = evaluate_file(p, save_dir=save_dir, iou_thr=iou_thr)
    return results


def evaluate_root(root: str, which: str, iou_thr: float = 0.5) -> Dict[str, EvalResult]:
    if which not in {"predictions", "last_predictions"}:
        raise ValueError("which must be 'predictions' or 'last_predictions'")

    pred_dir = os.path.join(root, which)
    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f"Not found: {pred_dir}")

    save_dir = os.path.join(root, f"eval_{which}")
    results = evaluate_dir(pred_dir, save_dir=save_dir, iou_thr=iou_thr)

    summary = {}
    for k, r in results.items():
        summary[k] = {
            "task": r.task,
            "n": r.n,
            "metric": r.metric,
            "score": r.score_mean,
            "critic_avg": r.critic_avg,
        }

    summary_path = os.path.join(root, f"summary_{which}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== {which} summary @ {root} (grounding iou_thr={iou_thr}) ===")
    for k, r in results.items():
        if r.metric == "acc":
            print(f"{k:18s} | task={r.task:10s} | acc={r.score_mean:6.2f}% | n={r.n}")
        else:
            extra = f" | critic_avg={r.critic_avg:.3f}" if r.critic_avg is not None else ""
            print(f"{k:18s} | task={r.task:10s} | rouge-l={r.score_mean:.4f} | n={r.n}{extra}")

    return results


if __name__ == "__main__":
    root = "/home/houzhiyan/qwen/cl_output/order1/mlp/Qwen2.5-VL-3B_moelora_expert_num6_gate_rank16_0"
    evaluate_root(root, which="predictions", iou_thr=0.5)
    evaluate_root(root, which="last_predictions", iou_thr=0.5)
