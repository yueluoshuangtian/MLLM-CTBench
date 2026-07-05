"""
最小评分模块, 兼容 train_cl.py 的接口:
    evaluate_xxx.eval_results(annotation_file, answers_file, output_file, is_reasoning=True) -> float

Annotation file: qwen_data/test/<task>.json, 每条含 {question_id, problem, solution, image?}.
Answers file:   model 推理产生的 jsonl, 每行 {"question_id", "prompt", "text", ...}.

针对每个任务做了任务特定的归一化:
- numglue / math : 提取数字 (容差 = 0)
- fomc          : 三分类 {hawkish, dovish, neutral} 关键词匹配
- 其他 (art/OCR/medical/science) : 文本归一化 + 子串/相似度匹配
"""
import json
import os
import re
import string
from difflib import SequenceMatcher


# ---------- 共用归一化 ----------
def _strip_think(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"<think>.*?</think>", " ", s, flags=re.DOTALL | re.IGNORECASE)


def _normalize_text(s: str) -> str:
    s = _strip_think(s or "").lower()
    s = re.sub(r"[%s]" % re.escape(string.punctuation), " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_NUM_RE = re.compile(r"-?\d+\.?\d*")

def _extract_number(s: str):
    s = _strip_think(s)
    m = _NUM_RE.findall(s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m[-1])  # 取最后一个数字, 通常是答案
    except ValueError:
        return None


def _load_anno(annotation_file):
    """读取 ground-truth, 兼容 conversations 格式 (llava_test/*.json) 与 简化格式 (test/*.json)."""
    with open(annotation_file) as f:
        data = json.load(f)
    out = {}
    for ex in data:
        qid = ex.get("question_id") or ex.get("id")
        gt = ex.get("solution") or ex.get("answer") or ex.get("original_answer")
        if gt is None and "conversations" in ex:
            for turn in ex["conversations"]:
                if turn.get("from") == "gpt":
                    gt = turn.get("value")
                    break
        if qid is not None and gt is not None:
            out[str(qid)] = gt
    return out


def _load_pred(answers_file):
    out = {}
    with open(answers_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            out[str(ex["question_id"])] = ex.get("text", "")
    return out


# ---------- 任务特定评分 ----------
def _score_text_loose(gt: str, pr: str) -> bool:
    """art / OCR / medical / science: 文本类宽松匹配."""
    gt_n = _normalize_text(gt)
    pr_n = _normalize_text(pr)
    if not gt_n:
        return False
    # 1) 完整 gt 出现在 pr
    if gt_n in pr_n:
        return True
    # 2) gt 前 30 字符 (短 gt 就用全部) 出现
    head = gt_n[:30] if len(gt_n) > 30 else gt_n
    if head and head in pr_n:
        return True
    # 3) 长 gt: 用模糊比 (>= 0.6)
    if len(gt_n) > 8:
        ratio = SequenceMatcher(None, gt_n, pr_n).ratio()
        return ratio >= 0.6
    return False


def _score_number(gt: str, pr: str) -> bool:
    """numglue / math: 数值匹配 (容差 1e-3)."""
    g = _extract_number(gt)
    p = _extract_number(pr)
    if g is None or p is None:
        return False
    return abs(g - p) < 1e-3


_FOMC_LABELS = {"hawkish", "dovish", "neutral"}

def _score_fomc(gt: str, pr: str) -> bool:
    """fomc: 三分类, 取 pr 中第一个出现的标签词."""
    gt_lower = (gt or "").lower()
    pr_lower = _strip_think(pr or "").lower()
    gt_label = next((lab for lab in _FOMC_LABELS if lab in gt_lower), None)
    if gt_label is None:
        return False
    # pr 中第一个出现的标签
    positions = [(pr_lower.find(lab), lab) for lab in _FOMC_LABELS]
    positions = [(p, l) for p, l in positions if p >= 0]
    if not positions:
        return False
    positions.sort()
    return positions[0][1] == gt_label


SCORERS = {
    "art": _score_text_loose,
    "OCR": _score_text_loose,
    "medical": _score_text_loose,
    "science": _score_text_loose,
    "math": _score_number,
    "numglue": _score_number,
    "fomc": _score_fomc,
    "object": _score_text_loose,
}


def _infer_task(annotation_file: str) -> str:
    base = os.path.basename(annotation_file).lower()
    for t in ["numglue", "art", "math", "fomc", "medical", "science", "object"]:
        if t in base:
            return t
    if "ocr" in base:
        return "OCR"
    return "art"  # fallback


def eval_results(annotation_file, answers_file, output_file=None, is_reasoning=True):
    anno = _load_anno(annotation_file)
    pred = _load_pred(answers_file)
    task = _infer_task(annotation_file)
    scorer = SCORERS.get(task, _score_text_loose)

    correct = 0
    total = 0
    details = []
    for qid, gt in anno.items():
        if qid not in pred:
            continue
        ok = scorer(gt, pred[qid])
        if ok:
            correct += 1
        total += 1
        details.append({"qid": qid, "gt": (gt or "")[:200], "pred": (pred[qid] or "")[:200], "ok": ok})

    acc = (correct / total) if total else 0.0
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump({"task": task, "accuracy": acc, "correct": correct, "total": total,
                       "details": details[:50]}, f, ensure_ascii=False, indent=2)
    return acc
