"""修正版评分: 先从 <answer>...</answer> 抠最终答案, 再按任务类型正确匹配.
对比旧版 _common.py 的修复:
  1) 答案提取: 取 <answer>..</answer> 内容(模型被训成 <think>..</think><answer>X</answer>);
     无 answer 标签时回退到去 think 后的尾部.
  2) fomc/math/science: 这些是【选择题】, GT 是字母 A/B/C/D(math 另有 yes/no/数值).
     -> 从答案里提取选项字母再比, 而不是关键词/数字.
  3) numglue: 数值; art/OCR/medical: 开放题文本(归一化 EM / 包含 / 模糊).
"""
import json, os, re, string
from difflib import SequenceMatcher

def _extract_answer(s: str) -> str:
    if not s: return ""
    m = re.findall(r"<answer>(.*?)</answer>", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m[-1].strip()
    # 无闭合 answer: 去掉 think 块后取尾部
    s2 = re.sub(r"<think>.*?</think>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s2 = re.sub(r"</?answer>", " ", s2, flags=re.IGNORECASE)
    # 若还残留未闭合 <think>, 去掉
    s2 = re.sub(r"<think>.*", " ", s2, flags=re.DOTALL | re.IGNORECASE)
    return s2.strip()

def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[%s]" % re.escape(string.punctuation), " ", s)
    return re.sub(r"\s+", " ", s).strip()

_LETTER_RE = re.compile(r"\b([A-H])\b")
def _extract_letter(ans: str):
    a = (ans or "").strip()
    # 优先: 开头就是单字母选项, 如 "C", "C.", "C. neutral", "(C)"
    m = re.match(r"^\(?\s*([A-Ha-h])\s*[\.\):,]", a) or re.match(r"^\(?\s*([A-Ha-h])\s*$", a)
    if m: return m.group(1).upper()
    m = _LETTER_RE.search(a.upper())
    return m.group(1) if m else None

_NUM_RE = re.compile(r"-?\d+\.?\d*")
def _extract_number(s: str):
    m = _NUM_RE.findall((s or "").replace(",", ""))
    if not m: return None
    try: return float(m[-1])
    except ValueError: return None

# ---- 任务打分(输入已是抠出来的 answer) ----
def _score_choice(gt, ans):
    """fomc/science, 以及 math 的字母题: GT 是字母, 比模型答案里的选项字母."""
    g = (gt or "").strip().upper()
    gl = _extract_letter(g) or (g if g in list("ABCDEFGH") else None)
    pl = _extract_letter(ans)
    if gl and pl: return gl == pl
    # GT 非字母(如 yes/no/数值) 的兜底
    return _score_open(gt, ans)

def _score_number(gt, ans):
    g = _extract_number(gt); p = _extract_number(ans)
    if g is None or p is None: return False
    return abs(g - p) < 1e-3

def _score_open(gt, ans):
    g = _norm(gt); p = _norm(ans)
    if not g: return False
    if g == p: return True
    if g in p: return True
    if len(g) > 8 and SequenceMatcher(None, g, p).ratio() >= 0.6: return True
    return False

def _score_math(gt, ans):
    g = (gt or "").strip()
    if re.fullmatch(r"[A-Ha-h]", g): return _score_choice(gt, ans)
    if g.lower() in ("yes", "no"): return _norm(ans).startswith(g.lower()) or g.lower() in _norm(ans).split()
    return _score_number(gt, ans) or _score_open(gt, ans)

SCORERS = {
    "fomc": _score_choice, "science": _score_choice,
    "math": _score_math, "numglue": _score_number,
    "art": _score_open, "OCR": _score_open, "medical": _score_open, "object": _score_open,
}

def _infer_task(annotation_file):
    base = os.path.basename(annotation_file).lower()
    for t in ["numglue", "art", "math", "fomc", "medical", "science", "object"]:
        if t in base: return t
    if "ocr" in base: return "OCR"
    return "art"

def _load_anno(f):
    data = json.load(open(f)); out = {}
    for ex in data:
        qid = ex.get("question_id") or ex.get("id")
        gt = ex.get("solution") or ex.get("answer") or ex.get("original_answer")
        if qid is not None and gt is not None: out[str(qid)] = gt
    return out

def _load_pred(f):
    out = {}
    for line in open(f):
        line = line.strip()
        if not line: continue
        ex = json.loads(line); out[str(ex["question_id"])] = ex.get("text", "")
    return out

def eval_results(annotation_file, answers_file, output_file=None, is_reasoning=True):
    anno = _load_anno(annotation_file); pred = _load_pred(answers_file)
    task = _infer_task(annotation_file); scorer = SCORERS.get(task, _score_open)
    correct = total = 0; details = []
    for qid, gt in anno.items():
        if qid not in pred: continue
        ans = _extract_answer(pred[qid])
        ok = scorer(gt, ans); correct += int(ok); total += 1
        details.append({"qid": qid, "gt": str(gt)[:80], "ans": ans[:80], "ok": ok})
    acc = correct / total if total else 0.0
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        json.dump({"task": task, "accuracy": acc, "correct": correct, "total": total, "details": details[:60]},
                  open(output_file, "w"), ensure_ascii=False, indent=2)
    return acc
