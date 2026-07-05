#!/usr/bin/env python3
"""CLMM(LLaVA) 正确口径重评: 从 <answer> 抠答案 + 按任务匹配, 算 AA/BWT 矩阵.
预测已完整(含<answer>), 只重评分, 不重推理. 纯 CPU.
"""
import json, os, re, string, glob
from difflib import SequenceMatcher

ORDER = ["art", "OCR", "fomc", "science", "numglue", "math", "medical"]
GT_DIR = "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test"
ROOT = "/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/checkpoints/llava-v1.5-7b/order3"
METHODS = ["none", "mas", "lwf", "freeze", "replay", "der", "l2p", "max_merge"]

def extract_answer(s):
    if not s: return ""
    m = re.findall(r"<answer>(.*?)</answer>", s, re.DOTALL | re.IGNORECASE)
    if m: return m[-1].strip()
    s2 = re.sub(r"<think>.*?</think>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    s2 = re.sub(r"<think>.*", " ", s2, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"</?answer>", " ", s2, flags=re.IGNORECASE).strip()

def norm(s):
    s = (s or "").lower()
    s = re.sub(r"[%s]" % re.escape(string.punctuation), " ", s)
    return re.sub(r"\s+", " ", s).strip()

def get_letter(a):
    a = (a or "").strip()
    m = re.match(r"^\(?\s*([A-Ha-h])\s*[\.\):,]", a) or re.match(r"^\(?\s*([A-Ha-h])\s*$", a)
    if m: return m.group(1).upper()
    m = re.search(r"\b([A-H])\b", a.upper())
    return m.group(1) if m else None

def get_num(s):
    m = re.findall(r"-?\d+\.?\d*", (s or "").replace(",", ""))
    if not m: return None
    try: return float(m[-1])
    except: return None

def sc_choice(gt, ans):
    g = (gt or "").strip().upper(); gl = g if g in list("ABCDEFGH") else get_letter(g)
    pl = get_letter(ans)
    if gl and pl: return gl == pl
    return sc_open(gt, ans)

def sc_num(gt, ans):
    g, p = get_num(gt), get_num(ans)
    return (g is not None and p is not None and abs(g - p) < 1e-3)

def sc_open(gt, ans):
    g, p = norm(gt), norm(ans)
    if not g: return False
    if g == p or g in p: return True
    return len(g) > 8 and SequenceMatcher(None, g, p).ratio() >= 0.6

def sc_math(gt, ans):
    g = (gt or "").strip()
    if re.fullmatch(r"[A-Ha-h]", g): return sc_choice(gt, ans)
    if g.lower() in ("yes", "no"): return g.lower() in norm(ans).split()[:3]
    return sc_num(gt, ans) or sc_open(gt, ans)

SCORER = {"fomc": sc_choice, "science": sc_choice, "math": sc_math,
          "numglue": sc_num, "art": sc_open, "OCR": sc_open, "medical": sc_open}

def load_gt(task):
    d = json.load(open(os.path.join(GT_DIR, f"{task}.json")))
    out = {}
    for ex in d:
        qid = ex.get("question_id") or ex.get("id")
        gt = ex.get("solution") or ex.get("answer")
        if qid is not None and gt is not None: out[str(qid)] = gt
    return out

def load_pred(path):
    out = {}
    if not os.path.isfile(path): return out
    for line in open(path):
        line = line.strip()
        if not line: continue
        r = json.loads(line); out[str(r["question_id"])] = r.get("text", "")
    return out

def score(task, pred_path):
    gt = load_gt(task); pred = load_pred(pred_path)
    if not pred: return None
    sc = SCORER.get(task, sc_open); c = t = 0
    for qid, g in gt.items():
        if qid not in pred: continue
        if sc(g, extract_answer(pred[qid])): c += 1
        t += 1
    return c / t if t else None

GT_CACHE = {}
def diag_path(m, k):
    task = ORDER[k-1]
    p = os.path.join(ROOT, f"{m}_default", str(k), "predictions", f"{k}-{task}.jsonl")
    if os.path.isfile(p): return p
    # k==7 或 中间也可能落在 7/ 下
    return os.path.join(ROOT, f"{m}_default", "7", "predictions", f"{k}-{task}.jsonl")

def last_path(m, j):
    task = ORDER[j-1]
    return os.path.join(ROOT, f"{m}_default", "7", "predictions", f"{j}-{task}.jsonl")

print(f"{'method':10s} | {'AA':>7s} {'BWT':>8s} | per-task last-row (P_Nj)")
print("-"*90)
results = {}
for m in METHODS:
    N = len(ORDER)
    diag = [score(ORDER[k-1], diag_path(m, k)) for k in range(1, N+1)]
    last = [score(ORDER[j-1], last_path(m, j)) for j in range(1, N+1)]
    valid_last = [x for x in last if x is not None]
    AA = sum(valid_last)/len(valid_last) if valid_last else float('nan')
    pairs = [last[j]-diag[j] for j in range(N-1) if last[j] is not None and diag[j] is not None]
    BWT = sum(pairs)/len(pairs) if pairs else float('nan')
    results[m] = {"diag": diag, "last": last, "AA": AA, "BWT": BWT}
    lr = " ".join(f"{ORDER[j][:4]}:{(last[j] if last[j] is not None else float('nan')):.2f}" for j in range(N))
    print(f"{m:10s} | {AA:7.4f} {BWT:+8.4f} | {lr}")

json.dump(results, open(os.path.join(ROOT, "reeval_v2_metrics.json"), "w"), indent=2)
print(f"\n[saved] {ROOT}/reeval_v2_metrics.json")
