"""
把两个 backbone（llava / qwen）各持续学习方法的预测文件，转成可打分的样本。

每个样本：
  {model, method, task, task_key, has_image, stage, question_id,
   question, cot_text, ref_answer, image_path}
stage ∈ {"post"（训完即测，对角 P_jj）, "final"（全训完，末行 P_Nj）}。
"""
import json, os, re
import cl_config as C

_GT_CACHE = {}


def load_gt(task):
    """金标：question_id -> (problem, solution, image_rel)。"""
    if task in _GT_CACHE:
        return _GT_CACHE[task]
    path = os.path.join(C.GT_DIR, f"{task}.json")
    out = {}
    if os.path.isfile(path):
        for ex in json.load(open(path, encoding="utf-8")):
            qid = ex.get("question_id") or ex.get("id")
            if qid is None:
                continue
            out[str(qid)] = (
                ex.get("problem") or ex.get("question") or "",
                ex.get("solution") or ex.get("answer") or "",
                ex.get("image") or "",
            )
    _GT_CACHE[task] = out
    return out


def extract_cot(text):
    """从模型输出里抽出『推理过程』部分（去掉 <answer> 标签）。"""
    if not text:
        return ""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 没有成对 think 标签：去掉 answer 段后整体当作 reasoning
    t = re.sub(r"<answer>.*?</answer>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"</?think>|</?answer>", " ", t, flags=re.IGNORECASE)
    return t.strip()


def _resolve_image(rel):
    if not rel:
        return None
    p = rel if os.path.isabs(rel) else os.path.join(C.IMAGE_ROOT, rel)
    return p if os.path.isfile(p) else None


def _read_pred_records(model, path):
    """读单个预测文件 -> [(question_id, question_or_None, gt_or_None, output_text)]。"""
    recs = []
    if not os.path.isfile(path):
        return recs
    fmt = C.MODELS[model]["fmt"]
    if fmt == "jsonl_text":          # llava: 每行 {question_id, prompt, text}
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            recs.append((str(r.get("question_id")), r.get("prompt"), None, r.get("text", "")))
    elif fmt == "json_results":      # qwen: {results:[{question_id, question, ground_truth, model_output}]}
        try:
            obj = json.load(open(path, encoding="utf-8"))
        except Exception:
            return recs
        for r in obj.get("results", obj if isinstance(obj, list) else []):
            recs.append((str(r.get("question_id")), r.get("question"),
                         r.get("ground_truth"), r.get("model_output", "")))
    return recs


def _candidate_paths(model, method, stage, j, task):
    spec = C.MODELS[model]
    md = spec["method_dir"].format(method=method)
    out = []
    for tmpl in spec[stage]:
        rel = tmpl.format(md=md, j=j, k=j, task=task)
        out.append(os.path.join(spec["root"], rel))
    return out


def iter_cells(models=None, methods=None, tasks=None, stages=("post", "final")):
    """枚举所有 (model, method, stage, task) 单元格，yield (meta, [samples])。

    - post  : stage j 的对角文件（P_jj）
    - final : 全训完后对该任务的预测（P_Nj）。最后一个任务的 final 即其 post，自动复用。
    """
    models  = models  or list(C.MODELS)
    methods = methods or C.METHODS
    tasks   = tasks   or C.ORDER
    N = len(C.ORDER)

    for model in models:
        for method in methods:
            for task in tasks:
                j = C.ORDER.index(task) + 1
                task_key, has_image = C.TASK_MAP[task]
                gt = load_gt(task)
                for stage in stages:
                    # 最后一个任务没有独立 final（== post），跳过避免重复
                    if stage == "final" and j == N:
                        continue
                    paths = _candidate_paths(model, method, stage, j, task)
                    path = next((p for p in paths if os.path.isfile(p)), None)
                    meta = {"model": model, "method": method, "task": task,
                            "task_key": task_key, "stage": stage, "j": j,
                            "path": path}
                    if not path:
                        yield meta, []
                        continue
                    samples = []
                    for qid, q_pred, gt_pred, out_text in _read_pred_records(model, path):
                        g = gt.get(qid)
                        question = (q_pred or (g[0] if g else "") or "").strip()
                        ref_ans  = (gt_pred or (g[1] if g else "") or "").strip()
                        img_rel  = g[2] if g else ""
                        cot = extract_cot(out_text)
                        if not cot:
                            continue
                        samples.append({
                            "model": model, "method": method, "task": task,
                            "task_key": task_key, "has_image": has_image,
                            "stage": stage, "question_id": qid,
                            "question": question, "cot_text": cot,
                            "ref_answer": ref_ans,
                            "image_path": _resolve_image(img_rel) if has_image else None,
                        })
                    yield meta, samples
