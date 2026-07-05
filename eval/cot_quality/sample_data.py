"""
从 7 个任务的 *_reasoning_samples.json 分层抽样 TOTAL_SAMPLES(=400) 条训练 CoT。
- 任务间均衡分层（每任务 ~400/7 条），保证每个任务可单独统计。
- 含子类的任务（science/math/OCR）在任务内再按子类分层，保证多样性覆盖。
- 清洗 science json 的非法控制字符。
- 解析 image 路径（相对 DATA_ROOT），记录是否可用。
- 拆分 CoT 正文与结论答案（若以 ",answer:X" 结尾）。
输出到 SAMPLE_FILE。
"""
import json
import os
import re
import random

import config
from rubrics import detect_subtype


def load_json_lenient(path):
    """容忍非法控制字符地加载 json（science 那份需要）。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    # strict=False 允许字符串内出现未转义的换行/制表等控制字符（science 那份需要）
    try:
        return json.loads(raw, strict=False)
    except json.JSONDecodeError:
        return salvage_array(raw)


def salvage_array(raw):
    """逐对象解析顶层数组，遇到损坏条目就跳到下一个对象起始，丢弃坏条目。
    science_reasoning_samples.json 里有个别 value 字符串缺结束引号导致结构损坏。"""
    dec = json.JSONDecoder(strict=False)
    n = len(raw)
    i = raw.find("[")
    i = (i + 1) if i != -1 else 0
    out, dropped = [], 0
    while i < n:
        while i < n and raw[i] in " \t\r\n,":
            i += 1
        if i >= n or raw[i] == "]":
            break
        try:
            obj, end = dec.raw_decode(raw, i)
            out.append(obj)
            i = end
        except json.JSONDecodeError:
            dropped += 1
            nxt = raw.find("\n    {", i + 1)
            if nxt == -1:
                nxt = raw.find("{", i + 1)
            if nxt == -1:
                break
            i = nxt
    if dropped:
        print(f"  [salvage] 跳过 {dropped} 个损坏条目，恢复 {len(out)} 条")
    return out


# VQA 文件把 CoT 推理拼在 human turn 的"答案格式指令"之后，gpt turn 只放答案。
# 下列指令短语用于切出 question 与 cot。
INSTRUCTION_MARKERS = [
    "answer with the option's letter from the given choices directly",
    "answer with the option's letter",
    "answer using a single word or phrase",
    "using a single word or phrase",
    "answer the question briefly",
    "then answer briefly",
    "answer briefly",
]


def split_human_cot(human):
    """从 human turn 切出 (question, cot)。VQA 的 CoT 在指令短语之后。"""
    h = human.replace("<image>", "").strip()
    hl = h.lower()
    pos = -1
    for mk in INSTRUCTION_MARKERS:
        i = hl.find(mk)
        if i != -1:
            pos = max(pos, i + len(mk))  # 取最靠后的指令结尾
    if pos != -1 and len(h) - pos > 15:   # 指令后还有实质内容才算 CoT
        return h[:pos].strip(" .\n"), h[pos:].lstrip(" .\n").strip()
    # 兜底（如 art：问题后直接接分析）：首行=问题，其余=CoT
    lines = [l for l in h.split("\n") if l.strip()]
    if len(lines) >= 2:
        return lines[0].strip(), " ".join(lines[1:]).strip()
    return h, ""


def parse_sample(task, has_image, item):
    """从一条原始样本抽出 question / cot / ref_answer / image 路径。
    兼容两种格式：文本任务 CoT 在 gpt turn；VQA 任务 CoT 在 human turn。"""
    convs = item.get("conversations", [])
    if len(convs) < 2:
        return None
    human_raw = convs[0]["value"]
    gpt_val = convs[1]["value"].strip()
    ref_answer = None
    if len(gpt_val) >= 40:
        # 文本任务风格：CoT 在 gpt turn
        question = human_raw.replace("<image>", "").strip()
        cot_text = gpt_val
        m = re.split(r",?\s*answer\s*:", gpt_val, flags=re.IGNORECASE)
        if len(m) >= 2:
            cot_text = m[0].strip().rstrip(",")
            ref_answer = m[-1].strip()
    else:
        # VQA 风格：CoT 在 human turn，gpt turn 是最终答案
        question, cot_text = split_human_cot(human_raw)
        ref_answer = gpt_val
        if not cot_text:   # 确实没有推理，丢弃该样本
            return None
    image_rel = item.get("image") if has_image else None
    image_abs = os.path.join(config.DATA_ROOT, image_rel) if image_rel else None
    image_ok = bool(image_abs and os.path.isfile(image_abs))
    return {
        "id": item.get("id"),
        "task": task,
        "subtype": detect_subtype(task, image_rel or "", item.get("id", "")),
        "question": question,
        "cot_text": cot_text,
        "ref_answer": ref_answer,
        "image_path": image_abs if image_ok else None,
        "image_missing": bool(has_image and not image_ok),
    }


def stratified_sample_task(task, has_image, n_target, rng=None):
    """对单个任务做（子类）分层抽样，返回 n_target 条 parsed 样本。
    用每任务独立种子，保证单任务抽样稳定、不受其它任务影响。"""
    rng = random.Random(f"{config.SEED}-{task}")
    path = os.path.join(config.DATA_ROOT, config.TASK_FILES[task][0])
    data = load_json_lenient(path)
    parsed = [p for p in (parse_sample(task, has_image, it) for it in data) if p]
    # 按子类分桶
    buckets = {}
    for p in parsed:
        buckets.setdefault(p["subtype"] or "_", []).append(p)
    # 子类间尽量均匀分配名额
    keys = list(buckets.keys())
    rng.shuffle(keys)
    per = max(1, n_target // len(keys))
    chosen = []
    for k in keys:
        pool = buckets[k]
        rng.shuffle(pool)
        chosen.extend(pool[:per])
    # 不足/超出再随机补齐到 n_target
    if len(chosen) < n_target:
        rest = [p for p in parsed if p not in chosen]
        rng.shuffle(rest)
        chosen.extend(rest[: n_target - len(chosen)])
    rng.shuffle(chosen)
    return chosen[:n_target]


def main():
    rng = random.Random(config.SEED)
    tasks = list(config.TASK_FILES.keys())
    per_task = config.TOTAL_SAMPLES // len(tasks)
    remainder = config.TOTAL_SAMPLES - per_task * len(tasks)
    all_samples = []
    for i, task in enumerate(tasks):
        n = per_task + (1 if i < remainder else 0)
        has_image = config.TASK_FILES[task][1]
        s = stratified_sample_task(task, has_image, n, rng)
        all_samples.extend(s)
        missing = sum(1 for x in s if x["image_missing"])
        print(f"{task:14s} 抽样 {len(s):3d} 条"
              f"{'  (图片缺失 %d)' % missing if missing else ''}")
    with open(config.SAMPLE_FILE, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"\n共 {len(all_samples)} 条 -> {config.SAMPLE_FILE}")


if __name__ == "__main__":
    main()
