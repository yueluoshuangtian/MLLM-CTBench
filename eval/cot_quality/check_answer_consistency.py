"""
纯本地启发式核查（不需 API key，现在就能跑出真值）：
统计训练 CoT 里「推理结论」与「标注答案 (,answer:X)」的一致性，作为 GPT-4 标注
噪声的一个廉价上界估计。

注意：这是启发式（字符串匹配），不如 LLM judge 准；正式数字以 LLM 的
answer_consistent 为准，本脚本用于快速预筛 + 给 rebuttal 一个可立即报告的下限/上限。

判定逻辑：对有 ",answer:X" 标记的样本，看答案串 X 是否出现在 CoT 正文尾部
（最后 300 字符）。不出现 -> 标为「潜在不一致」。
"""
import json
import os
import re
import sys

import config
from sample_data import load_json_lenient


def norm(s):
    return re.sub(r"[\s$\\{}().,]", "", str(s)).lower()


def tail_contains(cot_text, answer):
    """答案是否出现在 CoT 尾部（容忍空白/符号差异）。"""
    if not answer:
        return None  # 无显式答案标记，跳过
    tail = norm(cot_text[-300:])
    a = norm(answer)
    if not a:
        return None
    # 选择题字母答案：单字母直接找
    if len(a) <= 3:
        return a in tail
    return a in tail or tail.endswith(a)


def main():
    print(f"{'任务':14s} {'总数':>6s} {'有answer标记':>10s} {'潜在不一致':>10s} {'不一致率':>8s}")
    grand_total = grand_tagged = grand_bad = 0
    for task, (fname, has_image) in config.TASK_FILES.items():
        path = os.path.join(config.DATA_ROOT, fname)
        data = load_json_lenient(path)
        total = tagged = bad = 0
        for it in data:
            convs = it.get("conversations", [])
            if len(convs) < 2:
                continue
            total += 1
            val = convs[1]["value"]
            parts = re.split(r",?\s*answer\s*:", val, flags=re.IGNORECASE)
            if len(parts) < 2:
                continue
            tagged += 1
            cot, ans = parts[0], parts[-1].strip()
            res = tail_contains(cot, ans)
            if res is False:
                bad += 1
        rate = (100 * bad / tagged) if tagged else 0.0
        print(f"{task:14s} {total:6d} {tagged:10d} {bad:10d} {rate:7.1f}%")
        grand_total += total; grand_tagged += tagged; grand_bad += bad
    rate = (100 * grand_bad / grand_tagged) if grand_tagged else 0.0
    print(f"{'合计':14s} {grand_total:6d} {grand_tagged:10d} {grand_bad:10d} {rate:7.1f}%")
    print("\n注：'潜在不一致' = 标注答案未出现在 CoT 推理尾部（启发式，可能有误判）。"
          "\n正式一致性数字以 LLM judge 的 answer_consistent 为准。")


if __name__ == "__main__":
    main()
