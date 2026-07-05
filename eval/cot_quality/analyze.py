"""
统计分析 scores.jsonl，产出 quality_report.md：
  统计1 逐任务 3-judge overall 均分
  统计2 逐维度均分 (logic/grounding/knowledge)
  统计3 分值分布 + 达标率 (>=75) / 低质率 (<50)
  统计4 三 judge 一致性 (两两 Spearman；若装了 krippendorff 则附 alpha)
依赖: numpy, scipy（统计4）。
"""
import json
from collections import defaultdict

import config

try:
    import numpy as np
    from scipy.stats import spearmanr
    HAVE_STATS = True
except Exception:
    HAVE_STATS = False


def load_scores():
    rows = []
    with open(config.SCORE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if not r.get("error") and r.get("overall") is not None:
                    rows.append(r)
            except Exception:
                continue
    return rows


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(x):
    return f"{x:.1f}" if x == x else "-"  # nan 显示 -


def main():
    rows = load_scores()
    judges = list(config.JUDGES.keys())
    tasks = list(config.TASK_FILES.keys())
    lines = ["# 训练 CoT 质量审计报告\n",
             f"有效打分记录: {len(rows)}  |  judge: {', '.join(judges)}\n"]

    # ---- 统计1: 逐任务 overall 均分（按 judge + 3-judge 平均）----
    lines.append("\n## 统计1 逐任务 overall 质量均分\n")
    header = "| 任务 | n | " + " | ".join(judges) + " | 3-judge均值 |"
    lines.append(header)
    lines.append("|" + "---|" * (len(judges) + 3))
    for task in tasks:
        trow = [r for r in rows if r["task"] == task]
        n_samples = len({r["sample_id"] for r in trow})
        per_judge = [mean([r["overall"] for r in trow if r["judge"] == j]) for j in judges]
        allm = mean([r["overall"] for r in trow])
        lines.append(f"| {task} | {n_samples} | "
                     + " | ".join(fmt(x) for x in per_judge) + f" | **{fmt(allm)}** |")
    overall_all = mean([r["overall"] for r in rows])
    lines.append(f"| **全部** | {len({r['sample_id'] for r in rows})} | "
                 + " | ".join(fmt(mean([r['overall'] for r in rows if r['judge'] == j])) for j in judges)
                 + f" | **{fmt(overall_all)}** |")

    # ---- 统计2: 逐维度均分 ----
    lines.append("\n## 统计2 逐维度均分（全部任务，3-judge 合并）\n")
    for dim in ["logic", "grounding", "knowledge"]:
        lines.append(f"- {dim}: **{fmt(mean([r.get(dim) for r in rows]))}**")

    # ---- 统计3: 分值分布 + 达标率 ----
    lines.append("\n## 统计3 overall 分值分布与达标率\n")
    bins = {"0-25": 0, "26-50": 0, "51-75": 0, "76-100": 0}
    for r in rows:
        v = r["overall"]
        if v <= 25: bins["0-25"] += 1
        elif v <= 50: bins["26-50"] += 1
        elif v <= 75: bins["51-75"] += 1
        else: bins["76-100"] += 1
    tot = sum(bins.values()) or 1
    for k, v in bins.items():
        lines.append(f"- {k}: {v} ({100*v/tot:.1f}%)")
    ge75 = sum(1 for r in rows if r["overall"] >= 75) / tot * 100
    lt50 = sum(1 for r in rows if r["overall"] < 50) / tot * 100
    lines.append(f"\n**达标率 (>=75): {ge75:.1f}%  |  低质率 (<50): {lt50:.1f}%**")
    # CoT-答案一致性（LLM 判断）
    cons = [r.get("answer_consistent") for r in rows if r.get("answer_consistent") is not None]
    if cons:
        rate = sum(1 for c in cons if c) / len(cons) * 100
        lines.append(f"**judge 判定 CoT 与结论答案一致率: {rate:.1f}%**")

    # ---- 统计4: 三 judge 一致性 ----
    lines.append("\n## 统计4 三 judge 一致性\n")
    if HAVE_STATS:
        # 构造 sample_id × judge 的 overall 矩阵
        by_sample = defaultdict(dict)
        for r in rows:
            by_sample[r["sample_id"]][r["judge"]] = r["overall"]
        for a in range(len(judges)):
            for b in range(a + 1, len(judges)):
                ja, jb = judges[a], judges[b]
                pairs = [(v[ja], v[jb]) for v in by_sample.values() if ja in v and jb in v]
                if len(pairs) >= 3:
                    rho, _ = spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
                    lines.append(f"- Spearman ρ ({ja} vs {jb}), n={len(pairs)}: **{rho:.3f}**")
        try:
            import krippendorff
            mat = [[by_sample[s].get(j, np.nan) for s in by_sample] for j in judges]
            alpha = krippendorff.alpha(reliability_data=mat, level_of_measurement="interval")
            lines.append(f"- Krippendorff's α (interval): **{alpha:.3f}**")
        except Exception:
            lines.append("- (未安装 krippendorff 包，跳过 α；pip install krippendorff 可补)")
    else:
        lines.append("- (未安装 numpy/scipy，跳过一致性统计；pip install numpy scipy)")

    report = "\n".join(lines) + "\n"
    with open(config.REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"报告已写入 {config.REPORT_FILE}")


if __name__ == "__main__":
    main()
