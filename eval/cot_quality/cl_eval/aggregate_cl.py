"""
聚合 cl_cot_scores.jsonl，产出论文 Table V/VII 口径的逐方法×逐任务表：
  Score(全训完 P_Nj 的 CoT 均分) / Forget(P_Nj - P_jj) / AP(末行均值) / BWT(平均 Forget)。

CoT 总分 = 各适用维度(logic, grounding[仅含图], knowledge)的均值，与论文一致。
输出：outputs/reports/cot_table_<model>.md 和 cot_scores_summary.json
"""
import os, json, statistics
from collections import defaultdict
import cl_config as C

DIMS = ("logic", "grounding", "knowledge")


def overall_of(rec):
    """单条 CoT 总分：适用维度均值（缺维度跳过）。"""
    vals = [rec[d] for d in DIMS if isinstance(rec.get(d), (int, float))]
    return statistics.mean(vals) if vals else None


def load_scores():
    # (model, method, stage, task) -> list of per-example overall
    cell = defaultdict(list)
    dimcell = defaultdict(lambda: defaultdict(list))
    if not os.path.isfile(C.SCORE_FILE):
        return cell, dimcell
    for line in open(C.SCORE_FILE, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("error"):
            continue
        ov = overall_of(r)
        if ov is None:
            continue
        k = (r["model"], r["method"], r["stage"], r["task"])
        cell[k].append(ov)
        for d in DIMS:
            if isinstance(r.get(d), (int, float)):
                dimcell[(r["model"], r["method"], r["task"])][d].append(r[d])
    return cell, dimcell


def mean(xs):
    return statistics.mean(xs) if xs else None


def fmt(x, plus=False):
    if x is None:
        return "  –  "
    return (f"{x:+.2f}" if plus else f"{x:.2f}")


def build_tables():
    cell, dimcell = load_scores()
    summary = {}
    for model in C.MODELS:
        lines = []
        order = C.ORDER
        N = len(order)
        header = "| 方法 | " + " | ".join(order) + " | AP | BWT |"
        sep = "|" + "---|" * (N + 3)
        lines += [f"### {model.upper()}  CoT 过程级质量（Score / Forget）", "",
                  "> Score=全训完后该任务 CoT 总分(P_Nj)；Forget=P_Nj − P_jj（负=遗忘）。",
                  "", header, sep]
        for method in C.METHODS:
            post = {t: mean(cell.get((model, method, "post", t), [])) for t in order}
            # final：除最后一个任务外取 final；最后一个任务 final==post
            fin = {}
            for idx, t in enumerate(order):
                if idx == N - 1:
                    fin[t] = post[t]
                else:
                    fin[t] = mean(cell.get((model, method, "final", t), []))
            cells = []
            forgets = []
            for idx, t in enumerate(order):
                s, p = fin[t], post[t]
                if s is None:
                    cells.append("  –  ")
                    continue
                if idx == N - 1:
                    cells.append(f"{s:.2f} / –")     # 末任务无 forget
                else:
                    fg = (s - p) if (s is not None and p is not None) else None
                    if fg is not None:
                        forgets.append(fg)
                    cells.append(f"{s:.2f} / {fmt(fg, plus=True)}")
            ap_vals = [fin[t] for t in order if fin[t] is not None]
            AP = mean(ap_vals)
            BWT = mean(forgets) if forgets else None
            lines.append(f"| {method} | " + " | ".join(cells) +
                         f" | {fmt(AP)} | {fmt(BWT, plus=True)} |")
            summary[f"{model}/{method}"] = {
                "post": post, "final": fin, "AP": AP, "BWT": BWT,
            }
        out = os.path.join(C.REPORT_DIR, f"cot_table_{model}.md")
        open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(f"[saved] {out}")
        print("\n".join(lines)); print()

    json.dump(summary, open(os.path.join(C.REPORT_DIR, "cot_scores_summary.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"[saved] {os.path.join(C.REPORT_DIR, 'cot_scores_summary.json')}")


if __name__ == "__main__":
    build_tables()
