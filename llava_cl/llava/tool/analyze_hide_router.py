import os
import json
import argparse
from typing import Dict, Tuple, List

import torch
import numpy as np
import matplotlib.pyplot as plt


def load_metrics(exp_dir: str) -> Tuple[List[str], np.ndarray]:
    """
    从 metrics 目录加载：
      - router_metrics.json 里的任务列表 tasks
      - router_confusion.npy 里的混淆矩阵 (torch.save 存的 tensor)
    """
    metrics_dir = os.path.join(exp_dir, "metrics")
    metrics_path = os.path.join(metrics_dir, "router_metrics.json")
    cm_path = os.path.join(metrics_dir, "router_confusion.npy")

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics json not found: {metrics_path}")
    if not os.path.exists(cm_path):
        raise FileNotFoundError(f"confusion file not found: {cm_path}")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    tasks = metrics["tasks"]

    cm_tensor = torch.load(cm_path, map_location="cpu")
    cm = cm_tensor.numpy()

    return tasks, cm


def plot_confusion_heatmap(
    exp_dir: str,
    normalize: bool = True,
    figsize: Tuple[int, int] = (8, 7),
    cmap: str = "Blues",
    save_name: str = "router_confusion_heatmap.png",
):
    """
    画混淆矩阵热力图：
    - 行：真实任务（True）
    - 列：判别器预测的任务（Pred）
    - normalize=True 时按行归一化成百分比
    """
    tasks, cm = load_metrics(exp_dir)
    num_tasks = len(tasks)

    if normalize:
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(
            cm,
            cm_sum,
            out=np.zeros_like(cm, dtype=float),
            where=cm_sum != 0,
        )
        data = cm_norm
        value_fmt = "{:.2f}"
        title = "Router Confusion (row-normalized)"
    else:
        data = cm.astype(float)
        value_fmt = "{:d}"
        title = "Router Confusion (counts)"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap=cmap)

    # 坐标刻度
    ax.set_xticks(np.arange(num_tasks))
    ax.set_yticks(np.arange(num_tasks))
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_yticklabels(tasks)
    ax.set_xlabel("Predicted Task")
    ax.set_ylabel("True Task")
    ax.set_title(title)

    # 每个格子写数字
    for i in range(num_tasks):
        for j in range(num_tasks):
            val = data[i, j]
            if normalize:
                text = value_fmt.format(val * 100)  # 显示百分比
            else:
                text = value_fmt.format(int(val))
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="black" if val < data.max() * 0.7 else "white",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    metrics_dir = os.path.join(exp_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    save_path = os.path.join(metrics_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"[Analyze] Confusion heatmap saved to: {save_path}")


def analyze_confusion_pairs(
    exp_dir: str,
    topk: int = 10,
):
    """
    从 router_predictions.jsonl 中统计：
    - 最容易混淆的 (true_task -> pred_task) pair（true != pred）
    输出时给出：
      - 该 pair 的错误数量
      - 该 pair 占 true_task 所有样本的比例
    """
    metrics_dir = os.path.join(exp_dir, "metrics")
    pred_path = os.path.join(metrics_dir, "router_predictions.jsonl")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"router_predictions.jsonl not found: {pred_path}")

    # 加载任务列表 & 每个任务的总样本数
    metrics_path = os.path.join(metrics_dir, "router_metrics.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    tasks = metrics["tasks"]

    total_per_true: Dict[str, int] = {t: 0 for t in tasks}
    conf_pair_counts: Dict[Tuple[str, str], int] = {}

    with open(pred_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            true_t = rec["true_task"]
            pred_t = rec["pred_task"]
            correct = rec["correct"]

            total_per_true[true_t] += 1
            if not correct and true_t != pred_t:
                key = (true_t, pred_t)
                conf_pair_counts[key] = conf_pair_counts.get(key, 0) + 1

    if not conf_pair_counts:
        print("[Analyze] No mis-routed samples found (perfect routing?)")
        return

    # 按错误数量排序
    sorted_pairs = sorted(conf_pair_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\n[Analyze] Top-{topk} most confused task pairs (True -> Pred):")
    print("------------------------------------------------------------")
    for i, ((t_true, t_pred), cnt) in enumerate(sorted_pairs[:topk], start=1):
        denom = total_per_true.get(t_true, 1)
        ratio = cnt / max(denom, 1)
        print(
            f"{i:2d}. {t_true:10s} -> {t_pred:10s} | "
            f"count = {cnt:4d},  rate = {ratio*100:6.2f}% "
            f"( {cnt}/{denom} of all {t_true} samples )"
        )

    # 也可以把结果存一下
    pairs_path = os.path.join(metrics_dir, "router_confused_pairs_topk.json")
    out_list = []
    for (t_true, t_pred), cnt in sorted_pairs:
        denom = total_per_true.get(t_true, 1)
        ratio = cnt / max(denom, 1)
        out_list.append(
            {
                "true_task": t_true,
                "pred_task": t_pred,
                "count": cnt,
                "rate": ratio,
                "total_true": denom,
            }
        )
    with open(pairs_path, "w") as f:
        json.dump(out_list, f, indent=2, ensure_ascii=False)
    print(f"\n[Analyze] All confused pairs saved to: {pairs_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help=(
            "某个任务顺序的实验根目录，例如： "
            "/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/without_reasoning/hide_router_eval/numglue_art_math_fomc_medical_OCR_science"
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["heatmap", "pairs", "both"],
        default="both",
        help="执行内容：只画热力图 / 只分析混淆 pair / 两个都做",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="输出最容易混淆的 top-k 对",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="画热力图时不做 row-normalization（直接画数量）",
    )
    args = parser.parse_args()

    if args.mode in ["heatmap", "both"]:
        plot_confusion_heatmap(
            exp_dir=args.exp_dir,
            normalize=not args.no_normalize,
        )

    if args.mode in ["pairs", "both"]:
        analyze_confusion_pairs(
            exp_dir=args.exp_dir,
            topk=args.topk,
        )


if __name__ == "__main__":
    main()
