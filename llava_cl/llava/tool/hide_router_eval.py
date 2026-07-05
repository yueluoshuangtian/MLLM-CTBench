# tools/hide_router_eval.py
import os
import json
import argparse
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"


# ====== 你给的测试集路径 & 图像文件夹 ======
QUESTION_FILE_DICT = {
    "numglue": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/numglue.json",
    "art": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/art.json",
    "math": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/math.json",
    "fomc": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/fomc.json",
    "medical": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/medical.json",
    "OCR": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/OCR.json",
    "science": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/science.json"
}

IMAGE_FOLDER_DICT = {
    "numglue": "",
    "art": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "math": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "fomc": "",
    "medical": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "OCR": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "science": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark"
}


# ====== 小工具 ======
def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """既兼容 list-json 也兼容 jsonl（每行一个 json）"""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        assert isinstance(data, list)
        return data
    except Exception:
        data = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data


class ImageDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        """
        samples: [{"image_path": str}, ...]
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TextDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        """
        samples: [{"text": str}, ...]
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ====== CLIP 特征提取 ======
def extract_image_features(samples: List[Dict[str, Any]],
                           clip_model: CLIPModel,
                           clip_proc: CLIPProcessor,
                           batch_size: int = 32,
                           desc: str = "img") -> torch.Tensor:
    if not samples:
        return torch.empty(0, clip_model.visual_projection.out_features)

    ds = ImageDataset(samples)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    feats = []
    clip_model.eval()
    for batch in tqdm(dl, desc=f"[CLIP] {desc}", leave=False):
        paths = batch["image_path"]
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = clip_proc(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = clip_model.get_image_features(**inputs)  # [B, D]
        out = out / out.norm(dim=-1, keepdim=True)
        feats.append(out.cpu())
    return torch.cat(feats, dim=0)  # [N, D]


def extract_text_features(samples: List[Dict[str, Any]],
                          clip_model: CLIPModel,
                          clip_proc: CLIPProcessor,
                          batch_size: int = 32,
                          desc: str = "txt") -> torch.Tensor:
    if not samples:
        return torch.empty(0, clip_model.text_projection.out_features)

    ds = TextDataset(samples)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    feats = []
    clip_model.eval()
    for batch in tqdm(dl, desc=f"[CLIP] {desc}", leave=False):
        texts = batch["text"]
        inputs = clip_proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = clip_model.get_text_features(**inputs)  # [B, D]
        out = out / out.norm(dim=-1, keepdim=True)
        feats.append(out.cpu())
    return torch.cat(feats, dim=0)  # [N, D]


# ====== 用训练集构造 HiDe anchors ======
def build_train_samples_for_task(task: str, train_data_dir: str, train_image_root: str) -> List[Dict[str, Any]]:
    """
    读取 /home/.../llava/train/{task}.json
    每条样本结构和 LazySupervisedDataset 一样：
      {
        "conversations": [{"from": "human"/"gpt", "value": "..."} , ...],
        "image": "xxx.png"(可选)
      }
    返回： [{"text": "...", "image_path": "...(可选)"}, ...]
    """
    train_path = os.path.join(train_data_dir, f"{task}.json")
    data = json.load(open(train_path, "r"))
    samples = []
    for ex in data:
        convs = ex.get("conversations", [])
        # 把所有 human 的 value 拼起来当成指令文本
        human_texts = [c["value"] for c in convs if c.get("from", "").lower() == "human"]
        if human_texts:
            text = " ".join(human_texts)
        else:
            # 兜底：直接拼所有 value
            text = " ".join(c.get("value", "") for c in convs)
        sample = {"text": text}
        if "image" in ex and ex["image"]:
            img_file = ex["image"]
            sample["image_path"] = os.path.join(train_image_root, img_file)
        samples.append(sample)
    return samples


def compute_hide_anchors(tasks: List[str],
                         train_data_dir: str,
                         train_image_root: str,
                         clip_model: CLIPModel,
                         clip_proc: CLIPProcessor,
                         cache_dir: str,
                         batch_size: int = 32):
    """
    返回两个字典（全部放 CPU）：
      mv[task]   : 图像 anchor 向量 [1, D] 或 None
      mins[task] : 文本 anchor 向量 [1, D]
    同时会缓存到 cache_dir/anchors_clip.pt
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "anchors_clip.pt")
    if os.path.exists(cache_path):
        print(f"[HiDe] 发现 anchors 缓存，直接加载：{cache_path}")
        ckpt = torch.load(cache_path, map_location="cpu")
        return ckpt["mv"], ckpt["mins"]

    mv = {}
    mins = {}

    print("[HiDe] 未发现 anchors 缓存，开始从训练集构建 anchors...")
    for t in tqdm(tasks, desc="[HiDe] build anchors"):
        train_samples = build_train_samples_for_task(t, train_data_dir, train_image_root)

        # 文本样本
        text_samples = [{"text": s["text"]} for s in train_samples]
        z_text = extract_text_features(
            text_samples, clip_model, clip_proc,
            batch_size=batch_size, desc=f"{t}-train-text"
        )
        if z_text.size(0) == 0:
            raise RuntimeError(f"[HiDe] 任务 {t} 没有文本特征，检查训练数据。")
        mins[t] = z_text.mean(dim=0, keepdim=True)  # [1,D]

        # 图像样本（只取有 image_path 的）
        img_samples = [s for s in train_samples if "image_path" in s]
        if len(img_samples) > 0:
            z_img = extract_image_features(
                img_samples, clip_model, clip_proc,
                batch_size=batch_size, desc=f"{t}-train-img"
            )
            mv[t] = z_img.mean(dim=0, keepdim=True)  # [1,D]
        else:
            mv[t] = None  # 没有图像 anchor

    # 保存缓存
    torch.save({"mv": mv, "mins": mins}, cache_path)
    print(f"[HiDe] anchors 已缓存到：{cache_path}")
    return mv, mins


# ====== 构建测试集样本 ======
def build_test_samples_for_task(task: str) -> List[Dict[str, Any]]:
    """
    读取你给的测试集 question_file_dict[task],
    假设每条 json 至少有一个问题字段："question" / "text" / "instruction"
    图像字段假设为 "image"（和训练集一致），
    图像目录从 IMAGE_FOLDER_DICT[task] 取。
    """
    q_path = QUESTION_FILE_DICT[task]
    img_root = IMAGE_FOLDER_DICT[task]
    data = load_json_or_jsonl(q_path)

    samples = []
    for ex in data:
        text = ex.get("question") or ex.get("text") or ex.get("instruction") or ""
        sample = {"text": text}
        if img_root and ex.get("image"):
            sample["image_path"] = os.path.join(img_root, ex["image"])
        samples.append(sample)

    return samples


# ====== HiDe 路由 ======
def hide_predict_task(z_v: torch.Tensor,
                      z_ins: torch.Tensor,
                      mv: Dict[str, torch.Tensor],
                      mins: Dict[str, torch.Tensor],
                      tasks: List[str],
                      alpha: float = 0.5,
                      beta: float = 0.5,
                      temperature: float = 0.07) -> torch.Tensor:
    """
    z_v   : [N, D] 图像特征（若某样本无图像，用全零）
    z_ins : [N, D] 文本特征
    返回：pred_idx: [N]，是 tasks 下标
    """
    N, D = z_ins.shape
    scores = []

    for t in tasks:
        mv_t = mv[t]
        mins_t = mins[t]

        mins_t = mins_t.to(z_ins)
        r_ins = (z_ins * mins_t).sum(dim=-1)  # [N]

        if mv_t is not None:
            mv_t = mv_t.to(z_v)
            r_v = (z_v * mv_t).sum(dim=-1)
        else:
            r_v = torch.zeros_like(r_ins)

        score_t = alpha * r_v + beta * r_ins
        scores.append(score_t.unsqueeze(-1))

    scores = torch.cat(scores, dim=-1)  # [N, T]
    probs = F.softmax(scores / temperature, dim=-1)
    pred_idx = probs.argmax(dim=-1)
    return pred_idx


def compute_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t, p] += 1
    return cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help='任务列表，如 "numglue,art,math,fomc,medical,OCR,science"'
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/train",
        help="训练数据 json 所在目录"
    )
    parser.add_argument(
        "--train_image_root",
        type=str,
        default="/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
        help="训练集图像所在的根目录（和 DataArguments.image_folder 一致）"
    )
    parser.add_argument(
        "--clip_model_path",
        type=str,
        default="openai/clip-vit-large-patch14-336",
        help="CLIP 模型路径（可填本地：/mnt/cxzx/workspace/data_transfer/houzhiyan/models/clip-vit-large-patch14-336）"
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="图像相似度权重")
    parser.add_argument("--beta", type=float, default=0.5, help="文本相似度权重")
    parser.add_argument("--temperature", type=float, default=0.07, help="softmax 温度")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--cache_root",
        type=str,
        default="/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/without_reasoning/hide_router_eval",
        help="用于缓存 anchors / 特征 / 监控数据的根目录"
    )
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"[HiDe] 评测任务顺序: {tasks}")

    # 为该任务顺序单独建一个子目录
    order_name = "_".join(tasks)
    order_cache_dir = os.path.join(args.cache_root, order_name)
    anchors_cache_dir = os.path.join(order_cache_dir, "anchors")
    testfeats_cache_dir = os.path.join(order_cache_dir, "test_feats")
    metrics_dir = os.path.join(order_cache_dir, "metrics")
    os.makedirs(anchors_cache_dir, exist_ok=True)
    os.makedirs(testfeats_cache_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # ================= 1) 加载 CLIP =================
    print(f"[HiDe] 加载 CLIP 模型：{args.clip_model_path}")
    clip_model = CLIPModel.from_pretrained(args.clip_model_path).to(device)
    clip_proc = CLIPProcessor.from_pretrained(args.clip_model_path)

    # ================= 2) anchors（支持缓存） =================
    mv, mins = compute_hide_anchors(
        tasks,
        args.train_data_dir,
        args.train_image_root,
        clip_model,
        clip_proc,
        cache_dir=anchors_cache_dir,
        batch_size=args.batch_size,
    )

    # ================= 3) 测试集特征（支持缓存） =================
    test_feats_path = os.path.join(testfeats_cache_dir, "test_feats_clip.pt")
    if os.path.exists(test_feats_path):
        print(f"[HiDe] 发现测试特征缓存，直接加载：{test_feats_path}")
        ckpt = torch.load(test_feats_path, map_location="cpu")
        all_zv = ckpt["zv"]
        all_zins = ckpt["zins"]
        all_true = ckpt["labels"]
        task_offsets = ckpt["task_offsets"]
    else:
        print("[HiDe] 未发现测试特征缓存，开始从测试集提取特征...")
        all_zv_list, all_zins_list, all_true_list = [], [], []
        task_offsets = {}  # 记录每个 task 在拼接向量里的区间，便于之后分析

        cur_offset = 0
        for ti, t in enumerate(tasks):
            print(f"[HiDe] 提取测试集 {t} 的特征 ...")
            samples = build_test_samples_for_task(t)

            # 文本特征
            text_samples = [{"text": s["text"]} for s in samples]
            z_ins = extract_text_features(
                text_samples, clip_model, clip_proc,
                batch_size=args.batch_size, desc=f"{t}-test-text"
            )

            # 图像特征
            img_samples = []
            for s in samples:
                if "image_path" in s:
                    img_samples.append({"image_path": s["image_path"]})

            if img_samples:
                z_img_partial = extract_image_features(
                    img_samples, clip_model, clip_proc,
                    batch_size=args.batch_size, desc=f"{t}-test-img"
                )
                # 把有图像的样本填到对应位置，其余用 0 向量
                zv_full = torch.zeros_like(z_ins)
                img_idx = 0
                for i, s in enumerate(samples):
                    if "image_path" in s:
                        zv_full[i] = z_img_partial[img_idx]
                        img_idx += 1
                z_v = zv_full
            else:
                z_v = torch.zeros_like(z_ins)

            all_zv_list.append(z_v)
            all_zins_list.append(z_ins)
            labels_t = torch.full((z_ins.size(0),), ti, dtype=torch.long)
            all_true_list.append(labels_t)

            task_offsets[t] = {
                "start": cur_offset,
                "end": cur_offset + z_ins.size(0)
            }
            cur_offset += z_ins.size(0)

        all_zv = torch.cat(all_zv_list, dim=0)
        all_zins = torch.cat(all_zins_list, dim=0)
        all_true = torch.cat(all_true_list, dim=0)

        torch.save(
            {
                "zv": all_zv,
                "zins": all_zins,
                "labels": all_true,
                "task_offsets": task_offsets,
            },
            test_feats_path,
        )
        print(f"[HiDe] 测试集特征已缓存到：{test_feats_path}")

    print(f"[HiDe] 总测试样本数: {all_true.numel()}")

    # ================= 4) 计算路由预测 =================
    print("[HiDe] 开始计算路由预测...")
    pred_idx = hide_predict_task(
        all_zv, all_zins, mv, mins, tasks,
        alpha=args.alpha, beta=args.beta, temperature=args.temperature
    )

    acc = (pred_idx == all_true).float().mean().item()
    print(f"\n[HiDe] 总任务路由准确率: {acc:.4f}")

    # ================= 5) 统计 per-task acc + confusion matrix =================
    num_tasks = len(tasks)
    cm = compute_confusion_matrix(all_true, pred_idx, num_classes=num_tasks)

    per_task_acc = {}
    print("\n[HiDe] 各任务路由准确率：")
    for ti, t in enumerate(tasks):
        mask = (all_true == ti)
        n = int(mask.sum().item())
        if n == 0:
            print(f"  - {t}: (no test samples)")
            per_task_acc[t] = None
            continue
        acc_t = (pred_idx[mask] == all_true[mask]).float().mean().item()
        print(f"  - {t}: {acc_t:.4f}  (样本数 = {n})")
        per_task_acc[t] = acc_t

    # 保存混淆矩阵（行：真实任务；列：预测任务）
    cm_path = os.path.join(metrics_dir, "router_confusion.npy")
    torch.save(cm, cm_path)
    print(f"[HiDe] 混淆矩阵已保存到：{cm_path}")

    # 保存整体 metrics.json
    metrics_path = os.path.join(metrics_dir, "router_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "tasks": tasks,
                "overall_acc": acc,
                "per_task_acc": per_task_acc,
                "task_offsets": task_offsets,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[HiDe] 评测指标已保存到：{metrics_path}")

    # 保存每个样本的预测（便于后续分析具体路由错误）
    pred_log_path = os.path.join(metrics_dir, "router_predictions.jsonl")
    with open(pred_log_path, "w") as f:
        for i in range(all_true.size(0)):
            true_id = int(all_true[i].item())
            pred_id = int(pred_idx[i].item())
            rec = {
                "index": i,
                "true_task_id": true_id,
                "true_task": tasks[true_id],
                "pred_task_id": pred_id,
                "pred_task": tasks[pred_id],
                "correct": bool(true_id == pred_id),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[HiDe] 每样本路由结果已保存到：{pred_log_path}")


if __name__ == "__main__":
    main()
