# tools/run_mora_router_eval.py
import os
import json
import math
import argparse
import random
from datetime import datetime
from typing import List, Dict, Any, Optional

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from transformers import CLIPModel, CLIPProcessor
from llava.constants import DEFAULT_IMAGE_TOKEN


# ----------------- 全局配置 -----------------
TASKS = ["numglue", "art", "math", "fomc", "medical", "OCR", "science"]

QUESTION_FILE_DICT = {
    "numglue": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/numglue.json",
    "art": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/art.json",
    "math": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/math.json",
    "fomc": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/fomc.json",
    "medical": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/medical.json",
    "OCR": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/OCR.json",
    "science": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/science.json",
}

# 这里暂时不需要 annotation_file_dict 做路由评估，用不到答案
IMAGE_FOLDER_DICT = {
    "numglue": "",
    "art": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "math": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "fomc": "",
    "medical": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "OCR": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "science": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
}


# ===================== MoRA 风格 Router 定义 =====================
class MoRAStyleRouter(nn.Module):
    """
    简化版 MoRA 风格判别器：
    - A: [R, D] rank-1 key 向量（原始 MoRA 的 “rank-1 自激活” 信息）
    - gate: 对 A x 做归一化 + softmax 得到 w
    - 然后用 w -> 线性 -> task logits
    """
    def __init__(self, dim: int, num_ranks: int, num_tasks: int, tau: float = 1.0):
        super().__init__()
        self.A = nn.Parameter(torch.randn(num_ranks, dim) * 0.02)
        self.tau = tau
        self.fc = nn.Linear(num_ranks, num_tasks)

    def forward(self, x: torch.Tensor):
        # x: [B, D]
        # A x^T: [R, D] x [D, B] -> [R, B] -> [B, R]
        s = (self.A @ x.transpose(0, 1)).transpose(0, 1)  # [B, R]

        # 归一化（模仿 MoRA 中的 “能量” 归一）
        denom = torch.sqrt((s ** 2).sum(dim=-1, keepdim=True) + 1e-8)
        s = s / denom

        w = F.softmax(s / self.tau, dim=-1)  # [B, R]
        logits = self.fc(w)                  # [B, num_tasks]
        return logits, w


# ===================== 一些小工具 =====================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ===================== 特征提取相关 =====================
class MixedSampleDataset(Dataset):
    """
    用于特征提取：每个样本是 {image_path: str or None, text: str}
    """
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def extract_clip_features(
    samples: List[Dict[str, Any]],
    clip_model: CLIPModel,
    clip_proc: CLIPProcessor,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    对一组样本提 CLIP 特征：
    - 如果有 image & text：image_feat + text_feat 后再归一化
    - 如果只有 image：用 image_feat
    - 如果只有 text：用 text_feat
    返回 shape: [N, D]
    """
    ds = MixedSampleDataset(samples)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_feats = []
    clip_model.eval()

    for batch in tqdm(dl, desc="Extracting CLIP features", ncols=100):
        texts = batch["text"]
        image_paths = batch["image_path"]

        # 文本特征
        text_feats = None
        if texts is not None:
            # 允许有部分空文本
            texts_list = [t if t is not None else "" for t in texts]
            inputs_text = clip_proc(text=texts_list, return_tensors="pt", padding=True, truncation=True)
            inputs_text = {k: v.to(device) for k, v in inputs_text.items()}
            with torch.no_grad():
                text_feats = clip_model.get_text_features(**inputs_text)  # [B, D]
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        # 图像特征
        image_feats = None
        if image_paths is not None:
            imgs = []
            has_any_image = False
            for p in image_paths:
                if p is not None and len(p) > 0 and os.path.isfile(p):
                    imgs.append(Image.open(p).convert("RGB"))
                    has_any_image = True
                else:
                    imgs.append(None)
            if has_any_image:
                valid_indices = [i for i, img in enumerate(imgs) if img is not None]
                valid_imgs = [imgs[i] for i in valid_indices]
                inputs_img = clip_proc(images=valid_imgs, return_tensors="pt")
                inputs_img = {k: v.to(device) for k, v in inputs_img.items()}
                with torch.no_grad():
                    img_feats_all = clip_model.get_image_features(**inputs_img)
                    img_feats_all = img_feats_all / img_feats_all.norm(dim=-1, keepdim=True)

                # 填回到 batch 大小
                image_feats = torch.zeros(len(imgs), img_feats_all.size(-1), device=device)
                cnt = 0
                for i in range(len(imgs)):
                    if imgs[i] is not None:
                        image_feats[i] = img_feats_all[cnt]
                        cnt += 1
                    else:
                        image_feats[i] = 0.0
            else:
                image_feats = None

        # 融合
        if image_feats is not None and text_feats is not None:
            feats = image_feats + text_feats
        elif image_feats is not None:
            feats = image_feats
        elif text_feats is not None:
            feats = text_feats
        else:
            raise RuntimeError("Both image and text are None for a batch.")

        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        all_feats.append(feats.cpu())

    all_feats = torch.cat(all_feats, dim=0)
    return all_feats


# ===================== 读训练集 / 测试集 =====================
def load_train_samples_for_task(
    task: str,
    train_data_dir: str,
    image_root_default: str = "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    从 /home/.../llava/train/{task}.json 读取：
    每个元素类似：
    {
      "id": ...,
      "image": "xxx.png" (optional),
      "conversations": [
        {"from": "human", "value": "... <image> ..."},
        {"from": "gpt", "value": "..."},
        ...
      ]
    }
    我们提取：
    - text: 第一个 human 的 value 去掉 DEFAULT_IMAGE_TOKEN
    - image_path: 如果有 image 字段，就拼上 image_root_default
    """
    path = os.path.join(train_data_dir, f"{task}.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for ex in data:
        convs = ex.get("conversations", [])
        text = None
        for s in convs:
            if str(s.get("from", "")).lower() == "human":
                text = s.get("value", "")
                break
        if text is None and len(convs) > 0:
            text = convs[0].get("value", "")
        if text is None:
            text = ""

        # 去掉 image token
        text = text.replace(DEFAULT_IMAGE_TOKEN, "").strip()

        img_name = ex.get("image", None)
        if img_name and len(img_name) > 0:
            image_path = os.path.join(image_root_default, img_name)
        else:
            image_path = None

        samples.append({"text": text, "image_path": image_path})

    if max_samples is not None and len(samples) > max_samples:
        samples = samples[:max_samples]

    return samples


def guess_text_from_test_item(ex: Dict[str, Any]) -> str:
    """
    测试集的 JSON 结构不同任务可能不一样，这里做一些常见字段的猜测：
    如果你的实际字段不同，可以改这里。

    尝试顺序：
    - "question"
    - "input"
    - "instruction"
    - "conversations"（用第一个 human）
    - 否则转成 str(ex)
    """
    if "question" in ex:
        return str(ex["question"])
    if "input" in ex:
        return str(ex["input"])
    if "instruction" in ex:
        return str(ex["instruction"])
    if "conversations" in ex:
        convs = ex["conversations"]
        for s in convs:
            if str(s.get("from", "")).lower() == "human":
                return str(s.get("value", ""))
        if len(convs) > 0:
            return str(convs[0].get("value", ""))
    # fallback
    return str(ex)


def guess_image_from_test_item(ex: Dict[str, Any]) -> Optional[str]:
    """
    测试集里常见的 image 字段：
    - "image"
    - "image_path"
    如果没有，就返回 None
    """
    if "image" in ex:
        return ex["image"]
    if "image_path" in ex:
        return ex["image_path"]
    return None


def load_test_samples_for_task(
    task: str,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    从 QUESTION_FILE_DICT[task] 读测试样本：
    - text: 调用 guess_text_from_test_item
    - image_path: if IMAGE_FOLDER_DICT[task] != "" and item 有 image 文件名
    """
    q_path = QUESTION_FILE_DICT[task]
    img_root = IMAGE_FOLDER_DICT.get(task, "")

    with open(q_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for ex in data:
        text = guess_text_from_test_item(ex)
        img_name = guess_image_from_test_item(ex)
        if img_root and img_name:
            image_path = os.path.join(img_root, img_name)
        else:
            image_path = None

        samples.append({"text": text, "image_path": image_path})

    if max_samples is not None and len(samples) > max_samples:
        samples = samples[:max_samples]

    return samples


# ===================== 训练 & 评估 =====================
def train_mora_router(
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    dim: int,
    num_tasks: int,
    num_ranks: int = 32,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 128,
    device: torch.device = torch.device("cuda"),
) -> (MoRAStyleRouter, List[Dict[str, float]]):
    """
    训练 MoRA 风格 router，返回模型和每个 epoch 的统计信息
    """
    router = MoRAStyleRouter(dim=dim, num_ranks=num_ranks, num_tasks=num_tasks, tau=1.0).to(device)
    opt = torch.optim.Adam(router.parameters(), lr=lr)

    ds = TensorDataset(z_train, y_train)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    history = []

    for ep in range(1, epochs + 1):
        router.train()
        total_loss = 0.0
        correct = 0
        n_samples = 0

        pbar = tqdm(dl, desc=f"[Train] Epoch {ep}/{epochs}", ncols=100)
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            logits, _ = router(xb)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                total_loss += loss.item() * xb.size(0)
                pred = logits.argmax(dim=-1)
                correct_batch = (pred == yb).sum().item()
                correct += correct_batch
                n_samples += xb.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / max(1, n_samples):.4f}"})

        epoch_loss = total_loss / max(1, n_samples)
        epoch_acc = correct / max(1, n_samples)
        print(f"[Train] epoch={ep}, loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")

        history.append({"epoch": ep, "loss": epoch_loss, "acc": epoch_acc})

    return router.cpu(), history


def evaluate_router(
    router: MoRAStyleRouter,
    z: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    task_names: List[str],
) -> Dict[str, Any]:
    """
    评估 router：
    - overall acc
    - 每个任务 acc
    - 混淆矩阵（num_tasks x num_tasks）
    """
    router = router.to(device)
    router.eval()

    with torch.no_grad():
        logits, w = router(z.to(device))
        pred = logits.argmax(dim=-1).cpu()

    num_tasks = len(task_names)
    n = y.size(0)
    overall_acc = (pred == y).float().mean().item()

    # 混淆矩阵
    conf = [[0 for _ in range(num_tasks)] for _ in range(num_tasks)]
    for t_true, t_pred in zip(y.tolist(), pred.tolist()):
        conf[t_true][t_pred] += 1

    # 每任务 acc
    per_task_acc = {}
    for tid in range(num_tasks):
        mask = (y == tid)
        cnt = mask.sum().item()
        if cnt == 0:
            per_task_acc[task_names[tid]] = None
        else:
            correct = ((pred == y) & mask).sum().item()
            per_task_acc[task_names[tid]] = correct / cnt

    print("========== MoRA Router Evaluation ==========")
    print(f"Overall test acc: {overall_acc:.4f}")
    print("Per-task acc:")
    for t in task_names:
        v = per_task_acc[t]
        if v is None:
            print(f"  {t}: N/A (no samples)")
        else:
            print(f"  {t}: {v:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    header = "        " + " ".join([f"{t[:6]:>6}" for t in task_names])
    print(header)
    for i, t in enumerate(task_names):
        row = f"{t[:6]:>6} " + " ".join([f"{conf[i][j]:6d}" for j in range(num_tasks)])
        print(row)

    return {
        "overall_acc": overall_acc,
        "per_task_acc": per_task_acc,
        "confusion_matrix": {
            "task_names": task_names,
            "matrix": conf,
        },
    }


# ===================== 主流程 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str,
                        default="/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/train")
    parser.add_argument("--output_root", type=str,
                        default="/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/without_reasoning")
    parser.add_argument("--tasks", type=str,
                        default="numglue,art,math,fomc,medical,OCR,science")
    parser.add_argument("--max_train_per_task", type=int, default=None,
                        help="每个任务最多用多少训练样本（None 表示全用）")
    parser.add_argument("--max_test_per_task", type=int, default=None,
                        help="每个任务最多用多少测试样本（None 表示全用）")
    parser.add_argument("--num_ranks", type=int, default=32,
                        help="MoRA rank 数量")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    task_names = [t.strip() for t in args.tasks.split(",") if len(t.strip()) > 0]
    num_tasks = len(task_names)

    # 创建输出目录：/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/without_reasoning/mora_router_YYYYMMDD_HHMMSS
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_root, f"mora_router_{ts}")
    ensure_dir(out_dir)
    print(f"[INFO] Output dir: {out_dir}")

    # 保存 config
    config = vars(args)
    config["task_names"] = task_names
    save_json(config, os.path.join(out_dir, "config.json"))

    # 1) 加载 CLIP
    print("[INFO] Loading CLIP model (openai/clip-vit-large-patch14-336)...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    # 2) 构造训练样本：从 train_data_dir 读取
    print("[INFO] Loading train samples for each task...")
    train_samples_by_task: Dict[str, List[Dict[str, Any]]] = {}
    for t in task_names:
        print(f"  - Task {t}: loading train json...")
        samples = load_train_samples_for_task(
            task=t,
            train_data_dir=args.train_data_dir,
            max_samples=args.max_train_per_task,
        )
        train_samples_by_task[t] = samples
        print(f"    -> {len(samples)} train samples")

    # 3) 构造测试样本：从 QUESTION_FILE_DICT 读取
    print("[INFO] Loading test samples for each task...")
    test_samples_by_task: Dict[str, List[Dict[str, Any]]] = {}
    for t in task_names:
        print(f"  - Task {t}: loading test json...")
        samples = load_test_samples_for_task(
            task=t,
            max_samples=args.max_test_per_task,
        )
        test_samples_by_task[t] = samples
        print(f"    -> {len(samples)} test samples")

    # 4) 提取训练特征 + 标签
    print("[INFO] Extracting CLIP features for TRAIN...")
    z_tr_list = []
    y_tr_list = []
    for tid, t in enumerate(task_names):
        print(f"  >> Task {t}:")
        z_t = extract_clip_features(train_samples_by_task[t], clip_model, clip_proc, device)
        z_tr_list.append(z_t)
        y_tr_list.append(torch.full((z_t.size(0),), tid, dtype=torch.long))
    z_train = torch.cat(z_tr_list, dim=0)
    y_train = torch.cat(y_tr_list, dim=0)
    print(f"[INFO] Train features shape: {z_train.shape}, labels: {y_train.shape}")

    # 5) 提取测试特征 + 标签
    print("[INFO] Extracting CLIP features for TEST...")
    z_te_list = []
    y_te_list = []
    for tid, t in enumerate(task_names):
        print(f"  >> Task {t}:")
        z_t = extract_clip_features(test_samples_by_task[t], clip_model, clip_proc, device)
        z_te_list.append(z_t)
        y_te_list.append(torch.full((z_t.size(0),), tid, dtype=torch.long))
    z_test = torch.cat(z_te_list, dim=0)
    y_test = torch.cat(y_te_list, dim=0)
    print(f"[INFO] Test features shape: {z_test.shape}, labels: {y_test.shape}")

    # 6) 训练 MoRA router
    print("[INFO] Start training MoRA-style router...")
    router, history = train_mora_router(
        z_train=z_train,
        y_train=y_train,
        dim=z_train.size(1),
        num_tasks=num_tasks,
        num_ranks=args.num_ranks,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
    )
    save_json(history, os.path.join(out_dir, "train_metrics.json"))
    torch.save(router.state_dict(), os.path.join(out_dir, "router.pt"))
    print("[INFO] Router training finished and saved.")

    # 7) 评估
    print("[INFO] Evaluating MoRA router on TEST set...")
    metrics = evaluate_router(
        router=router,
        z=z_test,
        y=y_test,
        device=device,
        task_names=task_names,
    )
    save_json(metrics, os.path.join(out_dir, "test_metrics.json"))
    save_json(metrics["confusion_matrix"], os.path.join(out_dir, "confusion_matrix.json"))
    print("[INFO] All done.")


if __name__ == "__main__":
    main()
