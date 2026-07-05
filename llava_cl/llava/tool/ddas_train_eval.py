#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm  # 新增：用于实时进度展示


# ================== 你的测试集配置 ==================
question_file_dict = {
    "numglue": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/numglue.json",
    "art": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/art.json",
    "math": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/math.json",
    "fomc": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/fomc.json",
    "medical": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/medical.json",
    "OCR": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/OCR.json",
    "science": "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/test/science.json",
}

image_folder_dict = {
    "numglue": "",
    "art": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "math": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "fomc": "",
    "medical": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "OCR": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
    "science": "/mnt/cxzx/workspace/data_transfer/houzhiyan/clmm-benchmark",
}

# 训练数据所在目录（和 DataArguments.data_dir 一致）
TRAIN_DATA_DIR = "/mnt/cxzx/workspace/data_transfer/houzhiyan/qwen_data/train"

DEFAULT_TASKS = "numglue,art,math,fomc,medical,OCR,science"


# ================== 简单 Autoencoder 定义 ==================
class AutoEncoder(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        x_hat = self.decoder(h)
        return x_hat


# ================== Dataset & 特征抽取 ==================
class ImagePathDataset(Dataset):
    def __init__(self, image_paths: List[str]):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return {"image_path": self.image_paths[idx]}


class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx]}


def extract_image_features(
    image_paths: List[str],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
) -> torch.Tensor:
    """
    给定一串 image_paths，用 CLIP 抽取图像特征，返回 [N, D] 的 Tensor（L2-normalized）
    """
    if len(image_paths) == 0:
        return torch.empty(0, clip_model.config.projection_dim)

    ds = ImagePathDataset(image_paths)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    feats = []
    clip_model.eval()
    with torch.no_grad():
        for batch in tqdm(dl, desc="Extract image feats", total=len(dl)):
            paths = batch["image_path"]
            images = [Image.open(p).convert("RGB") for p in paths]
            inputs = clip_processor(images=images, return_tensors="pt").to(device)
            out = clip_model.get_image_features(**inputs)  # [B, D]
            out = out / out.norm(dim=-1, keepdim=True)     # L2 normalize
            feats.append(out.cpu())
    return torch.cat(feats, dim=0)


def extract_text_features(
    texts: List[str],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    batch_size: int = 64,
    num_workers: int = 0,
    device: str = "cuda",
) -> torch.Tensor:
    """
    给定一串文本，用 CLIP 文本 encoder 抽特征，返回 [N, D]（L2-normalized）
    """
    if len(texts) == 0:
        return torch.empty(0, clip_model.config.projection_dim)

    ds = TextDataset(texts)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    feats = []
    clip_model.eval()
    with torch.no_grad():
        for batch in tqdm(dl, desc="Extract text feats", total=len(dl)):
            t_list = batch["text"]
            inputs = clip_processor(text=t_list, return_tensors="pt", padding=True, truncation=True).to(device)
            out = clip_model.get_text_features(**inputs)  # [B, D]
            out = out / out.norm(dim=-1, keepdim=True)
            feats.append(out.cpu())
    return torch.cat(feats, dim=0)


# ================== JSON 读取 & 文本抽取 ==================
def load_json_or_ndjson(path: str):
    """
    兼容两种格式：
    1) 一个 list 的 json
    2) 每行一个 json（ndjson）
    """
    with open(path, "r") as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        else:
            return [data]
    except Exception:
        # 当成 ndjson
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return [json.loads(l) for l in lines]


def extract_text_from_conversations(conv_list: List[Dict]) -> Optional[str]:
    """
    从 LLaVA 格式的 conversations 里抽人类文本。
    策略：拼接所有 human 的 value。
    """
    if not isinstance(conv_list, list):
        return None
    human_texts = [c.get("value", "") for c in conv_list if str(c.get("from", "")).lower() == "human"]
    if human_texts:
        return " ".join(human_texts)
    # 没有人类标记，就把所有 value 拼起来兜底
    all_texts = [c.get("value", "") for c in conv_list]
    if all_texts:
        return " ".join(all_texts)
    return None


def extract_text_from_example(ex: Dict, is_train: bool = False) -> Optional[str]:
    """
    通用文本抽取函数：
    - 优先从 conversations（train 格式 / 有些 test 也可能有）拿；
    - 否则尝试 question / text / instruction / input / prompt。
    """
    if isinstance(ex, dict) and "conversations" in ex:
        txt = extract_text_from_conversations(ex["conversations"])
        if txt and txt.strip():
            return txt.strip()

    # 优先级顺序可以按需求调整
    for key in ["question", "text", "instruction", "input", "prompt"]:
        v = ex.get(key, None)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return None


# ================== 从 train / test 抽取图像 & 文本 ==================
def load_train_image_paths_for_task(task: str) -> List[str]:
    """
    只要训练集中有 image 字段且 image_folder_dict[task] 非空，就算一个样本。
    """
    train_file = os.path.join(TRAIN_DATA_DIR, f"{task}.json")
    if not os.path.isfile(train_file):
        print(f"[WARN] train file not found for task {task}: {train_file}")
        return []

    with open(train_file, "r") as f:
        data_list = json.load(f)

    img_folder = image_folder_dict.get(task, "") or ""
    if not img_folder:
        print(f"[INFO] task={task} has empty image_folder, no image samples.")
        return []

    image_paths = []
    for ex in data_list:
        img_name = ex.get("image", None)
        if img_name:
            image_paths.append(os.path.join(img_folder, img_name))

    print(f"[INFO] task={task} train image samples={len(image_paths)}")
    return image_paths


def load_train_texts_for_task(task: str) -> List[str]:
    train_file = os.path.join(TRAIN_DATA_DIR, f"{task}.json")
    if not os.path.isfile(train_file):
        print(f"[WARN] train file not found for task {task}: {train_file}")
        return []

    with open(train_file, "r") as f:
        data_list = json.load(f)

    texts = []
    for ex in data_list:
        txt = extract_text_from_example(ex, is_train=True)
        if txt:
            texts.append(txt)
    print(f"[INFO] task={task} train text samples={len(texts)}")
    return texts


def load_train_image_text_pairs_for_task(task: str) -> Tuple[List[str], List[str]]:
    """
    both 模式用：只保留同时有 image + 文本的样本，保证两个 list 同步。
    """
    train_file = os.path.join(TRAIN_DATA_DIR, f"{task}.json")
    if not os.path.isfile(train_file):
        print(f"[WARN] train file not found for task {task}: {train_file}")
        return [], []

    with open(train_file, "r") as f:
        data_list = json.load(f)

    img_folder = image_folder_dict.get(task, "") or ""
    if not img_folder:
        print(f"[INFO] task={task} has empty image_folder, cannot form image+text pairs.")
        return [], []

    image_paths, texts = [], []
    for ex in data_list:
        img_name = ex.get("image", None)
        if not img_name:
            continue
        txt = extract_text_from_example(ex, is_train=True)
        if not txt:
            continue
        image_paths.append(os.path.join(img_folder, img_name))
        texts.append(txt)

    print(f"[INFO] task={task} train image+text pair samples={len(image_paths)}")
    return image_paths, texts


def load_test_image_paths_for_task(task: str) -> List[str]:
    q_file = question_file_dict.get(task, None)
    if q_file is None or not os.path.isfile(q_file):
        print(f"[WARN] test file not found for task {task}: {q_file}")
        return []

    q_list = load_json_or_ndjson(q_file)

    img_folder = image_folder_dict.get(task, "") or ""
    if not img_folder:
        print(f"[INFO] task={task} has empty image_folder, no image test samples.")
        return []

    image_paths = []
    for ex in q_list:
        img_name = ex.get("image", None)
        if img_name:
            image_paths.append(os.path.join(img_folder, img_name))

    print(f"[INFO] task={task} test image samples={len(image_paths)}")
    return image_paths


def load_test_texts_for_task(task: str) -> List[str]:
    q_file = question_file_dict.get(task, None)
    if q_file is None or not os.path.isfile(q_file):
        print(f"[WARN] test file not found for task {task}: {q_file}")
        return []

    q_list = load_json_or_ndjson(q_file)

    texts = []
    for ex in q_list:
        txt = extract_text_from_example(ex, is_train=False)
        if txt:
            texts.append(txt)
    print(f"[INFO] task={task} test text samples={len(texts)}")
    return texts


def load_test_image_text_pairs_for_task(task: str) -> Tuple[List[str], List[str]]:
    q_file = question_file_dict.get(task, None)
    if q_file is None or not os.path.isfile(q_file):
        print(f"[WARN] test file not found for task {task}: {q_file}")
        return [], []

    q_list = load_json_or_ndjson(q_file)

    img_folder = image_folder_dict.get(task, "") or ""
    if not img_folder:
        print(f"[INFO] task={task} has empty image_folder, cannot form image+text pairs for test.")
        return [], []

    image_paths, texts = [], []
    for ex in q_list:
        img_name = ex.get("image", None)
        if not img_name:
            continue
        txt = extract_text_from_example(ex, is_train=False)
        if not txt:
            continue
        image_paths.append(os.path.join(img_folder, img_name))
        texts.append(txt)

    print(f"[INFO] task={task} test image+text pair samples={len(image_paths)}")
    return image_paths, texts


# ================== 训练 Autoencoder ==================
def train_autoencoder_for_task(
    z: torch.Tensor,
    hidden_dim: int = 512,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "cuda",
    task_name: str = "",
) -> AutoEncoder:
    """
    z: [N, D] 训练特征
    """
    n, d = z.size()
    if n == 0:
        raise ValueError(f"Empty feature set for training AE of task {task_name}.")

    ds = torch.utils.data.TensorDataset(z)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    ae = AutoEncoder(dim=d, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)

    pbar_epochs = tqdm(range(epochs), desc=f"Train AE [{task_name}]")
    for ep in pbar_epochs:
        ae.train()
        total_loss = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            recon = ae(xb)
            loss = ((recon - xb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / n
        pbar_epochs.set_postfix({"mse": f"{avg_loss:.6f}"})
    return ae.cpu()


# ================== DDAS: 计算重构误差并选任务 ==================
def ddas_predict_task(z: torch.Tensor, ae_dict: Dict[str, AutoEncoder], device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    z: [N, D]
    ae_dict: {task_name: AutoEncoder}
    返回：
      - pred_idx: [N]，表示选中哪一个任务（按照 tasks_eval 的顺序）
      - errors: [N, T]，每个样本在每个任务上的 MSE，用于后续分析
    """
    tasks = list(ae_dict.keys())
    z = z.to(device)
    all_errors = []
    with torch.no_grad():
        for t in tasks:
            ae = ae_dict[t].to(device)
            recon = ae(z)
            d = ((recon - z) ** 2).mean(dim=-1)  # [N]
            all_errors.append(d.unsqueeze(-1))
    errors = torch.cat(all_errors, dim=-1)  # [N, T]
    pred_idx = errors.argmin(dim=-1)        # [N]
    return pred_idx, errors


# ================== 主流程 ==================
@dataclass
class DDASConfig:
    clip_model_path: str = "/mnt/cxzx/workspace/data_transfer/houzhiyan/models/clip-vit-large-patch14-336"
    tasks: str = DEFAULT_TASKS
    mode: str = "image"  # "image" | "text" | "both"
    ae_hidden_dim: int = 512
    ae_epochs: int = 20
    ae_lr: float = 1e-3
    batch_size_feat: int = 64
    batch_size_ae: int = 256
    num_workers: int = 4
    device: str = "cuda"
    output_root: str = "/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/without_reasoning"


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_model_path", type=str,
                        default="/mnt/cxzx/workspace/data_transfer/houzhiyan/models/clip-vit-large-patch14-336")
    parser.add_argument("--tasks", type=str,
                        default=DEFAULT_TASKS,
                        help="任务列表，逗号分隔")
    parser.add_argument("--mode", type=str, default="image",
                        choices=["image", "text", "both"],
                        help="DDAS 判别所用模态：image / text / both(image+text concat)")
    parser.add_argument("--ae_hidden_dim", type=int, default=512)
    parser.add_argument("--ae_epochs", type=int, default=20)
    parser.add_argument("--ae_lr", type=float, default=1e-3)
    parser.add_argument("--batch_size_feat", type=int, default=64)
    parser.add_argument("--batch_size_ae", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_root", type=str,
                        default="/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/without_reasoning",
                        help="结果保存的根目录")
    args = parser.parse_args()

    cfg = DDASConfig(
        clip_model_path=args.clip_model_path,
        tasks=args.tasks,
        mode=args.mode,
        ae_hidden_dim=args.ae_hidden_dim,
        ae_epochs=args.ae_epochs,
        ae_lr=args.ae_lr,
        batch_size_feat=args.batch_size_feat,
        batch_size_ae=args.batch_size_ae,
        num_workers=args.num_workers,
        device=args.device,
        output_root=args.output_root,
    )

    device = cfg.device if torch.cuda.is_available() and "cuda" in cfg.device else "cpu"
    print(f"[INFO] using device={device}")
    print(f"[INFO] DDAS mode = {cfg.mode}")

    # ==== 1) 解析任务列表 ====
    all_tasks = [t.strip() for t in cfg.tasks.split(",") if t.strip()]
    print(f"[INFO] all_tasks={all_tasks}")

    # 根据 mode 决定哪些任务可以参与
    if cfg.mode == "image":
        tasks_for_train = [
            t for t in all_tasks
            if image_folder_dict.get(t, "") not in ("", None)
        ]
    elif cfg.mode == "text":
        # 文本不依赖 image_folder，所有任务都可以
        tasks_for_train = all_tasks
    else:  # both
        tasks_for_train = [
            t for t in all_tasks
            if image_folder_dict.get(t, "") not in ("", None)
        ]
    print(f"[INFO] tasks_for_train (by mode) = {tasks_for_train}")

    # ==== 2) 加载 CLIP 模型 ====
    print(f"[INFO] loading CLIP model from {cfg.clip_model_path}")
    clip_model = CLIPModel.from_pretrained(cfg.clip_model_path).to(device)
    clip_processor = CLIPProcessor.from_pretrained(cfg.clip_model_path)

    # ==== 3) 为每个任务提取训练特征并训练 AE ====
    ae_dict: Dict[str, AutoEncoder] = {}
    feat_dim = None

    print("\n[DDAS] ==== Train AutoEncoders for tasks ====")
    for t in tqdm(tasks_for_train, desc="Train AE over tasks"):
        print(f"\n[DDAS] ==== Task {t}: train AE (mode={cfg.mode}) ====")

        if cfg.mode == "image":
            train_image_paths = load_train_image_paths_for_task(t)
            if len(train_image_paths) == 0:
                print(f"[WARN] no train image samples for task {t}, skip.")
                continue
            z_train = extract_image_features(
                train_image_paths,
                clip_model,
                clip_processor,
                batch_size=cfg.batch_size_feat,
                num_workers=cfg.num_workers,
                device=device,
            )

        elif cfg.mode == "text":
            train_texts = load_train_texts_for_task(t)
            if len(train_texts) == 0:
                print(f"[WARN] no train text samples for task {t}, skip.")
                continue
            z_train = extract_text_features(
                train_texts,
                clip_model,
                clip_processor,
                batch_size=cfg.batch_size_feat,
                num_workers=0,
                device=device,
            )

        else:  # both
            train_image_paths, train_texts = load_train_image_text_pairs_for_task(t)
            if len(train_image_paths) == 0:
                print(f"[WARN] no train image+text paired samples for task {t}, skip.")
                continue
            z_img = extract_image_features(
                train_image_paths,
                clip_model,
                clip_processor,
                batch_size=cfg.batch_size_feat,
                num_workers=cfg.num_workers,
                device=device,
            )
            z_txt = extract_text_features(
                train_texts,
                clip_model,
                clip_processor,
                batch_size=cfg.batch_size_feat,
                num_workers=0,
                device=device,
            )
            if z_img.size(0) != z_txt.size(0):
                n = min(z_img.size(0), z_txt.size(0))
                z_img = z_img[:n]
                z_txt = z_txt[:n]
            z_train = torch.cat([z_img, z_txt], dim=-1)

        if z_train.numel() == 0:
            print(f"[WARN] empty z_train for task {t}, skip.")
            continue

        if feat_dim is None:
            feat_dim = z_train.size(1)
            print(f"[INFO] feature dim (mode={cfg.mode}) = {feat_dim}")

        ae = train_autoencoder_for_task(
            z_train,
            hidden_dim=cfg.ae_hidden_dim,
            epochs=cfg.ae_epochs,
            lr=cfg.ae_lr,
            batch_size=cfg.batch_size_ae,
            device=device,
            task_name=t,
        )
        ae_dict[t] = ae

    if not ae_dict:
        print("[FATAL] no autoencoder trained (no tasks with usable data).")
        return

    tasks_eval = list(ae_dict.keys())
    print(f"\n[DDAS] Trained AE for tasks: {tasks_eval}")

    # ==== 4) 在所有任务的 test 集上评估 DDAS 判别准确率 ====
    print("\n[DDAS] ==== Evaluate task routing accuracy on test sets ====")
    all_z_test = []
    all_true = []

    for ti, t in enumerate(tqdm(tasks_eval, desc="Eval over tasks")):
        print(f"\n[DDAS] Eval task {t} (mode={cfg.mode})")

        if cfg.mode == "image":
            test_image_paths = load_test_image_paths_for_task(t)
            if len(test_image_paths) == 0:
                print(f"[WARN] no test image samples for task {t}, skip in eval.")
                continue
            z_test = extract_image_features(
                test_image_paths,
                clip_model,
                clip_processor,
                batch_size=cfg.batch_size_feat,
                num_workers=cfg.num_workers,
                device=device,
            )

        elif cfg.mode == "text":
            test_texts = load_test_texts_for_task(t)
            if len(test_texts) == 0:
                print(f"[WARN] no test text samples for task {t}, skip in eval.")
                continue
            z_test = extract_text_features(
                test_texts,
                clip_model,
                clip_processor,
                batch_size=cfg.batch_size_feat,
                num_workers=0,
                device=device,
            )

        else:  # both
            test_image_paths, test_texts = load_test_image_text_pairs_for_task(t)
            if len(test_image_paths) == 0:
                print(f"[WARN] no test image+text paired samples for task {t}, skip in eval.")
                continue
            z_img = extract_image_features(
                test_image_paths,
                clip_model,
                clip_processor,
                batch_size=cfg.batch_size_feat,
                num_workers=cfg.num_workers,
                device=device,
            )
            z_txt = extract_text_features(
                test_texts,
                clip_model,
                clip_processor,
                batch_size=cfg.batch_size_feat,
                num_workers=0,
                device=device,
            )
            if z_img.size(0) != z_txt.size(0):
                n = min(z_img.size(0), z_txt.size(0))
                z_img = z_img[:n]
                z_txt = z_txt[:n]
            z_test = torch.cat([z_img, z_txt], dim=-1)

        if z_test.numel() == 0:
            print(f"[WARN] empty z_test for task {t}, skip in eval.")
            continue

        all_z_test.append(z_test)
        all_true.append(torch.full((z_test.size(0),), ti, dtype=torch.long))

    if len(all_z_test) == 0:
        print("[FATAL] no test features collected.")
        return

    all_z_test = torch.cat(all_z_test, dim=0)  # [N, D]
    all_true = torch.cat(all_true, dim=0)      # [N]
    print(f"[INFO] total test samples used = {all_z_test.size(0)}")

    # 预测 + 误差矩阵
    print("[DDAS] Computing prediction & errors ...")
    pred_idx, errors = ddas_predict_task(all_z_test, ae_dict, device=device)
    correct = (pred_idx.cpu() == all_true)
    acc = correct.float().mean().item()
    print(f"\n[DDAS] Overall task routing accuracy (mode={cfg.mode}) over tasks={tasks_eval}: {acc:.4f}")

    # 混淆矩阵
    num_tasks = len(tasks_eval)
    confusion = torch.zeros(num_tasks, num_tasks, dtype=torch.long)
    for t_true, t_pred in zip(all_true, pred_idx.cpu()):
        confusion[int(t_true), int(t_pred)] += 1

    # 每个任务的单独准确率
    print("\n[DDAS] Per-task routing accuracy:")
    per_task_metrics = {}
    per_task_acc_list = []
    for ti, t in enumerate(tasks_eval):
        mask = (all_true == ti)
        n_samples_t = int(mask.sum().item())
        if n_samples_t == 0:
            continue
        acc_t = (pred_idx[mask].cpu() == all_true[mask]).float().mean().item()
        per_task_acc_list.append(acc_t)
        per_task_metrics[t] = {
            "index": ti,
            "num_samples": n_samples_t,
            "accuracy": acc_t,
        }
        print(f"  - {t}: acc={acc_t:.4f}, num_samples={n_samples_t}")

    macro_acc = sum(per_task_acc_list) / max(len(per_task_acc_list), 1)
    print(f"\n[DDAS] Macro accuracy (mean of per-task acc): {macro_acc:.4f}")

    # ==== 5) 保存指标和预测结果到文件 ====
    # 结果根目录：/mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/CLMM/without_reasoning/ddas_router/<mode>/
    out_dir = os.path.join(cfg.output_root, "ddas_router", cfg.mode)
    os.makedirs(out_dir, exist_ok=True)

    # 5.1 metrics.json
    metrics = {
        "mode": cfg.mode,
        "tasks_eval": tasks_eval,
        "feature_dim": int(feat_dim) if feat_dim is not None else None,
        "num_test_samples": int(all_z_test.size(0)),
        "overall_accuracy": acc,
        "macro_accuracy": macro_acc,
        "per_task": per_task_metrics,
        "confusion_matrix": confusion.tolist(),  # [T, T]
    }
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] metrics saved to {metrics_path}")

    # 5.2 predictions.jsonl：每个样本的 true_task / pred_task / correct
    preds_path = os.path.join(out_dir, "predictions.jsonl")
    with open(preds_path, "w") as f:
        for i in range(all_true.size(0)):
            true_idx_i = int(all_true[i].item())
            pred_idx_i = int(pred_idx[i].item())
            item = {
                "index": i,
                "true_task_idx": true_idx_i,
                "true_task": tasks_eval[true_idx_i],
                "pred_task_idx": pred_idx_i,
                "pred_task": tasks_eval[pred_idx_i],
                "correct": bool(true_idx_i == pred_idx_i),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[SAVE] predictions saved to {preds_path}")

    # 5.3 简单的人类可读 summary.txt
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"DDAS Task Routing Evaluation (mode={cfg.mode})\n")
        f.write(f"Tasks: {tasks_eval}\n")
        f.write(f"Feature dim: {feat_dim}\n")
        f.write(f"Num test samples: {int(all_z_test.size(0))}\n")
        f.write(f"Overall accuracy: {acc:.4f}\n")
        f.write(f"Macro accuracy: {macro_acc:.4f}\n\n")

        f.write("Per-task accuracy:\n")
        for t, m in per_task_metrics.items():
            f.write(
                f"  - {t}: acc={m['accuracy']:.4f}, num_samples={m['num_samples']}\n"
            )

        f.write("\nConfusion matrix (rows=true, cols=pred):\n")
        f.write("Tasks order: " + ", ".join(tasks_eval) + "\n")
        for i, t in enumerate(tasks_eval):
            row = " ".join(str(int(x)) for x in confusion[i].tolist())
            f.write(f"{t:10s}: {row}\n")

    print(f"[SAVE] summary saved to {summary_path}")
    print(f"[DONE] all results saved under {out_dir}")


if __name__ == "__main__":
    main()
