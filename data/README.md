# 数据准备（MLLM-CTBench）

数据托管在 HuggingFace，**不随仓库分发**（img.zip 约 4.2GB）。
数据集：https://huggingface.co/datasets/yueluoshuangtian/MLLM-CITBench

## 一键下载
```bash
bash data/download_data.sh
```
它会把数据下到本仓库 `data/` 下并解压，得到：
```
data/
├── reasoning_train/<task>.json    # 训练集（含 CoT）
├── reasoning_test/<task>.json     # 测试集
└── images/…                        # img.zip 解压后的图片根
```
7 个任务：`art OCR fomc science numglue math medical`（fomc/numglue 为纯文本，无图）。

## 数据格式（reasoning_train / reasoning_test 的每条）
```json
{
  "problem":  "问题文本 + 答案格式约束",
  "solution": "<think> …推理过程… </think> 最终答案",   // 训练用 CoT 目标
  "image":    "art_vqa_datasets/AQUA/train/xxx.jpg"       // 相对 IMAGE_ROOT；纯文本任务无此字段
}
```
- 训练读 `solution`（CoT + 答案）作为监督；评测时从模型输出抽 `<answer>` 与金标 `solution` 比对。
- `image` 是相对路径，解析时与 `configs/paths.env` 的 `IMAGE_ROOT` 拼接。

## 与配置对接
下载后确认 `configs/paths.env` 里：
```bash
TRAIN_DIR=<repo>/data/reasoning_train
TEST_DIR=<repo>/data/reasoning_test
IMAGE_ROOT=<repo>/data/images
```
（默认已指向 `data/` 下，一键脚本下载到位后无需再改。）

## 备选：HF datasets 库加载
仓库同时提供了 HF-datasets 友好的 per-task 配置（字段 question_id/answer/image/…）：
```python
from datasets import load_dataset
ds = load_dataset("yueluoshuangtian/MLLM-CITBench", "art", split="train")
```
但**训练/评测代码直接读的是 `reasoning_train/reasoning_test` 的 json**（上面的格式），推荐用一键脚本。
