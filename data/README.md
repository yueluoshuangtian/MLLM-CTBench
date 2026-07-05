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

## LLaVA 训练所需的数据（格式与 Qwen 不同）
LLaVA 用 **conversations** 格式训练、用**问句 jsonl** 做逐任务推理,和 Qwen 直接读 `reasoning_*` 不同。
一键脚本已自动调用转换器生成;也可手动跑:
```bash
python llava_cl/scripts/convert_qwen_to_llava.py --src data --out data
```
产出:
```
data/
├── llava_train/<task>.json     # 训练 conversations: {id, image?, conversations:[human/gpt]}
└── llava_eval/<task>.jsonl     # 评测问句(每行): {question_id, text, image?}
```
对应 `configs/paths.env` 的 `LLAVA_TRAIN_DIR` / `LLAVA_QUESTION_DIR`(默认已指向 `data/llava_*`)。
金标仍复用 `reasoning_test`(`TEST_DIR`)。纯文本任务(numglue/fomc)无 image 字段。

## 数据格式对照
| 用途 | Qwen | LLaVA |
|---|---|---|
| 训练 | `reasoning_train/<task>.json`（problem/solution） | `llava_train/<task>.json`（conversations） |
| 推理问句 | 直接用 reasoning + 模板 | `llava_eval/<task>.jsonl`（question_id/text/image） |
| 评测金标 | `reasoning_test/<task>.json` | 同左（复用） |
| 图片 | `images/`（img.zip 解压） | 同左（复用） |

## 备选：HF datasets 库加载
仓库同时提供了 HF-datasets 友好的 per-task 配置（字段 question_id/answer/image/…）：
```python
from datasets import load_dataset
ds = load_dataset("yueluoshuangtian/MLLM-CITBench", "art", split="train")
```
但**训练/评测代码直接读的是 `reasoning_train/reasoning_test` 的 json**（上面的格式），推荐用一键脚本。
