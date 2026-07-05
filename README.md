# MLLM-CTBench

**A Benchmark for Continual Instruction Tuning of MLLMs with Reasoning Process Diagnosis**

MLLM-CTBench evaluates continual instruction tuning (CIT) of multimodal LLMs by jointly
measuring **final-answer accuracy** and **process-level Chain-of-Thought (CoT) quality**,
over **7 tasks / 6 domains**, under a unified protocol — across two backbones
(**LLaVA-1.5-7B**, **Qwen2.5-VL-3B**) and three training paradigms (**SFT**, **RFT/GRPO**, **Joint SFT→RFT**).

- 📄 Paper: arXiv:2508.08275   ·   🤗 Data: [yueluoshuangtian/MLLM-CITBench](https://huggingface.co/datasets/yueluoshuangtian/MLLM-CITBench)
- Tasks (order3): `art → OCR → fomc → science → numglue → math → medical`
- CL methods: `none(seqFT) · ewc · mas · lwf · freeze · replay · der · l2p · max_merge`

> This repo ships **code + data pipeline + reproduction instructions** only. Base models and
> trained checkpoints are **not** included (it is a benchmark). Bring your own base models;
> data is downloaded from HuggingFace.

---

## Quick start (TL;DR)
```bash
git clone <this repo> MLLM-CTBench && cd MLLM-CTBench
$EDITOR configs/paths.env          # ① 填 LLAVA_BASE / QWEN_BASE / EVALUATOR_MODEL
bash data/download_data.sh         # ② 下载数据(自动生成 LLaVA 格式)
bash scripts/verify_setup.sh       # ③ 自检：路径 + 语法 + 导入
# ④ 训练冒烟（0.5%数据验证链路）→ 全量：见下方「Training」
```

## Repository layout
```
MLLM-CTBench/
├── configs/paths.env         # ★ 唯一路径配置入口（模型/数据/输出/评估器），可被环境变量覆盖
├── data/                     # download_data.sh + 格式说明（数据从 HF 下载，不入库）
├── requirements/             # 三套依赖：llava_cl / qwen_cl / eval
├── llava_cl/                 # LLaVA-1.5-7B 持续学习（全参 SFT）
│   ├── llava/train/train_cl.py           # 训练主程序（含逐任务评测）
│   ├── cl_learner/                        # 8 种 CL 方法实现
│   └── scripts/llava_v1_5/finetune_cl_seqftback.sh
├── qwen_cl/                  # Qwen2.5-VL-3B 持续学习（LoRA SFT + RFT/Joint）
│   ├── src/scripts_cl/run_cl_order3.sh    # SFT-CL 主入口
│   ├── run_rft_worker.sh / run_jointft_worker.sh   # RFT(GRPO) / Joint 顺序 CL
│   └── src/src/open_r1/                   # sft.py / grpo_rec.py / trainer/
├── eval/
│   ├── accuracy/qwen_new_eval_tool/       # Qwen 评分（is_reasoning：剥离<think>+符号过滤）
│   ├── accuracy/reeval_clmm.py            # LLaVA 简化重评（抠<answer>）
│   └── cot_quality/cl_eval/               # CoT 过程级质量（评估器打三维度分）
└── scripts/verify_setup.sh
```

## 1. 环境安装
两个训练 backbone 用两套环境（CUDA 版本不同）。
```bash
# LLaVA (CUDA 11.8)
conda create -n clmm python=3.10 -y && conda activate clmm
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -e "llava_cl[train]" && pip install flash-attn --no-build-isolation

# Qwen (CUDA 12.4)
conda create -n qwen_cl python=3.10 -y && conda activate qwen_cl
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements/qwen_cl.txt
pip install -e qwen_cl/src/open-r1-multimodal && pip install flash-attn --no-build-isolation

# 评测（调用 OpenAI 兼容的评估器服务，轻量）
pip install -r requirements/eval.txt
```

## 2. 数据
```bash
bash data/download_data.sh    # 下载 reasoning_train/test + img.zip，并自动生成 llava_train/llava_eval
```
- **Qwen** 直接读 `data/reasoning_{train,test}/<task>.json`（`problem`/`solution`(含CoT)/`image`）。
- **LLaVA** 用转换生成的 `data/llava_train`（conversations）+ `data/llava_eval`（问句 jsonl）。
- 图片：`data/images`（img.zip 解压）。详见 [`data/README.md`](data/README.md)。

## 3. 配置路径（★ 唯一入口）
编辑 `configs/paths.env`（也可用同名环境变量在命令行覆盖）：
```bash
LLAVA_BASE=/path/to/llava-v1.5-7b
QWEN_BASE=/path/to/Qwen2.5-VL-3B-Instruct
EVALUATOR_MODEL=/path/to/CoT-Evaluator     # 过程级评测用
# 数据/输出/LLaVA 数据目录默认已指向 data/ 与 output/，一般无需改
bash scripts/verify_setup.sh               # 自检
```

## 4. 训练
所有脚本自动 `source configs/paths.env`。先用小样冒烟验证链路，再全量。

**Qwen2.5-VL-3B（LoRA）**
```bash
# SFT 持续学习：<method> ∈ 9 种方法之一
bash qwen_cl/src/scripts_cl/run_cl_order3.sh none 0.005 1     # 冒烟(0.5%数据,1epoch)
bash qwen_cl/src/scripts_cl/run_cl_order3.sh ewc  1.0   3     # 全量(论文3epoch)
# RFT(GRPO) 顺序 CL / Joint(每任务 1ep SFT → GRPO)
bash qwen_cl/run_rft_worker.sh
bash qwen_cl/run_jointft_worker.sh
```

**LLaVA-1.5-7B（全参）**
```bash
# 单任务冒烟（--max_steps 覆盖 epoch，几步即停）
deepspeed --include=localhost:0 llava_cl/llava/train/train_cl.py \
  --deepspeed llava_cl/scripts/zero2.json --model_name_or_path $LLAVA_BASE --version v1 \
  --tasks art --initial_tasks "" --image_folder $IMAGE_ROOT \
  --vision_tower /path/to/clip-vit-large-patch14-336 --bf16 True --output_dir $OUTPUT_ROOT/llava_smoke \
  --max_steps 2 --per_device_train_batch_size 1 --gradient_accumulation_steps 6 \
  --report_to none --cl none   # 其余超参见 finetune 脚本
# 全量顺序 CL（含各方法）
bash llava_cl/scripts/llava_v1_5/finetune_cl_seqftback.sh
```
产物写到 `OUTPUT_ROOT`（已 .gitignore）。

## 5. 推理 + 评测
两个 backbone 输出格式不同，**评测各用各的**：
```bash
# (a) 答案准确率（AA/BWT）
#  Qwen：模型输出 <think>…</think><answer>…</answer>，用 new_eval_tool(is_reasoning=True) 剥 <think>+符号过滤
python qwen_cl/src/src/open_r1/eval_matrix.py --root $OUTPUT_ROOT/qwen_cl/order3
#  LLaVA：抠 <answer> 按任务 exact/ROUGE-L（逐任务器在 llava_cl/eval_tool/）
python eval/accuracy/reeval_clmm.py

# (b) CoT 过程级质量（logic/grounding/knowledge + Forget/AP/BWT）
bash  eval/cot_quality/cl_eval/serve_vllm.sh                 # 起评估器服务(OpenAI兼容)
python eval/cot_quality/cl_eval/run_cl_eval.py --limit 10    # 每格随机10条(快)；去掉--limit为全量
python eval/cot_quality/cl_eval/aggregate_cl.py             # 出逐方法×逐任务表
```

## 已验证状态（冒烟实测，非纸面）
| 路径 | 状态 |
|---|---|
| Qwen SFT-CL | ✅ 真实模型+数据跑通(train→infer) |
| Qwen RFT(GRPO) | ✅ 真 GRPO 步(loss/kl/clip_ratio) |
| LLaVA SFT | ✅ 真训练步(deepspeed zero2 全参) |
| 所有训练/评测模块 import | ✅ 干净 |
| CoT 质量评测(双格式) | ✅ 2340 条实跑出表 |

完整训练（每方法数小时×8卡）需自行按上面命令跑；本仓库保证结构可运行 + 冒烟通过。

## Citation
```bibtex
@article{mllmctbench2025,
  title  = {MLLM-CTBench: A Benchmark for Continual Instruction Tuning with Reasoning Process Diagnosis},
  journal= {arXiv preprint arXiv:2508.08275},
  year   = {2025}
}
```
