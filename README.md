# MLLM-CTBench: Continual Instruction Tuning with Reasoning Process Diagnosis

A benchmark for **continual instruction tuning (CIT)** of multimodal LLMs that jointly
measures **final-answer accuracy** and **process-level CoT reasoning quality**, across
**7 tasks / 6 domains** under a unified protocol.

- 📄 Paper: *MLLM-CTBench* (arXiv:2508.08275)
- 🤗 Data: https://huggingface.co/datasets/yueluoshuangtian/MLLM-CITBench

Tasks (order3 curriculum): `art → OCR → fomc → science → numglue → math → medical`
(= Arts VQA, OCR VQA, Economics QA, Science VQA, Math QA, Math VQA, Medicine VQA).
CL methods: `none(seqFT) · ewc · mas · lwf · freeze · replay · der · l2p · max_merge`,
plus training paradigms `SFT · RFT(GRPO) · Joint(SFT→RFT)`.

---

## Repository layout
```
MLLM-CTBench/
├── configs/paths.env      # 唯一路径配置入口（模型/数据/输出/评估器）
├── data/                  # 数据下载脚本 + 格式说明（数据本体从 HF 下载，不入库）
├── requirements/          # 三套依赖：llava_cl / qwen_cl / eval
├── llava_cl/              # LLaVA-1.5-7B 持续学习（全参 SFT + 8 CL 方法）
├── qwen_cl/               # Qwen2.5-VL-3B 持续学习（LoRA SFT + RFT/Joint + CL 方法）
│   ├── src/scripts_cl/run_cl_order3.sh   # SFT-CL 主入口
│   ├── run_rft_worker.sh                 # RFT(GRPO) 顺序 CL
│   └── run_jointft_worker.sh             # Joint(SFT→RFT) 顺序 CL
├── eval/                  # 评测：答案准确率 + CoT 过程级质量
│   └── cot_quality/cl_eval/              # 用评估器给模型 CoT 打三维度质量分
└── scripts/               # 便捷封装 + 环境自检
```

## 1. 环境
```bash
# LLaVA 侧 (CUDA 11.8)
conda create -n clmm python=3.10 -y && conda activate clmm
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -e "llava_cl[train]" && pip install flash-attn --no-build-isolation

# Qwen 侧 (CUDA 12.4)
conda create -n qwen_cl python=3.10 -y && conda activate qwen_cl
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements/qwen_cl.txt
pip install -e qwen_cl/src/open-r1-multimodal && pip install flash-attn --no-build-isolation

# 评测侧（调用 OpenAI 兼容的评估器服务，轻量）
pip install -r requirements/eval.txt
```
详见 `requirements/*.txt`。

## 2. 数据
```bash
bash data/download_data.sh        # 从 HF 拉 reasoning_train/test + img.zip 并解压到 data/
```
格式与放置见 `data/README.md`。

## 3. 配置路径（关键）
所有脚本从 `configs/paths.env` 读取路径。**改这一个文件即可**：
```bash
$EDITOR configs/paths.env         # 填 LLAVA_BASE / QWEN_BASE / (数据默认已指 data/) / EVALUATOR_MODEL
bash scripts/verify_setup.sh      # 自检：路径存在性 + 脚本语法 + 依赖
```

## 4. 训练
```bash
# --- Qwen2.5-VL-3B ---
# SFT 持续学习（换 <method> 跑 9 种方法之一；先 0.005 比例冒烟）
bash qwen_cl/src/scripts_cl/run_cl_order3.sh none 0.005 1     # 冒烟
bash qwen_cl/src/scripts_cl/run_cl_order3.sh ewc  1.0   3     # 全量
# RFT(GRPO) 顺序 CL / Joint(SFT→RFT) 顺序 CL
bash qwen_cl/run_rft_worker.sh
bash qwen_cl/run_jointft_worker.sh

# --- LLaVA-1.5-7B ---
bash llava_cl/scripts/ewc_seq_mas_seqback.sh                  # 见 llava_cl/README
```
产物写到 `configs/paths.env` 的 `OUTPUT_ROOT`（已 .gitignore，仓库不含 checkpoint）。

## 5. 推理 + 评测
```bash
# 重新推理（写完 <answer>，供评测）
python qwen_cl/reinfer.py --method_root $OUTPUT_ROOT/qwen_cl/order3/none_tr1.0

# (a) 最终答案准确率（AA/BWT 矩阵）
python eval/accuracy/reeval_clmm.py            # LLaVA
python qwen_cl/src/src/open_r1/eval_matrix.py  # Qwen

# (b) CoT 过程级质量（三维度 logic/grounding/knowledge + Forget/AP/BWT）
#     先起评估器服务（vLLM/sglang，OpenAI 兼容），再打分聚合
bash  eval/cot_quality/cl_eval/serve_vllm.sh                 # 起评估器服务
python eval/cot_quality/cl_eval/run_cl_eval.py --limit 10   # 每格随机10条(快)；去掉--limit为全量
python eval/cot_quality/cl_eval/aggregate_cl.py             # 出逐方法×逐任务表
```
CoT 评测细节见 `eval/cot_quality/cl_eval/README.md`。

## 复现说明
- 论文 Qwen LoRA per-task 3 epochs、LLaVA 全参；超参见各方法脚本。
- RFT 与 Joint 的 GRPO 配置见 `run_rft_worker.sh` / `run_jointft_worker.sh`。
- 完整训练需 8×A800(80G) 量级，单方法数小时；`--limit`/`TRAIN_RATIO` 可小样冒烟。

## Citation
```bibtex
@article{mllmctbench2025,
  title  = {MLLM-CTBench: A Benchmark for Continual Instruction Tuning with Reasoning Process Diagnosis},
  journal= {arXiv preprint arXiv:2508.08275},
  year   = {2025}
}
```
