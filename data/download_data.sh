#!/usr/bin/env bash
# 从 HuggingFace 下载 MLLM-CTBench 数据到本仓库 data/ 并解压。
# 需要: pip install -U "huggingface_hub[cli]"  (或已有 huggingface-cli)
set -euo pipefail

REPO_ID="yueluoshuangtian/MLLM-CITBench"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # = <repo>/data
cd "$HERE"

echo "[1/3] 下载 reasoning_train / reasoning_test / img.zip ..."
# 只下需要的部分；--local-dir 直接落到 data/
huggingface-cli download "$REPO_ID" --repo-type dataset --local-dir . \
  --include "reasoning_train/*" "reasoning_test/*" "img.zip"

echo "[2/3] 解压 img.zip -> images/ ..."
mkdir -p images
if [ -f img.zip ]; then
  unzip -q -o img.zip -d images && echo "  解压完成" || { echo "  解压失败，请手动 unzip img.zip -d images"; exit 1; }
else
  echo "  未找到 img.zip，请检查下载"; exit 1
fi

echo "[3/4] 生成 LLaVA 专用数据 (llava_train conversations + llava_eval 问句) ..."
# Qwen 直接用 reasoning_train/test；LLaVA 需要 conversations/问句格式，用转换器生成
python "$HERE/../llava_cl/scripts/convert_qwen_to_llava.py" --src "$HERE" --out "$HERE" || \
  echo "  (LLaVA 转换失败可跳过；仅 LLaVA 训练需要)"

echo "[4/4] 校验目录结构 ..."
for d in reasoning_train reasoning_test images llava_train llava_eval; do
  [ -d "$d" ] && echo "  ✓ data/$d ($(find "$d" -maxdepth 1 | wc -l) 项)" || echo "  ✗ 缺少 data/$d"
done
echo "完成。若图片根层级与 image 字段不符，请把真正含 art_vqa_datasets/... 的目录设为 IMAGE_ROOT。"
