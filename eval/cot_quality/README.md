# 训练 CoT 质量审计（回应「GPT-4 对训练数据质量影响」审稿意见）

用 3 个跨厂商 LLM judge（GPT / Claude / Gemini，经 new-api 网关）对 GPT-4 生成的
**训练 CoT** 做 **pointwise 单点质量打分**，证明训练标注质量可靠。
模仿 `a_data_use/generate_{task}_reasoning.py` 的逐任务评分维度，但改为单条打分（非配对）。

## 文件
| 文件 | 作用 | 是否需要 key |
|---|---|---|
| `config.py` | base_url / key / 模型名 / 路径 / 抽样规模（均走环境变量） | - |
| `newapi_client.py` | new-api 统一客户端（OpenAI 兼容端点，三模型只换 model 名） | - |
| `rubrics.py` | 7 任务 pointwise rubric（已修源码 bug，映射论文三维度） | - |
| `sample_data.py` | 分层抽样 400 条（含 science json salvage、图片解析） | 否 |
| `check_answer_consistency.py` | CoT–答案一致性启发式核查（**纯本地，已出真值**） | 否 |
| `test_api.py` | 连通性自测（列模型 + 测三模型） | 是 |
| `run_scoring.py` | 主程序：400 条 × 3 judge 打分，可断点续跑 | 是 |
| `analyze.py` | 统计：逐任务/逐维度均分、达标率、三 judge 一致性 | 否 |

## 运行步骤
```bash
cd TMM/cot_quality_eval

# 0) 已完成（无需 key）：抽样 + 本地一致性核查
python sample_data.py                 # -> outputs/sampled_400.json
python check_answer_consistency.py    # 本地真值

# 1) 设置网关与 key（你提供）
export NEWAPI_BASE_URL="https://你的new-api域名"   # 不带 /v1
export NEWAPI_KEY="sk-xxxx"
# 若默认模型名不对，跑 test_api.py 看可用模型后覆盖：
# export NEWAPI_MODEL_GPT=...  NEWAPI_MODEL_CLAUDE=...  NEWAPI_MODEL_GEMINI=...

# 2) 连通性自测
python test_api.py

# 3) 正式打分（1200 次调用 = 400×3，可中断续跑）
python run_scoring.py                 # -> outputs/scores.jsonl

# 4) 统计出报告
pip install numpy scipy krippendorff  # 统计4 一致性需要
python analyze.py                     # -> outputs/quality_report.md
```

## 已得到的本地真值（启发式，CoT 结论 vs 标注答案）
仅覆盖有 `,answer:` 标记的 3 个任务；其余 VQA 任务答案是结尾短语、无标记，需 LLM judge：
- math_qa 23.4%、economics_qa 2.0%、ocr_vqa 14.8%（潜在不一致率，含启发式误判，偏高估）
- **正式一致性以 run_scoring 的 `answer_consistent` 为准**

## 待你提供
new-api 的 `BASE_URL` 和 `KEY`（以及若模型名不同，三个 judge 的真实模型名）。
