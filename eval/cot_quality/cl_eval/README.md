# CL 方法 CoT 过程级质量评测（本地 Qwen3.5-27B 评估器）

用新训练的评估器 `Qwen3.5-27B` 复刻论文 MLLM-CTBench 的过程级评测，
给 **LLaVA(CLMM)** 和 **Qwen(qwen_cl)** 两个 backbone 上各持续学习方法的
**模型生成 CoT** 打三维度质量分(logic/grounding/knowledge)，出
Table V/VII 口径的逐方法×逐任务 **Score / Forget / AP / BWT**。

任务顺序 = order3：`art → OCR → fomc → science → numglue → math → medical`
方法(各9)：`none ewc mas lwf freeze replay der l2p max_merge`

## 文件
| 文件 | 作用 |
|---|---|
| `cl_config.py` | 端点/模型/路径/任务映射/方法/输出/并发 |
| `cl_data.py` | 读两 backbone 预测文件 + join 金标 → 打分样本(抽 `<think>` CoT、解析图片) |
| `cl_rubrics.py` | 复用原三维度 rubric，framing 改为「评模型生成 CoT」 |
| `run_cl_eval.py` | 主程序：并发打分、增量写盘、断点续跑、可过滤/限量 |
| `aggregate_cl.py` | 聚合出 Score/Forget/AP/BWT 表 |
| `serve_evaluator.sh` | sglang 起评估器服务 |

## 执行步骤

### 1) 起评估器服务（claw_env_q35 + sglang）
```bash
cd /mnt/cxzx/workspace/data_transfer/houzhiyan/TMM/cot_quality_eval/cl_eval
GPUS=0,1 TP=2 PORT=30000 bash serve_evaluator.sh    # 前台；建议 tmux/nohup 后台
# 等日志出现 "The server is fired up and ready to roll!" 即就绪
```
就绪自测（任意带 openai 包的 env 均可）：
```bash
curl http://127.0.0.1:30000/v1/models
```

### 2) 冒烟（每单元格只评 5 条，验证端到端）
评测脚本本身只需要 `openai` 包，用原审计用的 env（如 base 或装了 openai 的任意 env）：
```bash
PY=/mnt/cxzx/workspace/data_transfer/houzhiyan/miniconda3/bin/python   # 需含 openai
$PY run_cl_eval.py --limit 5
$PY aggregate_cl.py        # 看小样表是否合理
```

### 3) 全量评测（后台，断点续跑）
```bash
nohup $PY run_cl_eval.py --workers 32 > outputs/run_full.log 2>&1 &
tail -f outputs/run_full.log
# 中断后重跑同一命令即自动续跑（按 model/method/stage/task/qid 去重）
```
可分批：`--models qwen`、`--methods der ewc`、`--tasks art OCR`、`--stages final`。

### 4) 出表
```bash
$PY aggregate_cl.py
# -> outputs/reports/cot_table_llava.md
#    outputs/reports/cot_table_qwen.md
#    outputs/reports/cot_scores_summary.json
```

## 说明
- 端点走 OpenAI 兼容协议（sglang），`run_cl_eval.py` 在 import 前把
  `NEWAPI_BASE_URL/NEWAPI_KEY` 指向本地服务并覆盖 `config` 的超时/温度等，复用 `newapi_client.py`。
- CoT 总分 = 适用维度均值；纯文本任务(fomc/numglue)无 grounding，与论文一致。
- `post`=训完即测(对角 P_jj)，`final`=全训完(末行 P_Nj)；末任务 final==post 自动复用。
- 图片根 `clmm-benchmark/`，金标 `qwen_data/test/<task>.json` 提供 image+solution。
