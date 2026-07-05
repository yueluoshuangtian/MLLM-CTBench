"""
全局配置：API 网关、judge 模型、路径、抽样规模。
所有密钥/host 都从环境变量读取，不在代码里硬编码（避免泄露）。

使用前在 shell 里 export：
    export NEWAPI_BASE_URL="https://你的new-api域名"   # 不要带结尾的 /v1
    export NEWAPI_KEY="sk-xxxxxxxx"                      # new-api 令牌
可选：用 NEWAPI_MODEL_GPT / NEWAPI_MODEL_CLAUDE / NEWAPI_MODEL_GEMINI 覆盖默认模型名。
"""
import os

# ---------------- API 网关 ----------------
# new-api 统一走 OpenAI 兼容端点 {BASE_URL}/v1/chat/completions
BASE_URL = os.environ.get("NEWAPI_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("NEWAPI_KEY", "")

# ---------------- 三个 judge 模型 ----------------
# 模型名取决于你们 new-api 后台实际启用的渠道；先用通用名，跑 test_api.py 时若报错
# 用 list_models() 打印可用模型再改这里，或用环境变量覆盖。
# 模型名用 lingleap 平台真实存在的（见 /api/pricing）。各模型所需分组：
#   gpt-5                      需 Azure 分组（或 GPT-直连/正价，视渠道而定）
#   claude-sonnet-4-5-20250929 需 Claude-AWS 分组
#   gemini-2.5-pro             需 Gemini-正价/直连 分组
JUDGES = {
    "gpt":    os.environ.get("NEWAPI_MODEL_GPT",    "gpt-5"),
    "claude": os.environ.get("NEWAPI_MODEL_CLAUDE", "claude-sonnet-4-5-20250929"),
    "gemini": os.environ.get("NEWAPI_MODEL_GEMINI", "gemini-2.5-pro"),
}

# 可选：若每个厂商需用不同令牌（分组按厂商隔离时），在此填各 judge 专用 key；
# 留空则统一用 NEWAPI_KEY。键名与 JUDGES 对应。
JUDGE_KEYS = {
    "gpt":    os.environ.get("NEWAPI_KEY_GPT", ""),
    "claude": os.environ.get("NEWAPI_KEY_CLAUDE", ""),
    "gemini": os.environ.get("NEWAPI_KEY_GEMINI", ""),
}

# 只跑部分 judge：设 NEWAPI_JUDGES="gpt" 或 "gpt,claude"（逗号分隔），留空跑全部。
_only = os.environ.get("NEWAPI_JUDGES", "").strip()
if _only:
    _keep = {j.strip() for j in _only.split(",")}
    JUDGES = {k: v for k, v in JUDGES.items() if k in _keep}

# ---------------- 数据路径 ----------------
# 训练 CoT 源数据所在目录（a_data_use），图片路径相对于此目录解析
DATA_ROOT = "/mnt/cxzx/workspace/data_transfer/houzhiyan/a_data_use"

# 7 个任务 -> (源 json 文件名, 是否含图像)
TASK_FILES = {
    "math_qa":      ("numglue_reasoning_samples.json", False),
    "economics_qa": ("fomc_reasoning_samples.json",    False),
    "science_vqa":  ("science_reasoning_samples.json", True),
    "math_vqa":     ("math_reasoning_samples.json",    True),
    "medicine_vqa": ("medical_reasoning_samples.json", True),
    "ocr_vqa":      ("OCR_reasoning_samples.json",     True),
    "arts_vqa":     ("art_reasoning_samples.json",     True),
}

# ---------------- 抽样与输出 ----------------
TOTAL_SAMPLES = 400          # 总抽样条数
SEED = 42                    # 复现实验用的随机种子
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SAMPLE_FILE = os.path.join(OUT_DIR, "sampled_400.json")          # 抽样结果
SCORE_FILE = os.path.join(OUT_DIR, "scores.jsonl")               # 打分结果（增量）
REPORT_FILE = os.path.join(OUT_DIR, "quality_report.md")         # 统计报告

# ---------------- 调用参数 ----------------
MAX_WORKERS = int(os.environ.get("NEWAPI_WORKERS", "4"))  # 并发线程数（I/O 密集，可调高；按 new-api 限流调整）
MAX_RETRIES = 4             # 单次调用失败重试次数
TEMPERATURE = 0.0           # 评分用确定性输出
REQUEST_TIMEOUT = 120       # 秒
MAX_IMAGE_BYTES = 4_000_000 # 超过则跳过图片（base64 太大）

os.makedirs(OUT_DIR, exist_ok=True)
