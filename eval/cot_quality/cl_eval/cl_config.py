"""
CL CoT 质量评测配置（用本地 Qwen3.5-27B 评估器给各持续学习方法的模型 CoT 打过程级分）。

与原 cot_quality_eval 的区别：
  - judge 从「三个 API 大模型」换成「本地 sglang 起的 Qwen3.5-27B 单评估器」；
  - 被打分对象从「GPT-4 训练标注 CoT」换成「CL 模型推理生成的 CoT」。
评测端点走 OpenAI 兼容协议，因此沿用 newapi_client.py。
"""
import os

# ---------------- 本地评估器服务（sglang，OpenAI 兼容）----------------
# 注意：复用的 newapi_client 会拼 BASE_URL + "/v1"，所以这里不要带 /v1。
EVAL_BASE_URL = os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:30000")
# 多副本端口（用满8卡：每副本TP=4，两副本在30000/30001），客户端轮询负载均衡。
# 可用 EVAL_PORTS="30000,30001" 覆盖；为空则用 EVAL_BASE_URL 单端点。
EVAL_PORTS    = [p for p in os.environ.get("EVAL_PORTS", "30000,30001").split(",") if p.strip()]
EVAL_HOST     = os.environ.get("EVAL_HOST", "127.0.0.1")
EVAL_API_KEY  = os.environ.get("EVAL_API_KEY", "EMPTY")          # vllm 不校验，但 OpenAI SDK 需非空
EVAL_MODEL    = os.environ.get("EVAL_SERVED_NAME", os.environ.get("EVAL_MODEL", "cot-evaluator"))  # 与 --served-model-name 一致
MODEL_PATH    = os.environ.get("EVALUATOR_MODEL", "/path/to/CoT-Evaluator")   # 评估器权重

# ---------------- 数据布局（均从 configs/paths.env 的环境变量读取）----------------
GT_DIR     = os.environ.get("TEST_DIR", "data/reasoning_test")   # 金标：<task>.json -> {problem, solution, image}
IMAGE_ROOT = os.environ.get("IMAGE_ROOT", "data/images")          # image 字段相对此根解析
_OUT       = os.environ.get("OUTPUT_ROOT", "output")              # 训练/推理产物根

# order3 任务顺序（两个 backbone 都用这个顺序训练）
ORDER = ["art", "OCR", "fomc", "science", "numglue", "math", "medical"]

# dir 任务名 -> (rubric 任务键, 是否含图)。映射到论文 7 任务。
TASK_MAP = {
    "art":     ("arts_vqa",     True),   # Arts VQA
    "OCR":     ("ocr_vqa",      True),   # OCR VQA
    "fomc":    ("economics_qa", False),  # Economics QA（纯文本）
    "science": ("science_vqa",  True),   # Science VQA
    "numglue": ("math_qa",      False),  # Math QA（纯文本）
    "math":    ("math_vqa",     True),   # Math VQA
    "medical": ("medicine_vqa", True),   # Medicine VQA
}

METHODS = ["none", "ewc", "mas", "lwf", "freeze", "replay", "der", "l2p", "max_merge"]

# 两个 backbone 的预测文件根 + 命名规则
MODELS = {
    "llava": {
        "root": os.environ.get("LLAVA_PRED_ROOT", f"{_OUT}/llava_cl/order3"),
        "method_dir": "{method}_default",
        # 预测 jsonl（每行 {question_id, prompt, text}）
        "post":  ["{md}/{k}/predictions/{k}-{task}.jsonl",          # 训完即测（对角 P_jj）
                  "{md}/7/predictions/{k}-{task}.jsonl"],           # 兜底：也可能落在 7/
        "final": ["{md}/7/predictions/{j}-{task}.jsonl"],           # 全训完（末行 P_Nj）
        "fmt": "jsonl_text",
    },
    "qwen": {
        "root": os.environ.get("QWEN_PRED_ROOT", f"{_OUT}/qwen_cl/order3"),
        "method_dir": "{method}_tr1.0",
        # 预测 json（{results:[{question_id, question, ground_truth, model_output}]}）
        "post":  ["{md}/predictions_v2/{k}-{task}.json"],
        "final": ["{md}/last_predictions_v2/{j}-{task}.json"],
        "fmt": "json_results",
    },
}

# ---------------- 输出 ----------------
OUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SCORE_FILE  = os.path.join(OUT_DIR, "cl_cot_scores.jsonl")     # 增量打分结果
REPORT_DIR  = os.path.join(OUT_DIR, "reports")

# ---------------- 调用参数 ----------------
MAX_WORKERS     = int(os.environ.get("EVAL_WORKERS", "32"))   # 并发；按服务吞吐调
MAX_RETRIES     = 4
TEMPERATURE     = 0.0
REQUEST_TIMEOUT = 180
MAX_IMAGE_BYTES = 8_000_000

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
