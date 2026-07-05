"""
用本地 Qwen3.5-27B 评估器给各 CL 方法的模型 CoT 打过程级质量分。

- 端点走 OpenAI 兼容（sglang），复用 newapi_client。
- 并发调用、失败重试；增量写 SCORE_FILE(jsonl)，按
  (model, method, stage, task, question_id) 去重，支持断点续跑。
- 可用 --models/--methods/--tasks/--stages/--limit 过滤，便于分批或冒烟。

用法：
  python run_cl_eval.py --limit 5          # 冒烟：每单元格只评 5 条
  python run_cl_eval.py                     # 全量
  python run_cl_eval.py --models qwen --methods der ewc
"""
import os, sys, json, re, argparse, random
from concurrent.futures import ThreadPoolExecutor, as_completed

# 在 import config / newapi_client 之前注入本地端点（它们在 import 时读环境变量）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 上级目录，含 config/newapi_client
import cl_config as C
os.environ.setdefault("NEWAPI_BASE_URL", C.EVAL_BASE_URL)
os.environ.setdefault("NEWAPI_KEY", C.EVAL_API_KEY)
os.environ.setdefault("EVAL_BASE_URL", C.EVAL_BASE_URL)

import config as _gcfg                      # noqa: E402 原全局 config（newapi_client 依赖）
# 用本地参数覆盖原 config（影响 newapi_client 的超时/重试/温度/图片上限）
_gcfg.BASE_URL = C.EVAL_BASE_URL.rstrip("/")
_gcfg.API_KEY = C.EVAL_API_KEY
_gcfg.TEMPERATURE = C.TEMPERATURE
_gcfg.MAX_RETRIES = C.MAX_RETRIES
_gcfg.REQUEST_TIMEOUT = C.REQUEST_TIMEOUT
_gcfg.MAX_IMAGE_BYTES = C.MAX_IMAGE_BYTES

import cl_data                              # noqa: E402
import rubrics                              # noqa: E402  直接复用 cot_quality_eval 的原始 prompt（保持一致）
from newapi_client import encode_image      # noqa: E402  仅复用图片编码
import time, itertools                      # noqa: E402
from openai import OpenAI                   # noqa: E402

# ---- 多副本轮询客户端（用满8卡：每副本TP=4，两副本30000/30001）----
_PORTS = C.EVAL_PORTS or [C.EVAL_BASE_URL.rsplit(":", 1)[-1]]
_CLIENTS = [OpenAI(api_key=C.EVAL_API_KEY,
                   base_url=f"http://{C.EVAL_HOST}:{p}/v1",
                   timeout=C.REQUEST_TIMEOUT) for p in _PORTS]
_rr = itertools.cycle(range(len(_CLIENTS)))  # itertools.cycle 的 next 是线程安全的


def chat(model, messages):
    """带重试 + 副本轮询的 chat completion，返回 (text, {})。"""
    last = None
    for attempt in range(C.MAX_RETRIES):
        cli = _CLIENTS[next(_rr)]
        try:
            resp = cli.chat.completions.create(
                model=model, messages=messages, temperature=C.TEMPERATURE,
                max_tokens=512,
                # Qwen3.5 评分关闭 thinking：只需吐 JSON 分数，提速约68×，
                # 也更贴近论文非推理评估器(Qwen2.5-VL-7B)的行为。
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return resp.choices[0].message.content, {}
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(2 ** attempt, 15))
    raise last


def parse_json_response(text):
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def cell_key(s):
    return (s["model"], s["method"], s["stage"], s["task"], s["question_id"])


def load_done(score_file):
    done = set()
    if os.path.isfile(score_file):
        for line in open(score_file, encoding="utf-8"):
            try:
                r = json.loads(line)
                if not r.get("error"):
                    done.add((r["model"], r["method"], r["stage"], r["task"], r["question_id"]))
            except Exception:
                continue
    return done


def score_one(s):
    img_uri = encode_image(s["image_path"]) if s.get("image_path") else None
    messages, has_g = rubrics.build_messages(
        task=s["task_key"], question=s["question"], cot_text=s["cot_text"],
        image_data_uri=img_uri, ref_answer=s.get("ref_answer"),
        image_path=s.get("image_path") or "", question_id=s["question_id"],
    )
    base = {k: s[k] for k in ("model", "method", "task", "task_key", "stage", "question_id")}
    try:
        text, _ = chat(C.EVAL_MODEL, messages)
        parsed = parse_json_response(text)
        if not parsed:
            return {**base, "error": "parse_failed", "raw": (text or "")[:500]}
        return {
            **base,
            "logic": parsed.get("logic"),
            "grounding": parsed.get("grounding") if has_g else None,
            "knowledge": parsed.get("knowledge"),
            "overall": parsed.get("overall"),
            "answer_consistent": parsed.get("answer_consistent"),
            "rationale": parsed.get("rationale"),
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        return {**base, "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, choices=list(C.MODELS))
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--stages", nargs="*", default=["post", "final"], choices=["post", "final"])
    ap.add_argument("--limit", type=int, default=0, help="每单元格随机抽多少条（0=全部）")
    ap.add_argument("--seed", type=int, default=42, help="随机抽样种子（可复现）")
    ap.add_argument("--workers", type=int, default=C.MAX_WORKERS)
    args = ap.parse_args()

    # 收集全部待评样本
    all_samples, missing = [], []
    for meta, samples in cl_data.iter_cells(args.models, args.methods, args.tasks, tuple(args.stages)):
        if not samples and meta["path"] is None:
            missing.append(meta)
        if args.limit and samples:
            # 每单元格随机抽样（按 cell 键 + seed 确定性，可复现且不依赖读取顺序）
            rng = random.Random(f"{meta['model']}/{meta['method']}/{meta['stage']}/{meta['task']}/{args.seed}")
            samples = rng.sample(samples, min(args.limit, len(samples)))
        all_samples.extend(samples)

    done = load_done(C.SCORE_FILE)
    todo = [s for s in all_samples if cell_key(s) not in done]
    print(f"样本总数 {len(all_samples)}，已完成 {len(all_samples)-len(todo)}，待评 {len(todo)}")
    if missing:
        print(f"⚠ 缺失预测文件的单元格 {len(missing)} 个（前5）：")
        for m in missing[:5]:
            print(f"   {m['model']}/{m['method']}/{m['stage']}/{m['task']}")
    if not todo:
        print("无待评样本。"); return

    f = open(C.SCORE_FILE, "a", encoding="utf-8")
    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(score_one, s) for s in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            n_err += bool(r.get("error")); n_ok += (not r.get("error"))
            if i % 200 == 0:
                print(f"  进度 {i}/{len(todo)}  成功 {n_ok} 失败 {n_err}")
    f.close()
    print(f"完成。成功 {n_ok}，失败 {n_err}。-> {C.SCORE_FILE}")
    if n_err:
        print("提示：重跑本脚本会自动续跑失败/未完成条目。")


if __name__ == "__main__":
    main()
