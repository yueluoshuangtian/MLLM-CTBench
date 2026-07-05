"""
对抽样的 400 条训练 CoT，用 3 个 judge 模型做 pointwise 质量打分。
- 并发调用、失败重试。
- 增量写入 SCORE_FILE(jsonl)，支持断点续跑（已打过的 (sample_id, judge) 跳过）。
每行结果: {sample_id, task, subtype, judge, logic, grounding, knowledge, overall,
           answer_consistent, rationale, raw, error}
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
import rubrics
from newapi_client import chat, encode_image


def parse_json_response(text):
    """从模型输出里抽出 JSON 对象，容忍前后多余文本/代码块。"""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
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


def load_done(score_file):
    """读取已完成的 (sample_id, judge) 集合，用于续跑。"""
    done = set()
    if os.path.isfile(score_file):
        with open(score_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if not r.get("error"):
                        done.add((r["sample_id"], r["judge"]))
                except Exception:
                    continue
    return done


def score_one(sample, judge_name, model):
    """对单条样本用单个 judge 打分，返回结果 dict。"""
    img_uri = encode_image(sample["image_path"]) if sample.get("image_path") else None
    messages, has_grounding = rubrics.build_messages(
        task=sample["task"],
        question=sample["question"],
        cot_text=sample["cot_text"],
        image_data_uri=img_uri,
        ref_answer=sample.get("ref_answer"),
        image_path=sample.get("image_path") or "",
        question_id=sample.get("id") or "",
    )
    base = {
        "sample_id": sample["id"], "task": sample["task"],
        "subtype": sample.get("subtype"), "judge": judge_name,
    }
    try:
        key = config.JUDGE_KEYS.get(judge_name) or None
        text, _usage = chat(model, messages, api_key=key)
        parsed = parse_json_response(text)
        if not parsed:
            return {**base, "error": "parse_failed", "raw": text}
        return {
            **base,
            "logic": parsed.get("logic"),
            "grounding": parsed.get("grounding") if has_grounding else None,
            "knowledge": parsed.get("knowledge"),
            "overall": parsed.get("overall"),
            "answer_consistent": parsed.get("answer_consistent"),
            "rationale": parsed.get("rationale"),
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        return {**base, "error": f"{type(e).__name__}: {e}"}


def main():
    with open(config.SAMPLE_FILE, "r", encoding="utf-8") as f:
        samples = json.load(f)
    done = load_done(config.SCORE_FILE)
    jobs = [(s, jn, jm) for s in samples for jn, jm in config.JUDGES.items()
            if (s["id"], jn) not in done]
    print(f"待打分任务: {len(jobs)} (= 样本 {len(samples)} × judge {len(config.JUDGES)} - 已完成 {len(done)})")

    lock_f = open(config.SCORE_FILE, "a", encoding="utf-8")
    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as ex:
        futs = {ex.submit(score_one, s, jn, jm): (s["id"], jn) for s, jn, jm in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            lock_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            lock_f.flush()
            if r.get("error"):
                n_err += 1
            else:
                n_ok += 1
            if i % 20 == 0:
                print(f"  进度 {i}/{len(jobs)}  成功 {n_ok} 失败 {n_err}")
    lock_f.close()
    print(f"完成。成功 {n_ok}，失败 {n_err}。结果 -> {config.SCORE_FILE}")
    if n_err:
        print("提示：重新运行本脚本会自动续跑失败/未完成的条目。")


if __name__ == "__main__":
    main()
