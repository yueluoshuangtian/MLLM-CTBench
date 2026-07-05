"""
把 reasoning_{train,test}/<task>.json (Qwen 格式 {problem, solution, image})
转成 LLaVA 持续学习所需的两类文件:

  1) 训练数据 (conversations)  ->  <out>/llava_train/<task>.json
       {"id", "image"?, "conversations":[{"from":"human","value":"<image>\n<problem>"},
                                          {"from":"gpt","value":"<solution>"}]}
  2) 评测问句 (question jsonl) ->  <out>/llava_eval/<task>.jsonl   (每行一条)
       {"question_id", "text":"<problem>", "image"?}

train_cl.py 用 llava_train 做训练、llava_eval 做逐任务推理、reasoning_test 做金标评分。

用法:
  python scripts/convert_qwen_to_llava.py \
      --src   $TEST_DIR/../                # 含 reasoning_train / reasoning_test 的目录（默认取 data/）
      --out   data                         # 输出根：生成 data/llava_train 与 data/llava_eval
"""
import argparse, json, os

TEXT_ONLY = {"numglue", "fomc"}


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def to_conversations(samples, src_name):
    out = []
    for i, s in enumerate(samples):
        problem = s.get("problem") or s.get("original_question") or ""
        solution = s.get("solution") or s.get("original_answer") or ""
        item = {"id": s.get("question_id", f"{src_name}_{i}"), "conversations": []}
        img = s.get("image")
        human = ("<image>\n" + problem) if img else problem
        if img:
            item["image"] = img
        item["conversations"] = [
            {"from": "human", "value": human},
            {"from": "gpt", "value": solution},
        ]
        out.append(item)
    return out


def to_questions(samples, src_name):
    """LLaVA 评测问句 jsonl（问题 + 图，不含答案）。"""
    lines = []
    for i, s in enumerate(samples):
        rec = {"question_id": s.get("question_id", f"{src_name}_{i}"),
               "text": s.get("problem") or s.get("original_question") or ""}
        if s.get("image"):
            rec["image"] = s["image"]
        lines.append(rec)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data",
                    help="含 reasoning_train / reasoning_test 的目录")
    ap.add_argument("--out", default="data", help="输出根（生成 llava_train / llava_eval）")
    ap.add_argument("--tasks", default="numglue,art,math,fomc,medical,OCR,science")
    args = ap.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",")]

    # 训练 conversations（来自 reasoning_train）
    for t in tasks:
        in_p = os.path.join(args.src, "reasoning_train", f"{t}.json")
        if not os.path.isfile(in_p):
            print(f"[skip] missing {in_p}"); continue
        out_p = os.path.join(args.out, "llava_train", f"{t}.json")
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        conv = to_conversations(_load(in_p), t)
        json.dump(conv, open(out_p, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"  train  {in_p} -> {out_p}  ({len(conv)})")

    # 评测问句 jsonl（来自 reasoning_test）
    for t in tasks:
        in_p = os.path.join(args.src, "reasoning_test", f"{t}.json")
        if not os.path.isfile(in_p):
            print(f"[skip] missing {in_p}"); continue
        out_p = os.path.join(args.out, "llava_eval", f"{t}.jsonl")
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            for rec in to_questions(_load(in_p), t):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  eval   {in_p} -> {out_p}")


if __name__ == "__main__":
    main()
